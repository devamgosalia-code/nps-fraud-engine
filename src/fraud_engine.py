# src/fraud_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# NPS Fraud Detection Engine — 5 Layers + Staff Coaching Detection
# Trends Retail / Reliance Retail — PRD v1.0
# ─────────────────────────────────────────────────────────────────────────────

import re
import pandas as pd
import numpy as np
from typing import Tuple

from config import (
    RRID_COL, STORE_COL, DATE_COL, NPS_COL,
    L1_WINDOW_DAYS, L1_HEAVY_DUP_THRESHOLD, L1_LIGHT_DUP_THRESHOLD,
    L2_DUP_RATIO_THRESHOLD, L2_HEAVY_DUP_COUNT_MIN,
    L2_ALL_PERFECT_RATE, L2_ALL_PERFECT_MIN_RESPONSES,
    L3_MIN_RESPONSES, L3_UNIQUE_RRID_RATIO_MAX, L3_PERFECT_RATE_MIN,
    L4_RRID_EXACT_THRESHOLD, L4_RRID_SIMILAR_THRESHOLD,
    L4_STORE_EXACT_THRESHOLD, L4_STORE_SIMILAR_THRESHOLD,
    L4_EXCLUDED_PHRASES,
    L5_MONOTONE_LOW_MAX, L5_EXTREME_CONTRADICTION_AVG, L5_EXTREME_NPS_MIN,
    L5_REVERSE_AVG_MIN, L5_REVERSE_NPS_MAX,
    LAYER_WEIGHTS, IS_FRAUD_THRESHOLD, COACHING_NAME_MIN_COUNT,
)
from src.loader import get_sub_rating_cols


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 1 — Duplicate RRID Detection
# ═════════════════════════════════════════════════════════════════════════════

def run_layer1_duplicate_rrid(df: pd.DataFrame) -> pd.Series:
    """
    Identify ballot-stuffing: same RRID submitting multiple surveys in 7 days.

    Rules (PRD §3.1):
      - 3+ submissions in 7-day window  → ALL submissions flagged  (RRID_HEAVY_DUP)
      - Exactly 2 submissions           → 2nd submission flagged   (RRID_LIGHT_DUP)
      - 1 submission                    → clean

    Returns: pd.Series of flag codes, indexed like df.
    """
    flags = pd.Series("", index=df.index, dtype=str)
    work  = df[[RRID_COL, DATE_COL]].copy()

    # Skip single-submission RRIDs (the vast majority)
    rrid_counts = work.groupby(RRID_COL)[DATE_COL].transform("count")
    multi_mask  = rrid_counts >= 2
    if not multi_mask.any():
        return flags

    multi = work[multi_mask].copy()
    grp   = multi.groupby(RRID_COL)

    # Date range per RRID to identify fast-path cases
    multi["_count"]      = rrid_counts[multi_mask]
    multi["_range_days"] = (
        grp[DATE_COL].transform("max") - grp[DATE_COL].transform("min")
    ).dt.days

    # ── Fast path: all submissions fit in one 7-day window ──────────────
    fits = multi["_range_days"] <= L1_WINDOW_DAYS

    # Heavy dups (3+ in window) → flag all
    heavy_idx = multi.index[fits.values & (multi["_count"].values >= L1_HEAVY_DUP_THRESHOLD)]
    flags.loc[heavy_idx] = "RRID_HEAVY_DUP"

    # Light dups (exactly 2 in window) → flag non-first
    light_cand = fits & (multi["_count"] == L1_LIGHT_DUP_THRESHOLD)
    if light_cand.any():
        light_sub = multi[light_cand]
        rank = light_sub.groupby(RRID_COL)[DATE_COL].rank(method="first")
        flags.loc[rank.index[rank.values > 1]] = "RRID_LIGHT_DUP"

    # ── Slow path: RRIDs spanning > 7 days (need rolling windows) ──────
    spread = multi[~fits]
    if len(spread):
        for _, grp_df in spread.groupby(RRID_COL):
            grp_df = grp_df.sort_values(DATE_COL)
            idxs   = grp_df.index.tolist()
            dates  = grp_df[DATE_COL].tolist()
            n      = len(idxs)
            for i in range(n):
                if pd.isnull(dates[i]):
                    continue
                window_idxs = [
                    idxs[j] for j in range(n)
                    if not pd.isnull(dates[j])
                    and abs((dates[j] - dates[i]).days) <= L1_WINDOW_DAYS
                ]
                window_n = len(window_idxs)
                if window_n >= L1_HEAVY_DUP_THRESHOLD:
                    for wi in window_idxs:
                        flags.loc[wi] = "RRID_HEAVY_DUP"
                elif window_n == L1_LIGHT_DUP_THRESHOLD:
                    if idxs[i] != window_idxs[0]:
                        if flags.loc[idxs[i]] == "":
                            flags.loc[idxs[i]] = "RRID_LIGHT_DUP"

    return flags


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Store Contamination
# ═════════════════════════════════════════════════════════════════════════════

def compute_store_contamination(
    df: pd.DataFrame, layer1_flags: pd.Series
) -> pd.DataFrame:
    """
    Classify each store as contaminated or clean.

    A store is contaminated if ANY condition is met (PRD §3.2):
      1. Duplicate RRID ratio > 30%
      2. Heavy duplicate count ≥ 5
      3. All-perfect rate > 90%  AND  ≥15 total responses

    Returns: DataFrame indexed by STORE_COL with health metrics.
    """
    tmp = df[[STORE_COL, RRID_COL, "is_all_perfect"]].copy()
    tmp["_l1"] = layer1_flags.values

    store_stats = tmp.groupby(STORE_COL).agg(
        total_responses    = (RRID_COL,        "count"),
        unique_rrids       = (RRID_COL,        "nunique"),
        all_perfect_count  = ("is_all_perfect", "sum"),
        heavy_dup_count    = ("_l1",            lambda x: (x == "RRID_HEAVY_DUP").sum()),
        dup_responses      = ("_l1",            lambda x: x.isin(["RRID_HEAVY_DUP", "RRID_LIGHT_DUP"]).sum()),
    ).reset_index()

    store_stats["dup_ratio"]    = store_stats["dup_responses"] / store_stats["total_responses"].clip(lower=1)
    store_stats["perfect_rate"] = store_stats["all_perfect_count"] / store_stats["total_responses"].clip(lower=1)

    crit1 = store_stats["dup_ratio"]     > L2_DUP_RATIO_THRESHOLD
    crit2 = store_stats["heavy_dup_count"] >= L2_HEAVY_DUP_COUNT_MIN
    crit3 = (store_stats["perfect_rate"] > L2_ALL_PERFECT_RATE) & \
            (store_stats["total_responses"] >= L2_ALL_PERFECT_MIN_RESPONSES)

    store_stats["contamination_reason"] = ""
    store_stats.loc[crit1, "contamination_reason"] += "HIGH_DUP_RATIO|"
    store_stats.loc[crit2, "contamination_reason"] += "HEAVY_DUP_COUNT|"
    store_stats.loc[crit3, "contamination_reason"] += "HIGH_PERFECT_RATE|"
    store_stats["contamination_reason"] = store_stats["contamination_reason"].str.rstrip("|")
    store_stats["is_contaminated"]      = crit1 | crit2 | crit3

    return store_stats


def run_layer2_store_contamination(
    df: pd.DataFrame, store_contamination: pd.DataFrame
) -> pd.Series:
    """
    Flag all-perfect responses from contaminated stores.
    Non-perfect responses at contaminated stores are NOT flagged.
    """
    contaminated = set(
        store_contamination.loc[store_contamination["is_contaminated"], STORE_COL]
    )
    flags = pd.Series("", index=df.index, dtype=str)
    mask  = df[STORE_COL].isin(contaminated) & df["is_all_perfect"]
    flags[mask] = "CONTAMINATED_STORE_PERFECT"
    return flags


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 3 — Response Velocity Anomaly
# ═════════════════════════════════════════════════════════════════════════════

def run_layer3_velocity_anomaly(df: pd.DataFrame) -> pd.Series:
    """
    Flag store-days with abnormal clustering (PRD §3.3):
      - ≥6 responses on one store-day
      - <70% unique RRIDs
      - >80% all-perfect
    All three conditions must be met simultaneously.
    """
    flags = pd.Series("", index=df.index, dtype=str)

    sd = df.groupby([STORE_COL, DATE_COL]).agg(
        response_count = (RRID_COL,         "count"),
        unique_rrids   = (RRID_COL,         "nunique"),
        perfect_count  = ("is_all_perfect",  "sum"),
    ).reset_index()

    sd["unique_rrid_ratio"] = sd["unique_rrids"]   / sd["response_count"].clip(lower=1)
    sd["perfect_rate"]      = sd["perfect_count"]  / sd["response_count"].clip(lower=1)

    anomaly_mask = (
        (sd["response_count"]   >= L3_MIN_RESPONSES)          &
        (sd["unique_rrid_ratio"] < L3_UNIQUE_RRID_RATIO_MAX)  &
        (sd["perfect_rate"]      > L3_PERFECT_RATE_MIN)
    )
    if anomaly_mask.any():
        anomaly_keys = sd.loc[anomaly_mask, [STORE_COL, DATE_COL]].copy()
        anomaly_keys["_anomaly"] = True
        merged = df[[STORE_COL, DATE_COL]].merge(
            anomaly_keys, on=[STORE_COL, DATE_COL], how="left"
        )
        flags.loc[merged.index[merged["_anomaly"] == True]] = "VELOCITY_ANOMALY"

    return flags


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 4 — Feedback Text Fingerprinting
# ═════════════════════════════════════════════════════════════════════════════

def run_layer4_text_fingerprint(df: pd.DataFrame) -> pd.Series:
    """
    Flag copy-paste feedback using 4 scenarios:
      1. RRID_EXACT_COPY    — Same RRID, exact same text ≥2 times
      2. RRID_SIMILAR_COPY  — Same RRID, similar fingerprint ≥2 times
      3. STORE_EXACT_COPY   — Same store, exact same text ≥3 times
      4. STORE_SIMILAR_COPY — Same store, similar fingerprint ≥4 times

    Priority order (highest confidence wins):
    RRID_EXACT > RRID_SIMILAR > STORE_EXACT > STORE_SIMILAR
    First occurrence is always kept clean.
    """
    import re as _re

    flags = pd.Series("", index=df.index, dtype=str)

    # Guard: check column exists
    if "feedback_clean" not in df.columns:
        print("  ⚠️  L4: 'feedback_clean' column not found — skipping")
        return flags

    # ── Step 1: Filter to non-trivial feedback only ──────────────────────────
    cleaned = df["feedback_clean"].fillna("").str.strip()
    non_trivial = (
        cleaned.str.len() > 3
    ) & (~cleaned.isin(L4_EXCLUDED_PHRASES))

    sub = df.loc[non_trivial, [RRID_COL, STORE_COL, "feedback_clean"]].copy()
    sub["feedback_clean"] = sub["feedback_clean"].fillna("").str.strip()

    if len(sub) == 0:
        print("  ⚠️  L4: No non-trivial feedback found after exclusion filter")
        return flags

    print(f"  L4: {len(sub):,} non-trivial responses to check")

    # ── Step 2: Compute fingerprint for near-duplicate matching ──────────────
    _FILLER = _re.compile(
        r"\b(very|really|so|quite|most|the|a|an|is|was|by|from|of|and|"
        r"for|with|to|in|has|have|were|are|its|my|our|your|their|this|"
        r"that|at|on|we|i|it|he|she|they|also|too|just|mr|ms|sir|madam)\b"
    )

    def _make_fingerprint(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        t = _FILLER.sub("", text.lower().strip())
        t = _re.sub(r"\s+", " ", t).strip()
        words = t.split()
        if not words:
            # Fingerprint collapsed to nothing — use original text truncated
            return text.lower().strip()[:30]
        return " ".join(words[:6])

    sub["fingerprint"] = sub["feedback_clean"].apply(_make_fingerprint)

    # Only remove truly empty fingerprints (empty string), not short ones
    sub = sub[sub["fingerprint"].str.len() > 0]

    if len(sub) == 0:
        print("  ⚠️  L4: All fingerprints empty after normalisation")
        return flags

    # ═══════════════════════════════════════════════════════════════════════
    # SCENARIO A — RRID_EXACT_COPY
    # Same RRID submitted exact same text 2+ times (any store)
    # ═══════════════════════════════════════════════════════════════════════
    grp_re = [RRID_COL, "feedback_clean"]
    sub["_re_count"] = sub.groupby(grp_re)["feedback_clean"].transform("count")
    sub["_re_rank"]  = sub.groupby(grp_re).cumcount()

    rrid_exact_mask = (
        (sub["_re_count"] >= L4_RRID_EXACT_THRESHOLD) &
        (sub["_re_rank"]  >= 1)
    ).values

    flags.loc[sub.index[rrid_exact_mask]] = "RRID_EXACT_COPY"
    print(f"  L4 Scenario A (RRID exact):    {rrid_exact_mask.sum():,} flagged")

    # ═══════════════════════════════════════════════════════════════════════
    # SCENARIO B — RRID_SIMILAR_COPY
    # Same RRID, similar fingerprint, 2+ times
    # ═══════════════════════════════════════════════════════════════════════
    grp_rs = [RRID_COL, "fingerprint"]
    sub["_rs_count"] = sub.groupby(grp_rs)["fingerprint"].transform("count")
    sub["_rs_rank"]  = sub.groupby(grp_rs).cumcount()

    already_flagged_a = (flags.loc[sub.index] != "").values
    rrid_sim_mask = (
        (sub["_rs_count"] >= L4_RRID_SIMILAR_THRESHOLD) &
        (sub["_rs_rank"]  >= 1)
    ).values & ~already_flagged_a

    flags.loc[sub.index[rrid_sim_mask]] = "RRID_SIMILAR_COPY"
    print(f"  L4 Scenario B (RRID similar):  {rrid_sim_mask.sum():,} flagged")

    # ═══════════════════════════════════════════════════════════════════════
    # SCENARIO C — STORE_EXACT_COPY
    # Different RRIDs, exact same text at same store, 3+ times
    # Must come from at least 2 different RRIDs
    # ═══════════════════════════════════════════════════════════════════════
    grp_se = [STORE_COL, "feedback_clean"]
    sub["_se_count"] = sub.groupby(grp_se)["feedback_clean"].transform("count")
    sub["_se_uniq"]  = sub.groupby(grp_se)[RRID_COL].transform("nunique")
    sub["_se_rank"]  = sub.groupby(grp_se).cumcount()

    already_flagged_ab = (flags.loc[sub.index] != "").values
    store_exact_mask = (
        (sub["_se_count"] >= L4_STORE_EXACT_THRESHOLD) &
        (sub["_se_uniq"]  >= 2) &
        (sub["_se_rank"]  >= 1)
    ).values & ~already_flagged_ab

    flags.loc[sub.index[store_exact_mask]] = "STORE_EXACT_COPY"
    print(f"  L4 Scenario C (store exact):   {store_exact_mask.sum():,} flagged")

    # ═══════════════════════════════════════════════════════════════════════
    # SCENARIO D — STORE_SIMILAR_COPY
    # Different RRIDs, similar fingerprint at same store, 4+ times
    # Catches verbal coaching
    # ═══════════════════════════════════════════════════════════════════════
    grp_ss = [STORE_COL, "fingerprint"]
    sub["_ss_count"] = sub.groupby(grp_ss)["fingerprint"].transform("count")
    sub["_ss_uniq"]  = sub.groupby(grp_ss)[RRID_COL].transform("nunique")
    sub["_ss_rank"]  = sub.groupby(grp_ss).cumcount()

    already_flagged_abc = (flags.loc[sub.index] != "").values
    store_sim_mask = (
        (sub["_ss_count"] >= L4_STORE_SIMILAR_THRESHOLD) &
        (sub["_ss_uniq"]  >= 2) &
        (sub["_ss_rank"]  >= 1)
    ).values & ~already_flagged_abc

    flags.loc[sub.index[store_sim_mask]] = "STORE_SIMILAR_COPY"
    print(f"  L4 Scenario D (store similar): {store_sim_mask.sum():,} flagged")

    total_l4 = (flags != "").sum()
    print(f"  L4 Total: {total_l4:,} flagged across all scenarios")

    return flags


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 5 — Scoring Contradiction Detection
# ═════════════════════════════════════════════════════════════════════════════

def run_layer5_scoring_contradiction(df: pd.DataFrame) -> pd.Series:
    """
    Flag logically impossible rating patterns (PRD §3.5):

    MONOTONE_MISMATCH:      All sub-ratings identical and ≤3 but NPS = 10
    EXTREME_CONTRADICTION:  Avg sub-rating ≤3 but NPS ≥9
    REVERSE_CONTRADICTION:  Avg sub-rating ≥4.5 but NPS ≤3
    """
    flags   = pd.Series("", index=df.index, dtype=str)
    nps     = df[NPS_COL]
    avg     = df["avg_sub_rating"]
    sr_cols = get_sub_rating_cols(df)

    if sr_cols:
        sub_df   = df[sr_cols]
        all_same = sub_df.nunique(axis=1) == 1
        all_low  = sub_df.max(axis=1)    <= L5_MONOTONE_LOW_MAX
        monotone = all_same & all_low & (nps == 10)
        flags[monotone] = "MONOTONE_MISMATCH"

    extreme = (avg <= L5_EXTREME_CONTRADICTION_AVG) & (nps >= L5_EXTREME_NPS_MIN)
    flags[extreme & (flags == "")] = "EXTREME_CONTRADICTION"

    reverse = (avg >= L5_REVERSE_AVG_MIN) & (nps <= L5_REVERSE_NPS_MAX)
    flags[reverse & (flags == "")] = "REVERSE_CONTRADICTION"

    return flags


# ═════════════════════════════════════════════════════════════════════════════
# STAFF COACHING DETECTION
# ═════════════════════════════════════════════════════════════════════════════

# Patterns that indicate a staff member's name in a verbatim response.
# Catches: "Good service by Ashok", "Mamta cashier excellent",
#          "thanks Dheeraj Kumar", "very helpful sir Rakesh"
_NAME_PATTERNS = [
    r'\bby\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b',
    r'\bfrom\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b',
    r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)?)\s+(?:cashier|manager|staff|sir|madam|bhaiya|didi|bhai|di|uncle|aunty)\b',
    r'\b(?:cashier|manager|staff|sir|madam|bhaiya|didi|bhai)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b',
    r'\bthanks?\s+(?:to\s+)?([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b',
    r'\bthank\s+you\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b',
    r'\bhelped?\s+by\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b',
    r'\bservice\s+(?:of|by|from)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b',
    r'\bname\s+(?:is\s+)?([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b',
]
_COMBINED_PATTERN = re.compile(
    "|".join(f"(?:{p})" for p in _NAME_PATTERNS),
    re.UNICODE,
)

# Words that should NOT be treated as names even if they're capitalised
_NOT_NAMES = {
    "Good", "Nice", "Very", "Service", "Excellent", "Super",
    "Best", "Great", "Happy", "Store", "Staff", "Team",
    "Reliance", "Trends", "Thank", "Thanks", "Please",
    "No", "Not", "All", "This", "The", "Product", "Quality",
    "Sir", "Madam", "Bhai", "Bhaiya", "Didi",
}


def _extract_staff_name(text: str) -> str:
    """
    Extract the most likely staff name from a verbatim response.
    Returns empty string if no name found.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    matches = _COMBINED_PATTERN.findall(text)
    # findall returns list of tuples (one per group); flatten
    candidates = [
        m for tup in matches
        for m in (tup if isinstance(tup, tuple) else [tup])
        if m and m.strip()
    ]
    for name in candidates:
        name = name.strip().title()
        if name and name not in _NOT_NAMES and len(name) > 2:
            return name
    return ""


def detect_staff_coaching(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect responses where staff coached the customer to mention their name.

    Two-step logic:
      1. Extract staff name candidate from every verbatim using regex patterns
      2. Flag responses where the SAME name appears ≥3 times at the SAME store

    Returns:
      df              — original df with three new columns:
                          suspected_staff_name  (str)
                          staff_name_detected   (bool)
                          staff_coaching_flag   (str: "STAFF_COACHING" or "")
      name_leaderboard — DataFrame: store × name × mention_count, sorted desc
    """
    verbatim_col = "verbatim" if "verbatim" in df.columns else "feedback_clean"

    df["suspected_staff_name"] = df[verbatim_col].apply(_extract_staff_name)
    df["staff_name_detected"]  = df["suspected_staff_name"] != ""

    # Count (store, name) pairs
    name_counts = (
        df[df["staff_name_detected"]]
        .groupby([STORE_COL, "suspected_staff_name"])
        .agg(
            mention_count  = ("suspected_staff_name", "count"),
            unique_rrids   = (RRID_COL,               "nunique"),
            first_seen     = (DATE_COL,                "min"),
            last_seen      = (DATE_COL,                "max"),
            sample_verbatim= (verbatim_col,            "first"),
        )
        .reset_index()
        .sort_values("mention_count", ascending=False)
    )

    # Only flag if same name repeated ≥ COACHING_NAME_MIN_COUNT times at one store
    coached_pairs = set(
        zip(
            name_counts.loc[name_counts["mention_count"] >= COACHING_NAME_MIN_COUNT, STORE_COL],
            name_counts.loc[name_counts["mention_count"] >= COACHING_NAME_MIN_COUNT, "suspected_staff_name"],
        )
    )

    if coached_pairs:
        coached_keys = pd.DataFrame(
            list(coached_pairs), columns=[STORE_COL, "suspected_staff_name"]
        )
        coached_keys["staff_coaching_flag"] = "STAFF_COACHING"
        df = df.merge(coached_keys, on=[STORE_COL, "suspected_staff_name"], how="left")
        df["staff_coaching_flag"] = df["staff_coaching_flag"].fillna("")
    else:
        df["staff_coaching_flag"] = ""

    return df, name_counts


# ═════════════════════════════════════════════════════════════════════════════
# COMPOSITE FRAUD SCORER
# ═════════════════════════════════════════════════════════════════════════════

def compute_fraud_scores(
    df: pd.DataFrame,
    l1: pd.Series,
    l2: pd.Series,
    l3: pd.Series,
    l4: pd.Series,
    l5: pd.Series,
) -> pd.DataFrame:
    """
    Combine all layer flags into a composite fraud score per response.

    Output columns appended:
      fraud_reasons       pipe-delimited list of triggered flag codes
      fraud_layer_count   number of layers triggered (0–5)
      fraud_score         weighted score 0–100
      disposition         CLEAN / QUARANTINED / REJECTED
      is_fraud            True if any layer triggered
    """
    result = df.copy()

    # Vectorized boolean masks for each layer
    b1 = l1.values != ""
    b2 = l2.values != ""
    b3 = l3.values != ""
    b4 = l4.values != ""
    b5 = l5.values != ""

    # Vectorized fraud_reasons (avoid row-by-row apply)
    reasons = pd.Series("", index=df.index, dtype=object)
    for flag_arr, flag_vals in [(b1, l1), (b2, l2), (b3, l3), (b4, l4), (b5, l5)]:
        has_existing = reasons != ""
        reasons = np.where(
            flag_arr & has_existing, reasons + "|" + flag_vals,
            np.where(flag_arr, flag_vals, reasons),
        )
    result["fraud_reasons"] = reasons

    # Vectorized layer count
    result["fraud_layer_count"] = (
        b1.astype(int) + b2.astype(int) + b3.astype(int) +
        b4.astype(int) + b5.astype(int)
    )

    # Vectorized L4 sub-score based on which scenario was flagged
    l4_scores = np.where(
        l4.values == "RRID_EXACT_COPY",   LAYER_WEIGHTS["layer4_rrid_exact"],
        np.where(
            l4.values == "RRID_SIMILAR_COPY", LAYER_WEIGHTS["layer4_rrid_similar"],
            np.where(
                l4.values == "STORE_EXACT_COPY",  LAYER_WEIGHTS["layer4_store_exact"],
                np.where(
                    l4.values == "STORE_SIMILAR_COPY", LAYER_WEIGHTS["layer4_store_similar"],
                    0,
                ),
            ),
        ),
    )

    # Vectorized score
    result["fraud_score"] = np.minimum(
        b1.astype(int) * LAYER_WEIGHTS["layer1"] +
        b2.astype(int) * LAYER_WEIGHTS["layer2"] +
        b3.astype(int) * LAYER_WEIGHTS["layer3"] +
        l4_scores +
        b5.astype(int) * LAYER_WEIGHTS["layer5"],
        100,
    )

    # Vectorized disposition
    n = result["fraud_layer_count"].values
    s = result["fraud_score"].values
    result["disposition"] = np.where(
        n == 0, "CLEAN",
        np.where((n >= 3) | (s >= 80), "REJECTED", "QUARANTINED"),
    )

    result["is_fraud"] = result["fraud_layer_count"] >= IS_FRAUD_THRESHOLD
    return result


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_fraud_engine(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the full fraud detection pipeline:
      Layer 1 — Duplicate RRID
      Layer 2 — Store Contamination
      Layer 3 — Velocity Anomaly
      Layer 4 — Text Fingerprinting
      Layer 5 — Scoring Contradictions
      + Staff Coaching Detection

    Returns:
      scored_df       — full DataFrame with all fraud columns
      store_health    — store-level contamination metrics
      name_leaderboard— staff name mention counts per store
    """
    print("Layer 1: Duplicate RRID detection…")
    l1 = run_layer1_duplicate_rrid(df)
    print(f"  → {(l1 != '').sum():,} flagged")

    print("Layer 2: Store contamination…")
    store_health = compute_store_contamination(df, l1)
    l2 = run_layer2_store_contamination(df, store_health)
    n_cont = store_health["is_contaminated"].sum()
    print(f"  → {n_cont} contaminated stores, {(l2 != '').sum():,} responses flagged")

    print("Layer 3: Velocity anomaly…")
    l3 = run_layer3_velocity_anomaly(df)
    print(f"  → {(l3 != '').sum():,} flagged")

    print("Layer 4: Text fingerprinting…")
    l4 = run_layer4_text_fingerprint(df)
    print(f"  → {(l4 != '').sum():,} flagged")

    print("Layer 5: Scoring contradictions…")
    l5 = run_layer5_scoring_contradiction(df)
    print(f"  → {(l5 != '').sum():,} flagged")

    print("Staff coaching detection…")
    df, name_leaderboard = detect_staff_coaching(df)
    coached = (df["staff_coaching_flag"] == "STAFF_COACHING").sum()
    print(f"  → {coached:,} coached responses, "
          f"{df['suspected_staff_name'].ne('').sum():,} with a staff name detected")

    print("Computing composite scores…")
    scored_df = compute_fraud_scores(df, l1, l2, l3, l4, l5)

    total   = len(scored_df)
    n_fraud = scored_df["is_fraud"].sum()
    print(f"\n✅ Engine complete: {n_fraud:,} / {total:,} responses flagged "
          f"({n_fraud/total*100:.1f}%)")

    return scored_df, store_health, name_leaderboard
