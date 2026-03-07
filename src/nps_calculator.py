# src/nps_calculator.py
# ─────────────────────────────────────────────────────────────────────────────
# NPS Calculations:
#   - Overall Reported NPS vs Clean NPS
#   - Per-store NPS breakdown
#   - Daily/weekly trend
#   - Layer-level fraud count breakdown
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from typing import Optional

from config import (
    RRID_COL, STORE_COL, DATE_COL, NPS_COL,
    NPS_PROMOTER_MIN, NPS_PASSIVE_MIN, NPS_DETRACTOR_MAX,
)


def _nps(scores: pd.Series) -> float:
    """Compute NPS from a series of 0–10 scores. Returns 0.0 if empty."""
    valid = scores.dropna()
    if len(valid) == 0:
        return 0.0
    n          = len(valid)
    promoters  = (valid >= NPS_PROMOTER_MIN).sum()
    detractors = (valid <= NPS_DETRACTOR_MAX).sum()
    return round((promoters / n - detractors / n) * 100, 1)


def _classify(score: float) -> str:
    if pd.isna(score):  return "Unknown"
    if score >= NPS_PROMOTER_MIN: return "Promoter"
    if score >= NPS_PASSIVE_MIN:  return "Passive"
    return "Detractor"


def compute_overall_nps(scored_df: pd.DataFrame) -> dict:
    """
    Compute Reported NPS (all responses) and Clean NPS (fraud excluded).

    Returns dict with keys:
      reported_nps, clean_nps, nps_inflation,
      reported_counts, clean_counts  (Promoter/Passive/Detractor breakdown),
      total_responses, clean_responses, fraud_count, fraud_pct
    """
    all_scores   = scored_df[NPS_COL]
    clean_mask   = ~scored_df["is_fraud"]
    clean_scores = scored_df.loc[clean_mask, NPS_COL]

    reported_nps = _nps(all_scores)
    clean_nps    = _nps(clean_scores)

    cats           = scored_df[NPS_COL].apply(_classify)
    clean_cats     = scored_df.loc[clean_mask, NPS_COL].apply(_classify)

    fraud_count = int(scored_df["is_fraud"].sum())
    total       = len(scored_df)

    return {
        "reported_nps":    reported_nps,
        "clean_nps":       clean_nps,
        "nps_inflation":   round(reported_nps - clean_nps, 1),
        "reported_counts": cats.value_counts().to_dict(),
        "clean_counts":    clean_cats.value_counts().to_dict(),
        "total_responses": total,
        "clean_responses": total - fraud_count,
        "fraud_count":     fraud_count,
        "fraud_pct":       round(fraud_count / total * 100, 1) if total else 0,
    }


def _nps_grouped(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute NPS per group using vectorized operations."""
    scores = df[[group_col, NPS_COL]].dropna(subset=[NPS_COL])
    n = scores.groupby(group_col)[NPS_COL].count()
    promoters = scores[scores[NPS_COL] >= NPS_PROMOTER_MIN].groupby(group_col).size()
    detractors = scores[scores[NPS_COL] <= NPS_DETRACTOR_MAX].groupby(group_col).size()
    promoters = promoters.reindex(n.index, fill_value=0)
    detractors = detractors.reindex(n.index, fill_value=0)
    nps = ((promoters / n - detractors / n) * 100).round(1)
    return nps


def compute_store_nps(
    scored_df: pd.DataFrame,
    store_health: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Per-store: Reported NPS, Clean NPS, fraud %, risk level.
    Merged with store_health for contamination signals.
    """
    # Aggregate all stores at once
    all_grp = scored_df.groupby(STORE_COL)
    out = all_grp.agg(
        total_responses=(NPS_COL, "count"),
        fraud_count=("is_fraud", "sum"),
        all_perfect_pct=("is_all_perfect", "mean"),
    ).reset_index()
    out["fraud_count"] = out["fraud_count"].astype(int)
    out["fraud_pct"] = (out["fraud_count"] / out["total_responses"] * 100).round(1)
    out["all_perfect_pct"] = (out["all_perfect_pct"] * 100).round(1)

    # Vectorized NPS per store — reported
    reported_nps = _nps_grouped(scored_df, STORE_COL)
    out["reported_nps"] = out[STORE_COL].map(reported_nps).fillna(0.0)

    # Clean NPS per store
    clean_df = scored_df[~scored_df["is_fraud"]]
    clean_nps = _nps_grouped(clean_df, STORE_COL)
    clean_counts = clean_df.groupby(STORE_COL).size()
    out["clean_nps"] = out[STORE_COL].map(clean_nps).fillna(0.0)
    out["clean_responses"] = out[STORE_COL].map(clean_counts).fillna(0).astype(int)

    out["nps_inflation"] = (out["reported_nps"] - out["clean_nps"]).round(1)

    # Join contamination data
    if store_health is not None:
        keep = [c for c in [
            STORE_COL, "is_contaminated", "dup_ratio",
            "heavy_dup_count", "perfect_rate", "contamination_reason"
        ] if c in store_health.columns]
        out = out.merge(store_health[keep], on=STORE_COL, how="left")

    # Vectorized risk level
    fp = out["fraud_pct"]
    con = out.get("is_contaminated", pd.Series(False, index=out.index)).fillna(False)
    out["risk_level"] = np.where(
        (fp >= 60) | (con & (fp >= 40)), "CRITICAL",
        np.where((fp >= 40) | con, "HIGH",
            np.where(fp >= 20, "MEDIUM", "LOW")),
    )

    return out.sort_values("fraud_count", ascending=False).reset_index(drop=True)


def compute_nps_trend(scored_df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Daily or weekly Reported NPS vs Clean NPS trend."""
    tmp = scored_df.copy()
    tmp["_period"] = tmp[DATE_COL].dt.to_period(freq).dt.to_timestamp()

    rows = []
    for period, grp in tmp.groupby("_period"):
        clean = grp[~grp["is_fraud"]]
        rows.append({
            "date":             period,
            "reported_nps":     _nps(grp[NPS_COL]),
            "clean_nps":        _nps(clean[NPS_COL]),
            "total_responses":  len(grp),
            "fraud_count":      int(grp["is_fraud"].sum()),
        })
    return pd.DataFrame(rows).sort_values("date")


def compute_layer_breakdown(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Count responses flagged by each detection layer code."""
    layer_map = {
        "RRID_HEAVY_DUP":             "L1 · Heavy Duplicate RRID",
        "RRID_LIGHT_DUP":             "L1 · Light Duplicate RRID",
        "CONTAMINATED_STORE_PERFECT": "L2 · Contaminated Store Perfect",
        "VELOCITY_ANOMALY":           "L3 · Velocity Anomaly",
        "REPEATED_FEEDBACK":          "L4 · Copy-Paste Feedback",
        "MONOTONE_MISMATCH":          "L5 · Monotone Mismatch",
        "EXTREME_CONTRADICTION":      "L5 · Extreme Contradiction",
        "REVERSE_CONTRADICTION":      "L5 · Reverse Contradiction",
    }
    total = len(scored_df)
    rows  = []
    for code, label in layer_map.items():
        count = int(scored_df["fraud_reasons"].str.contains(code, na=False).sum())
        rows.append({
            "layer_code":   code,
            "layer_label":  label,
            "count":        count,
            "pct_of_total": round(count / total * 100, 2) if total else 0,
        })
    return pd.DataFrame(rows).sort_values("count", ascending=False)
