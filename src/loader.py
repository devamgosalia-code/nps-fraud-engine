# src/loader.py
# ─────────────────────────────────────────────────────────────────────────────
# Loads the NPS CSV and extracts:
#   - Sub-ratings from the answers JSON column
#   - Verbatim feedback text from other_feedback inside answers JSON
#   - Derived fields used by all fraud layers
# Optimised for large files (770MB+) using Streamlit caching
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import json
import streamlit as st
from pathlib import Path

from config import (
    RRID_COL, STORE_COL, DATE_COL, NPS_COL,
    FEEDBACK_COL, ANSWERS_COL,
    SUB_RATING_QUESTIONS, VERBATIM_QUESTION_ID,
    BQ_PROJECT_ID, BQ_QUERY,
)


def _parse_answers_json(raw: str) -> dict:
    """
    Parse one cell of the answers JSON column.

    Extracts:
      - Numeric sub-ratings for each question in SUB_RATING_QUESTIONS
      - The verbatim text from the 'other_feedback' question_id

    Returns a flat dict:
      { "staff_friendliness_service": 5.0, ..., "_verbatim": "Good service by Ashok" }
    """
    result = {}
    if not isinstance(raw, str) or not raw.strip():
        return result
    try:
        data = json.loads(raw)
        answers = data.get("answers", [])
        for item in answers:
            qid = item.get("question_id") or item.get("question_code", "")

            # ── Sub-ratings (numeric) ──────────────────────────────────────
            if qid in SUB_RATING_QUESTIONS:
                val = item.get("answer_number")
                if val is not None:
                    try:
                        result[qid] = float(val)
                    except (ValueError, TypeError):
                        pass

            # ── Verbatim text ──────────────────────────────────────────────
            if qid == VERBATIM_QUESTION_ID:
                text = item.get("answer_string")
                if text and str(text).strip() not in ("", "null", "None"):
                    result["_verbatim"] = str(text).strip()

    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return result


def _extract_sub_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the answers JSON column.
    Creates:
      - sr_<question_id>  columns for each sub-rating (float)
      - verbatim          column (str) — actual customer/staff feedback text
    """
    if ANSWERS_COL not in df.columns:
        for q in SUB_RATING_QUESTIONS:
            df[f"sr_{q}"] = np.nan
        df["verbatim"] = ""
        return df

    parsed    = df[ANSWERS_COL].apply(_parse_answers_json)
    ratings_df = pd.DataFrame(parsed.tolist(), index=df.index)

    # Sub-rating columns
    for q in SUB_RATING_QUESTIONS:
        col = f"sr_{q}"
        df[col] = ratings_df[q] if q in ratings_df.columns else np.nan

    # Verbatim column — prefer value from answers JSON, fall back to top-level column
    if "_verbatim" in ratings_df.columns:
        df["verbatim"] = ratings_df["_verbatim"].fillna(
            df.get(FEEDBACK_COL, pd.Series("", index=df.index)).fillna("")
        )
    else:
        df["verbatim"] = df.get(FEEDBACK_COL, pd.Series("", index=df.index)).fillna("")

    df["verbatim"] = df["verbatim"].astype(str).str.strip()

    return df


def _compute_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add computed helper columns used by fraud detection layers.

    Added columns:
      - avg_sub_rating    : mean of all sr_* columns
      - is_all_perfect    : True if ALL sub-ratings = 5 AND nps_score = 10
      - feedback_clean    : lowercased, stripped verbatim for text matching
    """
    sr_cols = [f"sr_{q}" for q in SUB_RATING_QUESTIONS if f"sr_{q}" in df.columns]

    if sr_cols:
        df["avg_sub_rating"] = df[sr_cols].mean(axis=1, skipna=True)
        all_five             = (df[sr_cols] == 5.0).all(axis=1)
        df["is_all_perfect"] = all_five & (df[NPS_COL] == 10)
    else:
        df["avg_sub_rating"] = np.nan
        df["is_all_perfect"] = False

    # Normalised text for Layer 4 copy-paste detection
    df["feedback_clean"] = (
        df["verbatim"]
        .str.lower()
        .str.strip()
        .str.replace(r"\s+",       " ",  regex=True)
        .str.replace(r"[^\w\s]",   "",   regex=True)
    )

    # Ensure correct dtypes
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[NPS_COL]  = pd.to_numeric(df[NPS_COL],   errors="coerce")

    return df


def _get_csv_columns(filepath: str) -> list:
    """Peek at just the header row without reading the full file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
    return [c.strip().strip('"') for c in header.split(",")]


@st.cache_data(show_spinner="📂 Loading NPS data — please wait for large files…")
def load_nps_data(filepath: str) -> pd.DataFrame:
    """
    Load the NPS CSV, parse the answers JSON, and compute all derived fields.
    Streamlit-cached — only runs once per session per file.

    Args:
        filepath : Full path to the NPS CSV file

    Returns:
        Fully processed DataFrame ready for fraud detection
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {filepath}")

    size_mb = path.stat().st_size / 1e6
    print(f"Loading {path.name} ({size_mb:.0f} MB)…")

    df = pd.read_csv(filepath, low_memory=False)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Validate required columns
    required = [RRID_COL, STORE_COL, DATE_COL, NPS_COL]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}\n\n"
            f"Fix: update the column names at the top of config.py"
        )

    print("  Parsing answers JSON (sub-ratings + verbatims)…")
    df = _extract_sub_ratings(df)

    print("  Computing derived fields…")
    df = _compute_derived_fields(df)

    df = df.sort_values([RRID_COL, DATE_COL]).reset_index(drop=True)
    print(f"  ✅ Done. {len(df):,} responses ready.")
    return df


@st.cache_data(show_spinner="☁️ Loading NPS data from BigQuery…")
def load_nps_data_from_bigquery() -> pd.DataFrame:
    """
    Load NPS data directly from BigQuery, then run the same
    sub-rating extraction and derived-field pipeline as CSV loading.
    Uses gcloud CLI user credentials (no ADC file needed).
    """
    import google.auth
    import google.auth.transport.requests
    import subprocess, json as _json

    # Get access token from gcloud CLI directly
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gcloud auth failed: {result.stderr.strip()}\n"
            "Run: gcloud auth login"
        )
    access_token = result.stdout.strip()

    from google.oauth2.credentials import Credentials
    from google.cloud import bigquery

    credentials = Credentials(token=access_token)
    client = bigquery.Client(project=BQ_PROJECT_ID, credentials=credentials)
    df = client.query(BQ_QUERY).to_dataframe()
    print(f"  BigQuery returned {len(df):,} rows × {len(df.columns)} columns")

    # Validate required columns
    required = [RRID_COL, STORE_COL, DATE_COL, NPS_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}\n\n"
            f"Fix: update the column names at the top of config.py"
        )

    print("  Parsing answers JSON (sub-ratings + verbatims)…")
    df = _extract_sub_ratings(df)

    print("  Computing derived fields…")
    df = _compute_derived_fields(df)

    df = df.sort_values([RRID_COL, DATE_COL]).reset_index(drop=True)
    print(f"  ✅ Done. {len(df):,} responses ready.")
    return df


def load_nps_data_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    """
    Load NPS data from uploaded file bytes (for Streamlit file_uploader).
    Writes to a temp file then calls load_nps_data.
    """
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(file_bytes)
    tmp.close()
    try:
        df = load_nps_data(tmp.name)
    finally:
        os.unlink(tmp.name)
    return df


def get_sub_rating_cols(df: pd.DataFrame) -> list:
    """Return all sr_* column names present in the dataframe."""
    return [c for c in df.columns if c.startswith("sr_")]


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return a quick summary dict for the sidebar/header."""
    return {
        "total_responses":  len(df),
        "unique_stores":    df[STORE_COL].nunique(),
        "unique_rrids":     df[RRID_COL].nunique(),
        "date_min":         df[DATE_COL].min(),
        "date_max":         df[DATE_COL].max(),
        "all_perfect_count": int(df["is_all_perfect"].sum()),
        "all_perfect_pct":  round(df["is_all_perfect"].mean() * 100, 1),
        "has_verbatim":     int((df["verbatim"].str.len() > 0).sum()),
    }
