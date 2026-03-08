# app.py
# ─────────────────────────────────────────────────────────────────────────────
# NPS Fraud Detection Dashboard
# Trends Retail · Reliance Retail
# ─────────────────────────────────────────────────────────────────────────────

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from config import STORE_COL, RRID_COL, DATE_COL, NPS_COL


def format_date_with_ordinal(date_val):
    """
    Format date as "5th Mar, 2026" with ordinal suffix.
    Handles datetime, date, pd.Timestamp, and string inputs.
    """
    if pd.isna(date_val) or date_val is None:
        return ""
    
    # Convert to datetime if needed
    if isinstance(date_val, str):
        try:
            date_val = pd.to_datetime(date_val)
        except:
            return str(date_val)
    
    # Handle pandas Timestamp
    if isinstance(date_val, pd.Timestamp):
        date_val = date_val.to_pydatetime()
    
    # Handle datetime.date
    if hasattr(date_val, 'day') and hasattr(date_val, 'month') and hasattr(date_val, 'year'):
        day = date_val.day
        month = date_val.month
        year = date_val.year
    elif isinstance(date_val, datetime):
        day = date_val.day
        month = date_val.month
        year = date_val.year
    else:
        return str(date_val)
    
    # Add ordinal suffix
    if 10 <= day % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    
    month_abbr = datetime(year, month, day).strftime("%b")
    
    return f"{day}{suffix} {month_abbr}, {year}"
from src.loader import load_nps_data_from_bigquery, get_data_summary
from src.fraud_engine import run_fraud_engine
from src.nps_calculator import (
    compute_overall_nps,
    compute_store_nps,
    compute_nps_trend,
    compute_layer_breakdown,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NPS Fraud Engine — Trends Retail",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #f5f7fa;
    color: #1a2035;
}

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] * { color: #475569 !important; }
[data-testid="stSidebar"] .stButton button {
    background: #2563eb !important;
    color: #fff !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 11px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 500 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2rem !important;
    color: #0f172a !important;
    font-weight: 600 !important;
}

h1 { font-size: 1.6rem !important; font-weight: 700; color: #0f172a !important; }
h2 { font-size: 1.1rem !important; font-weight: 600; color: #1e293b !important;
     border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; margin-top: 28px !important; }
h3 { font-size: 0.95rem !important; font-weight: 500; color: #475569 !important; }

[data-testid="stTabs"] [role="tab"] {
    font-size: 13px;
    color: #64748b;
    font-weight: 500;
    padding: 10px 16px;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #2563eb !important;
    border-bottom-color: #2563eb !important;
    font-weight: 600 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.card-fraud {
    background: #fff5f5;
    border-left: 4px solid #dc2626;
    padding: 12px 16px;
    margin: 6px 0;
    border-radius: 0 8px 8px 0;
    box-shadow: 0 1px 3px rgba(220,38,38,0.08);
}
.card-fraud .text  { color: #7f1d1d; font-size: 13px; line-height: 1.6; }
.card-fraud .meta  { color: #94a3b8; font-size: 11px; margin-top: 6px; }
.card-fraud .name  { color: #dc2626; font-weight: 700; }

.card-clean {
    background: #f0fdf4;
    border-left: 4px solid #16a34a;
    padding: 12px 16px;
    margin: 6px 0;
    border-radius: 0 8px 8px 0;
    box-shadow: 0 1px 3px rgba(22,163,74,0.08);
}
.card-clean .text  { color: #14532d; font-size: 13px; line-height: 1.6; }
.card-clean .meta  { color: #94a3b8; font-size: 11px; margin-top: 6px; }

.card-coaching {
    background: #fffbeb;
    border-left: 4px solid #d97706;
    padding: 12px 16px;
    margin: 6px 0;
    border-radius: 0 8px 8px 0;
    box-shadow: 0 1px 3px rgba(217,119,6,0.08);
}
.card-coaching .text  { color: #78350f; font-size: 13px; line-height: 1.6; }
.card-coaching .meta  { color: #94a3b8; font-size: 11px; margin-top: 6px; }
.card-coaching .name  { color: #d97706; font-weight: 700; font-size: 12px; }

hr { border-color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT THEME
# ─────────────────────────────────────────────────────────────────────────────
DARK = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor ="#f8fafc",
    font = dict(family="Inter", color="#475569", size=11),
    xaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0", zerolinecolor="#e2e8f0"),
    yaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0", zerolinecolor="#e2e8f0"),
    margin=dict(t=44, b=36, l=40, r=16),
)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for key in ["scored_df", "store_health", "name_leaderboard",
            "overall_nps", "store_nps"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 NPS Fraud Engine")
    st.markdown("**Trends Retail · Reliance Retail**")
    st.markdown("---")

    st.markdown("##### Detection Layers")
    en_l1 = st.checkbox("L1 · Duplicate RRID",        value=True)
    en_l2 = st.checkbox("L2 · Store Contamination",   value=True)
    en_l3 = st.checkbox("L3 · Velocity Anomaly",      value=True)
    en_l4 = st.checkbox("L4 · Text Fingerprinting",   value=True)
    en_l5 = st.checkbox("L5 · Scoring Contradiction", value=True)

    if st.session_state.scored_df is not None:
        st.markdown("---")
        summ = get_data_summary(st.session_state.scored_df)
        st.caption(f"**{summ['total_responses']:,}** responses")
        st.caption(f"**{summ['unique_stores']:,}** stores")
        d0 = format_date_with_ordinal(summ["date_min"])
        d1 = format_date_with_ordinal(summ["date_max"])
        st.caption(f"{d0} → {d1}")

    st.markdown("---")
    st.caption("PRD v1.0 · March 2026")
    st.caption("Classification: Confidential")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _run_pipeline_bigquery():
    df = load_nps_data_from_bigquery()
    return run_fraud_engine(df)


def _store_results(scored_df, store_health, name_lb):
    st.session_state.scored_df        = scored_df
    st.session_state.store_health     = store_health
    st.session_state.name_leaderboard = name_lb
    st.session_state.overall_nps      = compute_overall_nps(scored_df)
    st.session_state.store_nps        = compute_store_nps(scored_df, store_health)


if st.session_state.scored_df is None:
    with st.spinner("Loading from BigQuery & running fraud engine…"):
        try:
            scored_df, store_health, name_lb = _run_pipeline_bigquery()
            _store_results(scored_df, store_health, name_lb)
            st.rerun()
        except Exception as e:
            st.error(f"Error loading from BigQuery: {e}")
            st.exception(e)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# NPS Fraud Detection Engine")
st.markdown("*Reported NPS · Clean NPS · Store Intelligence · Verbatim Analysis*")
st.markdown("---")

if st.session_state.scored_df is None:
    # ── Landing screen ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:60px 0 40px;color:#94a3b8">
        <div style="font-size:3.5rem;margin-bottom:14px">🔍</div>
        <div style="font-size:1.3rem;color:#475569;margin-bottom:6px">
            Loading data from BigQuery…
        </div>
        <div style="font-size:0.88rem;color:#94a3b8">
            5-layer detection · Staff coaching detection · Verbatim intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(5)
    labels = [
        ("L1", "Duplicate RRID", "Same respondent submitting multiple times"),
        ("L2", "Store Contamination", "All-perfect responses at fraud-flagged stores"),
        ("L3", "Velocity Anomaly", "Few people generating many responses in one day"),
        ("L4", "Text Fingerprint", "Copy-pasted feedback within a store"),
        ("L5", "Contradiction", "Logically impossible sub-rating vs NPS combinations"),
    ]
    for col, (badge, title, desc) in zip(cols, labels):
        with col:
            st.markdown(f"**{badge} · {title}**")
            st.caption(desc)
    st.stop()


# ── Pull from session state ───────────────────────────────────────────────────
scored_df     = st.session_state.scored_df
store_health  = st.session_state.store_health
name_lb       = st.session_state.name_leaderboard
nps           = st.session_state.overall_nps
store_nps     = st.session_state.store_nps


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊  National Overview",
    "🏪  Store Intelligence",
    "🗺️  State Intelligence",
    "💬  Verbatim Intelligence",
    "🚨  Top Fraud RRIDs",
    "🔬  Response Inspector",
    "📋  Export & Recommendations",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — NATIONAL OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## NPS — Reported vs Clean")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Reported NPS",     f"{nps['reported_nps']}")
    c2.metric("✅ Clean NPS",      f"{nps['clean_nps']}",
              delta=f"−{nps['nps_inflation']} pts inflation", delta_color="inverse")
    c3.metric("Total Responses",  f"{nps['total_responses']:,}")
    c4.metric("Fraud Flagged",    f"{nps['fraud_count']:,}",
              delta=f"{nps['fraud_pct']}%", delta_color="off")
    c5.metric("Clean Responses",  f"{nps['clean_responses']:,}")

    st.markdown("---")

    # ── NPS composition bars ──────────────────────────────────────────────
    def _nps_bar(counts: dict, title: str) -> go.Figure:
        cats   = ["Promoter", "Passive", "Detractor"]
        colors = {"Promoter": "#22c55e", "Passive": "#f59e0b", "Detractor": "#ef4444"}
        total  = sum(counts.get(c, 0) for c in cats) or 1
        fig = go.Figure(go.Bar(
            x=[counts.get(c, 0) / total * 100 for c in cats],
            y=cats, orientation="h",
            marker_color=[colors[c] for c in cats],
            text=[f"{counts.get(c, 0):,}  ({counts.get(c,0)/total*100:.0f}%)" for c in cats],
            textposition="inside",
            textfont=dict(color="white", size=11),
        ))
        fig.update_layout(**DARK, height=210, showlegend=False,
                          xaxis_title="% of responses",
                          title=dict(text=title, font=dict(size=13, color="#b0bfd8")))
        return fig

    col_r, col_c = st.columns(2)
    with col_r:
        st.plotly_chart(_nps_bar(nps["reported_counts"], "Reported NPS (all responses)"),
                        use_container_width=True)
    with col_c:
        st.plotly_chart(_nps_bar(nps["clean_counts"], "✅ Clean NPS (fraud excluded)"),
                        use_container_width=True)

    # ── Layer breakdown ───────────────────────────────────────────────────
    st.markdown("## Fraud by Detection Layer")
    ldf = compute_layer_breakdown(scored_df)

    fig_l = go.Figure(go.Bar(
        x=ldf["count"], y=ldf["layer_label"], orientation="h",
        marker_color="#1a56db", marker_line_width=0,
        text=[f"{r:,}  ({p:.1f}%)" for r, p in zip(ldf["count"], ldf["pct_of_total"])],
        textposition="outside",
        textfont=dict(color="#6b7fa3", size=10),
    ))
    fig_l.update_layout(**DARK, height=310, xaxis_title="Responses flagged",
                        title=dict(text="Layer breakdown", font=dict(size=13, color="#b0bfd8")))
    fig_l.update_yaxes(autorange="reversed", gridcolor="#181f2e", linecolor="#181f2e")
    st.plotly_chart(fig_l, use_container_width=True)

    # ── Layer Legend ──────────────────────────────────────────────────────
    with st.expander("📖 Detection Layer Legend — click to expand", expanded=False):

        st.markdown("#### How each layer works and what it flags")
        st.markdown("---")

        # ── L1 ────────────────────────────────────────────────────────────
        st.markdown("### L1 · Duplicate RRID")
        st.caption("Scope: rolling 7-day window per RRID · Weight: 35 pts")
        l1_col1, l1_col2 = st.columns(2)
        with l1_col1:
            st.markdown("""
**🔴 RRID_HEAVY_DUP** — Heavy Duplicate

Same phone number (RRID) submitted **3 or more times** within any 7-day window.

→ **ALL** submissions from that RRID in that window are removed, including the first.

*Why: If someone submits 3+ times, no submission from that RRID is trustworthy.*
            """)
        with l1_col2:
            st.markdown("""
**🟠 RRID_LIGHT_DUP** — Light Duplicate

Same phone number submitted **exactly twice** within any 7-day window.

→ Only the **second** (repeat) submission is removed. The first is kept clean.

*Why: A genuine customer might accidentally submit twice — we only remove the duplicate, not the original.*
            """)

        st.markdown("---")

        # ── L2 ────────────────────────────────────────────────────────────
        st.markdown("### L2 · Store Contamination")
        st.caption("Scope: rolling 7-day window per store · Weight: 20 pts")
        l2_col1, l2_col2 = st.columns(2)
        with l2_col1:
            st.markdown("""
**How a store window gets declared contaminated**

Within each 7-day window at a store, the engine checks three metrics.
**Any one** being true = window is contaminated:

| Metric | Threshold |
|---|---|
| Dup Ratio | >30% responses from repeat RRIDs |
| Heavy Dup Count | ≥5 responses from RRIDs submitting 3+ times |
| Perfect Rate | >90% all-perfect with ≥15 responses in window |

*All metrics are computed within the 7-day window only — clean history from previous months does not dilute current fraud.*
            """)
        with l2_col2:
            st.markdown("""
**🔴 CONTAMINATED_STORE_PERFECT**

Once a window is contaminated, every response in that window that is **all-perfect** gets flagged.

**All-perfect** = every sub-rating = 5/5 AND NPS = 10.

→ Non-perfect responses at a contaminated store are **not** removed.

→ All-perfect responses from the **same store in clean windows** are **not** removed.

*Why: A store being gamed cannot be trusted for perfect scores. But a genuine 7/10 response at a bad store is still real data.*
            """)

        st.markdown("---")

        # ── L3 ────────────────────────────────────────────────────────────
        st.markdown("### L3 · Velocity Anomaly")
        st.caption("Scope: single store-day · Weight: 20 pts")
        l3_col1, l3_col2 = st.columns(2)
        with l3_col1:
            st.markdown("""
**🔴 VELOCITY_ANOMALY**

A store-day is flagged when **all three** conditions are true simultaneously:

| Condition | Threshold |
|---|---|
| Responses that day | ≥ 6 |
| Unique RRID ratio | < 70% (few phones, many surveys) |
| All-perfect rate | > 80% |

→ Every response on that store-day is flagged.
            """)
        with l3_col2:
            st.markdown("""
**What this catches**

Staff filling surveys at the billing counter. Customers hand over their phone → staff tap through the survey → 5-5-5-5-5-5-5-5-5-5 and NPS 10.

The pattern is unmistakable: same day, same store, very few unique phones, almost all perfect.

*Why all three must be true: a genuine busy store on a Saturday might have many responses and high scores — but it would have near-100% unique RRIDs.*
            """)

        st.markdown("---")

        # ── L4 ────────────────────────────────────────────────────────────
        st.markdown("### L4 · Feedback Text Fingerprinting")
        st.caption("Scope: full dataset · 4 scenarios in priority order · First occurrence always clean")

        l4a, l4b = st.columns(2)
        with l4a:
            st.markdown("""
**🔴 RRID_EXACT_COPY** — Weight: 40 pts

Same phone number submitted the **exact same verbatim text** 2 or more times.

→ All submissions after the first are flagged.

---

**🟠 RRID_SIMILAR_COPY** — Weight: 35 pts

Same phone number submitted **similar text** (matching fingerprint) 2 or more times.

Fingerprint = first 6 meaningful words after removing filler words.

→ All submissions after the first are flagged.
            """)
        with l4b:
            st.markdown("""
**🟡 STORE_EXACT_COPY** — Weight: 25 pts

**Different** phones submitted the **exact same text** at the same store,
3 or more times, from at least 2 different RRIDs.

→ All submissions after the first are flagged. Catches verbally coached responses.

---

**⚪ STORE_SIMILAR_COPY** — Weight: 15 pts

**Different** phones submitted **similar text** at the same store,
4 or more times, from at least 2 different RRIDs.

→ All submissions after the first are flagged. Catches verbally coached paraphrases.
            """)

        st.caption(
            "Priority order: RRID_EXACT > RRID_SIMILAR > STORE_EXACT > STORE_SIMILAR. "
            "A response can only receive one L4 flag — highest confidence wins."
        )

        st.markdown("---")

        # ── L5 ────────────────────────────────────────────────────────────
        st.markdown("### L5 · Scoring Contradiction")
        st.caption("Scope: per response · Weight: 35 pts · Logically impossible rating patterns")

        l5a, l5b, l5c = st.columns(3)
        with l5a:
            st.markdown("""
**🔴 MONOTONE_MISMATCH**

All sub-ratings are **identical** AND all ≤ 3
BUT NPS = 10.

*Example: every sub-rating = 2 out of 5, yet customer is a promoter?*

Impossible. Sub-ratings were filled randomly, NPS was coached to 10.
            """)
        with l5b:
            st.markdown("""
**🔴 EXTREME_CONTRADICTION**

Average sub-rating ≤ 3
AND NPS ≥ 9.

*Example: avg store experience = 2.4/5 yet NPS = 10.*

Customer rates the store poorly on every dimension but is still a promoter. Contradictory.
            """)
        with l5c:
            st.markdown("""
**🔴 REVERSE_CONTRADICTION**

Average sub-rating ≥ 4.5
AND NPS ≤ 3.

*Example: avg sub-rating = 4.8/5 yet NPS = 1.*

Customer rates everything excellent but would not recommend. Contradictory — likely misunderstood the scale.
            """)

        st.markdown("---")

        # ── Fraud Score + Disposition ─────────────────────────────────────
        st.markdown("### Fraud Score & Disposition")
        fd_col1, fd_col2 = st.columns(2)
        with fd_col1:
            st.markdown("""
**How the fraud score is calculated**

Each triggered layer contributes its weight. Capped at 100.

| Layer | Flag | Weight |
|---|---|---|
| L1 | RRID_HEAVY_DUP / RRID_LIGHT_DUP | 35 |
| L2 | CONTAMINATED_STORE_PERFECT | 20 |
| L3 | VELOCITY_ANOMALY | 20 |
| L4 | RRID_EXACT_COPY | 40 |
| L4 | RRID_SIMILAR_COPY | 35 |
| L4 | STORE_EXACT_COPY | 25 |
| L4 | STORE_SIMILAR_COPY | 15 |
| L5 | Any contradiction | 35 |

*Example: L1 + L4 RRID_EXACT + L5 = 35+40+35 = 110 → capped at 100*
            """)
        with fd_col2:
            st.markdown("""
**How disposition is assigned**

| Disposition | Condition |
|---|---|
| ✅ CLEAN | No layers triggered (score = 0) |
| 🟡 QUARANTINED | ≥1 layer triggered AND layers < 3 AND score < 80 |
| 🔴 REJECTED | ≥3 layers triggered OR score ≥ 80 |

**Only CLEAN responses are used to compute Clean NPS.**

Both QUARANTINED and REJECTED are excluded from the Clean NPS calculation.

QUARANTINED = suspicious but not definitive evidence.
REJECTED = definitive fraud — multiple independent signals.
            """)

    # ── Disposition pie ───────────────────────────────────────────────────
    col_pie, col_trend = st.columns(2)
    with col_pie:
        st.markdown("## Disposition Split")
        dc = scored_df["disposition"].value_counts().reset_index()
        dc.columns = ["disposition", "count"]
        fig_pie = go.Figure(go.Pie(
            labels=dc["disposition"], values=dc["count"], hole=0.55,
            marker_colors=[
                {"CLEAN": "#22c55e", "QUARANTINED": "#f59e0b", "REJECTED": "#ef4444"}.get(d, "#334")
                for d in dc["disposition"]
            ],
            textinfo="label+percent",
            textfont=dict(size=12, color="#dde3f0"),
        ))
        fig_pie.update_layout(**DARK, height=300, showlegend=False,
                              title=dict(text="Disposition", font=dict(size=13, color="#b0bfd8")))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_trend:
        st.markdown("## Daily NPS Trend")
        tdf = compute_nps_trend(scored_df, freq="D")
        if len(tdf) > 1:
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=tdf["date"], y=tdf["reported_nps"], name="Reported",
                line=dict(color="#f59e0b", width=2), mode="lines+markers",
                marker=dict(size=4),
            ))
            fig_t.add_trace(go.Scatter(
                x=tdf["date"], y=tdf["clean_nps"], name="✅ Clean",
                line=dict(color="#22c55e", width=2), mode="lines+markers",
                marker=dict(size=4),
            ))
            fig_t.update_layout(**DARK, height=300, yaxis_title="NPS",
                                legend=dict(font=dict(color="#8896b3")),
                                title=dict(text="Reported vs Clean NPS",
                                           font=dict(size=13, color="#b0bfd8")))
            st.plotly_chart(fig_t, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — STORE INTELLIGENCE
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Store-Level Fraud Intelligence")

    n_cont = int(store_health[STORE_COL].nunique()) if (store_health is not None and len(store_health) > 0) else 0
    n_cont_windows = int(len(store_health)) if store_health is not None else 0
    n_critical = int((store_nps["risk_level"] == "CRITICAL").sum())
    n_high     = int((store_nps["risk_level"] == "HIGH").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Contaminated Stores", n_cont, 
              delta=f"{n_cont_windows} contaminated windows",
              delta_color="off")
    c2.metric("🔴 Critical Risk", n_critical)
    c3.metric("🟠 High Risk", n_high)

    # Filters row
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        risk_filter = st.selectbox("Filter by risk", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"], key="store_intel_risk")
    with filter_col2:
        # Get unique states for dropdown
        if "store_state" in scored_df.columns:
            unique_states = ["All"] + sorted(scored_df["store_state"].dropna().unique().tolist())
            selected_state = st.selectbox("🗺️ Filter by State", unique_states, key="store_intel_state")
        else:
            selected_state = "All"
    with filter_col3:
        # Get unique cities for dropdown (filtered by state if state is selected)
        if "store_city" in scored_df.columns:
            city_df = scored_df[["store_city", "store_state"]].dropna()
            if selected_state != "All" and "store_state" in scored_df.columns:
                city_df = city_df[city_df["store_state"] == selected_state]
            unique_cities = ["All"] + sorted(city_df["store_city"].dropna().unique().tolist())
            selected_city = st.selectbox("🏙️ Filter by City", unique_cities, key="store_intel_city")
        else:
            selected_city = "All"
    
    disp = store_nps.copy()

    # ── Build fraud reason columns per store ─────────────────────────────────
    fraud_only = scored_df[
        scored_df["is_fraud"] & (scored_df["fraud_reasons"].str.len() > 0)
    ].copy()

    if len(fraud_only) > 0:
        exploded = (
            fraud_only[[STORE_COL, "fraud_reasons"]]
            .assign(reason=fraud_only["fraud_reasons"].str.split("|"))
            .explode("reason")
        )
        exploded["reason"] = exploded["reason"].str.strip()
        exploded = exploded[exploded["reason"] != ""]

        # Friendly short labels for each flag code
        flag_labels = {
            "RRID_HEAVY_DUP":             "L1·HeavyDup",
            "RRID_LIGHT_DUP":             "L1·LightDup",
            "CONTAMINATED_STORE_PERFECT": "L2·ContamPerf",
            "VELOCITY_ANOMALY":           "L3·Velocity",
            "RRID_EXACT_COPY":            "L4·RRIDExact",
            "RRID_SIMILAR_COPY":          "L4·RRIDSimilar",
            "STORE_EXACT_COPY":           "L4·StoreExact",
            "STORE_SIMILAR_COPY":         "L4·StoreSimilar",
            "MONOTONE_MISMATCH":          "L5·Monotone",
            "EXTREME_CONTRADICTION":      "L5·Extreme",
            "REVERSE_CONTRADICTION":      "L5·Reverse",
        }
        exploded["reason_label"] = exploded["reason"].map(flag_labels).fillna(exploded["reason"])

        reason_pivot = (
            exploded
            .groupby([STORE_COL, "reason_label"])
            .size()
            .reset_index(name="cnt")
            .pivot_table(
                index=STORE_COL,
                columns="reason_label",
                values="cnt",
                fill_value=0,
                aggfunc="sum",
            )
            .reset_index()
        )
        reason_pivot.columns.name = None

        # Format each reason column as "label (count)" in a single 
        # "Fraud Reasons" text column for compact display
        reason_flag_cols = [c for c in reason_pivot.columns if c != STORE_COL]

        def _format_reasons(row):
            parts = []
            # Sort by count descending so biggest signal appears first
            sorted_cols = sorted(reason_flag_cols, key=lambda c: row.get(c, 0), reverse=True)
            for col in sorted_cols:
                val = int(row.get(col, 0))
                if val > 0:
                    parts.append(f"{col} ({val})")
            return " · ".join(parts) if parts else "—"

        reason_pivot["Fraud Reasons"] = reason_pivot[reason_flag_cols].apply(
            _format_reasons, axis=1
        )

        disp = disp.merge(
            reason_pivot[[STORE_COL, "Fraud Reasons"]],
            on=STORE_COL,
            how="left",
        )
        disp["Fraud Reasons"] = disp["Fraud Reasons"].fillna("—")
    else:
        disp["Fraud Reasons"] = "—"

    store_meta = scored_df[["entity_id", "store_name", "store_city", "store_state"]].drop_duplicates("entity_id")
    disp = disp.merge(store_meta, on="entity_id", how="left")

    if risk_filter != "All":
        disp = disp[disp["risk_level"] == risk_filter]
    if selected_state != "All" and "store_state" in disp.columns:
        # Case-insensitive and strip whitespace for robust matching
        state_mask = (
            disp["store_state"].notna() & 
            (disp["store_state"].astype(str).str.strip().str.lower() == selected_state.strip().lower())
        )
        disp = disp[state_mask]
    if selected_city != "All" and "store_city" in disp.columns:
        # Case-insensitive and strip whitespace for robust matching
        city_mask = (
            disp["store_city"].notna() & 
            (disp["store_city"].astype(str).str.strip().str.lower() == selected_city.strip().lower())
        )
        disp = disp[city_mask]

    show_cols = [c for c in [
        STORE_COL, "store_name", "store_city", "store_state",
        "total_responses", "fraud_count", "fraud_pct",
        "Fraud Reasons",
        "reported_nps", "clean_nps", "nps_inflation",
        "all_perfect_pct", "risk_level",
    ] if c in disp.columns]

    rename = {
        STORE_COL:          "Store",
        "store_name":       "Store Name",
        "store_city":       "City",
        "store_state":      "State",
        "total_responses":  "Total",
        "fraud_count":      "Fraud #",
        "fraud_pct":        "Fraud %",
        "Fraud Reasons":    "Fraud Reasons",
        "reported_nps":     "Reported NPS",
        "clean_nps":        "✅ Clean NPS",
        "nps_inflation":    "Inflation",
        "all_perfect_pct":  "All-Perfect %",
        "risk_level":       "Risk",
    }
    
    if len(disp) == 0:
        st.info(f"No stores found matching the selected filters. Try adjusting your filters (Risk: {risk_filter}, State: {selected_state}, City: {selected_city}).")
    else:
        st.dataframe(
            disp[show_cols].head(200).rename(columns=rename),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Fraud %": st.column_config.ProgressColumn(
                    "Fraud %", format="%.1f%%", min_value=0, max_value=100),
                "Fraud Reasons": st.column_config.TextColumn(
                    "Fraud Reasons", 
                    width="large",
                    help="Each fraud flag and how many responses it caught at this store"
                ),
                "✅ Clean NPS": st.column_config.NumberColumn("✅ Clean NPS", format="%.1f"),
                "Reported NPS": st.column_config.NumberColumn("Reported NPS", format="%.1f"),
                "Inflation":    st.column_config.NumberColumn("Inflation",    format="%.1f"),
            }
        )

    # ── Scatter ───────────────────────────────────────────────────────────
    st.markdown("## Fraud % vs Response Volume")
    scatter_df = disp.head(300).copy()
    if len(scatter_df) > 0 and "nps_inflation" in scatter_df.columns:
        scatter_df["_size"] = scatter_df["nps_inflation"].clip(lower=0.5).fillna(0.5)
    else:
        scatter_df["_size"] = 0.5
    
    fig_sc = px.scatter(
        scatter_df,
        x="total_responses", y="fraud_pct",
        size="_size", color="risk_level",
        hover_name=STORE_COL,
        color_discrete_map={
            "CRITICAL": "#ef4444", "HIGH": "#f97316",
            "MEDIUM": "#64748b",   "LOW":  "#22c55e",
        },
        labels={"total_responses": "Total Responses", "fraud_pct": "Fraud %"},
    )
    fig_sc.update_layout(**DARK, height=380,
                         title=dict(text="Fraud Map (bubble size = NPS inflation)",
                                    font=dict(size=13, color="#b0bfd8")),
                         legend=dict(font=dict(color="#8896b3")))
    st.plotly_chart(fig_sc, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — STATE INTELLIGENCE
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## State-Level Fraud & NPS Intelligence")

    if "store_state" in scored_df.columns and scored_df["store_state"].notna().any():
        state_col = "store_state"
    elif "state" in scored_df.columns:
        state_col = "state"
    else:
        st.warning("No state data found in dataset.")
        state_col = None

    if state_col is None:
        st.info("State-level analysis requires a 'state' or 'entity_geo' column.")
    else:
        state_rows = []
        from src.nps_calculator import nps as calc_nps
        for state, grp in scored_df.groupby(state_col):
            clean = grp[~grp["is_fraud"]]
            state_rows.append({
                "State": state,
                "Total Responses": len(grp),
                "Genuine": len(clean),
                "Fraud": int(grp["is_fraud"].sum()),
                "Fraud %": round(grp["is_fraud"].mean() * 100, 1),
                "Reported NPS": calc_nps(grp[NPS_COL]),
                "Clean NPS": calc_nps(clean[NPS_COL]),
                "NPS Delta": round(calc_nps(clean[NPS_COL]) - calc_nps(grp[NPS_COL]), 1),
            })

        state_df = pd.DataFrame(state_rows).sort_values("Fraud %", ascending=False).reset_index(drop=True)

        worst_state = state_df.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Worst State (Fraud %)", f"{worst_state['State']}", f"{worst_state['Fraud %']}% fraud")
        c2.metric("States with >20% Fraud", str(len(state_df[state_df["Fraud %"] > 20])))
        c3.metric("Max NPS Inflation", f"{state_df['NPS Delta'].min()} pts", "most negative = worst")

        st.markdown("### State Fraud Rankings")
        st.dataframe(
            state_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Fraud %": st.column_config.ProgressColumn("Fraud %", format="%.1f%%", min_value=0, max_value=100),
                "Reported NPS": st.column_config.NumberColumn("Reported NPS", format="%.1f"),
                "Clean NPS": st.column_config.NumberColumn("✅ Clean NPS", format="%.1f"),
                "NPS Delta": st.column_config.NumberColumn("NPS Delta", format="%.1f"),
            }
        )

        top_states = state_df[state_df["Fraud %"] > 5].head(20)
        fig_state = go.Figure()
        fig_state.add_trace(go.Bar(
            x=top_states["State"], y=top_states["Reported NPS"],
            name="Reported NPS", marker_color="#f59e0b"
        ))
        fig_state.add_trace(go.Bar(
            x=top_states["State"], y=top_states["Clean NPS"],
            name="✅ Clean NPS", marker_color="#22c55e"
        ))
        fig_state.update_layout(
            barmode="group",
            title=dict(text="Reported vs Clean NPS by State", font=dict(size=13)),
            xaxis_tickangle=-45,
            height=420,
            legend=dict(orientation="h"),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(family="Inter", color="#475569", size=11),
            margin=dict(t=44, b=80, l=40, r=16),
        )
        st.plotly_chart(fig_state, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — VERBATIM INTELLIGENCE
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Verbatim Intelligence — Staff Coaching Detection")
    st.markdown(
        "Identifies stores where staff are coaching customers to name them in the survey. "
        "A staff name appearing **3+ times at the same store** is a coaching signal."
    )

    # ── Top-line numbers ──────────────────────────────────────────────────
    coached_df    = scored_df[scored_df["staff_coaching_flag"] == "STAFF_COACHING"]
    name_detected = scored_df["staff_name_detected"].sum() if "staff_name_detected" in scored_df.columns else 0
    coached_stores= coached_df[STORE_COL].nunique() if len(coached_df) else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Responses with Staff Name",  f"{int(name_detected):,}")
    c2.metric("Stores with Coaching Signal",f"{coached_stores}")
    c3.metric("Coached Responses Flagged",  f"{len(coached_df):,}")
    c4.metric("Unique Staff Names Detected",
              f"{scored_df['suspected_staff_name'].replace('', np.nan).nunique():,}"
              if "suspected_staff_name" in scored_df.columns else "—")

    st.markdown("---")

    # ── Staff Name Leaderboard ─────────────────────────────────────────────
    st.markdown("### 🚨 Staff Name Leaderboard")
    st.caption("Same staff name repeated 3+ times at one store = coaching signal")

    if name_lb is not None and len(name_lb) > 0:
        coaching_lb = name_lb[name_lb["mention_count"] >= 3].copy()
        if len(coaching_lb):
            coaching_lb["first_seen"] = pd.to_datetime(coaching_lb["first_seen"]).apply(format_date_with_ordinal)
            coaching_lb["last_seen"]  = pd.to_datetime(coaching_lb["last_seen"]).apply(format_date_with_ordinal)

            lb_store_meta = scored_df[["entity_id", "store_name", "store_city", "store_state"]].drop_duplicates("entity_id")
            coaching_lb = coaching_lb.merge(lb_store_meta, on="entity_id", how="left")

            lb_display = coaching_lb.rename(columns={
                STORE_COL:          "Store",
                "store_name":       "Store Name",
                "store_city":       "City",
                "store_state":      "State",
                "suspected_staff_name": "Staff Name",
                "mention_count":    "Mentions",
                "unique_rrids":     "Unique RRIDs",
                "first_seen":       "First Seen",
                "last_seen":        "Last Seen",
                "sample_verbatim":  "Sample Verbatim",
            })
            lb_col_order = [c for c in [
                "Store", "Store Name", "City", "State", "Staff Name",
                "Mentions", "Unique RRIDs", "First Seen", "Last Seen", "Sample Verbatim",
            ] if c in lb_display.columns]

            st.dataframe(
                lb_display[lb_col_order].head(100),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Mentions": st.column_config.ProgressColumn(
                        "Mentions", format="%d", min_value=0,
                        max_value=int(name_lb["mention_count"].max()) if len(name_lb) else 10
                    ),
                    "Sample Verbatim": st.column_config.TextColumn("Sample Verbatim", width="large"),
                }
            )

            # ── Bar chart: top names ──────────────────────────────────────
            top20 = coaching_lb.head(20)
            fig_names = go.Figure(go.Bar(
                x=top20["mention_count"],
                y=top20["suspected_staff_name"] + "  (" + top20[STORE_COL].astype(str) + ")",
                orientation="h",
                marker_color="#f59e0b",
                marker_line_width=0,
                text=top20["mention_count"],
                textposition="outside",
                textfont=dict(color="#6b7fa3", size=10),
            ))
            fig_names.update_layout(
                **DARK, height=max(250, len(top20) * 26),
                xaxis_title="Mentions",
                title=dict(text="Top staff names by store (coaching signals)",
                           font=dict(size=13, color="#b0bfd8")),
            )
            fig_names.update_yaxes(autorange="reversed", gridcolor="#181f2e", linecolor="#181f2e")
            st.plotly_chart(fig_names, use_container_width=True)
        else:
            st.info("No coaching signals detected (no staff name mentioned 3+ times at same store).")
    else:
        st.info("No staff name data available.")

    st.markdown("---")

    # ── Per-Store Verbatim Explorer ───────────────────────────────────────
    st.markdown("### 📖 Store Verbatim Explorer")
    st.caption("Select a store to read all verbatim responses side by side — fraud vs genuine")

    # Sort stores by coached_responses desc for easier discovery
    if len(coached_df):
        coached_store_order = (
            coached_df.groupby(STORE_COL).size()
            .sort_values(ascending=False).index.tolist()
        )
        other_stores = [
            s for s in sorted(scored_df[STORE_COL].dropna().unique())
            if s not in coached_store_order
        ]
        store_order = coached_store_order + other_stores
    else:
        store_order = sorted(scored_df[STORE_COL].dropna().unique().tolist())

    selected_store = st.selectbox("Select Store", store_order,
                                   help="Stores with coaching signals are listed first", key="verbatim_store_select")

    store_df = scored_df[scored_df[STORE_COL] == selected_store].copy()

    store_meta_row = store_df[["store_name", "store_city", "store_state"]].iloc[0] if len(store_df) else None
    if store_meta_row is not None:
        st.markdown(f"**{store_meta_row['store_name']}** · {store_meta_row['store_city']}, {store_meta_row['store_state']}")

    # Mini store summary
    s_total   = len(store_df)
    s_fraud   = int(store_df["is_fraud"].sum())
    s_coached = int((store_df.get("staff_coaching_flag", "") == "STAFF_COACHING").sum())
    s_names   = store_df["suspected_staff_name"].replace("", np.nan).dropna().unique() \
        if "suspected_staff_name" in store_df.columns else []

    info_c = st.columns(4)
    info_c[0].metric("Total Responses", s_total)
    info_c[1].metric("Fraud Flagged",   s_fraud)
    info_c[2].metric("Coached Responses", s_coached)
    info_c[3].metric("Staff Names Found", len(s_names))

    if len(s_names):
        names_str = ", ".join(str(n) for n in s_names[:10])
        st.markdown(f"**Detected names:** `{names_str}`")

    # ── Two-column verbatim layout ────────────────────────────────────────
    col_fraud, col_clean = st.columns(2)

    verbatim_col = "verbatim" if "verbatim" in store_df.columns else "feedback_clean"

    # ── FRAUD side ────────────────────────────────────────────────────────
    with col_fraud:
        st.markdown("#### 🔴 Fraud-Flagged Verbatims")
        fraud_rows = store_df[
            store_df["is_fraud"] &
            (store_df[verbatim_col].str.len() > 0)
        ].sort_values(DATE_COL)

        if len(fraud_rows) == 0:
            st.caption("No fraud responses with verbatim at this store.")

        for _, row in fraud_rows.head(50).iterrows():
            verbatim = str(row.get(verbatim_col, "")).strip()
            if not verbatim:
                continue
            nps_val = int(row[NPS_COL]) if not pd.isna(row.get(NPS_COL)) else "—"
            date_str = format_date_with_ordinal(row.get(DATE_COL, ""))
            staff_name = str(row.get("suspected_staff_name", "")).strip()
            reasons = str(row.get("fraud_reasons", "")).replace("|", " · ")
            coaching_badge = "🚨 COACHED · " if row.get("staff_coaching_flag") == "STAFF_COACHING" else ""
            name_tag = f"👤 <strong>{staff_name}</strong> · " if staff_name else ""
            store_city_val = str(row.get("store_city", "")).strip()
            store_state_val = str(row.get("store_state", "")).strip()
            location_tag = f"{store_city_val}, {store_state_val} · " if store_city_val else ""

            display_text = verbatim
            if staff_name:
                import re as _re
                display_text = _re.sub(
                    rf"\b{_re.escape(staff_name)}\b",
                    f'<strong style="color:#dc2626">{staff_name}</strong>',
                    display_text, flags=_re.IGNORECASE
                )

            st.markdown(f"""
            <div class="card-fraud">
                <div class="text">"{display_text}"</div>
                <div class="meta">{coaching_badge}{name_tag}{location_tag}NPS <strong>{nps_val}</strong> · {date_str}</div>
                <div class="meta" style="margin-top:3px;color:#dc2626;font-size:10px;font-family:monospace">{reasons}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── GENUINE side ──────────────────────────────────────────────────────
    with col_clean:
        st.markdown("#### ✅ Genuine Verbatims")
        clean_rows = store_df[
            ~store_df["is_fraud"] &
            (store_df[verbatim_col].str.len() > 0)
        ].sort_values(DATE_COL)

        if len(clean_rows) == 0:
            st.caption("No genuine responses with verbatim at this store.")

        for _, row in clean_rows.head(50).iterrows():
            verbatim = str(row.get(verbatim_col, "")).strip()
            if not verbatim:
                continue
            date_str = format_date_with_ordinal(row.get(DATE_COL, ""))
            nps_val  = int(row[NPS_COL]) if not pd.isna(row.get(NPS_COL)) else "—"
            store_city_val = str(row.get("store_city", "")).strip()
            store_state_val = str(row.get("store_state", "")).strip()
            location_tag = f"{store_city_val}, {store_state_val} · " if store_city_val else ""

            st.markdown(f"""
            <div class="card-clean">
                <div class="text">"{verbatim}"</div>
                <div class="meta">{location_tag}NPS <strong>{nps_val}</strong> · {date_str}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── All verbatims with staff names (across all stores) ────────────────
    st.markdown("---")
    st.markdown("### 🔎 All Coaching Verbatims Across Network")
    st.caption("Every response where a staff name was detected and meets the 3+ threshold")

    if len(coached_df) and verbatim_col in coached_df.columns:
        coaching_view = coached_df[coached_df[verbatim_col].str.len() > 0][[
            STORE_COL, verbatim_col, "suspected_staff_name",
            NPS_COL, DATE_COL, "fraud_reasons"
        ]].rename(columns={
            STORE_COL:               "Store",
            verbatim_col:            "Verbatim",
            "suspected_staff_name":  "Staff Name",
            NPS_COL:                 "NPS",
            DATE_COL:                "Date",
            "fraud_reasons":         "Fraud Flags",
        })
        coaching_view["Date"] = pd.to_datetime(coaching_view["Date"]).apply(format_date_with_ordinal)
        st.dataframe(
            coaching_view.sort_values("Store").head(500),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Verbatim":    st.column_config.TextColumn("Verbatim",    width="large"),
                "Fraud Flags": st.column_config.TextColumn("Fraud Flags", width="medium"),
            }
        )
    else:
        st.info("No coaching verbatims found.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — TOP FRAUD RRIDs
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 🚨 Top Fraud RRIDs — Ballot Stuffing Evidence")
    st.markdown("Respondent IDs with the most submissions. These are the strongest evidence of store staff filling surveys.")

    if "fraud_layer_count" in scored_df.columns:
        rrid_summary = (
            scored_df[scored_df["is_fraud"]]
            .groupby(RRID_COL)
            .agg(
                submissions=(RRID_COL, "count"),
                stores=(STORE_COL, "nunique"),
                store_list=(STORE_COL, lambda x: ", ".join(x.unique()[:3])),
                all_perfect=("is_all_perfect", "all"),
                avg_nps=(NPS_COL, "mean"),
                fraud_layers=("fraud_reasons", "first"),
                first_seen=(DATE_COL, "min"),
                last_seen=(DATE_COL, "max"),
            )
            .reset_index()
            .sort_values("submissions", ascending=False)
        )
        rrid_summary["all_perfect"] = rrid_summary["all_perfect"].map({True: "YES ✓", False: "No"})
        rrid_summary["avg_nps"] = rrid_summary["avg_nps"].round(1)
        rrid_summary["first_seen"] = pd.to_datetime(rrid_summary["first_seen"]).apply(format_date_with_ordinal)
        rrid_summary["last_seen"] = pd.to_datetime(rrid_summary["last_seen"]).apply(format_date_with_ordinal)

        # Merge store metadata — pick first store per RRID for display
        rrid_first_store = scored_df[scored_df["is_fraud"]].drop_duplicates(RRID_COL)[[RRID_COL, STORE_COL]]
        rrid_summary = rrid_summary.merge(rrid_first_store, on=RRID_COL, how="left")
        rrid_store_meta = scored_df[["entity_id", "store_name", "store_city", "store_state"]].drop_duplicates("entity_id")
        rrid_summary = rrid_summary.merge(rrid_store_meta, on="entity_id", how="left")

        c1, c2, c3 = st.columns(3)
        c1.metric("Unique Fraud RRIDs", f"{len(rrid_summary):,}")
        c2.metric("Max Submissions (1 RRID)", f"{int(rrid_summary['submissions'].max())}")
        c3.metric("RRIDs with 5+ Submissions", f"{len(rrid_summary[rrid_summary['submissions'] >= 5]):,}")

        top20 = rrid_summary.head(20)
        fig_rrid = go.Figure(go.Bar(
            x=top20["submissions"],
            y=top20[RRID_COL].str[:16] + "…",
            orientation="h",
            marker_color="#ef4444",
            text=top20["submissions"],
            textposition="outside",
        ))
        fig_rrid.update_layout(
            title="Top 20 Worst Offender RRIDs (by submission count)",
            yaxis=dict(autorange="reversed"),
            xaxis_title="Number of submissions",
            height=500,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(family="Inter", color="#475569", size=11),
            margin=dict(t=44, b=36, l=180, r=60),
        )
        st.plotly_chart(fig_rrid, use_container_width=True)

        st.markdown("### Full RRID Leaderboard")
        rrid_display = rrid_summary.head(100).copy()
        # Ensure dates are formatted as strings
        rrid_display["first_seen"] = rrid_display["first_seen"].astype(str)
        rrid_display["last_seen"] = rrid_display["last_seen"].astype(str)
        rrid_display = rrid_display.rename(columns={
            RRID_COL: "RRID",
            "submissions": "Submissions",
            "stores": "Stores",
            "store_list": "Store Codes",
            STORE_COL: "Store",
            "store_name":  "Store Name",
            "store_city":  "City",
            "store_state": "State",
            "all_perfect": "All Perfect?",
            "avg_nps": "Avg NPS",
            "fraud_layers": "Fraud Flags",
            "first_seen": "First Seen",
            "last_seen": "Last Seen",
        })
        rrid_col_order = [c for c in [
            "RRID", "Submissions", "Store Codes", "Store Name", "City", "State",
            "Stores", "All Perfect?", "Avg NPS", "Fraud Flags", "First Seen", "Last Seen",
        ] if c in rrid_display.columns]

        st.dataframe(
            rrid_display[rrid_col_order],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Submissions": st.column_config.ProgressColumn(
                    "Submissions", format="%d", min_value=0,
                    max_value=int(rrid_summary["submissions"].max())
                ),
                "Fraud Flags": st.column_config.TextColumn("Fraud Flags", width="large"),
                "First Seen": st.column_config.TextColumn("First Seen"),
                "Last Seen": st.column_config.TextColumn("Last Seen"),
            }
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — RESPONSE INSPECTOR
# ═════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## Response Inspector")

    s_col, r_col, state_col, city_col = st.columns(4)
    with s_col:
        store_search = st.text_input("🏪 Filter by Store Code")
    with r_col:
        rrid_search = st.text_input("👤 Filter by RRID")
    with state_col:
        # Get unique states for dropdown
        if "store_state" in scored_df.columns:
            unique_states = ["All"] + sorted(scored_df["store_state"].dropna().unique().tolist())
            selected_state = st.selectbox("🗺️ Filter by State", unique_states, key="response_inspector_state")
        else:
            selected_state = "All"
    with city_col:
        # Get unique cities for dropdown (filtered by state if state is selected)
        if "store_city" in scored_df.columns:
            city_df = scored_df[["store_city", "store_state"]].dropna()
            if selected_state != "All" and "store_state" in scored_df.columns:
                city_df = city_df[city_df["store_state"] == selected_state]
            unique_cities = ["All"] + sorted(city_df["store_city"].dropna().unique().tolist())
            selected_city = st.selectbox("🏙️ Filter by City", unique_cities, key="response_inspector_city")
        else:
            selected_city = "All"

    fraud_reason_col, fraud_only_col = st.columns(2)
    with fraud_reason_col:
        # Get unique fraud reasons from pipe-delimited fraud_reasons column
        if "fraud_reasons" in scored_df.columns:
            all_reasons = scored_df["fraud_reasons"].dropna()
            if len(all_reasons) > 0:
                # Split pipe-delimited reasons and get unique values
                unique_reasons = set()
                for reasons_str in all_reasons:
                    if reasons_str and reasons_str.strip():
                        reasons_list = [r.strip() for r in str(reasons_str).split("|") if r.strip()]
                        unique_reasons.update(reasons_list)
                unique_reasons = ["All"] + sorted(list(unique_reasons))
                selected_fraud_reason = st.selectbox("🚨 Filter by Fraud Reason", unique_reasons, key="response_inspector_fraud_reason")
            else:
                selected_fraud_reason = "All"
        else:
            selected_fraud_reason = "All"
    with fraud_only_col:
        fraud_only = st.checkbox("Show fraud responses only", value=True)

    ins = scored_df.copy()
    if store_search.strip():
        ins = ins[ins[STORE_COL].astype(str).str.contains(store_search.strip(), case=False)]
    if rrid_search.strip():
        ins = ins[ins[RRID_COL].astype(str).str.contains(rrid_search.strip(), case=False)]
    if selected_state != "All" and "store_state" in ins.columns:
        ins = ins[ins["store_state"] == selected_state]
    if selected_city != "All" and "store_city" in ins.columns:
        ins = ins[ins["store_city"] == selected_city]
    if selected_fraud_reason != "All" and "fraud_reasons" in ins.columns:
        ins = ins[ins["fraud_reasons"].str.contains(selected_fraud_reason, na=False, case=False)]
    if fraud_only:
        ins = ins[ins["is_fraud"]]

    st.caption(f"{len(ins):,} responses")

    vb_col = "verbatim" if "verbatim" in ins.columns else "feedback_clean"
    show = [c for c in [
        RRID_COL, STORE_COL, "store_name", "store_city", "store_state", 
        DATE_COL, NPS_COL, vb_col,
        "suspected_staff_name", "fraud_score", "fraud_layer_count",
        "fraud_reasons", "disposition",
    ] if c in ins.columns]

    # Format dates before displaying
    ins_display = ins[show].head(500).copy()
    if DATE_COL in ins_display.columns:
        ins_display[DATE_COL] = ins_display[DATE_COL].apply(format_date_with_ordinal)
    
    # Rename columns for display
    display_rename = {
        STORE_COL: "Store Code",
        RRID_COL: "Phone Number",
        DATE_COL: "Date",
        NPS_COL: "NPS",
        "store_name": "Store Name",
        "store_city": "City",
        "store_state": "State",
        "fraud_score": "Fraud Score",
        "fraud_layer_count": "Layers",
        "fraud_reasons": "Fraud Reasons",
        "disposition": "Disposition",
        "suspected_staff_name": "Staff Name",
        vb_col: "Verbatim",
    }
    
    st.dataframe(
        ins_display.rename(columns=display_rename),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Fraud Score": st.column_config.ProgressColumn(
                "Fraud Score", format="%d", min_value=0, max_value=100),
            "Fraud Reasons":         st.column_config.TextColumn("Fraud Reasons", width="large"),
            "Verbatim":              st.column_config.TextColumn("Verbatim",      width="large"),
            "Staff Name":            st.column_config.TextColumn("Staff Name"),
            "Date":                  st.column_config.TextColumn("Date"),
        }
    )

    # RRID timeline
    if rrid_search.strip():
        st.markdown("### RRID Submission Timeline")
        rrid_hist = scored_df[
            scored_df[RRID_COL].astype(str).str.contains(rrid_search.strip(), case=False)
        ][[RRID_COL, STORE_COL, DATE_COL, NPS_COL, vb_col, "fraud_reasons", "disposition"]].copy()
        if len(rrid_hist):
            # Sort first, then format dates
            rrid_hist = rrid_hist.sort_values(DATE_COL)
            rrid_hist["Date_Formatted"] = rrid_hist[DATE_COL].apply(format_date_with_ordinal)
            rrid_display = rrid_hist.rename(columns={
                RRID_COL: "Phone Number",
                STORE_COL: "Store Code",
                NPS_COL: "NPS",
                vb_col: "Verbatim",
                "fraud_reasons": "Fraud Reasons",
                "disposition": "Disposition",
            })
            rrid_display = rrid_display.drop(columns=[DATE_COL]).rename(columns={"Date_Formatted": "Date"})
            st.dataframe(rrid_display, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 — EXPORT & RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("## Data Export")

    fraud_df   = scored_df[scored_df["is_fraud"]].copy()
    coached_ex = scored_df[scored_df.get("staff_coaching_flag", pd.Series("")) == "STAFF_COACHING"].copy() \
        if "staff_coaching_flag" in scored_df.columns else pd.DataFrame()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "⬇  Fraud Responses (CSV)",
            data=fraud_df.to_csv(index=False).encode("utf-8"),
            file_name="nps_fraud_responses.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "⬇  Store Intelligence (CSV)",
            data=store_nps.to_csv(index=False).encode("utf-8"),
            file_name="nps_store_intelligence.csv",
            mime="text/csv",
        )
    with c3:
        if len(coached_ex):
            st.download_button(
                "⬇  Coaching Verbatims (CSV)",
                data=coached_ex.to_csv(index=False).encode("utf-8"),
                file_name="nps_coaching_verbatims.csv",
                mime="text/csv",
            )

    st.markdown("### Fraud Summary by Layer")
    ldf = compute_layer_breakdown(scored_df)
    st.dataframe(
        ldf[["layer_label", "count", "pct_of_total"]].rename(columns={
            "layer_label":  "Layer",
            "count":        "Flagged Responses",
            "pct_of_total": "% of Total",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Preview: Fraud Responses (first 1,000)")
    vb_col = "verbatim" if "verbatim" in fraud_df.columns else "feedback_clean"
    pr_cols = [c for c in [
        RRID_COL, STORE_COL, DATE_COL, NPS_COL, vb_col,
        "suspected_staff_name", "staff_coaching_flag",
        "fraud_score", "fraud_layer_count", "fraud_reasons", "disposition",
    ] if c in fraud_df.columns]
    st.dataframe(fraud_df[pr_cols].head(1000), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("## 📌 Recommendations & Action Plan")
    st.caption("Generated dynamically based on fraud patterns detected in this dataset.")

    recs = []

    # Pull key stats
    total = len(scored_df)
    fraud_pct = nps["fraud_pct"]
    reported = nps["reported_nps"]
    clean = nps["clean_nps"]
    inflation = nps["nps_inflation"]

    # Top critical stores
    critical_stores = store_nps[store_nps["risk_level"] == "CRITICAL"].head(10)
    critical_store_list = ", ".join(critical_stores[STORE_COL].astype(str).tolist())
    if len(critical_stores) > 0:
        recs.append({
            "Priority": "🔴 IMMEDIATE",
            "Action": f"Investigate {len(critical_stores)} CRITICAL stores: {critical_store_list}",
            "Rationale": f"These stores have >40% fraud rates. Store staff are almost certainly filling surveys.",
            "Impact": f"Removes majority of definitive fraud responses"
        })

    # Duplicate RRID blocking
    l1_count = int(scored_df["fraud_reasons"].str.contains("RRID_HEAVY_DUP", na=False).sum())
    max_submissions = scored_df.groupby(RRID_COL).size().max() if len(scored_df) else 0
    if l1_count > 0:
        recs.append({
            "Priority": "🔴 IMMEDIATE",
            "Action": f"Block repeat submissions — enforce 1 response per RRID per transaction",
            "Rationale": f"{l1_count:,} responses flagged for duplicate RRID. Worst offender submitted {max_submissions}x. This single fix eliminates Layer 1 fraud entirely.",
            "Impact": f"Eliminates {l1_count:,} duplicate responses instantly"
        })

    # Velocity anomaly = survey at billing counter
    l3_count = int(scored_df["fraud_reasons"].str.contains("VELOCITY_ANOMALY", na=False).sum())
    if l3_count > 0:
        recs.append({
            "Priority": "🟠 SHORT-TERM",
            "Action": "Delay survey delivery by 2–24 hours post-transaction",
            "Rationale": f"{l3_count:,} responses show velocity clustering — many perfect responses in one store on one day. Staff are filling surveys at the billing counter. Delayed delivery prevents this.",
            "Impact": "Structurally prevents on-site staff-driven fraud"
        })

    # Copy-paste feedback = attention checks needed
    l4_rrid_exact  = int(scored_df["fraud_reasons"].str.contains("RRID_EXACT_COPY", na=False).sum())
    l4_rrid_similar = int(scored_df["fraud_reasons"].str.contains("RRID_SIMILAR_COPY", na=False).sum())
    l4_store_exact = int(scored_df["fraud_reasons"].str.contains("STORE_EXACT_COPY", na=False).sum())
    l4_store_similar = int(scored_df["fraud_reasons"].str.contains("STORE_SIMILAR_COPY", na=False).sum())
    l4_count = l4_rrid_exact + l4_rrid_similar + l4_store_exact + l4_store_similar
    if l4_count > 0:
        breakdown = []
        if l4_rrid_exact:  breakdown.append(f"RRID exact: {l4_rrid_exact:,}")
        if l4_rrid_similar: breakdown.append(f"RRID similar: {l4_rrid_similar:,}")
        if l4_store_exact: breakdown.append(f"Store exact: {l4_store_exact:,}")
        if l4_store_similar: breakdown.append(f"Store similar: {l4_store_similar:,}")
        recs.append({
            "Priority": "🟠 SHORT-TERM",
            "Action": "Introduce 2–3 attention-check questions randomly in survey",
            "Rationale": f"{l4_count:,} responses flagged for copy-paste feedback ({', '.join(breakdown)}). Identical or near-identical text submitted repeatedly by the same RRID or at the same store. Attention checks will fail staff speed-filling.",
            "Impact": "Separates lazy-genuine responses from coordinated robo-filling"
        })

    # Staff coaching detected
    coaching_count = int((scored_df.get("staff_coaching_flag", pd.Series("")) == "STAFF_COACHING").sum())
    coached_stores = scored_df[scored_df.get("staff_coaching_flag", pd.Series("")) == "STAFF_COACHING"][STORE_COL].nunique() if coaching_count > 0 else 0
    if coaching_count > 0:
        recs.append({
            "Priority": "🟠 SHORT-TERM",
            "Action": f"Conduct staff interviews at {coached_stores} stores with coaching signals",
            "Rationale": f"{coaching_count:,} responses show repeated staff names in verbatims — a strong indicator of managers coaching staff to prompt customers to name them. Investigate these stores directly.",
            "Impact": "Addresses root behaviour driving coached responses"
        })

    # High contaminated store count = device fingerprinting
    cont_count = int(store_health[STORE_COL].nunique()) if (store_health is not None and len(store_health) > 0) else 0
    if cont_count > 10:
        recs.append({
            "Priority": "🟠 SHORT-TERM",
            "Action": "Add device fingerprinting beyond RRID (IP address, browser, device ID)",
            "Rationale": f"{cont_count} stores are contaminated. Staff may create new RRIDs to bypass RRID-based blocking. Device fingerprinting catches this workaround.",
            "Impact": "Closes the gap after RRID blocking is enforced"
        })

    # Always: real-time pipeline
    recs.append({
        "Priority": "🟡 MEDIUM-TERM",
        "Action": "Build automated fraud scoring into the live NPS pipeline",
        "Rationale": "Currently fraud is detected retrospectively. These 5 detection layers should run in real-time so fraudulent responses never enter the NPS calculation.",
        "Impact": "Permanently clean NPS going forward — no manual analysis needed"
    })

    # High national fraud % = incentive misalignment
    if fraud_pct > 10:
        recs.append({
            "Priority": "🔵 POLICY",
            "Action": "Decouple store-level NPS from staff incentives and performance targets",
            "Rationale": f"National fraud rate is {fraud_pct}%. The root cause is incentive misalignment — if staff bonuses depend on NPS, they will game it. Shift to operational metrics (billing speed, return rates).",
            "Impact": "Removes the motive for fraud at source"
        })

    # Always: publish clean NPS
    recs.append({
        "Priority": "🔵 POLICY",
        "Action": f"Publish Clean NPS ({clean}) in all internal dashboards — not Reported NPS ({reported})",
        "Rationale": f"Reported NPS of {reported} is inflated by {inflation} points. Decisions made on this number are based on false data. Clean NPS of {clean} is the only actionable figure.",
        "Impact": "Ensures every business decision is based on real customer sentiment"
    })

    st.dataframe(
        pd.DataFrame(recs),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Priority": st.column_config.TextColumn("Priority", width="small"),
            "Action": st.column_config.TextColumn("Action", width="large"),
            "Rationale": st.column_config.TextColumn("Rationale", width="large"),
            "Impact": st.column_config.TextColumn("Impact", width="medium"),
        }
    )
