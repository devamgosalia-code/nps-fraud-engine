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

from config import STORE_COL, RRID_COL, DATE_COL, NPS_COL
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
        d0 = str(summ["date_min"])[:10]
        d1 = str(summ["date_max"])[:10]
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  National Overview",
    "🏪  Store Intelligence",
    "💬  Verbatim Intelligence",
    "🔬  Response Inspector",
    "📋  Export",
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

    n_cont     = int(store_health["is_contaminated"].sum()) if store_health is not None else "—"
    n_critical = int((store_nps["risk_level"] == "CRITICAL").sum())
    n_high     = int((store_nps["risk_level"] == "HIGH").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Contaminated Stores", n_cont)
    c2.metric("🔴 Critical Risk", n_critical)
    c3.metric("🟠 High Risk", n_high)

    risk_filter = st.selectbox("Filter by risk", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    disp = store_nps.copy()
    if risk_filter != "All":
        disp = disp[disp["risk_level"] == risk_filter]

    show_cols = [c for c in [
        STORE_COL, "total_responses", "fraud_count", "fraud_pct",
        "reported_nps", "clean_nps", "nps_inflation",
        "all_perfect_pct", "risk_level",
    ] if c in disp.columns]

    rename = {
        STORE_COL:          "Store",
        "total_responses":  "Total",
        "fraud_count":      "Fraud #",
        "fraud_pct":        "Fraud %",
        "reported_nps":     "Reported NPS",
        "clean_nps":        "✅ Clean NPS",
        "nps_inflation":    "Inflation",
        "all_perfect_pct":  "All-Perfect %",
        "risk_level":       "Risk",
    }
    st.dataframe(
        disp[show_cols].head(200).rename(columns=rename),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Fraud %": st.column_config.ProgressColumn(
                "Fraud %", format="%.1f%%", min_value=0, max_value=100),
            "✅ Clean NPS": st.column_config.NumberColumn("✅ Clean NPS", format="%.1f"),
            "Reported NPS": st.column_config.NumberColumn("Reported NPS", format="%.1f"),
            "Inflation":    st.column_config.NumberColumn("Inflation",    format="%.1f"),
        }
    )

    # ── Scatter ───────────────────────────────────────────────────────────
    st.markdown("## Fraud % vs Response Volume")
    sz = disp.head(300)["nps_inflation"].clip(lower=0.5).fillna(0.5)
    fig_sc = px.scatter(
        disp.head(300),
        x="total_responses", y="fraud_pct",
        size=sz, color="risk_level",
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
# TAB 3 — VERBATIM INTELLIGENCE
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
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
            coaching_lb["first_seen"] = pd.to_datetime(coaching_lb["first_seen"]).dt.date
            coaching_lb["last_seen"]  = pd.to_datetime(coaching_lb["last_seen"]).dt.date

            st.dataframe(
                coaching_lb.rename(columns={
                    STORE_COL:          "Store",
                    "suspected_staff_name": "Staff Name",
                    "mention_count":    "Mentions",
                    "unique_rrids":     "Unique RRIDs",
                    "first_seen":       "First Seen",
                    "last_seen":        "Last Seen",
                    "sample_verbatim":  "Sample Verbatim",
                }).head(100),
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
                                   help="Stores with coaching signals are listed first")

    store_df = scored_df[scored_df[STORE_COL] == selected_store].copy()

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
            date_str = str(row.get(DATE_COL, ""))[:10]
            staff_name = str(row.get("suspected_staff_name", "")).strip()
            reasons = str(row.get("fraud_reasons", "")).replace("|", " · ")
            coaching_badge = "🚨 COACHED · " if row.get("staff_coaching_flag") == "STAFF_COACHING" else ""
            name_tag = f"👤 <strong>{staff_name}</strong> · " if staff_name else ""

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
                <div class="meta">{coaching_badge}{name_tag}NPS <strong>{nps_val}</strong> · {date_str}</div>
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
            date_str = str(row.get(DATE_COL, ""))[:10]
            nps_val  = int(row[NPS_COL]) if not pd.isna(row.get(NPS_COL)) else "—"

            st.markdown(f"""
            <div class="card-clean">
                <div class="text">"{verbatim}"</div>
                <div class="meta">NPS <strong>{nps_val}</strong> · {date_str}</div>
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
        coaching_view["Date"] = pd.to_datetime(coaching_view["Date"]).dt.date
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
# TAB 4 — RESPONSE INSPECTOR
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Response Inspector")

    s_col, r_col = st.columns(2)
    with s_col:
        store_search = st.text_input("🏪 Filter by Store Code")
    with r_col:
        rrid_search = st.text_input("👤 Filter by RRID")

    fraud_only = st.checkbox("Show fraud responses only", value=True)

    ins = scored_df.copy()
    if store_search.strip():
        ins = ins[ins[STORE_COL].astype(str).str.contains(store_search.strip(), case=False)]
    if rrid_search.strip():
        ins = ins[ins[RRID_COL].astype(str).str.contains(rrid_search.strip(), case=False)]
    if fraud_only:
        ins = ins[ins["is_fraud"]]

    st.caption(f"{len(ins):,} responses")

    vb_col = "verbatim" if "verbatim" in ins.columns else "feedback_clean"
    show = [c for c in [
        RRID_COL, STORE_COL, DATE_COL, NPS_COL, vb_col,
        "suspected_staff_name", "fraud_score", "fraud_layer_count",
        "fraud_reasons", "disposition",
    ] if c in ins.columns]

    st.dataframe(
        ins[show].head(500),
        use_container_width=True,
        hide_index=True,
        column_config={
            "fraud_score": st.column_config.ProgressColumn(
                "Fraud Score", format="%d", min_value=0, max_value=100),
            "fraud_reasons":         st.column_config.TextColumn("Fraud Reasons", width="large"),
            vb_col:                  st.column_config.TextColumn("Verbatim",      width="large"),
            "suspected_staff_name":  st.column_config.TextColumn("Staff Name"),
        }
    )

    # RRID timeline
    if rrid_search.strip():
        st.markdown("### RRID Submission Timeline")
        rrid_hist = scored_df[
            scored_df[RRID_COL].astype(str).str.contains(rrid_search.strip(), case=False)
        ][[RRID_COL, STORE_COL, DATE_COL, NPS_COL, vb_col, "fraud_reasons", "disposition"]]
        if len(rrid_hist):
            st.dataframe(rrid_hist.sort_values(DATE_COL), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXPORT
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
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
