import sqlite3
from pathlib import Path

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DB_PATH = Path("models/monitoring.db")
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Model Monitoring",
    page_icon="📡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS -- cards, badges, typography. Injected once at the top.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1100px; }

    .kpi-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        border-left: 4px solid #4C8BF5;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #9aa0a6;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 600;
        line-height: 1.1;
    }

    .status-banner {
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        font-weight: 500;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .status-ok { background: rgba(52,168,83,0.12); color: #34a853; border: 1px solid rgba(52,168,83,0.35); }
    .status-drift { background: rgba(234,67,53,0.12); color: #ea4335; border: 1px solid rgba(234,67,53,0.35); }
    .status-wait { background: rgba(66,133,244,0.12); color: #4c8bf5; border: 1px solid rgba(66,133,244,0.35); }

    .feature-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.55rem 0.9rem;
        border-radius: 8px;
        margin-bottom: 0.4rem;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
    }
    .feature-name { font-size: 0.92rem; }
    .feature-pval { font-size: 0.8rem; color: #9aa0a6; margin-right: 0.6rem; }
    .badge {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
    }
    .badge-drift { background: rgba(234,67,53,0.18); color: #ea4335; }
    .badge-ok { background: rgba(52,168,83,0.18); color: #34a853; }

    h1 { font-size: 1.6rem !important; font-weight: 600 !important; }
    h3 { font-size: 1.05rem !important; font-weight: 600 !important; margin-top: 1.6rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("# 📡 Subscription upgrade model — monitoring")
st.caption("Live view of production traffic, prediction outcomes, and data drift.")

if not DB_PATH.exists():
    st.info("No predictions logged yet. Send a few requests to the `/predict` endpoint first.")
    st.stop()

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM predictions ORDER BY id ASC", conn)
conn.close()

if df.empty:
    st.info("No predictions logged yet.")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
kpis = [
    (c1, "Total predictions logged", f"{len(df):,}"),
    (c2, "Upgrade rate (predicted)", f"{df['prediction'].mean() * 100:.1f}%"),
    (c3, "Avg. probability", f"{df['probability'].mean():.3f}"),
]
for col, label, value in kpis:
    col.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Drift status
# ---------------------------------------------------------------------------
st.markdown("### Drift status")

drift = None
try:
    drift = httpx.get(f"{API_URL}/monitoring/drift", timeout=5).json()
except httpx.ConnectError:
    st.markdown(
        '<div class="status-banner status-drift">⚠️ Can\'t reach the API at localhost:8000 — is it running?</div>',
        unsafe_allow_html=True,
    )

if drift is not None:
    if drift.get("reason"):
        st.markdown(
            f'<div class="status-banner status-wait">⏳ {drift["reason"]}</div>',
            unsafe_allow_html=True,
        )
    elif drift["drifted"]:
        st.markdown(
            '<div class="status-banner status-drift">🚨 Drift detected in one or more features — consider retraining.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-banner status-ok">✅ No drift detected — production data still matches the training distribution.</div>',
            unsafe_allow_html=True,
        )

    if drift.get("details"):
        left, right = st.columns([1, 1])

        with left:
            for feature, info in drift["details"].items():
                badge_class = "badge-drift" if info["drifted"] else "badge-ok"
                badge_text = "drifted" if info["drifted"] else "stable"
                st.markdown(
                    f'<div class="feature-row">'
                    f'<span class="feature-name">{feature.replace("_", " ")}</span>'
                    f'<span><span class="feature-pval">p={info["p_value"]}</span>'
                    f'<span class="badge {badge_class}">{badge_text}</span></span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with right:
            pvals = drift["details"]
            fig = go.Figure(
                go.Bar(
                    y=[f.replace("_", " ") for f in pvals.keys()],
                    x=[v["p_value"] for v in pvals.values()],
                    orientation="h",
                    marker_color=["#ea4335" if v["drifted"] else "#34a853" for v in pvals.values()],
                )
            )
            fig.add_vline(x=0.05, line_dash="dash", line_color="#9aa0a6", annotation_text="threshold (0.05)")
            fig.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="p-value",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0a6"),
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction trend over time
# ---------------------------------------------------------------------------
st.markdown("### Prediction probability over time")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["probability"],
        mode="lines+markers",
        line=dict(color="#4c8bf5", width=2),
        marker=dict(
            size=6,
            color=["#34a853" if p == 1 else "#9aa0a6" for p in df["prediction"]],
        ),
        name="upgrade probability",
    )
)
fig.add_hline(y=0.5, line_dash="dash", line_color="#9aa0a6", annotation_text="decision threshold")
fig.update_layout(
    height=320,
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis_title="upgrade probability",
    xaxis_title="time",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#9aa0a6"),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Recent predictions table
# ---------------------------------------------------------------------------
st.markdown("### Recent predictions")

display_df = df.sort_values("id", ascending=False).head(50).copy()
display_df["prediction"] = display_df["prediction"].map({1: "upgrade", 0: "no upgrade"})
display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
display_df = display_df.drop(columns=["id"]).rename(columns=lambda c: c.replace("_", " "))

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "probability": st.column_config.ProgressColumn(
            "probability", min_value=0, max_value=1, format="%.3f"
        ),
    },
)
