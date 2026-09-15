"""
app.py
======
Fleet Readiness Copilot — Streamlit dashboard for the predictive maintenance
pipeline.  Loads pre-computed predictions and service records, builds the
maintenance queue via prioritize.py, generates Granite briefings via
explain.py, and renders an operational readiness interface.

Supports a mixed fleet with two asset modalities:
  ✈  engine  — C-MAPSS turbofan engines (asset IDs 1–100)
  ⚙  gearbox — IMS bearing / gearbox assets (asset IDs 1001–1004)

Both asset types are loaded and passed to run_readiness_check(), which routes
them to the appropriate model and returns a unified asset list.  The dashboard
shows an asset-type badge (✈ / ⚙) on every card and row so fleet composition
is immediately obvious.

Usage:
    streamlit run app.py
"""

import os
import warnings
import logging

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Suppress noisy library logging before any imports fire
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

# ── Pipeline imports ──────────────────────────────────────────────────────────
from pipeline import run_readiness_check          # single orchestration entrypoint
from explain import _humanize_feature_name        # UI helper only
from prioritize import W_URGENCY, W_CRITICALITY, W_TIME_PRESS, W_DISCREPANCY, W_PART_AVAIL


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fleet Readiness Copilot",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design system: muted dark-professional palette ────────────────────────────
CSS = """
<style>
/* ── Base ─────────────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f1117;
    color: #d4d8e0;
    font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #161b22; }

/* ── Typography ───────────────────────────────────────────────────────────── */
h1, h2, h3 { color: #e2e8f0; letter-spacing: 0.04em; font-weight: 600; }
.caption    { color: #6b7280; font-size: 0.78rem; }

/* ── Readiness strip ──────────────────────────────────────────────────────── */
.readiness-strip {
    display: flex; gap: 16px; margin-bottom: 24px;
    flex-wrap: wrap;
}
.stat-tile {
    background: #1c2130;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 18px 28px;
    min-width: 150px;
    flex: 1;
}
.stat-tile .label  { font-size: 0.72rem; text-transform: uppercase;
                     letter-spacing: 0.1em; color: #6b7280; margin-bottom: 4px; }
.stat-tile .number { font-size: 2.2rem; font-weight: 700; line-height: 1; }
.stat-tile .sub    { font-size: 0.78rem; color: #9ca3af; margin-top: 4px; }
.num-total   { color: #93c5fd; }
.num-ready   { color: #4ade80; }
.num-notready{ color: #f87171; }
.num-crit    { color: #fb923c; }

/* ── Asset cards ──────────────────────────────────────────────────────────── */
.asset-card {
    background: #1c2130;
    border: 1px solid #2d3748;
    border-left: 4px solid #374151;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 10px;
    transition: border-left-color 0.15s;
}
.asset-card.rank-1  { border-left-color: #ef4444; }
.asset-card.rank-2  { border-left-color: #f97316; }
.asset-card.rank-3  { border-left-color: #f59e0b; }
.asset-card.rank-4, .asset-card.rank-5 { border-left-color: #a78bfa; }

.card-header    { display: flex; align-items: baseline; gap: 12px; margin-bottom: 8px; }
.rank-badge     { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
                  color: #6b7280; text-transform: uppercase; }
.asset-id       { font-size: 1.15rem; font-weight: 700; color: #e2e8f0; }
.crit-badge     { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
                  background: #7c3aed22; color: #a78bfa;
                  border: 1px solid #7c3aed55; border-radius: 4px;
                  padding: 2px 8px; text-transform: uppercase; }
.pill           { display: inline-block; border-radius: 4px;
                  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
                  padding: 2px 10px; text-transform: uppercase; }
.pill-red       { background: #ef444422; color: #fca5a5; border: 1px solid #ef444455; }
.pill-green     { background: #22c55e22; color: #86efac; border: 1px solid #22c55e55; }
.pill-amber     { background: #f59e0b22; color: #fcd34d; border: 1px solid #f59e0b55; }

.meta-grid      { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
                  gap: 8px 20px; margin-top: 10px; }
.meta-item .k   { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em;
                  color: #6b7280; }
.meta-item .v   { font-size: 0.9rem; color: #d4d8e0; font-weight: 500; }

.briefing-box   { background: #111827; border: 1px solid #1f2937;
                  border-radius: 6px; padding: 14px 16px; margin-top: 12px;
                  font-size: 0.88rem; line-height: 1.65; color: #c9d1dc; }
.briefing-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em;
                  color: #4b5563; margin-bottom: 6px; }

/* ── Ask copilot ──────────────────────────────────────────────────────────── */
.copilot-response {
    background: #111827;
    border: 1px solid #1f2937;
    border-left: 3px solid #3b82f6;
    border-radius: 6px;
    padding: 16px 18px;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #c9d1dc;
}
.copilot-label  { font-size: 0.68rem; text-transform: uppercase;
                  letter-spacing: 0.1em; color: #3b82f6; margin-bottom: 8px; }

/* ── Score bar ────────────────────────────────────────────────────────────── */
.score-bar-wrap { margin: 10px 0 2px; }
.score-bar-bg   { background: #1f2937; border-radius: 4px; height: 6px;
                  overflow: hidden; }
.score-bar-fill { height: 6px; border-radius: 4px;
                  background: linear-gradient(90deg, #f59e0b, #ef4444); }

/* ── Divider ──────────────────────────────────────────────────────────────── */
hr { border: none; border-top: 1px solid #1f2937; margin: 20px 0; }

/* ── Streamlit widget overrides ───────────────────────────────────────────── */
.stTextInput > div > div > input {
    background: #161b22 !important;
    color: #d4d8e0 !important;
    border-color: #2d3748 !important;
    border-radius: 6px !important;
}
.stButton > button {
    background: #1d4ed8 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
}
.stButton > button:hover { background: #2563eb !important; }
div[data-testid="stExpander"] {
    background: #1c2130;
    border: 1px solid #2d3748 !important;
    border-radius: 8px !important;
}
div[data-testid="stExpander"] summary {
    color: #9ca3af !important;
    font-size: 0.82rem !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rul_color(rul: float) -> str:
    if rul <= 20:  return "#ef4444"
    if rul <= 40:  return "#f59e0b"
    return "#4ade80"

def _ready_pill(is_ready: bool, rul: float) -> str:
    if is_ready:
        return '<span class="pill pill-green">GO</span>'
    if rul <= 20:
        return '<span class="pill pill-red">NO-GO</span>'
    return '<span class="pill pill-amber">MARGINAL</span>'

def _stock_pill(in_stock: bool, lead: int) -> str:
    if in_stock:
        return '<span class="pill pill-green">IN STOCK</span>'
    color = "pill-red" if lead >= 14 else "pill-amber"
    return f'<span class="pill {color}">PROCURE · {lead}d</span>'

def _rank_class(rank: int) -> str:
    return f"rank-{rank}" if rank <= 5 else ""

def _score_bar_html(score: float) -> str:
    pct = min(100, int(score * 100))
    return (
        f'<div class="score-bar-wrap">'
        f'  <div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%"></div></div>'
        f'</div>'
    )

def _humanize(fname: str) -> str:
    return _humanize_feature_name(fname)


# ── Data loading (cached so the heavy pipeline runs only once) ────────────────

@st.cache_data(show_spinner="Running readiness assessment…")
def load_pipeline_data():
    """
    Load merged asset dataframes for both modalities and pass them to
    pipeline.run_readiness_check() as a single combined fleet DataFrame.

    Engine assets come from output_real/train_FD001_merged.parquet.
    Gearbox assets come from output_real/ims_bearing_merged.parquet
    (if present — missing file is skipped gracefully so the dashboard
    still works with engine-only data).

    Both frames are tagged with an asset_type column before concatenation
    so pipeline.py can route each row to the correct model.

    Paths are resolved relative to this file so the dashboard works
    regardless of which directory `streamlit run` is invoked from.
    """
    # Resolve paths relative to this file's directory, not CWD
    _here = os.path.dirname(os.path.abspath(__file__))
    dfs = []

    # Engine assets (C-MAPSS)
    engine_path = os.path.join(_here, "output_real", "train_FD001_merged.parquet")
    if os.path.exists(engine_path):
        eng_df = pd.read_parquet(engine_path)
        eng_df["asset_type"] = "engine"
        dfs.append(eng_df)
    else:
        st.warning(f"Engine data not found at: {engine_path}")

    # Gearbox / bearing assets (IMS)
    bearing_path = os.path.join(_here, "output_real", "ims_bearing_merged.parquet")
    if os.path.exists(bearing_path):
        gb_df = pd.read_parquet(bearing_path)
        gb_df["asset_type"] = "gearbox"
        dfs.append(gb_df)
    else:
        st.warning(f"Gearbox data not found at: {bearing_path}")

    if not dfs:
        st.error(
            "No asset data found.  Run etl_cmapss.py and/or etl_ims.py first."
        )
        st.stop()

    combined = pd.concat(dfs, ignore_index=True)
    return run_readiness_check(combined)


# ── Render helpers ────────────────────────────────────────────────────────────

def _asset_type_badge(asset_type: str) -> str:
    """Return a small inline HTML badge for engine (✈) or gearbox (⚙) assets."""
    if asset_type == "gearbox":
        return (
            '<span style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;'
            'background:#16313122;color:#67e8f9;border:1px solid #67e8f955;'
            'border-radius:4px;padding:2px 8px;text-transform:uppercase">⚙ GEARBOX</span>'
        )
    return (
        '<span style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;'
        'background:#1e293b;color:#93c5fd;border:1px solid #3b82f655;'
        'border-radius:4px;padding:2px 8px;text-transform:uppercase">✈ ENGINE</span>'
    )


def render_fleet_strip(assets: list, queue: list):
    total      = len(assets)
    ready      = sum(1 for a in assets if a["is_ready"])
    not_ready  = total - ready
    crit_nr    = sum(1 for a in assets if not a["is_ready"] and a["mission_critical_flag"])
    n_engines  = sum(1 for a in assets if a.get("asset_type") == "engine")
    n_gearbox  = sum(1 for a in assets if a.get("asset_type") == "gearbox")

    st.markdown(
        f"""
        <div class="readiness-strip">
          <div class="stat-tile">
            <div class="label">Fleet size</div>
            <div class="number num-total">{total}</div>
            <div class="sub">✈ {n_engines} engines &nbsp;·&nbsp; ⚙ {n_gearbox} gearboxes</div>
          </div>
          <div class="stat-tile">
            <div class="label">Mission Ready</div>
            <div class="number num-ready">{ready}</div>
            <div class="sub">{ready/total*100:.0f}% of fleet</div>
          </div>
          <div class="stat-tile">
            <div class="label">Not Ready</div>
            <div class="number num-notready">{not_ready}</div>
            <div class="sub">{not_ready/total*100:.0f}% of fleet</div>
          </div>
          <div class="stat-tile" style="border-left:3px solid #fb923c">
            <div class="label">⚠ Critical · Not Ready</div>
            <div class="number num-crit">{crit_nr}</div>
            <div class="sub">immediate action required</div>
          </div>
          <div class="stat-tile">
            <div class="label">In Maint. Queue</div>
            <div class="number" style="color:#c084fc">{len(queue)}</div>
            <div class="sub">ranked by priority</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_asset_card(item: dict, rank: int):
    aid       = item["asset_id"]
    rul       = item["predicted_rul"]
    is_rdy    = item["is_ready"]
    crit      = item.get("mission_critical_flag", False)
    score     = item.get("priority_score", 0.0)
    comps     = item.get("score_components", {})
    stock     = item.get("part_in_stock", True)
    lead      = item.get("part_lead_time_days", 0)
    disc      = item.get("open_discrepancies", 0)
    maint     = str(item.get("next_scheduled_maintenance", ""))[:10]
    insp      = str(item.get("last_inspection_date", ""))[:10]
    cum_fh    = item.get("cumulative_flight_hours", 0)
    briefing  = item.get("explanation", "")
    top_feats = item.get("top_features", [])
    atype     = item.get("asset_type", "engine")

    crit_html  = '<span class="crit-badge">MISSION CRITICAL</span>' if crit else ""
    type_badge = _asset_type_badge(atype)
    rul_col    = _rul_color(rul)
    rank_cls   = _rank_class(rank)
    # For gearbox assets use "UNIT" label instead of "TAIL" to distinguish visually
    id_label   = "UNIT" if atype == "gearbox" else "TAIL"
    rul_unit   = "steps" if atype == "gearbox" else "cycles"

    st.markdown(
        f"""
        <div class="asset-card {rank_cls}">
          <div class="card-header">
            <span class="rank-badge">#{rank}</span>
            <span class="asset-id">{id_label} {aid:04d}</span>
            {type_badge}
            {crit_html}
            {_ready_pill(is_rdy, rul)}
            {_stock_pill(stock, lead)}
          </div>
          <div class="meta-grid">
            <div class="meta-item">
              <div class="k">Predicted RUL</div>
              <div class="v" style="color:{rul_col};font-weight:700">{rul:.1f} {rul_unit}</div>
            </div>
            <div class="meta-item">
              <div class="k">Priority Score</div>
              <div class="v">{score:.4f}</div>
            </div>
            <div class="meta-item">
              <div class="k">Open Discrepancies</div>
              <div class="v">{disc}</div>
            </div>
            <div class="meta-item">
              <div class="k">Next Maint.</div>
              <div class="v">{maint}</div>
            </div>
            <div class="meta-item">
              <div class="k">Last Inspection</div>
              <div class="v">{insp}</div>
            </div>
            <div class="meta-item">
              <div class="k">Cumulative FH</div>
              <div class="v">{cum_fh:.0f} h</div>
            </div>
          </div>
          {_score_bar_html(score)}
          <div class="briefing-label">MAINTENANCE BRIEFING</div>
          <div class="briefing-box">{briefing}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Expandable evidence panel
    with st.expander(f"  Evidence · Score Components · Sensor Readings — Tail {aid:03d}"):
        col_a, col_b = st.columns([1, 1.4])
        with col_a:
            st.markdown("**Score Breakdown**")
            weight_map = {
                "urgency_score":           ("RUL Urgency",       W_URGENCY),
                "criticality_score":       ("Mission Critical",  W_CRITICALITY),
                "time_pressure_score":     ("Time Pressure",     W_TIME_PRESS),
                "discrepancy_score":       ("Open Write-ups",    W_DISCREPANCY),
                "part_availability_score": ("Parts Lead Time",   W_PART_AVAIL),
            }
            rows = []
            for k, (label, w) in weight_map.items():
                raw = comps.get(k, 0.0)
                weighted = round(raw * w, 4)
                rows.append({"Component": label, "Weight": f"{w:.0%}",
                             "Sub-score": f"{raw:.4f}", "Contribution": f"{weighted:.4f}"})
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                use_container_width=True,
            )

        with col_b:
            st.markdown("**Top Contributing Sensor Readings**")
            if top_feats:
                fig = go.Figure()
                labels = [_humanize(f) for f, _ in top_feats[:8]]
                values = [v for _, v in top_feats[:8]]
                colors = ["#ef4444" if v > np.percentile(values, 75)
                          else "#f59e0b" if v > np.percentile(values, 50)
                          else "#4ade80" for v in values]
                fig.add_trace(go.Bar(
                    x=values, y=labels,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:.2f}" for v in values],
                    textposition="outside",
                    textfont=dict(color="#9ca3af", size=10),
                ))
                fig.update_layout(
                    paper_bgcolor="#111827", plot_bgcolor="#111827",
                    font=dict(color="#9ca3af", size=10),
                    margin=dict(l=10, r=60, t=10, b=10),
                    height=260,
                    xaxis=dict(showgrid=False, zeroline=False,
                               tickfont=dict(color="#4b5563")),
                    yaxis=dict(tickfont=dict(color="#c9d1dc"), autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No sensor feature data available.")


def render_ready_asset_row(item: dict):
    """Compact single-line row for mission-ready assets."""
    aid   = item["asset_id"]
    rul   = item["predicted_rul"]
    crit  = item.get("mission_critical_flag", False)
    disc  = item.get("open_discrepancies", 0)
    maint = str(item.get("next_scheduled_maintenance", ""))[:10]
    atype = item.get("asset_type", "engine")
    icon  = "⚙" if atype == "gearbox" else "✈"
    label = "UNIT" if atype == "gearbox" else "TAIL"
    unit  = "steps" if atype == "gearbox" else "cycles"
    crit_tag = "⬡ CRITICAL" if crit else ""
    st.markdown(
        f"{icon} {label} **{aid:04d}** &nbsp;·&nbsp; "
        f"<span style='color:#4ade80;font-weight:700'>{rul:.0f} {unit}</span> remaining &nbsp;·&nbsp; "
        f"{disc} discrepancies &nbsp;·&nbsp; next maint {maint} &nbsp;&nbsp;"
        f"<span style='color:#a78bfa;font-size:0.78rem'>{crit_tag}</span>",
        unsafe_allow_html=True,
    )


def render_copilot(query: str, all_map: dict, queue: list):
    """Simple keyword-matched copilot response."""
    query_clean = query.strip().lower()
    if not query_clean:
        return

    # ── Asset ID extraction ───────────────────────────────────────────────────
    # Match up to 5 digits so gearbox IDs (1001-1004) are captured.
    # Also recognise "G-1001", "G1001", "unit 1001", "bearing 1001" prefixes.
    import re
    matched_asset = None

    # First pass: explicit G-prefix (e.g. "G-1001", "G1001")
    g_hits = re.findall(r"\bg-?(\d{3,5})\b", query_clean)
    for n in g_hits:
        aid = int(n)
        if aid in all_map:
            matched_asset = all_map[aid]
            break

    # Second pass: bare number (1-5 digits)
    if matched_asset is None:
        numbers = re.findall(r"\b(\d{1,5})\b", query_clean)
        for n in numbers:
            aid = int(n)
            if aid in all_map:
                matched_asset = all_map[aid]
                break

    # ── Keyword fallbacks ─────────────────────────────────────────────────────
    if matched_asset is None:
        urgency_kws  = ["worst", "most urgent", "highest priority", "critical first", "top"]
        ready_kws    = ["ready", "available", "flyable", "go"]
        gearbox_kws  = ["bearing", "vibration", "gearbox", "bpfo", "bpfi", "bsf", "ftf",
                         "unit", "gear"]

        if any(kw in query_clean for kw in urgency_kws):
            if queue:
                matched_asset = {**queue[0], **all_map.get(queue[0]["asset_id"], {})}

        elif any(kw in query_clean for kw in gearbox_kws):
            # "What gearbox/bearing assets exist?" — pick worst gearbox by RUL
            gearbox_assets = [a for a in all_map.values() if a.get("asset_type") == "gearbox"]
            if gearbox_assets:
                matched_asset = min(gearbox_assets, key=lambda x: x["predicted_rul"])

        elif any(kw in query_clean for kw in ready_kws):
            ready_assets = [a for a in all_map.values() if a["is_ready"]]
            if ready_assets:
                matched_asset = max(ready_assets, key=lambda x: x["predicted_rul"])

    if matched_asset is None:
        response_html = (
            "I couldn't find an asset matching that query. Try asking "
            "<em>\"Is Tail 43 ready?\"</em>, <em>\"Is Unit G-1003 ready?\"</em>, "
            "<em>\"What's the highest priority asset?\"</em>, "
            "<em>\"What's the worst bearing?\"</em>, "
            "or <em>\"Show me the most flyable asset.\"</em>"
        )
    else:
        aid   = matched_asset["asset_id"]
        rul   = matched_asset["predicted_rul"]
        rdy   = matched_asset["is_ready"]
        crit  = matched_asset.get("mission_critical_flag", False)
        brief = matched_asset.get("explanation", "No briefing available.")
        disc  = matched_asset.get("open_discrepancies", 0)
        stock = matched_asset.get("part_in_stock", True)
        lead  = matched_asset.get("part_lead_time_days", 0)
        atype = matched_asset.get("asset_type", "engine")

        # Modality-aware labels
        id_label  = "Unit" if atype == "gearbox" else "Tail"
        rul_unit  = "steps" if atype == "gearbox" else "cycles"
        aid_fmt   = f"G-{aid}" if atype == "gearbox" else f"{aid:03d}"

        verdict = (
            "<span style='color:#4ade80;font-weight:700'>MISSION READY (GO)</span>"
            if rdy else
            "<span style='color:#f87171;font-weight:700'>NOT MISSION READY (NO-GO)</span>"
        )
        crit_note = " This asset is flagged <strong>MISSION CRITICAL</strong>." if crit else ""
        parts_note = (
            f" Required part is <span style='color:#4ade80'>in stock</span>."
            if stock else
            f" Required part is <span style='color:#f59e0b'>not in stock</span> — "
            f"procurement lead time: <strong>{lead} days</strong>."
        )

        # Find its queue rank if it's in the queue
        queue_rank = next((q["priority_rank"] for q in queue if q["asset_id"] == aid), None)
        rank_note = (
            f" It is ranked <strong>#{queue_rank}</strong> in the maintenance queue."
            if queue_rank else
            " It does not currently appear in the maintenance queue."
        )

        response_html = (
            f"<strong>{id_label} {aid_fmt}</strong> is {verdict}.{crit_note}{parts_note}{rank_note}"
            f"<br><br>{brief}"
            f"<br><br><span style='color:#4b5563;font-size:0.8rem'>Predicted RUL: {rul:.1f} {rul_unit} &nbsp;·&nbsp; "
            f"Open discrepancies: {disc}</span>"
        )

    st.markdown(
        f'<div class="copilot-label">◆ COPILOT RESPONSE</div>'
        f'<div class="copilot-response">{response_html}</div>',
        unsafe_allow_html=True,
    )


# ── Main layout ───────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown(
        "<h1 style='margin-bottom:4px'>✈ Fleet Readiness Copilot</h1>"
        "<div class='caption' style='margin-bottom:24px'>"
        "Predictive Maintenance · Powered by XGBoost + IBM Granite Time Series + Granite 3.3"
        "</div>",
        unsafe_allow_html=True,
    )

    # Load data
    with st.spinner("Running readiness assessment…"):
        assets, queue, all_map = load_pipeline_data()

    # ── Fleet Readiness Board ─────────────────────────────────────────────────
    render_fleet_strip(assets, queue)

    # ── Ask the Copilot ───────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='margin-bottom:8px'>◆ Ask the Copilot</h3>"
        "<div class='caption'>Ask about any asset by tail number, or use natural language.</div>",
        unsafe_allow_html=True,
    )
    ask_col, btn_col = st.columns([6, 1])
    with ask_col:
        user_query = st.text_input(
            label="copilot_input",
            placeholder='e.g.  "Is Tail 43 ready?"  |  "Highest priority asset?"  |  "Most flyable tail?"',
            label_visibility="collapsed",
        )
    with btn_col:
        ask_clicked = st.button("Ask", use_container_width=True)

    if ask_clicked and user_query:
        render_copilot(user_query, all_map, queue)
    elif user_query:
        render_copilot(user_query, all_map, queue)

    # ── Maintenance Queue ─────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f"<h3 style='margin-bottom:4px'>Maintenance Queue &nbsp;"
        f"<span style='color:#6b7280;font-size:1rem;font-weight:400'>"
        f"{len(queue)} assets requiring action</span></h3>"
        "<div class='caption'>Ranked by composite priority score. "
        "Expand any asset for full evidence and sensor breakdown.</div>",
        unsafe_allow_html=True,
    )

    # Controls
    filter_col, type_col, sort_col, _ = st.columns([2, 2, 2, 2])
    with filter_col:
        show_crit_only = st.checkbox("Mission-critical only", value=False)
    with type_col:
        asset_type_options = ["All types", "Engine only", "Gearbox only"]
        asset_type_filter = st.selectbox("Asset type", asset_type_options, index=0)
    with sort_col:
        show_top_n = st.selectbox("Show top", [5, 10, 20, 50, len(queue)],
                                  format_func=lambda x: f"Top {x}" if x != len(queue) else "All",
                                  index=0)

    filtered_queue = queue
    if show_crit_only:
        filtered_queue = [q for q in filtered_queue if q.get("mission_critical_flag")]
    if asset_type_filter == "Engine only":
        filtered_queue = [q for q in filtered_queue if q.get("asset_type", "engine") == "engine"]
    elif asset_type_filter == "Gearbox only":
        filtered_queue = [q for q in filtered_queue if q.get("asset_type") == "gearbox"]

    displayed = filtered_queue[:show_top_n]

    if not displayed:
        st.info("No assets match the current filter.")
    else:
        for item in displayed:
            # Merge back any fields not carried through build_maintenance_queue
            aid  = item["asset_id"]
            full = {**all_map.get(aid, {}), **item}
            render_asset_card(full, rank=item["priority_rank"])

    # ── Mission-Ready Fleet ───────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    ready_assets = sorted(
        [a for a in assets if a["is_ready"]],
        key=lambda x: -x["predicted_rul"],
    )
    ready_engines   = [a for a in ready_assets if a.get("asset_type", "engine") == "engine"]
    ready_gearboxes = [a for a in ready_assets if a.get("asset_type") == "gearbox"]
    st.markdown(
        f"<h3 style='margin-bottom:4px'>Mission-Ready Fleet &nbsp;"
        f"<span style='color:#4ade80;font-size:1rem;font-weight:400'>"
        f"{len(ready_assets)} assets cleared for tasking</span></h3>"
        f"<div class='caption'>✈ {len(ready_engines)} engines &nbsp;·&nbsp; "
        f"⚙ {len(ready_gearboxes)} gearboxes</div>",
        unsafe_allow_html=True,
    )
    # Asset-type filter for the ready fleet
    ready_filter = st.radio(
        "Show", ["All", "Engines", "Gearboxes"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if ready_filter == "Engines":
        display_ready = ready_engines
    elif ready_filter == "Gearboxes":
        display_ready = ready_gearboxes
    else:
        display_ready = ready_assets

    if display_ready:
        for a in display_ready:
            render_ready_asset_row(a)
    else:
        st.warning("No mission-ready assets match the current filter.")

    # ── Fleet RUL Distribution ────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-bottom:8px'>Fleet RUL Distribution</h3>",
                unsafe_allow_html=True)

    ruls   = [a["predicted_rul"] for a in assets]
    colors = [_rul_color(r) for r in ruls]
    aids   = [
        ("G" if a["asset_type"] == "gearbox" else "T") + f"{a['asset_id']:04d}"
        for a in assets
    ]
    # Per-asset hover unit so engine=cycles, gearbox=steps
    units  = ["steps" if a["asset_type"] == "gearbox" else "cycles" for a in assets]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=aids, y=ruls,
        marker_color=colors,
        customdata=units,
        hovertemplate="%{x}<br>RUL: %{y:.1f} %{customdata}<extra></extra>",
    ))
    fig.add_hline(y=20, line_dash="dash", line_color="#6b7280",
                  annotation_text="Mission threshold (20 cycles / steps)",
                  annotation_font_color="#6b7280",
                  annotation_position="bottom right")
    fig.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(color="#9ca3af", size=10),
        margin=dict(l=10, r=10, t=10, b=40),
        height=220,
        xaxis=dict(showgrid=False, tickfont=dict(color="#4b5563"), tickangle=-45),
        yaxis=dict(showgrid=True, gridcolor="#1f2937",
                   tickfont=dict(color="#4b5563"), title=""),
        bargap=0.25,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown(
        "<div style='text-align:center;color:#374151;font-size:0.75rem;"
        "border-top:1px solid #1f2937;padding-top:16px;margin-top:24px'>"
        "Fleet Readiness Copilot · XGBoost + IBM Granite TinyTimeMixer + Granite 3.3 · "
        "All predictions are machine-generated and subject to maintenance officer review."
        "</div>",
        unsafe_allow_html=True,
    )


main()
