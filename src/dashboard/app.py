"""
ASTRA — Ultra-Premium Dashboard v3
Full dark-mode, glassmorphism, animated, multi-tab Streamlit app.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASTRA | AI Weather Blending",
    page_icon="🌩️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ────────────────────────────────────────────────────────────────────────────
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
  --bg-deep:    #050d1e;
  --bg-mid:     #0a1628;
  --bg-card:    rgba(14,28,56,0.85);
  --border:     rgba(88,166,255,0.14);
  --accent:     #58a6ff;
  --accent2:    #7ee787;
  --accent3:    #d2a8ff;
  --text-hi:    #e6edf3;
  --text-mid:   #8b9bb4;
  --text-lo:    #444d56;
}

/* ── Base ── */
*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp {
  background: radial-gradient(ellipse 120% 80% at 50% -10%, #0d2149 0%, #050d1e 60%);
  min-height: 100vh;
}
.main .block-container { padding: 1.2rem 2rem 3rem 2rem; max-width: 1440px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background:#21345a; border-radius:4px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #060f22 0%, #050d1e 100%) !important;
  border-right: 1px solid var(--border) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(14,28,56,0.7);
  border-radius: 14px;
  padding: 4px 6px;
  gap: 4px;
  border: 1px solid var(--border);
  backdrop-filter: blur(8px);
}
.stTabs [data-baseweb="tab"] {
  border-radius: 10px;
  font-weight: 600;
  font-size: 0.88rem;
  color: var(--text-mid) !important;
  padding: 8px 18px;
  transition: all .2s;
  border: none !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(88,166,255,0.2) 0%, rgba(88,166,255,0.08) 100%) !important;
  color: var(--accent) !important;
  border: 1px solid rgba(88,166,255,0.25) !important;
}

/* ── Cards ── */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1.4rem 1.6rem;
  backdrop-filter: blur(12px);
  transition: all .3s;
}
.card:hover {
  border-color: rgba(88,166,255,0.35);
  box-shadow: 0 12px 40px rgba(88,166,255,0.12);
  transform: translateY(-2px);
}
.card-title {
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--text-lo);
  margin-bottom: .4rem;
}
.card-value {
  font-size: 1.9rem;
  font-weight: 800;
  color: var(--text-hi);
  font-family: 'JetBrains Mono', monospace;
  line-height: 1.1;
}
.card-sub {
  font-size: .78rem;
  color: var(--text-mid);
  margin-top: .25rem;
}

/* ── Hero ── */
.hero {
  position: relative;
  overflow: hidden;
  background: linear-gradient(135deg,
    rgba(88,166,255,.07) 0%,
    rgba(126,231,135,.05) 45%,
    rgba(210,168,255,.07) 100%);
  border: 1px solid rgba(88,166,255,.18);
  border-radius: 24px;
  padding: 2.8rem 3.2rem;
  margin-bottom: 1.8rem;
}
.hero::before {
  content:'';
  position:absolute;
  top:-40%; left:-20%; width:140%; height:200%;
  background: radial-gradient(ellipse, rgba(88,166,255,.06) 0%, transparent 70%);
  animation: shimmer 8s ease-in-out infinite alternate;
  pointer-events: none;
}
@keyframes shimmer {
  from { transform: translate(0,0) scale(1); }
  to   { transform: translate(5%,5%) scale(1.08); }
}
.hero-badges { margin-bottom: .9rem; }
.badge {
  display: inline-block;
  background: rgba(88,166,255,.12);
  border: 1px solid rgba(88,166,255,.28);
  color: var(--accent);
  padding: .18rem .7rem;
  border-radius: 20px;
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .06em;
  margin-right: .4rem;
}
.badge.green  { background:rgba(126,231,135,.1); border-color:rgba(126,231,135,.3); color:#7ee787; }
.badge.purple { background:rgba(210,168,255,.1); border-color:rgba(210,168,255,.3); color:#d2a8ff; }
.badge.orange { background:rgba(251,188,5,.1);   border-color:rgba(251,188,5,.3);   color:#fbbf05; }
.hero-title {
  font-size: 3.2rem; font-weight: 900;
  background: linear-gradient(100deg, #58a6ff 0%, #7ee787 50%, #d2a8ff 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  margin: 0 0 .5rem; letter-spacing: -.04em; line-height: 1;
}
.hero-sub { font-size: 1.05rem; color: var(--text-mid); max-width: 720px; line-height: 1.6; }

/* ── Section titles ── */
.sec-title {
  font-size: 1.05rem; font-weight: 700;
  color: var(--text-hi);
  border-left: 3px solid var(--accent);
  padding-left: .7rem;
  margin: 1.6rem 0 .9rem;
}

/* ── Info pill ── */
.pill {
  display:inline-block;
  padding:.2rem .75rem;
  border-radius:20px;
  font-size:.78rem; font-weight:600;
}
.pill.blue   { background:rgba(88,166,255,.12);  color:#58a6ff; }
.pill.green  { background:rgba(126,231,135,.12); color:#7ee787; }
.pill.red    { background:rgba(248,81,73,.12);   color:#f85149; }
.pill.purple { background:rgba(210,168,255,.12); color:#d2a8ff; }

/* ── Stat row ── */
.stat-row {
  display:flex; gap:.5rem; flex-wrap:wrap; margin:.7rem 0;
}
.stat-item {
  background: rgba(22,40,80,.7);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: .45rem .9rem;
  font-size: .82rem;
}
.stat-item b { color: var(--text-hi); }
.stat-item span { color: var(--text-mid); margin-left:.3rem; }

/* ── Tables ── */
.stDataFrame { border-radius:14px !important; overflow:hidden; }
.stDataFrame thead { background:rgba(88,166,255,.08) !important; }

/* ── Divider ── */
hr { border-color: rgba(88,166,255,.08) !important; margin:1.2rem 0 !important; }

/* ── Metrics ── */
[data-testid="stMetricValue"]  { color: var(--text-hi)  !important; font-weight:700 !important; }
[data-testid="stMetricLabel"]  { color: var(--text-lo)  !important; text-transform:uppercase; font-size:.7rem !important; letter-spacing:.07em; }

/* ── Download btn ── */
.stDownloadButton button {
  background: linear-gradient(135deg,rgba(88,166,255,.18),rgba(88,166,255,.08)) !important;
  border: 1px solid rgba(88,166,255,.35) !important;
  color: var(--accent) !important;
  font-weight: 600 !important;
  border-radius: 10px !important;
  transition: all .2s;
}
.stDownloadButton button:hover {
  background: rgba(88,166,255,.25) !important;
  box-shadow: 0 0 20px rgba(88,166,255,.25) !important;
}
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# CONSTANTS / DATA
# ────────────────────────────────────────────────────────────────────────────
DATA_NC   = "data/blended_forecast.nc"
SKILL_CSV = "data/skill_scores.csv"
XGB_JSON  = "data/xgb_adaptive_weighter.json"

FEATURE_NAMES = ["latitude","longitude","tp_hres","tp_gfs","tp_spread",
                 "tp_model_diff","tp_model_mean","lat_sin","lon_cos"]

@st.cache_data(show_spinner=False)
def load_data():
    ds    = xr.open_dataset(DATA_NC) if os.path.exists(DATA_NC) else None
    skill = pd.read_csv(SKILL_CSV, index_col=0) if os.path.exists(SKILL_CSV) else None
    return ds, skill

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(XGB_JSON): return None
    import xgboost as xgb
    m = xgb.XGBRegressor(); m.load_model(XGB_JSON); return m

ds, skill_df = load_data()
xgb_model    = load_model()

# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.2rem 0 .8rem">
      <div style="font-size:2.8rem;line-height:1">🌩️</div>
      <div style="font-size:1.4rem;font-weight:900;background:linear-gradient(90deg,#58a6ff,#7ee787);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-top:.3rem;">
        ASTRA
      </div>
      <div style="font-size:.65rem;color:#444d56;letter-spacing:.14em;margin-top:.2rem;">
        AI FORECAST BLENDING SYSTEM
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Controls")

    cmap = st.selectbox("Precipitation Colormap",
        ["YlGnBu","Blues","Viridis","Plasma","Cividis","Turbo","RdYlBu_r"], index=0)
    show_contours = st.checkbox("Contour lines", value=True)
    opacity = st.slider("Map opacity", 0.5, 1.0, 0.92, 0.02)

    st.markdown("---")
    st.markdown("### 🛰️ Data Sources")
    sources = [
        ("🌍", "#58a6ff", "ECMWF HRES", "Deterministic 9 km"),
        ("🌐", "#7ee787", "NOAA GFS",   "Global 0.25°"),
        ("📡", "#d2a8ff", "ECMWF ENS",  "50-member spread"),
        ("✅", "#fbbf05", "ERA5",        "Ground truth labels"),
    ]
    for icon, clr, name, desc in sources:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:.7rem;padding:.5rem .6rem;
             background:rgba(14,28,56,.7);border:1px solid rgba(88,166,255,.1);
             border-radius:10px;margin-bottom:.4rem;">
          <div style="font-size:1.3rem">{icon}</div>
          <div>
            <div style="font-size:.82rem;font-weight:700;color:{clr};">{name}</div>
            <div style="font-size:.72rem;color:#444d56;">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Pipeline status
    nc_ok    = os.path.exists(DATA_NC)
    skill_ok = os.path.exists(SKILL_CSV)
    xgb_ok   = os.path.exists(XGB_JSON)
    st.markdown("### 🔄 Pipeline Status")
    for label, ok in [("Blended NC", nc_ok), ("Skill CSV", skill_ok), ("XGB Model", xgb_ok)]:
        icon = "✅" if ok else "⏳"
        clr  = "#7ee787" if ok else "#f85149"
        st.markdown(f'<div style="font-size:.82rem;color:{clr};margin:.2rem 0">{icon} {label}</div>',
                    unsafe_allow_html=True)
    if not nc_ok:
        st.code("python src/run_pipeline.py", language="bash")

    st.markdown("---")
    st.markdown('<div style="text-align:center;font-size:.65rem;color:#2a3547;">SIH 2026 · ASTRA v3 · XGBoost + Streamlit</div>',
                unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badges">
    <span class="badge">🏆 SIH 2026</span>
    <span class="badge green">🤖 XGBoost v2</span>
    <span class="badge purple">🌏 India 0.25°</span>
    <span class="badge orange">⚡ Live Dashboard</span>
  </div>
  <div class="hero-title">ASTRA Dashboard</div>
  <p class="hero-sub">
    Adaptive System for Temperature &amp; Rainfall Analytics —
    dynamically blends ECMWF HRES + NOAA GFS precipitation forecasts using
    XGBoost spatial weights calibrated on 2+ years of ERA5 reanalysis.
    Features ensemble-spread uncertainty maps and full NWP skill verification.
  </p>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# NO DATA
# ────────────────────────────────────────────────────────────────────────────
if ds is None:
    st.error("⚠️ No forecast data. Run: `python src/run_pipeline.py`")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ────────────────────────────────────────────────────────────────────────────
lats  = ds.latitude.values
lons  = ds.longitude.values
n_lat = len(lats); n_lon = len(lons)

blend = ds["tp_blended"].values if "tp_blended" in ds else np.zeros((n_lat,n_lon))
hres  = ds["tp_hres"].values    if "tp_hres"    in ds else np.zeros((n_lat,n_lon))
gfs   = ds["tp_gfs"].values     if "tp_gfs"     in ds else np.zeros((n_lat,n_lon))
spr   = ds["tp_spread"].values  if "tp_spread"  in ds else np.zeros((n_lat,n_lon))

kpis = [
    ("🗺️", "Grid Points",   f"{n_lat*n_lon:,}",          f"{n_lat}×{n_lon}"),
    ("🧭", "Coverage",      f"{lats.min():.0f}–{lats.max():.0f}°N", "India Region"),
    ("🌧️", "Mean Blend",    f"{np.nanmean(blend)*1000:.3f} mm", "Avg precipitation"),
    ("⬆️", "Peak Blend",    f"{np.nanmax(blend)*1000:.3f} mm",  "Grid maximum"),
    ("📡", "ENS Spread",    f"{np.nanmean(spr)*1000:.4f} mm",   "Mean uncertainty"),
]
cols = st.columns(5)
for col, (icon, lbl, val, sub) in zip(cols, kpis):
    with col:
        st.markdown(f"""
        <div class="card" style="text-align:center">
          <div style="font-size:1.6rem;margin-bottom:.3rem">{icon}</div>
          <div class="card-title">{lbl}</div>
          <div class="card-value" style="font-size:1.5rem">{val}</div>
          <div class="card-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────
DARK_BG = "rgba(5,13,30,0)"
GRID_CLR = "rgba(88,166,255,0.07)"

def _base_layout(title="", height=500):
    return dict(
        title=dict(text=title, font=dict(color="#e6edf3", size=13, family="Inter"),
                   x=0.01, xanchor="left", y=0.97),
        template="plotly_dark",
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        margin=dict(l=0,r=0,t=42,b=0), height=height,
        xaxis=dict(title="Longitude (°E)", color="#6e7681",
                   gridcolor=GRID_CLR, showgrid=True, zeroline=False),
        yaxis=dict(title="Latitude (°N)", color="#6e7681",
                   gridcolor=GRID_CLR, showgrid=True, zeroline=False,
                   scaleanchor="x", scaleratio=1),
    )

def precip_map(data_m, title, colorscale="YlGnBu", height=520,
               show_cont=True, zmax_mm=None, opacity=0.92):
    data_mm = data_m * 1000
    zmax = zmax_mm if zmax_mm else float(np.nanpercentile(data_mm[np.isfinite(data_mm)], 97))
    cbar = dict(title=dict(text="mm",side="right"),thickness=13,
                tickfont=dict(color="#6e7681",size=10),
                titlefont=dict(color="#6e7681",size=11),
                bgcolor="rgba(5,13,30,0.8)",
                bordercolor=GRID_CLR, borderwidth=1)
    fig = go.Figure()
    if show_cont:
        fig.add_trace(go.Contour(
            z=data_mm, x=lons, y=lats,
            colorscale=colorscale, zmin=0, zmax=zmax,
            contours=dict(showlines=True, showlabels=False, coloring="heatmap"),
            line=dict(width=0.6, color="rgba(255,255,255,0.12)"),
            opacity=opacity, colorbar=cbar, name=title,
        ))
    else:
        fig.add_trace(go.Heatmap(
            z=data_mm, x=lons, y=lats,
            colorscale=colorscale, zmin=0, zmax=zmax,
            opacity=opacity, colorbar=cbar, name=title,
        ))
    fig.update_layout(**_base_layout(title, height))
    return fig

def diff_map(data_m, title, height=460):
    data_mm = data_m * 1000
    lim = float(np.nanpercentile(np.abs(data_mm[np.isfinite(data_mm)]), 95))
    cbar = dict(title=dict(text="Δ mm",side="right"),thickness=13,
                tickfont=dict(color="#6e7681",size=10),
                titlefont=dict(color="#6e7681",size=11))
    fig = go.Figure(go.Heatmap(
        z=data_mm, x=lons, y=lats,
        colorscale="RdBu", zmid=0, zmin=-lim, zmax=lim,
        colorbar=cbar, name=title, opacity=opacity,
    ))
    fig.update_layout(**_base_layout(title, height))
    return fig

# ────────────────────────────────────────────────────────────────────────────
# TABS
# ────────────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6 = st.tabs([
    "🌧️ Blended Forecast",
    "⚖️ Model Comparison",
    "📈 Skill Metrics",
    "🔍 Feature Analysis",
    "📡 Uncertainty",
    "🗃️ Data Explorer",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1  ──  BLENDED FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    st.markdown('<div class="sec-title">🌧️ ASTRA AI-Blended Total Precipitation</div>', unsafe_allow_html=True)

    col_main, col_side = st.columns([3, 1.1])
    with col_main:
        fig = precip_map(blend, "ASTRA Blended Precipitation (mm)",
                         cmap, 540, show_contours, opacity=opacity)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":True,"modeBarButtonsToRemove":["lasso2d","select2d"]})

    with col_side:
        st.markdown('<div class="sec-title" style="margin-top:0">📊 Grid Stats</div>', unsafe_allow_html=True)
        vals_mm = blend.ravel() * 1000
        vals_mm = vals_mm[np.isfinite(vals_mm)]
        stats = pd.DataFrame({
            "Stat": ["Min","P10","P25","Median","P75","P90","Max","Mean","Std"],
            "mm":   [f"{np.percentile(vals_mm,q):.5f}" for q in [0,10,25,50,75,90,100]]
                    + [f"{vals_mm.mean():.5f}", f"{vals_mm.std():.5f}"],
        })
        st.dataframe(stats, hide_index=True, use_container_width=True, height=280)

        st.markdown('<div class="sec-title">📦 Distribution</div>', unsafe_allow_html=True)
        fig_h = go.Figure(go.Histogram(
            x=vals_mm, nbinsx=50,
            marker=dict(color="#58a6ff", opacity=.75,
                        line=dict(color="rgba(88,166,255,.3)", width=.4)),
        ))
        fig_h.update_layout(
            template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            margin=dict(l=0,r=0,t=6,b=30), height=200, showlegend=False,
            xaxis=dict(title="mm", color="#6e7681", gridcolor=GRID_CLR),
            yaxis=dict(title="", color="#6e7681", gridcolor=GRID_CLR),
        )
        st.plotly_chart(fig_h, use_container_width=True)

        # Download
        st.markdown('<div class="sec-title">⬇️ Export</div>', unsafe_allow_html=True)
        csv_df = ds["tp_blended"].to_dataframe(name="tp_blended_mm").reset_index()
        csv_df["tp_blended_mm"] *= 1000
        st.download_button("📥 Download CSV", csv_df.to_csv(index=False),
                           "astra_blended.csv", "text/csv", use_container_width=True)

    # 3D surface
    with st.expander("🏔️ 3-D Surface View"):
        fig3d = go.Figure(go.Surface(
            z=blend * 1000, x=lons, y=lats,
            colorscale=cmap, opacity=.92,
            contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="#58a6ff", project_z=True)),
            colorbar=dict(title="mm", thickness=14, tickfont=dict(color="#8b9bb4")),
        ))
        fig3d.update_layout(
            scene=dict(
                xaxis=dict(title="Longitude", backgroundcolor=DARK_BG, gridcolor="#1a2d4f"),
                yaxis=dict(title="Latitude",  backgroundcolor=DARK_BG, gridcolor="#1a2d4f"),
                zaxis=dict(title="Precip (mm)", backgroundcolor=DARK_BG, gridcolor="#1a2d4f"),
                bgcolor=DARK_BG,
            ),
            template="plotly_dark", paper_bgcolor=DARK_BG,
            margin=dict(l=0,r=0,t=20,b=0), height=520,
        )
        st.plotly_chart(fig3d, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2  ──  MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    st.markdown('<div class="sec-title">⚖️ Side-by-Side: HRES vs GFS vs ASTRA</div>', unsafe_allow_html=True)
    avail = {k:v for k,v in {"ECMWF HRES":"tp_hres","NOAA GFS":"tp_gfs","ASTRA Blend":"tp_blended"}.items() if v in ds}
    cmaps_model = {"ECMWF HRES":"Blues","NOAA GFS":"Greens","ASTRA Blend":cmap}

    cols = st.columns(len(avail))
    for col,(name,var) in zip(cols,avail.items()):
        with col:
            fig = precip_map(ds[var].values, name, cmaps_model.get(name,cmap), 400, show_contours, opacity=opacity)
            st.plotly_chart(fig, use_container_width=True)

    # Difference maps
    st.markdown('<div class="sec-title">↕️ Bias Maps (Blended − Source Model)</div>', unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)
    for col, var, label in [(dc1,"tp_hres","Blended − HRES"), (dc2,"tp_gfs","Blended − GFS")]:
        if var in ds and "tp_blended" in ds:
            with col:
                d = ds["tp_blended"].values - ds[var].values
                fig_d = diff_map(d, label, 420)
                st.plotly_chart(fig_d, use_container_width=True)
                pct_hi = float(np.nansum(d > 0) / np.sum(np.isfinite(d)) * 100)
                st.markdown(f"""
                <div style="background:rgba(14,28,56,.8);border:1px solid rgba(88,166,255,.12);
                     border-radius:12px;padding:.8rem 1rem;font-size:.85rem;color:#8b9bb4;">
                  <span class="pill green">{pct_hi:.1f}% Higher</span>
                  <span class="pill red">{100-pct_hi:.1f}% Lower</span>
                  <span style="margin-left:.5rem;">than {label.split(' − ')[1]}</span>
                </div>
                """, unsafe_allow_html=True)

    # HRES vs GFS scatter
    st.markdown('<div class="sec-title">🔵 HRES vs GFS Scatter (coloured by Blend)</div>', unsafe_allow_html=True)
    if "tp_hres" in ds and "tp_gfs" in ds and "tp_blended" in ds:
        h_f = hres.ravel()*1000; g_f = gfs.ravel()*1000; b_f = blend.ravel()*1000
        mask = np.isfinite(h_f) & np.isfinite(g_f) & np.isfinite(b_f)
        idx  = np.where(mask)[0]
        if len(idx) > 3000: idx = np.random.choice(idx, 3000, replace=False)
        mx = max(float(np.nanpercentile(h_f[mask],99)), float(np.nanpercentile(g_f[mask],99)))
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=h_f[idx], y=g_f[idx], mode="markers",
            marker=dict(color=b_f[idx], colorscale="YlGnBu", size=5, opacity=.75,
                        colorbar=dict(title="Blended mm",thickness=12,tickfont=dict(color="#8b9bb4"))),
        ))
        fig_sc.add_trace(go.Scatter(x=[0,mx],y=[0,mx],mode="lines",
            line=dict(color="rgba(255,255,255,.2)",dash="dash",width=1.5),showlegend=False))
        fig_sc.update_layout(
            template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            xaxis=dict(title="HRES (mm)", color="#6e7681", gridcolor=GRID_CLR),
            yaxis=dict(title="GFS (mm)",  color="#6e7681", gridcolor=GRID_CLR),
            height=380, margin=dict(l=0,r=0,t=20,b=0), showlegend=False,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3  ──  SKILL METRICS
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    st.markdown('<div class="sec-title">📈 NWP Skill Verification vs ERA5 Ground Truth</div>', unsafe_allow_html=True)

    if skill_df is not None and not skill_df.empty:
        col_tbl, col_radar = st.columns([3, 2])
        with col_tbl:
            styled = (skill_df.style
                .format("{:.5f}")
                .background_gradient(subset=["RMSE","MAE"], cmap="Reds_r")
                .background_gradient(subset=["Correlation","Skill Score"], cmap="Greens")
                .set_table_styles([
                    {"selector":"th","props":[("background","rgba(88,166,255,.08)"),
                                              ("color","#cdd9e5"),("font-size","0.82rem"),("padding","8px")]},
                    {"selector":"td","props":[("color","#e6edf3"),("font-size","0.85rem"),("padding","7px")]},
                    {"selector":"tr:hover td","props":[("background","rgba(88,166,255,.06)")]},
                ])
            )
            st.dataframe(styled, use_container_width=True)
            with st.expander("📖 Metric Definitions"):
                st.markdown("""
                | Metric | Best | Formula |
                |--------|------|---------|
                | **RMSE** | ↓ Lower | √mean((pred−truth)²) |
                | **MAE** | ↓ Lower | mean(|pred−truth|) |
                | **Bias** | → 0 | mean(pred−truth) |
                | **Correlation** | ↑ Higher | Pearson r |
                | **Skill Score** | ↑ Higher (>0 beats climatology) | 1 − RMSE/RMSE_clim |
                """)

        with col_radar:
            metrics = [c for c in ["RMSE","MAE","Correlation","Skill Score"] if c in skill_df.columns]
            if len(metrics) >= 3:
                rdf = skill_df[metrics].copy()
                for col in rdf.columns:
                    rng = rdf[col].max() - rdf[col].min() + 1e-12
                    if col in ("RMSE","MAE"):
                        rdf[col] = 1 - (rdf[col]-rdf[col].min()) / rng
                    else:
                        rdf[col] = (rdf[col]-rdf[col].min()) / rng
                colors = ["#58a6ff","#7ee787","#d2a8ff"]
                fig_r = go.Figure()
                for i,(mn,row) in enumerate(rdf.iterrows()):
                    cats = metrics + [metrics[0]]
                    vals = list(row.values) + [row.values[0]]
                    r,g,b = bytes.fromhex(colors[i%3].lstrip("#"))
                    fig_r.add_trace(go.Scatterpolar(
                        r=vals, theta=cats, name=mn, fill="toself",
                        fillcolor=f"rgba({r},{g},{b},0.12)",
                        line=dict(color=colors[i%3], width=2.5),
                    ))
                fig_r.update_layout(
                    polar=dict(
                        bgcolor="rgba(14,28,56,0.6)",
                        radialaxis=dict(visible=True,range=[0,1],color="#6e7681",
                                        gridcolor="rgba(88,166,255,0.1)"),
                        angularaxis=dict(color="#cdd9e5", tickfont=dict(size=11)),
                    ),
                    showlegend=True,
                    legend=dict(font=dict(color="#cdd9e5",size=11), bgcolor="rgba(0,0,0,0)"),
                    template="plotly_dark", paper_bgcolor=DARK_BG,
                    height=390, margin=dict(l=20,r=20,t=20,b=20),
                    title=dict(text="Normalised Skill Radar", font=dict(color="#e6edf3",size=13)),
                )
                st.plotly_chart(fig_r, use_container_width=True)

        # Bar chart grid
        st.markdown('<div class="sec-title">📊 Metric Breakdown</div>', unsafe_allow_html=True)
        plot_metrics = [c for c in ["RMSE","MAE","Bias","Skill Score"] if c in skill_df.columns]
        bar_cols = st.columns(len(plot_metrics))
        mcolors  = ["#58a6ff","#7ee787","#d2a8ff"]
        for c, metric in zip(bar_cols, plot_metrics):
            with c:
                fig_b = go.Figure(go.Bar(
                    x=list(skill_df.index), y=skill_df[metric].values,
                    marker=dict(color=mcolors[:len(skill_df)], opacity=.88,
                                line=dict(width=1, color="rgba(255,255,255,.06)")),
                    text=[f"{v:.4f}" for v in skill_df[metric].values],
                    textposition="outside", textfont=dict(color="#8b9bb4",size=10),
                ))
                fig_b.update_layout(
                    title=dict(text=metric,font=dict(color="#e6edf3",size=12),x=0.04),
                    template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                    xaxis=dict(color="#6e7681", tickfont=dict(size=10), gridcolor=GRID_CLR),
                    yaxis=dict(color="#6e7681", gridcolor=GRID_CLR),
                    margin=dict(l=0,r=0,t=38,b=0), height=240, showlegend=False,
                )
                st.plotly_chart(fig_b, use_container_width=True)
    else:
        st.info("Skill scores will appear here after running the full pipeline:\n```bash\npython src/run_pipeline.py\n```")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4  ──  FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with t4:
    st.markdown('<div class="sec-title">🔍 XGBoost Feature Importance</div>', unsafe_allow_html=True)
    if xgb_model is not None:
        fi  = pd.Series(xgb_model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
        max_fi = fi.max()

        col_bar, col_info = st.columns([2, 1])
        with col_bar:
            fig_fi = go.Figure(go.Bar(
                x=fi.values[::-1], y=fi.index[::-1], orientation="h",
                marker=dict(
                    color=fi.values[::-1],
                    colorscale=[[0,"rgba(88,166,255,.3)"], [1,"rgba(88,166,255,1)"]],
                    opacity=.9, line=dict(width=0),
                ),
                text=[f"  {v:.4f}" for v in fi.values[::-1]],
                textposition="outside", textfont=dict(color="#8b9bb4", size=10),
            ))
            fig_fi.update_layout(
                title=dict(text="Feature Importance (Gain-based)", font=dict(color="#e6edf3",size=13), x=0.01),
                template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                xaxis=dict(title="Importance",color="#6e7681",gridcolor=GRID_CLR),
                yaxis=dict(color="#cdd9e5",tickfont=dict(family="JetBrains Mono",size=11)),
                margin=dict(l=0,r=70,t=42,b=0), height=420, showlegend=False,
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        with col_info:
            st.markdown("#### Feature Details")
            descs = {
                "latitude":      ("Geographic latitude",       "Latitudinal bias patterns"),
                "longitude":     ("Geographic longitude",      "Orographic/coastal effects"),
                "tp_hres":       ("ECMWF HRES precipitation",  "Direct predictor"),
                "tp_gfs":        ("NOAA GFS precipitation",    "Direct predictor"),
                "tp_spread":     ("ENS inter-member std-dev",  "Uncertainty proxy"),
                "tp_model_diff": ("HRES − GFS disagreement",   "Model spread signal"),
                "tp_model_mean": ("(HRES+GFS)/2",              "Consensus signal"),
                "lat_sin":       ("sin(latitude)",             "Cyclic lat encoding"),
                "lon_cos":       ("cos(longitude)",            "Cyclic lon encoding"),
            }
            for feat, imp in fi.items():
                pct = int(imp / max_fi * 100)
                short, long_ = descs.get(feat, (feat,""))
                st.markdown(f"""
                <div style="margin-bottom:.9rem">
                  <div style="display:flex;justify-content:space-between;margin-bottom:.2rem">
                    <span style="font-family:'JetBrains Mono',mono;font-size:.78rem;color:#cdd9e5;">{feat}</span>
                    <span style="font-size:.78rem;color:#58a6ff;font-weight:700">{imp:.4f}</span>
                  </div>
                  <div style="background:#0a1628;border-radius:4px;height:5px;margin-bottom:.2rem">
                    <div style="background:linear-gradient(90deg,#1a4080,#58a6ff);
                         width:{pct}%;height:5px;border-radius:4px;
                         box-shadow:0 0 8px rgba(88,166,255,.4)"></div>
                  </div>
                  <div style="font-size:.7rem;color:#444d56">{short} — {long_}</div>
                </div>
                """, unsafe_allow_html=True)

        # Pie
        st.markdown('<div class="sec-title">🥧 Importance Breakdown</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=fi.index, values=fi.values, hole=0.55,
            marker=dict(colors=px.colors.sequential.Blues_r[:len(fi)],
                        line=dict(color="#050d1e", width=2)),
            textfont=dict(color="#cdd9e5",size=11),
            hovertemplate="<b>%{label}</b><br>Importance: %{value:.4f}<br>Share: %{percent}<extra></extra>",
        ))
        fig_pie.add_annotation(text="Features", x=0.5, y=0.5, showarrow=False,
                               font=dict(color="#8b9bb4",size=13,family="Inter"))
        fig_pie.update_layout(
            title=dict(text="Relative Feature Contributions",
                       font=dict(color="#e6edf3",size=13), x=0.5, xanchor="center"),
            template="plotly_dark", paper_bgcolor=DARK_BG,
            legend=dict(font=dict(color="#cdd9e5",size=11), bgcolor="rgba(0,0,0,0)"),
            height=380, margin=dict(l=0,r=0,t=42,b=0),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("Model not found. Run `python src/run_pipeline.py` first.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5  ──  UNCERTAINTY (ENS SPREAD)
# ══════════════════════════════════════════════════════════════════════════════
with t5:
    st.markdown('<div class="sec-title">📡 Ensemble Uncertainty Analysis</div>', unsafe_allow_html=True)

    if "tp_spread" in ds:
        col_map, col_stat = st.columns([3, 1.1])
        with col_map:
            fig_sp = precip_map(spr, "ENS Inter-Member Spread (mm)", "Oranges", 500,
                                show_contours, opacity=opacity)
            st.plotly_chart(fig_sp, use_container_width=True)
        with col_stat:
            st.markdown('<div class="sec-title" style="margin-top:0">Spread Stats</div>', unsafe_allow_html=True)
            sp_mm = spr.ravel()*1000; sp_mm = sp_mm[np.isfinite(sp_mm)]
            stats_sp = pd.DataFrame({
                "Stat": ["Min","Mean","Median","P90","Max","Std"],
                "mm":   [f"{np.min(sp_mm):.5f}",f"{np.mean(sp_mm):.5f}",
                         f"{np.median(sp_mm):.5f}",f"{np.percentile(sp_mm,90):.5f}",
                         f"{np.max(sp_mm):.5f}",f"{np.std(sp_mm):.5f}"],
            })
            st.dataframe(stats_sp, hide_index=True, use_container_width=True, height=220)
            st.markdown("""
            <div class="info-panel" style="background:rgba(210,168,255,.06);border-color:rgba(210,168,255,.2);border-radius:12px;padding:.9rem;">
              <p style="color:#8b9bb4;font-size:.82rem;margin:0">
                <b style="color:#d2a8ff;">Spread interpretation:</b><br/>
                High spread = high uncertainty = model members disagree strongly.
                ASTRA uses spread as a feature to down-weight uncertain predictions.
              </p>
            </div>
            """, unsafe_allow_html=True)

        # Spread vs Blend scatter
        st.markdown('<div class="sec-title">📊 Spread vs Blended Precipitation</div>', unsafe_allow_html=True)
        sp_flat = spr.ravel()*1000; bl_flat = blend.ravel()*1000
        mask2 = np.isfinite(sp_flat) & np.isfinite(bl_flat)
        idx2  = np.where(mask2)[0]
        if len(idx2) > 3000: idx2 = np.random.choice(idx2, 3000, replace=False)
        fig_sv = go.Figure(go.Scatter(
            x=sp_flat[idx2], y=bl_flat[idx2], mode="markers",
            marker=dict(color=bl_flat[idx2], colorscale="Plasma", size=5, opacity=.65,
                        colorbar=dict(title="Blend mm",thickness=12,tickfont=dict(color="#8b9bb4"))),
        ))
        fig_sv.update_layout(
            template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            xaxis=dict(title="ENS Spread (mm)", color="#6e7681", gridcolor=GRID_CLR),
            yaxis=dict(title="Blended (mm)",    color="#6e7681", gridcolor=GRID_CLR),
            height=360, margin=dict(l=0,r=0,t=20,b=0), showlegend=False,
        )
        st.plotly_chart(fig_sv, use_container_width=True)

        # Uncertainty heatmap subplot
        st.markdown('<div class="sec-title">🗺️ Confidence Map</div>', unsafe_allow_html=True)
        # Confidence = 1 - normalised spread
        spread_norm = (spr - np.nanmin(spr)) / (np.nanmax(spr) - np.nanmin(spr) + 1e-12)
        confidence  = 1 - spread_norm
        fig_conf = go.Figure(go.Heatmap(
            z=confidence, x=lons, y=lats,
            colorscale="RdYlGn", zmin=0, zmax=1,
            colorbar=dict(title=dict(text="Confidence",side="right"),
                          thickness=13, tickfont=dict(color="#6e7681",size=10),
                          titlefont=dict(color="#6e7681",size=11)),
            opacity=opacity,
        ))
        fig_conf.update_layout(**_base_layout("Forecast Confidence (1 = most confident)", 420))
        st.plotly_chart(fig_conf, use_container_width=True)
    else:
        st.info("ENS spread not available. Check that `ens.grib2` is present and pipeline ran successfully.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6  ──  DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with t6:
    st.markdown('<div class="sec-title">🗃️ Interactive Data Explorer</div>', unsafe_allow_html=True)

    sel_var = st.selectbox("Variable", list(ds.data_vars), index=list(ds.data_vars).index("tp_blended") if "tp_blended" in ds.data_vars else 0)
    var_data = ds[sel_var].values * 1000

    col_z, col_m = st.columns(2)
    with col_z:
        lat_sl = st.slider("Zonal profile: Latitude (°N)", float(lats.min()), float(lats.max()),
                           float(np.median(lats)), 0.25)
        li = int(np.argmin(np.abs(lats - lat_sl)))
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(x=lons, y=var_data[li,:], mode="lines",
            line=dict(color="#58a6ff",width=2.5), fill="tozeroy",
            fillcolor="rgba(88,166,255,.07)"))
        fig_z.update_layout(
            title=dict(text=f"Zonal slice at {lat_sl:.2f}°N — {sel_var}", font=dict(color="#e6edf3",size=12),x=0.01),
            template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            xaxis=dict(title="Longitude (°E)",color="#6e7681",gridcolor=GRID_CLR),
            yaxis=dict(title="mm",color="#6e7681",gridcolor=GRID_CLR),
            height=300, margin=dict(l=0,r=0,t=40,b=0), showlegend=False,
        )
        st.plotly_chart(fig_z, use_container_width=True)

    with col_m:
        lon_sl = st.slider("Meridional profile: Longitude (°E)", float(lons.min()), float(lons.max()),
                           float(np.median(lons)), 0.25)
        lni = int(np.argmin(np.abs(lons - lon_sl)))
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=lats, y=var_data[:,lni], mode="lines",
            line=dict(color="#7ee787",width=2.5), fill="tozeroy",
            fillcolor="rgba(126,231,135,.07)"))
        fig_m.update_layout(
            title=dict(text=f"Meridional slice at {lon_sl:.2f}°E — {sel_var}", font=dict(color="#e6edf3",size=12),x=0.01),
            template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            xaxis=dict(title="Latitude (°N)",color="#6e7681",gridcolor=GRID_CLR),
            yaxis=dict(title="mm",color="#6e7681",gridcolor=GRID_CLR),
            height=300, margin=dict(l=0,r=0,t=40,b=0), showlegend=False,
        )
        st.plotly_chart(fig_m, use_container_width=True)

    # Schema table
    st.markdown('<div class="sec-title">📋 Dataset Schema</div>', unsafe_allow_html=True)
    rows = []
    for v in ds.data_vars:
        arr = ds[v].values
        rows.append({"Variable":v, "Shape":str(arr.shape),
                     "Units":ds[v].attrs.get("units","—"),
                     "Min (mm)":f"{float(np.nanmin(arr))*1000:.5f}",
                     "Max (mm)":f"{float(np.nanmax(arr))*1000:.5f}",
                     "Mean (mm)":f"{float(np.nanmean(arr))*1000:.5f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # All-variable comparison bar
    st.markdown('<div class="sec-title">📊 All-Variable Mean Comparison</div>', unsafe_allow_html=True)
    fig_comp = go.Figure()
    model_vars = [(v, ds[v].values*1000) for v in ds.data_vars]
    fig_comp.add_trace(go.Bar(
        x=[v for v,_ in model_vars],
        y=[float(np.nanmean(a)) for _,a in model_vars],
        marker=dict(color=["#58a6ff","#7ee787","#d2a8ff","#fbbf05"][:len(model_vars)], opacity=.88),
        text=[f"{float(np.nanmean(a)):.4f}" for _,a in model_vars],
        textposition="outside", textfont=dict(color="#8b9bb4",size=11),
    ))
    fig_comp.update_layout(
        template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        xaxis=dict(color="#cdd9e5",gridcolor=GRID_CLR),
        yaxis=dict(title="Mean (mm)",color="#6e7681",gridcolor=GRID_CLR),
        margin=dict(l=0,r=0,t=20,b=0), height=280, showlegend=False,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# FOOTER
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr/>
<div style="display:flex;justify-content:space-between;align-items:center;
     padding:.6rem 0;flex-wrap:wrap;gap:.5rem;">
  <div style="font-size:.72rem;color:#2a3547">
    🌩️ <b style="color:#444d56">ASTRA v3</b> — Adaptive System for Temperature &amp; Rainfall Analytics
  </div>
  <div style="font-size:.72rem;color:#2a3547">
    ECMWF Open Data · NOAA GFS · ERA5 · XGBoost · Streamlit
  </div>
  <div style="font-size:.72rem;color:#2a3547">
    Smart India Hackathon 2026 · India 0.25° · 113×113 Grid
  </div>
</div>
""", unsafe_allow_html=True)
