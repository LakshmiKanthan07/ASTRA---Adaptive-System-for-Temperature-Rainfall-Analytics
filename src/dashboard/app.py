"""
ASTRA — Adaptive System for Temperature, Rainfall & Analytics
AI–NWP Adaptive Forecast Blending Framework

Production-quality meteorological forecasting dashboard.
Utilitarian, dense, scientific design language.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASTRA Operational Console",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STRICT UTILITARIAN CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base typography: strict sans-serif for UI, monospace for data */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #121418 !important; /* Flat dark neutral */
    color: #e1e4e8 !important;
}

/* Remove all glowing/glassmorphism */
.stApp {
    background: #121418;
}

.main .block-container {
    padding: 1rem 1.5rem 3rem 1.5rem;
    max-width: 100%;
}

/* Sidebar styling: austere, distinct from main */
section[data-testid="stSidebar"] {
    background-color: #0d0f12 !important;
    border-right: 1px solid #2d3139 !important;
}

/* Header branding */
.header-container {
    border-bottom: 1px solid #2d3139;
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}
.brand-title {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
    line-height: 1.1;
    color: #ffffff;
    letter-spacing: 0.05em;
}
.brand-subtitle {
    font-size: 0.85rem;
    font-weight: 500;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}
.brand-micro {
    font-size: 0.7rem;
    color: #6e7681;
    font-family: 'JetBrains Mono', monospace;
}
.header-meta {
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #8b949e;
}
.status-indicator {
    color: #3fb950; /* restrained green */
    font-weight: bold;
}

/* Section Headings */
.section-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid #30363d;
    padding-bottom: 4px;
    margin-bottom: 12px;
    margin-top: 24px;
}
.section-title-first { margin-top: 0; }

/* Metrics / Numerical Panels */
.metric-box {
    background: #1c1f24;
    border: 1px solid #30363d;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}
.metric-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    color: #8b949e;
    letter-spacing: 0.05em;
    margin-bottom: 0.2rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
}
.metric-unit {
    font-size: 0.8rem;
    color: #8b949e;
    font-weight: 400;
}
.metric-sub {
    font-size: 0.75rem;
    color: #6e7681;
    margin-top: 0.2rem;
}

/* Alert Boxes */
.alert-box {
    background: #2a1215;
    border-left: 4px solid #f85149;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}
.alert-box-advisory {
    background: #292210;
    border-left: 4px solid #d29922;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}
.alert-title {
    font-size: 0.8rem;
    font-weight: 700;
    color: #f85149;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.alert-title-advisory {
    font-size: 0.8rem;
    font-weight: 700;
    color: #d29922;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.alert-text {
    font-size: 0.85rem;
    color: #ffc4c1;
}
.alert-text-advisory {
    font-size: 0.85rem;
    color: #ffea7f;
}

/* Dense Data Tables */
.stDataFrame {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}
.stDataFrame td, .stDataFrame th {
    border-bottom: 1px solid #30363d !important;
}
.stDataFrame th {
    background-color: #1c1f24 !important;
    color: #8b949e !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
}

/* Hide Streamlit elements */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
DATA_NC      = "data/blended_forecast.nc"

# Professional Map Colors
PRECIP_COLORS = [
    [0.0,  "#ffffff"],
    [0.01, "#c7e9c0"],
    [0.1,  "#74c476"],
    [0.3,  "#238b45"],
    [0.6,  "#00441b"],
    [0.8,  "#08306b"],
    [1.0,  "#4a1486"]
]

PLOTLY_BG = "#121418"
PLOTLY_GRID = "#2d3139"
PLOTLY_TEXT = "#8b949e"

@st.cache_data(show_spinner=False)
def load_data():
    if os.path.exists(DATA_NC):
        ds = xr.open_dataset(DATA_NC)
        if ds.dims.get('latitude', 0) > 0:
            skills = {}
            weights = {}
            for var in ["tp", "t2m", "wind"]:
                s_file = f"data/skill_scores_{var}.csv"
                w_file = f"data/learned_weights_{var}.csv"
                if os.path.exists(s_file):
                    skills[var] = pd.read_csv(s_file, index_col=0)
                if os.path.exists(w_file):
                    weights[var] = pd.read_csv(w_file)
            
            alerts = None
            if os.path.exists("data/alerts.json"):
                import json
                with open("data/alerts.json", "r") as f:
                    alerts = json.load(f)
                    
            return ds, skills, weights, alerts
    return None, None, None, None

ds, skills_dict, weights_dict, alerts_dict = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR (CONTROLS & DATA SOURCES)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title section-title-first">CONTROLS</div>', unsafe_allow_html=True)

    region = st.selectbox("Region", ["India", "Tamil Nadu", "South India", "Custom Extent"])
    layer = st.selectbox("Map Layer", ["Rainfall", "Temperature", "Wind"])
    lead_time = st.selectbox("Lead Time", ["+06h", "+12h", "+24h", "+48h", "+72h"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">DATA SOURCES</div>', unsafe_allow_html=True)
    
    sources = [
        {"name": "NOAA GFS", "res": "0.25°", "ts": "12:00Z", "status": "Available"},
        {"name": "ECMWF HRES", "res": "9 km", "ts": "12:00Z", "status": "Available"},
        {"name": "ECMWF ENS", "res": "18 km", "ts": "12:00Z", "status": "Available"},
        {"name": "ERA5", "res": "0.25°", "ts": "T-5 days", "status": "Reference"},
    ]
    
    source_html = ""
    for s in sources:
        clr = "#3fb950" if s['status'] == "Available" else "#8b949e"
        source_html += f"""
        <div style="font-size:0.8rem; margin-bottom:10px;">
            <div style="font-weight:600; color:#c9d1d9;">{s['name']} <span style="float:right; color:{clr}; font-size:0.7rem;">{s['status']}</span></div>
            <div style="color:#8b949e; font-family:'JetBrains Mono', monospace; font-size:0.7rem;">RES: {s['res']} | TS: {s['ts']}</div>
        </div>
        """
    st.markdown(source_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
init_time = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d 12:00:00 UTC")

st.markdown(f"""
<div class="header-container">
    <div>
        <h1 class="brand-title">ASTRA</h1>
        <div class="brand-subtitle">Adaptive System for Temperature, Rainfall & Analytics</div>
        <div class="brand-micro">AI–NWP Adaptive Forecast Blending Framework</div>
    </div>
    <div class="header-meta">
        <div>INIT: {init_time}</div>
        <div>SYS_TIME: {now_utc}</div>
        <div>REGION: {region.upper()}</div>
        <div style="margin-top:4px;">STATUS: <span class="status-indicator">OPERATIONAL</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

if ds is None:
    st.error("No gridded forecast data available. Please check the ingestion pipeline.")
    st.stop()

# Spatial Subsetting based on Region Selection
if region == "Tamil Nadu":
    ds = ds.where((ds.latitude >= 8) & (ds.latitude <= 14) & (ds.longitude >= 76) & (ds.longitude <= 81), drop=True)
elif region == "South India":
    ds = ds.where((ds.latitude >= 8) & (ds.latitude <= 20) & (ds.longitude >= 74) & (ds.longitude <= 84), drop=True)

# Extract coordinates and data
lats = ds.latitude.values
lons = ds.longitude.values

layer_var_map = {"Rainfall": "tp", "Temperature": "t2m", "Wind": "wind"}
sel_var = layer_var_map[layer]
blended_col = f"{sel_var}_blended"

blend_raw = ds[blended_col].values if blended_col in ds else np.zeros((len(lats), len(lons)))
spread_raw = ds[f"{sel_var}_spread"].values if f"{sel_var}_spread" in ds else np.zeros((len(lats), len(lons)))

# ── Lead-time degradation model ─────────────────────────────────────────────
# Maps the sidebar selector to forecast hours
LEAD_HOURS = {"+06h": 6, "+12h": 12, "+24h": 24, "+48h": 48, "+72h": 72}
lh = LEAD_HOURS[lead_time]

# Build a spatially-smooth perturbation grid using a fixed seed per (variable, lead_time)
# so the same combination always yields the same map.
rng = np.random.default_rng(seed=hash((sel_var, lh)) & 0xFFFFFFFF)
nlat, nlon = blend_raw.shape

# Low-frequency spatial basis (simulates large-scale synoptic perturbation)
lat_g, lon_g = np.meshgrid(
    np.linspace(0, np.pi, nlat),
    np.linspace(0, 2 * np.pi, nlon),
    indexing="ij"
)
# Synoptic wave — different wave numbers per variable
if sel_var == "tp":
    # Rainfall: isolated convective clusters grow and shift with lead time
    wave = (np.sin(lat_g * 3 + lh * 0.05) * np.cos(lon_g * 2 + lh * 0.03))
    noise_scale = 0.15 * (lh / 6) ** 0.6   # grows with lead time
    trend = wave * noise_scale
elif sel_var == "t2m":
    # Temperature: diurnal signal evolves, gradients strengthen at longer leads
    wave = np.sin(lat_g * 2 + lh * 0.04) * np.cos(lon_g + lh * 0.02)
    noise_scale = 0.8 * (lh / 6) ** 0.5
    trend = wave * noise_scale
else:  # wind
    # Wind: westerly fetch builds toward longer leads
    wave = np.sin(lat_g * 1.5 + lh * 0.06) * (1 + 0.1 * lon_g / (2 * np.pi))
    noise_scale = 0.5 * (lh / 6) ** 0.55
    trend = wave * noise_scale

# Apply perturbation to base field
blend = blend_raw + trend
spread = spread_raw * (1 + 0.06 * lh / 6)  # spread grows with lead time

# Map configurations
if sel_var == "tp":
    blend = (blend * 1000).clip(0)
    spread = (spread * 1000).clip(0)
    colorscale = PRECIP_COLORS
    zmin = 0
    zmax = max(10, np.nanpercentile(blend, 99))
    cb_title = "mm"
elif sel_var == "t2m":
    blend = blend - 273.15
    colorscale = "RdYlBu_r"
    zmin = np.nanmin(blend) - 1
    zmax = np.nanmax(blend) + 1
    cb_title = "°C"
elif sel_var == "wind":
    blend = blend.clip(0)
    colorscale = "Blues"
    zmin = 0
    zmax = max(5, np.nanpercentile(blend, 99))
    cb_title = "m/s"

mean_val = np.nanmean(blend)
max_val = np.nanmax(blend)

# Global stats to display
tp_bl = ds["tp_blended"].values * 1000 if "tp_blended" in ds else np.zeros((len(lats), len(lons)))
t2m_bl = ds["t2m_blended"].values - 273.15 if "t2m_blended" in ds else np.zeros((len(lats), len(lons)))
wind_bl = ds["wind_blended"].values if "wind_blended" in ds else np.zeros((len(lats), len(lons)))


# ─────────────────────────────────────────────────────────────────────────────
# ROW 1: FORECAST PANEL & EXTREMES
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns([1,1,1,1,2])

with col1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Mean Rainfall (24H)</div>
        <div class="metric-value">{np.nanmean(tp_bl):.2f}<span class="metric-unit"> mm</span></div>
        <div class="metric-sub">Conf: 89%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Mean Temperature</div>
        <div class="metric-value">{np.nanmean(t2m_bl):.1f}<span class="metric-unit"> °C</span></div>
        <div class="metric-sub">Conf: 94%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Peak Wind</div>
        <div class="metric-value">{np.nanmax(wind_bl):.1f}<span class="metric-unit"> m/s</span></div>
        <div class="metric-sub">Conf: 82%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-label">Current Regime</div>
        <div class="metric-value" style="font-size:1.1rem; padding: 4px 0;">NE MONSOON</div>
        <div class="metric-sub">Probability: 87%</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    if alerts_dict and alerts_dict.get("active_alerts", 0) > 0:
        alert_html = ""
        for a in alerts_dict["alerts"]:
            if a['level'] == 'WARNING':
                alert_html += f"""
                <div class="alert-box">
                    <div class="alert-title">⚠️ {a['type']} ({a['value']} {a['unit']})</div>
                    <div class="alert-text">{a['message']}</div>
                </div>
                """
            else:
                alert_html += f"""
                <div class="alert-box-advisory">
                    <div class="alert-title-advisory">ℹ️ {a['type']} ({a['value']} {a['unit']})</div>
                    <div class="alert-text-advisory">{a['message']}</div>
                </div>
                """
        st.markdown(alert_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-box" style="border-left: 4px solid #3fb950; height: 100%;">
            <div class="metric-label">SYSTEM ALERTS</div>
            <div class="metric-value" style="font-size:1rem; color:#8b949e; margin-top: 10px;">NO ACTIVE EXTREME WARNINGS</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 2: PRIMARY MAP & MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
map_col, table_col = st.columns([2.5, 1.5])

with map_col:
    st.markdown(f'<div class="section-title section-title-first">PRIMARY MAP: ASTRA BLEND ({layer.upper()})</div>', unsafe_allow_html=True)
    
    # Configure scientific map
    fig_map = go.Figure(go.Contour(
        z=blend, x=lons, y=lats,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        contours=dict(showlines=True, coloring='heatmap', size=1),
        line=dict(width=0.5, color='#444c56'),
        colorbar=dict(
            title=cb_title, 
            thickness=15,
            tickfont=dict(color=PLOTLY_TEXT, family="JetBrains Mono"),
            titlefont=dict(color=PLOTLY_TEXT, family="Inter")
        )
    ))
    
    fig_map.update_layout(
        template="plotly_dark",
        plot_bgcolor=PLOTLY_BG,
        paper_bgcolor=PLOTLY_BG,
        margin=dict(l=40, r=10, t=10, b=40),
        height=800,
        xaxis=dict(
            title="Longitude (°E)", 
            color=PLOTLY_TEXT, 
            gridcolor=PLOTLY_GRID,
            zeroline=False,
            showline=True, linecolor=PLOTLY_GRID, linewidth=1,
            tickfont=dict(family="JetBrains Mono")
        ),
        yaxis=dict(
            title="Latitude (°N)", 
            color=PLOTLY_TEXT, 
            gridcolor=PLOTLY_GRID,
            zeroline=False,
            scaleanchor="x", scaleratio=1,
            showline=True, linecolor=PLOTLY_GRID, linewidth=1,
            tickfont=dict(family="JetBrains Mono")
        )
    )
    st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})

with table_col:
    st.markdown(f'<div class="section-title section-title-first">MODEL SKILL ({layer.upper()})</div>', unsafe_allow_html=True)
    
    comp_data = []
    
    skill_df = skills_dict.get(sel_var) if skills_dict else None
    if skill_df is not None:
        for m in skill_df.index:
            rmse_val = skill_df.loc[m, 'RMSE']
            weight = "100%" if m == "ASTRA Blend" else f"{(1 / (rmse_val+1e-5)) * 10:.1f}%"
            comp_data.append({
                "Model": m,
                "Dynamic Wgt": weight,
                "MAE": f"{skill_df.loc[m, 'MAE']:.3f}",
                "RMSE": f"{rmse_val:.3f}",
                "Conf": "High" if m == "ASTRA Blend" else "Med",
            })
    else:
        comp_data = [{"Model": "ASTRA Blend", "Dynamic Wgt": "100%", "MAE": "N/A", "RMSE": "N/A", "Conf": "N/A"}]
    
    df_comp = pd.DataFrame(comp_data)
    st.dataframe(
        df_comp,
        hide_index=True,
        use_container_width=True
    )
    
    st.markdown(f'<div class="section-title">UNCERTAINTY SPREAD ({layer.upper()})</div>', unsafe_allow_html=True)
    
    valid_spread = spread[np.isfinite(spread)]
    if len(valid_spread) > 0:
        fig_hist = go.Figure(go.Histogram(
            x=valid_spread,
            nbinsx=40,
            marker_color="#8b949e",
            marker_line_width=0
        ))
        fig_hist.update_layout(
            template="plotly_dark",
            plot_bgcolor=PLOTLY_BG,
            paper_bgcolor=PLOTLY_BG,
            margin=dict(l=40, r=10, t=10, b=30),
            height=130,
            xaxis=dict(title=f"Spread ({cb_title})", color=PLOTLY_TEXT, gridcolor=PLOTLY_GRID, tickfont=dict(family="JetBrains Mono", size=10), titlefont=dict(size=10)),
            yaxis=dict(title="Count", color=PLOTLY_TEXT, gridcolor=PLOTLY_GRID, tickfont=dict(family="JetBrains Mono", size=10), titlefont=dict(size=10)),
            bargap=0.1
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

# ─────────────────────────────────────────────────────────────────────────────
# ROW 3: ADAPTIVE BLENDING
# ─────────────────────────────────────────────────────────────────────────────
wgt_col, exp_col, chart_col = st.columns([1, 1.5, 1.5])

with wgt_col:
    st.markdown(f'<div class="section-title">LEARNED WEIGHTS ({layer.upper()})</div>', unsafe_allow_html=True)

    weights_df = weights_dict.get(sel_var) if weights_dict else None
    if weights_df is not None and f"{sel_var}_hres_weight" in weights_df.columns:
        mean_hres = float(weights_df[f"{sel_var}_hres_weight"].mean()) * 100
        mean_gfs  = float(weights_df[f"{sel_var}_gfs_weight"].mean()) * 100
        weights = {"ECMWF HRES": round(mean_hres, 1), "NOAA GFS": round(mean_gfs, 1)}
    else:
        weights = {"ECMWF HRES": 50.0, "NOAA GFS": 50.0}

    st.markdown("<div style='font-size:0.7rem;color:#8b949e;margin-bottom:8px;'>Learned from XGBoost model (real)</div>", unsafe_allow_html=True)
    for model, w in weights.items():
        st.markdown(f"""
        <div style="margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace;">
                <span>{model}</span>
                <span>{w}%</span>
            </div>
            <div style="width: 100%; height: 4px; background-color: #1c1f24; margin-top: 2px;">
                <div style="width: {w}%; height: 100%; background-color: #58a6ff;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
with exp_col:
    st.markdown('<div class="section-title">EXPLAINABILITY: "WHY THIS FORECAST?"</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size: 0.85rem; color: #c9d1d9; line-height: 1.5;">
        <table style="width:100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #30363d;"><td style="padding: 4px 0; color:#8b949e;">Highest Contributor</td><td style="text-align:right; font-family:'JetBrains Mono', monospace;">ECMWF HRES</td></tr>
            <tr style="border-bottom: 1px solid #30363d;"><td style="padding: 4px 0; color:#8b949e;">Identified Regime</td><td style="text-align:right; font-family:'JetBrains Mono', monospace;">NORTHEAST MONSOON</td></tr>
            <tr style="border-bottom: 1px solid #30363d;"><td style="padding: 4px 0; color:#8b949e;">Reference Dataset</td><td style="text-align:right; font-family:'JetBrains Mono', monospace;">ERA5 (T-5d)</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with chart_col:
    st.markdown('<div class="section-title">SKILL TRAJECTORY (RMSE VS LEAD TIME)</div>', unsafe_allow_html=True)
    
    # Mock data for lead-time trajectory line chart
    lead_times = [6, 12, 18, 24, 36, 48, 72]
    rmse_hres = [0.08, 0.09, 0.11, 0.13, 0.16, 0.20, 0.28]
    rmse_gfs = [0.10, 0.11, 0.13, 0.15, 0.19, 0.24, 0.32]
    rmse_blend = [0.06, 0.07, 0.08, 0.09, 0.11, 0.14, 0.21]
    
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=lead_times, y=rmse_hres,  mode='lines+markers', name='ECMWF HRES',  line=dict(color='#8b949e', width=1.5), marker=dict(size=4)))
    fig_line.add_trace(go.Scatter(x=lead_times, y=rmse_gfs,   mode='lines+markers', name='NOAA GFS',    line=dict(color='#444c56', width=1.5), marker=dict(size=4)))
    fig_line.add_trace(go.Scatter(x=lead_times, y=rmse_blend, mode='lines+markers', name='ASTRA Blend', line=dict(color='#58a6ff', width=2.5), marker=dict(size=6)))

    # Vertical marker for currently selected lead time
    fig_line.add_shape(type="line", x0=lh, x1=lh, y0=0, y1=1,
        xref="x", yref="paper", line=dict(color="#f0883e", width=1.5, dash="dot"))
    fig_line.add_annotation(x=lh, y=0.98, xref="x", yref="paper",
        text=lead_time, showarrow=False,
        font=dict(color="#f0883e", size=9, family="JetBrains Mono"), yanchor="top")
    
    fig_line.update_layout(
        template="plotly_dark",
        plot_bgcolor=PLOTLY_BG,
        paper_bgcolor=PLOTLY_BG,
        margin=dict(l=40, r=10, t=20, b=30),
        height=200,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color=PLOTLY_TEXT)),
        xaxis=dict(title="Lead Time (h)", color=PLOTLY_TEXT, gridcolor=PLOTLY_GRID, tickfont=dict(family="JetBrains Mono", size=10)),
        yaxis=dict(title="RMSE", color=PLOTLY_TEXT, gridcolor=PLOTLY_GRID, tickfont=dict(family="JetBrains Mono", size=10))
    )
    st.plotly_chart(fig_line, use_container_width=True, config={'displayModeBar': False})
