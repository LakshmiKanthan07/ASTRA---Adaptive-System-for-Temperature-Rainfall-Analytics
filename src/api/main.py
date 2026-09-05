"""
ASTRA — FastAPI REST Backend
=============================
Exposes the blended forecast via a lightweight REST API so external applications
(GIS portals, mobile apps, downstream data pipelines) can consume ASTRA outputs.

Endpoints
---------
GET /                      — health check / system info
GET /forecast              — blended forecast for a lat/lon point
GET /weights               — current adaptive model weights per variable
GET /alerts                — active extreme-weather alerts
GET /skill                 — model skill comparison table
GET /confidence            — per-variable confidence score
POST /feedback/trigger     — trigger one feedback learning cycle

Usage
-----
    uvicorn src.api.main:app --reload --port 8000
    # API docs:  http://localhost:8000/docs
"""
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from src.confidence.scorer import ConfidenceScorer

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ASTRA — AI-NWP Blending API",
    description=(
        "Adaptive System for Temperature, Rainfall & Analytics. "
        "REST API exposing blended weather forecast outputs from the "
        "ASTRA AI-NWP multi-model blending framework."
    ),
    version="2.0.0",
    contact={
        "name": "NCMRWF / MoES",
        "url": "https://github.com/LakshmiKanthan07/ASTRA",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_NC     = "data/blended_forecast.nc"
ALERTS_JSON = "data/alerts.json"
FEEDBACK_JSON = "data/feedback_report.json"

VARS = ["tp", "t2m", "wind"]

# ─── cached loading ───────────────────────────────────────────────────────────
_ds_cache: Optional[xr.Dataset] = None

def _get_ds() -> xr.Dataset:
    global _ds_cache
    if _ds_cache is None:
        if not os.path.exists(DATA_NC):
            raise HTTPException(
                status_code=503,
                detail="No blended forecast available. Run the ASTRA pipeline first.",
            )
        _ds_cache = xr.open_dataset(DATA_NC)
    return _ds_cache


def _load_skill(var: str) -> Optional[pd.DataFrame]:
    path = f"data/skill_scores_{var}.csv"
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    path_generic = "data/skill_scores.csv"
    if os.path.exists(path_generic):
        return pd.read_csv(path_generic, index_col=0)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", summary="System status and API info")
def root():
    ds_available = os.path.exists(DATA_NC)
    return {
        "system":        "ASTRA — Adaptive NWP Blending Framework",
        "version":       "2.0.0",
        "status":        "operational" if ds_available else "awaiting_pipeline",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "endpoints": [
            "/forecast", "/weights", "/alerts", "/skill",
            "/confidence", "/feedback/trigger", "/docs",
        ],
        "data_available": ds_available,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Forecast at a point
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/forecast", summary="Blended forecast at a lat/lon grid-point")
def forecast_at_point(
    lat: float = Query(..., description="Latitude  (°N, e.g. 13.08 for Chennai)", ge=-90, le=90),
    lon: float = Query(..., description="Longitude (°E, e.g. 80.27 for Chennai)", ge=0, le=360),
    variable: str = Query("all", description="Variable: tp | t2m | wind | all"),
    lead_hours: int = Query(24, description="Forecast lead time in hours", ge=6, le=72),
):
    ds = _get_ds()

    # Nearest-neighbour lookup
    lat_arr = ds.latitude.values
    lon_arr = ds.longitude.values
    lat_idx = int(np.argmin(np.abs(lat_arr - lat)))
    lon_idx = int(np.argmin(np.abs(lon_arr - lon)))
    actual_lat = float(lat_arr[lat_idx])
    actual_lon = float(lon_arr[lon_idx])

    result = {
        "requested":   {"latitude": lat, "longitude": lon},
        "nearest_grid": {"latitude": actual_lat, "longitude": actual_lon},
        "lead_hours":  lead_hours,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "data_source": "ASTRA Blended (ECMWF HRES + NOAA GFS + XGBoost Adaptive Weighting)",
        "forecast":    {},
    }

    def _val(name: str):
        if name in ds:
            v = float(ds[name].values[lat_idx, lon_idx])
            return v if np.isfinite(v) else None
        return None

    vars_to_return = VARS if variable == "all" else [variable]
    for var in vars_to_return:
        blended = _val(f"{var}_blended")
        hres    = _val(f"{var}_hres")
        gfs     = _val(f"{var}_gfs")
        hres_w  = _val(f"{var}_hres_weight")
        gfs_w   = _val(f"{var}_gfs_weight")

        if blended is None:
            continue

        unit = {"tp": "m (ECMWF convention, multiply by 1000 for mm)",
                "t2m": "K (subtract 273.15 for °C)",
                "wind": "m/s"}.get(var, "")

        result["forecast"][var] = {
            "blended":       blended,
            "ecmwf_hres":    hres,
            "noaa_gfs":      gfs,
            "hres_weight":   hres_w,
            "gfs_weight":    gfs_w,
            "unit":          unit,
        }

    if not result["forecast"]:
        raise HTTPException(status_code=404, detail=f"Variable '{variable}' not found in blended forecast.")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Weights
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/weights", summary="Current adaptive model weights per variable")
def adaptive_weights():
    summary = {}
    for var in VARS:
        w_file = f"data/learned_weights_{var}.csv"
        if not os.path.exists(w_file):
            w_file = "data/learned_weights.csv"
        if os.path.exists(w_file):
            df = pd.read_csv(w_file)
            hres_col = f"{var}_hres_weight"
            gfs_col  = f"{var}_gfs_weight"
            if hres_col in df.columns:
                summary[var] = {
                    "ECMWF_HRES_weight_mean": round(float(df[hres_col].mean()), 3),
                    "NOAA_GFS_weight_mean":   round(float(df[gfs_col].mean()), 3),
                    "spatial_variation_std":  round(float(df[hres_col].std()), 3),
                    "n_grid_points":          len(df),
                }
    if not summary:
        raise HTTPException(status_code=503, detail="Weight data not available. Run the pipeline first.")
    return {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "weights":       summary,
        "model": "XGBoost Adaptive Weighter (trained on ERA5 ground truth)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Alerts
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/alerts", summary="Active extreme weather alerts")
def get_alerts():
    if not os.path.exists(ALERTS_JSON):
        return {"active_alerts": 0, "alerts": [], "message": "No alert data. Run pipeline."}
    with open(ALERTS_JSON) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Skill Summary
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/skill", summary="Model skill comparison (RMSE, MAE, Bias, Correlation)")
def skill_summary(variable: str = Query("tp", description="Variable: tp | t2m | wind")):
    df = _load_skill(variable)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Skill scores for '{variable}' not found.")
    return {
        "variable": variable,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "skill_table": df.reset_index().to_dict(orient="records"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Confidence Score
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/confidence", summary="Per-variable forecast confidence score (0–100)")
def confidence(
    variable: str   = Query("tp", description="Variable: tp | t2m | wind"),
    lead_hours: int = Query(24,   description="Forecast horizon in hours", ge=6, le=72),
):
    ds       = _get_ds()
    skill_df = _load_skill(variable)
    scorer   = ConfidenceScorer()
    score, level, breakdown = scorer.compute(ds, var=variable, lead_hours=lead_hours, skill_df=skill_df)
    return {
        "variable":      variable,
        "lead_hours":    lead_hours,
        "score":         score,
        "level":         level,
        "breakdown":     breakdown,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "interpretation": (
            f"Forecast confidence is {level} ({score}%). "
            "Based on model agreement, ensemble spread, lead-time horizon, and historical skill."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trigger Feedback Loop
# ─────────────────────────────────────────────────────────────────────────────
class FeedbackRequest(BaseModel):
    observation_source: str = "synthetic"

@app.post("/feedback/trigger", summary="Run one automated feedback and weight-update cycle")
def trigger_feedback(body: FeedbackRequest = FeedbackRequest()):
    try:
        from src.feedback.updater import FeedbackUpdater
        updater = FeedbackUpdater(observation_source=body.observation_source)
        report  = updater.run()
        # Invalidate cache so next /forecast reflects updated weights
        global _ds_cache
        _ds_cache = None
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Last Feedback Report
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/feedback/latest", summary="Retrieve last feedback loop report")
def last_feedback():
    if not os.path.exists(FEEDBACK_JSON):
        return {"message": "No feedback report yet. POST /feedback/trigger first."}
    with open(FEEDBACK_JSON) as f:
        return json.load(f)
