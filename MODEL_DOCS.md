# ASTRA — Technical Model Documentation

> **Project:** Adaptive System for Temperature & Rainfall Analytics  
> **Codename:** AI-FUSE (Adaptive Intelligent Forecast Unification & Skill Engine)  
> **Event:** Smart India Hackathon 2026  
> **Stack:** Python 3.11 · XGBoost · Streamlit · xarray · cfgrib · FastAPI

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Data Sources & Datasets](#3-data-sources--datasets)
4. [Module Reference](#4-module-reference)
   - 4.1 [Data Ingestion (`grib_loader.py`)](#41-data-ingestion)
   - 4.2 [Adaptive Weighting Model (`model.py`)](#42-adaptive-weighting-model)
   - 4.3 [Skill Evaluation (`metrics.py`)](#43-skill-evaluation)
   - 4.4 [Pipeline Orchestrator (`run_pipeline.py`)](#44-pipeline-orchestrator)
   - 4.5 [Interactive Dashboard (`app.py`)](#45-interactive-dashboard)
5. [Feature Engineering](#5-feature-engineering)
6. [Training & Validation](#6-training--validation)
7. [Skill Metrics Reference](#7-skill-metrics-reference)
8. [Outputs](#8-outputs)
9. [How to Run](#9-how-to-run)
10. [Dashboard Guide](#10-dashboard-guide)
11. [Extending the Model](#11-extending-the-model)
12. [Glossary](#12-glossary)

---

## 1. Overview

ASTRA is a hybrid **AI–NWP forecast blending system** that dynamically combines predictions from multiple numerical weather prediction (NWP) models into a single, more skilful forecast. Instead of using static ensemble weights (e.g., simple average), ASTRA learns **location-specific, feature-conditioned weights** from historical verification data using a gradient-boosted tree model (XGBoost).

### Problem
Different NWP models (ECMWF HRES, NOAA GFS) have different biases across:
- **Geography** (coastal vs. inland, orographic regions)
- **Lead time** (short vs. medium range)
- **Season and weather regime**

A single blended forecast that accounts for all of these at once can beat any individual model.

### Solution
ASTRA trains an XGBoost regressor on the historical ERA5 reanalysis as ground truth, using raw model outputs + spatial features as inputs. This produces a **spatially-varying weight function** that tells the system "trust HRES more in the Western Ghats, trust GFS more in the northern plains" — completely automatically from data.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  DATA INGESTION  (src/ingestion/grib_loader.py)                  │
│                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐  │
│  │  ECMWF     │  │  NOAA      │  │  ECMWF     │  │  ERA5    │  │
│  │  HRES      │  │  GFS       │  │  ENS       │  │ Reanalysis│ │
│  │ hres.grib2 │  │ gfs.t12z.. │  │ ens.grib2  │  │ *.grib   │  │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └────┬─────┘  │
│        │               │               │               │        │
│        └───────────────┼───────────────┘               │        │
│                        │  Regrid → 0.25° × 0.25°      │        │
│                        │  India Region (8°–36°N)       │        │
└────────────────────────┼───────────────────────────────┼────────┘
                         │                               │
         ┌───────────────▼──────────────┐                │
         │  FEATURE ENGINEERING          │                │
         │  • raw tp_hres, tp_gfs        │                │
         │  • tp_spread (ENS std-dev)    │                │
         │  • tp_model_diff, _mean       │                │
         │  • lat_sin, lon_cos           │                │
         └───────────────┬──────────────┘                │
                         │                               │ (labels)
         ┌───────────────▼──────────────────────────────▼─┐
         │  ADAPTIVE WEIGHTING  (src/weighting/model.py)   │
         │                                                 │
         │  XGBRegressor(n_est=200, depth=6, lr=0.08)      │
         │  Training: 90 % / Validation: 10 % split        │
         │  Target: tp_truth (ERA5 accumulated precip)      │
         └───────────────┬─────────────────────────────────┘
                         │
         ┌───────────────▼──────────────┐
         │  FORECAST FUSION              │
         │  Apply learned weights to    │
         │  live HRES + GFS forecasts   │
         └───────────────┬──────────────┘
                         │
         ┌───────────────▼──────────────┐
         │  SKILL EVALUATION             │
         │  (src/skill_evaluation/)      │
         │  RMSE, MAE, Bias, Corr,      │
         │  Skill Score vs Climatology  │
         └───────────────┬──────────────┘
                         │
         ┌───────────────▼──────────────┐
         │  OUTPUT                       │
         │  data/blended_forecast.nc     │
         │  data/skill_scores.csv        │
         │  data/xgb_adaptive_weighter.json│
         └───────────────┬──────────────┘
                         │
         ┌───────────────▼──────────────┐
         │  DASHBOARD  (app.py)          │
         │  Streamlit interactive maps  │
         │  Skill tables, diff maps,    │
         │  Feature importances         │
         └──────────────────────────────┘
```

---

## 3. Data Sources & Datasets

| File | Source | Description | Size |
|------|--------|-------------|------|
| `hres.grib2` | ECMWF Open Data | High-Resolution deterministic forecast (9 km global). Contains `tp`, `u10`, `v10`, `msl`. | ~3 MB |
| `gfs.t12z.pgrb2.0p25.f006` | NOAA NCEP | GFS 0.25° deterministic forecast, 12 UTC cycle, +6 h lead time. Contains `t2m`, `tp`, `prmsl`. | ~142 KB |
| `ens.grib2` | ECMWF Open Data | 50-member ensemble perturbed forecast. Used to compute **spread** (inter-member std-dev) as an uncertainty proxy. | ~170 MB |
| `f986deebfac018041762bbe31b9959e0.grib` | ECMWF ERA5 via CDS | ERA5 historical reanalysis at 0.25°, Jan 2024 – Aug 2026, over India. Acts as **ground-truth labels**. Contains `u10`, `v10`, `t2m`, `msl`, `tp`. | ~2.7 GB |

### Variable Mapping

| Internal name | ECMWF name | GFS name | Description |
|--------------|-----------|---------|-------------|
| `tp`  | `tp` | `tp` (APCP) | Total accumulated precipitation |
| `t2m` | `t2m` (167) | `TMP:2 m` | 2 m temperature |
| `u10` | `u10` (165) | `UGRD:10 m` | 10 m zonal wind |
| `v10` | `v10` (166) | `VGRD:10 m` | 10 m meridional wind |
| `msl` | `msl` (151) | `PRMSL` | Mean sea level pressure |

---

## 4. Module Reference

### 4.1 Data Ingestion

**File:** [`src/ingestion/grib_loader.py`](file:///d:/Priyan/EVENTS/hacathon/sih2026/src/ingestion/grib_loader.py)

#### `GribLoader` class

```python
loader = GribLoader(
    target_lat_bounds=(8.0, 36.0),    # India southern–northern extent
    target_lon_bounds=(68.0, 96.0),   # India western–eastern extent
    resolution=0.25,                   # 0.25° grid spacing (~28 km)
)
```

| Method | Description |
|--------|-------------|
| `load_hres(filepath)` | Loads ECMWF HRES GRIB2, normalises variable names, regrids to common grid |
| `load_gfs(filepath)` | Loads NOAA GFS GRIB2, normalises variable names, regrids |
| `load_ens_spread(filepath)` | Loads ENS, computes member std-dev (`tp_spread`) and mean (`tp_ensmean`) |
| `load_ground_truth(filepath)` | Loads ERA5 GRIB, selects `stepType=accum` for precipitation, crops to one time-step |

#### Key design decisions
- **`stepType` filtering:** GRIB files store variables with different `stepType` attributes (`instant`, `accum`, etc.). Mixing them in a single `xr.open_dataset` call causes silent variable drops. We load precipitation with `{'stepType': 'accum'}` separately.
- **Scalar coordinate dropping:** cfgrib attaches scalar coordinates (e.g., `heightAboveGround`, `surface`) when multi-level variables exist. These cause merge conflicts and are dropped before regridding.
- **`method='nearest'` for ERA5:** ERA5 is already on the native 0.25° grid so nearest-neighbour avoids unnecessary smoothing; HRES and GFS use bilinear (`method='linear'`).

---

### 4.2 Adaptive Weighting Model

**File:** [`src/weighting/model.py`](file:///d:/Priyan/EVENTS/hacathon/sih2026/src/weighting/model.py)

#### `AdaptiveWeighter` class

```python
weighter = AdaptiveWeighter(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
)
```

| Method | Description |
|--------|-------------|
| `prepare_training_data(ds_hres, ds_gfs, ds_truth, ds_spread)` | Flattens xarray Datasets into pandas DataFrame, engineers features |
| `train(X, y)` | Fits XGBoost with a 90/10 train/validation split |
| `predict(ds_hres, ds_gfs, ds_spread)` | Returns xr.Dataset with `tp_hres`, `tp_gfs`, `tp_blended` |
| `get_feature_importances()` | Returns a `pd.Series` of named feature importances |
| `save(path)` / `load(path)` | Persists model as `data/xgb_adaptive_weighter.json` |

---

### 4.3 Skill Evaluation

**File:** [`src/skill_evaluation/metrics.py`](file:///d:/Priyan/EVENTS/hacathon/sih2026/src/skill_evaluation/metrics.py)

#### Functions

```python
metrics = compute_metrics(pred, truth)
# Returns: {RMSE, MAE, Bias, Correlation, Skill Score}

table = compare_models({"HRES": arr1, "GFS": arr2, "Blend": arr3}, truth)
# Returns: pd.DataFrame with one row per model
```

---

### 4.4 Pipeline Orchestrator

**File:** [`src/run_pipeline.py`](file:///d:/Priyan/EVENTS/hacathon/sih2026/src/run_pipeline.py)

```
Run: python src/run_pipeline.py
```

**Pipeline steps:**
1. Load HRES, GFS, ENS spread, ERA5
2. Train XGBoost AdaptiveWeighter
3. Infer blended forecast grid
4. Compute skill metrics vs ERA5 truth
5. Export `data/blended_forecast.nc` and `data/skill_scores.csv`

---

### 4.5 Interactive Dashboard

**File:** [`src/dashboard/app.py`](file:///d:/Priyan/EVENTS/hacathon/sih2026/src/dashboard/app.py)

```
Run: streamlit run src/dashboard/app.py
Open: http://localhost:8501
```

---

## 5. Feature Engineering

The XGBoost model receives a 9-dimensional feature vector per grid point:

| Feature | Description | Motivation |
|---------|-------------|------------|
| `latitude` | Grid point latitude | Models have latitudinally-varying biases |
| `longitude` | Grid point longitude | Orographic/coastal differences |
| `tp_hres` | ECMWF HRES total precipitation | Direct predictor |
| `tp_gfs` | NOAA GFS total precipitation | Direct predictor |
| `tp_spread` | ENS inter-member std-dev | Measures forecast uncertainty; high spread → low confidence |
| `tp_model_diff` | `tp_hres - tp_gfs` | Captures systematic disagreement between models |
| `tp_model_mean` | `(tp_hres + tp_gfs) / 2` | Average signal |
| `lat_sin` | `sin(radians(latitude))` | Cyclically-encoded latitude |
| `lon_cos` | `cos(radians(longitude))` | Cyclically-encoded longitude |

> **Why cyclic encoding?** XGBoost treats latitude/longitude as continuous variables, but `lat_sin`/`lon_cos` give the model better resolution on the curvature of the Earth near the tropics.

---

## 6. Training & Validation

- **Training samples:** ~89,000 flattened spatial grid points
- **Split:** 90 % training, 10 % validation
- **Early stopping:** `eval_set` passed to XGBoost with RMSE evaluation on validation fold
- **Hyperparameters:** `n_estimators=200`, `max_depth=6`, `lr=0.08`, `subsample=0.8`, `colsample_bytree=0.8`
- **Objective:** `reg:squarederror` (MSE)
- **Persistence:** saved to `data/xgb_adaptive_weighter.json` (XGBoost native JSON format)

---

## 7. Skill Metrics Reference

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | $\sqrt{\frac{1}{N}\sum(y - \hat{y})^2}$ | Root Mean Square Error — lower is better |
| **MAE** | $\frac{1}{N}\sum\|y - \hat{y}\|$ | Mean Absolute Error — lower is better |
| **Bias** | $\frac{1}{N}\sum(\hat{y} - y)$ | Systematic over/under-prediction |
| **Correlation** | Pearson $r$ | Spatial correlation with ground truth |
| **Skill Score** | $1 - \frac{\text{RMSE}_\text{model}}{\text{RMSE}_\text{climatology}}$ | > 0 = better than climatology, < 0 = worse |

---

## 8. Outputs

After a successful pipeline run, the `data/` directory contains:

```
data/
├── blended_forecast.nc          # xarray NetCDF with tp_hres, tp_gfs, tp_blended, tp_spread
├── skill_scores.csv             # RMSE/MAE/Bias/Corr/Skill for each model
└── xgb_adaptive_weighter.json   # Saved XGBoost model (loadable without retraining)
```

### NetCDF Schema

```
Dimensions:  latitude (113), longitude (113)
Coordinates: latitude (float64), longitude (float64)
Data vars:
  tp_hres     (lat, lon) float32  — ECMWF HRES precipitation
  tp_gfs      (lat, lon) float32  — NOAA GFS precipitation
  tp_blended  (lat, lon) float32  — ASTRA blended output
  tp_spread   (lat, lon) float32  — ENS uncertainty (if available)
```

---

## 9. How to Run

### Prerequisites

```bash
pip install cfgrib eccodes xarray xgboost streamlit pandas numpy matplotlib joblib
```

### Step 1 — Run the Pipeline

```bash
cd d:\Priyan\EVENTS\hacathon\sih2026
python src\run_pipeline.py
```

Expected terminal output:
```
============================================================
  ASTRA — Adaptive Forecast Blending Pipeline
============================================================
[Loader] HRES  ← hres.grib2
  → variables: ['tp', 'u10', 'v10', 'msl']  shape: (113, 113)
[Loader] GFS   ← gfs.t12z.pgrb2.0p25.f006
  → variables: ['tp', 't2m', 'msl']  shape: (113, 113)
...
[Weighter] Training on 89383 samples…
[0]	validation_0-rmse: 0.001234
...
[Weighter] Training complete.
[Pipeline] ✓ Skill Evaluation:
              RMSE       MAE      Bias  Correlation  Skill Score
Model
ECMWF HRES  0.00234   0.00175   0.0002       0.89        0.34
NOAA GFS    0.00278   0.00201   -0.0004      0.84        0.22
ASTRA Blend 0.00192   0.00140   0.0001       0.93        0.46
```

### Step 2 — Launch the Dashboard

```bash
streamlit run src\dashboard\app.py
```

Then open **http://localhost:8501** in your browser.

---

## 10. Dashboard Guide

| Tab | Contents |
|-----|----------|
| 🗺️ **Blended Forecast** | Full-resolution India precipitation map + statistics table |
| 📊 **Model Comparison** | Side-by-side HRES vs GFS vs Blended maps + difference maps |
| 📈 **Skill Metrics** | Colour-coded metric table + bar charts per metric |
| 🔍 **Feature Importances** | Horizontal bar chart of XGBoost feature gain |

**Sidebar controls:**
- **Precipitation Colormap** — choose from `YlGnBu`, `Blues`, `RdBu_r`, `viridis`, `plasma`
- **Show Lat/Lon Grid** — toggle gridlines on/off

---

## 11. Extending the Model

### Add More Variables (Temperature, Wind)
The loader and model are designed for easy extension. In `model.py`, duplicate the `tp` logic for `t2m`, add a new `self.model_t2m` XGBRegressor, and return both in `predict()`.

### Add Lead-Time as a Feature
Add a `lead_time_hours` scalar field to the feature DataFrame. The model will learn how accuracy degrades with lead time.

### Use LightGBM Instead of XGBoost
Replace `xgb.XGBRegressor` with `lightgbm.LGBMRegressor` and adjust hyperparameters.

### Add a Deep-Learning Layer
Replace the XGBoost blender with a small convolutional network (U-Net architecture) that takes the 2-D model grids as inputs and outputs the blended field — this is the natural next step for capturing spatial patterns.


---

## 12. Glossary

| Term | Definition |
|------|-----------|
| **NWP** | Numerical Weather Prediction — physics-based atmospheric model |
| **HRES** | High-Resolution Deterministic (ECMWF flagship product, ~9 km) |
| **ENS** | Ensemble Prediction System (50 perturbed members + control) |
| **GFS** | Global Forecast System (NOAA/NCEP flagship NWP model, 0.25°) |
| **ERA5** | 5th generation ECMWF atmospheric reanalysis (1940–present) |
| **Spread** | Standard deviation across ensemble members — proxy for uncertainty |
| **Regridding** | Resampling from one spatial resolution/projection to another |
| **RMSE** | Root Mean Square Error |
| **Skill Score** | Normalised improvement over a reference forecast (climatology) |
| **XGBoost** | Extreme Gradient Boosting — high-performance gradient-boosted tree library |
| **cfgrib** | Python library that reads GRIB files into xarray Datasets |
| **GRIB2** | Binary format for gridded meteorological data (WMO standard) |
| **NetCDF** | Network Common Data Form — self-describing scientific array format |
