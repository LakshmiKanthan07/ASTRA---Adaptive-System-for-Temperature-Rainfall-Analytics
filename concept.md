# AI–NWP Adaptive Forecast Blending Framework — Execution Plan

**Project codename:** AI-FUSE (Adaptive Intelligent Forecast Unification & Skill Engine)
**Purpose of this document:** Complete build spec for an autonomous/agentic dev tool. Contains architecture, module-by-module implementation detail, data contracts, algorithms, folder structure, and a phased execution schedule. Follow phases in order — each phase produces a runnable artifact before the next begins.

---

## 1. Problem Statement (restated)

Different forecasting systems (physical NWP models, ensemble forecasts, AI/ML weather models) perform differently depending on region, season, lead time, and weather situation. Build a hybrid **AI–NWP blending framework** that assigns **adaptive weights** to different forecast sources based on **historical skill, forecast lead time, region, season, and weather regime**, producing one optimized forecast for **rainfall, temperature, wind, and extreme-weather indicators**.

## 2. Expected Deliverables (from Problem Statement)

| Deliverable | Description |
|---|---|
| Dynamically blended forecast | Best-combined forecast from multiple model sources |
| Model weight maps | Which model is more reliable, per region/lead time |
| Improved forecast skill | Measurably better than any individual model |
| Extreme weather guidance | Improved signal for heavy rainfall, heatwave, high-wind events |
| Operational workflow | Automated script/dashboard for routine forecast blending |

---

## 3. System Architecture

```
                    ┌────────────────────┐
                    │   DATA INGESTION    │
                    │  (NWP, ENS, AI/ML,  │
                    │   Obs, Reanalysis)  │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  PREPROCESSING &    │
                    │  COMMON REGRIDDING  │
                    └─────────┬──────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
      ┌───────▼──────┐ ┌──────▼───────┐ ┌─────▼────────┐
      │  NWP Sources  │ │  Ensemble    │ │  AI/ML Models │
      │ (GFS/ECMWF/   │ │  Spread      │ │ (LSTM/Transf/ │
      │  WRF)         │ │  Products    │ │  GNN)         │
      └───────┬──────┘ └──────┬───────┘ └─────┬────────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  SKILL EVALUATION   │
                    │  (RMSE, CRPS,       │
                    │   Brier Score)      │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │ ADAPTIVE WEIGHTING  │
                    │  MODULE (XGBoost /  │
                    │  LightGBM)          │
                    │  features: region,  │
                    │  lead time, season, │
                    │  weather regime     │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  FORECAST FUSION    │
                    │  ENGINE             │
                    └─────────┬──────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
      ┌───────▼──────┐ ┌──────▼───────┐ ┌─────▼────────┐
      │ Blended       │ │ Model Weight │ │ Extreme Event │
      │ Forecast      │ │ Maps         │ │ Flags         │
      │ (rain/temp/   │ │              │ │ (heavy rain,  │
      │  wind)        │ │              │ │  heatwave,    │
      └───────┬──────┘ └──────┬───────┘ │  high wind)   │
              │               │         └─────┬────────┘
              └───────────────┼───────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  DASHBOARD + API    │
                    │ (Streamlit/Grafana, │
                    │  REST API)          │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  ORCHESTRATION      │
                    │  (Docker + Grafana, │
                    │  scheduled runs)    │
                    └─────────────────────┘
```

---

## 4. Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| Language | Python 3.11+ | Single language across the stack |
| AI/ML frameworks | PyTorch (primary), TensorFlow (optional) | LSTM, Transformer, GNN implementations |
| Weighting model | XGBoost, LightGBM | Adaptive weighting layer |
| Gridded data handling | xarray, NetCDF4, cfgrib | NWP/ensemble data in NetCDF/GRIB2 |
| Data validation | pydantic | Schema validation on ingested data |
| Numerical/scientific | NumPy, SciPy, pandas | General computation |
| Geospatial regridding | xESMF or scipy.interpolate | Common-grid preprocessing |
| Experiment tracking | MLflow (optional but recommended) | Track weighting-model retraining runs |
| Containerization | Docker, docker-compose | One container per pipeline stage |
| Monitoring    | Grafana        | Real-time system monitoring |
| API | FastAPI | REST API for forecast retrieval |
| Dashboard | Streamlit (fast prototype) / Grafana (ops-grade) | Forecaster-facing UI |
| Storage | PostgreSQL (metadata/skill scores) + object storage (S3/MinIO) for gridded data | |
| CI | GitHub Actions | Lint, unit tests, build checks |

---

## 5. Data Sources & Access

| Source | Use | Link |
|---|---|---|
| IMD (India Meteorological Department) | NWP output, verification | https://mausam.imd.gov.in/ |
| NCMRWF | NWP model documentation & output | https://www.ncmrwf.gov.in/ |
| ECMWF Open Data (ENS + HRES) | Ensemble & high-res deterministic forecasts | https://www.ecmwf.int/en/forecasts/datasets/open-data |
| ECMWF Open Data technical docs | Access method reference | https://confluence.ecmwf.int/display/DAC/ECMWF+open+data%3A+real-time+forecasts |
| NOAA NCEP (GFS) | Global deterministic NWP | https://www.ncep.noaa.gov/ |
| ERA5 Reanalysis | Historical training data | https://cds.climate.copernicus.eu/ |
| Krishnamurti et al. (1999) | Superensemble methodology reference | https://doi.org/10.1126/science.285.5433.1548 (open copy: https://eprints.iisc.ac.in/243/1/tnkrish.pdf) |
| Ground station + INSAT satellite obs | Verification / ground truth | via IMD data portal |

**Data acquisition libraries:** `ecmwf-opendata` (pip) for ECMWF; `herbie` (pip) for GFS/ECMWF archive pulls; `cdsapi` (pip) for ERA5/Copernicus CDS.

---

## 5a. Exact Dataset Specifications

Concrete product identifiers, variable codes, and request parameters for each source — use these directly in `src/ingestion/`.

### GFS (NOAA NCEP) — via `herbie`

- **Model:** `gfs` | **Product:** `pgrb2.0p25` (0.25° global, common fields)
- **Cycles:** `00z, 06z, 12z, 18z` | **Lead times:** `f000`–`f384` (3-hourly steps typical)
- **Variables (GRIB short names) to pull:**
  - `TMP:2 m above ground` → 2 m temperature
  - `APCP:surface` → accumulated precipitation
  - `UGRD:10 m above ground`, `VGRD:10 m above ground` → 10 m wind components
  - `RH:2 m above ground` → 2 m relative humidity (optional, useful covariate)
- **Example:**
  ```python
  from herbie import Herbie
  H = Herbie("2026-09-04 00:00", model="gfs", product="pgrb2.0p25", fxx=24)
  ds = H.xarray(":(TMP:2 m above|APCP:surface|UGRD:10 m above|VGRD:10 m above):")
  ```
- **Sources checked automatically by herbie:** AWS, NOMADS, Google, Azure, NCAR RDA (falls back through them).

### ECMWF Open Data (HRES + ENS) — via `ecmwf-opendata`

- **Streams:** `oper` (HRES deterministic, 9 km) | `enfo` (ENS ensemble, 18 km, 51 members)
- **Params (ECMWF short names):** `2t` (2 m temp), `tp` (total precip), `10u`/`10v` (10 m wind components), `msl` (mean sea level pressure)
- **Model runs:** `00z, 06z, 12z, 18z`
- **Example (HRES):**
  ```python
  from ecmwf.opendata import Client
  client = Client(source="ecmwf")
  client.retrieve(time=0, stream="oper", type="fc", step=24,
                   param=["2t", "tp", "10u", "10v"], target="hres_f24.grib2")
  ```
- **Example (ENS, all 50 perturbed + control members):**
  ```python
  client.retrieve(time=0, stream="enfo", type="pf", step=24,
                   param="2t", target="ens_f24.grib2")   # type="cf" for control member
  ```
- **Note:** only the last 4 days are on ECMWF's own open-data endpoint; for archive/backfill use `herbie` with `model="ecmwf"` (pulls from Azure/AWS mirrors going back further).

### ERA5 Reanalysis — via `cdsapi` (Copernicus Climate Data Store)

- **Dataset name:** `reanalysis-era5-single-levels`
- **Variables:** `2m_temperature`, `total_precipitation`, `10m_u_component_of_wind`, `10m_v_component_of_wind`, `mean_sea_level_pressure`, `2m_dewpoint_temperature`
- **Requires:** free CDS account + API key in `~/.cdsapirc` (`url: https://cds.climate.copernicus.eu/api`, `key: <token>`); accept dataset Terms of Use on the CDS site first.
- **Example:**
  ```python
  import cdsapi
  client = cdsapi.Client()
  client.retrieve("reanalysis-era5-single-levels", {
      "product_type": "reanalysis",
      "variable": ["2m_temperature", "total_precipitation",
                    "10m_u_component_of_wind", "10m_v_component_of_wind"],
      "year": "2025", "month": "06", "day": [f"{d:02d}" for d in range(1, 31)],
      "time": [f"{h:02d}:00" for h in range(24)],
      "area": [37, 68, 8, 92],          # [North, West, South, East] — India bbox
      "grid": [0.25, 0.25],
      "data_format": "netcdf",
  }, "era5_india_june2025.nc")
  ```
- **Known gotcha:** request *instantaneous* params (temp, wind, pressure) and *accumulated* params (precip, radiation) in **separate requests** — mixing both in one call has caused silent drops of the accumulated fields.

### IMD (India Meteorological Department) — mausam.imd.gov.in

- **No open, documented bulk-download API.** The public portal serves rendered maps/bulletins and city-level forecast text for human consumption, not a machine-readable feed.
- **What's realistically usable for this project:**
  - City/district forecast and warning bulletins (for verification/labels, scraped or manually collected)
  - Published extreme-event criteria (heavy rain / heatwave / gale thresholds) — used in Section 7.7, not as a data feed
- **Action item for the build agent:** treat IMD as a **verification/labels source**, not an ingestion pipeline target, unless the team obtains a data-sharing arrangement (IMD data requests typically go through an official request process, not an anonymous API).

### NCMRWF — Regional Data Sharing (RDS) Portal

- **Portal:** https://rds.ncmrwf.gov.in (public access, free registration)
- **Datasets available:**
  - **IMDAA Reanalysis** — 12 km, hourly, regional reanalysis over India, 1979–2020
  - **NGFS Reanalysis** — 25 km, 6-hourly, global reanalysis, 1999–2018
  - **IMDAA-Like products** (post-2020 continuity, from operational NCUM global NWP, 12 km; multi-level fields at 18 pressure levels)
- **Access method:** web portal / THREDDS-style download (confirm exact download protocol on the portal after registration — it is not a simple REST endpoint like ECMWF/CDS).
- **Use in this project:** primary **India-region ground-truth/training** source (regional resolution is much finer than global GFS/ERA5), and as the NWP baseline most directly comparable to IMD's own operational forecasts.

### Summary table

| Source | Dataset ID / Product | Access method | Role |
|---|---|---|---|
| GFS | `gfs`, product `pgrb2.0p25` | `herbie` | NWP input stream |
| ECMWF HRES | stream `oper` | `ecmwf-opendata` / `herbie` | NWP input stream |
| ECMWF ENS | stream `enfo` | `ecmwf-opendata` | Ensemble spread input stream |
| ERA5 | `reanalysis-era5-single-levels` | `cdsapi` | Training / historical skill backfill |
| NCMRWF | IMDAA / NGFS / IMDAA-Like | RDS portal (rds.ncmrwf.gov.in) | India-region ground truth + regional NWP baseline |
| IMD | mausam.imd.gov.in bulletins | manual/scrape | Verification labels + extreme-event thresholds only |

---

## 6. Repository Structure

```
ai-fuse/
├── README.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── config/
│   ├── sources.yaml          # data source endpoints, credentials refs
│   ├── regions.yaml          # region definitions (bounding boxes/zones)
│   └── model_config.yaml     # hyperparameters for AI models & weighting model
├── data/
│   ├── raw/                  # untouched pulls (gitignored)
│   ├── interim/              # regridded/preprocessed
│   └── processed/            # model-ready tensors/frames
├── src/
│   ├── ingestion/
│   │   ├── fetch_nwp.py          # GFS/ECMWF/WRF pull
│   │   ├── fetch_ensemble.py     # ENS spread products
│   │   ├── fetch_reanalysis.py   # ERA5 via cdsapi
│   │   └── fetch_observations.py # ground stations + INSAT
│   ├── preprocessing/
│   │   ├── regrid.py             # common-grid interpolation
│   │   ├── normalize.py          # unit/variable standardization
│   │   └── align_timestamps.py   # lead-time alignment across sources
│   ├── models/
│   │   ├── lstm_model.py
│   │   ├── transformer_model.py
│   │   ├── gnn_model.py
│   │   └── inference.py          # unified inference wrapper for all 3
│   ├── skill_evaluation/
│   │   ├── metrics.py            # RMSE, CRPS, Brier score implementations
│   │   └── rolling_skill.py      # rolling-window skill computation
│   ├── weighting/
│   │   ├── feature_builder.py    # region, lead time, season, regime → features
│   │   ├── train_weighting_model.py  # XGBoost/LightGBM training
│   │   └── predict_weights.py
│   ├── fusion/
│   │   ├── fuse_forecasts.py     # weighted combination logic
│   │   └── confidence_score.py
│   ├── extremes/
│   │   └── flag_extremes.py      # heavy rain / heatwave / high wind thresholds
│   ├── api/
│   │   └── main.py               # FastAPI app
│   ├── dashboard/
│   │   └── app.py                # Streamlit app
│   └── utils/
│       ├── logging_config.py
│       └── schema.py             # pydantic schemas

│   └── dags/
│       └── blending_pipeline_dag.py
├── tests/
│   ├── test_ingestion.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_weighting.py
│   ├── test_fusion.py
│   └── test_api.py
└── notebooks/
    └── eda_and_prototyping.ipynb
```

---

## 7. Module Specifications

### 7.1 Ingestion (`src/ingestion/`)

- Use the exact dataset IDs, variable codes, and libraries from **Section 5a** — do not substitute other product names/resolutions without updating that section first, since downstream regridding and skill-scoring assume these specific fields.
- Each fetch script exposes `fetch(date: str, region: str, lead_times: list[int]) -> xarray.Dataset`.
- Output written to `data/raw/<source>/<date>/<region>.nc`.
- Must be idempotent — re-running for an already-fetched date/region is a no-op unless `--force`.
- Retry with exponential backoff (max 3 retries) on network failure; log and continue on partial failure (don't crash the whole pipeline for one missing source).

### 7.2 Preprocessing (`src/preprocessing/`)

- `regrid.py`: interpolate all sources onto a **common lat/lon grid** (define target resolution in `config/regions.yaml`, e.g. 0.25°).
- `normalize.py`: convert all variables to consistent units (°C, mm/hr, m/s) and consistent naming (`t2m`, `precip`, `wind_speed`).
- `align_timestamps.py`: map each source's native lead-time steps onto a shared lead-time index (e.g. `[6h, 12h, 24h, 48h, 72h, 120h]`).
- Output: `data/interim/<date>/<region>_<variable>.nc`, one aligned tensor per variable per region.

### 7.3 AI/ML Forecast Models (`src/models/`)

| Model | Role | Input | Output |
|---|---|---|---|
| LSTM | Short/medium-range sequence forecasting | Time series of past N timesteps per station/grid cell (t2m, precip, wind, pressure) | Point forecast + variance per lead time |
| Transformer | Medium-range sequence forecasting, better long-range dependency capture | Same as LSTM, with positional lead-time encoding | Point forecast + variance per lead time |
| GNN | Spatial correlation across stations/grid cells | Graph with nodes = grid cells/stations, edges = spatial adjacency or correlation-weighted | Per-node forecast, spatially consistent |

- `inference.py` exposes: `run_all_models(input_tensor) -> dict[str, xarray.Dataset]` returning one forecast dataset per model, all on the common grid/lead-time index defined in preprocessing.
- Model checkpoints stored under `models/checkpoints/<model_name>/<version>/`.
- Training scripts are separate from inference (`train_lstm.py`, `train_transformer.py`, `train_gnn.py` — add under `src/models/training/` if training from scratch is in scope for this hackathon; otherwise document assumed pretrained checkpoints).

### 7.4 Skill Evaluation (`src/skill_evaluation/`)

Implement these metrics in `metrics.py`:

- **RMSE** — `sqrt(mean((forecast - observation)^2))`, per variable, per region, per lead time.
- **CRPS** (Continuous Ranked Probability Score) — for ensemble/probabilistic sources; use `properscoring.crps_ensemble` or implement via empirical CDF.
- **Brier Score** — for extreme-event probability forecasts: `mean((forecast_prob - observed_binary)^2)`.

`rolling_skill.py`: maintain a rolling window (configurable, default 30 days) of per-source skill scores, keyed by `(source, region, lead_time, variable, season, weather_regime)`. Persist to PostgreSQL table `skill_scores`.

### 7.5 Adaptive Weighting Module (`src/weighting/`)

**Feature vector per (region, lead_time, variable) prediction instance:**

```
features = [
    region_id,             # categorical, one-hot or embedded
    lead_time_hours,        # numeric
    season,                 # categorical (winter/pre-monsoon/monsoon/post-monsoon)
    weather_regime,         # categorical (clear-sky/monsoon-active/western-disturbance/cyclonic)
    source_rmse_30d,        # rolling skill, per source
    source_crps_30d,
    source_brier_30d,
    day_of_year_sin_cos,    # cyclical encoding
]
```

**Model:** Gradient Boosted Trees (XGBoost or LightGBM) — multi-output regression, one weight per source, trained so that `sum(weights) == 1` (softmax the raw outputs).

**Training procedure (`train_weighting_model.py`):**
1. Pull historical (feature, per-source-error) pairs from `skill_scores` + preprocessed observation data.
2. Target: weights that minimize blended-forecast error (frame as a differentiable proxy — e.g. train GBT to predict per-source error, then invert to weights via inverse-error normalization, OR directly optimize blended RMSE via a wrapper search).
3. Retrain on a **fixed schedule** (not per-request) — daily or weekly, configurable in `config/model_config.yaml`.
4. Version and store model artifact: `models/checkpoints/weighting_model/<date>.pkl`.

`predict_weights.py`: `get_weights(region, lead_time, season, weather_regime) -> dict[source, weight]`.

### 7.6 Forecast Fusion Engine (`src/fusion/`)

```python
def fuse_forecasts(forecasts: dict[str, xr.Dataset], weights: dict[str, float]) -> xr.Dataset:
    """
    forecasts: {"nwp_gfs": ds, "nwp_ecmwf": ds, "ens_mean": ds,
                "lstm": ds, "transformer": ds, "gnn": ds}
    weights:   {"nwp_gfs": 0.15, "nwp_ecmwf": 0.25, "ens_mean": 0.10,
                "lstm": 0.20, "transformer": 0.20, "gnn": 0.10}
    Returns a single blended xr.Dataset with the same variables/grid/lead-times.
    """
```

- Weighted sum per grid cell / lead time / variable.
- `confidence_score.py`: derive a confidence score per grid cell from **weight entropy** (low entropy = one source dominates = higher confidence) combined with **ensemble spread** at that point.

### 7.7 Extreme Weather Flagging (`src/extremes/`)

Threshold-based + statistical flagging on the blended output:

| Indicator | Rule (example — calibrate against IMD definitions) |
|---|---|
| Heavy rainfall | 24h accumulated precip ≥ 64.5 mm (IMD "heavy rain" threshold) |
| Heat wave | Max temp ≥ 45°C, or ≥ 4.5°C above normal for ≥ 2 consecutive days (region-dependent, per IMD criteria) |
| High wind | Sustained wind speed ≥ 62 km/h (IMD gale-force threshold) or per-region climatological 95th percentile |

Output: a boolean/probability flag layer per grid cell per lead time, plus a summary alert list.

### 7.8 REST API (`src/api/main.py`, FastAPI)

| Endpoint | Method | Description |
|---|---|---|
| `/forecast/{region}` | GET | Latest blended forecast for a region (all variables, all lead times) |
| `/forecast/{region}/weights` | GET | Current model weight map for that region |
| `/forecast/{region}/extremes` | GET | Active extreme-weather flags |
| `/skill/{source}` | GET | Rolling skill score history for a given source |
| `/health` | GET | Liveness/readiness check |

### 7.9 Dashboard (`src/dashboard/app.py`, Streamlit)

- Map view: blended forecast by variable, selectable lead time.
- Weight map view: which source dominates per region (choropleth or grid heatmap).
- Extreme event panel: active alerts, highlighted regions.
- Skill trend view: rolling skill score per source over time.

### 7.10 Monitoring (Grafana)
Grafana provides system-level metrics for the ASTRA platform, monitoring hardware usage, container health, and API latency.

### 7.11 Orchestration (`airflow/dags/blending_pipeline_dag.py`)

DAG stages (each a task, each a Docker container):

```
fetch_nwp >> fetch_ensemble >> fetch_ai_models >> preprocess_regrid
    >> run_skill_evaluation >> compute_adaptive_weights
    >> fuse_forecasts >> flag_extremes >> publish_to_api_and_dashboard
```

- Schedule: every 6 hours (aligned to NWP model run times: 00/06/12/18 UTC).
- Weighting-model retraining is a **separate, less-frequent DAG** (daily/weekly) — do not retrain on every blending run.
- Failure handling: if one AI/ML model or one NWP source fails to fetch/infer, proceed with the remaining sources and re-normalize weights (never block the whole pipeline on one source).

---

## 8. Phased Execution Plan

| Phase | Goal | Key Tasks | Exit Criteria |
|---|---|---|---|
| **Phase 0 — Setup** | Repo, env, config scaffolding | Create repo structure above; `requirements.txt`; Docker base image; `config/*.yaml` stubs | `docker-compose up` starts an empty but running stack |
| **Phase 1 — Data Ingestion** | Pull real data from all sources | Implement `fetch_nwp.py`, `fetch_ensemble.py`, `fetch_reanalysis.py`, `fetch_observations.py`; validate against `config/sources.yaml` | Can fetch 1 day of data from all 4 sources for 1 test region, saved to `data/raw/` |
| **Phase 2 — Preprocessing** | Common grid + aligned lead times | Implement `regrid.py`, `normalize.py`, `align_timestamps.py` | All sources for the test day land on the same grid/lead-time index, unit-consistent |
| **Phase 3 — AI/ML Models** | Working inference from 3 model types | Implement/wire `lstm_model.py`, `transformer_model.py`, `gnn_model.py`, `inference.py` (use pretrained/lightweight versions first — full training is a stretch goal) | `run_all_models()` returns forecasts from all 3 models on test data |
| **Phase 4 — Skill Evaluation** | Historical skill scoring | Implement `metrics.py`, `rolling_skill.py`; backfill `skill_scores` table with ≥30 days of history (can use ERA5 as ground truth for backfill) | Rolling RMSE/CRPS/Brier queryable per source/region/lead-time |
| **Phase 5 — Adaptive Weighting** | Trained weighting model | Implement `feature_builder.py`, `train_weighting_model.py`, `predict_weights.py`; first training run | Weighting model outputs a valid weight distribution (sums to 1) for a given (region, lead_time, season, regime) |
| **Phase 6 — Fusion + Confidence** | Blended forecast output | Implement `fuse_forecasts.py`, `confidence_score.py` | Single blended `xr.Dataset` produced end-to-end from raw data pull to fused output |
| **Phase 7 — Extreme Flagging** | Alert layer | Implement `flag_extremes.py` with calibrated thresholds | Flags correctly trigger on known historical extreme-event dates (backtest) |
| **Phase 8 — API + Dashboard** | User-facing layer | Implement FastAPI endpoints; Streamlit dashboard | API returns forecast/weights/extremes for a test region; dashboard renders map + weight map + alerts |
| **Phase 9 — Monitoring** | Automated metric tracking | Configure Grafana; Dockerize each stage | Monitoring completes end-to-end without manual intervention |
| **Phase 10 — Validation & Demo Prep** | Prove skill improvement | Backtest blended forecast vs. each individual source over a held-out period; compute skill improvement %; prepare demo script/slides | Blended forecast shows measurable RMSE/CRPS improvement over best individual source on held-out data |

---

## 9. Evaluation Plan (for proving "Improved forecast skill")

1. Hold out a test period (e.g. last 30 days of available data) not used in weighting-model training.
2. For each variable (rainfall, temperature, wind) and each lead time, compute RMSE/CRPS for:
   - Each individual NWP source
   - Each individual AI/ML model
   - The blended (AI-FUSE) output
3. Report **% skill improvement** of blended forecast vs. the best-performing individual source, per region/season.
4. For extreme events specifically, report **Brier Skill Score** and a confusion matrix (hit/miss/false-alarm) for the flagging layer against historical IMD-recorded extreme events.

---

## 10. Configuration Files (starter templates)

**`config/regions.yaml`**
```yaml
regions:
  - id: north_india
    bbox: [68.0, 24.0, 80.0, 37.0]   # [min_lon, min_lat, max_lon, max_lat]
  - id: south_india
    bbox: [74.0, 8.0, 84.0, 18.0]
  - id: coastal_east
    bbox: [80.0, 8.0, 92.0, 22.0]
grid_resolution_deg: 0.25
lead_times_hours: [6, 12, 24, 48, 72, 120]
```

**`config/sources.yaml`**
```yaml
sources:
  nwp_gfs:
    provider: noaa_ncep
    fetch_method: herbie
  nwp_ecmwf:
    provider: ecmwf_open_data
    fetch_method: ecmwf-opendata
  ensemble_ecmwf_ens:
    provider: ecmwf_open_data
    fetch_method: ecmwf-opendata
  reanalysis_era5:
    provider: copernicus_cds
    fetch_method: cdsapi
```

**`config/model_config.yaml`**
```yaml
weighting_model:
  algorithm: lightgbm
  retrain_schedule: daily
  rolling_skill_window_days: 30
ai_models:
  lstm:
    lookback_steps: 24
  transformer:
    lookback_steps: 48
    num_heads: 4
  gnn:
    edge_definition: spatial_knn
    k_neighbors: 8
```

---

## 11. Testing Strategy

- **Unit tests** (`tests/`) for every module: ingestion (mock API responses), preprocessing (grid alignment correctness), metrics (known-value checks against hand-computed RMSE/CRPS), fusion (weights sum to 1, output shape matches input).
- **Integration test**: full pipeline run on a small fixed date range with cached/mocked data, asserting the DAG completes and produces a valid blended output.
- **Backtest validation**: as described in Section 9, run against a historical period with known outcomes.

---

## 12. Known Risks & Mitigations (carried from Feasibility slide)

| Risk | Mitigation |
|---|---|
| Heterogeneous data formats/resolutions | Common regridding/preprocessing layer (Section 6.2) |
| Latency mismatch between NWP release and AI inference | Pre-fetch and cache NWP releases; align pipeline schedule to model run times |
| Sparse ground-truth for rare extreme events | Skill-score smoothing + Bayesian shrinkage in extreme-event-sparse regions |
| Compute cost of frequent retraining | Scheduled (not per-request) retraining of the weighting model |

---

## 13. References

- IMD Forecast Verification: https://mausam.imd.gov.in/
- NCMRWF: https://www.ncmrwf.gov.in/
- NCMRWF Regional Data Sharing Portal (IMDAA/NGFS datasets): https://rds.ncmrwf.gov.in
- Copernicus CDS — ERA5 single levels dataset page: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
- ECMWF Open Data: https://www.ecmwf.int/en/forecasts/datasets/open-data
- ECMWF Open Data technical docs: https://confluence.ecmwf.int/display/DAC/ECMWF+open+data%3A+real-time+forecasts
- NOAA NCEP (GFS): https://www.ncep.noaa.gov/
- Copernicus Climate Data Store (ERA5): https://cds.climate.copernicus.eu/
- Krishnamurti, T.N. et al. (1999), "Improved Weather and Seasonal Climate Forecasts from Multimodel Superensemble," *Science* 285(5433): https://doi.org/10.1126/science.285.5433.1548 (open copy: https://eprints.iisc.ac.in/243/1/tnkrish.pdf)

---

*End of execution plan. Build in phase order; each phase should leave the repo in a runnable state before the next begins.*