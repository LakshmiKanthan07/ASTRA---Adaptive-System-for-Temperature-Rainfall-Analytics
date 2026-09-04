# ASTRA — Adaptive System for Temperature & Rainfall Analytics
*(Also known as AI-FUSE: Adaptive Intelligent Forecast Unification & Skill Engine)*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](Dockerfile)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io)

**ASTRA** is an enterprise-grade, hybrid **AI–NWP (Numerical Weather Prediction) forecast blending framework** developed for SIH 2026. ASTRA dynamically computes adaptive weights for heterogeneous weather models (such as ECMWF HRES/IFS, GFS, WRF, and Deep-Learning Weather Prediction models like GraphCast/FourCastNet) based on historical skill score, forecast lead time, geographic region, season, and active weather regimes.

---

## 🌟 Key Features

- 🛰️ **Multi-Source Data Ingestion & Regridding**: Ingests GRIB2/NetCDF files from ECMWF, GFS, and AI models; standardizes onto a unified spatial grid ($0.25^\circ \times 0.25^\circ$).
- 📊 **Skill Evaluation Engine**: Computes real-time and rolling skill metrics including RMSE, MAE, CRPS (Continuous Ranked Probability Score), Brier Score, and Anomaly Correlation Coefficient (ACC).
- ⚖️ **Adaptive Weighting Engine**: Utilizes gradient-boosted trees (XGBoost / LightGBM) to dynamically assign spatial-temporal weights based on lead time, terrain, regime, and historical error.
- 🌪️ **Extreme Event Guidance**: Specialized alert pipeline for heavy precipitation, heatwaves, and high-wind events.
- 🚀 **REST API & Interactive Dashboard**: Built with FastAPI for high-performance inference and Streamlit for rich spatial-temporal geospatial maps and model reliability analytics.
- 🐳 **Containerized & Orchestrated**: Out-of-the-box support for Docker, Docker Compose, and Apache Airflow pipelines.

---

## 🏗️ Architecture

```
                    ┌────────────────────┐
                    │   DATA INGESTION   │
                    │  (NWP, ENS, AI/ML, │
                    │   Obs, Reanalysis) │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  PREPROCESSING &   │
                    │  COMMON REGRIDDING │
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
                    │  SKILL EVALUATION  │
                    │  (RMSE, CRPS, ACC) │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │ ADAPTIVE WEIGHTING │
                    │ (XGBoost/LightGBM) │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  FORECAST FUSION   │
                    │      ENGINE        │
                    └─────────┬──────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
      ┌───────▼──────┐ ┌──────▼───────┐ ┌─────▼────────┐
      │ Blended      │ │ Model Weight │ │ Extreme Event│
      │ Forecast     │ │ Maps         │ │ Signals      │
      └───────┬──────┘ └──────┬───────┘ └─────┬────────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  DASHBOARD + API   │
                    │ (Streamlit/FastAPI)│
                    └────────────────────┘
```

---

## 📁 Repository Structure

```
sih2026/
├── config/                  # Model & regional configuration files
│   ├── model_config.yaml    # Hyperparameters & model sources definition
│   ├── regions.yaml         # Region masks and coordinate bounding boxes
│   └── sources.yaml         # NWP and AI dataset source endpoints
├── src/                     # Core Python package
│   ├── api/                 # FastAPI web REST endpoints (`main.py`)
│   ├── dashboard/           # Streamlit analytics application (`app.py`)
│   ├── ingestion/           # Data fetchers for ECMWF/GFS & GRIB2 parsers
│   ├── preprocessing/       # Xarray regridding & normalization utilities
│   ├── skill_evaluation/    # RMSE, CRPS, Brier score calculation modules
│   ├── weighting/           # XGBoost dynamic weight predictor
│   ├── fusion/              # Blending engine & ensemble combination
│   ├── extremes/            # Extreme weather anomaly detection
│   ├── models/              # PyTorch weather deep-learning architectures
│   └── utils/               # Geospatial & temporal helper functions
├── tests/                   # Unit and integration test suite
├── airflow/                 # Apache Airflow DAGs for operational runs
├── concept.md               # Detailed technical specification
├── download_hres.py         # ECMWF open data automated download script
├── docker-compose.yml       # Multi-container orchestration (API + Dashboard)
├── Dockerfile               # Production container image setup
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Project metadata & build tool configuration
└── .gitignore               # Git ignore rules for datasets & binaries
```

---

## ⚡ Quick Start

### 1. Prerequisites

- Python 3.11+
- Git & Docker (optional for containerized deployment)

### 2. Local Setup

```bash
# Clone the repository
git clone https://github.com/LakshmiKanthan07/ASTRA---Adaptive-System-for-Temperature-Rainfall-Analytics.git
cd ASTRA---Adaptive-System-for-Temperature-Rainfall-Analytics

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Fetch Sample ECMWF Data

```bash
python download_hres.py
```

### 4. Run REST API

```bash
uvicorn src.api.main:app --reload --port 8000
```
*API docs available at:* `http://localhost:8000/docs`

### 5. Launch Interactive Dashboard

```bash
streamlit run src/dashboard/app.py
```

---

## 🐳 Docker Deployment

To launch both the REST API and Streamlit Dashboard in containerized mode:

```bash
docker-compose up --build -d
```

- **Dashboard**: `http://localhost:8501`
- **FastAPI Endpoint**: `http://localhost:8000`

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any features, bug fixes, or enhancements.
