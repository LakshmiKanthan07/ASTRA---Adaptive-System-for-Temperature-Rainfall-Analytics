# ASTRA: Adaptive System for Temperature & Rainfall Analytics

**Project Codename:** ASTRA (Adaptive Intelligent Forecast Unification & Skill Engine)
**Theme:** Disaster Management
**Target Hackathon:** Smart India Hackathon (SIH) 2026

---

## 1. Introduction & The Idea

**ASTRA** is an intelligent, automated weather forecasting framework that blends multiple Numerical Weather Prediction (NWP) models and AI/ML predictions into a single, high-accuracy forecast. Instead of relying on a single weather model, ASTRA dynamically evaluates the historical performance ("skill") of various models and assigns adaptive weights to them based on current conditions, region, season, and lead time.

---

## 2. The Problem Statement & Current Gap

**The Problem:** 
Meteorologists and disaster management agencies have access to dozens of weather models (GFS, ECMWF, NCMRWF, etc.). However, **no single model is perfectly accurate all the time**. 
- Model A might be excellent at predicting monsoon rainfall in Kerala but terrible at predicting winter fog in Delhi.
- Model B might be great at short-term forecasts (24h) but lose accuracy at long-term predictions (120h).

**The Current Gap:**
Currently, forecasters have to manually look at multiple models, compare them, and use human intuition to decide which one to trust. This manual consensus process is slow, subjective, and prone to human error, especially during rapidly developing extreme weather events (e.g., flash floods or heatwaves). 

---

## 3. Our Proposed Solution

ASTRA completely automates and optimizes the consensus process using Machine Learning.

Our solution ingests data from major global and regional weather models, evaluates how accurate they have been over the past 30 days for specific regions, and uses an AI weighting model (Gradient Boosted Trees - XGBoost/LightGBM) to blend them. 

If ECMWF has been highly accurate for rainfall in South India this week, ASTRA automatically gives it a higher weight. If GFS is better for wind speed, GFS gets the higher weight. The result is a **"Super-Forecast"** that is statistically more accurate than any individual model.

---

## 4. Key Features & Unique Selling Proposition (USP)

- **Multi-Model Intelligence (The USP):** ASTRA does not reinvent the wheel by making a new physics model. It acts as an intelligent layer *on top* of existing models, extracting the best parts of each.
- **Context-Aware Adaptation:** Weights are not static. They change dynamically based on Region, Lead Time, Season, and current Weather Regime.
- **Automated Extreme Weather Flagging:** The system automatically issues alerts for heavy rainfall, heatwaves, and high winds based on calibrated thresholds.
- **Explainable AI (XAI):** Unlike black-box AI models, ASTRA provides "Confidence Scores" and "Weight Maps" so meteorologists can see *exactly* which underlying model is driving the current forecast and why.
- **Forecaster Dashboard:** A ready-to-use Streamlit interactive web dashboard for visualizing the blended maps and active alerts.

---

## 5. Datasets Used

ASTRA utilizes a heterogeneous mix of reliable, official, and open-access data providers:

1. **GFS (NOAA NCEP):** Global Forecast System for deterministic NWP input.
2. **ECMWF Open Data:** High-resolution deterministic (HRES) and Ensemble (ENS) forecasts.
3. **NCMRWF (IMDAA/NGFS):** Regional high-resolution reanalysis models acting as the Indian baseline.
4. **ERA5 (Copernicus Climate Data Store):** Historical ground truth data used for training the weighting model and evaluating past skill.
5. **IMD (India Meteorological Department):** Used for verification labels and extreme-event threshold definitions.

---

## 6. Technical Approach & Architecture

The ASTRA pipeline operates in an automated, scheduled loop:

1. **Ingestion & Preprocessing:** GRIB/NetCDF files from GFS, ECMWF, and historical ground truth are downloaded. They are spatially regridded to a common 0.25° resolution map of India.
2. **Skill Evaluation:** The system continuously compares yesterday's forecasts against today's actual ground truth to compute rolling error metrics (RMSE, CRPS, Brier Score).
3. **Adaptive Weighting:** An XGBoost machine learning model takes the rolling skill scores, time of year, and geographic coordinates as inputs to output optimal percentage weights for each source.
4. **Forecast Fusion:** The raw NWP models are multiplied by their respective AI-generated weights and summed together to create the final blended forecast.
5. **Deployment:** The processed data is served via a **FastAPI REST API** and visualized on a **Streamlit Dashboard** deployed on cloud infrastructure (e.g., Render/AWS).

---

## 7. Feasibility & Viability

**Technical Feasibility:** 
Highly feasible. ASTRA is built entirely on proven open-source technologies (Python, PyTorch, XGBoost, xarray). The pipeline relies on publicly available, standard meteorological data formats (GRIB2/NetCDF4).

**Viability & Scalability:**
- **Modular Pipeline:** The system is containerized (Docker) and designed in micro-stages. If a new weather model is invented tomorrow, it can be plugged into the ingestion layer without breaking the system.
- **Cost-Effective:** Because it relies on open data and lightweight tree-based ML for the weighting (rather than training massive deep learning models from scratch), the compute costs for operational deployment are extremely low.

**Challenges & Mitigation:**
- *Challenge:* Handling massive GRIB data files.
- *Mitigation:* We run the heavy data pipeline securely on backend servers and only expose lightweight, compressed processed netCDF/JSON outputs to the frontend API.

---

## 8. Impact & Benefits

**Operational Impact:**
- Drastically reduces the manual workload of IMD/NCMRWF forecasters during critical weather events.
- Provides a mathematically sound, unbiased consensus forecast.

**Social Impact:**
- Improved lead time and accuracy for extreme weather warnings (cyclones, flash floods) directly saves lives and allows disaster management forces (NDRF) to mobilize efficiently.

**Economic & Environmental Impact:**
- **Agriculture:** Farmers receive more accurate localized rainfall predictions, reducing crop loss.
- **Energy:** Power-grid operators can optimize solar and wind dispatch planning based on highly accurate localized cloud-cover and wind-speed blends.
- **Aviation & Logistics:** Reduced disruptions due to unexpected fog or storms.

---

## 9. Future Scope

1. **Deep Learning Integration:** Incorporating Graph Neural Networks (GNNs) or Transformers to not just weight the models, but to physically correct spatial biases (e.g., fixing a model's tendency to under-predict rain over mountainous terrain).
2. **Hyper-Local Downscaling:** Extending the resolution from 0.25° (~25km) down to 3km using generative AI (Diffusion models) for village-level forecasting.
3. **Mobile App:** Releasing a lightweight mobile application for direct consumer/farmer access to the ASTRA blended forecast.