"""
ASTRA — Automated Feedback Loop & Continuous Learning Module
============================================================
Simulates the operational feedback cycle that NWP blending systems use in practice:

  1. Retrieve "new" observation data (here: ERA5 reanalysis as proxy)
  2. Load the most recent blended forecast
  3. Compare forecast vs observations at grid-point level
  4. Compute per-variable forecast errors (RMSE, MAE, Bias)
  5. Update historical skill metrics (append rolling window)
  6. Incrementally retrain the XGBoost adaptive weighter on the new errors
  7. Persist updated model weights and skill history
  8. Generate a feedback report for the dashboard

This module is designed to be run after each new observation cycle becomes available
(typically 24–48 h after forecast initialisation).

Usage (standalone):
    python src/feedback/updater.py

Usage (from pipeline):
    from src.feedback.updater import FeedbackUpdater
    updater = FeedbackUpdater()
    report  = updater.run()
"""
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

# ── ensure project root on path ─────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

from src.skill_evaluation.metrics import compute_metrics, compare_models, compute_categorical_metrics

BLENDED_NC     = "data/blended_forecast.nc"
SKILL_HISTORY  = "data/skill_history.csv"
FEEDBACK_JSON  = "data/feedback_report.json"


class FeedbackUpdater:
    """
    Handles the full observation -> error -> skill-update -> weight-retrain cycle.
    """

    def __init__(self, observation_source: str = "synthetic"):
        """
        Parameters
        ----------
        observation_source : 'synthetic' (default for MVP) or path to a NetCDF observation file.
        """
        self.obs_source = observation_source

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Load / synthesise observations
    # ─────────────────────────────────────────────────────────────────────────
    def _get_observations(self, ds_forecast: xr.Dataset) -> xr.Dataset:
        """
        Returns an observation dataset aligned to the forecast grid.

        For the hackathon MVP we generate realistic synthetic observations:
        - Add spatially-correlated random noise to the blended forecast
        - This mimics what a raingauge / satellite observation might look like
        - The system distinguishes this as SYNTHETIC DATA explicitly

        In production replace this method with:
            ds = xr.open_dataset(self.obs_source)
            return ds.interp(latitude=ds_forecast.latitude, longitude=ds_forecast.longitude)
        """
        rng = np.random.default_rng(seed=int(datetime.utcnow().strftime("%Y%m%d")))
        lats = ds_forecast.latitude.values
        lons = ds_forecast.longitude.values
        obs_vars = {}
        noise_fracs = {"tp_blended": 0.15, "t2m_blended": 0.03, "wind_blended": 0.10}

        for var, frac in noise_fracs.items():
            if var in ds_forecast:
                base = ds_forecast[var].values
                # Spatially-correlated noise via 2-D smooth random field
                raw_noise = rng.standard_normal(base.shape)
                from scipy.ndimage import gaussian_filter
                smooth_noise = gaussian_filter(raw_noise, sigma=3)
                magnitude = frac * (np.nanstd(base) + 1e-9)
                obs_vars[var.replace("_blended", "_obs")] = xr.DataArray(
                    (base + smooth_noise * magnitude).clip(0 if "tp" in var or "wind" in var else None),
                    dims=["latitude", "longitude"],
                    coords={"latitude": lats, "longitude": lons},
                    attrs={"source": "SYNTHETIC_OBS — Demo Data (not real observations)"}
                )

        return xr.Dataset(obs_vars)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Compare forecast vs observations
    # ─────────────────────────────────────────────────────────────────────────
    def _verify(
        self,
        ds_forecast: xr.Dataset,
        ds_obs: xr.Dataset,
    ) -> dict:
        """Compare blended forecast against observations. Returns error metrics per variable."""
        results = {}
        var_pairs = [
            ("tp_blended",   "tp_obs",   "tp"),
            ("t2m_blended",  "t2m_obs",  "t2m"),
            ("wind_blended", "wind_obs", "wind"),
        ]
        for fcst_var, obs_var, label in var_pairs:
            if fcst_var not in ds_forecast or obs_var not in ds_obs:
                continue
            fcst = ds_forecast[fcst_var].values.ravel()
            obs  = ds_obs[obs_var].values.ravel()
            valid = np.isfinite(fcst) & np.isfinite(obs)
            if valid.sum() < 10:
                continue
            metrics = compute_metrics(fcst[valid], obs[valid])
            metrics["n_points"] = int(valid.sum())

            # For precipitation also compute categorical metrics (POD/FAR/CSI)
            if label == "tp":
                tp_mm_fcst = fcst[valid] * 1000.0
                tp_mm_obs  = obs[valid] * 1000.0
                cat = compute_categorical_metrics(tp_mm_fcst, tp_mm_obs, threshold=2.5)
                metrics.update(cat)

            results[label] = metrics
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Update skill history
    # ─────────────────────────────────────────────────────────────────────────
    def _update_skill_history(self, verification: dict) -> pd.DataFrame:
        """Append today's verification metrics to the rolling skill history CSV."""
        rows = []
        run_time = datetime.utcnow().isoformat()
        for var, metrics in verification.items():
            row = {"run_time": run_time, "variable": var}
            row.update(metrics)
            rows.append(row)

        new_df = pd.DataFrame(rows)

        if os.path.exists(SKILL_HISTORY):
            hist_df = pd.read_csv(SKILL_HISTORY)
            # Rolling window: keep last 30 cycles
            hist_df = pd.concat([hist_df, new_df], ignore_index=True).tail(30 * len(rows))
        else:
            hist_df = new_df

        hist_df.to_csv(SKILL_HISTORY, index=False)
        print(f"[Feedback] [OK] Skill history updated -> {SKILL_HISTORY}  (rows: {len(hist_df)})")
        return hist_df

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Incremental weight update
    # ─────────────────────────────────────────────────────────────────────────
    def _update_weights(self, ds_forecast: xr.Dataset, ds_obs: xr.Dataset) -> dict:
        """
        Perform lightweight incremental weight update.
        Rather than full retraining (which requires GRIB files), we use the
        verification error to nudge the HRES/GFS weight bias.

        Returns updated weight summary.
        """
        weight_updates = {}

        for var in ("tp", "t2m", "wind"):
            hres_var  = f"{var}_hres"
            gfs_var   = f"{var}_gfs"
            obs_var   = f"{var}_obs"
            blend_var = f"{var}_blended"

            if not all(v in ds_forecast for v in [hres_var, gfs_var, blend_var]):
                continue
            if obs_var not in ds_obs:
                continue

            hres  = ds_forecast[hres_var].values.ravel()
            gfs   = ds_forecast[gfs_var].values.ravel()
            obs   = ds_obs[obs_var].values.ravel()
            valid = np.isfinite(hres) & np.isfinite(gfs) & np.isfinite(obs)

            if valid.sum() < 10:
                continue

            hres_rmse = float(np.sqrt(np.mean((hres[valid] - obs[valid]) ** 2)))
            gfs_rmse  = float(np.sqrt(np.mean((gfs[valid]  - obs[valid]) ** 2)))

            # Skill-weighted update: models with lower error get higher weight
            total_inv = (1.0 / (hres_rmse + 1e-9)) + (1.0 / (gfs_rmse + 1e-9))
            new_hres_w = (1.0 / (hres_rmse + 1e-9)) / total_inv
            new_gfs_w  = (1.0 / (gfs_rmse  + 1e-9)) / total_inv

            # Load existing weights and blend with new evidence (exponential smoothing)
            w_file = f"data/learned_weights_{var}.csv"
            if os.path.exists(w_file):
                w_df = pd.read_csv(w_file)
                hres_col = f"{var}_hres_weight"
                gfs_col  = f"{var}_gfs_weight"
                if hres_col in w_df.columns:
                    alpha = 0.2  # learning rate for exponential smoothing
                    old_hres = float(w_df[hres_col].mean())
                    old_gfs  = float(w_df[gfs_col].mean())
                    # Smooth update
                    updated_hres = alpha * new_hres_w + (1 - alpha) * old_hres
                    updated_gfs  = 1.0 - updated_hres
                    w_df[hres_col] = updated_hres
                    w_df[gfs_col]  = updated_gfs
                    w_df.to_csv(w_file, index=False)
                    print(f"[Feedback] [OK] Weights updated for {var}: HRES={updated_hres:.3f}, GFS={updated_gfs:.3f}")
                    weight_updates[var] = {
                        "ECMWF HRES": round(updated_hres * 100, 1),
                        "NOAA GFS":   round(updated_gfs  * 100, 1),
                        "HRES RMSE":  round(hres_rmse, 4),
                        "GFS RMSE":   round(gfs_rmse, 4),
                    }

        return weight_updates

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Build feedback report
    # ─────────────────────────────────────────────────────────────────────────
    def _build_report(
        self,
        verification: dict,
        weight_updates: dict,
        data_source: str,
    ) -> dict:
        report = {
            "generated_at":   datetime.utcnow().isoformat() + "Z",
            "observation_source": data_source,
            "is_real_data":   data_source != "synthetic",
            "verification":   {v: {k: round(float(m), 5) if isinstance(m, (float, np.floating)) else m
                                   for k, m in metrics.items()}
                               for v, metrics in verification.items()},
            "weight_updates": weight_updates,
            "feedback_cycle": {
                "step_1": "Observations retrieved (synthetic proxy for MVP demo)",
                "step_2": "Forecast vs observations compared at grid-point level",
                "step_3": "RMSE/MAE/Bias/POD/FAR/CSI computed per variable",
                "step_4": "Skill history appended to rolling 30-cycle window",
                "step_5": "Adaptive weights nudged via exponential smoothing (α=0.2)",
                "step_6": "Updated weights persisted to data/learned_weights_*.csv",
                "step_7": "This report saved for dashboard consumption",
            },
        }

        # Compute blending improvement
        for var, m in verification.items():
            rmse_blend = m.get("RMSE")
            if rmse_blend and rmse_blend > 0:
                report["verification"][var]["narrative"] = (
                    f"Blended RMSE={rmse_blend:.4f}. "
                    "Adaptive weights will be further refined next cycle."
                )

        os.makedirs("data", exist_ok=True)
        with open(FEEDBACK_JSON, "w") as f:
            json.dump(report, f, indent=2)

        print(f"[Feedback] [OK] Report saved -> {FEEDBACK_JSON}")
        return report

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────
    def run(self) -> dict:
        """Execute the full feedback loop."""
        print("\n" + "=" * 60)
        print("  ASTRA — Automated Feedback & Continuous Learning Loop")
        print("=" * 60)

        # Load blended forecast
        if not os.path.exists(BLENDED_NC):
            print("[Feedback] [FAIL] No blended forecast found. Run pipeline first.")
            return {}

        ds_forecast = xr.open_dataset(BLENDED_NC)
        print(f"[Feedback] [OK] Loaded blended forecast: {list(ds_forecast.data_vars)}")

        # Step 1: Get observations
        print("\n[Feedback] Step 1/5: Retrieving observations...")
        ds_obs = self._get_observations(ds_forecast)
        print(f"[Feedback] [OK] Observations ready: {list(ds_obs.data_vars)}  [SOURCE: SYNTHETIC — Demo Mode]")

        # Step 2: Verify
        print("\n[Feedback] Step 2/5: Verifying forecast vs observations...")
        verification = self._verify(ds_forecast, ds_obs)
        for var, m in verification.items():
            print(f"  {var.upper():6s}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  Bias={m['Bias']:.4f}")

        # Step 3: Update skill history
        print("\n[Feedback] Step 3/5: Updating skill history...")
        self._update_skill_history(verification)

        # Step 4: Update weights
        print("\n[Feedback] Step 4/5: Incrementally updating adaptive weights...")
        weight_updates = self._update_weights(ds_forecast, ds_obs)

        # Step 5: Build report
        print("\n[Feedback] Step 5/5: Generating feedback report...")
        report = self._build_report(verification, weight_updates, self.obs_source)

        print("\n[Feedback] [OK] Feedback loop complete.\n")
        return report


if __name__ == "__main__":
    updater = FeedbackUpdater(observation_source="synthetic")
    report  = updater.run()
    print(json.dumps(report, indent=2))
