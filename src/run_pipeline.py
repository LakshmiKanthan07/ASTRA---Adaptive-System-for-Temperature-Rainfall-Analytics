"""
ASTRA — Master Pipeline Script
Runs the full ingestion → training → blending → evaluation → export workflow.

Usage:
    python src/run_pipeline.py
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from src.ingestion.grib_loader import GribLoader
from src.weighting.model import AdaptiveWeighter
from src.skill_evaluation.metrics import compare_models

# ---------------------------------------------------------------------------
# Dataset paths  (relative to project root — adjust if files live elsewhere)
# ---------------------------------------------------------------------------
HRES_FILE  = "hres.grib2"
GFS_FILE   = "gfs.t12z.pgrb2.0p25.f006"
ENS_FILE   = "ens.grib2"
ERA5_FILE  = "f986deebfac018041762bbe31b9959e0.grib"
OUTPUT_NC  = "data/blended_forecast.nc"


def main():
    print("=" * 60)
    print("  ASTRA — Adaptive Forecast Blending Pipeline")
    print("=" * 60)

    os.makedirs("data", exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Ingestion & Regridding
    # -----------------------------------------------------------------------
    loader = GribLoader()

    ds_hres  = loader.load_hres(HRES_FILE)
    ds_gfs   = loader.load_gfs(GFS_FILE)
    ds_truth = loader.load_ground_truth(ERA5_FILE)

    # ENS spread — non-fatal if it fails
    ds_spread = None
    try:
        ds_spread = loader.load_ens_spread(ENS_FILE)
    except Exception as exc:
        print(f"[Pipeline] ENS spread skipped: {exc}")

    print("\n[Pipeline] ✓ All datasets loaded and regridded.")

    # -----------------------------------------------------------------------
    # 2. Adaptive Weighting — Train XGBoost
    # -----------------------------------------------------------------------
    weighter = AdaptiveWeighter()
    X, y = weighter.prepare_training_data(ds_hres, ds_gfs, ds_truth, ds_spread)
    weighter.train(X, y)

    # -----------------------------------------------------------------------
    # 3. Blended Forecast Inference
    # -----------------------------------------------------------------------
    ds_blended = weighter.predict(ds_hres, ds_gfs, ds_spread)
    print("\n[Pipeline] ✓ Blended forecast generated.")
    print(ds_blended)

    # -----------------------------------------------------------------------
    # 4. Skill Evaluation — align all grids to blended output grid
    # -----------------------------------------------------------------------
    import numpy as _np
    blended_lats = ds_blended.latitude
    blended_lons = ds_blended.longitude

    # -----------------------------------------------------------------------
    # 5. Export first (delete stale file to avoid file-lock)
    # -----------------------------------------------------------------------
    if os.path.exists(OUTPUT_NC):
        os.remove(OUTPUT_NC)
    ds_blended.to_netcdf(OUTPUT_NC, engine="scipy")
    print(f"\n[Pipeline] ✓ Blended forecast saved → {OUTPUT_NC}")

    # Now compute skill metrics (non-fatal)
    try:
        truth_grid = ds_truth["tp_truth"].interp(
            latitude=blended_lats, longitude=blended_lons, method="nearest"
        ).values.ravel()
        hres_grid = ds_hres["tp"].interp(
            latitude=blended_lats, longitude=blended_lons, method="linear"
        ).values.ravel()
        gfs_grid = ds_gfs["tp"].interp(
            latitude=blended_lats, longitude=blended_lons, method="linear"
        ).values.ravel()
        blended_grid = ds_blended["tp_blended"].values.ravel()

        # All arrays must be same length — truncate to min length
        n = min(len(truth_grid), len(hres_grid), len(gfs_grid), len(blended_grid))
        valid_mask = (
            _np.isfinite(truth_grid[:n]) & _np.isfinite(hres_grid[:n]) &
            _np.isfinite(gfs_grid[:n])   & _np.isfinite(blended_grid[:n])
        )
        skill_table = compare_models(
            {
                "ECMWF HRES":  hres_grid[:n][valid_mask],
                "NOAA GFS":    gfs_grid[:n][valid_mask],
                "ASTRA Blend": blended_grid[:n][valid_mask],
            },
            truth_grid[:n][valid_mask],
        )
        print("\n[Pipeline] ✓ Skill Evaluation:")
        print(skill_table.to_string())
        skill_table.to_csv("data/skill_scores.csv")
        print("[Pipeline] ✓ Skill scores saved  → data/skill_scores.csv")
    except Exception as e:
        print(f"[Pipeline] Warning: Skill evaluation skipped: {e}")


    print("\n" + "=" * 60)
    print("  Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
