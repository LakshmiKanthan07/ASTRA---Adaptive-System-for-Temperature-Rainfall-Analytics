"""
ASTRA — Master Pipeline v2.1
Runs: ingestion → multi-variable weight-target training → blending → skill evaluation → extremes → export

Usage:
    python src/run_pipeline.py
"""
import sys, os, warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import xarray as xr
from src.ingestion.grib_loader import GribLoader
from src.weighting.model import AdaptiveWeighter
from src.skill_evaluation.metrics import compare_models
from src.extremes.guidance import ExtremeWeatherDetector

HRES_FILE = "hres.grib2"
GFS_FILE  = "gfs.t12z.pgrb2.0p25.f006"
ENS_FILE  = "ens.grib2"
ERA5_FILE = "f986deebfac018041762bbe31b9959e0.grib"
OUTPUT_NC = "data/blended_forecast.nc"

# Variables to blend
TARGETS = ["tp", "t2m", "wind"]

def main():
    print("=" * 60)
    print("  ASTRA — Adaptive Forecast Blending Pipeline v2.1")
    print("=" * 60)

    os.makedirs("data", exist_ok=True)

    # ── 1. Ingestion ────────────────────────────────────────────
    loader   = GribLoader()
    ds_hres  = loader.load_hres(HRES_FILE)
    ds_gfs   = loader.load_gfs(GFS_FILE)
    ds_truth = loader.load_ground_truth(ERA5_FILE)

    ds_spread = None
    try:
        ds_spread = loader.load_ens_spread(ENS_FILE)
    except Exception as exc:
        print(f"[Pipeline] ENS spread skipped: {exc}")

    print("\n[Pipeline] ✓ All datasets loaded.")

    # We will accumulate all blended variables into one dataset
    ds_final = None

    # ── 2 & 3. Train & Inference (Per Variable) ──────────────────
    for var in TARGETS:
        print(f"\n--- Processing variable: {var.upper()} ---")
        ds_blended = None
        if var in ds_hres and var in ds_gfs and f"{var}_truth" in ds_truth:
            weighter = AdaptiveWeighter(target_var=var)
            X, y = weighter.prepare_training_data(ds_hres, ds_gfs, ds_truth, ds_spread)
            weighter.train(X, y)
            ds_blended = weighter.predict(ds_hres, ds_gfs, ds_spread)
        else:
            print(f"[Pipeline] Warning: {var} missing from some sources. Using fallback.")
            ds_blended = xr.Dataset()
            if var in ds_hres:
                ds_blended[f"{var}_blended"] = ds_hres[var]
            elif var in ds_gfs:
                ds_blended[f"{var}_blended"] = ds_gfs[var]
            else:
                print(f"[Pipeline] Skipping {var}: completely missing.")
                continue
        
        # Merge into final dataset
        if ds_final is None:
            ds_final = ds_blended
        else:
            ds_final = xr.merge([ds_final, ds_blended])
            
        # Skill Evaluation for this variable
        if var in ds_hres and var in ds_gfs and f"{var}_truth" in ds_truth:
            try:
                df = weighter._train_df.copy()
                X_eval = df[weighter.FEATURE_NAMES].fillna(0.0)
                w_eval = weighter.model.predict(X_eval).clip(0, 1)
                
                blended_col = f"{var}_blended"
                df[blended_col] = w_eval * df[f"{var}_hres"] + (1 - w_eval) * df[f"{var}_gfs"]
                
                if var == "tp":
                    df[blended_col] = df[blended_col].clip(
                        lower=0, upper=np.maximum(df[f"{var}_hres"], df[f"{var}_gfs"])
                    )
                
                valid = df[[f"{var}_truth", f"{var}_hres", f"{var}_gfs", blended_col]].notnull().all(axis=1)
                df_v = df[valid]
                
                if len(df_v) > 10:
                    skill_table = compare_models(
                        {"ECMWF HRES":  df_v[f"{var}_hres"].values,
                         "NOAA GFS":    df_v[f"{var}_gfs"].values,
                         "ASTRA Blend": df_v[blended_col].values},
                        df_v[f"{var}_truth"].values,
                    )
                    print(f"\n[Pipeline] ✓ Skill Evaluation ({var.upper()}):")
                    print(skill_table.to_string())
                    skill_table.to_csv(f"data/skill_scores_{var}.csv")
                else:
                    print(f"[Pipeline] Warning: only {len(df_v)} valid points — skipping skill scores for {var}")
                    
                # Save per-grid learned weights
                if f"{var}_hres_weight" in ds_blended:
                    import pandas as pd
                    w_df = ds_blended[[f"{var}_hres_weight", f"{var}_gfs_weight"]].to_dataframe().reset_index()
                    w_df.to_csv(f"data/learned_weights_{var}.csv", index=False)
                    
            except Exception as e:
                import traceback
                print(f"[Pipeline] Skill eval error for {var}: {e}")
                traceback.print_exc()

    print("\n[Pipeline] ✓ All variables processed.")

    # ── 4. Extreme Weather Guidance ─────────────────────────────
    if ds_final is not None:
        print("\n[Pipeline] Running Extreme Weather Detection...")
        detector = ExtremeWeatherDetector()
        detector.detect(ds_final)
        
    # ── 5. Export (atomic replace to bypass Windows file locks) ─
    if ds_final is not None:
        TEMP_NC = OUTPUT_NC + ".tmp"
        if os.path.exists(TEMP_NC):
            os.remove(TEMP_NC)
        ds_final.to_netcdf(TEMP_NC, engine="scipy")
        
        try:
            os.replace(TEMP_NC, OUTPUT_NC)
            print(f"\n[Pipeline] ✓ Blended forecast saved → {OUTPUT_NC}")
        except Exception as e:
            print(f"\n[Pipeline] Warning: Could not replace {OUTPUT_NC} because it is locked: {e}")
            print(f"             Forecast is saved at {TEMP_NC} instead.")

    print("\n" + "=" * 60)
    print("  Pipeline v2.1 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
