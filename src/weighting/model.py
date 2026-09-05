"""
ASTRA — Adaptive Weighting Model v2.1
────────────────────────────────────
Key improvements over v1/v2:
  1. CORRECT TARGET: instead of predicting truth directly, we predict the
     optimal per-grid-point HRES weight  w ∈ [0,1].
  2. MULTI-VARIABLE: Supports dynamic target variables (tp, t2m, wind).
  3. TIME FEATURES: day_of_year (season proxy) and lead_time.
"""
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
import os


class AdaptiveWeighter:
    """
    Learns spatially-varying HRES blending weights using gradient-boosted trees.
    Supports generic variables (precipitation, temperature, wind).
    """

    FEATURE_NAMES = [
        # Geographic
        "latitude", "longitude",
        "lat_sin", "lat_cos", "lon_sin", "lon_cos",
        # Time / Season
        "day_of_year", "lead_time",
        # Raw model predictions (might be log-scaled for precip)
        "val_hres", "val_gfs",
        # Interaction / meteorological
        "val_spread",
        "val_model_mean",
        "frac_bias",
        "agree_flag",
        # Spatial context
        "dist_coast_proxy",
    ]

    def __init__(
        self,
        target_var: str = "tp",
        n_estimators: int = 400,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0.1,
        min_child_weight: int = 5,
        reg_lambda: float = 1.5,
    ):
        self.target_var = target_var
        self.model_file = f"data/xgb_adaptive_weighter_{target_var}.json"
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            objective="reg:squarederror",
            eval_metric="rmse",
            verbosity=0,
            tree_method="hist",
            early_stopping_rounds=25,
        )
        self.trained: bool = False

    # ──────────────────────────────────────────────────────────────────────
    # Feature engineering
    # ──────────────────────────────────────────────────────────────────────

    def _make_features(
        self,
        df_hres: pd.DataFrame,
        df_gfs: pd.DataFrame,
        df_spread: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Engineer the full feature matrix from raw NWP dataframes."""
        
        # We assume the columns are named according to target_var
        df = pd.merge(
            df_hres[["latitude", "longitude", self.target_var]].rename(columns={self.target_var: "val_hres"}),
            df_gfs[["latitude", "longitude", self.target_var]].rename(columns={self.target_var: "val_gfs"}),
            on=["latitude", "longitude"],
        )

        # Merge ensemble spread if available
        spread_var = f"{self.target_var}_spread"
        if df_spread is not None and spread_var in df_spread.columns:
            df = pd.merge(df, df_spread[["latitude", "longitude", spread_var]],
                          on=["latitude", "longitude"], how="left")
            df["val_spread"] = df[spread_var].fillna(0.0)
        else:
            df["val_spread"] = 0.0

        # Optional Time Features (if valid_time or time exists)
        df["day_of_year"] = 1.0  # Default fallback
        df["lead_time"] = 6.0    # Default fallback (+6h)
        if "time" in df_hres.columns:
            try:
                times = pd.to_datetime(df_hres["time"])
                df["day_of_year"] = times.dt.dayofyear
                # If we had an init time, we'd calculate lead time. For now, fallback to 6
            except:
                pass
        elif "valid_time" in df_hres.columns:
            try:
                times = pd.to_datetime(df_hres["valid_time"])
                df["day_of_year"] = times.dt.dayofyear
            except:
                pass

        eps = 1e-7
        # For precipitation, log scaling helps. For temperature and wind, we use raw.
        if self.target_var == "tp":
            df["val_hres_feat"] = np.log1p(df["val_hres"].clip(0))
            df["val_gfs_feat"]  = np.log1p(df["val_gfs"].clip(0))
            df["val_spread"]    = np.log1p(df["val_spread"].clip(0))
        else:
            df["val_hres_feat"] = df["val_hres"]
            df["val_gfs_feat"]  = df["val_gfs"]
            df["val_spread"]    = df["val_spread"]

        df["val_model_mean"] = (df["val_hres_feat"] + df["val_gfs_feat"]) / 2.0

        # ── Interaction features ──
        denom = np.abs(df["val_hres"]) + np.abs(df["val_gfs"]) + eps
        df["frac_bias"] = (df["val_hres"] - df["val_gfs"]) / denom
        max_val = np.maximum(np.abs(df["val_hres"]), np.abs(df["val_gfs"])) + eps
        df["agree_flag"] = (np.abs(df["val_hres"] - df["val_gfs"]) < 0.2 * max_val).astype(float)

        # ── Geographic / cyclic encoding ──
        df["lat_sin"]  = np.sin(np.radians(df["latitude"]))
        df["lat_cos"]  = np.cos(np.radians(df["latitude"]))
        df["lon_sin"]  = np.sin(np.radians(df["longitude"]))
        df["lon_cos"]  = np.cos(np.radians(df["longitude"]))

        # ── Spatial context ──
        df["dist_coast_proxy"] = np.abs(df["longitude"] - 80.0)   # crude E-coast proxy

        # Map back feature names to match FEATURE_NAMES
        df["val_hres"] = df["val_hres_feat"]
        df["val_gfs"]  = df["val_gfs_feat"]

        return df

    @staticmethod
    def _compute_optimal_weight(
        val_hres: np.ndarray, val_gfs: np.ndarray, val_truth: np.ndarray
    ) -> np.ndarray:
        """
        Closed-form optimal HRES weight for each grid point:
            w* = (truth - gfs) / (hres - gfs)
        Clipped to [0, 1] so the blend stays between the two models.
        """
        denom = val_hres - val_gfs
        # Where models agree (|denom| < eps), equal weighting is optimal
        safe = np.abs(denom) > 1e-8
        w = np.where(safe, (val_truth - val_gfs) / (denom + 1e-12), 0.5)
        return np.clip(w, 0.0, 1.0)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def prepare_training_data(
        self,
        ds_hres: xr.Dataset,
        ds_gfs: xr.Dataset,
        ds_truth: xr.Dataset,
        ds_spread: xr.Dataset | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build (X, y) where y = optimal per-point HRES weight."""
        print(f"[Weighter] Preparing training data for {self.target_var}…")

        df_hres  = ds_hres.to_dataframe().reset_index()
        df_gfs   = ds_gfs.to_dataframe().reset_index()
        df_spread = ds_spread.to_dataframe().reset_index() if ds_spread is not None else None

        df = self._make_features(df_hres, df_gfs, df_spread)

        truth_var = f"{self.target_var}_truth"
        # ── Align ERA5 truth to the HRES/GFS grid via nearest-neighbour ──
        truth_on_grid = ds_truth[truth_var].interp(
            latitude=ds_hres.latitude,
            longitude=ds_hres.longitude,
            method="nearest",
        )
        df_truth = truth_on_grid.to_dataframe(name=truth_var).reset_index()
        df = pd.merge(df, df_truth[["latitude", "longitude", truth_var]],
                      on=["latitude", "longitude"])

        df = df.dropna(subset=["val_hres", "val_gfs"])
        df[truth_var] = df[truth_var].fillna(df[["val_hres","val_gfs"]].mean(axis=1))

        # Compute optimal weight target
        # For computing weight, we need original values (not log scaled).
        # We must re-extract them since 'val_hres' in df is log-scaled for 'tp'
        orig_hres = df_hres.set_index(["latitude", "longitude"])[self.target_var].values
        orig_gfs  = df_gfs.set_index(["latitude", "longitude"])[self.target_var].values
        # But wait, df order might be different. Let's merge originals back
        df_orig = pd.merge(
            df_hres[["latitude", "longitude", self.target_var]].rename(columns={self.target_var: "orig_hres"}),
            df_gfs[["latitude", "longitude", self.target_var]].rename(columns={self.target_var: "orig_gfs"}),
            on=["latitude", "longitude"]
        )
        df = pd.merge(df, df_orig, on=["latitude", "longitude"])
        
        df["w_opt"] = self._compute_optimal_weight(
            df["orig_hres"].values,
            df["orig_gfs"].values,
            df[truth_var].values,
        )

        X = df[self.FEATURE_NAMES].fillna(0.0)
        y = df["w_opt"]

        self._train_df = df
        
        # Add original names back so evaluation can use them
        self._train_df[f"{self.target_var}_hres"] = df["orig_hres"]
        self._train_df[f"{self.target_var}_gfs"] = df["orig_gfs"]
        self._train_df[f"{self.target_var}_truth"] = df[truth_var]

        print(f"  → {len(X)} training samples")
        print(f"  → Target: optimal HRES weight  μ={y.mean():.3f}  σ={y.std():.3f}")
        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit with spatially-stratified 80/20 split (every 5th latitude band → val)."""
        print(f"[Weighter] Training on {len(X)} samples for {self.target_var}…")

        n = len(X)
        idx = np.arange(n)
        val_mask = (idx % 5) == 0
        X_tr, X_val = X.loc[~val_mask], X.loc[val_mask]
        y_tr, y_val = y.loc[~val_mask], y.loc[val_mask]

        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        self.trained = True
        self.save()

        print("[Weighter] Training complete.")
        w_pred = self.model.predict(X)
        print(f"  Learned mean HRES weight: {w_pred.mean():.3f} ± {w_pred.std():.3f}")
        print(f"  Learned mean GFS  weight: {(1-w_pred).mean():.3f}")

    def predict(
        self,
        ds_hres: xr.Dataset,
        ds_gfs: xr.Dataset,
        ds_spread: xr.Dataset | None = None,
    ) -> xr.Dataset:
        """Generate blended output using learned optimal weights."""
        if not self.trained:
            self.load()

        print(f"[Weighter] Generating blended forecast for {self.target_var}…")
        df_hres  = ds_hres.to_dataframe().reset_index()
        df_gfs   = ds_gfs.to_dataframe().reset_index()
        df_spread = ds_spread.to_dataframe().reset_index() if ds_spread is not None else None

        df = self._make_features(df_hres, df_gfs, df_spread)
        
        # Get raw values to apply weights to
        df_orig = pd.merge(
            df_hres[["latitude", "longitude", self.target_var]].rename(columns={self.target_var: "orig_hres"}),
            df_gfs[["latitude", "longitude", self.target_var]].rename(columns={self.target_var: "orig_gfs"}),
            on=["latitude", "longitude"]
        )
        df = pd.merge(df, df_orig, on=["latitude", "longitude"])
        
        valid = df[["orig_hres", "orig_gfs"]].notnull().all(axis=1)
        df_v  = df[valid].copy()

        X_pred = df_v[self.FEATURE_NAMES].fillna(0.0)
        w = self.model.predict(X_pred).clip(0, 1)

        # Weighted blend
        blended_col = f"{self.target_var}_blended"
        df_v[blended_col] = w * df_v["orig_hres"] + (1 - w) * df_v["orig_gfs"]

        if self.target_var == "tp":
            # Physical clip: blended cannot exceed max(HRES, GFS) or go negative for precip
            df_v[blended_col] = df_v[blended_col].clip(
                lower=0,
                upper=np.maximum(df_v["orig_hres"], df_v["orig_gfs"]),
            )
        
        hres_col = f"{self.target_var}_hres"
        gfs_col = f"{self.target_var}_gfs"
        
        df_v[hres_col] = df_v["orig_hres"]
        df_v[gfs_col]  = df_v["orig_gfs"]
        df_v[f"{self.target_var}_hres_weight"] = w
        df_v[f"{self.target_var}_gfs_weight"]  = 1 - w

        out_cols = [hres_col, gfs_col, blended_col, f"{self.target_var}_hres_weight", f"{self.target_var}_gfs_weight"]
        ds_out = df_v.set_index(["latitude", "longitude"])[out_cols].to_xarray()

        spread_var = f"{self.target_var}_spread"
        if ds_spread is not None and spread_var in ds_spread:
            ds_out[spread_var] = ds_spread[spread_var]

        print(f"  → output shape: {ds_out[blended_col].shape}")
        return ds_out

    def save(self, path: str = None) -> None:
        if path is None:
            path = self.model_file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)

    def load(self, path: str = None) -> None:
        if path is None:
            path = self.model_file
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model at {path}. Run pipeline first.")
        self.model.load_model(path)
        self.trained = True
