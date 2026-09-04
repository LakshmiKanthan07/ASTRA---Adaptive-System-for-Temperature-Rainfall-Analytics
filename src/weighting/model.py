"""
ASTRA — Adaptive Weighting Model
XGBoost-based spatial weight learner that blends forecasts from multiple
NWP sources (ECMWF HRES + GFS) using ERA5 historical skill as supervision.
"""
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
import joblib
import os


class AdaptiveWeighter:
    """
    Learns spatially-varying blending weights using gradient-boosted trees.

    Feature set:
        latitude, longitude       — geographic location
        tp_hres, tp_gfs           — raw model precipitation predictions
        tp_spread                 — ENS uncertainty (if available)
        lat_sin, lon_cos          — cyclically-encoded location features

    Target:
        tp_truth                  — ERA5 accumulated precipitation (ground truth)
    """

    MODEL_FILE = "data/xgb_adaptive_weighter.json"

    def __init__(self, n_estimators: int = 200, max_depth: int = 6,
                 learning_rate: float = 0.08, subsample: float = 0.8):
        self.model_tp = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="rmse",
            verbosity=0,
        )
        self.feature_names: list[str] = []
        self.trained: bool = False

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _make_features(
        self,
        df_hres: pd.DataFrame,
        df_gfs: pd.DataFrame,
        df_spread: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Merge source dataframes and engineer features."""
        df = pd.merge(
            df_hres[["latitude", "longitude", "tp"]],
            df_gfs[["latitude", "longitude", "tp"]],
            on=["latitude", "longitude"],
            suffixes=("_hres", "_gfs"),
        )

        # Ensemble spread as uncertainty proxy (optional)
        if df_spread is not None and "tp_spread" in df_spread.columns:
            df = pd.merge(
                df,
                df_spread[["latitude", "longitude", "tp_spread"]],
                on=["latitude", "longitude"],
                how="left",
            )
        else:
            df["tp_spread"] = np.nan

        # Derived / interaction features
        df["tp_model_diff"]  = df["tp_hres"] - df["tp_gfs"]
        df["tp_model_mean"]  = (df["tp_hres"] + df["tp_gfs"]) / 2.0
        df["lat_sin"]        = np.sin(np.radians(df["latitude"]))
        df["lon_cos"]        = np.cos(np.radians(df["longitude"]))

        self.feature_names = [
            "latitude", "longitude",
            "tp_hres", "tp_gfs",
            "tp_spread",
            "tp_model_diff", "tp_model_mean",
            "lat_sin", "lon_cos",
        ]
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_training_data(
        self,
        ds_hres: xr.Dataset,
        ds_gfs: xr.Dataset,
        ds_truth: xr.Dataset,
        ds_spread: xr.Dataset | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build and return (X, y) for XGBoost training."""
        print("[Weighter] Preparing training data…")

        df_hres  = ds_hres.to_dataframe().reset_index()
        df_gfs   = ds_gfs.to_dataframe().reset_index()
        df_truth = ds_truth.to_dataframe().reset_index()
        df_spread = ds_spread.to_dataframe().reset_index() if ds_spread is not None else None

        df = self._make_features(df_hres, df_gfs, df_spread)

        # Merge ground truth
        df = pd.merge(df, df_truth[["latitude", "longitude", "tp_truth"]],
                      on=["latitude", "longitude"])

        # Drop NaN rows on critical columns
        df = df.dropna(subset=["tp_hres", "tp_gfs", "tp_truth"])

        X = df[self.feature_names].fillna(0.0)
        y = df["tp_truth"]

        print(f"  → {len(X)} training samples | features: {self.feature_names}")
        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit XGBoost model with early stopping on a 10 % validation split."""
        print(f"[Weighter] Training on {len(X)} samples…")
        split = int(len(X) * 0.9)
        X_tr, X_val = X.iloc[:split], X.iloc[split:]
        y_tr, y_val = y.iloc[:split], y.iloc[split:]

        self.model_tp.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        self.trained = True
        self.save()
        print("[Weighter] Training complete.")
        print(f"  Feature importances:")
        for name, imp in sorted(
            zip(self.feature_names, self.model_tp.feature_importances_),
            key=lambda x: -x[1],
        ):
            print(f"    {name:<25}: {imp:.4f}")

    def predict(
        self,
        ds_hres: xr.Dataset,
        ds_gfs: xr.Dataset,
        ds_spread: xr.Dataset | None = None,
    ) -> xr.Dataset:
        """
        Generate blended precipitation grid.
        Returns an xr.Dataset with variables: tp_blended, tp_hres, tp_gfs.
        """
        if not self.trained:
            self.load()

        print("[Weighter] Generating blended forecast…")
        df_hres  = ds_hres.to_dataframe().reset_index()
        df_gfs   = ds_gfs.to_dataframe().reset_index()
        df_spread = ds_spread.to_dataframe().reset_index() if ds_spread is not None else None

        df = self._make_features(df_hres, df_gfs, df_spread)
        valid_mask = df[["tp_hres", "tp_gfs"]].notnull().all(axis=1)
        df_valid = df[valid_mask].copy()

        X_pred = df_valid[self.feature_names].fillna(0.0)
        df_valid["tp_blended"] = self.model_tp.predict(X_pred)

        # Reconstruct multi-variable xarray Dataset
        ds_out = df_valid.set_index(["latitude", "longitude"])[
            ["tp_hres", "tp_gfs", "tp_blended"]
        ].to_xarray()

        # Add ENS spread if available
        if ds_spread is not None and "tp_spread" in ds_spread:
            ds_out["tp_spread"] = ds_spread["tp_spread"]

        print(f"  → output shape: {ds_out['tp_blended'].shape}")
        return ds_out

    def get_feature_importances(self) -> pd.Series:
        """Return feature importances as a named Series."""
        return pd.Series(self.model_tp.feature_importances_, index=self.feature_names).sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = MODEL_FILE) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model_tp.save_model(path)
        print(f"  [Weighter] Model saved → {path}")

    def load(self, path: str = MODEL_FILE) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model found at {path}. Run pipeline first.")
        self.model_tp.load_model(path)
        self.trained = True
        print(f"  [Weighter] Model loaded ← {path}")
