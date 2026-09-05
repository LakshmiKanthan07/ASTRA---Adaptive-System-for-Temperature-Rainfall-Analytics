"""
ASTRA — Ingestion Module
Handles loading, variable extraction, and spatial regridding of GRIB/NetCDF files
from ECMWF HRES, ECMWF ENS, GFS, and ERA5 Reanalysis ground truth.
"""
import xarray as xr
import numpy as np


class GribLoader:
    """
    Unified data loader that ingests GRIB2/GRIB1 files from heterogeneous
    NWP sources and reprojects them to a common India-region lat/lon grid.

    Default grid: India Region  8°N–36°N, 68°E–96°E at 0.25° resolution.
    """

    VARS_RENAME = {
        # ECMWF standard short names → internal standard names
        "t2m": "t2m",
        "tp":  "tp",
        "msl": "msl",
        "u10": "u10",
        "v10": "v10",
        # GFS can use GRIB2 name 't' for temperature on pressure levels
        "t":   "t2m",
        "prmsl": "msl",
    }

    def __init__(
        self,
        target_lat_bounds: tuple = (8.0, 36.0),
        target_lon_bounds: tuple = (68.0, 96.0),
        resolution: float = 0.25,
    ):
        self.target_lats = np.arange(
            target_lat_bounds[1], target_lat_bounds[0] - resolution, -resolution
        )
        self.target_lons = np.arange(
            target_lon_bounds[0], target_lon_bounds[1] + resolution, resolution
        )
        self.resolution = resolution

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_load_cfgrib(self, filepath: str, filter_by_keys: dict | None = None) -> xr.Dataset | None:
        """Try to open a GRIB file with cfgrib, returning None on failure."""
        try:
            return xr.open_dataset(
                filepath, engine="cfgrib",
                filter_by_keys=filter_by_keys or {},
                backend_kwargs={"errors": "ignore"},
            )
        except Exception as exc:
            print(f"  [GribLoader] Warning: could not open {filepath} "
                  f"with keys={filter_by_keys}: {exc}")
            return None

    def _regrid(self, ds: xr.Dataset, method: str = "linear") -> xr.Dataset:
        """Regrid dataset to the common India grid."""
        return ds.interp(
            latitude=self.target_lats,
            longitude=self.target_lons,
            method=method,
        )

    def _rename_vars(self, ds: xr.Dataset) -> xr.Dataset:
        """Normalise variable names to internal standard."""
        # Only rename if the source key exists, it differs from the target name,
        # AND the target name does not already exist (avoids rename conflicts).
        rename_map = {
            k: v for k, v in self.VARS_RENAME.items()
            if k in ds and k != v and v not in ds
        }
        return ds.rename_vars(rename_map) if rename_map else ds

    def _drop_scalar_coords(self, ds: xr.Dataset) -> xr.Dataset:
        """Drop scalar (non-dimensional) coordinate variables that cause merge issues."""
        to_drop = [c for c in ds.coords if ds[c].dims == ()]
        return ds.drop_vars(to_drop, errors="ignore")

    def _compute_wind_speed(self, ds: xr.Dataset) -> xr.Dataset:
        if "u10" in ds and "v10" in ds:
            ds["wind"] = np.sqrt(ds["u10"]**2 + ds["v10"]**2)
        return ds

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------

    def load_hres(self, filepath: str) -> xr.Dataset:
        """Load ECMWF HRES deterministic forecast → common grid."""
        print(f"[Loader] HRES  ← {filepath}")
        ds = self._safe_load_cfgrib(filepath)
        if ds is None:
            raise RuntimeError(f"Cannot open HRES file: {filepath}")
        ds = self._rename_vars(ds)
        ds = self._compute_wind_speed(ds)
        keep = [v for v in ("tp", "t2m", "u10", "v10", "wind", "msl") if v in ds]
        ds = ds[keep]
        ds = self._drop_scalar_coords(ds)
        ds = self._regrid(ds, method="linear")
        print(f"  → variables: {list(ds.data_vars)}  shape: {ds['tp'].shape if 'tp' in ds else '?'}")
        return ds

    def load_gfs(self, filepath: str) -> xr.Dataset:
        """Load NOAA GFS → common grid."""
        print(f"[Loader] GFS   ← {filepath}")
        ds = self._safe_load_cfgrib(filepath)
        if ds is None:
            raise RuntimeError(f"Cannot open GFS file: {filepath}")
        ds = self._rename_vars(ds)
        ds = self._compute_wind_speed(ds)
        keep = [v for v in ("tp", "t2m", "u10", "v10", "wind", "msl") if v in ds]
        ds = ds[keep]
        ds = self._drop_scalar_coords(ds)
        ds = self._regrid(ds, method="linear")
        print(f"  → variables: {list(ds.data_vars)}  shape: {ds['tp'].shape if 'tp' in ds else '?'}")
        return ds

    def load_ens_spread(self, filepath: str) -> xr.Dataset:
        """
        Load ECMWF ENS perturbed members, compute ensemble spread
        (inter-member std-dev) as a model uncertainty proxy.
        """
        print(f"[Loader] ENS   ← {filepath}")
        ds = self._safe_load_cfgrib(filepath)
        if ds is None:
            raise RuntimeError(f"Cannot open ENS file: {filepath}")
        ds = self._rename_vars(ds)
        ds = self._compute_wind_speed(ds)
        spread_vars = {}
        for var in ("tp", "t2m", "wind"):
            if var in ds and "number" in ds[var].dims:
                spread_vars[f"{var}_spread"] = ds[var].std(dim="number")
                spread_vars[f"{var}_ensmean"] = ds[var].mean(dim="number")
        ds_spread = xr.Dataset(spread_vars)
        ds_spread = self._drop_scalar_coords(ds_spread)
        ds_spread = self._regrid(ds_spread, method="linear")
        print(f"  → spread variables: {list(ds_spread.data_vars)}")
        return ds_spread

    def load_ground_truth(self, filepath: str) -> xr.Dataset:
        """
        Load ERA5 historical reanalysis as ground-truth labels.
        Selects only the first time-step to avoid IOProblem on 4 GB files.
        """
        print(f"[Loader] ERA5  ← {filepath}")
        
        # Load accumulated (tp)
        ds_accum = self._safe_load_cfgrib(filepath, filter_by_keys={"stepType": "accum"})
        if ds_accum is not None:
            if "time" in ds_accum.dims:
                ds_accum = ds_accum.isel(time=0).drop_vars("time", errors="ignore")
            ds_accum = ds_accum.load()  # Load into memory to avoid IO locks later
            ds_accum = self._rename_vars(ds_accum)

        # Load instant (t2m, u10, v10)
        ds_inst = self._safe_load_cfgrib(filepath, filter_by_keys={"stepType": "instant"})
        if ds_inst is not None:
            if "time" in ds_inst.dims:
                ds_inst = ds_inst.isel(time=0).drop_vars("time", errors="ignore")
            ds_inst = ds_inst.load()  # Load into memory to avoid IO locks later
            ds_inst = self._rename_vars(ds_inst)
            ds_inst = self._compute_wind_speed(ds_inst)

        # Merge them
        ds_merged = xr.Dataset()
        if ds_accum is not None and "tp" in ds_accum:
            ds_merged["tp"] = ds_accum["tp"]
            
        if ds_inst is not None:
            for v in ("t2m", "u10", "v10", "wind"):
                if v in ds_inst:
                    ds_merged[v] = ds_inst[v]
                    
        if len(ds_merged.data_vars) == 0:
             raise RuntimeError(f"Cannot extract truth variables from ERA5 file: {filepath}")

        ds = self._drop_scalar_coords(ds_merged)

        # Rename for disambiguation
        rename_truth = {v: f"{v}_truth" for v in ds.data_vars}
        ds = ds.rename_vars(rename_truth)

        ds = self._regrid(ds, method="nearest")
        print(f"  → truth variables: {list(ds.data_vars)}")
        return ds
