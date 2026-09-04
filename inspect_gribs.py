import xarray as xr
import glob
import traceback

files = ['ens.grib2', 'hres.grib2', 'gfs.t12z.pgrb2.0p25.f006', 'f986deebfac018041762bbe31b9959e0.grib']

for f in files:
    print(f"\n--- {f} ---")
    try:
        ds = xr.open_dataset(f, engine='cfgrib')
        print(ds)
        ds.close()
    except Exception as e:
        print(f"Failed to open with default cfgrib: {e}")
        try:
            # Try specific filters for GRIB if needed
            ds = xr.open_dataset(f, engine='cfgrib', filter_by_keys={'typeOfLevel': 'surface'})
            print("Opened with surface filter:")
            print(ds)
            ds.close()
        except Exception as e2:
            print(f"Failed to open with surface filter: {e2}")