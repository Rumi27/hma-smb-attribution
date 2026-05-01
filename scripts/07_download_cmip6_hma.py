"""
Script 07 — Download CMIP6 Monthly T+P for HMA via PANGEO Google Cloud

Loads monthly tas and pr directly from public PANGEO Zarr stores (no account).
Subsets to HMA domain, saves as NetCDF.

Models   : MIROC6, MPI-ESM1-2-LR, IPSL-CM6A-LR
Periods  : historical 2000–2014  |  ssp245 2015–2100  |  ssp585 2015–2100
Domain   : lat 25–47°N, lon 64–106°E (0–360 lon convention handled)

Output   : data/cmip6/{model}_{experiment}_{variable}.nc
"""

import pandas as pd
import numpy as np
import xarray as xr
import intake
import gcsfs
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import sys

OUT_DIR = Path(__file__).parent.parent / "data" / "cmip6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAT_MIN, LAT_MAX = 25.0, 47.0
LON_MIN, LON_MAX = 64.0, 106.0   # 0–360 equiv: same (< 180)

CATALOG_URL = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"

# Models confirmed to have all 6 zarr stores (hist + ssp245 + ssp585) × (tas + pr)
MODELS = ["MIROC6", "MPI-ESM1-2-LR", "IPSL-CM6A-LR"]

EXPERIMENTS = {
    "historical": (2000, 2014),
    "ssp245":     (2015, 2100),
    "ssp585":     (2015, 2100),
}
VARIABLES = ["tas", "pr"]

print("=" * 70)
print("SCRIPT 07 — CMIP6 HMA DOWNLOAD (PANGEO ZARR)")
print("=" * 70)
print(f"  Models  : {MODELS}")
print(f"  Exps    : {list(EXPERIMENTS.keys())}")
print(f"  Vars    : {VARIABLES}")

# ── Load catalog once ─────────────────────────────────────────────────────────
print("\n[1] Loading PANGEO catalog ...")
col = intake.open_esm_datastore(CATALOG_URL)
print(f"    {len(col.df):,} entries")

fs = gcsfs.GCSFileSystem(token="anon")

downloaded, skipped, failed = [], [], []

# ── Download loop ─────────────────────────────────────────────────────────────
for model in MODELS:
    for experiment, (yr0, yr1) in EXPERIMENTS.items():
        for variable in VARIABLES:
            out_file = OUT_DIR / f"{model}_{experiment}_{variable}.nc"

            if out_file.exists():
                print(f"\n  SKIP — {out_file.name}  ({out_file.stat().st_size/1e6:.1f} MB)")
                skipped.append(out_file.name)
                continue

            print(f"\n  {model} / {experiment} / {variable} ({yr0}–{yr1}) ...",
                  end=" ", flush=True)

            try:
                # Find zarr URL from catalog
                cat = col.search(
                    source_id=model,
                    experiment_id=experiment,
                    variable_id=variable,
                    table_id="Amon",
                    member_id="r1i1p1f1",
                )
                if len(cat.df) == 0:
                    # CESM2 uses f2
                    cat = col.search(source_id=model, experiment_id=experiment,
                                     variable_id=variable, table_id="Amon",
                                     member_id="r1i1p1f2")
                if len(cat.df) == 0:
                    # Take any member
                    cat = col.search(source_id=model, experiment_id=experiment,
                                     variable_id=variable, table_id="Amon")
                if len(cat.df) == 0:
                    print("NOT FOUND in catalog")
                    failed.append(f"{model}/{experiment}/{variable}")
                    continue

                zstore = cat.df["zstore"].iloc[0]

                # Open zarr directly — lazy
                store = fs.get_mapper(zstore)
                ds    = xr.open_zarr(store, consolidated=True)

                # Normalise coord names
                rename = {}
                if "longitude" in ds.coords: rename["longitude"] = "lon"
                if "latitude"  in ds.coords: rename["latitude"]  = "lat"
                if rename: ds = ds.rename(rename)

                # Some zarr stores have reversed/non-monotonic time — sort first
                ds = ds.sortby("time")

                # Subset time
                ds = ds.sel(time=slice(str(yr0), str(yr1)))

                # Subset space — models use 0–360 longitude
                lons = ds["lon"].values
                if lons.max() > 180:
                    # 0–360: LON_MIN=64, LON_MAX=106 are valid as-is
                    ds = ds.where(
                        (ds["lon"] >= LON_MIN) & (ds["lon"] <= LON_MAX) &
                        (ds["lat"] >= LAT_MIN) & (ds["lat"] <= LAT_MAX),
                        drop=True)
                else:
                    ds = ds.sel(lat=slice(LAT_MIN, LAT_MAX),
                                lon=slice(LON_MIN, LON_MAX))

                # Keep only target variable + drop heavy bounds vars
                drop = [v for v in ds.data_vars if v != variable]
                ds   = ds.drop_vars(drop + ["time_bnds","lat_bnds","lon_bnds"],
                                    errors="ignore")

                n_time = len(ds["time"])
                print(f"shape=({n_time}, {len(ds.lat)}, {len(ds.lon)}) ... loading ...",
                      end=" ", flush=True)

                # Actually load into memory and save
                ds.load().to_netcdf(str(out_file))
                size_mb = out_file.stat().st_size / 1e6
                print(f"OK ({size_mb:.1f} MB)")
                downloaded.append(out_file.name)

            except Exception as e:
                print(f"\n    ERROR: {e}")
                failed.append(f"{model}/{experiment}/{variable}")
                if out_file.exists():
                    out_file.unlink()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DOWNLOAD SUMMARY")
print(f"  Downloaded : {len(downloaded)}")
print(f"  Skipped    : {len(skipped)}")
print(f"  Failed     : {len(failed)}")
if failed:
    for f in failed:
        print(f"    {f}")

nc_files = sorted(OUT_DIR.glob("*.nc"))
print(f"\n  Total CMIP6 NetCDF files: {len(nc_files)}")
for f in nc_files:
    print(f"    {f.name}  ({f.stat().st_size/1e6:.1f} MB)")

if nc_files:
    ds = xr.open_dataset(nc_files[0])
    tc = "time"
    print(f"\n  Sample: {nc_files[0].name}")
    print(f"    vars : {list(ds.data_vars)}")
    print(f"    time : {len(ds[tc])} steps  "
          f"{str(ds[tc].values[0])[:7]} – {str(ds[tc].values[-1])[:7]}")
    print(f"    lat  : {float(ds['lat'].min()):.1f} – {float(ds['lat'].max()):.1f}")
    print(f"    lon  : {float(ds['lon'].min()):.1f} – {float(ds['lon'].max()):.1f}")
    ds.close()

print("\n  Next: run scripts/08_project_smb_cmip6.py")
print("=" * 70)
