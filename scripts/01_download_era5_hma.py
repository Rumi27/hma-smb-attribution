"""
Script 01 — Download ERA5 Monthly Means for Full HMA

Coverage:  High Mountain Asia (regions 13, 14, 15)
           lat 25–47°N, lon 64–106°E  (1° buffer on every side)
Variables: 2m air temperature (t2m), total precipitation (tp)
Period:    2000–2018 (19 years × 12 months)
Format:    One NetCDF per year, both variables in same file
Output:    paper_HMA_attribution/data/era5/era5_hma_{year}.nc

Estimated size: ~19 files × ~8 MB = ~150 MB total
Estimated time: 30–90 min depending on CDS queue

Resume-safe: already-downloaded years are skipped.

Usage:
  python scripts/01_download_era5_hma.py
  python scripts/01_download_era5_hma.py --years 2000 2001  # specific years
"""

import cdsapi
import argparse
from pathlib import Path
import time
import sys

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR   = Path(__file__).parent.parent / "data" / "era5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# HMA bounding box [North, West, South, East] — 1° buffer beyond glacier extent
AREA      = [47, 64, 25, 106]
MONTHS    = [f"{m:02d}" for m in range(1, 13)]
YEARS     = list(range(2000, 2019))  # 2000–2018 inclusive
VARIABLES = ["2m_temperature", "total_precipitation"]

# ── Parse arguments ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--years", nargs="+", type=int, default=YEARS,
                    help="Specific years to download (default: all 2000–2018)")
parser.add_argument("--force", action="store_true",
                    help="Re-download even if file already exists")
args = parser.parse_args()

print("=" * 70)
print("SCRIPT 01 — ERA5 HMA DOWNLOAD")
print("=" * 70)
print(f"  Output dir : {OUT_DIR}")
print(f"  Area       : N={AREA[0]} W={AREA[1]} S={AREA[2]} E={AREA[3]}")
print(f"  Variables  : {VARIABLES}")
print(f"  Years      : {args.years[0]}–{args.years[-1]} ({len(args.years)} years)")
print(f"  Months     : all 12")

# ── CDS client ────────────────────────────────────────────────────────────────
try:
    c = cdsapi.Client(quiet=False)
    print("\n  CDS client initialised successfully.")
except Exception as e:
    print(f"\n  ERROR: CDS client failed: {e}")
    print("  Make sure ~/.cdsapirc is set up with your UID and API key.")
    print("  Register at: https://cds.climate.copernicus.eu")
    sys.exit(1)

# ── Download loop ─────────────────────────────────────────────────────────────
downloaded, skipped, failed = [], [], []

for year in args.years:
    out_file = OUT_DIR / f"era5_hma_{year}.nc"

    if out_file.exists() and not args.force:
        size_mb = out_file.stat().st_size / 1e6
        print(f"\n  [{year}] SKIP — already exists ({size_mb:.1f} MB): {out_file.name}")
        skipped.append(year)
        continue

    print(f"\n  [{year}] Requesting from CDS ...")
    t0 = time.time()

    # New CDS API (2024+) returns a ZIP containing separate NC files per variable.
    # We download to a temp zip, extract, merge t2m + tp, save as one NC per year.
    zip_tmp = OUT_DIR / f"_tmp_{year}.zip"

    try:
        c.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis",
                "variable"    : VARIABLES,
                "year"        : str(year),
                "month"       : MONTHS,
                "time"        : "00:00",
                "area"        : AREA,
                "format"      : "netcdf",
            },
            str(zip_tmp),
        )

        # Unzip and merge variables into single file
        import zipfile, xarray as xr, shutil, tempfile
        tmp_dir = OUT_DIR / f"_unzip_{year}"
        tmp_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_tmp, "r") as zf:
            zf.extractall(tmp_dir)

        nc_files = sorted(tmp_dir.glob("*.nc"))
        datasets = [xr.open_dataset(f, engine="netcdf4") for f in nc_files]
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            merged = xr.merge(datasets, join="outer", compat="no_conflicts")
        for ds in datasets:
            ds.close()
        merged.to_netcdf(str(out_file))
        merged.close()

        # Cleanup
        shutil.rmtree(tmp_dir)
        zip_tmp.unlink()

        elapsed = time.time() - t0
        size_mb = out_file.stat().st_size / 1e6
        print(f"  [{year}] DONE — {size_mb:.1f} MB in {elapsed:.0f}s → {out_file.name}")
        downloaded.append(year)

    except Exception as e:
        print(f"  [{year}] FAILED: {e}")
        failed.append(year)
        for p in [out_file, zip_tmp]:
            if p.exists():
                p.unlink()
        import shutil
        tmp_dir = OUT_DIR / f"_unzip_{year}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    # Brief pause between requests to be polite to the CDS queue
    if year != args.years[-1]:
        time.sleep(2)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DOWNLOAD SUMMARY")
print(f"  Downloaded : {len(downloaded)} years {downloaded}")
print(f"  Skipped    : {len(skipped)} years (already existed)")
print(f"  Failed     : {len(failed)} years {failed}")

if failed:
    print(f"\n  Retry failed years with:")
    print(f"  python scripts/01_download_era5_hma.py --years {' '.join(map(str, failed))}")

# ── Validate downloaded files ─────────────────────────────────────────────────
print("\n  Validating downloaded files ...")
import xarray as xr

all_files = sorted(OUT_DIR.glob("era5_hma_*.nc"))
print(f"  Total ERA5 files in {OUT_DIR.name}/: {len(all_files)}")

if all_files:
    print("\n  Checking first file ...")
    import xarray as xr
    ds = xr.open_dataset(all_files[0], engine="netcdf4")
    print(f"    Variables : {list(ds.data_vars)}")
    tc = "valid_time" if "valid_time" in ds.coords else "time"
    print(f"    Lat range : {float(ds.latitude.min()):.1f} to {float(ds.latitude.max()):.1f}")
    print(f"    Lon range : {float(ds.longitude.min()):.1f} to {float(ds.longitude.max()):.1f}")
    print(f"    Time steps: {len(ds[tc])}")
    ds.close()

print("\n  Next step: run scripts/02_build_glacier_centroids.py")
print("=" * 70)
