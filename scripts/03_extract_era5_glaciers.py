"""
Script 03 — Extract ERA5 Climate Time Series at Glacier Centroids

For each glacier in glacier_centroids_hma.parquet, samples ERA5 monthly
temperature and precipitation at the nearest 0.25° grid point.

Input:
  data/era5/era5_hma_{year}.nc     — ERA5 downloaded by script 01
  data/glacier_centroids_hma.parquet — centroids from script 02

Output:
  data/monthly_climate_hma.parquet
    columns: RGIId, Year, Month, Temp_C, Precip_mm
    one row per glacier per month (19 yr × 12 months × N glaciers)

  data/era5_climate_metrics_hma.parquet
    columns: RGIId + 18 climate feature columns (means, trends, variability)
    one row per glacier — ready to merge into feature matrix
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PAPER_DIR   = Path(__file__).parent.parent
DATA_DIR    = PAPER_DIR / "data"
ERA5_DIR    = DATA_DIR / "era5"
CENTROIDS   = DATA_DIR / "glacier_centroids_hma.parquet"

YEARS       = list(range(2000, 2019))
MONTHS      = list(range(1, 13))

print("=" * 70)
print("SCRIPT 03 — EXTRACT ERA5 AT GLACIER CENTROIDS")
print("=" * 70)

# ── Load centroids ─────────────────────────────────────────────────────────────
print("\n[1] Loading glacier centroids ...")
cents = pd.read_parquet(CENTROIDS)
print(f"    {len(cents):,} glaciers loaded")

lats_all = cents["cenlat"].values
lons_all = cents["cenlon"].values
rgi_ids  = cents["RGIId"].values   # numeric IDs (e.g. 13.12345)

# ── Check available ERA5 files ─────────────────────────────────────────────────
era5_files = {int(f.stem.split("_")[-1]): f
              for f in sorted(ERA5_DIR.glob("era5_hma_*.nc"))}
available_years = sorted(era5_files.keys())
missing_years   = [y for y in YEARS if y not in era5_files]

print(f"\n[2] ERA5 files: {len(available_years)} available, "
      f"{len(missing_years)} missing")
if missing_years:
    print(f"    Missing years: {missing_years}")
    print(f"    Run script 01 first to download missing files.")
    if not available_years:
        raise SystemExit("No ERA5 files found. Run script 01 first.")

years_to_process = available_years  # process what we have

# ── Extract loop ───────────────────────────────────────────────────────────────
print(f"\n[3] Extracting ERA5 at {len(cents):,} glacier centroids ...")
print(f"    Years to process: {years_to_process[0]}–{years_to_process[-1]}")

all_records = []

for year in years_to_process:
    era5_file = era5_files[year]
    print(f"    Processing {year} ... ", end="", flush=True)

    try:
        ds = xr.open_dataset(era5_file)

        # Identify variable names (may vary by ERA5 version)
        t2m_var = next((v for v in ds.data_vars if v in ("t2m", "2m_temperature")), None)
        tp_var  = next((v for v in ds.data_vars if v in ("tp",  "total_precipitation")), None)

        if t2m_var is None or tp_var is None:
            print(f"SKIP — vars not found (got: {list(ds.data_vars)})")
            ds.close()
            continue

        era5_lats = ds["latitude"].values
        era5_lons = ds["longitude"].values

        # Time coordinate: handle 'valid_time' or 'time'
        time_coord = "valid_time" if "valid_time" in ds.coords else "time"
        all_times  = pd.to_datetime(ds[time_coord].values)

        # CDS new API interleaves t2m (00:00) and tp (06:00) time steps.
        # Select valid (non-NaN) time steps for each variable independently
        # by checking which steps have data (no NaN at a sample point).
        mid_lat_i = len(era5_lats) // 2
        mid_lon_i = len(era5_lons) // 2

        t2m_all = ds[t2m_var].values          # (n_times, lat, lon)  K
        tp_all  = ds[tp_var].values           # (n_times, lat, lon)  m

        # Valid step = not NaN at centre pixel
        t2m_valid = ~np.isnan(t2m_all[:, mid_lat_i, mid_lon_i])
        tp_valid  = ~np.isnan(tp_all[:,  mid_lat_i, mid_lon_i])
        ds.close()

        t2m_data   = (t2m_all[t2m_valid] - 273.15)    # K → °C
        tp_data    = (tp_all[tp_valid]   * 1000.0)    # m → mm
        t2m_months = [t.month for t in all_times[t2m_valid]]
        tp_months  = [t.month for t in all_times[tp_valid]]

        if len(t2m_months) != 12 or len(tp_months) != 12:
            print(f"WARN — unexpected month count "
                  f"t2m:{len(t2m_months)} tp:{len(tp_months)}, using available")

        # Find nearest ERA5 grid index for each glacier (vectorised, paired indexing)
        # lat_idx[i] = row index of nearest ERA5 lat for glacier i
        # lon_idx[i] = col index of nearest ERA5 lon for glacier i
        lat_idx = np.abs(era5_lats[:, None] - lats_all[None, :]).argmin(axis=0)
        lon_idx = np.abs(era5_lons[:, None] - lons_all[None, :]).argmin(axis=0)

        # t2m_data shape: (12, n_lat, n_lon)
        # t2m_data[i, lat_idx, lon_idx] → (n_glaciers,) via paired advanced indexing
        tp_month_map = {m: i for i, m in enumerate(tp_months)}

        for i, month in enumerate(t2m_months):
            t_sampled = t2m_data[i, lat_idx, lon_idx]   # (n_glaciers,)
            j = tp_month_map.get(month)
            p_sampled = tp_data[j, lat_idx, lon_idx] if j is not None \
                        else np.zeros(len(rgi_ids))      # (n_glaciers,)

            df_month = pd.DataFrame({
                "RGIId"    : rgi_ids,
                "Year"     : year,
                "Month"    : month,
                "Temp_C"   : t_sampled,
                "Precip_mm": p_sampled,
            })
            all_records.append(df_month)

        print(f"OK ({len(t2m_months)} months)")

    except Exception as e:
        print(f"ERROR: {e}")
        continue

# ── Save monthly time series ───────────────────────────────────────────────────
print(f"\n[4] Building monthly climate DataFrame ...")
monthly = pd.concat(all_records, ignore_index=True)
print(f"    Rows: {len(monthly):,}  (expected ~{len(cents)*12*len(years_to_process):,})")

out_monthly = DATA_DIR / "monthly_climate_hma.parquet"
monthly.to_parquet(out_monthly, index=False)
print(f"    Saved: {out_monthly}  ({out_monthly.stat().st_size/1e6:.1f} MB)")

# ── Compute climate metrics per glacier ────────────────────────────────────────
print(f"\n[5] Computing per-glacier climate metrics ...")

monthly["season"] = monthly["Month"].map({
    12:"DJF", 1:"DJF", 2:"DJF",
    3:"MAM",  4:"MAM", 5:"MAM",
    6:"JJA",  7:"JJA", 8:"JJA",
    9:"SON",  10:"SON",11:"SON",
})

# Annual means
ann_T = (monthly.groupby(["RGIId","Year"])["Temp_C"]
                 .mean().reset_index(name="T_ann"))
ann_P = (monthly.groupby(["RGIId","Year"])["Precip_mm"]
                 .sum().reset_index(name="P_ann"))
jja_T = (monthly[monthly["season"]=="JJA"]
                 .groupby(["RGIId","Year"])["Temp_C"]
                 .mean().reset_index(name="T_JJA"))
djf_P = (monthly[monthly["season"]=="DJF"]
                 .groupby(["RGIId","Year"])["Precip_mm"]
                 .sum().reset_index(name="P_DJF"))

# Climatological means (full period)
clim = (monthly.groupby("RGIId")
               .agg(T_mean_C=("Temp_C","mean"),
                    P_mean_mm=("Precip_mm","mean"))
        )
for season in ["JJA","DJF","MAM","SON"]:
    sub = (monthly[monthly["season"]==season]
                   .groupby("RGIId")
                   .agg(**{f"T_{season}_C": ("Temp_C","mean"),
                           f"P_{season}_mm":("Precip_mm","mean")}))
    clim = clim.join(sub)

# Trends [units per decade]
def compute_trends(df_yr, id_col, yr_col, val_col, out_name):
    out = {}
    for rid, grp in df_yr.groupby(id_col):
        x = grp[yr_col].values.astype(float)
        y = grp[val_col].values.astype(float)
        mask = ~np.isnan(y)
        if mask.sum() >= 5:
            slope, *_ = stats.linregress(x[mask], y[mask])
            out[rid] = slope * 10
        else:
            out[rid] = np.nan
    return pd.Series(out, name=out_name)

print("    Computing trends ...")
trend_T_ann = compute_trends(ann_T,"RGIId","Year","T_ann", "T_trend_C_dec")
trend_T_JJA = compute_trends(jja_T,"RGIId","Year","T_JJA", "T_JJA_trend_C_dec")
trend_P_ann = compute_trends(ann_P,"RGIId","Year","P_ann", "P_trend_mm_dec")

# Variability
std_T_ann = ann_T.groupby("RGIId")["T_ann"].std().rename("T_std_C")
std_T_JJA = jja_T.groupby("RGIId")["T_JJA"].std().rename("T_JJA_std_C")
std_P_ann = ann_P.groupby("RGIId")["P_ann"].std().rename("P_std_mm")

# Extremes
max_T_JJA = jja_T.groupby("RGIId")["T_JJA"].max().rename("T_JJA_max_C")
min_T_DJF = (monthly[monthly["season"]=="DJF"]
                     .groupby("RGIId")["Temp_C"].min().rename("T_DJF_min_C"))

# Assemble
era5_metrics = (clim
                .join(trend_T_ann).join(trend_T_JJA).join(trend_P_ann)
                .join(std_T_ann).join(std_T_JJA).join(std_P_ann)
                .join(max_T_JJA).join(min_T_DJF))

print(f"    ERA5 metrics: {era5_metrics.shape[1]} features for "
      f"{len(era5_metrics):,} glaciers")

out_metrics = DATA_DIR / "era5_climate_metrics_hma.parquet"
era5_metrics.reset_index().to_parquet(out_metrics, index=False)
print(f"    Saved: {out_metrics}  ({out_metrics.stat().st_size/1e6:.1f} MB)")

print("\n" + "=" * 70)
print("Script 03 complete.")
print("Next step: run scripts/04_build_feature_matrix_hma.py")
print("=" * 70)
