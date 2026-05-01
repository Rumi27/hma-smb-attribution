"""
Script 04 — Build Full HMA Feature Matrix

Merges:
  ERA5 climate metrics (script 03)  +  RGI/Shean topography (script 02)
  → one row per glacier, 33 features + target SMB_mwea

Output: data/feature_matrix_hma.parquet
"""
import pandas as pd
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"

print("=" * 70)
print("SCRIPT 04 — BUILD FULL HMA FEATURE MATRIX")
print("=" * 70)

# Load
climate = pd.read_parquet(DATA / "era5_climate_metrics_hma.parquet")
cents   = pd.read_parquet(DATA / "glacier_centroids_hma.parquet")

print(f"\n  ERA5 metrics  : {climate.shape}")
print(f"  Centroids+MB  : {cents.shape}")

# Merge
feat = cents.merge(climate, on="RGIId", how="inner")
print(f"  After merge   : {len(feat):,} glaciers")

# Quality filters
n0   = len(feat)
feat = feat[
    (feat["SMB_sigma"]   < 1.0)           &
    (feat["SMB_mwea"].abs() < 5.0)        &
    feat["zmean_m"].notna()               &
    feat["slope_deg"].notna()             &
    feat["area_km2"].notna()
].copy()
print(f"  After filters : {len(feat):,}  (removed {n0-len(feat):,})")

# ── Derived features ──────────────────────────────────────────────────────────
feat["log_area_km2"] = np.log10(feat["area_km2"].clip(0.001))

# Lapse-rate correction: adjust ERA5 T2m (valid at ERA5 grid-cell orography
# elevation) to glacier mean elevation using Δz = z_glacier - z_ERA5_orog.
# Standard atmospheric lapse rate 6.5 °C / 1000 m.
LAPSE = 6.5 / 1000.0   # °C m⁻¹
_orog_ds = xr.open_dataset(DATA / "era5" / "era5_hma_orography.nc")
_z_orog  = (_orog_ds["z"].squeeze() / 9.80665).values   # m
_lats    = _orog_ds["latitude"].values
_lons    = _orog_ds["longitude"].values
_lon2d, _lat2d = np.meshgrid(_lons, _lats)
_pts  = np.column_stack([_lat2d.ravel(), _lon2d.ravel()])
_tree = cKDTree(_pts)
_, _idx = _tree.query(feat[["cenlat", "cenlon"]].values)
feat["z_era5_orog_m"] = _z_orog.ravel()[_idx]
feat["T_JJA_lapse_C"] = feat["T_JJA_C"]  - (feat["zmean_m"] - feat["z_era5_orog_m"]) * LAPSE
feat["T_ann_lapse_C"] = feat["T_mean_C"] - (feat["zmean_m"] - feat["z_era5_orog_m"]) * LAPSE
print(f"\n  Lapse correction Δz: mean={( feat['zmean_m']-feat['z_era5_orog_m']).mean():.0f} m  "
      f"T_JJA_lapse mean={feat['T_JJA_lapse_C'].mean():.1f} °C  "
      f"(was −21.8 °C with old formula)")

# Continentality: large annual temperature range = continental interior
# (Tian Shan, Pamir) vs small range = maritime-influenced (Hindu Kush flanks)
feat["continentality"] = feat["T_JJA_C"] - feat["T_DJF_C"]

# Precipitation seasonality: winter-dominated = snow accumulation regime
# spring-melt; summer-dominated = monsoon regime
denom = (feat["P_mean_mm"] * 12).clip(lower=1.0)
feat["precip_winter_frac"] = (feat["P_DJF_mm"] * 3 / denom).clip(0, 1)
feat["P_snow_mm"]           = feat["P_DJF_mm"] + feat["P_MAM_mm"]   # cold-season accum

# Cryo-balance index: ratio of cold-season accumulation to summer melt forcing.
# Add 10 so denominator stays positive for cold glaciers where T_JJA_lapse < 0.
feat["cryo_balance"] = feat["P_snow_mm"] / (feat["T_JJA_lapse_C"] + 10.0).clip(lower=0.1)

# Melt index: lapse-corrected summer T weighted by glacier size proxy.
# Large warm glaciers experience greater total ablation.
feat["melt_index"] = feat["T_JJA_lapse_C"] * np.sqrt(feat["area_km2"].clip(lower=0.001))

# Precipitation trend × winter fraction interaction: where precip is declining
# AND winter-dominated, accumulation loss is amplified.
feat["P_trend_x_wfrac"] = feat["P_trend_mm_dec"] * feat["precip_winter_frac"]

print(f"\n  Final: {len(feat):,} glaciers, {feat.shape[1]} columns")
print(f"  SMB_mwea: {feat['SMB_mwea'].mean():.3f} ± {feat['SMB_mwea'].std():.3f}")

print("\n  By region:")
for r, grp in feat.groupby("rgi_region"):
    name = {13:"Central Asia", 14:"S Asia West", 15:"S Asia East"}.get(r,"")
    print(f"    Region {r} ({name}): {len(grp):6,}  "
          f"SMB={grp['SMB_mwea'].mean():.3f}±{grp['SMB_mwea'].std():.3f}")

out = DATA / "feature_matrix_hma.parquet"
feat.to_parquet(out, index=False)
print(f"\n  Saved: {out}  ({out.stat().st_size/1e6:.1f} MB)")
print("  Next: run scripts/05_train_models_hma.py")
print("=" * 70)
