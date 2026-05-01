"""
Script 02 — Build Glacier Centroid Table for All HMA Glaciers

Sources:
  Shean et al. 2020 geodetic MB CSV  → target variable + x,y coordinates
  RGI v7 attributes (region 13)      → authoritative sub-region codes via
                                        nearest-neighbour spatial join

Steps:
  1. Convert Shean x,y → lat/lon using AzEqDist 35N/85E projection
  2. Apply quality filters (valid_area_perc > 60%, dt > 5 yr, sigma < 1.0)
  3. Assign sub-region labels:
       Region 13: KD-tree nearest-neighbour join to RGI v7 → o2region code
       Regions 14, 15: geographic bounding-box approximation (no v7 file available)
  4. Save final centroid + MB + topography table

Sub-region codes (region 13, from RGI v7 o2region field):
  13-01  Pamir-Alay       13-02  Pamir
  13-03  W_Tian_Shan      13-04  E_Tian_Shan
  13-05  Karakoram_N      13-06  Kunlun
  13-07  Qilian_Shan      13-08  Inner_Tibet
  13-09  Hengduan_Shan_N

Sub-region codes (regions 14-15, geographic approximation):
  Hindu_Kush   Karakoram   W_Himalaya   W_Nepal
  C_Himalaya   E_Himalaya  Hengduan_Shan_S

Output:
  data/glacier_centroids_hma.parquet
    columns: RGIId, cenlat, cenlon, SMB_mwea, SMB_sigma,
             area_km2, zmean_m, zmin_m, zmax_m, slope_deg, aspect_deg,
             elev_range_m, subregion, rgi_region, perc_debris
"""

import os, ctypes, glob as _glob
# Fix: libproj.so requires CXXABI_1.3.15 not in system libstdc++.
# Preload the conda-bundled libstdc++ before importing pyproj.
_stdc_candidates = _glob.glob(
    "/home/chunlab/anaconda3/pkgs/libstdcxx-*/lib/libstdc++.so.6")
if _stdc_candidates:
    ctypes.CDLL(_stdc_candidates[0])
os.environ["PROJ_NETWORK"] = "OFF"

import pandas as pd
import numpy as np
from pathlib import Path
from pyproj import Transformer
from scipy.spatial import KDTree

# ── Paths ─────────────────────────────────────────────────────────────────────
PAPER_DIR  = Path(__file__).parent.parent
DATA_DIR   = PAPER_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

SHEAN_CSV  = (Path("/home/chunlab/Desktop/writing_paper/tajikistan/"
                   "revised TC paper/data_work/shean/mb/"
                   "hma_mb_20190214_1015_nmad.csv"))

RGI13_CSV  = (Path("/home/chunlab/Desktop/writing_paper/tajikistan/"
                   "revised TC paper/data_work/new_v7/"
                   "RGI2000-v7.0-G-13_central_asia(1)/"
                   "RGI2000-v7.0-G-13_central_asia-attributes.csv"))

# ── Quality filter thresholds ─────────────────────────────────────────────────
MIN_VALID_AREA_PERC = 60.0   # % of glacier area covered by observations
MIN_DT_YEARS        = 5.0    # minimum observation period
MAX_SMB_SIGMA       = 1.0    # maximum allowed uncertainty (m w.e. yr⁻¹)
MAX_ABS_SMB         = 5.0    # remove extreme outliers (data artefacts)

# ── Projection (validated in preliminary analysis) ────────────────────────────
# Azimuthal Equidistant centred at 35°N, 85°E — matches Shean et al. 2020
SHEAN_PROJ = "+proj=aeqd +lat_0=35 +lon_0=85 +datum=WGS84 +units=m"

print("=" * 70)
print("SCRIPT 02 — BUILD GLACIER CENTROID TABLE")
print("=" * 70)

# ════════════════════════════════════════════════════════════════════════════
# 1. LOAD AND FILTER SHEAN DATA
# ════════════════════════════════════════════════════════════════════════════
print("\n[1] Loading Shean geodetic MB ...")
shean = pd.read_csv(SHEAN_CSV)
print(f"    Raw rows: {len(shean):,}")
print(f"    Columns : {list(shean.columns)}")

# Standardise RGIId format: float 13.12345 → string 'RGI60-13.12345'
shean["RGIId_str"] = shean["RGIId"].apply(
    lambda x: f"RGI60-{x:.5f}".replace("RGI60-", "RGI60-")
              if not isinstance(x, str) else x
)
# Cleaner: keep numeric region for filtering
shean["rgi_region"] = shean["RGIId"].astype(str).str.split(".").str[0].astype(int)

print(f"\n    By RGI region:")
for r, grp in shean.groupby("rgi_region"):
    region_name = {13:"Central Asia", 14:"S Asia West", 15:"S Asia East"}.get(r, "Other")
    print(f"      Region {r:2d} ({region_name}): {len(grp):6,} glaciers")

# Apply quality filters
n0 = len(shean)
shean = shean[
    (shean["valid_area_perc"] >= MIN_VALID_AREA_PERC) &
    (shean["dt"]              >= MIN_DT_YEARS)        &
    (shean["mb_mwea_sigma"]   <  MAX_SMB_SIGMA)       &
    (shean["mb_mwea"].abs()   <  MAX_ABS_SMB)         &
    shean["x"].notna() & shean["y"].notna()
].copy()

print(f"\n    After quality filter: {len(shean):,} (removed {n0 - len(shean):,})")
print(f"    SMB range: {shean['mb_mwea'].min():.3f} to {shean['mb_mwea'].max():.3f} m w.e. yr⁻¹")
print(f"    SMB mean : {shean['mb_mwea'].mean():.3f} ± {shean['mb_mwea'].std():.3f}")

# ════════════════════════════════════════════════════════════════════════════
# 2. CONVERT x,y → lat/lon
# ════════════════════════════════════════════════════════════════════════════
print("\n[2] Converting projected coordinates → WGS84 lat/lon ...")
transformer = Transformer.from_crs(SHEAN_PROJ, "EPSG:4326", always_xy=True)
lons, lats   = transformer.transform(shean["x"].values, shean["y"].values)
shean["cenlat"] = lats
shean["cenlon"] = lons

# Sanity check: all coordinates should be within HMA bounds
lat_ok = (lats >= 24) & (lats <= 48)
lon_ok = (lons >= 63) & (lons <= 107)
n_bad  = (~(lat_ok & lon_ok)).sum()
print(f"    Coordinates outside HMA bounds: {n_bad} (should be near 0)")
if n_bad > 0:
    print(f"    WARNING: {n_bad} glaciers have coordinates outside expected HMA range")
    shean = shean[lat_ok & lon_ok].copy()
    print(f"    After removing: {len(shean):,} glaciers remain")

print(f"    Lat range: {shean['cenlat'].min():.2f} to {shean['cenlat'].max():.2f}°N")
print(f"    Lon range: {shean['cenlon'].min():.2f} to {shean['cenlon'].max():.2f}°E")

# ════════════════════════════════════════════════════════════════════════════
# 3. LOAD RGI v7 FOR REGION 13 — for authoritative sub-region assignment
# ════════════════════════════════════════════════════════════════════════════
print("\n[3] Loading RGI v7 region 13 for sub-region assignment ...")
rgi13 = pd.read_csv(RGI13_CSV)
print(f"    RGI v7 region 13: {len(rgi13):,} glaciers")
print(f"    o2region values : {sorted(rgi13['o2region'].unique())}")

# Official RGI v7 / RGI 6.0 sub-region names for region 13
# (from RGI technical report, Arendt et al. 2017)
# Notes on naming:
#   13-02 Pamir is split post-hoc into W_Pamir (lon < 73.5°E) / E_Pamir (lon >= 73.5°E)
#   13-03 W Tian Shan → labelled C_Tian_Shan (central/western, large; ~10k glaciers)
#   13-04 E Tian Shan → labelled N_Tian_Shan (northern/eastern, small; ~400 glaciers)
#   13-05 Karakoram → labelled Karakoram (consistent with regions 14 label)
O2REGION_NAMES_13 = {
    "13-01": "Pamir_Alay",
    "13-02": "Pamir",          # will be split into W_Pamir / E_Pamir below
    "13-03": "C_Tian_Shan",
    "13-04": "N_Tian_Shan",
    "13-05": "Karakoram",
    "13-06": "Kunlun",
    "13-07": "Qilian_Shan",
    "13-08": "Inner_Tibet",
    "13-09": "Hengduan_Shan_N",
}

# Build KD-tree on RGI v7 centroids (region 13) for nearest-neighbour lookup.
# Scale lon by cos(mean_lat) to approximate equal-area distances in degrees.
LAT_SCALE = np.cos(np.deg2rad(37.0))   # mean HMA latitude
rgi13_xy = np.column_stack([
    rgi13["cenlat"].values,
    rgi13["cenlon"].values * LAT_SCALE,
])
kd_r13 = KDTree(rgi13_xy)
print(f"    KD-tree built on {len(rgi13):,} RGI v7 region 13 centroids.")

def assign_subregion_r13_kdtree(lats, lons):
    """
    Assign RGI v7 o2region sub-region to Shean glaciers in region 13
    via nearest-neighbour lookup in the RGI v7 centroid table.
    Returns a list of sub-region name strings.
    """
    query_xy = np.column_stack([lats, lons * LAT_SCALE])
    _, idx = kd_r13.query(query_xy)
    o2codes = rgi13["o2region"].iloc[idx].values
    return [O2REGION_NAMES_13.get(c, "Unknown_R13") for c in o2codes]

# Geographic assignment for regions 14 and 15 (no RGI v7 files available).
# Boundaries are approximate and based on the published RGI 6.0 sub-region
# geographic descriptions (Pfeffer et al. 2014; Arendt et al. 2017).
def assign_subregion_r14(lat, lon):
    """Region 14: South Asia West."""
    if lon < 73.5:
        return "Hindu_Kush"        # 14-01: Hindu Kush
    elif lon < 79.0 and lat > 34.0:
        return "Karakoram"         # 14-02: Karakoram (south slopes)
    elif lon < 79.0:
        return "W_Himalaya"        # 14-03: W. Himalaya / Indus headwaters
    else:
        return "W_Himalaya"        # 14-03: extends east to ~lon 82

def assign_subregion_r15(lat, lon):
    """Region 15: South Asia East."""
    if lon < 84.0:
        return "W_Nepal"           # 15-01: W. Nepal / Karnali
    elif lon < 92.0:
        return "C_Himalaya"        # 15-02: Central Himalaya
    else:
        return "E_Himalaya"        # 15-03: Eastern Himalaya / Hengduan S.

# ════════════════════════════════════════════════════════════════════════════
# 4. BUILD FINAL CENTROID TABLE
# ════════════════════════════════════════════════════════════════════════════
print("\n[4] Building final centroid table ...")

cols_keep = [
    "RGIId", "RGIId_str", "rgi_region",
    "cenlat", "cenlon",
    "mb_mwea", "mb_mwea_sigma",
    "dhdt_ma", "dhdt_ma_sigma",
    "area_m2",
    "z_med", "z_min", "z_max", "z_slope", "z_aspect",
    "valid_area_perc", "dt", "t1", "t2",
    "perc_debris",
]
# Only keep columns that actually exist
cols_keep = [c for c in cols_keep if c in shean.columns]
centroids = shean[cols_keep].copy()

# Derived features
centroids["area_km2"]     = centroids["area_m2"] / 1e6
centroids["elev_range_m"] = centroids["z_max"] - centroids["z_min"]
centroids["log_area_km2"] = np.log10(centroids["area_km2"].clip(0.001))
centroids["aspect_sin"]   = np.sin(np.deg2rad(centroids["z_aspect"]))
centroids["aspect_cos"]   = np.cos(np.deg2rad(centroids["z_aspect"]))

# ── Sub-region assignment ──────────────────────────────────────────────────
# Region 13: authoritative KD-tree lookup against RGI v7 o2region
mask_r13 = centroids["rgi_region"] == 13
if mask_r13.any():
    sr_r13 = assign_subregion_r13_kdtree(
        centroids.loc[mask_r13, "cenlat"].values,
        centroids.loc[mask_r13, "cenlon"].values,
    )
    centroids.loc[mask_r13, "subregion"] = sr_r13
    print(f"\n    Region 13 sub-regions assigned via RGI v7 KD-tree "
          f"({mask_r13.sum():,} glaciers).")

# Post-process: split RGI-13 "Pamir" (13-02) into W_Pamir / E_Pamir by longitude.
# Threshold 73.5°E separates the Tajik (western) Pamir from the Chinese (eastern) Pamir.
pamir_mask = centroids["subregion"] == "Pamir"
if pamir_mask.any():
    centroids.loc[pamir_mask & (centroids["cenlon"] < 73.5), "subregion"] = "W_Pamir"
    centroids.loc[pamir_mask & (centroids["cenlon"] >= 73.5), "subregion"] = "E_Pamir"
    n_wp = (centroids["subregion"] == "W_Pamir").sum()
    n_ep = (centroids["subregion"] == "E_Pamir").sum()
    print(f"    Pamir split into W_Pamir ({n_wp:,}) / E_Pamir ({n_ep:,}) at lon=73.5°E.")

# Regions 14 and 15: geographic approximation
mask_r14 = centroids["rgi_region"] == 14
mask_r15 = centroids["rgi_region"] == 15
if mask_r14.any():
    centroids.loc[mask_r14, "subregion"] = centroids[mask_r14].apply(
        lambda r: assign_subregion_r14(r["cenlat"], r["cenlon"]), axis=1)
    print(f"    Region 14 sub-regions assigned via geographic bounds "
          f"({mask_r14.sum():,} glaciers).")
if mask_r15.any():
    centroids.loc[mask_r15, "subregion"] = centroids[mask_r15].apply(
        lambda r: assign_subregion_r15(r["cenlat"], r["cenlon"]), axis=1)
    print(f"    Region 15 sub-regions assigned via geographic bounds "
          f"({mask_r15.sum():,} glaciers).")

# Rename for consistency with existing pipeline
centroids = centroids.rename(columns={
    "mb_mwea":       "SMB_mwea",
    "mb_mwea_sigma": "SMB_sigma",
    "z_med":         "zmean_m",
    "z_min":         "zmin_m",
    "z_max":         "zmax_m",
    "z_slope":       "slope_deg",
    "z_aspect":      "aspect_deg",
})

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n    Final glacier centroid table:")
print(f"      Total glaciers : {len(centroids):,}")
print(f"      Columns        : {list(centroids.columns)}")
print(f"\n    By region:")
for r, grp in centroids.groupby("rgi_region"):
    name = {13:"Central Asia", 14:"S Asia West", 15:"S Asia East"}.get(r, "Other")
    print(f"      Region {r:2d} ({name:15s}): {len(grp):6,} glaciers  |  "
          f"SMB = {grp['SMB_mwea'].mean():.3f} ± {grp['SMB_mwea'].std():.3f}")

print(f"\n    By sub-region:")
for sr, grp in centroids.groupby("subregion"):
    print(f"      {sr:20s}: {len(grp):6,} glaciers  |  "
          f"SMB = {grp['SMB_mwea'].mean():.3f} ± {grp['SMB_mwea'].std():.3f}")

# ── Save ───────────────────────────────────────────────────────────────────
out_path = DATA_DIR / "glacier_centroids_hma.parquet"
centroids.to_parquet(out_path, index=False)
print(f"\n    Saved: {out_path}")
print(f"    File size: {out_path.stat().st_size / 1e6:.1f} MB")

print("\n" + "=" * 70)
print("Script 02 complete.")
print("Next step: run scripts/03_extract_era5_glaciers.py")
print("(Requires ERA5 files from script 01 to be downloaded first)")
print("=" * 70)
