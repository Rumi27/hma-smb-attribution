"""
Script 15 — σT_JJA Variance-Preservation Sensitivity for Projections

The delta bias-correction in script 08 creates a synthetic monthly series
where every year in the future window has the same temperature (ERA5 clim +
CMIP6 delta), yielding σT_JJA = 0 for all future projections.

This script tests the impact of preserving each glacier's historical
σT_JJA (from ERA5 2000-2018) in the projected feature set.

Method:
  After computing projected features via script 08's monthly_to_features(),
  replace the projected T_JJA_std_C with the ERA5 historical value for
  each glacier. Re-run the trained XGBoost predictor and compare regional
  mean SMB projections.

Output:
  outputs/smb_projections_variance_preserved.parquet
  outputs/smb_projections_comparison.csv
"""
import pandas as pd, numpy as np, json
from pathlib import Path
import xgboost as xgb

BASE    = Path(__file__).parent.parent
DATA    = BASE / "data"
OUT_DIR = BASE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("SCRIPT 15 — σT_JJA VARIANCE-PRESERVATION SENSITIVITY")
print("=" * 70)

feat    = pd.read_parquet(DATA / "feature_matrix_hma.parquet")
proj    = pd.read_parquet(OUT_DIR / "smb_projections_hma.parquet")
results = json.loads((OUT_DIR / "model_results_hma.json").read_text())
FEATURE_COLS = results["features"]

# Train full-dataset XGBoost
X_full = feat[FEATURE_COLS].values.astype(float)
y_full = feat["SMB_mwea"].values

print(f"  Training XGBoost on {len(X_full):,} glaciers ...")
xgb_model = xgb.XGBRegressor(
    n_estimators=600, learning_rate=0.04, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
    reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, random_state=42, verbosity=0)
xgb_model.fit(X_full, y_full)

# Historical σT_JJA per glacier (from training set)
sigma_hist = feat.set_index("RGIId")["T_JJA_std_C"].to_dict()

# For each CMIP6 scenario × window, recompute projections with variance preserved
SCENARIOS = ["ssp245", "ssp585"]
WINDOWS   = ["2020-2039", "2040-2059", "2060-2079", "2080-2099"]

# We need the per-glacier feature sets for each projection.
# Since script 08 doesn't save them, we re-derive a simplified version:
# For each glacier, substitute σT_JJA = historical ERA5 value in the projected feature set.
# The simplest approximation: take the historical features and add only the
# climatological shift in other features.

# For a clean sensitivity test without re-running the full delta pipeline,
# we compute: projected SMB (var-preserved) = model(X_proj with T_JJA_std_C
# replaced by ERA5 historical value).
# Since we don't have X_proj stored, we reconstruct from proj_df predictions:
# We know y_proj = model(X_proj). We want model(X_proj with sigma replaced).
# Approach: use feat as X_hist; apply only temperature and precipitation shifts
# from the delta correction. For simplicity, use feat (historical) with
# sigma already present and apply CMIP6 temperature mean shift only.

# Better approach: use the historical feature matrix but substitute the
# climatological temperature change (CMIP6 delta) while keeping sigma unchanged.
# This is an approximation of the variance-preserved projection.

print("\n  NOTE: Using historical feature matrix with CMIP6 temperature/precip")
print("  mean shifts applied, while retaining historical σT_JJA.")
print("  This approximates what a variance-preserving bias correction would give.")

# Load CMIP6 historical and future means for each model to get temperature delta
import xarray as xr
from scipy.spatial import cKDTree

CMIP6_DIR = DATA / "cmip6"
nc_files = sorted(CMIP6_DIR.glob("*.nc"))

# Parse available files
available = {}
for f in nc_files:
    parts = f.stem.split("_")
    var = parts[-1]
    for exp in ["historical", "ssp245", "ssp585"]:
        exp_parts = exp.split("_")
        n = len(exp_parts)
        if parts[-(1+n):-1] == exp_parts:
            model = "_".join(parts[:-(1+n)])
            available.setdefault(model, {}).setdefault(exp, {})[var] = f
            break

lats_all = feat["cenlat"].values
lons_all = feat["cenlon"].values
rgi_ids  = feat["RGIId"].values

def load_cmip6_monthly(nc_path):
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    tc  = next((c for c in ["time","valid_time"] if c in ds.coords), None)
    lat_c = next((c for c in ["lat","latitude"] if c in ds.coords), None)
    lon_c = next((c for c in ["lon","longitude"] if c in ds.coords), None)
    var  = list(ds.data_vars)[0]
    data = ds[var].values
    times = pd.to_datetime(ds[tc].values)
    lats = ds[lat_c].values
    lons = ds[lon_c].values
    ds.close()
    return times, lats, lons, data

LAPSE = 6.5 / 1000.0
_orog_ds = xr.open_dataset(DATA / "era5" / "era5_hma_orography.nc")
_z_orog  = (_orog_ds["z"].squeeze() / 9.80665).values
_lat_o   = _orog_ds["latitude"].values
_lon_o   = _orog_ds["longitude"].values
_lon2d, _lat2d = np.meshgrid(_lon_o, _lat_o)
_pts  = np.column_stack([_lat2d.ravel(), _lon2d.ravel()])
_tree = cKDTree(_pts)
_, _idx_orog = _tree.query(feat[["cenlat","cenlon"]].values)
z_era5_orog = _z_orog.ravel()[_idx_orog]

WINDOWS_YEARS = {"2020-2039":(2020,2039),"2040-2059":(2040,2059),
                 "2060-2079":(2060,2079),"2080-2099":(2080,2099)}

comparison_rows = []
proj_vp = proj[["RGIId","subregion","rgi_region","SMB_hist"]].copy()

for model_name, exps in available.items():
    if "historical" not in exps: continue
    hist_files = exps["historical"]
    t_key = next((k for k in hist_files if k in ("tas","near_surface_air_temperature")), None)
    p_key = next((k for k in hist_files if k in ("pr","precipitation")), None)
    if not t_key or not p_key: continue

    print(f"\n  Model: {model_name}")
    t_times_h, c_lats, c_lons, t_data_h = load_cmip6_monthly(hist_files[t_key])
    p_times_h, _, _, p_data_h = load_cmip6_monthly(hist_files[p_key])
    if t_data_h.mean() > 100: t_data_h -= 273.15
    if p_data_h.mean() < 1:   p_data_h *= 86400 * 30

    c_lat_idx = np.abs(c_lats[:,None] - lats_all[None,:]).argmin(axis=0)
    c_lon_idx = np.abs(c_lons[:,None] - lons_all[None,:]).argmin(axis=0)

    hist_T_clim = np.zeros((12, len(rgi_ids)))
    hist_P_clim = np.zeros((12, len(rgi_ids)))
    for m in range(1, 13):
        tm = np.array([t.month==m for t in t_times_h])
        pm = np.array([t.month==m for t in p_times_h])
        if tm.sum(): hist_T_clim[m-1] = t_data_h[tm].mean(axis=0)[c_lat_idx, c_lon_idx]
        if pm.sum(): hist_P_clim[m-1] = p_data_h[pm].mean(axis=0)[c_lat_idx, c_lon_idx]

    for scenario in SCENARIOS:
        if scenario not in exps: continue
        fut_files = exps[scenario]
        t_key_f = next((k for k in fut_files if k in ("tas","near_surface_air_temperature")), None)
        p_key_f = next((k for k in fut_files if k in ("pr","precipitation")), None)
        if not t_key_f or not p_key_f: continue

        t_times_f, _, _, t_data_f = load_cmip6_monthly(fut_files[t_key_f])
        p_times_f, _, _, p_data_f = load_cmip6_monthly(fut_files[p_key_f])
        if t_data_f.mean() > 100: t_data_f -= 273.15
        if p_data_f.mean() < 1:   p_data_f *= 86400 * 30

        for win_label, (y0, y1) in WINDOWS_YEARS.items():
            t_win = np.array([(y0 <= t.year <= y1) for t in t_times_f])
            p_win = np.array([(y0 <= t.year <= y1) for t in p_times_f])
            if t_win.sum() == 0: continue

            # Future climatological delta
            fut_T = np.zeros((12, len(rgi_ids)))
            fut_P = np.zeros((12, len(rgi_ids)))
            t_sub = t_data_f[t_win]; t_sub_times = t_times_f[t_win]
            p_sub = p_data_f[p_win]; p_sub_times = p_times_f[p_win]
            for m in range(1, 13):
                tm = np.array([t.month==m for t in t_sub_times])
                pm = np.array([t.month==m for t in p_sub_times])
                if tm.sum(): fut_T[m-1] = t_sub[tm].mean(axis=0)[c_lat_idx, c_lon_idx]
                if pm.sum(): fut_P[m-1] = p_sub[pm].mean(axis=0)[c_lat_idx, c_lon_idx]

            delta_T  = fut_T - hist_T_clim        # (12, n_glaciers)
            ratio_P  = np.where(hist_P_clim > 0.1, fut_P / hist_P_clim, 1.0)

            # Apply delta shifts to historical features
            # Seasonal T shifts: use mean delta over the relevant months
            jja_mask = np.array([6, 7, 8]) - 1    # indices for JJA
            djf_mask = np.array([12, 1, 2]) - 1
            djf_mask[djf_mask < 0] += 12
            mam_mask = np.array([3, 4, 5]) - 1
            son_mask = np.array([9, 10, 11]) - 1

            delta_T_ann = delta_T.mean(axis=0)
            delta_T_JJA = delta_T[jja_mask].mean(axis=0)
            delta_T_DJF = delta_T[djf_mask].mean(axis=0)
            delta_T_MAM = delta_T[mam_mask].mean(axis=0)
            delta_T_SON = delta_T[son_mask].mean(axis=0)

            ratio_P_ann = ratio_P.mean(axis=0)
            ratio_P_JJA = ratio_P[jja_mask].mean(axis=0)
            ratio_P_DJF = ratio_P[djf_mask].mean(axis=0)
            ratio_P_MAM = ratio_P[mam_mask].mean(axis=0)

            # Build projected feature set from historical features + delta
            X_proj = feat[FEATURE_COLS].values.copy().astype(float)
            fi = {f: i for i, f in enumerate(FEATURE_COLS)}

            # Apply shifts (column indices may vary by run; use dict)
            if "T_mean_C" in fi:   X_proj[:, fi["T_mean_C"]]  += delta_T_ann
            if "T_JJA_C"  in fi:   X_proj[:, fi["T_JJA_C"]]   += delta_T_JJA
            if "T_DJF_C"  in fi:   X_proj[:, fi["T_DJF_C"]]   += delta_T_DJF
            if "T_MAM_C"  in fi:   X_proj[:, fi["T_MAM_C"]]   += delta_T_MAM
            if "T_SON_C"  in fi:   X_proj[:, fi["T_SON_C"]]   += delta_T_SON
            if "P_mean_mm" in fi:  X_proj[:, fi["P_mean_mm"]] *= ratio_P_ann
            if "P_JJA_mm"  in fi:  X_proj[:, fi["P_JJA_mm"]]  *= ratio_P_JJA
            if "P_DJF_mm"  in fi:  X_proj[:, fi["P_DJF_mm"]]  *= ratio_P_DJF
            if "P_MAM_mm"  in fi:  X_proj[:, fi["P_MAM_mm"]]  *= ratio_P_MAM

            # Recompute derived features
            if "T_JJA_lapse_C" in fi:
                X_proj[:, fi["T_JJA_lapse_C"]] = (X_proj[:, fi["T_JJA_C"]]
                    - (feat["zmean_m"].values - z_era5_orog) * LAPSE)
            if "T_ann_lapse_C" in fi:
                X_proj[:, fi["T_ann_lapse_C"]] = (X_proj[:, fi["T_mean_C"]]
                    - (feat["zmean_m"].values - z_era5_orog) * LAPSE)
            if "continentality" in fi:
                X_proj[:, fi["continentality"]] = (X_proj[:, fi["T_JJA_C"]]
                                                   - X_proj[:, fi["T_DJF_C"]])
            if "P_snow_mm" in fi:
                X_proj[:, fi["P_snow_mm"]] = (X_proj[:, fi["P_DJF_mm"]]
                                               + X_proj[:, fi["P_MAM_mm"]])
            # NOTE: T_JJA_std_C is left at historical value (σ preserved!)
            # trends also left at historical (same limitation as script 08)

            y_vp = xgb_model.predict(X_proj)

            # Also get the original (zero-sigma) projection
            col_orig = f"SMB_{scenario}_{win_label}_ensmean"
            y_orig = proj[col_orig].values if col_orig in proj.columns else None

            col_vp = f"SMB_vp_{scenario}_{win_label}_{model_name}"
            proj_vp[col_vp] = y_vp

            if y_orig is not None:
                delta_orig = y_orig.mean() - proj["SMB_hist"].mean()
                delta_vp   = y_vp.mean()   - proj["SMB_hist"].mean()
                print(f"    {scenario} {win_label}: orig ΔSMB={delta_orig:.4f}  "
                      f"var-pres ΔSMB={delta_vp:.4f}  diff={delta_vp-delta_orig:.4f}")
                comparison_rows.append({
                    "model": model_name, "scenario": scenario, "window": win_label,
                    "delta_smb_original":  round(delta_orig, 4),
                    "delta_smb_var_pres":  round(delta_vp,   4),
                    "difference":          round(delta_vp - delta_orig, 4),
                })

# Regional summary comparison
print("\n  Regional mean SMB: variance-preserved vs original (2080-2099 SSP5-8.5)")
print(f"  {'Sub-region':<25} {'Orig ΔSMB':>10} {'VarPres ΔSMB':>13} {'Diff':>7}")
print("-" * 60)
for reg, grp in proj_vp.groupby("subregion"):
    hist_m = grp["SMB_hist"].mean()
    orig_col = "SMB_ssp585_2080-2099_ensmean"
    vp_cols  = [c for c in proj_vp.columns if "vp_ssp585_2080" in c]
    if not vp_cols or orig_col not in proj.columns: continue
    vp_m = grp[vp_cols].mean(axis=1).mean()
    orig_m = proj.loc[proj["subregion"]==reg, orig_col].mean()
    print(f"  {reg:<25} {orig_m - hist_m:>10.3f} {vp_m - hist_m:>13.3f} {vp_m - orig_m:>7.3f}")

pd.DataFrame(comparison_rows).to_csv(OUT_DIR / "smb_projections_comparison.csv", index=False)
print(f"\n  Saved: smb_projections_comparison.csv")
print("=" * 70)
