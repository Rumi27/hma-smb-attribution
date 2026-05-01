"""
Script 11 — Lapse-Rate Sensitivity Analysis

Runs the full feature-matrix → XGBoost pipeline for three lapse rates:
  Γ = 5.0, 6.5 (default), 8.0 °C/km

Reports R², RMSE, and top-5 SHAP feature importances for each Γ.
Output: outputs/lapse_sensitivity.csv
"""
import pandas as pd, numpy as np, json
from pathlib import Path
import xarray as xr
from scipy.spatial import cKDTree
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import shap

DATA = Path(__file__).parent.parent / "data"
OUT  = Path(__file__).parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

print("=" * 70)
print("SCRIPT 11 — LAPSE-RATE SENSITIVITY (Γ = 5.0 / 6.5 / 8.0 °C/km)")
print("=" * 70)

# Load base data (no lapse correction yet)
climate = pd.read_parquet(DATA / "era5_climate_metrics_hma.parquet")
cents   = pd.read_parquet(DATA / "glacier_centroids_hma.parquet")
feat_base = cents.merge(climate, on="RGIId", how="inner")

# Quality filters (same as script 04)
feat_base = feat_base[
    (feat_base["SMB_sigma"]   < 1.0) &
    (feat_base["SMB_mwea"].abs() < 5.0) &
    feat_base["zmean_m"].notna() &
    feat_base["slope_deg"].notna() &
    feat_base["area_km2"].notna()
].copy()

# Base derived features (lapse-independent)
feat_base["log_area_km2"] = np.log10(feat_base["area_km2"].clip(0.001))
feat_base["continentality"] = feat_base["T_JJA_C"] - feat_base["T_DJF_C"]
denom = (feat_base["P_mean_mm"] * 12).clip(lower=1.0)
feat_base["precip_winter_frac"] = (feat_base["P_DJF_mm"] * 3 / denom).clip(0, 1)
feat_base["P_snow_mm"] = feat_base["P_DJF_mm"] + feat_base["P_MAM_mm"]
feat_base["P_trend_x_wfrac"] = feat_base["P_trend_mm_dec"] * feat_base["precip_winter_frac"]
feat_base["aspect_sin"] = np.sin(np.radians(feat_base["aspect_deg"])) if "aspect_sin" not in feat_base.columns else feat_base["aspect_sin"]
feat_base["aspect_cos"] = np.cos(np.radians(feat_base["aspect_deg"])) if "aspect_cos" not in feat_base.columns else feat_base["aspect_cos"]
feat_base["elev_range_m"] = feat_base["zmax_m"] - feat_base["zmin_m"] if "elev_range_m" not in feat_base.columns else feat_base["elev_range_m"]

# Load ERA5 orography
_orog_ds = xr.open_dataset(DATA / "era5" / "era5_hma_orography.nc")
_z_orog  = (_orog_ds["z"].squeeze() / 9.80665).values
_lats    = _orog_ds["latitude"].values
_lons    = _orog_ds["longitude"].values
_lon2d, _lat2d = np.meshgrid(_lons, _lats)
_pts  = np.column_stack([_lat2d.ravel(), _lon2d.ravel()])
_tree = cKDTree(_pts)
_, _idx = _tree.query(feat_base[["cenlat", "cenlon"]].values)
feat_base["z_era5_orog_m"] = _z_orog.ravel()[_idx]

EXCLUDE_BASE = ["RGIId","RGIId_str","SMB_mwea","SMB_sigma","rgi_region",
                "subregion","dhdt_ma","dhdt_ma_sigma","area_m2",
                "valid_area_perc","dt","t1","t2","perc_debris",
                "aspect_deg","zmax_m","zmin_m","cenlat","cenlon",
                "z_era5_orog_m"]

XGB_PARAMS = dict(n_estimators=600, learning_rate=0.04, max_depth=5,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                  reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1,
                  random_state=42, verbosity=0)

results = []

for gamma_km in [5.0, 6.5, 8.0]:
    LAPSE = gamma_km / 1000.0
    feat = feat_base.copy()
    feat["T_JJA_lapse_C"] = feat["T_JJA_C"]  - (feat["zmean_m"] - feat["z_era5_orog_m"]) * LAPSE
    feat["T_ann_lapse_C"] = feat["T_mean_C"] - (feat["zmean_m"] - feat["z_era5_orog_m"]) * LAPSE
    # cryo_balance with +10 offset (chosen for corrected T range; kept constant across sensitivity)
    feat["cryo_balance"] = feat["P_snow_mm"] / (feat["T_JJA_lapse_C"] + 10.0).clip(lower=0.1)
    feat["melt_index"]   = feat["T_JJA_lapse_C"] * np.sqrt(feat["area_km2"].clip(lower=0.001))

    FEATURE_COLS = [c for c in feat.columns
                    if c not in EXCLUDE_BASE and feat[c].dtype != object
                    and not feat[c].isna().any()]
    X = feat[FEATURE_COLS].values
    y = feat["SMB_mwea"].values
    regions = feat["subregion"].values

    kf = GroupKFold(n_splits=5)
    y_pred = np.zeros_like(y)
    fold_r2 = []
    for tr, te in kf.split(X, y, groups=regions):
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X[tr], y[tr])
        y_pred[te] = m.predict(X[te])
        fold_r2.append(r2_score(y[te], y_pred[te]))

    r2   = r2_score(y, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

    # SHAP on full dataset
    xgb_full = xgb.XGBRegressor(**XGB_PARAMS)
    xgb_full.fit(X, y)
    explainer = shap.TreeExplainer(xgb_full)
    shap_vals = explainer.shap_values(X)
    top5 = (pd.Series(np.abs(shap_vals).mean(axis=0), index=FEATURE_COLS)
              .sort_values(ascending=False).head(5))

    print(f"\n  Γ = {gamma_km:.1f} °C/km")
    print(f"    T_JJA_lapse mean = {feat['T_JJA_lapse_C'].mean():.2f} °C  "
          f"range [{feat['T_JJA_lapse_C'].min():.1f}, {feat['T_JJA_lapse_C'].max():.1f}]")
    print(f"    Spatial CV  R² = {r2:.3f}  RMSE = {rmse:.4f}")
    print(f"    Fold R²: {[round(v,3) for v in fold_r2]}  mean={np.mean(fold_r2):.3f}±{np.std(fold_r2):.3f}")
    print(f"    Top-5 SHAP: {dict(top5.round(5))}")

    results.append({
        "gamma_C_per_km": gamma_km,
        "T_JJA_lapse_mean_C": round(float(feat["T_JJA_lapse_C"].mean()), 2),
        "spatial_cv_r2": round(r2, 4),
        "spatial_cv_rmse": round(rmse, 4),
        "fold_r2_mean": round(float(np.mean(fold_r2)), 4),
        "fold_r2_std":  round(float(np.std(fold_r2)), 4),
        "fold_r2_values": [round(v, 4) for v in fold_r2],
        "top5_shap": top5.round(5).to_dict()
    })

df = pd.DataFrame(results)
df.to_csv(OUT / "lapse_sensitivity.csv", index=False)
print(f"\n  Saved: {OUT / 'lapse_sensitivity.csv'}")

# Summary comparison
print("\n  Summary:")
print(f"  {'Γ (°C/km)':>10}  {'R²':>6}  {'RMSE':>7}  {'fold R² mean±std':>20}")
for r in results:
    print(f"  {r['gamma_C_per_km']:>10.1f}  {r['spatial_cv_r2']:>6.3f}  "
          f"{r['spatial_cv_rmse']:>7.4f}  "
          f"{r['fold_r2_mean']:.3f}±{r['fold_r2_std']:.3f}")

(OUT / "lapse_sensitivity.json").write_text(json.dumps(results, indent=2))
print("=" * 70)
