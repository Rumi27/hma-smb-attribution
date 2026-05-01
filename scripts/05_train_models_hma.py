"""
Script 05 — Train Models + SHAP on Full HMA Dataset

Models: Ridge (baseline), Random Forest, XGBoost (primary)
Validation: 5-fold spatial cross-validation stratified by sub-region
SHAP: TreeExplainer on full-dataset XGBoost

Outputs:
  outputs/model_results_hma.json
  outputs/cv_predictions_hma.parquet
  outputs/shap_values_hma.parquet
  outputs/shap_importance_hma.csv
  outputs/model_comparison_hma.csv
"""
import pandas as pd, numpy as np, json
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import shap

DATA = Path(__file__).parent.parent / "data"
OUT  = Path(__file__).parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

print("=" * 70)
print("SCRIPT 05 — TRAIN MODELS + SHAP (FULL HMA)")
print("=" * 70)

feat = pd.read_parquet(DATA / "feature_matrix_hma.parquet")
print(f"\n  Dataset: {len(feat):,} glaciers, {feat.shape[1]} columns")

EXCLUDE = ["RGIId","RGIId_str","SMB_mwea","SMB_sigma","rgi_region",
           "subregion","dhdt_ma","dhdt_ma_sigma","area_m2",
           "valid_area_perc","dt","t1","t2","perc_debris",
           "aspect_deg","zmax_m","zmin_m",
           "cenlat","cenlon",
           "z_era5_orog_m"]    # intermediate for lapse correction only, not a model feature
FEATURE_COLS = [c for c in feat.columns
                if c not in EXCLUDE and feat[c].dtype != object
                and not feat[c].isna().any()]

TARGET = "SMB_mwea"
X = feat[FEATURE_COLS].values
y = feat[TARGET].values
regions = feat["subregion"].values

print(f"  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"  Target range: {y.min():.3f} to {y.max():.3f}  mean={y.mean():.3f}")

def metrics(y_true, y_pred, label):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    print(f"  {label:22s}: R²={r2:.3f}  RMSE={rmse:.4f}  MAE={mae:.4f}  bias={bias:.4f}")
    return {"r2":round(r2,4),"rmse":round(rmse,4),"mae":round(mae,4),"bias":round(bias,4)}

# Spatial CV: hold out complete sub-regions so train/test never share the same
# geographic unit.  GroupKFold assigns each unique subregion label to exactly
# one fold — no spatial leakage.
kf = GroupKFold(n_splits=5)

unique_regs = np.unique(regions)
print(f"  Spatial CV: {len(unique_regs)} sub-regions → 5 folds (each fold holds out "
      f"~{len(unique_regs)//5} sub-regions)")

# ── Ridge ─────────────────────────────────────────────────────────────────────
print("\n[1] Ridge Regression ...")
lr_pipe = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))])
y_lr = np.zeros_like(y)
for tr, te in kf.split(X, y, groups=regions):
    lr_pipe.fit(X[tr], y[tr]); y_lr[te] = lr_pipe.predict(X[te])
m_lr = metrics(y, y_lr, "Ridge")

# ── Random Forest ─────────────────────────────────────────────────────────────
print("\n[2] Random Forest ...")
rf = RandomForestRegressor(n_estimators=300, max_depth=10,
                            min_samples_leaf=20, max_features=0.6,
                            n_jobs=-1, random_state=42)
y_rf = np.zeros_like(y)
for tr, te in kf.split(X, y, groups=regions):
    rf.fit(X[tr], y[tr]); y_rf[te] = rf.predict(X[te])
m_rf = metrics(y, y_rf, "Random Forest")

# ── XGBoost ───────────────────────────────────────────────────────────────────
print("\n[3] XGBoost ...")
xgb_cv = xgb.XGBRegressor(n_estimators=600, learning_rate=0.04, max_depth=5,
                            subsample=0.8, colsample_bytree=0.8,
                            min_child_weight=20, reg_alpha=0.1, reg_lambda=1.0,
                            n_jobs=-1, random_state=42, verbosity=0)
y_xgb = np.zeros_like(y)
fold_r2 = []
fold_regions_held = []
for fold_i, (tr, te) in enumerate(kf.split(X, y, groups=regions)):
    xgb_cv.fit(X[tr], y[tr])
    y_xgb[te] = xgb_cv.predict(X[te])
    fold_r2.append(r2_score(y[te], y_xgb[te]))
    held = np.unique(regions[te])
    fold_regions_held.append(held)
    print(f"    Fold {fold_i+1}: held-out regions={list(held)}  R²={fold_r2[-1]:.3f}")
m_xgb = metrics(y, y_xgb, "XGBoost")
print(f"    Fold R² range: {min(fold_r2):.3f} – {max(fold_r2):.3f}  "
      f"mean={np.mean(fold_r2):.3f}  std={np.std(fold_r2):.3f}")

# ── Per-region XGBoost ────────────────────────────────────────────────────────
print("\n  XGBoost by sub-region:")
region_metrics = {}
for reg in np.unique(regions):
    mask = regions == reg
    if mask.sum() < 50: continue
    rm = metrics(y[mask], y_xgb[mask], f"  {reg}")
    region_metrics[reg] = rm

# ── SHAP ─────────────────────────────────────────────────────────────────────
print("\n[4] SHAP (full-dataset XGBoost) ...")
xgb_full = xgb.XGBRegressor(n_estimators=600, learning_rate=0.04, max_depth=5,
                              subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=20, reg_alpha=0.1, reg_lambda=1.0,
                              n_jobs=-1, random_state=42, verbosity=0)
xgb_full.fit(X, y)

explainer = shap.TreeExplainer(xgb_full)
shap_vals  = explainer.shap_values(X)

shap_df = pd.DataFrame(shap_vals, columns=FEATURE_COLS)
shap_df["RGIId"]       = feat["RGIId"].values
shap_df["subregion"]   = feat["subregion"].values
shap_df["rgi_region"]  = feat["rgi_region"].values
shap_df["SMB_mwea"]    = y
shap_df["y_pred_xgb"]  = xgb_full.predict(X)
shap_df.to_parquet(OUT / "shap_values_hma.parquet", index=False)

mean_abs_shap = (pd.Series(np.abs(shap_vals).mean(axis=0), index=FEATURE_COLS)
                   .sort_values(ascending=False))
print("\n  Top 12 features by mean |SHAP|:")
for f, v in mean_abs_shap.head(12).items():
    print(f"    {f:30s}: {v:.5f}")

mean_abs_shap.reset_index().rename(columns={"index":"feature"}).to_csv(
    OUT / "shap_importance_hma.csv", index=False)

# ── Grouped SHAP analysis ─────────────────────────────────────────────────────
# Addresses collinearity: zmean_m enters the model five times (directly plus
# inside four engineered features).  Summing |SHAP| within conceptual groups
# gives the true total attribution share of each physical driver cluster.
print("\n[5] Grouped SHAP analysis (resolves elevation-variability collinearity) ...")

SHAP_GROUPS = {
    "Elevation":          ["zmean_m", "T_JJA_lapse_C", "T_ann_lapse_C",
                           "melt_index", "cryo_balance"],
    "JJA_T_Variability":  ["T_JJA_std_C", "T_std_C", "T_JJA_max_C"],
    "JJA_Precipitation":  ["P_JJA_mm", "P_std_mm"],
    "Precip_Seasonality": ["precip_winter_frac", "P_snow_mm",
                           "P_DJF_mm", "P_MAM_mm", "P_trend_x_wfrac",
                           "P_SON_mm"],
    "Temperature_Trend":  ["T_trend_C_dec", "T_JJA_trend_C_dec"],
    "Continentality":     ["continentality"],
    "Mean_Temperature":   ["T_mean_C", "T_JJA_C", "T_DJF_C",
                           "T_MAM_C", "T_SON_C", "T_DJF_min_C"],
    "Precip_Mean":        ["P_mean_mm", "P_trend_mm_dec"],
    "Glacier_Morphology": ["slope_deg", "area_km2", "log_area_km2",
                           "elev_range_m", "aspect_sin", "aspect_cos"],
}

# Sum mean |SHAP| over group members that exist in FEATURE_COLS
shap_abs = np.abs(shap_vals)
grouped_shap = {}
for group, members in SHAP_GROUPS.items():
    present = [m for m in members if m in FEATURE_COLS]
    idxs    = [FEATURE_COLS.index(m) for m in present]
    grouped_shap[group] = float(shap_abs[:, idxs].mean(axis=0).sum()) if idxs else 0.0

grouped_series = pd.Series(grouped_shap).sort_values(ascending=False)
print("\n  Grouped SHAP (sum of mean |SHAP| within conceptual cluster):")
for g, v in grouped_series.items():
    bar = "█" * int(v / grouped_series.max() * 30)
    print(f"    {g:25s}: {v:.5f}  {bar}")

grouped_df = grouped_series.reset_index()
grouped_df.columns = ["group", "grouped_mean_abs_shap"]
grouped_df.to_csv(OUT / "shap_grouped_hma.csv", index=False)
print(f"\n  Saved: {OUT / 'shap_grouped_hma.csv'}")

# ── Random CV vs spatial CV comparison ───────────────────────────────────────
print("\n[6] Random CV vs Spatial CV comparison (spatial leakage quantification) ...")
from sklearn.model_selection import KFold

kf_random = KFold(n_splits=5, shuffle=True, random_state=42)
y_xgb_random = np.zeros_like(y)
for tr, te in kf_random.split(X):
    xgb_tmp = xgb.XGBRegressor(n_estimators=600, learning_rate=0.04, max_depth=5,
                                subsample=0.8, colsample_bytree=0.8,
                                min_child_weight=20, reg_alpha=0.1, reg_lambda=1.0,
                                n_jobs=-1, random_state=42, verbosity=0)
    xgb_tmp.fit(X[tr], y[tr])
    y_xgb_random[te] = xgb_tmp.predict(X[te])

r2_random_cv = r2_score(y, y_xgb_random)
rmse_random  = float(np.sqrt(mean_squared_error(y, y_xgb_random)))
print(f"\n  Random 5-fold CV  : R²={r2_random_cv:.3f}  RMSE={rmse_random:.4f}")
print(f"  Spatial 5-fold CV : R²={m_xgb['r2']:.3f}  RMSE={m_xgb['rmse']:.4f}")
print(f"  Spatial leakage inflation: {r2_random_cv/m_xgb['r2']:.1f}× (random / spatial R²)")

# ── Save all ──────────────────────────────────────────────────────────────────
cv = feat[["RGIId","subregion","rgi_region","SMB_mwea","SMB_sigma"]].copy()
cv["pred_lr"] = y_lr; cv["pred_rf"] = y_rf; cv["pred_xgb"] = y_xgb
cv.to_parquet(OUT / "cv_predictions_hma.parquet", index=False)

comp = pd.DataFrame({"Model":["Ridge","Random Forest","XGBoost"],
                     "R2":[m_lr["r2"],m_rf["r2"],m_xgb["r2"]],
                     "RMSE":[m_lr["rmse"],m_rf["rmse"],m_xgb["rmse"]],
                     "MAE":[m_lr["mae"],m_rf["mae"],m_xgb["mae"]]})
comp.to_csv(OUT / "model_comparison_hma.csv", index=False)
print("\n  Model comparison:"); print(comp.to_string(index=False))

results = {"n_glaciers":len(feat),"n_features":len(FEATURE_COLS),
           "features":FEATURE_COLS,
           "cv_type": "spatial_GroupKFold_by_subregion",
           "cv_folds": 5,
           "xgb_fold_r2": [round(v,4) for v in fold_r2],
           "xgb_fold_r2_mean": round(float(np.mean(fold_r2)),4),
           "xgb_fold_r2_std":  round(float(np.std(fold_r2)),4),
           "metrics":{"ridge":m_lr,"random_forest":m_rf,"xgboost":m_xgb},
           "random_cv_r2":  round(r2_random_cv, 4),
           "random_cv_rmse": round(rmse_random, 4),
           "spatial_leakage_factor": round(r2_random_cv / m_xgb["r2"], 2),
           "xgb_by_subregion":region_metrics,
           "shap_top10":mean_abs_shap.head(10).to_dict(),
           "shap_grouped":grouped_shap}
(OUT / "model_results_hma.json").write_text(json.dumps(results, indent=2))

print("\n  All outputs saved to:", OUT)
print("  Next: run scripts/06_generate_figures_hma.py")
print("=" * 70)
