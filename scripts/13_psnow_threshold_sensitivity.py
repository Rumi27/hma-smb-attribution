"""
Script 13 — Temperature-Threshold P_snow Sensitivity

Replaces the calendar-based P_snow = P_DJF + P_MAM with a
temperature-threshold version: P_snow_thresh = sum of monthly P
where monthly T < 2 °C (a standard degree-day threshold).

Reports:
  1. Correlation between calendar-based and T-threshold P_snow
  2. Spatial CV R² and RMSE with T-threshold P_snow replacing calendar P_snow
  3. Top SHAP feature changes

Output:
  outputs/psnow_threshold_sensitivity.csv
"""
import pandas as pd, numpy as np, json
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

BASE = Path(__file__).parent.parent
DATA = BASE / "data"
OUT  = BASE / "outputs"
OUT.mkdir(exist_ok=True)

print("=" * 70)
print("SCRIPT 13 — P_snow TEMPERATURE-THRESHOLD SENSITIVITY")
print("=" * 70)

# ── Load monthly climate data ─────────────────────────────────────────────────
monthly = pd.read_parquet(DATA / "monthly_climate_hma.parquet")
feat    = pd.read_parquet(DATA / "feature_matrix_hma.parquet")
results = json.loads((OUT / "model_results_hma.json").read_text())
FEATURE_COLS = results["features"]

print(f"  Monthly climate: {len(monthly):,} rows  ({monthly['RGIId'].nunique():,} unique glaciers)")

# ── Compute T-threshold P_snow per glacier ────────────────────────────────────
# P_snow_thresh = total monthly P where T < 2°C, expressed as mm/month average
T_THRESH = 2.0   # °C

monthly_snow = monthly[monthly["Temp_C"] < T_THRESH].copy()
psnow_thresh = (monthly_snow.groupby("RGIId")["Precip_mm"]
                             .mean()       # mean monthly accumulation-season P
                             .rename("P_snow_thresh_mm")
                             .reset_index())

print(f"  Months with T < {T_THRESH}°C: {len(monthly_snow):,} / {len(monthly):,} "
      f"({100*len(monthly_snow)/len(monthly):.1f}%)")

# Merge into feature matrix
feat2 = feat.merge(psnow_thresh, on="RGIId", how="left")
feat2["P_snow_thresh_mm"] = feat2["P_snow_thresh_mm"].fillna(0.0)

# Correlation with calendar-based
from scipy.stats import pearsonr, spearmanr
r_pearson, _ = pearsonr(feat2["P_snow_mm"], feat2["P_snow_thresh_mm"])
r_spearman, _ = spearmanr(feat2["P_snow_mm"], feat2["P_snow_thresh_mm"])
print(f"\n  Pearson r(calendar P_snow, T-thresh P_snow) = {r_pearson:.3f}")
print(f"  Spearman r                                 = {r_spearman:.3f}")
diff = feat2["P_snow_thresh_mm"] - feat2["P_snow_mm"]
print(f"  Mean difference (thresh - calendar): {diff.mean():.2f} mm/month")
print(f"  Std difference                     : {diff.std():.2f} mm/month")
print(f"  % glaciers with |diff| > 10 mm/month: {(np.abs(diff) > 10).mean()*100:.1f}%")

# ── Build modified feature set: replace P_snow_mm with P_snow_thresh_mm ──────
FEATURE_THRESH = [c if c != "P_snow_mm" else "P_snow_thresh_mm"
                  for c in FEATURE_COLS]
# Also replace inside cryo_balance (requires recomputing it)
feat2["cryo_balance_thresh"] = (feat2["P_snow_thresh_mm"]
                                / (feat2["T_JJA_lapse_C"] + 10.0).clip(lower=0.1))

# Replace cryo_balance with cryo_balance_thresh
FEATURE_THRESH2 = [c if c != "cryo_balance" else "cryo_balance_thresh"
                   for c in FEATURE_THRESH]

# Remove duplicates that don't exist
missing = [c for c in FEATURE_THRESH2 if c not in feat2.columns]
print(f"\n  Missing features: {missing}")

FINAL_FEATS = [c for c in FEATURE_THRESH2 if c in feat2.columns and not feat2[c].isna().any()]
print(f"  Features for T-threshold model: {len(FINAL_FEATS)}")

X_thresh = feat2[FINAL_FEATS].values.astype(float)
X_base   = feat2[FEATURE_COLS].values.astype(float)
y        = feat2["SMB_mwea"].values
regions  = feat2["subregion"].values

XGB_PARAMS = dict(n_estimators=600, learning_rate=0.04, max_depth=5,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                  reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1,
                  random_state=42, verbosity=0)

kf = GroupKFold(n_splits=5)

results_summary = []
for label, Xc, fc in [("Calendar P_snow (baseline)", X_base, FEATURE_COLS),
                       ("T-threshold P_snow (<2 degC)", X_thresh, FINAL_FEATS)]:
    y_pred = np.zeros_like(y)
    for tr, te in kf.split(Xc, y, groups=regions):
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(Xc[tr], y[tr])
        y_pred[te] = m.predict(Xc[te])
    r2   = r2_score(y, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    print(f"\n  {label}")
    print(f"    Spatial CV: R²={r2:.3f}  RMSE={rmse:.4f}")
    results_summary.append({"method": label, "r2": round(r2, 4), "rmse": round(rmse, 4)})

pd.DataFrame(results_summary).to_csv(OUT / "psnow_threshold_sensitivity.csv", index=False)
print(f"\n  Saved: {OUT / 'psnow_threshold_sensitivity.csv'}")
print("=" * 70)
