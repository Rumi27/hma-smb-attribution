"""
Script 17 — Cross-Validated SHAP Attribution (Q4, pot_com5)

Computes SHAP values in a strict cross-validated manner: for each CV fold,
train XGBoost on the training fold only and compute SHAP for the held-out
test glaciers. Aggregates all out-of-fold SHAP values to report honest
attribution under spatial CV.

Outputs:
  outputs/cv_shap_values.parquet        — per-glacier out-of-fold SHAP values
  outputs/cv_shap_importance.csv        — mean |SHAP| ranked (CV vs full-data)
"""
import pandas as pd, numpy as np, json
from pathlib import Path
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import shap

BASE = Path(__file__).parent.parent
DATA = BASE / "data"
OUT  = BASE / "outputs"
OUT.mkdir(exist_ok=True)

print("=" * 70)
print("SCRIPT 17 — CROSS-VALIDATED SHAP ATTRIBUTION")
print("=" * 70)

feat    = pd.read_parquet(DATA / "feature_matrix_hma.parquet")
results = json.loads((OUT / "model_results_hma.json").read_text())
FEATURE_COLS = results["features"]

X       = feat[FEATURE_COLS].values.astype(float)
y       = feat["SMB_mwea"].values
regions = feat["subregion"].values

XGB_PARAMS = dict(n_estimators=600, learning_rate=0.04, max_depth=5,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                  reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1,
                  random_state=42, verbosity=0)

kf = GroupKFold(n_splits=5)

cv_shap = np.zeros_like(X)
cv_bias = np.zeros(len(X))

print(f"\n  Running {kf.n_splits}-fold CV SHAP ({len(FEATURE_COLS)} features, "
      f"{len(X):,} glaciers) ...")

for fold_i, (tr, te) in enumerate(kf.split(X, y, groups=regions)):
    print(f"  Fold {fold_i+1}/{kf.n_splits}: train={len(tr):,}  test={len(te):,} "
          f"[{', '.join(np.unique(regions[te])[:3])}...]")
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(X[tr], y[tr])
    expl = shap.TreeExplainer(m)
    sv   = expl.shap_values(X[te])
    cv_shap[te] = sv
    cv_bias[te] = expl.expected_value

# Save per-glacier SHAP
shap_df = pd.DataFrame(cv_shap, columns=FEATURE_COLS)
shap_df.insert(0, "RGIId",     feat["RGIId"].values)
shap_df.insert(1, "subregion", feat["subregion"].values)
shap_df.insert(2, "shap_bias", cv_bias)
shap_df.to_parquet(OUT / "cv_shap_values.parquet", index=False)

# Ranked importance
full_shap = pd.read_csv(OUT / "shap_importance_hma.csv")  # full-data SHAP
mean_abs_cv = np.abs(cv_shap).mean(axis=0)

imp = pd.DataFrame({
    "feature":       FEATURE_COLS,
    "mean_abs_shap_cv":   mean_abs_cv.round(6),
    "mean_abs_shap_full": full_shap.set_index("feature")["mean_abs_shap"].reindex(FEATURE_COLS).values,
}).sort_values("mean_abs_shap_cv", ascending=False)

print("\n  Top 15 — CV vs Full-data SHAP ranking:")
print(f"  {'Feature':30s} {'CV |SHAP|':>10} {'Full |SHAP|':>12} {'Rank shift':>11}")
full_rank = {r["feature"]: i+1 for i, r in full_shap.iterrows()}
for rank, (_, row) in enumerate(imp.head(15).iterrows(), 1):
    shift = full_rank.get(row["feature"], 99) - rank
    arrow = f"+{shift}" if shift > 0 else str(shift) if shift < 0 else "="
    print(f"  {row['feature']:30s} {row['mean_abs_shap_cv']:>10.5f} "
          f"{row['mean_abs_shap_full']:>12.5f}  {arrow:>10}")

imp.to_csv(OUT / "cv_shap_importance.csv", index=False)
print(f"\n  Saved: cv_shap_values.parquet  cv_shap_importance.csv")
print("=" * 70)
