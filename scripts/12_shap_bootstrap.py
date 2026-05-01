"""
Script 12 — SHAP Bootstrap Stability Analysis

Trains 30 bootstrap XGBoost models on the full dataset.
Reports mean ± SD of mean |SHAP| for the top 15 features to assess
ranking stability under sampling uncertainty.

Outputs:
  outputs/shap_bootstrap_stability.csv   — mean/std/cv per feature
  outputs/shap_bootstrap_raw.parquet     — all 30 × n_features values
"""
import pandas as pd, numpy as np, json
from pathlib import Path
import xgboost as xgb
import shap

BASE = Path(__file__).parent.parent
DATA = BASE / "data"
OUT  = BASE / "outputs"
OUT.mkdir(exist_ok=True)

print("=" * 70)
print("SCRIPT 12 — SHAP BOOTSTRAP STABILITY (30 resamples)")
print("=" * 70)

feat    = pd.read_parquet(DATA / "feature_matrix_hma.parquet")
results = json.loads((OUT / "model_results_hma.json").read_text())
FEATURE_COLS = results["features"]

X = feat[FEATURE_COLS].values.astype(float)
y = feat["SMB_mwea"].values

XGB_PARAMS = dict(n_estimators=600, learning_rate=0.04, max_depth=5,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                  reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1,
                  random_state=42, verbosity=0)

N_BOOT = 30
rng    = np.random.default_rng(seed=0)

boot_shap = np.zeros((N_BOOT, len(FEATURE_COLS)))

for b in range(N_BOOT):
    idx = rng.choice(len(X), size=len(X), replace=True)
    Xb, yb = X[idx], y[idx]
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(Xb, yb)
    expl  = shap.TreeExplainer(model)
    sv    = expl.shap_values(X)           # evaluate on full (unbootstrapped) set
    boot_shap[b] = np.abs(sv).mean(axis=0)
    if (b + 1) % 5 == 0:
        print(f"  Bootstrap {b+1:2d}/{N_BOOT} done")

# Stability stats
mean_shap = boot_shap.mean(axis=0)
std_shap  = boot_shap.std(axis=0)
cv_shap   = std_shap / (mean_shap + 1e-9)   # coefficient of variation

stab = pd.DataFrame({
    "feature": FEATURE_COLS,
    "mean_abs_shap":  mean_shap.round(6),
    "std_abs_shap":   std_shap.round(6),
    "cv_pct":         (cv_shap * 100).round(1),
}).sort_values("mean_abs_shap", ascending=False)

print("\n  Top 15 features (mean ± std of mean |SHAP| across 30 bootstraps):")
for _, row in stab.head(15).iterrows():
    bar = "█" * int(row["mean_abs_shap"] / stab["mean_abs_shap"].max() * 25)
    print(f"    {row['feature']:30s}: {row['mean_abs_shap']:.5f} ± {row['std_abs_shap']:.5f}"
          f"  CV={row['cv_pct']:.1f}%  {bar}")

stab.to_csv(OUT / "shap_bootstrap_stability.csv", index=False)

raw = pd.DataFrame(boot_shap, columns=FEATURE_COLS)
raw.to_parquet(OUT / "shap_bootstrap_raw.parquet", index=False)
print(f"\n  Saved: shap_bootstrap_stability.csv")
print(f"  Saved: shap_bootstrap_raw.parquet")
print("=" * 70)
