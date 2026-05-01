"""
Script 19 — OOD Threshold Sensitivity + RF Permutation Importance
(Q6, Q8 — pot_com5)

Q6: Compute per-sub-region OOD fractions at four Mahalanobis D² thresholds
    (90th, 95th, 97.5th, 99th percentiles of training D²).

Q8: Compute permutation importance for Random Forest (out-of-fold, so no leakage)
    and compare top-10 ranking to XGBoost SHAP.

Outputs:
  outputs/ood_threshold_sensitivity.csv
  outputs/rf_permutation_importance.csv
"""
import pandas as pd, numpy as np, json
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

BASE = Path(__file__).parent.parent
DATA = BASE / "data"
OUT  = BASE / "outputs"

print("=" * 70)
print("SCRIPT 19 — OOD THRESHOLD SENSITIVITY + RF PERMUTATION IMPORTANCE")
print("=" * 70)

feat    = pd.read_parquet(DATA / "feature_matrix_hma.parquet")
ood_df  = pd.read_parquet(OUT / "ood_mahalanobis_per_glacier.parquet")
results = json.loads((OUT / "model_results_hma.json").read_text())
FEATURE_COLS = results["features"]
shap_cv = pd.read_csv(OUT / "cv_shap_importance.csv")

X       = feat[FEATURE_COLS].values.astype(float)
y       = feat["SMB_mwea"].values
regions = feat["subregion"].values
mah2    = ood_df["mah2"].values

# ── Q6: OOD threshold sensitivity ─────────────────────────────────────────────
print("\n  Q6: OOD threshold sensitivity ...")
THRESHOLDS_PCT = [90, 95, 97.5, 99]
threshold_vals = {p: np.percentile(mah2, p) for p in THRESHOLDS_PCT}
print(f"\n  Threshold D² values: "
      + "  ".join(f"{p}th={v:.1f}" for p, v in threshold_vals.items()))

rows_ood = []
header = f"  {'Sub-region':<25}" + "".join(f" {'D²>'+str(p)+'th':>10}" for p in THRESHOLDS_PCT)
print(f"\n{header}")
print("  " + "-" * (25 + 10 * len(THRESHOLDS_PCT)))
for reg in sorted(np.unique(regions)):
    mask = regions == reg
    row  = {"subregion": reg}
    vals = []
    for p in THRESHOLDS_PCT:
        frac = 100 * (mah2[mask] > threshold_vals[p]).mean()
        row[f"ood_pct_{p}th"] = round(frac, 1)
        vals.append(f"{frac:>10.1f}")
    rows_ood.append(row)
    print(f"  {reg:<25}" + "".join(vals))

# HMA-wide
row_all = {"subregion": "HMA-wide"}
for p in THRESHOLDS_PCT:
    row_all[f"ood_pct_{p}th"] = round(100 * (mah2 > threshold_vals[p]).mean(), 1)
rows_ood.append(row_all)

pd.DataFrame(rows_ood).to_csv(OUT / "ood_threshold_sensitivity.csv", index=False)

# Key check: does masking threshold change ΔSMB conclusion for E/C Himalaya?
proj = pd.read_parquet(OUT / "smb_projections_hma.parquet")
print("\n  OOD threshold impact on projection conclusions:")
print(f"  {'Threshold':>12} {'E_Himal OOD%':>14} {'C_Himal OOD%':>14} "
      f"{'Karakoram OOD%':>16}")
for p in THRESHOLDS_PCT:
    col = f"ood_pct_{p}th"
    df  = pd.DataFrame(rows_ood).set_index("subregion")
    e   = df.loc["E_Himalaya", col] if "E_Himalaya" in df.index else np.nan
    c   = df.loc["C_Himalaya", col] if "C_Himalaya" in df.index else np.nan
    k   = df.loc["Karakoram",  col] if "Karakoram"  in df.index else np.nan
    print(f"  D²>{p:4.1f}th pct {e:>14.1f} {c:>14.1f} {k:>16.1f}")

# ── Q8: RF permutation importance ─────────────────────────────────────────────
print("\n  Q8: RF permutation importance (out-of-fold) ...")
kf = GroupKFold(n_splits=5)

RF_PARAMS = dict(n_estimators=300, max_features=0.6, min_samples_leaf=20,
                 n_jobs=-1, random_state=42)

perm_imp_accum = np.zeros(len(FEATURE_COLS))
n_perm_runs    = 0

for fold_i, (tr, te) in enumerate(kf.split(X, y, groups=regions)):
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X[tr], y[tr])
    # Permutation importance on test fold (honest out-of-region)
    pi = permutation_importance(rf, X[te], y[te], n_repeats=5,
                                random_state=42, n_jobs=-1)
    perm_imp_accum += pi.importances_mean
    n_perm_runs += 1
    print(f"    Fold {fold_i+1}/5 done")

perm_mean = perm_imp_accum / n_perm_runs

rf_imp = pd.DataFrame({
    "feature":        FEATURE_COLS,
    "rf_perm_importance": perm_mean.round(6),
}).sort_values("rf_perm_importance", ascending=False)

# Compare to XGBoost CV-SHAP
xgb_rank = {r["feature"]: i+1 for i, r in shap_cv.iterrows()}

print("\n  Top 12 — RF Permutation vs XGBoost CV-SHAP:")
print(f"  {'Feature':30s} {'RF rank':>8} {'XGB rank':>9} {'Agreement':>10}")
for rf_rank, (_, row) in enumerate(rf_imp.head(12).iterrows(), 1):
    xr = xgb_rank.get(row["feature"], 99)
    agree = "✓" if abs(rf_rank - xr) <= 2 else ("~" if abs(rf_rank - xr) <= 4 else "✗")
    print(f"  {row['feature']:30s} {rf_rank:>8} {xr:>9}  {agree:>10}")

rf_imp.to_csv(OUT / "rf_permutation_importance.csv", index=False)
print(f"\n  Saved: ood_threshold_sensitivity.csv  rf_permutation_importance.csv")
print("=" * 70)
