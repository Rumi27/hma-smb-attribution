"""
Script 18 — Feature Sensitivity: Hypsometry, σSMB Weighting, Melt-index Ablation
(Q3, Q5, Q10 — pot_com5)

Tests three independent sensitivity analyses:

  Q3: Add hypsometric integral HI = (zmean − zmin)/(zmax − zmin) as feature.
  Q5: Train with sample weights = 1/SMB_sigma² (σSMB-weighted).
  Q10: Ablation of √A scaling in melt_index (compare melt_index vs T_JJA_lapse only).

Outputs:
  outputs/sensitivity_hyp_sigma_melt.csv
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
print("SCRIPT 18 — HYPSOMETRY / σSMB-WEIGHT / MELT-INDEX SENSITIVITY")
print("=" * 70)

feat    = pd.read_parquet(DATA / "feature_matrix_hma.parquet")
results = json.loads((OUT / "model_results_hma.json").read_text())
FEATURE_COLS = results["features"]

y       = feat["SMB_mwea"].values
regions = feat["subregion"].values

XGB_PARAMS = dict(n_estimators=600, learning_rate=0.04, max_depth=5,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                  reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1,
                  random_state=42, verbosity=0)
kf = GroupKFold(n_splits=5)

def run_cv(Xc, y, regions, weights=None, label=""):
    y_pred = np.zeros_like(y)
    for tr, te in kf.split(Xc, y, groups=regions):
        m = xgb.XGBRegressor(**XGB_PARAMS)
        w_tr = weights[tr] if weights is not None else None
        m.fit(Xc[tr], y[tr], sample_weight=w_tr)
        y_pred[te] = m.predict(Xc[te])
    r2   = r2_score(y, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    print(f"  {label:55s}: R²={r2:.4f}  RMSE={rmse:.4f}")
    return r2, rmse

rows = []

# ── Baseline (reproduce) ──────────────────────────────────────────────────────
X_base = feat[FEATURE_COLS].values.astype(float)
r2, rmse = run_cv(X_base, y, regions, label="Baseline (no change)")
rows.append({"experiment": "Baseline", "r2": r2, "rmse": rmse, "note": "reference"})

# ── Q3: Hypsometric integral ──────────────────────────────────────────────────
print("\n  Q3: Hypsometric integral ...")
feat2 = feat.copy()
feat2["hyp_integral"] = ((feat2["zmean_m"] - feat2["zmin_m"])
                         / (feat2["zmax_m"] - feat2["zmin_m"]).clip(lower=1.0))
FEAT_HYP = FEATURE_COLS + ["hyp_integral"]
X_hyp = feat2[FEAT_HYP].values.astype(float)
r2, rmse = run_cv(X_hyp, y, regions, label="+ hypsometric integral (HI)")
rows.append({"experiment": "+hypsometric_integral", "r2": r2, "rmse": rmse,
             "note": "HI=(zmean-zmin)/(zmax-zmin)"})

# HI stats
hi = feat2["hyp_integral"]
print(f"    HI range: {hi.min():.3f}–{hi.max():.3f}  mean={hi.mean():.3f}  "
      f"std={hi.std():.3f}")

# ── Q5: σSMB-weighted training ────────────────────────────────────────────────
print("\n  Q5: σSMB-weighted training ...")
sigma = feat["SMB_sigma"].values.clip(min=0.01)
weights = 1.0 / (sigma ** 2)
weights /= weights.mean()   # normalise to mean=1 for numerical stability
r2, rmse = run_cv(X_base, y, regions, weights=weights,
                  label="σSMB-weighted (w = 1/σ²)")
rows.append({"experiment": "sigma_smb_weighted", "r2": r2, "rmse": rmse,
             "note": "sample_weight=1/SMB_sigma²"})

print(f"    σSMB range: {sigma.min():.3f}–{sigma.max():.3f}  "
      f"mean={sigma.mean():.3f}")
high_sigma_frac = (sigma > 0.5).mean()
print(f"    Glaciers with σSMB > 0.5 m w.e. yr⁻¹: {100*high_sigma_frac:.1f}%")

# ── Q10: Melt index ablation — remove √A scaling ─────────────────────────────
print("\n  Q10: Melt index ablation ...")
# Current: melt_index = T_JJA_lapse_C / (area_km2^0.5 + ε) or similar
# We test: replace melt_index with T_JJA_lapse_C directly (no area scaling)
fi = FEATURE_COLS.index("melt_index") if "melt_index" in FEATURE_COLS else -1
if fi >= 0:
    X_noscale = X_base.copy()
    X_noscale[:, fi] = feat["T_JJA_lapse_C"].values   # replace with unscaled version
    r2, rmse = run_cv(X_noscale, y, regions,
                      label="melt_index → T_JJA_lapse (no √A)")
    rows.append({"experiment": "melt_index_no_sqrt_area", "r2": r2, "rmse": rmse,
                 "note": "melt_index replaced by T_JJA_lapse_C (drop √A scaling)"})

    # Also test removing melt_index entirely
    FEAT_NOMELT = [c for c in FEATURE_COLS if c != "melt_index"]
    X_nomelt = feat[FEAT_NOMELT].values.astype(float)
    r2, rmse = run_cv(X_nomelt, y, regions, label="drop melt_index feature entirely")
    rows.append({"experiment": "no_melt_index", "r2": r2, "rmse": rmse,
                 "note": "melt_index removed from feature set"})

    # Current melt_index formula check
    mi = feat["melt_index"].values
    t_lapse = feat["T_JJA_lapse_C"].values
    a_sqrt = np.sqrt(feat["area_km2"].values)
    corr = np.corrcoef(mi, t_lapse / a_sqrt.clip(min=0.01))[0, 1]
    print(f"    Pearson r(melt_index, T_JJA_lapse/√A) = {corr:.4f}")
    corr2 = np.corrcoef(mi, t_lapse)[0, 1]
    print(f"    Pearson r(melt_index, T_JJA_lapse)    = {corr2:.4f}")
else:
    print("    melt_index not found in FEATURE_COLS")

# Save
pd.DataFrame(rows).to_csv(OUT / "sensitivity_hyp_sigma_melt.csv", index=False)
print(f"\n  Saved: sensitivity_hyp_sigma_melt.csv")
print("=" * 70)
