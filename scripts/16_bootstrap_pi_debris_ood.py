"""
Script 16 — Bootstrap Prediction Intervals, Debris Analysis, OOD Projections

Addresses pot_com4 reviewer questions:
  Q1: Projected OOD fractions (σT_JJA=0 under delta method)
  Q4: Bootstrap prediction intervals (30 resamples, 5–95th percentile)
  Q5: Debris-cover residual and SHAP analysis
  Q7: Negative R² sub-region error decomposition

Outputs:
  outputs/bootstrap_pi_hma.parquet      — per-glacier PI width and bounds
  outputs/bootstrap_pi_subregion.csv    — PI summary by sub-region
  outputs/debris_residual_analysis.csv  — debris quantile residual/SHAP stats
  outputs/subregion_error_decomp.csv    — error decomposition per sub-region
  outputs/projected_ood_fractions.csv   — per-sub-region projected OOD fractions
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
print("SCRIPT 16 — BOOTSTRAP PI / DEBRIS / OOD PROJECTION ANALYSIS")
print("=" * 70)

feat    = pd.read_parquet(DATA / "feature_matrix_hma.parquet")
cv_pred = pd.read_parquet(OUT / "cv_predictions_hma.parquet")
results = json.loads((OUT / "model_results_hma.json").read_text())
FEATURE_COLS = results["features"]

X       = feat[FEATURE_COLS].values.astype(float)
y       = feat["SMB_mwea"].values
regions = feat["subregion"].values

XGB_PARAMS = dict(n_estimators=600, learning_rate=0.04, max_depth=5,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                  reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1,
                  random_state=42, verbosity=0)

# ── Q4: Bootstrap Prediction Intervals ───────────────────────────────────────
# Strategy: for each CV fold, train N_BOOT bootstrap models on training fold,
# predict test fold with each, collect per-glacier distribution → PI width.
N_BOOT = 30
kf     = GroupKFold(n_splits=5)
rng    = np.random.default_rng(seed=42)

print(f"\n  Q4: Bootstrap PI ({N_BOOT} resamples × 5 folds) ...")
all_preds = np.full((N_BOOT, len(X)), np.nan)

for fold_i, (tr, te) in enumerate(kf.split(X, y, groups=regions)):
    Xtr, ytr = X[tr], y[tr]
    fold_preds = np.zeros((N_BOOT, len(te)))
    for b in range(N_BOOT):
        idx = rng.choice(len(Xtr), size=len(Xtr), replace=True)
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(Xtr[idx], ytr[idx])
        fold_preds[b] = m.predict(X[te])
    all_preds[:, te] = fold_preds
    print(f"    Fold {fold_i+1}/5 done  (n_test={len(te):,})")

pi_lo   = np.nanpercentile(all_preds, 5,  axis=0)
pi_hi   = np.nanpercentile(all_preds, 95, axis=0)
pi_med  = np.nanmedian(all_preds, axis=0)
pi_width = pi_hi - pi_lo

# Calibration: fraction of observed values within 90% PI
covered = ((y >= pi_lo) & (y <= pi_hi)).mean()
print(f"\n  Bootstrap 90% PI calibration: {100*covered:.1f}% of observations covered")
print(f"  Mean PI width (90%): {pi_width.mean():.3f}  Median: {np.median(pi_width):.3f}  m w.e. yr⁻¹")

pi_df = feat[["RGIId","subregion","SMB_mwea","area_km2","perc_debris"]].copy()
pi_df["pi_lo"]      = pi_lo
pi_df["pi_hi"]      = pi_hi
pi_df["pi_med"]     = pi_med
pi_df["pi_width"]   = pi_width
pi_df["cv_residual"]= cv_pred["y_pred"].values - y if "y_pred" in cv_pred.columns else np.nan

pi_df.to_parquet(OUT / "bootstrap_pi_hma.parquet", index=False)

# Per-sub-region summary
print("\n  Bootstrap PI by sub-region:")
print(f"  {'Sub-region':<25} {'Mean PI':>9} {'Median PI':>10} {'Coverage':>9} {'n':>7}")
sub_pi_rows = []
for reg, grp in pi_df.groupby("subregion"):
    cov  = ((y[grp.index] >= grp["pi_lo"]) & (y[grp.index] <= grp["pi_hi"])).mean()
    row  = {"subregion": reg, "mean_pi_width": round(grp["pi_width"].mean(), 3),
            "median_pi_width": round(grp["pi_width"].median(), 3),
            "coverage_90pct": round(100*cov, 1), "n": len(grp)}
    sub_pi_rows.append(row)
    print(f"  {reg:<25} {row['mean_pi_width']:>9.3f} {row['median_pi_width']:>10.3f} "
          f"{row['coverage_90pct']:>8.1f}% {len(grp):>7,}")

pd.DataFrame(sub_pi_rows).to_csv(OUT / "bootstrap_pi_subregion.csv", index=False)

# ── Q5: Debris-cover residual analysis ────────────────────────────────────────
print("\n  Q5: Debris-cover residual analysis ...")

# Merge CV predictions with debris
cv = cv_pred.rename(columns={"pred_xgb": "y_pred"}).copy()
cv = cv.merge(feat[["RGIId","perc_debris","area_km2"]], on="RGIId", how="left")
cv["residual"] = cv["y_pred"] - cv["SMB_mwea"]

# Bin by debris quartile
cv["debris_q"] = pd.qcut(cv["perc_debris"], q=[0, 0.25, 0.5, 0.75, 0.9, 1.0],
                          labels=["0–25%", "25–50%", "50–75%", "75–90%", ">90%"],
                          duplicates="drop")

print(f"\n  Debris cover > 0 in {(feat['perc_debris'] > 0).mean()*100:.1f}% of glaciers")
print(f"  Mean perc_debris: {feat['perc_debris'].mean():.1f}%  Max: {feat['perc_debris'].max():.1f}%")

debris_rows = []
print(f"\n  {'Debris bin':<12} {'n':>7} {'Mean resid':>11} {'RMSE':>8} {'|SHAP_σT|':>10}")
for q in ["0–25%", "25–50%", "50–75%", "75–90%", ">90%"]:
    grp = cv[cv["debris_q"] == q]
    if len(grp) == 0: continue
    rmse = float(np.sqrt(mean_squared_error(grp["SMB_mwea"], grp["y_pred"])))
    debris_rows.append({"debris_bin": q, "n": len(grp),
                        "mean_residual": round(grp["residual"].mean(), 4),
                        "rmse": round(rmse, 4)})
    print(f"  {q:<12} {len(grp):>7,} {grp['residual'].mean():>11.4f} {rmse:>8.4f}")

pd.DataFrame(debris_rows).to_csv(OUT / "debris_residual_analysis.csv", index=False)

# High-debris (>50th percentile) vs low-debris
high_debris_mask = feat["perc_debris"] > feat["perc_debris"].median()
cv_high = cv[cv["RGIId"].isin(feat.loc[high_debris_mask, "RGIId"])]
cv_low  = cv[~cv["RGIId"].isin(feat.loc[high_debris_mask, "RGIId"])]
r2_high = r2_score(cv_high["SMB_mwea"], cv_high["y_pred"]) if len(cv_high) else np.nan
r2_low  = r2_score(cv_low["SMB_mwea"],  cv_low["y_pred"])  if len(cv_low)  else np.nan
print(f"\n  R² high-debris (>median={feat['perc_debris'].median():.1f}%): {r2_high:.3f}  "
      f"R² low-debris: {r2_low:.3f}")

# ── Q7: Sub-region error decomposition ───────────────────────────────────────
print("\n  Q7: Sub-region error decomposition ...")

decomp_rows = []
print(f"\n  {'Sub-region':<25} {'R²':>7} {'RMSE':>7} {'SMB_std':>9} {'mean_resid':>11}")
for reg, grp_f in feat.groupby("subregion"):
    grp_cv = cv[cv["RGIId"].isin(grp_f["RGIId"])]
    if len(grp_cv) < 5: continue
    r2   = r2_score(grp_cv["SMB_mwea"], grp_cv["y_pred"])
    rmse = float(np.sqrt(mean_squared_error(grp_cv["SMB_mwea"], grp_cv["y_pred"])))
    smb_std = grp_cv["SMB_mwea"].std()
    mean_r  = grp_cv["residual"].mean()
    debris_m = grp_f["perc_debris"].mean()
    decomp_rows.append({
        "subregion": reg, "n": len(grp_cv),
        "r2": round(r2, 3), "rmse": round(rmse, 4),
        "smb_std": round(smb_std, 4), "mean_residual": round(mean_r, 4),
        "mean_perc_debris": round(debris_m, 1),
    })
    flag = " ← negative" if r2 < 0 else ""
    print(f"  {reg:<25} {r2:>7.3f} {rmse:>7.4f} {smb_std:>9.4f} {mean_r:>11.4f}{flag}")

pd.DataFrame(decomp_rows).to_csv(OUT / "subregion_error_decomp.csv", index=False)

# ── Q1: Projected OOD fractions (σT_JJA = 0 under delta method) ──────────────
print("\n  Q1: Projected OOD fractions ...")

# Compute Mahalanobis D² for projected feature set: historical features but σT_JJA = 0
mu      = X.mean(axis=0)
cov_inv = np.linalg.pinv(np.cov(X.T))

# Historical D²
diff_hist  = X - mu
mah2_hist  = np.einsum("ij,jk,ik->i", diff_hist, cov_inv, diff_hist)
thresh     = np.percentile(mah2_hist, 97.5)

# Projected: set σT_JJA = 0
fi = {f: i for i, f in enumerate(FEATURE_COLS)}
X_proj = X.copy()
if "T_JJA_std_C" in fi:
    X_proj[:, fi["T_JJA_std_C"]] = 0.0
    diff_proj  = X_proj - mu
    mah2_proj  = np.einsum("ij,jk,ik->i", diff_proj, cov_inv, diff_proj)
    ood_proj   = (mah2_proj > thresh).mean()
    print(f"\n  With σT_JJA = 0 (delta-method projection):")
    print(f"  Training D² 97.5th-percentile threshold: {thresh:.1f}")
    print(f"  OOD fraction (all glaciers): {100*ood_proj:.1f}%")
    print(f"  Median projected D²: {np.median(mah2_proj):.1f}  vs historical median: {np.median(mah2_hist):.1f}")

    ood_proj_rows = []
    print(f"\n  {'Sub-region':<25} {'Hist OOD%':>10} {'Proj OOD%':>11}")
    for reg in np.unique(regions):
        mask = regions == reg
        h_frac = 100 * (mah2_hist[mask] > thresh).mean()
        p_frac = 100 * (mah2_proj[mask] > thresh).mean()
        ood_proj_rows.append({"subregion": reg,
                              "hist_ood_pct": round(h_frac, 1),
                              "proj_ood_pct_sigma0": round(p_frac, 1)})
        print(f"  {reg:<25} {h_frac:>10.1f} {p_frac:>11.1f}")

    pd.DataFrame(ood_proj_rows).to_csv(OUT / "projected_ood_fractions.csv", index=False)

print(f"\n  Outputs saved to {OUT}/")
print("=" * 70)
