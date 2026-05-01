"""
Script 10 — Generate Manuscript Tables

Tables (saved as CSV + LaTeX):
  table1_model_comparison.csv/.tex     — Ridge / RF / XGBoost CV metrics
  table2_regional_performance.csv/.tex — Per sub-region XGBoost metrics
  table3_shap_importance.csv/.tex      — Top-15 SHAP feature importance
  table4_projections.csv/.tex          — Projected SMB by region × scenario × window
  table5_dataset_summary.csv/.tex      — Dataset counts and SMB stats by RGI region

All outputs saved to outputs/tables/
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

OUT_DIR   = Path(__file__).parent.parent / "outputs"
TAB_DIR   = OUT_DIR / "tables"
TAB_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("SCRIPT 10 — MANUSCRIPT TABLES")
print("=" * 70)

# ── Load inputs ───────────────────────────────────────────────────────────────
results  = json.loads((OUT_DIR / "model_results_hma.json").read_text())
imp_df   = pd.read_csv(OUT_DIR / "shap_importance_hma.csv")
proj_sum = pd.read_csv(OUT_DIR / "smb_projection_summary.csv")
feat_df  = pd.read_parquet(
    Path(__file__).parent.parent / "data" / "feature_matrix_hma.parquet")

REGION_LABELS = {
    "C_Himalaya":     "C. Himalaya",
    "C_Tian_Shan":    "C. Tian Shan",
    "E_Himalaya":     "E. Himalaya",
    "E_Pamir":        "E. Pamir",
    "Hindu_Kush":     "Hindu Kush",
    "Hengduan_Shan_N":"Hengduan Shan N.",
    "Inner_Tibet":    "Inner Tibet",
    "Karakoram":      "Karakoram",
    "Kunlun":         "Kunlun",
    "N_Tian_Shan":    "N. Tian Shan",
    "Other_CA":       "Other C.A.",      # legacy fallback
    "Pamir_Alay":     "Pamir-Alay",
    "Qilian_Shan":    "Qilian Shan",
    "W_Himalaya":     "W. Himalaya",
    "W_Nepal":        "W. Nepal",
    "W_Pamir":        "W. Pamir",
}

PRETTY_FEAT = {
    "T_JJA_std_C":       "JJA temperature variability ($\\sigma$, $^{\circ}$C)",
    "P_JJA_mm":          "JJA precipitation (mm month$^{-1}$)",
    "zmean_m":           "Mean glacier elevation (m a.s.l.)",
    "slope_deg":         "Glacier slope ($^{\circ}$)",
    "continentality":    "Continentality ($T_{\\rm JJA} - T_{\\rm DJF}$, $^{\circ}$C)",
    "precip_winter_frac":"Winter precipitation fraction",
    "P_SON_mm":          "SON precipitation (mm month$^{-1}$)",
    "elev_range_m":      "Elevation range (m)",
    "T_MAM_C":           "MAM temperature ($^{\circ}$C)",
    "area_km2":          "Glacier area (km$^2$)",
    "cryo_balance":      "Cryo-balance index",
    "T_JJA_max_C":       "JJA maximum temperature ($^{\circ}$C)",
    "T_JJA_lapse_C":     "JJA temperature, lapse-corrected ($^{\circ}$C)",
    "P_snow_mm":         "Cold-season precipitation (DJF+MAM, mm month$^{-1}$)",
    "P_std_mm":          "Precipitation variability ($\\sigma$, mm month$^{-1}$)",
    "T_ann_lapse_C":     "Annual mean temperature, lapse-corrected ($^{\circ}$C)",
    "T_trend_C_dec":     "Annual temperature trend ($^{\circ}$C decade$^{-1}$)",
    "T_JJA_trend_C_dec": "JJA temperature trend ($^{\circ}$C decade$^{-1}$)",
    "P_trend_mm_dec":    "Annual precipitation trend (mm decade$^{-1}$)",
    "T_std_C":           "Annual temperature variability ($\\sigma$, $^{\circ}$C)",
    "T_JJA_max_C":       "JJA maximum temperature ($^{\circ}$C)",
    "T_DJF_min_C":       "DJF minimum temperature ($^{\circ}$C)",
    "log_area_km2":      "log$_{10}$(glacier area, km$^2$)",
    "aspect_sin":        "Aspect (sin component)",
    "aspect_cos":        "Aspect (cos component)",
    "melt_index":        "Melt index ($T_{\\rm JJA,lapse} \\times \\sqrt{A}$)",
    "P_trend_x_wfrac":   "Precip. trend $\\times$ winter fraction",
    "P_mean_mm":         "Mean monthly precipitation (mm month$^{-1}$)",
    "T_mean_C":          "Annual mean temperature ($^{\circ}$C)",
}

def to_latex(df, caption, label, col_format=None, float_fmt="{:.3f}"):
    """Convert DataFrame to LaTeX tabular with caption and label."""
    n_cols = len(df.columns) + (1 if df.index.name else 0)
    if col_format is None:
        col_format = "l" + "r" * (n_cols - 1)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_format}}}",
        "\\hline",
    ]
    # Header
    header = " & ".join(str(c) for c in df.columns)
    lines.append(header + " \\\\")
    lines.append("\\hline")
    # Rows
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(float_fmt.format(v))
            else:
                cells.append(str(v))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# Table 1 — Model comparison
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Table 1 — Model comparison ...")

fold_r2   = results["xgb_fold_r2"]
fold_mean = results["xgb_fold_r2_mean"]
fold_std  = results["xgb_fold_r2_std"]

# Area-weighted metrics for XGBoost CV predictions
cv_preds = pd.read_parquet(OUT_DIR / "cv_predictions_hma.parquet")
cv_preds = cv_preds.merge(feat_df[["RGIId","area_km2"]], on="RGIId", how="left")
_y = cv_preds["SMB_mwea"].values
_p = cv_preds["pred_xgb"].values
_w = cv_preds["area_km2"].values
_wsum = _w.sum()
_yw_mean = np.sum(_w * _y) / _wsum
_r2_area  = 1 - np.sum(_w * (_y - _p)**2) / np.sum(_w * (_y - _yw_mean)**2)
_rmse_area = float(np.sqrt(np.sum(_w * (_y - _p)**2) / _wsum))

t1 = pd.DataFrame([
    {
        "Model":             "Ridge regression",
        "CV $R^2$":          results["metrics"]["ridge"]["r2"],
        "RMSE": results["metrics"]["ridge"]["rmse"],
        "MAE":  results["metrics"]["ridge"]["mae"],
        "Bias":              results["metrics"]["ridge"]["bias"],
        "Spatial fold $R^2$ (mean $\\pm$ $\\sigma$)": "--",
        "Area-wtd $R^2$": "--",
        "Area-wtd RMSE": "--",
    },
    {
        "Model":             "Random Forest",
        "CV $R^2$":          results["metrics"]["random_forest"]["r2"],
        "RMSE": results["metrics"]["random_forest"]["rmse"],
        "MAE":  results["metrics"]["random_forest"]["mae"],
        "Bias":              results["metrics"]["random_forest"]["bias"],
        "Spatial fold $R^2$ (mean $\\pm$ $\\sigma$)": "--",
        "Area-wtd $R^2$": "--",
        "Area-wtd RMSE": "--",
    },
    {
        "Model":             "XGBoost (primary)",
        "CV $R^2$":          results["metrics"]["xgboost"]["r2"],
        "RMSE": results["metrics"]["xgboost"]["rmse"],
        "MAE":  results["metrics"]["xgboost"]["mae"],
        "Bias":              results["metrics"]["xgboost"]["bias"],
        "Spatial fold $R^2$ (mean $\\pm$ $\\sigma$)": f"{fold_mean:.3f} $\\pm$ {fold_std:.3f}",
        "Area-wtd $R^2$": f"{_r2_area:.3f}",
        "Area-wtd RMSE": f"{_rmse_area:.4f}",
    },
])

t1.to_csv(TAB_DIR / "table1_model_comparison.csv", index=False)
(TAB_DIR / "table1_model_comparison.tex").write_text(
    to_latex(t1,
             caption="Performance of candidate models under 5-fold spatial "
                     "cross-validation (GroupKFold by sub-region, $n$=94,463 glaciers). "
                     "CV R$^2$ and RMSE (m~w.e.~yr$^{-1}$) are glacier-count-weighted; "
                     "area-weighted metrics (final two columns) use glacier area as weight "
                     "and are reported for XGBoost only. "
                     "Spatial fold R$^2$ reports the mean $\\pm$ standard deviation "
                     "across the five folds for XGBoost.",
             label="tab:model_comparison",
             col_format="lrrrrlrr"))
print("    Saved table1")
print(t1.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# Table 2 — Per sub-region performance
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Table 2 — Regional performance ...")

reg_stats = feat_df.groupby("subregion").agg(
    n=("SMB_mwea","count"),
    SMB_mean=("SMB_mwea","mean"),
    SMB_std=("SMB_mwea","std"),
).reset_index()

reg_metrics = results["xgb_by_subregion"]
rows = []
for reg, grp in reg_stats.iterrows():
    subr = grp["subregion"]
    rm = reg_metrics.get(subr, {})
    rows.append({
        "Sub-region":                REGION_LABELS.get(subr, subr),
        "n":                         int(grp["n"]),
        "Mean SMB":                  round(grp["SMB_mean"], 3),
        "$\\sigma$ SMB":             round(grp["SMB_std"],  3),
        "XGBoost $R^2$":             rm.get("r2", float("nan")),
        "RMSE":                      rm.get("rmse", float("nan")),
        "MAE":                       rm.get("mae",  float("nan")),
    })

t2 = pd.DataFrame(rows).sort_values("Mean SMB")
t2.to_csv(TAB_DIR / "table2_regional_performance.csv", index=False)
(TAB_DIR / "table2_regional_performance.tex").write_text(
    to_latex(t2,
             caption="XGBoost prediction performance and observed mass balance "
                     "statistics by sub-region. Metrics are computed on out-of-fold "
                     "predictions from 5-fold spatial cross-validation. "
                     "SMB = surface mass balance (m~w.e.~yr$^{-1}$).",
             label="tab:regional_performance",
             col_format="lrrrrrr"))
print("    Saved table2")
print(t2.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# Table 3 — SHAP feature importance (top 15)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Table 3 — SHAP importance ...")

val_col = [c for c in imp_df.columns if c != "feature"][0]
top15   = imp_df.head(15).copy()
top15["Feature"] = top15["feature"].map(lambda f: PRETTY_FEAT.get(f, f))
top15["Mean |SHAP|"] = top15[val_col].round(5)
top15["Rank"] = range(1, 16)
t3 = top15[["Rank","Feature","Mean |SHAP|"]].reset_index(drop=True)

t3.to_csv(TAB_DIR / "table3_shap_importance.csv", index=False)
(TAB_DIR / "table3_shap_importance.tex").write_text(
    to_latex(t3,
             caption="Top-15 features ranked by mean absolute SHAP value "
                     "(m~w.e.~yr$^{-1}$) from the full-dataset XGBoost model "
                     "($n$=94,463 glaciers). Geographic coordinates (latitude, "
                     "longitude) were excluded from the feature set to ensure "
                     "attribution reflects climate--terrain mechanisms rather "
                     "than spatial memorisation.",
             label="tab:shap_importance",
             col_format="clr"))
print("    Saved table3")
print(t3.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# Table 4 — Projections by sub-region × scenario × window
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Table 4 — Projections ...")

WINDOWS   = ["2020-2039","2040-2059","2060-2079","2080-2099"]
SCENARIOS = ["ssp245","ssp585"]
SC_LABEL  = {"ssp245":"SSP2-4.5","ssp585":"SSP5-8.5"}

proj_sum["Sub-region"] = proj_sum["subregion"].map(REGION_LABELS)
proj_sum = proj_sum.sort_values("SMB_hist")

rows4 = []
for _, row in proj_sum.iterrows():
    base = {
        "Sub-region": row["Sub-region"],
        "Hist.\\ SMB (2000--2014)": round(row["SMB_hist"], 3),
    }
    for sc in SCENARIOS:
        for w in WINDOWS:
            col = f"SMB_{sc}_{w}_ensmean"
            if col in row.index:
                delta = round(row[col] - row["SMB_hist"], 3)
                base[f"$\\Delta$ {SC_LABEL[sc]} {w}"] = delta
    rows4.append(base)

t4 = pd.DataFrame(rows4)
t4.to_csv(TAB_DIR / "table4_projections.csv", index=False)

# OOD sub-regions (E. Himalaya, C. Himalaya) get a dagger marker
OOD_REGS = {"E. Himalaya", "C. Himalaya"}

def make_proj_split_latex():
    """Generate split two-panel LaTeX table (SSP2-4.5 top, SSP5-8.5 bottom)."""
    caption = ("Projected change in glacier mass balance ($\\Delta$SMB, "
               "m~w.e.~yr$^{-1}$) relative to the 2000--2014 historical mean. "
               "Values are ensemble means across three CMIP6 models (MIROC6, "
               "MPI-ESM1-2-LR, IPSL-CM6A-LR). Negative $\\Delta$SMB = accelerated mass "
               "loss. $\\dagger$ marks sub-regions where projections exceed the model's "
               "training distribution (OOD; see Sect.~4.5).")

    col5 = "lrrrrr"
    win_hdr = " & ".join(["2020--2039","2040--2059","2060--2079","2080--2099"])

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:projections}",
        "\\small",
        # ── SSP2-4.5 panel ────────────────────────────────────────────────
        "\\textbf{(a) SSP2-4.5}\\\\[2pt]",
        f"\\begin{{tabular}}{{{col5}}}",
        "\\hline",
        f"Sub-region & Hist.\\ SMB & {win_hdr} \\\\",
        "\\hline",
    ]
    for _, row in proj_sum.iterrows():
        reg = row["Sub-region"]
        marker = "$^{\\dagger}$" if reg in OOD_REGS else ""
        hist = f"{row['SMB_hist']:.3f}"
        vals = []
        for w in WINDOWS:
            col = f"SMB_ssp245_{w}_ensmean"
            if col in row.index:
                vals.append(f"{row[col] - row['SMB_hist']:+.3f}")
            else:
                vals.append("--")
        lines.append(f"{reg}{marker} & {hist} & " + " & ".join(vals) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "", "\\vspace{4pt}", ""]

    # ── SSP5-8.5 panel ────────────────────────────────────────────────────
    lines += [
        "\\textbf{(b) SSP5-8.5}\\\\[2pt]",
        f"\\begin{{tabular}}{{{col5}}}",
        "\\hline",
        f"Sub-region & Hist.\\ SMB & {win_hdr} \\\\",
        "\\hline",
    ]
    for _, row in proj_sum.iterrows():
        reg = row["Sub-region"]
        marker = "$^{\\dagger}$" if reg in OOD_REGS else ""
        hist = f"{row['SMB_hist']:.3f}"
        vals = []
        for w in WINDOWS:
            col = f"SMB_ssp585_{w}_ensmean"
            if col in row.index:
                vals.append(f"{row[col] - row['SMB_hist']:+.3f}")
            else:
                vals.append("--")
        lines.append(f"{reg}{marker} & {hist} & " + " & ".join(vals) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)

(TAB_DIR / "table4_projections.tex").write_text(make_proj_split_latex())
print("    Saved table4")
print(t4.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# Table 5 — Dataset summary by RGI macro-region
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Table 5 — Dataset summary ...")

RGI_NAME = {13:"RGI-13 Central Asia",
            14:"RGI-14 South Asia West",
            15:"RGI-15 South Asia East"}

rows5 = []
for r, grp in feat_df.groupby("rgi_region"):
    rows5.append({
        "Region":                RGI_NAME.get(r, str(r)),
        "n glaciers":            len(grp),
        "Total area (km$^2$)":   round(grp["area_km2"].sum(), 0),
        "Mean SMB":              round(grp["SMB_mwea"].mean(), 3),
        "$\\sigma$ SMB":         round(grp["SMB_mwea"].std(),  3),
        "Min SMB":               round(grp["SMB_mwea"].min(),  3),
        "Max SMB":               round(grp["SMB_mwea"].max(),  3),
    })
# Total row
rows5.append({
    "Region":                "All HMA",
    "n glaciers":            len(feat_df),
    "Total area (km$^2$)":   round(feat_df["area_km2"].sum(), 0),
    "Mean SMB":              round(feat_df["SMB_mwea"].mean(), 3),
    "$\\sigma$ SMB":         round(feat_df["SMB_mwea"].std(),  3),
    "Min SMB":               round(feat_df["SMB_mwea"].min(),  3),
    "Max SMB":               round(feat_df["SMB_mwea"].max(),  3),
})

t5 = pd.DataFrame(rows5)
t5.to_csv(TAB_DIR / "table5_dataset_summary.csv", index=False)
(TAB_DIR / "table5_dataset_summary.tex").write_text(
    to_latex(t5,
             caption="Summary statistics of the geodetic mass balance dataset "
                     "(Shean et al., 2020) used in this study. "
                     "SMB = surface mass balance (m~w.e.~yr$^{-1}$, 2000--2018). "
                     "Glaciers with $|$SMB$|$ $>$ 5.0~m~w.e.~yr$^{-1}$ or "
                     "$\\sigma_{\\rm SMB}$ $>$ 1.0 were excluded as outliers.",
             label="tab:dataset_summary",
             col_format="lrrrrrr"))
print("    Saved table5")
print(t5.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
tables = sorted(TAB_DIR.glob("*"))
print(f"\n  {len(tables)} files saved to: {TAB_DIR}")
for f in tables:
    print(f"    {f.name}  ({f.stat().st_size/1e3:.0f} kB)")
print("\n  Next: write manuscript draft")
print("=" * 70)
