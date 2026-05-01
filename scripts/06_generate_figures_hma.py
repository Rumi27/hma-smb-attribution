"""
Script 06 — Publication-Quality Figures (Full HMA)

Style guide: The Cryosphere / JGR format
  - Vector PDF output (editable in Illustrator / Inkscape)
  - Column widths: single = 8.3 cm, double = 17.1 cm
  - Font: 8 pt sans-serif body, 9 pt axis labels
  - pdf.fonttype = 42  (TrueType — text stays editable)
  - No top/right spines; outward ticks; minimal grid

Figures:
  fig01_study_area.pdf          — HMA glacier map coloured by SMB
  fig02_cv_scatter_grid.pdf     — Per-sub-region predicted vs observed (Peng et al. Fig 4 style)
  fig03_shap_beeswarm.pdf       — SHAP beeswarm top 15 features
  fig04_shap_bar.pdf            — Mean |SHAP| ranked bar
  fig05_shap_heatmap.pdf        — Signed SHAP by region × feature
  fig06_shap_dependence.pdf     — Dependence plots top 3 drivers
  fig07_smb_violin.pdf          — SMB violin by sub-region
"""

import pandas as pd
import numpy as np
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path
import contextily as cx
import geopandas as gpd
from pyproj import Transformer

# ── Paths ─────────────────────────────────────────────────────────────────────
PAPER_DIR = Path(__file__).parent.parent
OUT_DIR   = PAPER_DIR / "outputs"
FIG_DIR   = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("SCRIPT 06 — PUBLICATION FIGURES (vector PDF)")
print("=" * 70)

# ── Global style ──────────────────────────────────────────────────────────────
CM = 1 / 2.54          # 1 cm in inches
W1 = 8.3  * CM        # single column
W2 = 17.1 * CM        # double column

mpl.rcParams.update({
    # Font
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size":            8,
    "axes.labelsize":       9,
    "axes.titlesize":       9,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      7.5,
    "legend.title_fontsize":8,
    # Axes
    "axes.linewidth":       0.7,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.linewidth":       0.4,
    "grid.color":           "#e0e0e0",
    "grid.alpha":           1.0,
    # Ticks
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.width":    0.7,
    "ytick.major.width":    0.7,
    "xtick.major.size":     3.0,
    "ytick.major.size":     3.0,
    # Lines
    "lines.linewidth":      1.0,
    # Save
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.03,
    "figure.dpi":           150,
    # PDF: keep fonts editable
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
})

def savefig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path)
    print(f"    Saved {name}  ({path.stat().st_size/1e3:.0f} kB)")
    plt.close(fig)

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n[1] Loading data ...")
shap_df  = pd.read_parquet(OUT_DIR / "shap_values_hma.parquet")
cv_df    = pd.read_parquet(OUT_DIR / "cv_predictions_hma.parquet")
imp_df   = pd.read_csv(OUT_DIR / "shap_importance_hma.csv")
results  = json.loads((OUT_DIR / "model_results_hma.json").read_text())
feat_df  = pd.read_parquet(PAPER_DIR / "data" / "feature_matrix_hma.parquet")

FEATURE_COLS = results["features"]
VAL_COL = [c for c in imp_df.columns if c != "feature"][0]

# ── Consistent palette ────────────────────────────────────────────────────────
SUBREGIONS = sorted(feat_df["subregion"].unique())
_cmap = mpl.colormaps["tab20"]
PALETTE = {r: _cmap(i / len(SUBREGIONS)) for i, r in enumerate(SUBREGIONS)}

REG_LABEL = {
    "C_Himalaya":     "C. Himalaya",
    "C_Tian_Shan":    "C. Tian Shan",
    "E_Himalaya":     "E. Himalaya",
    "E_Pamir":        "E. Pamir",
    "Hengduan_Shan_N":"Hengduan Shan N.",
    "Hindu_Kush":     "Hindu Kush",
    "Inner_Tibet":    "Inner Tibet",
    "Karakoram":      "Karakoram",
    "Kunlun":         "Kunlun",
    "N_Tian_Shan":    "N. Tian Shan",
    "Other_CA":       "Other C.A.",     # legacy fallback
    "Pamir_Alay":     "Pamir-Alay",
    "Qilian_Shan":    "Qilian Shan",
    "W_Himalaya":     "W. Himalaya",
    "W_Nepal":        "W. Nepal",
    "W_Pamir":        "W. Pamir",
}

FEAT_LABEL = {
    "T_JJA_std_C":       r"JJA T variability ($\sigma$, °C)",
    "P_JJA_mm":          r"JJA precipitation (mm mo$^{-1}$)",
    "zmean_m":           r"Mean elevation (m a.s.l.)",
    "slope_deg":         r"Slope (°)",
    "continentality":    r"Continentality ($T_\mathrm{JJA}-T_\mathrm{DJF}$, °C)",
    "precip_winter_frac":r"Winter precip. fraction",
    "P_SON_mm":          r"SON precipitation (mm mo$^{-1}$)",
    "elev_range_m":      r"Elevation range (m)",
    "T_MAM_C":           r"MAM temperature (°C)",
    "area_km2":          r"Glacier area (km$^2$)",
    "cryo_balance":      r"Cryo-balance index",
    "T_JJA_max_C":       r"JJA max temperature (°C)",
    "T_JJA_lapse_C":     r"JJA T lapse-corrected (°C)",
    "P_snow_mm":         r"Cold-season precip. (mm mo$^{-1}$)",
    "P_std_mm":          r"Precip. variability ($\sigma$, mm mo$^{-1}$)",
}

SMB_UNIT = r"SMB (m w.e. yr$^{-1}$)"

def annot_box(ax, text, loc="upper left", pad=0.04):
    """Add a clean metric annotation box (Peng et al. style)."""
    props = dict(boxstyle="round,pad=0.3", facecolor="white",
                 edgecolor="#aaaaaa", linewidth=0.6, alpha=0.92)
    kw = dict(transform=ax.transAxes, fontsize=7.5,
              verticalalignment="top", bbox=props)
    if loc == "upper left":
        ax.text(pad, 1 - pad, text, ha="left", **kw)
    elif loc == "upper right":
        ax.text(1 - pad, 1 - pad, text, ha="right", **kw)
    elif loc == "lower right":
        ax.text(1 - pad, pad, text, ha="right",
                verticalalignment="bottom", bbox=props,
                transform=ax.transAxes, fontsize=7.5)

# ══════════════════════════════════════════════════════════════════════════════
# Fig 01 — Study area map with satellite basemap + global inset
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Fig 01 — Study area map (satellite + inset) ...")

smb = feat_df["SMB_mwea"].values
lon = feat_df["cenlon"].values
lat = feat_df["cenlat"].values

# Coordinate transformer: WGS84 ↔ Web Mercator (required by contextily)
t_fwd = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
t_inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

LON0,LON1,LAT0,LAT1 = 63.0, 107.5, 23.5, 48.5
xmin,ymin = t_fwd.transform(LON0, LAT0)
xmax,ymax = t_fwd.transform(LON1, LAT1)
gx, gy = t_fwd.transform(lon, lat)

fig = plt.figure(figsize=(W2, W2 * 0.60), facecolor="white")
ax  = fig.add_axes([0.06, 0.08, 0.88, 0.84])
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

# Satellite basemap
cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery,
               zoom=5, attribution=False)

# Glacier scatter
sc = ax.scatter(gx, gy, c=smb, cmap="RdBu_r", vmin=-1.2, vmax=1.2,
                s=2.0, alpha=0.80, linewidths=0, rasterized=True, zorder=4)

# Sub-region labels
reg_cents = feat_df.groupby("subregion")[["cenlon","cenlat"]].median()
for reg, row in reg_cents.iterrows():
    rx, ry = t_fwd.transform(row["cenlon"], row["cenlat"])
    ax.text(rx, ry, REG_LABEL.get(reg, reg),
            fontsize=5.8, ha="center", va="center",
            color="white", fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.20", fc="black",
                      alpha=0.50, ec="none"))

# Axis tick formatters (degrees from Web Mercator)
def lon_fmt(x, pos): return f"{t_inv.transform(x,0)[0]:.0f}°E"
def lat_fmt(y, pos): return f"{t_inv.transform(0,y)[1]:.0f}°N"
ax.set_xticks([t_fwd.transform(v,0)[0] for v in [70,80,90,100]])
ax.set_yticks([t_fwd.transform(0,v)[1] for v in [30,35,40,45]])
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lon_fmt))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lat_fmt))
ax.tick_params(color="#333", labelcolor="#222")
for sp in ax.spines.values():
    sp.set_edgecolor("#444"); sp.set_linewidth(0.7); sp.set_visible(True)
ax.set_xlabel("Longitude", fontsize=8, labelpad=3)
ax.set_ylabel("Latitude",  fontsize=8, labelpad=3)

# Colorbar
cb = fig.colorbar(sc, ax=ax, fraction=0.020, pad=0.012,
                  extend="both", shrink=0.80)
cb.set_label(r"Mass balance (m w.e. yr$^{-1}$)", fontsize=8)
cb.ax.tick_params(labelsize=7.5)
cb.outline.set_linewidth(0.5)
cb.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])

ax.set_title(
    f"Geodetic glacier mass balance, 2000–2018  ·  "
    f"$n$ = {len(feat_df):,} glaciers  ·  "
    f"HMA mean = {smb.mean():.3f} m w.e. yr$^{{-1}}$",
    fontsize=8.5, pad=5, color="#111")

# ── Global inset map (bottom-left) ───────────────────────────────────────────
NE_SHP = Path("/tmp/ne_110m/ne_110m_admin_0_countries.shp")
if not NE_SHP.exists():
    import urllib.request, zipfile, io
    NE_SHP.parent.mkdir(exist_ok=True)
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    data = urllib.request.urlopen(url, timeout=20).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(NE_SHP.parent)

world   = gpd.read_file(NE_SHP)
ax_ins  = fig.add_axes([0.065, 0.09, 0.195, 0.26])
world.plot(ax=ax_ins, color="#c8d8e8", edgecolor="#777", linewidth=0.2, zorder=2)
ax_ins.set_facecolor("#a8c4d4")

for alpha, fc in [(0.30, "#e63329"), (0.0, "none")]:
    ax_ins.add_patch(mpatches.FancyBboxPatch(
        (LON0, LAT0), LON1-LON0, LAT1-LAT0,
        boxstyle="square,pad=0", linewidth=1.4,
        edgecolor="#e63329", facecolor=fc, alpha=alpha, zorder=4+int(alpha==0)))

ax_ins.text((LON0+LON1)/2, (LAT0+LAT1)/2, "HMA",
            fontsize=5.5, ha="center", va="center",
            color="white", fontweight="bold", zorder=6)
ax_ins.set_xlim(-180,180); ax_ins.set_ylim(-75,85)
ax_ins.set_xticks([]); ax_ins.set_yticks([])
for sp in ax_ins.spines.values():
    sp.set_edgecolor("#555"); sp.set_linewidth(0.6); sp.set_visible(True)
ax_ins.text(0.5, 1.015, "Location", transform=ax_ins.transAxes,
            ha="center", va="bottom", fontsize=6.2, color="#333", style="italic")

savefig(fig, "fig01_study_area.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 02 — Per-sub-region predicted vs observed (Peng et al. Fig 4 style)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Fig 02 — CV scatter grid ...")

REGS_SORTED = sorted(SUBREGIONS,
    key=lambda r: feat_df.loc[feat_df["subregion"]==r,"SMB_mwea"].mean())

n_regs = len(REGS_SORTED)
ncols = 4
nrows = (n_regs + ncols - 1) // ncols   # ceiling division → 4 rows for 15 sub-regions
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(W2, W2 * (nrows / 3) * 0.90),
                         sharex=True, sharey=True,
                         constrained_layout=True)

lim = (-2.4, 1.6)

for ax_i, reg in enumerate(REGS_SORTED):
    row_i, col_i = ax_i // ncols, ax_i % ncols
    ax = axes[row_i, col_i]
    mask = cv_df["subregion"] == reg

    y_obs  = cv_df.loc[mask, "SMB_mwea"].values
    y_pred = cv_df.loc[mask, "pred_xgb"].values

    ax.scatter(y_obs, y_pred, color=PALETTE[reg],
               s=2.0, alpha=0.40, linewidths=0, rasterized=True, zorder=3)

    ax.plot(lim, lim, "k-", lw=0.7, zorder=4)
    ax.axhline(0, color="#bbb", lw=0.4, ls="--", zorder=2)
    ax.axvline(0, color="#bbb", lw=0.4, ls="--", zorder=2)

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xticks([-2, -1, 0, 1])
    ax.set_yticks([-2, -1, 0, 1])
    ax.tick_params(labelsize=7)

    r2   = r2_score(y_obs, y_pred)
    rmse = np.sqrt(mean_squared_error(y_obs, y_pred))
    mae  = mean_absolute_error(y_obs, y_pred)
    n    = mask.sum()

    # Region + n count as title (one clean line)
    ax.set_title(f"{REG_LABEL.get(reg, reg)}  ($n$={n:,})",
                 fontsize=7.5, pad=3)

    # Metrics as a compact text block inside plot, bottom-right corner
    # (that corner is sparse: positive obs rarely has very negative pred)
    stats_txt = f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}\nMAE={mae:.3f}"
    ax.text(0.97, 0.03, stats_txt,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=6.5, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#bbbbbb", linewidth=0.5, alpha=0.90),
            zorder=10)

    # Axis labels only on edges
    if col_i == 0:
        ax.set_ylabel(r"Predicted (m w.e. yr$^{-1}$)", fontsize=7.5)
    if row_i == nrows - 1:
        ax.set_xlabel(r"Observed (m w.e. yr$^{-1}$)", fontsize=7.5)

# Hide unused axes in the last row
for empty_i in range(n_regs, nrows * ncols):
    axes[empty_i // ncols, empty_i % ncols].set_visible(False)

# Overall title
fold_mean = results["xgb_fold_r2_mean"]
fold_std  = results["xgb_fold_r2_std"]
fig.suptitle(
    f"XGBoost — 5-fold spatial cross-validation  |  "
    f"Overall $R^2$ = {results['metrics']['xgboost']['r2']:.3f}  "
    f"RMSE = {results['metrics']['xgboost']['rmse']:.3f}  "
    f"(fold $R^2$: {fold_mean:.3f} ± {fold_std:.3f})",
    fontsize=8.5, y=1.01)

savefig(fig, "fig02_cv_scatter_grid.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 03 — SHAP beeswarm
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Fig 03 — SHAP beeswarm ...")

TOP_N = 15
top_feats = imp_df.head(TOP_N)["feature"].tolist()
shap_mat  = shap_df[top_feats].values
feat_mat  = feat_df[top_feats].values

fig, ax = plt.subplots(figsize=(W2 * 0.62, TOP_N * 0.38 * CM * 2.54))
rng = np.random.default_rng(0)

for i, fname in enumerate(reversed(top_feats)):
    col = len(top_feats) - 1 - i
    sv  = shap_mat[:, col]
    fv  = feat_mat[:, col]
    yj  = i + rng.uniform(-0.32, 0.32, size=len(sv))

    p1, p99 = np.nanpercentile(fv, 1), np.nanpercentile(fv, 99)
    fv_n = np.clip((fv - p1) / (p99 - p1 + 1e-9), 0, 1)

    ax.scatter(sv, yj, c=fv_n, cmap="RdYlBu_r",
               s=1.8, alpha=0.35, linewidths=0, rasterized=True,
               vmin=0, vmax=1)

ax.axvline(0, color="k", lw=0.7)
ax.set_yticks(range(TOP_N))
ax.set_yticklabels([FEAT_LABEL.get(f, f) for f in reversed(top_feats)], fontsize=7.8)
ax.set_xlabel(r"SHAP value  (impact on predicted SMB, m w.e. yr$^{-1}$)")
ax.set_title("Feature contributions — XGBoost SHAP values\n"
             "(colour: blue = low feature value, red = high)", fontsize=8.5)

# Colorbar
sm = plt.cm.ScalarMappable(cmap="RdYlBu_r",
                            norm=mcolors.Normalize(vmin=0, vmax=1))
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02,
                  ticks=[0, 0.5, 1])
cb.ax.set_yticklabels(["Low", "Mid", "High"], fontsize=7)
cb.set_label("Feature value", fontsize=7.5)
cb.outline.set_linewidth(0.5)

fig.tight_layout()
savefig(fig, "fig03_shap_beeswarm.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 04 — Mean |SHAP| bar chart
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Fig 04 — SHAP importance bar ...")

TOP20 = imp_df.head(20).copy()
TOP20["label"] = TOP20["feature"].map(lambda f: FEAT_LABEL.get(f, f))
vals = TOP20[VAL_COL].values[::-1]
labels = TOP20["label"].values[::-1]

# Colour by category
def feat_category(f):
    if any(x in f for x in ["T_JJA","T_ann","T_DJF","T_MAM","T_SON",
                              "T_mean","T_std","T_trend","continentality",
                              "melt_index"]):
        return "#d73027"   # temperature → red
    if any(x in f for x in ["P_","precip","cryo","P_trend"]):
        return "#4575b4"   # precipitation → blue
    return "#74add1"       # topography → teal

colours = [feat_category(f) for f in TOP20["feature"].values[::-1]]

fig, ax = plt.subplots(figsize=(W1 * 1.05, W1 * 1.3))
bars = ax.barh(range(len(vals)), vals, color=colours,
               edgecolor="white", linewidth=0.4, height=0.72)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=7.5)
ax.set_xlabel(r"Mean $|$SHAP$|$ (m w.e. yr$^{-1}$)")
ax.set_title("Feature importance — XGBoost\n(full HMA, $n$=94,463)", fontsize=8.5)
ax.set_xlim(0, vals.max() * 1.18)

# Value labels on bars
for bar, val in zip(bars, vals):
    ax.text(val + vals.max() * 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=6.5)

# Legend
leg_handles = [
    mpatches.Patch(color="#d73027", label="Temperature"),
    mpatches.Patch(color="#4575b4", label="Precipitation"),
    mpatches.Patch(color="#74add1", label="Topography/derived"),
]
leg = ax.legend(handles=leg_handles, fontsize=7, loc="lower right",
                framealpha=0.85, edgecolor="#cccccc")
leg.get_frame().set_linewidth(0.5)

fig.tight_layout()
savefig(fig, "fig04_shap_bar.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 05 — Regional SHAP heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Fig 05 — SHAP regional heatmap ...")

TOP10 = imp_df.head(10)["feature"].tolist()
reg_shap = (shap_df.groupby("subregion")[TOP10]
                   .mean()
                   .rename(index=REG_LABEL))

# Short two-line labels for heatmap columns — avoids x-axis overlap
HEATMAP_COL = {
    "T_JJA_std_C":        "JJA T\nvariability",
    "P_JJA_mm":           "JJA\nprecip.",
    "zmean_m":            "Mean\nelevation",
    "slope_deg":          "Slope",
    "continentality":     "Continen-\ntality",
    "precip_winter_frac": "Winter\nprecip. frac.",
    "P_SON_mm":           "SON\nprecip.",
    "elev_range_m":       "Elev.\nrange",
    "T_MAM_C":            "MAM\ntemp.",
    "area_km2":           "Glacier\narea",
    "cryo_balance":       "Cryo-\nbalance",
    "T_JJA_max_C":        "JJA T\nmax",
    "T_JJA_lapse_C":      "JJA T\nlapse",
    "P_snow_mm":          "Cold-season\nprecip.",
    "P_std_mm":           "Precip.\nvariability",
}
reg_shap.columns = [HEATMAP_COL.get(c, c) for c in reg_shap.columns]

# Sort rows by overall mass balance (most negative at top)
smb_order = (feat_df.groupby("subregion")["SMB_mwea"].mean()
                    .rename(index=REG_LABEL)
                    .sort_values().index.tolist())
reg_shap = reg_shap.reindex(smb_order)

vabs = np.abs(reg_shap.values).max()
# Taller figure so 12 rows × 10 cols have room for annotations
fig, ax = plt.subplots(figsize=(W2, W2 * 0.72))

sns.heatmap(reg_shap, ax=ax, cmap="RdBu_r", center=0,
            vmin=-vabs * 0.9, vmax=vabs * 0.9,
            annot=True, fmt=".2f",
            annot_kws={"fontsize": 6.0, "fontfamily": "sans-serif"},
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": r"Mean SHAP (m w.e. yr$^{-1}$)",
                      "shrink": 0.75, "aspect": 25, "pad": 0.02})

ax.set_title(
    "Regional SHAP contributions — signed mean per sub-region\n"
    "(red = drives mass loss  ·  blue = drives mass gain)",
    fontsize=8.5, pad=6)
ax.set_xlabel(""); ax.set_ylabel("")
# x-labels: 45° right-aligned avoids collision between long two-line labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                   ha="right", va="top", fontsize=7.5, multialignment="center")
ax.tick_params(axis="y", rotation=0, labelsize=7.8)
ax.tick_params(axis="x", length=0)   # suppress tick marks — labels are enough

fig.tight_layout(pad=0.5)
savefig(fig, "fig05_shap_heatmap.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 06 — SHAP dependence plots top 3
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Fig 06 — SHAP dependence top 3 ...")

TOP3 = imp_df.head(3)["feature"].tolist()
from scipy.stats import binned_statistic

# Panel layout: taller, with room for legend below
fig, axes = plt.subplots(1, 3, figsize=(W2, W2 * 0.52))
fig.subplots_adjust(wspace=0.38, bottom=0.28)

PANEL_LETTERS = ["(a)", "(b)", "(c)"]
rng = np.random.default_rng(42)

for ax_i, (ax, fname) in enumerate(zip(axes, TOP3)):
    fv  = feat_df[fname].values
    sv  = shap_df[fname].values
    reg = shap_df["subregion"].values

    # ── Layer 1: subsampled scatter coloured by region ──────────────────────
    # 300 pts/region keeps the distribution visible without overplotting
    for r in SUBREGIONS:
        idx = np.where(reg == r)[0]
        chosen = rng.choice(idx, min(300, len(idx)), replace=False)
        ax.scatter(fv[chosen], sv[chosen],
                   color=PALETTE[r], s=3.5, alpha=0.45, linewidths=0,
                   rasterized=True, zorder=2,
                   label=REG_LABEL.get(r, r))

    # ── Layer 2: binned IQR band + median trend ─────────────────────────────
    valid_mask = ~(np.isnan(fv) | np.isnan(sv))
    fv_v, sv_v = fv[valid_mask], sv[valid_mask]
    bin_edges = np.percentile(fv_v, np.linspace(2, 98, 26))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) > 5:
        med, _, _  = binned_statistic(fv_v, sv_v, statistic="median",             bins=bin_edges)
        p25, _, _  = binned_statistic(fv_v, sv_v, statistic=lambda x: np.percentile(x, 25), bins=bin_edges)
        p75, _, _  = binned_statistic(fv_v, sv_v, statistic=lambda x: np.percentile(x, 75), bins=bin_edges)
        xmid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ok   = ~np.isnan(med)
        ax.fill_between(xmid[ok], p25[ok], p75[ok],
                        color="#333333", alpha=0.13, zorder=4,
                        label="_nolegend_")
        ax.plot(xmid[ok], med[ok], color="k", lw=2.2, zorder=5,
                solid_capstyle="round", label="_nolegend_")

    ax.axhline(0, color="#777", lw=0.7, ls="--", zorder=3)

    # ── Annotations ─────────────────────────────────────────────────────────
    mean_abs = np.abs(sv).mean()
    # Panel letter — upper left
    ax.text(0.03, 0.97, PANEL_LETTERS[ax_i],
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, fontweight="bold")
    # Mean |SHAP| — upper right, small clean box
    ax.text(0.97, 0.97,
            f"mean $|$SHAP$|$\n= {mean_abs:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#bbbbbb", lw=0.5, alpha=0.90),
            zorder=10)

    # Feature name as x-label; y-label only on leftmost panel
    ax.set_xlabel(FEAT_LABEL.get(fname, fname), fontsize=8)
    if ax_i == 0:
        ax.set_ylabel(r"SHAP value (m w.e. yr$^{-1}$)", fontsize=8)
    else:
        ax.set_ylabel("")

# ── Shared legend ────────────────────────────────────────────────────────────
reg_handles = [mpatches.Patch(color=PALETTE[r], label=REG_LABEL.get(r, r))
               for r in sorted(SUBREGIONS)]
trend_handle = Line2D([0],[0], color="k", lw=2.2, label="Median trend")
iqr_handle   = mpatches.Patch(color="#333333", alpha=0.25,
                               label="IQR (25th–75th pctile)")
all_handles  = reg_handles + [trend_handle, iqr_handle]

leg6 = fig.legend(handles=all_handles, loc="lower center", ncol=7,
                  fontsize=7, bbox_to_anchor=(0.5, 0.0),
                  framealpha=0.95, edgecolor="#cccccc",
                  handlelength=1.4, handleheight=0.9, borderpad=0.6,
                  columnspacing=0.8)
leg6.get_frame().set_linewidth(0.5)

fig.suptitle("SHAP dependence — top 3 climate–terrain drivers",
             fontsize=9, y=1.02)
savefig(fig, "fig06_shap_dependence.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 07 — SMB violin by sub-region
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] Fig 07 — SMB violin ...")

smb_df = feat_df[["subregion","SMB_mwea"]].copy()
smb_df["Region"] = smb_df["subregion"].map(REG_LABEL)
palette_label = {REG_LABEL[k]: v for k, v in PALETTE.items()}

order = (smb_df.groupby("Region")["SMB_mwea"]
               .median().sort_values().index.tolist())

fig, ax = plt.subplots(figsize=(W2, W2 * 0.42))

sns.violinplot(data=smb_df, x="Region", y="SMB_mwea",
               order=order, hue="Region", palette=palette_label,
               inner="quartile", linewidth=0.7, ax=ax,
               cut=0, legend=False)

ax.axhline(0, color="k", lw=0.7, ls="--", zorder=5)
ax.axhline(smb_df["SMB_mwea"].mean(), color="#555", lw=0.7,
           ls=":", zorder=5, label=f"HMA mean = {smb_df['SMB_mwea'].mean():.3f}")

ax.set_xlabel("")
ax.set_ylabel(SMB_UNIT)
ax.set_title("Geodetic mass balance distribution by sub-region  (2000–2018)\n"
             "Inner lines: 25th, 50th, 75th percentiles", fontsize=8.5)
ax.tick_params(axis="x", rotation=30, labelsize=7.8)
leg7 = ax.legend(fontsize=7.5, framealpha=0.85, edgecolor="#cccccc",
                 loc="lower right")
leg7.get_frame().set_linewidth(0.5)

for i, reg in enumerate(order):
    n = (smb_df["Region"] == reg).sum()
    ax.text(i, ax.get_ylim()[0] + 0.04, f"$n$={n:,}",
            ha="center", va="bottom", fontsize=5.8, color="#555")

fig.tight_layout()
savefig(fig, "fig07_smb_violin.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
pdfs = sorted(FIG_DIR.glob("fig0*.pdf"))
print(f"\n  {len(pdfs)} vector PDF figures saved to: {FIG_DIR}")
for f in pdfs:
    print(f"    {f.name}  ({f.stat().st_size/1e3:.0f} kB)")
print("\n  Next: run scripts/09_figures_projections_hma.py (also needs PDF upgrade)")
print("=" * 70)
