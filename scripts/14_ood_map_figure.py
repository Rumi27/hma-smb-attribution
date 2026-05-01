"""
Script 14 — OOD Mahalanobis Map Figure

Computes per-glacier Mahalanobis D² from the training distribution
and plots a geographic map colouring glaciers by OOD status.
Threshold: empirical 97.5th percentile of training D² (= 81.0).

Output:
  outputs/figures/fig_ood_mahalanobis.pdf
  outputs/ood_mahalanobis_per_glacier.parquet
"""
import pandas as pd, numpy as np, json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from scipy.stats import chi2

BASE    = Path(__file__).parent.parent
DATA    = BASE / "data"
OUT_DIR = BASE / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42,
})

CM = 1 / 2.54
W2 = 17.1 * CM

print("=" * 70)
print("SCRIPT 14 — OOD MAHALANOBIS MAP")
print("=" * 70)

feat    = pd.read_parquet(DATA / "feature_matrix_hma.parquet")
results = json.loads((OUT_DIR / "model_results_hma.json").read_text())
FEATURE_COLS = results["features"]

X  = feat[FEATURE_COLS].values.astype(float)
mu = X.mean(axis=0)
cov_inv = np.linalg.pinv(np.cov(X.T))

# Vectorised Mahalanobis²
diff = X - mu
mah2 = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
feat = feat.copy()
feat["mah2"] = mah2

THRESHOLD = np.percentile(mah2, 97.5)
feat["ood"]  = mah2 > THRESHOLD
feat["ood_level"] = pd.cut(mah2,
    bins=[0, np.percentile(mah2, 75), THRESHOLD, np.percentile(mah2, 99), mah2.max() + 1],
    labels=["Low (< 75th)", "Moderate (75–97.5th)", "High (97.5–99th)", "Extreme (> 99th)"])

print(f"  Mahalanobis² 97.5th percentile: {THRESHOLD:.1f}")
print(f"  Total OOD glaciers (>97.5th): {feat['ood'].sum():,} ({100*feat['ood'].mean():.1f}%)")

# Per sub-region fractions
print("\n  OOD fraction by sub-region:")
for reg, grp in feat.groupby("subregion"):
    frac = 100 * grp["ood"].mean()
    print(f"    {reg:<25}: {frac:.1f}%")

# Save per-glacier table
feat[["RGIId","subregion","rgi_region","cenlat","cenlon",
      "SMB_mwea","area_km2","mah2","ood","ood_level"]].to_parquet(
    OUT_DIR / "ood_mahalanobis_per_glacier.parquet", index=False)

# ── Map figure ────────────────────────────────────────────────────────────────
try:
    import geopandas as gpd
    NE_SHP = Path("/tmp/ne_110m/ne_110m_admin_0_countries.shp")
    world  = gpd.read_file(NE_SHP) if NE_SHP.exists() else None
except Exception:
    world = None

fig, ax = plt.subplots(figsize=(W2, W2 * 0.55))

# Background land
if world is not None:
    world.plot(ax=ax, color="#f0f0f0", edgecolor="#999999",
               linewidth=0.35, zorder=0)

# Plot glaciers by OOD level
LEVEL_COLORS = {
    "Low (< 75th)":           "#2166ac",   # blue
    "Moderate (75–97.5th)":   "#92c5de",   # light blue
    "High (97.5–99th)":       "#f4a582",   # orange
    "Extreme (> 99th)":       "#d73027",   # red
}
SIZES = {
    "Low (< 75th)": 1.5,
    "Moderate (75–97.5th)": 2.0,
    "High (97.5–99th)": 3.5,
    "Extreme (> 99th)": 5.0,
}

for level, color in LEVEL_COLORS.items():
    mask = feat["ood_level"] == level
    if not mask.any(): continue
    ax.scatter(feat.loc[mask, "cenlon"], feat.loc[mask, "cenlat"],
               s=SIZES[level], c=color, alpha=0.65, linewidths=0,
               rasterized=True, zorder=3, label=level)

# Sub-region text labels for the two OOD hot-spots
for lon, lat, txt in [(92, 28.2, "E. Himalaya"), (84, 28.5, "C. Himalaya")]:
    ax.text(lon, lat, txt, fontsize=6.5, color="#d73027",
            ha="center", va="center", style="italic", fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75))

ax.set_xlim(63.5, 107); ax.set_ylim(24, 48)
ax.set_aspect("equal")
ax.set_xlabel("Longitude (°E)", fontsize=8)
ax.set_ylabel("Latitude (°N)", fontsize=8)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f°E"))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f°N"))
ax.tick_params(labelsize=7.5)
ax.grid(False)

leg = ax.legend(title=f"Mahalanobis D² percentile\n(threshold = 97.5th = {THRESHOLD:.0f})",
                fontsize=6.5, title_fontsize=7, loc="upper right",
                framealpha=0.92, edgecolor="#cccccc", markerscale=2.5,
                handlelength=1.0)
leg.get_frame().set_linewidth(0.5)

ax.set_title(
    "Per-glacier out-of-distribution (OOD) status — Mahalanobis distance from training distribution\n"
    f"Training D² 97.5th percentile = {THRESHOLD:.0f}; "
    f"OOD fraction: E. Himalaya 17.3%, C. Himalaya 7.8%, others <4.4%",
    fontsize=7.5, pad=4)

fig.tight_layout()
out_path = FIG_DIR / "fig_ood_mahalanobis.pdf"
fig.savefig(out_path)
plt.close(fig)
print(f"\n  Saved: {out_path}  ({out_path.stat().st_size/1e3:.0f} kB)")
print("=" * 70)
