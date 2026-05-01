"""
Script 11 — Methodology Framework Figures (vector PDF, publication quality)

figA_model_architecture.pdf  — Model layers + two-stage training paradigm
figB_technical_framework.pdf — Full data→attribution→projection pipeline
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

CM     = 1 / 2.54
W1     = 8.3  * CM
W2     = 17.1 * CM

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150,
})

OUT_DIR = Path(__file__).parent.parent / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
C_GRAY   = "#e8e8e8"    # stage-1 background
C_ORANGE = "#fff0de"    # stage-2 background
C_BLUE   = "#ddeeff"    # input boxes
C_GREEN  = "#e6f4ea"    # output / result boxes
C_RED    = "#fde8e8"    # XGBoost highlight
C_PURPLE = "#ede8f4"    # projection boxes
C_WHITE  = "#ffffff"
EDGE     = "#666666"
ARROW    = "#444444"

def box(ax, x, y, w, h, label, sublabel=None,
        fc=C_WHITE, ec=EDGE, lw=0.8,
        fontsize=7.5, subfontsize=6.5,
        bold=False, radius=0.02):
    """Draw a rounded rectangle with centred text."""
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad={radius}",
                          fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x, y + (0.012 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight=weight, zorder=4, wrap=False)
    if sublabel:
        ax.text(x, y - 0.022, sublabel,
                ha="center", va="center", fontsize=subfontsize,
                color="#555555", zorder=4)

def arrow(ax, x0, y0, x1, y1, label=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=ARROW,
                                lw=0.9, mutation_scale=8),
                zorder=5)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx+0.01, my, label, fontsize=6, color="#555555",
                ha="left", va="center", zorder=6)

def bg_rect(ax, x0, y0, x1, y1, fc, label=None, ec="#cccccc"):
    r = FancyBboxPatch((x0, y0), x1-x0, y1-y0,
                       boxstyle="round,pad=0.005",
                       fc=fc, ec=ec, lw=0.6, zorder=1, alpha=0.85)
    ax.add_patch(r)
    if label:
        ax.text(x0+0.008, y1-0.01, label,
                fontsize=6, color="#888888", va="top", ha="left",
                fontstyle="italic", zorder=2)

def savefig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    print(f"  Saved {name}  ({path.stat().st_size/1e3:.0f} kB)")
    plt.close(fig)

print("=" * 70)
print("SCRIPT 11 — FRAMEWORK FIGURES")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# Fig A — Model Architecture  (portrait-ish, single column)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Fig A — Model architecture ...")

fig, ax = plt.subplots(figsize=(W2, W2 * 0.95))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis("off")

# ── Stage backgrounds ──────────────────────────────────────────────────────
bg_rect(ax, 0.02, 0.60, 0.98, 0.99, C_GRAY,
        label="Stage 1 — Data preparation & global training")
bg_rect(ax, 0.02, 0.20, 0.98, 0.59, C_ORANGE,
        label="Stage 2 — Model comparison & selection")
bg_rect(ax, 0.02, 0.01, 0.98, 0.19, C_GREEN,
        label="Outputs")

# ── Stage 1 — Row 1: Raw inputs (y=0.88) ──────────────────────────────────
box(ax, 0.18, 0.88, 0.28, 0.08,
    "ERA5 Reanalysis", "Monthly T & P · 2000–2018\n0.25° × 0.25° grid",
    fc=C_BLUE, fontsize=7.5, subfontsize=6.2)
box(ax, 0.50, 0.88, 0.26, 0.08,
    "RGI 6.0 Inventory", "94,463 glaciers · HMA\nArea, elevation, slope, aspect",
    fc=C_BLUE, fontsize=7.5, subfontsize=6.2)
box(ax, 0.82, 0.88, 0.28, 0.08,
    "Geodetic SMB", "Shean et al. 2020\n2000–2018 · m w.e. yr-1  (labels)",
    fc=C_BLUE, fontsize=7.5, subfontsize=6.2)

# arrows down to feature engineering
for xc in [0.18, 0.50, 0.82]:
    arrow(ax, xc, 0.84, xc, 0.79)

# ── Stage 1 — Row 2: Feature engineering (y=0.73) ─────────────────────────
box(ax, 0.18, 0.74, 0.28, 0.08,
    "Seasonal Climate Indices",
    "JJA/DJF/MAM/SON T & P\nTrends · variability (σ)",
    fc=C_GRAY, fontsize=7.5, subfontsize=6.2)
box(ax, 0.50, 0.74, 0.26, 0.08,
    "Physics-Based Features",
    "Lapse-rate T · Cryo-balance\nMelt index · Continentality",
    fc=C_GRAY, fontsize=7.5, subfontsize=6.2)
box(ax, 0.82, 0.74, 0.28, 0.08,
    "Glacier Morphology",
    "log(area) · Elev. range\nAspect sin/cos · Slope",
    fc=C_GRAY, fontsize=7.5, subfontsize=6.2)

# merge arrows to CV box
for xc in [0.18, 0.50, 0.82]:
    arrow(ax, xc, 0.70, xc, 0.65)
ax.annotate("", xy=(0.50, 0.64), xytext=(0.18, 0.64),
            arrowprops=dict(arrowstyle="-", color=ARROW, lw=0.8), zorder=5)
ax.annotate("", xy=(0.50, 0.64), xytext=(0.82, 0.64),
            arrowprops=dict(arrowstyle="-", color=ARROW, lw=0.8), zorder=5)
arrow(ax, 0.50, 0.64, 0.50, 0.615)

# ── Stage 1 — Row 3: Spatial CV (y=0.59) ──────────────────────────────────
box(ax, 0.50, 0.61, 0.50, 0.075,
    "5-Fold Spatial Cross-Validation",
    "GroupKFold by sub-region · prevents spatial data leakage · n = 94,463",
    fc=C_WHITE, ec="#888888", lw=1.0, fontsize=7.5, subfontsize=6.2, bold=False)

arrow(ax, 0.50, 0.572, 0.50, 0.545)

# ── Stage 2 — Row 4: Three models (y=0.48) ────────────────────────────────
box(ax, 0.18, 0.49, 0.28, 0.08,
    "Ridge Regression",
    "Linear baseline\nR² = 0.109",
    fc=C_WHITE, fontsize=7.5, subfontsize=6.5)
box(ax, 0.50, 0.49, 0.26, 0.08,
    "Random Forest",
    "100 trees · max_depth=8\nR² = 0.154",
    fc=C_WHITE, fontsize=7.5, subfontsize=6.5)
box(ax, 0.82, 0.49, 0.28, 0.08,
    "XGBoost  (best)",
    "300 trees · lr=0.05\nR² = 0.164  ← selected",
    fc=C_RED, ec="#cc4444", lw=1.1, fontsize=7.5, subfontsize=6.5, bold=True)

# arrows from CV to models
for xc in [0.18, 0.50, 0.82]:
    arrow(ax, 0.50, 0.545, xc, 0.53)

# arrows from models to evaluation
for xc in [0.18, 0.50, 0.82]:
    arrow(ax, xc, 0.45, xc, 0.405)
ax.annotate("", xy=(0.50, 0.405), xytext=(0.18, 0.405),
            arrowprops=dict(arrowstyle="-", color=ARROW, lw=0.8), zorder=5)
ax.annotate("", xy=(0.50, 0.405), xytext=(0.82, 0.405),
            arrowprops=dict(arrowstyle="-", color=ARROW, lw=0.8), zorder=5)
arrow(ax, 0.50, 0.405, 0.50, 0.38)

# ── Stage 2 — Row 5: Evaluation & selection (y=0.36) ──────────────────────
box(ax, 0.50, 0.36, 0.50, 0.065,
    "Model Evaluation & Selection",
    "R² · RMSE · MAE · Bias · Spatial fold R² (mean ± σ)  →  XGBoost selected",
    fc=C_WHITE, ec="#888888", fontsize=7.5, subfontsize=6.2)

arrow(ax, 0.50, 0.327, 0.50, 0.295)

# ── Stage 2 — Row 6: SHAP + Projections (y=0.26) ─────────────────────────
box(ax, 0.28, 0.265, 0.40, 0.065,
    "SHAP Attribution",
    "TreeExplainer · feature importance\nregional signed SHAP patterns",
    fc=C_ORANGE, fontsize=7.2, subfontsize=6.2)
box(ax, 0.74, 0.265, 0.40, 0.065,
    "CMIP6 Bias-Corrected Projections",
    "Delta method · SSP2-4.5 / SSP5-8.5\n3-model ensemble · 2020–2099",
    fc=C_PURPLE, fontsize=7.2, subfontsize=6.2)

arrow(ax, 0.50, 0.327, 0.28, 0.298)
arrow(ax, 0.50, 0.327, 0.74, 0.298)

# arrows to outputs
arrow(ax, 0.28, 0.232, 0.28, 0.185)
arrow(ax, 0.74, 0.232, 0.74, 0.185)

# ── Outputs (y=0.12) ───────────────────────────────────────────────────────
box(ax, 0.18, 0.135, 0.28, 0.07,
    "Attribution Maps",
    "Top-15 SHAP features\nRegional SHAP heatmap",
    fc=C_GREEN, fontsize=7.2, subfontsize=6.2)
box(ax, 0.50, 0.135, 0.26, 0.07,
    "Performance Tables",
    "CV R²/RMSE by model\n& sub-region",
    fc=C_GREEN, fontsize=7.2, subfontsize=6.2)
box(ax, 0.82, 0.135, 0.28, 0.07,
    "Projection Maps",
    "ΔSMB 2080–2099\nSSP2-4.5 & SSP5-8.5",
    fc=C_GREEN, fontsize=7.2, subfontsize=6.2)

arrow(ax, 0.28, 0.185, 0.18, 0.17)
arrow(ax, 0.28, 0.185, 0.50, 0.17)
arrow(ax, 0.74, 0.185, 0.82, 0.17)

ax.set_title("Model architecture and training paradigm",
             fontsize=9, fontweight="bold", pad=6)
savefig(fig, "figA_model_architecture.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig B — Technical Framework  (horizontal flowchart, landscape)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Fig B — Technical framework ...")

fig, ax = plt.subplots(figsize=(W2, W2 * 0.68))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis("off")

# Five column x-centres
XC = [0.09, 0.28, 0.50, 0.72, 0.91]
COL_LABELS = ["Data Sources", "Pre-processing", "ML Modelling",
              "Attribution", "Projections"]
COL_FC     = [C_BLUE, C_GRAY, C_ORANGE, "#ede8f4", C_PURPLE]

# Column header bands
for xc, lbl, fc in zip(XC, COL_LABELS, COL_FC):
    bg_rect(ax, xc-0.087, 0.01, xc+0.087, 0.99, fc, ec="#cccccc")
    ax.text(xc, 0.94, lbl, ha="center", va="center",
            fontsize=8, fontweight="bold", color="#333333", zorder=4)
    ax.plot([xc-0.087, xc+0.087], [0.91, 0.91],
            color="#cccccc", lw=0.6, zorder=3)

# ── Column 1: Data Sources ─────────────────────────────────────────────────
box(ax, XC[0], 0.80, 0.155, 0.075,
    "ERA5 Reanalysis",
    "Monthly T & P\n2000–2018, 0.25°",
    fc=C_BLUE, fontsize=7, subfontsize=6)
box(ax, XC[0], 0.67, 0.155, 0.075,
    "RGI 6.0 Glacier\nInventory",
    "94,463 glaciers\nHMA (RGI-13/14/15)",
    fc=C_BLUE, fontsize=7, subfontsize=6)
box(ax, XC[0], 0.54, 0.155, 0.075,
    "Geodetic SMB",
    "Shean et al. 2020\nm w.e. yr-1 labels",
    fc=C_BLUE, fontsize=7, subfontsize=6)
box(ax, XC[0], 0.35, 0.155, 0.075,
    "CMIP6 Models",
    "MIROC6\nMPI-ESM1-2-LR\nIPSL-CM6A-LR",
    fc=C_BLUE, fontsize=7, subfontsize=6)

# ── Column 2: Pre-processing ───────────────────────────────────────────────
box(ax, XC[1], 0.80, 0.155, 0.075,
    "ERA5 Extraction",
    "Nearest-neighbour at\nglacier centroids",
    fc=C_GRAY, fontsize=7, subfontsize=6)
box(ax, XC[1], 0.67, 0.155, 0.075,
    "Feature Engineering",
    "Seasonal indices\nPhysics-based features\n(lapse, cryo-balance…)",
    fc=C_GRAY, fontsize=7, subfontsize=6)
box(ax, XC[1], 0.54, 0.155, 0.075,
    "Outlier Filtering",
    "|SMB| > 5.0 removed\nσ_SMB > 1.0 removed",
    fc=C_GRAY, fontsize=7, subfontsize=6)
box(ax, XC[1], 0.35, 0.155, 0.075,
    "Delta Bias\nCorrection",
    "T_bc = ERA5 + ΔCMIP6\nP_bc = ERA5 × ratio",
    fc=C_GRAY, fontsize=7, subfontsize=6)

# ── Column 3: ML Modelling ─────────────────────────────────────────────────
box(ax, XC[2], 0.82, 0.155, 0.055,
    "Ridge Regression",
    "Linear baseline",
    fc=C_WHITE, fontsize=7, subfontsize=6)
box(ax, XC[2], 0.74, 0.155, 0.055,
    "Random Forest",
    "100 trees",
    fc=C_WHITE, fontsize=7, subfontsize=6)
box(ax, XC[2], 0.66, 0.155, 0.055,
    "XGBoost  (primary)",
    "300 trees · lr=0.05",
    fc=C_RED, ec="#cc4444", fontsize=7, subfontsize=6, bold=True)
box(ax, XC[2], 0.54, 0.155, 0.07,
    "Spatial GroupKFold CV",
    "5 folds · grouped by\nsub-region · n=94,463",
    fc=C_WHITE, ec="#888888", lw=1.0, fontsize=7, subfontsize=6)

# bracket linking 3 models
ax.plot([XC[2]-0.077, XC[2]-0.077], [0.635, 0.845],
        color="#888888", lw=0.7, zorder=5)
ax.plot([XC[2]-0.077, XC[2]-0.06], [0.74, 0.74],
        color="#888888", lw=0.7, zorder=5)

# ── Column 4: Attribution ──────────────────────────────────────────────────
box(ax, XC[3], 0.80, 0.155, 0.075,
    "SHAP TreeExplainer",
    "Global feature\nimportance ranking",
    fc=C_PURPLE, fontsize=7, subfontsize=6)
box(ax, XC[3], 0.67, 0.155, 0.075,
    "Regional SHAP\nPatterns",
    "Signed mean SHAP\nper sub-region",
    fc=C_PURPLE, fontsize=7, subfontsize=6)
box(ax, XC[3], 0.54, 0.155, 0.075,
    "Dependence Plots",
    "SHAP vs feature value\ntop 3 drivers",
    fc=C_PURPLE, fontsize=7, subfontsize=6)

# ── Column 5: Projections ──────────────────────────────────────────────────
box(ax, XC[4], 0.80, 0.155, 0.075,
    "SSP2-4.5 Projection",
    "2020–2099 · 4 windows\n3-model ensemble mean",
    fc=C_GREEN, fontsize=7, subfontsize=6)
box(ax, XC[4], 0.67, 0.155, 0.075,
    "SSP5-8.5 Projection",
    "2020–2099 · 4 windows\n3-model ensemble mean",
    fc=C_GREEN, fontsize=7, subfontsize=6)
box(ax, XC[4], 0.54, 0.155, 0.075,
    "Uncertainty\nQuantification",
    "Ensemble σ by\nregion & window",
    fc=C_GREEN, fontsize=7, subfontsize=6)

# ── Lower row: Key outputs ─────────────────────────────────────────────────
OUT_Y = 0.20
out_items = [
    (XC[0], "94,463\nGlaciers"),
    (XC[1], "29 Input\nFeatures"),
    (XC[2], "XGBoost\nR²=0.164"),
    (XC[3], "JJA T var.\n#1 Driver"),
    (XC[4], "ΔSMB\n2080–2099"),
]
for xc, lbl in out_items:
    box(ax, xc, OUT_Y, 0.145, 0.07, lbl,
        fc="#fffde7", ec="#ccaa00", lw=0.8, fontsize=7.5, bold=False)

ax.plot([0.02, 0.98], [0.315, 0.315], color="#cccccc", lw=0.6, ls="--", zorder=2)
ax.text(0.5, 0.305, "Key metrics / outputs per stage",
        ha="center", va="top", fontsize=6, color="#888888",
        fontstyle="italic", zorder=3)

# ── Horizontal flow arrows between columns ─────────────────────────────────
for i in range(4):
    for yc in [0.80, 0.67, 0.54]:
        arrow(ax, XC[i]+0.078, yc, XC[i+1]-0.078, yc)

# CMIP6 arrow goes to projections column (col5) via col2 bias correction
arrow(ax, XC[0]+0.078, 0.35, XC[1]-0.078, 0.35)
arrow(ax, XC[1]+0.078, 0.35, XC[4]-0.078, 0.67)

ax.set_title("Technical framework: data sources → ML attribution → future projections",
             fontsize=9, fontweight="bold", pad=6)
savefig(fig, "figB_technical_framework.pdf")

print("\n" + "=" * 70)
pdfs = [FIG_DIR / "figA_model_architecture.pdf",
        FIG_DIR / "figB_technical_framework.pdf"]
for f in pdfs:
    if f.exists():
        print(f"  {f.name}  ({f.stat().st_size/1e3:.0f} kB)")
print("=" * 70)
