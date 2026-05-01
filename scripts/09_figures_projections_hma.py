"""
Script 09 — Projection Figures (vector PDF, publication quality)

Figures:
  fig08_projection_timeseries.pdf  — Regional mean SMB vs time, SSP2-4.5 & SSP5-8.5
  fig09_projection_map_2080.pdf    — Spatial map of projected SMB change (2080–2099)
  fig10_scenario_comparison.pdf    — Bar: ΔSMB by sub-region for both scenarios
  fig11_uncertainty_spread.pdf     — Ensemble spread by region
"""

import pandas as pd
import numpy as np
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path

CM = 1 / 2.54
W1 = 8.3  * CM
W2 = 17.1 * CM

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.linewidth": 0.4, "grid.color": "#e0e0e0",
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42,
})

OUT_DIR = Path(__file__).parent.parent / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("SCRIPT 09 — PROJECTION FIGURES")
print("=" * 70)

proj_df  = pd.read_parquet(OUT_DIR / "smb_projections_hma.parquet")
summary  = pd.read_csv(OUT_DIR / "smb_projection_summary.csv")
feat_df  = pd.read_parquet(
    Path(__file__).parent.parent / "data" / "feature_matrix_hma.parquet")

SUBREGIONS = sorted(proj_df["subregion"].unique())
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
    "Other_CA":       "Other C.A.",
    "Pamir_Alay":     "Pamir-Alay",
    "Qilian_Shan":    "Qilian Shan",
    "W_Himalaya":     "W. Himalaya",
    "W_Nepal":        "W. Nepal",
    "W_Pamir":        "W. Pamir",
}

WINDOWS     = ["2020-2039","2040-2059","2060-2079","2080-2099"]
WIN_CENTERS = [2029.5, 2049.5, 2069.5, 2089.5]
SCENARIOS   = {"ssp245":"SSP2-4.5","ssp585":"SSP5-8.5"}
SC_COLOURS  = {"ssp245":"#2166ac","ssp585":"#d73027"}
SMB_UNIT    = r"SMB (m w.e. yr$^{-1}$)"
DSMB_UNIT   = r"$\Delta$SMB (m w.e. yr$^{-1}$)"

def get_ensmean(sc, w):
    col = f"SMB_{sc}_{w}_ensmean"
    return col if col in proj_df.columns else None

def get_ensstd(sc, w):
    col = f"SMB_{sc}_{w}_ensstd"
    return col if col in proj_df.columns else None

def savefig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path)
    print(f"    Saved {name}  ({path.stat().st_size/1e3:.0f} kB)")
    plt.close(fig)

# Sub-regions where the model is out-of-distribution (OOD): projected
# monsoon precipitation increases push feature values outside the training
# range, producing spurious positive ΔSMB that are not physical.
OOD_SUBREGIONS = {"E_Himalaya", "C_Himalaya"}

available_combos = [(sc, w) for sc in SCENARIOS for w in WINDOWS
                    if get_ensmean(sc, w) is not None]
print(f"  Available scenario×window combos: {len(available_combos)}")
if not available_combos:
    print("  No projection columns found. Run script 08 first.")
    raise SystemExit(0)

# Sort sub-regions by historical SMB
REGS_SORTED = sorted(SUBREGIONS,
    key=lambda r: proj_df.loc[proj_df["subregion"]==r,"SMB_hist"].mean())

# ══════════════════════════════════════════════════════════════════════════════
# Fig 08 — Regional time series (3×4 grid)  — fig02-style layout
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Fig 08 — Projection time series ...")

import string
LETTERS = list(string.ascii_lowercase)   # a–l for panel labels

n_regs = len(REGS_SORTED)
_ncols = 4
_nrows = (n_regs + _ncols - 1) // _ncols
fig, axes = plt.subplots(_nrows, _ncols,
                         figsize=(W2, W2 * (_nrows / 3) * 0.80),
                         sharex=True, sharey=False,
                         constrained_layout=True)

for ax_i, reg in enumerate(REGS_SORTED):
    ax   = axes[ax_i // _ncols, ax_i % _ncols]
    mask = proj_df["subregion"] == reg
    smb_hist = proj_df.loc[mask, "SMB_hist"].mean()

    # ── Shade historical period ───────────────────────────────────────────
    ax.axvspan(2000, 2015, color="#e8e8e8", zorder=0, lw=0)

    # ── Historical mean reference line ───────────────────────────────────
    ax.axhline(smb_hist, color="#444", lw=0.9, ls="--", zorder=3, alpha=0.8)
    ax.axhline(0,        color="#bbb", lw=0.4, zorder=2)

    # ── Scenario lines + ensemble spread ─────────────────────────────────
    delta_ssp585 = None
    for sc, sc_label in SCENARIOS.items():
        xs = [2007] + WIN_CENTERS
        ys, lo, hi = [smb_hist], [smb_hist], [smb_hist]
        for w in WINDOWS:
            cm = get_ensmean(sc, w); cs = get_ensstd(sc, w)
            if cm is None: continue
            mv = proj_df.loc[mask, cm].mean()
            sv = proj_df.loc[mask, cs].mean() if cs else 0
            ys.append(mv); lo.append(mv - sv); hi.append(mv + sv)

        ax.fill_between(xs[:len(ys)], lo, hi,
                        color=SC_COLOURS[sc], alpha=0.18, zorder=1, lw=0)
        ax.plot(xs[:len(ys)], ys, color=SC_COLOURS[sc],
                lw=1.5, marker="o", ms=3.2, zorder=4,
                markeredgewidth=0, clip_on=False)

        if sc == "ssp585" and len(ys) == 5:
            delta_ssp585 = ys[-1] - smb_hist   # Δ at 2080–2099

    # ── Panel letter (upper-left) ─────────────────────────────────────────
    ax.text(0.03, 0.97, f"({LETTERS[ax_i]})",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, fontweight="bold")

    # ── Region title (centred) ────────────────────────────────────────────
    ax.set_title(REG_LABEL.get(reg, reg), fontsize=8, pad=2)

    # ── ΔSMB annotation (bottom-right, fig02 style) ───────────────────────
    if delta_ssp585 is not None:
        sign = "+" if delta_ssp585 > 0 else ""
        ax.text(0.97, 0.97,
                f"SSP5-8.5\n2080–99: {sign}{delta_ssp585:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6.2, linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#bbbbbb", lw=0.5, alpha=0.92),
                zorder=10)

    # ── Axis labels — only outer edges ───────────────────────────────────
    ax.set_xlim(2000, 2100)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(40))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(20))
    if ax_i % _ncols == 0:
        ax.set_ylabel(SMB_UNIT, fontsize=7.5)
    if ax_i // _ncols == _nrows - 1:
        ax.set_xlabel("Year", fontsize=7.5)

# Hide unused panels in the last row
for _ei in range(n_regs, _nrows * _ncols):
    axes[_ei // _ncols, _ei % _ncols].set_visible(False)

# ── Shared legend (bottom-centre) ────────────────────────────────────────────
handles = [
    mpatches.Patch(color="#e8e8e8", ec="#cccccc", lw=0.5, label="Historical period (2000–2014)"),
    plt.Line2D([0],[0], color="#444", ls="--", lw=0.9, label="2000–2014 mean SMB"),
    plt.Line2D([0],[0], color=SC_COLOURS["ssp245"], lw=1.5,
               marker="o", ms=3.2, markeredgewidth=0, label="SSP2-4.5 ensemble mean"),
    plt.Line2D([0],[0], color=SC_COLOURS["ssp585"], lw=1.5,
               marker="o", ms=3.2, markeredgewidth=0, label="SSP5-8.5 ensemble mean"),
    mpatches.Patch(color=SC_COLOURS["ssp245"], alpha=0.35, label="SSP2-4.5 ±1σ"),
    mpatches.Patch(color=SC_COLOURS["ssp585"], alpha=0.35, label="SSP5-8.5 ±1σ"),
]
leg8 = fig.legend(handles=handles, loc="outside lower center", ncol=3,
                  fontsize=7.2, framealpha=0.95, edgecolor="#cccccc",
                  handlelength=1.8, borderpad=0.6, columnspacing=1.0)
leg8.get_frame().set_linewidth(0.5)

fig.suptitle(
    "Projected glacier SMB — CMIP6 ensemble (MIROC6, MPI-ESM1-2-LR, IPSL-CM6A-LR)",
    fontsize=8.5)
savefig(fig, "fig08_projection_timeseries.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 09 — Spatial map ΔSMB 2080–2099  (Q1 journal style)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Fig 09 — Projection map 2080–2099 ...")

import geopandas as gpd

lon_f = feat_df.set_index("RGIId").reindex(proj_df["RGIId"])["cenlon"].values
lat_f = feat_df.set_index("RGIId").reindex(proj_df["RGIId"])["cenlat"].values

# ── Shared colour scale: 95th-percentile of BOTH scenarios combined ──────────
all_delta = []
for sc in SCENARIOS:
    col = get_ensmean(sc, "2080-2099")
    if col is not None:
        all_delta.append(proj_df[col].values - proj_df["SMB_hist"].values)
vmax = np.percentile(np.abs(np.concatenate(all_delta)), 95)

# ── Natural Earth country borders ────────────────────────────────────────────
NE_SHP = Path("/tmp/ne_110m/ne_110m_admin_0_countries.shp")
world  = gpd.read_file(NE_SHP) if NE_SHP.exists() else None

# ── Sub-region text labels (lon, lat, label) ─────────────────────────────────
REG_ANCHORS = [
    (74.5, 36.8, "Karakoram"),
    (76.5, 32.2, "W. Himalaya"),
    (84.5, 28.8, "C. Himalaya"),
    (92.0, 28.2, "E. Himalaya"),
    (73.5, 38.8, "Pamir"),
    (79.5, 42.5, "Tian Shan"),
    (70.2, 35.8, "Hindu Kush"),
]

# ── Layout: 2 map panels + thin shared colorbar on the right ─────────────────
fig = plt.figure(figsize=(W2, W2 * 0.52))
gs  = GridSpec(1, 3, figure=fig,
               width_ratios=[1, 1, 0.045],
               wspace=0.06,
               left=0.07, right=0.94, top=0.88, bottom=0.12)

axes_map = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
ax_cb    = fig.add_subplot(gs[0, 2])

for ax_i, (ax, (sc, sc_label)) in enumerate(zip(axes_map, SCENARIOS.items())):
    col = get_ensmean(sc, "2080-2099")
    if col is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        continue
    delta = proj_df[col].values - proj_df["SMB_hist"].values

    # ── Background: land fill + country borders ───────────────────────────
    if world is not None:
        world.plot(ax=ax, color="#f0f0f0", edgecolor="#999999",
                   linewidth=0.35, zorder=0)

    # ── Glacier points ────────────────────────────────────────────────────
    sc_p = ax.scatter(lon_f, lat_f, c=delta, cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax,
                      s=3.5, alpha=0.80, linewidths=0,
                      rasterized=True, zorder=3)

    # ── OOD stippling: grey × markers on top of OOD glacier points ──────
    ood_mask = proj_df["subregion"].isin(OOD_SUBREGIONS)
    if ood_mask.any():
        ax.scatter(lon_f[ood_mask], lat_f[ood_mask],
                   marker="x", s=9, linewidths=0.6,
                   color="#333333", alpha=0.55, zorder=5,
                   label="_nolegend_")

    # ── Sub-region labels ─────────────────────────────────────────────────
    for lx, ly, lbl in REG_ANCHORS:
        ax.text(lx, ly, lbl, fontsize=5.2, color="#222222",
                ha="center", va="center", style="italic", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.65))

    # ── Panel letter ──────────────────────────────────────────────────────
    ax.text(0.02, 0.97, f"({'ab'[ax_i]})",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec="none", alpha=0.75))

    ax.set_xlim(63.5, 107); ax.set_ylim(24, 48)
    ax.set_aspect("equal")
    ax.set_title(f"{sc_label}", fontsize=8.5, pad=4)
    ax.set_xlabel("Longitude (°E)", fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f°E"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f°N"))
    ax.tick_params(labelsize=7.5)
    ax.grid(False)

    if ax_i == 0:
        ax.set_ylabel("Latitude (°N)", fontsize=8)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")

# ── Single shared colorbar ────────────────────────────────────────────────────
norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
sm   = mpl.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, cax=ax_cb, extend="both")
cb.set_label(DSMB_UNIT, fontsize=7.5, labelpad=4)
cb.ax.tick_params(labelsize=7)
cb.outline.set_linewidth(0.5)

# ── OOD legend entry ─────────────────────────────────────────────────────────
ood_handle = plt.Line2D([0], [0], marker="x", color="#333333", linestyle="None",
                         markersize=5, markeredgewidth=0.9, alpha=0.7,
                         label="OOD region (E./C. Himalaya — extrapolation artefact)")
axes_map[1].legend(handles=[ood_handle], loc="lower right", fontsize=6.5,
                    framealpha=0.9, edgecolor="#cccccc").get_frame().set_linewidth(0.5)

fig.suptitle(
    r"Projected $\Delta$SMB (2080–2099 vs 2000–2014) — CMIP6 ensemble mean",
    fontsize=8.5, y=0.97)
savefig(fig, "fig09_projection_map_2080.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 10 — ΔSMB by sub-region × scenario × window
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Fig 10 — Scenario comparison bar ...")

bar_data = []
for reg, grp in proj_df.groupby("subregion"):
    smb_h = grp["SMB_hist"].mean()
    row = {"subregion": REG_LABEL.get(reg, reg), "SMB_hist": smb_h}
    for sc in SCENARIOS:
        for w in ["2040-2059","2080-2099"]:   # two key windows
            cm = get_ensmean(sc, w); cs = get_ensstd(sc, w)
            if cm:
                row[f"d_{sc}_{w}"]     = grp[cm].mean() - smb_h
                row[f"d_{sc}_{w}_std"] = grp[cs].mean() if cs else 0
    bar_data.append(row)

bar_df = pd.DataFrame(bar_data)
order  = bar_df.sort_values("SMB_hist")["subregion"].tolist()
bar_df = bar_df.set_index("subregion").reindex(order)

x     = np.arange(len(order))
width = 0.20
offsets = {"ssp245_2040-2059": -1.5, "ssp245_2080-2099": -0.5,
           "ssp585_2040-2059":  0.5, "ssp585_2080-2099":  1.5}
colours_w = {"ssp245_2040-2059": "#92c5de", "ssp245_2080-2099": "#2166ac",
             "ssp585_2040-2059": "#f4a582", "ssp585_2080-2099": "#d73027"}
labels_w  = {"ssp245_2040-2059": "SSP2-4.5 2040–2059",
             "ssp245_2080-2099": "SSP2-4.5 2080–2099",
             "ssp585_2040-2059": "SSP5-8.5 2040–2059",
             "ssp585_2080-2099": "SSP5-8.5 2080–2099"}

fig, ax = plt.subplots(figsize=(W2, W2 * 0.42))
for key, off in offsets.items():
    sc, w = key[:6], key[7:]
    dcol = f"d_{sc}_{w}"; scol = f"d_{sc}_{w}_std"
    if dcol not in bar_df.columns: continue
    vals = bar_df[dcol].values
    errs = bar_df[scol].values if scol in bar_df.columns else None
    ax.bar(x + off * width, vals, width,
           label=labels_w[key], color=colours_w[key],
           edgecolor="white", linewidth=0.3,
           yerr=errs, capsize=2,
           error_kw={"linewidth": 0.6, "capthick": 0.6})

ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels(order, rotation=30, ha="right", fontsize=7.8)
ax.set_ylabel(DSMB_UNIT)
ax.set_title(r"Projected $\Delta$SMB vs 2000–2014  |  Error bars: ensemble ±1σ",
             fontsize=8.5, pad=4)
leg10 = ax.legend(fontsize=7, ncol=2, framealpha=0.9, edgecolor="#cccccc",
                  loc="lower left")
leg10.get_frame().set_linewidth(0.5)
fig.tight_layout()
savefig(fig, "fig10_scenario_comparison.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 11 — Ensemble uncertainty by region
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Fig 11 — Ensemble uncertainty ...")

fig, axes = plt.subplots(1, 2, figsize=(W2, W2 * 0.38), sharey=True)
fig.subplots_adjust(wspace=0.08)
win_colours = {"2020-2039":"#fee090","2040-2059":"#fdae61",
               "2060-2079":"#f46d43","2080-2099":"#d73027"}

for ax, (sc, sc_label) in zip(axes, SCENARIOS.items()):
    unc_rows = []
    for reg, grp in proj_df.groupby("subregion"):
        for w in WINDOWS:
            cs = get_ensstd(sc, w)
            if cs is None: continue
            unc_rows.append({"region": REG_LABEL.get(reg,reg),
                             "window": w, "spread": grp[cs].mean()})
    if not unc_rows: continue
    unc_df = pd.DataFrame(unc_rows)
    pivot  = (unc_df.pivot(index="region", columns="window", values="spread")
                    .reindex([REG_LABEL.get(r,r) for r in REGS_SORTED]))

    pivot.plot(kind="bar", ax=ax, color=[win_colours[w] for w in WINDOWS],
               edgecolor="white", linewidth=0.3, width=0.72, legend=(ax==axes[0]))
    ax.set_title(f"{sc_label}", fontsize=8.5, pad=3)
    ax.set_xlabel("")
    if ax == axes[0]:
        ax.set_ylabel(r"Ensemble $\sigma$ (m w.e. yr$^{-1}$)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7.5)
    if ax == axes[0]:
        leg11 = ax.legend(title="Window", fontsize=7, title_fontsize=7.5,
                          framealpha=0.9, edgecolor="#cccccc")
        leg11.get_frame().set_linewidth(0.5)

fig.suptitle("Multi-model ensemble spread (model uncertainty) by sub-region",
             fontsize=8.5, y=1.02)
fig.tight_layout()
savefig(fig, "fig11_uncertainty_spread.pdf")

# ── Summary ───────────────────────────────────────────────────────────────────
pdfs = sorted(FIG_DIR.glob("fig*.pdf"))
print(f"\n  Total PDF figures: {len(pdfs)}")
for f in pdfs:
    print(f"    {f.name}  ({f.stat().st_size/1e3:.0f} kB)")
print("=" * 70)
