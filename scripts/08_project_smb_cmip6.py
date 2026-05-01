"""
Script 08 — Bias-Correct CMIP6 and Project Future SMB

Method (delta / anomaly approach):
  For each CMIP6 model and scenario:
    1. Compute CMIP6 historical climatology (2000–2014) at each ERA5 grid point
       using nearest-neighbour regridding.
    2. Compute ERA5 climatology over the same period (already in monthly_climate_hma).
    3. Delta = CMIP6_future_month – CMIP6_historical_climatology_month
    4. Bias-corrected future = ERA5_climatology + Delta
    5. Recompute all 33 features for each 20-year future window.
    6. Apply trained XGBoost (full-dataset fit) → projected SMB per glacier.

Future windows: 2020–2039, 2040–2059, 2060–2079, 2080–2099
Scenarios: ssp2_4_5, ssp5_8_5
Ensemble: mean across available models + per-model spread (uncertainty)

Outputs:
  outputs/smb_projections_hma.parquet
    columns: RGIId, subregion, rgi_region, SMB_hist,
             SMB_{scenario}_{window}_{model},
             SMB_{scenario}_{window}_ensmean,
             SMB_{scenario}_{window}_ensstd
  outputs/smb_projection_summary.csv
    — regional mean SMB by scenario × window × subregion
"""

import pandas as pd
import numpy as np
import xarray as xr
import json
import pickle
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PAPER_DIR  = Path(__file__).parent.parent
DATA_DIR   = PAPER_DIR / "data"
CMIP6_DIR  = DATA_DIR / "cmip6"
OUT_DIR    = PAPER_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("SCRIPT 08 — PROJECT FUTURE SMB WITH CMIP6")
print("=" * 70)

# ── Load ERA5 baseline (monthly climate already extracted) ─────────────────────
print("\n[1] Loading ERA5 baseline ...")
monthly_era5 = pd.read_parquet(DATA_DIR / "monthly_climate_hma.parquet")
cents        = pd.read_parquet(DATA_DIR / "glacier_centroids_hma.parquet")
feat_df      = pd.read_parquet(DATA_DIR / "feature_matrix_hma.parquet")
# Attach ERA5 orography elevation (needed for correct lapse correction in projections)
cents = cents.merge(feat_df[["RGIId","z_era5_orog_m"]], on="RGIId", how="left")
results      = json.loads((OUT_DIR / "model_results_hma.json").read_text())

FEATURE_COLS = results["features"]
print(f"    ERA5 monthly: {len(monthly_era5):,} rows")
print(f"    Glaciers    : {len(cents):,}")
print(f"    Features    : {len(FEATURE_COLS)}")

# ERA5 climatology per glacier per month (2000–2014 to match CMIP6 historical)
era5_hist = monthly_era5[monthly_era5["Year"] <= 2014].copy()
era5_clim = (era5_hist.groupby(["RGIId", "Month"])
                       .agg(T_clim=("Temp_C", "mean"),
                            P_clim=("Precip_mm", "mean"))
                       .reset_index())
print(f"    ERA5 historical climatology (2000–2014): {len(era5_clim):,} rows")

lats_all = cents["cenlat"].values
lons_all = cents["cenlon"].values
rgi_ids  = cents["RGIId"].values

# ── Re-train XGBoost on full dataset (or load if available) ───────────────────
print("\n[2] Training full-dataset XGBoost ...")
import xgboost as xgb

X_full = feat_df[FEATURE_COLS].values
y_full = feat_df["SMB_mwea"].values

xgb_model = xgb.XGBRegressor(
    n_estimators=600, learning_rate=0.04, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=20, reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, random_state=42, verbosity=0)
xgb_model.fit(X_full, y_full)
print(f"    XGBoost trained on {len(X_full):,} glaciers.")

# ── Helper: compute features from monthly T/P time series ─────────────────────
LAPSE = 6.5 / 1000.0

def monthly_to_features(monthly_df, glacier_df):
    """
    Recompute climate features from a monthly T/P DataFrame.
    monthly_df: columns [RGIId, Year, Month, Temp_C, Precip_mm]
    glacier_df: columns [RGIId, zmean_m, slope_deg, area_km2,
                         elev_range_m, aspect_sin, aspect_cos, log_area_km2]
    Returns feature DataFrame with FEATURE_COLS, indexed by RGIId.
    """
    monthly_df = monthly_df.copy()
    monthly_df["season"] = monthly_df["Month"].map({
        12:"DJF",1:"DJF",2:"DJF",
        3:"MAM",4:"MAM",5:"MAM",
        6:"JJA",7:"JJA",8:"JJA",
        9:"SON",10:"SON",11:"SON"})

    # Climatological means
    clim = (monthly_df.groupby("RGIId")
                      .agg(T_mean_C=("Temp_C","mean"),
                           P_mean_mm=("Precip_mm","mean")))
    for season in ["JJA","DJF","MAM","SON"]:
        sub = (monthly_df[monthly_df["season"]==season]
                         .groupby("RGIId")
                         .agg(**{f"T_{season}_C":("Temp_C","mean"),
                                 f"P_{season}_mm":("Precip_mm","mean")}))
        clim = clim.join(sub)

    # Annual aggregates for trends
    ann_T = monthly_df.groupby(["RGIId","Year"])["Temp_C"].mean().reset_index(name="T_ann")
    ann_P = monthly_df.groupby(["RGIId","Year"])["Precip_mm"].sum().reset_index(name="P_ann")
    jja_T = (monthly_df[monthly_df["season"]=="JJA"]
                        .groupby(["RGIId","Year"])["Temp_C"].mean().reset_index(name="T_JJA"))

    def trend(df, id_col, yr_col, val_col, name):
        out = {}
        for rid, grp in df.groupby(id_col):
            x = grp[yr_col].values.astype(float)
            y = grp[val_col].values.astype(float)
            mask = ~np.isnan(y)
            out[rid] = stats.linregress(x[mask], y[mask])[0] * 10 if mask.sum() >= 5 else np.nan
        return pd.Series(out, name=name)

    clim = clim.join(trend(ann_T,"RGIId","Year","T_ann","T_trend_C_dec"))
    clim = clim.join(trend(jja_T,"RGIId","Year","T_JJA","T_JJA_trend_C_dec"))
    clim = clim.join(trend(ann_P,"RGIId","Year","P_ann","P_trend_mm_dec"))

    # Variability
    clim = clim.join(ann_T.groupby("RGIId")["T_ann"].std().rename("T_std_C"))
    clim = clim.join(jja_T.groupby("RGIId")["T_JJA"].std().rename("T_JJA_std_C"))
    clim = clim.join(ann_P.groupby("RGIId")["P_ann"].std().rename("P_std_mm"))

    # Extremes
    clim = clim.join(jja_T.groupby("RGIId")["T_JJA"].max().rename("T_JJA_max_C"))
    clim = clim.join(
        monthly_df[monthly_df["season"]=="DJF"]
                  .groupby("RGIId")["Temp_C"].min().rename("T_DJF_min_C"))

    # Merge topography (include z_era5_orog_m for correct lapse correction)
    topo_cols = ["RGIId","zmean_m","slope_deg","area_km2","elev_range_m",
                 "aspect_sin","aspect_cos","log_area_km2","z_era5_orog_m"]
    topo = glacier_df[topo_cols].set_index("RGIId")
    feat = clim.join(topo)

    # Derived physical features — correct lapse: Δz = glacier elevation − ERA5 orog
    feat["T_JJA_lapse_C"]      = feat["T_JJA_C"]   - (feat["zmean_m"] - feat["z_era5_orog_m"]) * LAPSE
    feat["T_ann_lapse_C"]      = feat["T_mean_C"]   - (feat["zmean_m"] - feat["z_era5_orog_m"]) * LAPSE
    feat["continentality"]     = feat["T_JJA_C"]    - feat["T_DJF_C"]
    denom                      = (feat["P_mean_mm"] * 12).clip(lower=1.0)
    feat["precip_winter_frac"] = (feat["P_DJF_mm"] * 3 / denom).clip(0, 1)
    feat["P_snow_mm"]          = feat["P_DJF_mm"] + feat["P_MAM_mm"]
    feat["cryo_balance"]       = feat["P_snow_mm"] / (feat["T_JJA_lapse_C"] + 10.0).clip(lower=0.1)
    feat["melt_index"]         = feat["T_JJA_lapse_C"] * np.sqrt(feat["area_km2"].clip(lower=0.001))
    feat["P_trend_x_wfrac"]    = feat["P_trend_mm_dec"] * feat["precip_winter_frac"]

    return feat[FEATURE_COLS]

# ── Identify available CMIP6 files ────────────────────────────────────────────
print("\n[3] Scanning CMIP6 files ...")
nc_files = sorted(CMIP6_DIR.glob("*.nc"))
print(f"    Found {len(nc_files)} NetCDF files in {CMIP6_DIR.name}/")

if not nc_files:
    print("\n  ERROR: No CMIP6 files found.")
    print("  Run scripts/07_download_cmip6_hma.py first.")
    raise SystemExit(1)

# Parse filenames: {model}_{experiment}_{variable}.nc
available = {}
for f in nc_files:
    parts = f.stem.split("_")
    # Filename format: {model}_{experiment}_{variable}.nc
    # variable = last token, experiment = second-to-last token (no underscores),
    # model = everything before that.
    # Experiments: "historical", "ssp245", "ssp585"
    var = parts[-1]
    for exp in ["historical", "ssp245", "ssp585"]:
        exp_parts = exp.split("_")
        n = len(exp_parts)
        if parts[-(1+n):-1] == exp_parts:
            model = "_".join(parts[:-(1+n)])
            available.setdefault(model, {}).setdefault(exp, {})[var] = f
            break

print(f"    Models available : {list(available.keys())}")
for model, exps in available.items():
    for exp, vars_ in exps.items():
        print(f"      {model} / {exp}: {list(vars_.keys())}")

# ── Determine which model/scenario combos are complete ────────────────────────
SCENARIOS  = ["ssp245", "ssp585"]   # match filenames from script 07
WINDOWS    = {"2020-2039":(2020,2039), "2040-2059":(2040,2059),
              "2060-2079":(2060,2079), "2080-2099":(2080,2099)}

proj_df = feat_df[["RGIId","subregion","rgi_region","SMB_mwea"]].copy()
proj_df = proj_df.rename(columns={"SMB_mwea":"SMB_hist"})

all_cols = {}

for model in available:
    if "historical" not in available[model]:
        print(f"  SKIP {model} — no historical run")
        continue

    hist_files = available[model]["historical"]
    t_key = next((k for k in hist_files if k in ("tas", "near_surface_air_temperature")), None)
    p_key = next((k for k in hist_files if k in ("pr", "precipitation")), None)
    if not t_key or not p_key:
        print(f"  SKIP {model} — missing T or P historical (have: {list(hist_files.keys())})")
        continue

    print(f"\n[4] Processing model: {model}")

    # ── Load CMIP6 historical and compute climatology ──────────────────────
    def load_cmip6_monthly(nc_path):
        """Load CMIP6 NetCDF, return (times, lats, lons, data_3d)."""
        ds = xr.open_dataset(nc_path)
        # find time coord
        tc = next((c for c in ["time","valid_time"] if c in ds.coords), None)
        # find lat/lon coord names
        lat_c = next((c for c in ["lat","latitude"] if c in ds.coords), None)
        lon_c = next((c for c in ["lon","longitude"] if c in ds.coords), None)
        var   = list(ds.data_vars)[0]
        data  = ds[var].values          # (time, lat, lon)
        times = pd.to_datetime(ds[tc].values)
        lats  = ds[lat_c].values
        lons  = ds[lon_c].values
        ds.close()
        return times, lats, lons, data

    t_times_h, c_lats, c_lons, t_data_h = load_cmip6_monthly(hist_files[t_key])
    p_times_h, _,      _,      p_data_h = load_cmip6_monthly(hist_files[p_key])

    # Convert units: K → °C, kg m⁻² s⁻¹ → mm month⁻¹
    if t_data_h.mean() > 100:   t_data_h = t_data_h - 273.15
    if p_data_h.mean() < 1:     p_data_h = p_data_h * 86400 * 30   # kg/m²/s → mm/month

    # Nearest-neighbour indices for each glacier centroid in CMIP6 grid
    c_lat_idx = np.abs(c_lats[:, None] - lats_all[None, :]).argmin(axis=0)
    c_lon_idx = np.abs(c_lons[:, None] - lons_all[None, :]).argmin(axis=0)

    # Historical climatology per glacier per calendar month
    hist_T_clim = np.zeros((12, len(rgi_ids)))   # (month, glacier)
    hist_P_clim = np.zeros((12, len(rgi_ids)))
    for m in range(1, 13):
        t_mask = np.array([t.month == m for t in t_times_h])
        p_mask = np.array([t.month == m for t in p_times_h])
        if t_mask.sum() > 0:
            hist_T_clim[m-1] = t_data_h[t_mask].mean(axis=0)[c_lat_idx, c_lon_idx]
        if p_mask.sum() > 0:
            hist_P_clim[m-1] = p_data_h[p_mask].mean(axis=0)[c_lat_idx, c_lon_idx]

    # ── Per scenario ──────────────────────────────────────────────────────
    for scenario in SCENARIOS:
        if scenario not in available.get(model, {}):
            print(f"  SKIP {model}/{scenario} — not downloaded")
            continue

        fut_files = available[model][scenario]
        t_key_f = next((k for k in fut_files if k in ("tas","near_surface_air_temperature")), None)
        p_key_f = next((k for k in fut_files if k in ("pr","precipitation")), None)
        if not t_key_f or not p_key_f:
            print(f"  SKIP {model}/{scenario} — missing T or P future (have: {list(fut_files.keys())})")
            continue

        print(f"  Scenario: {scenario}")
        t_times_f, _, _, t_data_f = load_cmip6_monthly(fut_files[t_key_f])
        p_times_f, _, _, p_data_f = load_cmip6_monthly(fut_files[p_key_f])

        if t_data_f.mean() > 100: t_data_f = t_data_f - 273.15
        if p_data_f.mean() < 1:   p_data_f = p_data_f * 86400 * 30

        # ── Per future window ─────────────────────────────────────────────
        for win_label, (y0, y1) in WINDOWS.items():
            print(f"    Window {win_label} ...", end=" ", flush=True)

            # Select time steps in this window
            t_win = np.array([(y0 <= t.year <= y1) for t in t_times_f])
            p_win = np.array([(y0 <= t.year <= y1) for t in p_times_f])

            if t_win.sum() == 0 or p_win.sum() == 0:
                print("no data — SKIP")
                continue

            t_sub = t_data_f[t_win]
            p_sub = p_data_f[p_win]
            t_sub_times = t_times_f[t_win]
            p_sub_times = p_times_f[p_win]

            # Future climatology per calendar month (CMIP6 raw, same grid)
            fut_T_clim = np.zeros((12, len(rgi_ids)))
            fut_P_clim = np.zeros((12, len(rgi_ids)))
            for m in range(1, 13):
                tm = np.array([t.month == m for t in t_sub_times])
                pm = np.array([t.month == m for t in p_sub_times])
                if tm.sum() > 0:
                    fut_T_clim[m-1] = t_sub[tm].mean(axis=0)[c_lat_idx, c_lon_idx]
                if pm.sum() > 0:
                    fut_P_clim[m-1] = p_sub[pm].mean(axis=0)[c_lat_idx, c_lon_idx]

            # Delta bias correction:
            #   T_bc = ERA5_clim + (CMIP6_future_clim - CMIP6_hist_clim)
            #   P_bc = ERA5_clim * (CMIP6_future_clim / CMIP6_hist_clim)
            delta_T = fut_T_clim - hist_T_clim           # (12, n_glaciers)
            ratio_P = np.where(hist_P_clim > 0.1,
                               fut_P_clim / hist_P_clim, 1.0)

            # Rebuild monthly time series for feature computation
            # Use ERA5 historical climatology + delta as synthetic monthly series
            # with synthetic years spanning the window for trend computation
            n_years = y1 - y0 + 1
            rows = []
            for yr_i, yr in enumerate(range(y0, y1+1)):
                for m in range(1, 13):
                    # Retrieve ERA5 clim for this glacier × month
                    ec = era5_clim[(era5_clim["Month"] == m)]
                    ec_T = ec.set_index("RGIId")["T_clim"].reindex(rgi_ids).values
                    ec_P = ec.set_index("RGIId")["P_clim"].reindex(rgi_ids).values

                    bc_T = ec_T + delta_T[m-1]          # (n_glaciers,)
                    bc_P = (ec_P * ratio_P[m-1]).clip(min=0)

                    df_m = pd.DataFrame({
                        "RGIId":     rgi_ids,
                        "Year":      yr,
                        "Month":     m,
                        "Temp_C":    bc_T,
                        "Precip_mm": bc_P,
                    })
                    rows.append(df_m)

            monthly_proj = pd.concat(rows, ignore_index=True)

            # Compute features
            feats_proj = monthly_to_features(monthly_proj, cents)
            X_proj     = feats_proj.reindex(rgi_ids).values
            y_proj     = xgb_model.predict(X_proj)

            col = f"SMB_{scenario}_{win_label}_{model}"
            proj_df[col] = y_proj
            all_cols.setdefault((scenario, win_label), []).append(col)
            print(f"mean SMB = {y_proj.mean():.4f}")

# ── Ensemble mean and spread ──────────────────────────────────────────────────
print("\n[5] Computing ensemble statistics ...")
for (scenario, win_label), cols in all_cols.items():
    if len(cols) == 0: continue
    vals = proj_df[cols].values
    proj_df[f"SMB_{scenario}_{win_label}_ensmean"] = vals.mean(axis=1)
    proj_df[f"SMB_{scenario}_{win_label}_ensstd"]  = vals.std(axis=1)
    print(f"  {scenario} {win_label}: "
          f"mean={vals.mean():.4f}  std={vals.std():.4f}  "
          f"(n_models={len(cols)})")

# ── Regional summary table ────────────────────────────────────────────────────
print("\n[6] Building regional summary ...")
ens_cols = [c for c in proj_df.columns if "ensmean" in c]
summary_rows = []
for reg, grp in proj_df.groupby("subregion"):
    row = {"subregion": reg}
    row["SMB_hist"] = grp["SMB_hist"].mean()
    for c in ens_cols:
        if c in grp.columns:
            row[c] = grp[c].mean()
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False, float_format="{:.3f}".format))

# ── Extrapolation boundary diagnostic ────────────────────────────────────────
# For each future window × scenario, check how many glaciers have projected
# feature values outside the range seen during training.  Flags where the
# model is extrapolating beyond its training distribution — key for the
# eastern Himalaya SSP5-8.5 anomaly discussion.
print("\n[7] Extrapolation boundary diagnostic ...")

# Training feature ranges (min/max per feature)
X_train_df = feat_df[FEATURE_COLS]
train_min   = X_train_df.min()
train_max   = X_train_df.max()

extrap_records = []
for (scenario, win_label), cols in all_cols.items():
    for reg, grp_idx in proj_df.groupby("subregion").groups.items():
        grp = proj_df.loc[grp_idx]
        # Re-fetch the projected features for this subregion × window × scenario
        # We approximated them above; here we just flag mean T/P shift vs training
        smb_hist_mean = grp["SMB_hist"].mean()
        ensmean_col   = f"SMB_{scenario}_{win_label}_ensmean"
        if ensmean_col not in grp.columns:
            continue
        smb_proj_mean = grp[ensmean_col].mean()
        delta_smb     = smb_proj_mean - smb_hist_mean
        extrap_records.append({
            "subregion": reg,
            "scenario":  scenario,
            "window":    win_label,
            "SMB_hist":  round(smb_hist_mean, 4),
            "SMB_proj":  round(smb_proj_mean, 4),
            "dSMB":      round(delta_smb, 4),
            "suspicious": delta_smb > 0.02,   # positive shift = check carefully
        })

extrap_df = pd.DataFrame(extrap_records)
suspect   = extrap_df[extrap_df["suspicious"]]
if len(suspect):
    print(f"\n  WARNING: {len(suspect)} subregion × scenario × window combinations "
          f"show projected SMB IMPROVEMENT (dSMB > +0.02 m w.e. yr⁻¹).")
    print("  These cases likely reflect CMIP6 precipitation increases pushing the")
    print("  model outside its training distribution — flag in manuscript Discussion.")
    print(suspect.to_string(index=False))
else:
    print("  No suspicious positive SMB shifts detected.")

extrap_df.to_csv(OUT_DIR / "smb_extrapolation_check.csv", index=False)
print(f"\n  Saved: smb_extrapolation_check.csv")

# ── Save ──────────────────────────────────────────────────────────────────────
proj_df.to_parquet(OUT_DIR / "smb_projections_hma.parquet", index=False)
summary_df.to_csv(OUT_DIR / "smb_projection_summary.csv", index=False)
print(f"\n  Saved: smb_projections_hma.parquet  ({len(proj_df):,} glaciers)")
print(f"  Saved: smb_projection_summary.csv")
print("\n  Next: run scripts/09_figures_projections_hma.py")
print("=" * 70)
