# Data Documentation

All files are in Apache Parquet format (columnar, compressed) unless noted.
Units and variable definitions follow The Cryosphere data policy requirements.

---

## feature_matrix_hma.parquet

**Description:** Main analysis dataset. One row per glacier, 94,463 glaciers total.
Combines RGI v6.0 geometry, Hugonnet et al. (2021) surface mass balance observations,
and ERA5-derived climate indices.

**Dimensions:** 94,463 rows × ~45 columns

| Column | Units | Description |
|--------|-------|-------------|
| RGIId | — | RGI v6.0 glacier identifier (e.g. RGI60-13.00001) |
| subregion | — | HMA sub-region label (15 sub-regions; see Table 2) |
| lon | °E | Glacier centroid longitude |
| lat | °N | Glacier centroid latitude |
| SMB_mwea | m w.e. yr⁻¹ | Surface mass balance 2000–2019 (Hugonnet et al. 2021) |
| SMB_sigma | m w.e. yr⁻¹ | 1σ uncertainty on SMB_mwea (Hugonnet et al. 2021) |
| area_km2 | km² | Glacier area (RGI v6.0) |
| zmean_m | m a.s.l. | Mean glacier elevation (RGI v6.0) |
| zmin_m | m a.s.l. | Minimum glacier elevation (RGI v6.0) |
| zmax_m | m a.s.l. | Maximum glacier elevation (RGI v6.0) |
| slope_deg | degrees | Mean glacier surface slope (RGI v6.0) |
| aspect_sin | — | sin(aspect); encodes north/south orientation |
| aspect_cos | — | cos(aspect); encodes east/west orientation |
| T_annual_C | °C | ERA5 mean annual 2 m air temperature (2000–2019), lapse-rate corrected to glacier elevation |
| T_JJA_C | °C | ERA5 June–July–August mean 2 m temperature, lapse-rate corrected |
| T_DJF_C | °C | ERA5 December–January–February mean 2 m temperature, lapse-rate corrected |
| T_JJA_lapse_C | °C | T_JJA corrected to glacier mean elevation using Γ=6.5 °C km⁻¹ |
| T_JJA_std_C | °C | Standard deviation of inter-annual JJA temperature (σT_JJA); key variability driver |
| precip_annual_m | m yr⁻¹ | ERA5 total annual precipitation |
| P_JJA_m | m | ERA5 June–August total precipitation |
| P_DJF_m | m | ERA5 December–February total precipitation |
| P_snow_m | m | Estimated solid precipitation (T < 2°C threshold) |
| precip_winter_frac | — | Fraction of annual precipitation falling in DJF–MAM (0–1) |
| swe_max_m | m | Maximum snow water equivalent (ERA5) |
| melt_index | °C km | Melt index = T_JJA_lapse_C × √(area_km2); combines thermal forcing and ablation tongue extent |
| debris_frac | — | Supraglacial debris cover fraction (0–1); from optical remote sensing |

**Note:** Columns labelled as ERA5 climate indices are derived quantities (means, standard deviations, or seasonal aggregates over 2000–2019) calculated in script 03. They are not raw ERA5 grid-cell values — lapse-rate correction to glacier mean elevation has been applied using Γ=6.5 °C km⁻¹.

**Source:** Hugonnet et al. (2021) https://doi.org/10.1038/s41586-021-03436-z; RGI Consortium (2017); ERA5 (Hersbach et al. 2020).

---

## glacier_centroids_hma.parquet

**Description:** Glacier centroid coordinates and basic geometry attributes for all
RGI v6.0 glaciers in the HMA domain (lon 60°–105°E, lat 25°–50°N).

**Dimensions:** ~97,000 rows (includes glaciers outside the 94,463 with valid SMB data)

| Column | Units | Description |
|--------|-------|-------------|
| RGIId | — | RGI v6.0 glacier identifier |
| CenLon | °E | Centroid longitude |
| CenLat | °N | Centroid latitude |
| Area | km² | Glacier area |
| Zmed | m a.s.l. | Median glacier elevation |
| Slope | degrees | Mean slope |
| Aspect | degrees | Mean aspect (0=N, 90=E, 180=S, 270=W) |
| subregion | — | HMA sub-region assignment |

**Source:** RGI Consortium (2017). Randolph Glacier Inventory v6.0. https://doi.org/10.7265/N5-RGI-60.

---

## era5_climate_metrics_hma.parquet

**Description:** ERA5-derived climate indices for each glacier centroid grid cell,
aggregated over 2000–2019. These are the intermediate climate variables before
feature engineering (script 04).

**Dimensions:** ~94,500 rows

| Column | Units | Description |
|--------|-------|-------------|
| RGIId | — | RGI v6.0 glacier identifier |
| ERA5_lon | °E | Nearest ERA5 grid cell longitude (0.25° resolution) |
| ERA5_lat | °N | Nearest ERA5 grid cell latitude |
| z_ERA5_orog_m | m | ERA5 orographic height at grid cell |
| T_annual_era5 | °C | Raw ERA5 2 m annual mean temperature (not lapse-corrected) |
| T_JJA_era5 | °C | Raw ERA5 JJA mean temperature |
| T_JJA_std_era5 | °C | ERA5 inter-annual standard deviation of JJA temperature |
| precip_annual_era5 | m yr⁻¹ | ERA5 total annual precipitation |
| P_JJA_era5 | m | ERA5 JJA total precipitation |
| P_snow_era5 | m | ERA5 snowfall |
| swe_max_era5 | m | ERA5 maximum SWE |

**Source:** ERA5 single-level reanalysis (Hersbach et al. 2020). https://doi.org/10.24381/cds.adbb2d47.
Downloaded via CDS API (script 01). Raw monthly files (131 MB) available at Zenodo [DOI: TBD].

---

## monthly_climate_hma.parquet [NOT included — Zenodo only]

**Description:** Monthly ERA5 climate fields extracted at each glacier centroid,
2000–2019. This file (131 MB) exceeds GitHub's recommended file size limit and
is deposited at Zenodo [DOI: TBD].

Required only to re-run scripts 01–04. All downstream analyses (scripts 05–19)
use `feature_matrix_hma.parquet` which is derived from this file.

---

## Coordinate Reference System

All coordinates are geographic (WGS84, EPSG:4326).

## Temporal Coverage

Climate indices: 2000–2019 (20-year mean/std).
SMB observations: 2000–2019 (Hugonnet et al. 2021 period 1).

## Missing Values

`NaN` indicates missing data. Glaciers with missing SMB observations (e.g., very small glaciers below the Hugonnet detection threshold) are excluded from the feature_matrix_hma.parquet dataset.

## Data Sharing Policy

All derived data files in this repository are released under CC BY 4.0. The underlying ERA5 and RGI source data are subject to their respective licenses (Copernicus C3S/ECMWF and RGI Consortium). Users are responsible for compliance with those upstream licenses.
