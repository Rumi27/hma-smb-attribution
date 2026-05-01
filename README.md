# HMA Glacier Mass Balance Attribution

**XGBoost-SHAP attribution of High Mountain Asia glacier surface mass balance drivers**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repository contains all code, derived data, and outputs to reproduce the analyses in:

> Avzalshoev, Z., Chun, P., and Navruzshoev, H. (2026). *Climate Controls on High Mountain Asia Glacier Mass Balance Using Machine Learning: Attribution, Spatial Cross-Validation, and CMIP6 Projections.* Journal of Geophysical Research: Atmospheres (submitted).

---

## Overview

We train an XGBoost model on 94,463 RGI v6.0 glaciers across 15 High Mountain Asia sub-regions, using 33 climate and topographic features derived from ERA5 reanalysis and the Randolph Glacier Inventory. SHAP (SHapley Additive exPlanations) values provide physically interpretable attribution of mass balance variability to individual drivers.

Key findings:
- JJA temperature variability (σ_T_JJA) is the dominant driver of inter-glacier SMB differences across HMA — confirmed by both XGBoost SHAP and Random Forest SHAP
- Strict spatial cross-validation (GroupKFold by sub-region) gives R²=0.168 vs. random-fold R²=0.442 — a 2.6× skill-inflation factor
- Mahalanobis-distance OOD diagnostics flag 52.5% of HMA glaciers as out-of-distribution under delta-method CMIP6 forcing due to the σ_T_JJA=0 artefact
- Under SSP5-8.5 (2080–2099), variance-preserved projections give pan-HMA ΔMB of −0.09 to −0.42 m w.e. yr⁻¹ relative to 2000–2014

---

## Repository Structure

```
hmadata/
├── scripts/                        # Numbered analysis scripts 01–20
│   ├── 01_download_era5_hma.py
│   ├── 02_build_glacier_centroids.py
│   ├── ...
│   ├── 19_ood_threshold_rf_perm.py
│   └── 20_supplementary_stats.py   # Supplementary stats (size-CV, Hugonnet, distances)
├── data/
│   ├── feature_matrix_hma.parquet     # 94,463 glaciers × 33 features + metadata
│   ├── glacier_centroids_hma.parquet  # RGI geometry centroids
│   ├── era5_climate_metrics_hma.parquet  # ERA5-derived climate indices
│   └── README_data.md                 # Per-variable metadata
├── outputs/
│   ├── model_results_hma.json             # CV metrics, feature list
│   ├── shap_importance_hma.csv            # XGBoost full-data SHAP importance
│   ├── rf_shap_importance.csv             # Random Forest SHAP importance (n=10k subsample)
│   ├── rf_permutation_importance.csv      # RF permutation importance (all 94k glaciers)
│   ├── cv_shap_importance.csv             # Cross-validated SHAP importance
│   ├── cv_predictions_hma.parquet         # Out-of-fold predictions
│   ├── shap_values_hma.parquet            # Full-dataset SHAP values (33 features)
│   ├── cv_shap_values.parquet             # Cross-validated SHAP values
│   ├── shap_bootstrap_stability.csv       # Bootstrap SHAP rank stability
│   ├── shap_grouped_hma.csv               # Grouped SHAP by feature cluster
│   ├── smb_projections_hma.parquet        # CMIP6 SMB projections per glacier
│   ├── smb_projection_summary.csv         # Sub-regional projection summary
│   ├── smb_projections_comparison.csv     # σ=0 vs. variance-preserved comparison
│   ├── ood_mahalanobis_per_glacier.parquet  # Mahalanobis D² OOD scores
│   ├── projected_ood_fractions.csv        # OOD fractions by sub-region × scenario
│   ├── ood_threshold_sensitivity.csv      # OOD sensitivity to threshold choice
│   ├── size_stratified_cv.csv             # R², RMSE by glacier size class
│   ├── supplementary_stats.json           # Supplementary statistics (script 20)
│   ├── psnow_threshold_sensitivity.csv    # P_snow definition sensitivity test
│   ├── lapse_sensitivity.csv / .json      # Fixed vs. variable lapse rate sensitivity
│   ├── sensitivity_hyp_sigma_melt.csv     # Hypsometric / σ-weighting sensitivity
│   ├── debris_residual_analysis.csv       # Debris-cover residual analysis
│   ├── subregion_error_decomp.csv         # Sub-regional error decomposition
│   ├── smb_extrapolation_check.csv        # Extrapolation diagnostics
│   ├── bootstrap_pi_hma.parquet           # Bootstrap prediction intervals
│   ├── bootstrap_pi_subregion.csv         # Bootstrap PI by sub-region
│   └── tables/                            # LaTeX and CSV manuscript tables
│       ├── table1_model_comparison.*
│       ├── table2_regional_performance.*
│       ├── table3_shap_importance.*
│       ├── table4_projections.*
│       ├── table5_dataset_summary.*
│       ├── tableS_proj_ood.tex            # Table S1: OOD fractions
│       ├── tableS_bootstrap_pi.tex        # Table S2: Bootstrap PI coverage
│       └── tableS_shap_sensitivity.tex    # Table S3: SHAP method sensitivity
├── figures/                           # All manuscript figures (PDF + PNG)
├── environment.yml                    # Conda environment
├── CITATION.cff                       # Citation metadata
└── LICENSE                            # CC BY 4.0
```

**Note:** `monthly_climate_hma.parquet` (131 MB) is deposited separately at Zenodo [DOI: TBD upon acceptance] and is required only to re-run scripts 03–04.

---

## Installation

This repository uses **Git LFS** to store large `.parquet` and `.pdf` files. Install Git LFS before cloning:

```bash
# Install Git LFS (once per machine)
git lfs install

# Clone with LFS objects
git clone https://github.com/[username]/hma-smb-attribution
cd hma-smb-attribution

# Create conda environment
conda env create -f environment.yml
conda activate hma_attribution
```

Requirements: Python 3.12, XGBoost 2.x, SHAP 0.46+, scikit-learn 1.5+, pandas 2.x, geopandas.

---

## Reproduction

Scripts are numbered in execution order. Most analyses can be reproduced from the provided derived data (starting at step 5):

| Script | Input | Output | Notes |
|--------|-------|--------|-------|
| 01 | ERA5 CDS API | `monthly_climate_hma.parquet` | Requires CDS API key; output at Zenodo |
| 02 | RGI v6.0 shapefiles | `glacier_centroids_hma.parquet` | RGI from NSIDC |
| 03 | monthly_climate, centroids | `era5_climate_metrics_hma.parquet` | Lapse-rate correction |
| 04 | era5_metrics, RGI, Shean SMB | `feature_matrix_hma.parquet` | Main feature matrix |
| 05 | feature_matrix | `model_results_hma.json`, `cv_predictions_hma.parquet` | XGBoost + GroupKFold CV |
| 06 | model_results, feature_matrix | manuscript figures | SHAP beeswarm, bar, heatmap, dependence |
| 07 | CMIP6 PANGEO | CMIP6 climate files | Requires PANGEO access |
| 08 | CMIP6, feature_matrix | `smb_projections_hma.parquet` | SSP2-4.5, SSP5-8.5 |
| 09 | smb_projections | projection figures | Timeseries, maps |
| 10 | all outputs | LaTeX tables | Tables 1–5 |
| 11 | feature_matrix | framework figures | Study area, CV scatter |
| 12 | feature_matrix | `shap_bootstrap_stability.csv` | Bootstrap SHAP rank stability |
| 13 | feature_matrix | `psnow_threshold_sensitivity.csv` | P_snow definition sensitivity |
| 14 | feature_matrix | `ood_mahalanobis_per_glacier.parquet` | OOD Mahalanobis map |
| 15 | feature_matrix, smb_projections | `smb_projections_comparison.csv` | σ_T_JJA variance test |
| 16 | feature_matrix | bootstrap PI, debris, projected OOD | Uncertainty quantification |
| 17 | feature_matrix | `cv_shap_importance.csv`, `cv_shap_values.parquet` | Cross-validated SHAP |
| 18 | feature_matrix | `sensitivity_hyp_sigma_melt.csv` | Hypsometric sensitivity |
| 19 | feature_matrix | `ood_threshold_sensitivity.csv`, `rf_permutation_importance.csv` | OOD threshold & RF permutation |
| 20 | feature_matrix, cv_predictions | `supplementary_stats.json`, `size_stratified_cv.csv` | Supplementary statistics |

Script 19 also produces `rf_permutation_importance.csv`. RF SHAP values (`rf_shap_importance.csv`) are produced by script 19 and reproduced using the same RF hyperparameters as the main text.

### Quick start (reproduce main results)

```bash
conda activate hma_attribution

# Reproduce CV performance, SHAP, and figures
python scripts/05_train_models_hma.py
python scripts/06_generate_figures_hma.py

# Reproduce all sensitivity analyses
python scripts/12_shap_bootstrap.py
python scripts/17_cv_shap.py
python scripts/19_ood_threshold_rf_perm.py
python scripts/20_supplementary_stats.py
```

---

## Data Sources

| Dataset | Reference | Access |
|---------|-----------|--------|
| Shean et al. (2020) geodetic SMB | Shean et al. (2020) *Front. Earth Sci.* | NSIDC |
| RGI v6.0 | RGI Consortium (2017) | https://www.glims.org/RGI/ |
| ERA5 reanalysis | Hersbach et al. (2020) *QJRMS* | https://doi.org/10.24381/cds.adbb2d47 |
| CMIP6 projections | Eyring et al. (2016) | https://doi.org/10.5194/gmd-9-1937-2016 via PANGEO |
| Hugonnet et al. (2021) glacier SMB | Hugonnet et al. (2021) *Nature* | https://doi.org/10.6096/13 |
| OGGM standard projections | Schuster et al. (2023) | https://doi.org/10.5281/zenodo.8286065 |

---

## License

All code and derived data in this repository are released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

Please cite the associated paper (see `CITATION.cff`) when using this data or code.

---

## Contact

Corresponding author: Zafar Avzalshoev — zavzalshoev@gmail.com
