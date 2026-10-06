# chatgpt__01_WHERE_TO_LOOK — the specific files behind each headline claim

Each row names the claim, the report figure it appears in, the data file it is
computed from, and the code that computes it. Flat names only.

## Error accounting and the two calculation dialects

| Claim | Data | Code |
|---|---|---|
| Coarse-grid error splits into a simulation term and an aggregation term | `resultdata__premodel__flood_error.json`, `resultdata__premodel__variance_decomposition.json` | `scripts__premodel_04_decomp.py` |
| Fine-grid MSE equals coarse-grid MSE plus within-block variance (exact identity) | `resultdata__premodel__scale_dialect.json` | `scripts__premodel_09_dialect.py` |

## Cross-resolution laws

| Claim | Data | Code |
|---|---|---|
| Terrain correlates across resolutions; flood depth and velocity do not | `resultdata__premodel__cross_resolution_consistency.json`, `velocity_correlation.json` | `scripts__premodel_10_consistency.py` |
| Terrain distortion grows with grid size | `resultdata__premodel__terrain_distortion.json` | same |
| Only neighbouring resolutions transfer | `resultdata__premodel__cross_resolution_consistency.json` (`vmeasure`) | same |

## Spatial pattern similarity (five independent families)

| Family | Data | Figure |
|---|---|---|
| V-measure | `resultdata__premodel__cross_resolution_consistency.json` | `reportfig__fig74_vmeasure.png` |
| Chance-corrected (ARI, FM, NMI, AMI, kappa) | `resultdata__premodel__pattern_extra.json` | `reportfig__fig76_pattern_agreement.png` |
| Areal overlap (IoU, total variation) | same | `reportfig__fig77_pattern_areal.png` |
| Wet-front boundary / Hausdorff | same | `reportfig__fig78_pattern_boundary.png` |
| Spatial autocorrelation (Moran, Geary, join counts) | same | `reportfig__fig79_pattern_autocorr.png` |
| Distance- and value-resolved | same | `reportfig__fig80_pattern_distance.png` |
| Variogram / scale space | same | `reportfig__fig82_variogram_scale.png` |
| Contiguity-constrained clustering | same | `reportfig__fig83_contiguity_map.png`, `fig84_contiguity_metrics.png` |
| Cluster-count sensitivity K=2..8 | same | `reportfig__fig85_k_sensitivity.png` |
| SABRE (R) cross-check | `resultdata__premodel__sabre_r_vmeasure.csv`, `sabre_python_vmeasure.json` | `scripts__sabre_crosscheck.R` |

## The deep-water deficit and its correction

| Claim | Data | Figure |
|---|---|---|
| Point-by-point deep water is underestimated | `resultdata__premodel__premodel_results.json` | `reportfig__fig24_density.png` |
| Loss-weight sweep trades deep-water gain against overall cost | `resultdata__*` under the sweep runs | `reportfig__fig09_sweep.png` |
| Fine-tune reduces deep-water error without degrading overall skill | `resultdata__premodel__premodel_results.json` | `reportfig__fig06_before_after.png` |
| The improvement is significant under paired bootstrap | same | `reportfig__fig07_forest.png`, `fig27_bootstrap.png` |
| Improvement is spatially concentrated | `resultdata__report_fields__*` | `reportfig__fig23_domain_maps.png` |

## Cheapest high-value checks

1. Pick five numbers from `report.md`, find them in
   `resultdata__premodel__*.json`, confirm they match.
2. Confirm every value in `report_brief.md` also appears in `report.md`.
3. Re-derive one V-measure cell from `resultdata__premodel__sabre_input__*.tif`
   and compare with `resultdata__premodel__sabre_r_vmeasure.csv`.
4. Read the report's own limitation section and check no figure contradicts it.

*Full inventory: `FILE_INDEX.md`.*
