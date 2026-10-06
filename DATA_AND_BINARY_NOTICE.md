# DATA AND BINARY NOTICE — what was deliberately left out

This mirror is a **flat, GitHub-sized snapshot** of a much larger working
project. The source tree is about **41 GB across 1700+ files**. GitHub rejects any
single file over 100 MB, and this snapshot therefore carries the auditable core
only: **about 760 files, roughly 149 MB.**

**Nothing that determines a number in the report was omitted.** Every figure and
every table can be traced to a file that is present. What is absent is raw input
volume and model weights, which are re-derivable.

---

## 1. What is NOT in this repository

| Omitted group | Size | Why | Manifest prefix if you have the source |
|---|---|---|---|
| `dataset/grids/**/*.nc` | ~6.0 GB | Per-file 0.6–1.9 GB; the multi-resolution hydrodynamic fields | `dataset__` (code and rain data only are kept) |
| `20260924-备份数据-不动不改/static_geo_data/` | ~27.0 GB | Raw 2 m static terrain rasters as `.txt`, up to 610 MB each, three scenarios | — (frozen backup, not used by the pipeline) |
| `20260924-备份数据-不动不改/wellington-output-data/` | ~4.9 GB | Raw HiPIMS flood output fields (`flood_h`, `flood_hux`, `flood_hvy`) | — (frozen backup) |
| `outputs/**/*.pt` | ~2.5 GB | 158 checkpoints. The four small pre-model ones are kept | `resultdata__premodel__ckpt_*.pt` are kept |
| `outputs/**/*.npz` large | ~0.6 GB | Per-tile training/validation grids over 5 MB | `resultdata__premodel__tile_*.npz` under 5 MB are kept |
| `outputs/finetune_deep/`, `v0_10m2m_hmax/`, `deep_sweep/` epoch snapshots | ~2.8 GB | Intermediate training artefacts | small JSON/CSV kept as `resultdata__*` |

## 2. Files under 100 MB but over 50 MB

GitHub warns at 50 MB. Seven files in this snapshot exceed that and are
intentional, because they are the deliverables themselves:

| File | Size | Why kept |
|---|---|---|
| `report.html` | 18.6 MB | Full report, figures inlined as base64 |
| `report.pdf` | 15.2 MB | Full report, print form |
| `reportbackup__report.html` | 12.6 MB | Earlier snapshot, for comparison |
| `reportbackup__report.pdf` | 10.8 MB | Earlier snapshot |
| `report_brief.html` | 8.1 MB | Condensed report, self-contained |
| `report_brief.pdf` | 6.8 MB | Condensed report, print form |
| `resultdata__premodel__tile_metrics.json` | 4.6 MB | Per-tile metric table behind several figures |
If you clone with `git clone --depth 1` you still get all of them; Git LFS is not
used, so no extra step is needed to fetch the reports.

## 3. What each kept artefact lets you re-derive

- **Every number in the report** — the source of truth is under
  `resultdata__premodel__*.json`, plus `resultdata__premodel__premodel_results.json`,
  `cross_resolution_consistency.json`, `pattern_extra.json`, `cell_contrasts.json`,
  `factor_contrasts.json`, `flood_error.json`, `variance_decomposition.json`,
  `tile_metrics.json`, `scale_dialect.json`, `velocity_correlation.json`.
- **Every figure** — the generator is a `scripts__premodel_05_figs.py` or
  `scripts__make_result_figs.py` function; the embedded plate is
  `reportfig__fig*.png`.
- **The four pre-model checkpoints** — `resultdata__premodel__ckpt_2m.pt`,
  `ckpt_5m.pt`, `ckpt_10m.pt`, `ckpt_20m.pt`, `ckpt_30m.pt` (about 90 KB each).
- **The SABRE cross-check** — the exported GeoTIFFs are
  `resultdata__premodel__sabre_input__*.tif`, the R script is
  `scripts__sabre_crosscheck.R`, the R output is
  `resultdata__premodel__sabre_r_vmeasure.csv`, and the run log is
  `resultdata__premodel__sabre_r_run.log`.

## 4. How to regenerate the omitted raw data

The raw fields come from the HiPIMS two-dimensional hydrodynamic solver run on
the Wellington urban flood dataset at five grid sizes (2, 5, 10, 20, 30 m). They
are produced upstream of this project, not by code in this repository. The
derived multi-resolution grids under `dataset/grids/` are rebuilt by the
`dataset__*.py` adapters from those raw fields; the report states, where
relevant, which resolution a quantity belongs to.

Model checkpoints are produced by
`scripts__train_fixed.py` (V0), the `scripts__run_deep_*.py` arms, and the sweeps; the exact
configurations are the `configs__*.yaml` files, and the resulting metrics are the
`resultdata__*` JSON/CSV files, so a re-run can be checked against the values
already shipped here.

---

*Generated 2026-10-06. Matching manifest: `_manifest.json`.*
