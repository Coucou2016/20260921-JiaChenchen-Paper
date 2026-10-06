# START HERE — why this repository has no subfolders

**This repository is a deliberately flattened, single-directory mirror of the
`20260921-JiaChenchen-Paper` project (HydroGeo-SRNO, urban flood depth
super-resolution).** There are no subdirectories under the repository root, and
that is intentional.

Every source file, document, figure, table, machine result and verification
script sits side by side in one flat namespace so that an automated reader — a
browsing LLM, another coding agent, a script, or a human — can enumerate and
fetch the *entire* project without walking a tree, guessing paths, or being
silently blocked by directory-level access limits.

If you are an automated reviewer (ChatGPT, another agent, a crawler), the useful
pattern is:

1. read this file,
2. read `FILE_INDEX.md` for the complete inventory,
3. read `README_PROJECT.md` for the science,
4. read `chatgpt__00_TASK_BRIEF.md` and `chatgpt__01_WHERE_TO_LOOK.md` for what
   is worth cross-checking,
5. then fetch the files you need by their flat name.

---

## 1. How the flat names encode provenance

Because there are no folders, the original directory is encoded in each filename
as an ASCII prefix. Nothing about where a file came from is lost — it is in the
name. Sub-paths that were more than one level deep are joined with `__`.

| Flat prefix | Original location | Contents |
|---|---|---|
| *(none)* | repository root | Canonical entry points: `report.html`, `report.md`, `report.pdf`, the three `report_brief.*` condensed files, `README.md`, `STATUS.md`, `VISUALIZATION.md`, `IMPLEMENTATION_CHECKLIST.md`, `requirements.txt`, `PLAN_20260924.md` |
| `scripts__` | `scripts/` | Pipeline stages, figure generators, report builders, audits. A leading `_` marks a verification/audit helper that is excluded from the report body |
| `configs__` | `configs/` | YAML experiment configuration (V0, ablations, multiscale, diffusion stubs) |
| `models__` | `models/` | HydroGeo-SRNO, Galerkin operator, baselines |
| `losses__` | `losses/` | `FloodLoss` and the deep-water weighted variants |
| `metrics__` | `metrics/` | CSI / F1 / RMSE_wet / rank metrics |
| `engine__` | `engine/` | Trainer, evaluator, checkpoint handling |
| `viz__` | `viz/` | Dashboards and plotting helpers |
| `tests__` | `tests/` | Unit checks |
| `dataset__` | `dataset/` | Dataset adapters, normalisation, rain `.npy`, `manifest.json`, `DATASET.md` |
| `figure__` | `figures/` | Published figures (upstream set) |
| `reportfig__` | `outputs/report_figs/*.png` | The figures actually embedded in the report, in render order (`fig00`…`fig85`) |
| `resultdata__premodel__` | `outputs/premodel/` | Pre-model analysis results: JSON, CSV, GeoTIFF, the small `.npz` grids and the four `resultdata__premodel__ckpt_*.pt` |
| `resultdata__report_fields__` | `outputs/report_figs/result_fields/` | Per-tile field tables and the tile metadata behind the domain maps |
| `resultdata__<sub>__` | other `outputs/<sub>/` | Small machine-readable result files (JSON/CSV/YAML) from training runs, sweeps and benchmarks |
| `reportbackup__` | `20261002-报告备份-1/` | An earlier snapshot of the three full-report artifacts, kept for audit |

**Example.** `scripts/premodel_10_consistency.py` becomes
`scripts__premodel_10_consistency.py`. `outputs/premodel/cross_resolution_consistency.json`
becomes `resultdata__premodel__cross_resolution_consistency.json`.

---

## 2. What is deliberately NOT here

The working project is about 41 GB across 1705 files. GitHub rejects any file
over 100 MB and warns above 50 MB, so the multi-gigabyte raw simulation fields
and the full checkpoint set cannot be published this way. This mirror carries
the **auditable core: 715 files, 148.5 MB.**
`DATA_AND_BINARY_NOTICE.md` lists what was omitted, its size, and how to
regenerate it. Nothing that affects a reported number was omitted — every value
quoted in the report can still be traced from the files that are here.

---

## 3. Reproducing the flat mirror

`scripts__build_flat_mirror.py` regenerates this layout from the source tree:

```bash
python scripts/build_flat_mirror.py <DEST_DIR>
```

`_manifest.json` records, for every file, its flat name, its original repo
relative path, its byte size and its SHA-256. Use it to verify that a file you
fetched is the one this snapshot describes.

---

*Generated 2026-10-06. Source root: `20260921-JiaChenchen-Paper`.*
