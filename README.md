# HydroGeo-SRNO

Geography-guided arbitrary-scale neural operator for urban flood super-resolution.

Primary experiment (V0): **10 m → 2 m**, target `h_max`, deterministic residual SR conditioned on the 2 m static stack.

Plan source of truth: `../20260924-方案.md` (this repo root). Derived grids live in `dataset/` (do not rebuild unless required).

## Revision status (2026-10-07)

An external system review returned **Major Revision**: the conclusions, the code and
the statistical evidence were not fully consistent. The code-level and report-level
defects are fixed and, where no retraining was needed, recomputed. Model-level claims
that require retraining are deferred and are **not** asserted anywhere in the report.

Key changes:

- statistics move from **training epochs** to **independent seeds**; a spatial block
  bootstrap was run on all 242 test tiles and shows spatial autocorrelation inflates
  the CI width by $1.25\times$–$1.63\times$;
- the **volume relative error and CSI@0.30 m cross zero** under block bootstrap, so
  the volume-improvement claim is withdrawn;
- the ablation interfaces that made A4/A6 invalid are fixed (trainable direct head,
  wet-head supervision), but the corrected ablation must be **rerun** before use;
- the multiscale trainer now honours its YAML and strictly isolates `5→2`, `20→2`,
  `30→2` as unseen scales;
- generalisation is bounded to **within-Wellington spatial extrapolation** — cross-city
  and cross-event transfer are labelled unverified.

Full plan and pre-submission checklist: `RERUN_PLAN_AND_CHECKLIST.md`.
Ledger: `STATUS.md`, `IMPLEMENTATION_CHECKLIST.md`.

## Layout

```
configs/          YAML for V0 / ablation / multiscale / LDM
dataset/          WellingtonSRDataset + fixed/multiscale adapters
models/           HydroGeo-SRNO, Galerkin, baselines, diffusion stubs
losses/           FloodLoss (masked, no LR conservation)
metrics/          CSI/F1/RMSE_wet/... (PSNR secondary)
engine/           trainer, evaluator, checkpoints
scripts/          train_fixed, evaluate, infer_*, multiscale, LDM stub
tests/            unit checks
```

## Setup

```bash
pip install torch pyyaml netCDF4 numpy
```

Run all commands from the repo root (`20260921-JiaChenchen-Paper`).

## Main experiment

```bash
# 1) Bilinear benchmark
python scripts/evaluate.py --config configs/v0_10m2m_hmax.yaml --model bilinear --split val

# 2) Smoke train (CPU-safe; real patches, flood loss, checkpoint)
python scripts/train_fixed.py --config configs/v0_10m2m_hmax_smoke.yaml

# 3) Full V0 train (GPU recommended, 300 epochs)
python scripts/train_fixed.py --config configs/v0_10m2m_hmax.yaml

# 4) Ablation A0..A7
python scripts/train_fixed.py --config configs/v0_10m2m_hmax.yaml --ablation A7
```

Checkpoints: `outputs/.../best_csi.pt` (by val CSI@0.05) and `best_rmse_wet.pt`.

## Constraints

- Geographic split only (`dataset/index/patches_480m.json`)
- Nodata masked; LR uses explicit `lr_valid`
- Landuse never z-scored
- No mass-conservation loss vs coarse simulation
- No flip/rotation in V0
