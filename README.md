# HydroGeo-SRNO

> **This repository has no subfolders, on purpose.** It is a deliberately
> flattened, single-directory mirror so an automated reader (ChatGPT, another
> agent, a crawler) can enumerate the whole project at once. Start at
> **`START_HERE_FLAT_LAYOUT.md`**, then `FILE_INDEX.md` for the full inventory,
> then `README_PROJECT.md` for the science. Current directory references in the
> notes below map to flat names by an ASCII prefix, for example `scripts/train_fixed.py`
> is `scripts__train_fixed.py`.

Geography-guided arbitrary-scale neural operator for urban flood super-resolution.

Primary experiment (V0): **10 m → 2 m**, target `h_max`, deterministic residual SR conditioned on the 2 m static stack.

Plan source of truth: `../20260924-方案.md` (this repo root). Derived grids live in `dataset/` (do not rebuild unless required).

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
