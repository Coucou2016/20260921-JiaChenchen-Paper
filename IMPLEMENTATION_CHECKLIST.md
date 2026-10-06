# HydroGeo-SRNO implementation checklist

Derived strictly from `20260924-方案.md`. Primary experiment: **10 m → 2 m, `h_max`, deterministic residual SR**.

## Phase map (plan §二十八)

| ID | Plan item | Deliverable | Status |
|----|-----------|-------------|--------|
| P0 | Reader sanity | Fixed-SR adapter, mask/NaN handling, geographic split untouched | done |
| P1 | Bilinear + metrics | `baselines/bilinear.py`, `metrics/flood_metrics.py`, benchmark script | done |
| P2 | ResUNet baseline | `models/baselines/resunet.py` | done |
| P3 | Vanilla SRNO | `models/baselines/srno_single.py` | done |
| P4 | GeoEncoder | `models/geo_encoder.py` (landuse embedding, no z-score) | done |
| P5 | HydroGeo-SRNO dual branch | `models/hydrogeo_srno.py` | done |
| P6 | Residual + flood loss | depth residual in log1p space + wet head + `FloodLoss` | done |
| P7 | Ablation configs | `configs/v0_10m2m_hmax_ablation.yaml` | done |
| P8 | Multi-resolution loader | `dataset/wellington_multiscale.py` | done |
| P9 | Arbitrary scale (5→2) | `allow_fractional_scale` + scale embedding | scaffolded |
| P10 | Dynamic h/hux/hvy | HydroEncoder `in_channels=4` path | scaffolded |
| P11 | Residual-LDM | stubs under `models/diffusion/` | stub only |
| P12 | Cross-city transfer | out of scope (no second city) | blocked |

## First-batch files (plan closing list)

- [x] `dataset/wellington_fixed_sr.py`
- [x] `dataset/normalization.py`
- [x] `models/galerkin.py`
- [x] `models/hydro_encoder.py`
- [x] `models/geo_encoder.py`
- [x] `models/scale_embedding.py`
- [x] `models/hydrogeo_srno.py`
- [x] `losses/flood_loss.py`
- [x] `metrics/flood_metrics.py`
- [x] `scripts/train_fixed.py`

## Hard constraints (honored)

1. Geographic split only (`patches_480m.json`); no reshuffle.
2. Nodata → NaN from reader; losses use mask; LR gets explicit `lr_valid` channel.
3. Landuse codes 1–7 categorical; embedding, never z-score.
4. No conservation loss forcing `pool(pred_hr) ≈ lr` (independent simulations).
5. No random flip/rotation in V0 (aspect / future flow vectors).
6. Best checkpoint by `val_CSI_005` / wet RMSE, not PSNR.
7. Coarse flood is independent simulation, not downsample of 2 m.
8. Raw `wellington-output-data/` and `static_geo_data/` remain read-only.

## Experiment matrix (V0)

| ID | Model | Script / config |
|----|-------|-----------------|
| B0 | Nearest | evaluate `--model nearest` |
| B1 | Bilinear | evaluate `--model bilinear` |
| B2 | ResUNet | `--model resunet` |
| B3 | EDSR | `--model edsr` |
| B4 | Residual SwinUNet | lightweight stub / optional |
| B5 | vanilla SRNO | `--model srno_single` |
| M1 | SRNO + DEM | ablation `geo_mode=dem` |
| M2 | HydroGeo-SRNO | `configs/v0_10m2m_hmax.yaml` |
| M3 | + flood loss | default V0 train |

## How to run main experiment

```bash
# Bilinear benchmark (no train)
python scripts/evaluate.py --config configs/v0_10m2m_hmax.yaml --model bilinear --split val

# Smoke train (pipeline proof)
python scripts/train_fixed.py --config configs/v0_10m2m_hmax_smoke.yaml

# Full V0 train (GPU recommended; 300 epochs)
python scripts/train_fixed.py --config configs/v0_10m2m_hmax.yaml
```
