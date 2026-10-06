# Implementation status (HydroGeo-SRNO)

Generated against plan `20260924-方案.md`. Workspace: `E:\Projects\20260921-JiaChenchen-Paper`.

## Completed (plan sections → files)

| Plan § | Item | Paths |
|--------|------|-------|
| 一, 二十八 P0–P6 | Project layout + V0 core | `configs/`, `models/`, `losses/`, `metrics/`, `engine/`, `scripts/`, `tests/` |
| 二–三 | Fixed SR adapter + `lr_valid` | `dataset/wellington_fixed_sr.py` |
| 四 | GeoEncoder + landuse embed | `models/geo_encoder.py`, `dataset/normalization.py` |
| 五 | HydroEncoder | `models/hydro_encoder.py` |
| 六–八 | HydroGeo-SRNO residual log1p | `models/hydrogeo_srno.py`, `models/implicit_query.py`, `models/galerkin.py`, `models/heads.py` |
| 九 | No LR conservation loss | enforced in `losses/flood_loss.py` + test |
| 十–十一 | FloodLoss + wet head | `losses/flood_loss.py` |
| 十二 | Flood metrics | `metrics/flood_metrics.py` |
| 十三 | V0 YAML | `configs/v0_10m2m_hmax.yaml`, `configs/v0_10m2m_hmax_smoke.yaml` |
| 十四 | No flip/rot | `dataset/transforms.py`, configs |
| 十五–十六 | Baselines + ablations | `models/baselines/*`, `configs/v0_10m2m_hmax_ablation.yaml` |
| 十七–二十 | Multiscale + fractional | `dataset/wellington_multiscale.py`, `allow_fractional_scale` in `wellington_sr.py`, `configs/v1_*.yaml` |
| 十八 | ScaleEmbedding | `models/scale_embedding.py` |
| 二十二–二十四 | Residual-LDM stubs | `models/diffusion/*`, `scripts/train_residual_ldm.py`, `configs/v2_residual_ldm.yaml` |
| 二十五–二十六 | Unified forward + trainer | `engine/trainer.py`, `scripts/train_fixed.py` |
| 二十七 | Tiled infer | `scripts/infer_domain.py`, `scripts/infer_tile.py` |
| Closing list | First 10 files | all present (see `IMPLEMENTATION_CHECKLIST.md`) |

## Stage / intermediate visualisation (CPU-only, runs beside training)

Full guide: `VISUALIZATION.md`. Live HTML report:
`outputs/v0_10m2m_hmax/visualizations/index.html`.

| Kind | Figure | Trigger |
|------|--------|---------|
| Process | `01_training_dashboard.png`, `01_progress_overview.png` | every refresh cycle |
| Dataset | `02_domain_splits.png`, `02_error_geography.png` | every 4th cycle |
| Physics | `03_diagnostics_{val,test}.png` (bias vs depth, error vs slope, land-use, model-vs-bilinear, mass, residual anchoring) | every 4th cycle; both splits in the final pass |
| Deep water | `04_bias_evolution.png` (depth-binned bias/RMSE per checkpoint, deep-water trend) | every 4th cycle |
| Intermediate | `epNNN/tiles_{val,test}/tile_*.png` (coarse → bilinear → model → truth, \|error\|, TP/FP/FN, Δz) | each cycle (representatives) / 4th cycle (full scan) |
| Milestone | `epNNN/tiles_{split}_overview.png`, `epNNN/metrics_vs_baseline_{split}.png` | each cycle |
| Evolution | `evolution/<tile>/evolution_predictions.png` | 4th cycle (needs `snapshots/`) |
| Tables | `epNNN/per_tile_metrics_*.csv/.json`, `diagnostics_*.json`, `bias_evolution.json` | with the scans |

Background jobs (survive agent exit):

| Job | PID file | Behaviour |
|-----|----------|-----------|
| `scripts/watch_checkpoints.py` | `snapshots/watcher.pid` | snapshots `last.pt`/`best_csi.pt` every 10 epochs into `snapshots/epNNNN.pt` |
| `scripts/auto_visualize.py` | `visualizations/auto_visualize.pid` | 25 min cycle: curves + representative tiles + gallery; every 4th cycle also full scan, diagnostics, evolution, bias evolution; on trainer exit runs an exhaustive final pass (every tile of both splits) and exits |

One-shot refresh: `python scripts/visualize_all.py` (add `--full` to scan every tile), or
`python scripts/auto_visualize.py --final-only` for the exhaustive pass immediately.

Key findings at epoch 130 (`best_csi.pt`) — val/test averages over the scanned tiles:

**Authoritative full-split evaluation (2026-09-27, `scripts/evaluate.py`, CUDA, every tile).**
Frozen V0 checkpoint `best_csi.pt` (ep155) vs the plan's baselines:

| Model | Split | Tiles | CSI_005 | CSI_100 | RMSE_wet |
|---|---|---|---|---|---|
| **HydroGeo-SRNO** | val | 182 | **0.5392** | **0.1686** | **0.5118** |
| **HydroGeo-SRNO** | test | 242 | **0.5987** | **0.2547** | **0.3986** |
| bilinear | val | 182 | 0.3141 | 0.1088 | 0.6767 |
| bilinear | test | 242 | 0.3762 | 0.1507 | 0.5465 |
| nearest | val | 182 | 0.3047 | 0.1047 | 0.7907 |
| nearest | test | 242 | 0.3798 | 0.1475 | 0.6632 |

Advantage over bilinear: val CSI_005 **+0.225** (+72% relative), test CSI_005 **+0.222**;
CSI@1.00 m val +55%, test **+69%**; RMSE_wet lower on both splits. These are the numbers to
quote — they scan **every** tile and come from the evaluator the plan designates for the final
table. Files: `outputs/post_sweep/metrics/*.json`, log `outputs/post_sweep/post_sweep.log`.

Note these are higher than the subset scans (test CSI_005 0.5987 vs 0.5546 on 60 tiles) because
the 60-tile subsets were not randomly sampled.

Earlier 60-tile subset scans, kept only for the same-tile checkpoint comparison:

| Split | CSI_005 | CSI_030 | CSI_100 | MAE_wet | RMSE_wet |
|-------|---------|---------|---------|---------|----------|
| val (60 tiles) | 0.5151 | 0.2984 | 0.1436 | 0.3227 | 0.5978 |
| test (60 tiles) | 0.5546 | 0.3788 | 0.1964 | 0.2421 | 0.4548 |

Checkpoint `ep155` (`best_csi.pt`, best_csi=0.5380). These supersede the earlier ep130 row
(val 0.5147 / test 0.5515). On the **identical** 60-tile sets the ep130 → ep155 change is
within noise (~0.000, CSI_005 −0.0004 val / +0.0031 test), i.e. the checkpoint is not
meaningfully better than ep130 despite the tiny new `best_csi`.

(Full-split single-tile scans earlier at ep78, 182 val / 242 test tiles, gave val CSI_005
0.526 / test 0.583 vs bilinear 0.313 / 0.375 — model better on **100%** of tiles for
CSI@0.05 m and volume accuracy on both splits.)

Mass: model has lower volume error than bilinear on 93% of val tiles and 100% of test tiles.
Residual anchoring correlation r = 0.547 (val) / 0.658 (test). On test tiles the model raises
CSI@1.00 m from 0.146 (bilinear) to ~0.19–0.29 (deepest-water class).

**Plateau confirmed (ep98 → ep155, identical 60-tile sets):**

| Split | Metric | ep098 | ep130 | ep142 | ep155 | Δ(ep98→ep155) |
|---|---|---|---|---|---|---|
| val | CSI_005 | 0.5157 | 0.5147 | 0.5173 | 0.5151 | **−0.0006** |
| val | CSI_100 | 0.1607 | 0.1527 | 0.1599 | 0.1436 | **−0.0171** |
| val | RMSE_wet | 0.5896 | 0.5798 | 0.5831 | 0.5978 | +0.0082 |
| test | CSI_005 | 0.5525 | 0.5515 | 0.5553 | 0.5546 | **+0.0021** |
| test | CSI_100 | 0.2044 | 0.1949 | 0.2090 | 0.1964 | **−0.0080** |
| test | RMSE_wet | 0.4546 | 0.4498 | 0.4508 | 0.4548 | +0.0002 |

Everything is flat or slightly worse. Over the last 40 epochs the fitted slopes are
CSI_005 −0.00001/ep, CSI_030 −0.00006/ep, CSI_100 −0.00004/ep, RMSE_wet +0.00002/ep.
**The model stopped improving around epoch 98–130 and 57 further epochs bought nothing.**
This is the single most important finding of this checkpoint review.

**Open risk (flagged, tracked):** depth-binned bias grows monotonically with depth
(val: +0.006 m on dry cells → −0.36 m at 0.5–0.75 m → −1.47 m above 3 m). Deep-water
accuracy is the weakest area (CSI@1.00 m is the lowest of the three CSIs). The extreme-depth
term in `FloodLoss` (`w_extreme=0.1`) is the first knob to investigate if this plateaus.

Deep-water bias trend over epochs 80→160 (30 fixed val tiles, fitted drift vs residual scatter):

| Quantity | Total drift | Noise scatter | Verdict |
|---|---|---|---|
| bias > 3 m | 0.133 m | 0.071 m | DRIFT > NOISE, but **worsening** (−1.94 → −2.02) |
| bias 1.0–1.5 m | 0.056 m | 0.023 m | DRIFT > NOISE, **worsening** (−0.69 → −0.72) |
| rmse > 3 m | 0.008 m | 0.034 m | FLAT vs NOISE |
| mean bias, all wet bins | 0.003 m | 0.001 m | DRIFT > NOISE, **worsening** (−0.012 → −0.014) |

Reading: with the longer, denser trace (6 checkpoints through ep160) the shallow drifts now
exceed the scatter, but every one of them points **the wrong way** — the deep-water bias and
the overall wet bias are slowly *growing*, and the >3 m RMSE is flat. So extra training is
not fixing the deep-water underestimation; it is mildly aggravating it. This is consistent
with the plateau above and makes a loss-side change (`w_extreme`) the clear next step.
CSI@1.00 m best-so-far is 0.1897 @ep129.

### Snapshot deduplication fix (2026-09-26)

`scripts/watch_checkpoints.py` previously copied `best_csi.pt` to `best_epNNNN.pt` **every
epoch**, with no check that the best had actually moved. Because `best_csi.pt` only changes
on a minority of epochs, 64 of the files were byte-identical duplicates (verified by MD5),
consuming 1.18 GB and — worse — making `visualize_evolution.py` render long flat segments
that looked like stagnation between genuine improvements (e.g. ep106/114/123 were identical).

Fixed:
- The watcher now only snapshots when `best_csi` differs from the last captured value, and
  persists that value in `snapshots/.watcher_state.json` so restarts do not re-capture.
- Added `python scripts/watch_checkpoints.py --prune`, which deletes every `best_ep*.pt`
  that is not a genuine best-so-far epoch per `history.jsonl`. Ran once: removed 59 files,
  reclaimed 1.05 GB, kept 26 genuine improvements.
- Watcher restarted (PID `15328`) with the fixed logic and seeded state.

No information was lost: stride `ep*.pt` snapshots already cover the pruned epochs, and every
epoch that actually set a new best is retained.

### Summary-reporting fix (2026-09-26)

`scripts/visualize_tiles.py` computed `summary_{split}.json`'s `avg_metrics` from the
per-tile scan **when a scan ran**, but silently fell back to the **4 hand-picked
representative tiles** when it did not. Because `auto_visualize.py` runs
`--representatives-only` every 25-minute cycle, each cycle overwrote a real 60-tile average
with a 4-tile one, and the gallery (`build_gallery.py`) reported that inflated number as the
model's score.

Evidence of the damage (val, `avg_basis` missing ⇒ stale 4-tile value):

| Tag | Reported CSI_005 | True 60-tile CSI_005 |
|---|---|---|
| ep078 | 0.5543 | 0.4838 |
| ep098 | 0.5682 | 0.5157 |
| ep142 | 0.5653 | 0.5173 |
| ep155 | 0.5671 | 0.5151 |

The reported val CSI_005 was inflated by roughly **+0.05** — larger than any real epoch-over-
epoch gain, which could have produced a false "improvement" claim in the paper.

Fixed:
- `avg_metrics` now always records provenance in a new `avg_basis` field
  (`full_scan_N_tiles` vs `representative_only_N_tiles`).
- A `--representatives-only` run no longer downgrades an existing full-scan average; it keeps
  the scanned value and prints that it is doing so.
- Added `python scripts/repair_summaries.py [--dry-run]` to recompute already-written
  summaries from their `per_tile_metrics_*.json` tables. Ran once: recomputed 14 summary
  files (idempotent; re-running reports `would fix: 0  already-correct: 14`).

Because this pipeline acts as the de-facto evaluator while training runs, the corrected
numbers are what should be quoted. The authoritative end-of-training evaluation (via
`scripts/run_v0_pipeline.ps1` → `scripts/evaluate.py`, full splits) still has to run.

### Root cause of the deep-water failure: two compounding loss defects (2026-09-26)

Measured with `python scripts/diagnose_loss.py` (CPU) and a per-tile probe. Component
magnitudes on val, checkpoint ep155:

| term | mean | weight | weighted | share |
|---|---|---|---|---|
| `l_depth` | 0.07274 | 1.00 | 0.07274 | 31.1% |
| `l_wet` | 0.01636 | 0.30 | 0.00491 | 2.1% |
| `l_log` | 0.16781 | 0.20 | 0.03356 | 14.4% |
| `l_boundary` | 0.34214 | 0.10 | 0.03421 | 14.6% |
| `l_extreme` | 0.88433 | 0.10 | 0.08843 | 37.8% |
| **total** | | | **0.23386** | 100% |

**Defect 1 — the log-space terms are nearly blind to deep magnitude.** `l_depth` (Huber) and
`l_log` (L1) are computed on the log1p residual `z = log1p(h/0.1)`, and together they are
**45.5% of the loss**. In log space a large absolute error is cheap:

| true h | error | log-space cost | cost in metres | ratio |
|---|---|---|---|---|
| 3.0 | 1.0 | 0.280 | 1.00 | 0.28 |
| 5.0 | 2.0 | 0.331 | 2.00 | 0.17 |
| 9.0 | 2.0 | 0.199 | 2.00 | **0.10** |

So a 2 m error at 5–9 m depth costs **6–10x less** in log space than in metres. The
best-fitting loss is therefore satisfied by under-predicting deep water, which is exactly the
observed monotone bias growth with depth.

**Defect 2 — the "extreme" term does not reliably cover the tail.** `l_extreme` *is* linear
(the only deep-sensitive term), but its mask is `target >= quantile(target, 0.95)` computed
**per tile**. Because most val tiles are mostly dry, that per-tile threshold is:

| statistic | value |
|---|---|
| min | 0.051 m |
| p25 | 0.095 m |
| median | **0.190 m** |
| p75 | 0.736 m |
| max | 5.053 m |

The mask covers 5.0% of valid cells and is ~100% wet — but at a tile-dependent depth that is
usually only ~0.19 m. So on typical tiles the term supervises shallow water, and where the
mask does reach deep water its contribution is gradient-clipped by `max_grad_norm=1.0`.

(Note: an earlier inference from the *aggregated* val histogram put this threshold at
0.3–0.5 m and concluded the term never saw deep water. The per-tile probe shows that was
wrong — coverage of true >1 m cells within the mask is 92.7% *where such cells exist*. The
defect is the tile-dependent threshold, not a total absence of deep supervision.)

**Also fixed while here:** the linear wet L1 was already being computed but discarded
(`+ 0.0 * l_metric`). It is now recorded as `l_wet_mae_linear` in `history.jsonl`, so
"objective progress" (what RMSE/MAE_wet measure) can be told apart from "training-loss
progress" — they are not the same quantity, since the loss is dominated by log-space and
per-tile-quantile terms.

### Fix: `l_deep`, a linear target-masked deep-water term

Added to `losses/flood_loss.py`, **off by default (`w_deep=0.0`) so legacy behaviour and all
published numbers reproduce exactly** (verified: with `w_deep=0` the loss is bit-identical to
before). It adds a Huber loss in **metre space** on the deep tail:

- mask = valid & `target >= deep_threshold` (default 1.0 m), or the `deep_quantile` of the
  **wet** cells floored at `deep_threshold`, so the mask cannot collapse onto shallow water;
- the source of truth is always the target mask — no fabricated depth tensor is built, so no
  gradient can flow through a constant;
- `deep_delta` (default 5.0 m) keeps Huber linear for realistic errors but bounds the influence
  of absurd ones.

On a synthetic deep pool (3 m cells predicted as 0.5 m), the total penalty for the deep error
rises from **0.1196 → 1.6821 (14x)** while `l_depth`/`l_log` stay unchanged at 0.0179/0.0257.
Gradient flow was verified to reach the deep region.

### Deep-water loss sweep (RUNNING: focused 2-arm comparison)

`scripts/run_deep_sweep.py --only D0 D2 --epochs 40 --patience 40` — sequential, restart-safe,
one arm at a time (single 4 GB GPU). It was launched automatically by
`scripts/chain_after_pid.py` the moment the baseline evaluation exited, so the GPU never hosts
two jobs at once and nothing needed babysitting.

Controlled: only the deep term varies; seed, architecture, data order, lr schedule, epoch
budget and per-epoch val subset are identical, and `loss.name` is unchanged so every other
term is byte-identical. D0 = control (`w_deep=0`, reproduces the original loss exactly);
D2 = `w_deep=0.5, deep_delta=5.0`.

**Timing correction (important).** The first launch used 5 arms x 40 epochs, which I initially
(and wrongly) estimated at ~5 h. Measured cost is **791 s = 13.2 min per epoch** (including the
full val evaluation), so 5x40 = 200 epochs is **~44 h**, not 5 h. I had conflated "40 epochs per
arm" with the total. The sweep was stopped at D0 epoch 3 to avoid burning the GPU on a
mis-stated plan. Measured from D0: ep1 sec=795, ep2 sec=788, ep3 sec=791.

A second consideration killed the "many arms, few epochs" shortcut: at ep1-3 the control is
still rising steeply (CSI_005 0.386 -> 0.405 -> 0.431), so a 10-12 epoch sweep would have
ranked arms on transient behaviour rather than on the deep term. Short sweeps are not valid
here, so the budget was not reduced that way.

Revised plan (running): **2 arms x 40 epochs = 80 epochs = ~17.6 h**. Arms D1/D3/D4 remain
defined for a later sensitivity pass.

**Cost model for any remaining work:** ~13.2 min per epoch per arm. 2 arms x 40 ep = 17.6 h;
3 arms x 25 ep = 16.5 h; 5 arms x 40 ep = 44 h.

`test` is held out and read only once for the winner. Status per arm:
`outputs/deep_sweep/<tag>/status.json`; summary: `outputs/deep_sweep/sweep_summary.json`;
completion marker: `outputs/deep_sweep/SWEEP_DONE`.

#### FINAL result (2026-09-27 21:47, both arms complete)

Both arms finished 40 epochs. D2 finished at 18:29:50 (8.8 h). Formally decided by
`post_sweep.py` under the pre-registered rule:

```
arms done: ['D0', 'D2']
selection: ref D0 best_csi=0.4977 CSI_100=0.1487 tolerance=0.01
selection: rejected by CSI_005 guardrail: D2(csi005=0.4487 < 0.4877)
selection: eligible -> D0(CSI_100=0.1487)
WINNER = D0  (no arm beat the control on CSI_100)
```

**D2 was rejected and nothing was resumed — exactly as predicted.** Note the second half of the
verdict is worth reading carefully: once D2 is rejected, D0 is the only eligible arm, so the
comparison degenerates to `D0 vs D0` and the message "no arm beat the control on CSI_100" is
*not* evidence that the deep term failed to raise CSI_100. **It did raise it**
(0.1487 → 0.1707, +15%, on the identical 60-tile subset; and 1.88 m vs 2.66 m peak depth error).
The rule rejected D2 purely on the overall-accuracy guardrail, and the logged CSI_100 comparison
is vacuous by construction. `D2`'s own `status.json` is the number to quote for the deep gain:

| arm | best CSI_005 | CSI_030 | CSI_100 | RMSE_wet | MAE_wet | PeakDepthError | VolErr |
|---|---|---|---|---|---|---|---|
| D0 (control) | **0.4977** | 0.2868 | 0.1487 | 0.6030 | 0.3219 | 2.661 m | 0.386 |
| D2 (w_deep=0.5) | 0.4487 | 0.2784 | **0.1707** | 0.6540 | 0.3608 | **1.881 m** | **0.309** |

So the honest summary of the from-scratch sweep: `l_deep` **does** improve the deep tail
(CSI_100 +15%, peak depth error −0.78 m, volume error −20%) but at a cost to overall CSI_005
(−0.049) that no 40-epoch budget recovers under this weighting. See the depth-binned evidence
and the measured weighting defect below.

#### Interim result (2026-09-27 14:35, D0 done / D2 at epoch 22 of 40)

**Decisive evidence — depth-binned val error, each arm at its own `best_csi.pt`**
(D0 = ep18, D2 = ep20; 182 val tiles, 10 340 656 valid pixels, from
`outputs/deep_sweep/*/visualizations/diagnostics_val.json`):

| depth bin | n | D0 MAE | D2 MAE | Δ | D0 RMSE | D2 RMSE | D0 bias | D2 bias | D0 under% | D2 under% |
|---|---|---|---|---|---|---|---|---|---|---|
| 0–0.05 | 8 847 913 | 0.0135 | 0.0267 | **+98%** | 0.058 | 0.135 | +0.008 | +0.021 | 7.3 | 8.5 |
| 0.05–0.1 | 468 001 | 0.0744 | 0.1024 | +38% | 0.166 | 0.286 | +0.033 | +0.057 | 48.9 | 54.7 |
| 0.1–0.2 | 303 884 | 0.1266 | 0.1701 | +34% | 0.219 | 0.363 | +0.029 | +0.057 | 56.0 | 64.3 |
| 0.2–0.3 | 147 053 | 0.1877 | 0.2512 | +34% | 0.276 | 0.417 | −0.019 | +0.019 | 64.2 | 69.8 |
| 0.3–0.5 | 185 304 | 0.2683 | 0.3408 | +27% | 0.341 | 0.469 | −0.117 | −0.086 | 73.2 | 75.9 |
| 0.5–0.75 | 144 928 | 0.3772 | 0.4688 | +24% | 0.446 | 0.566 | −0.267 | −0.235 | 81.6 | 80.0 |
| 0.75–1 | 78 968 | 0.5284 | 0.5987 | +13% | 0.620 | 0.705 | −0.422 | −0.354 | 85.5 | 80.4 |
| **1–1.5** | 70 164 | 0.8451 | 0.8222 | **−2.7%** | 0.948 | 0.947 | −0.750 | −0.580 | 90.3 | 82.0 |
| **1.5–2** | 40 045 | 1.0927 | 1.0102 | **−7.6%** | 1.249 | 1.172 | −1.013 | −0.829 | 91.9 | 84.7 |
| **2–3** | 33 074 | 1.5366 | 1.4256 | **−7.2%** | 1.783 | 1.645 | −1.358 | −1.227 | 90.1 | 87.3 |
| **3+** | 21 322 | 2.5811 | 2.4688 | **−4.3%** | 3.013 | 2.872 | **−1.755** | **−2.103** | 75.0 | 84.6 |
| **all** | 10 340 656 | 0.0556 | 0.0724 | **+30.3%** | — | — | — | — | — | — |

Deep aggregate (≥ 1 m, n = 164 605): MAE **1.2691 → 1.2025 (−5.3%)**, RMSE
1.6052 → 1.5257 (−5.0%), bias −1.0661 → −0.9681. The underestimation *fraction* falls
at every deep bin below 3 m (e.g. 1–1.5 m: 90.3% → 82.0%).

So `l_deep` **is** doing what it was designed to do — it is the first change that has moved
deep-water error at all, and it moves it in the right direction at 1–3 m. But three caveats
matter and are the honest reading:

1. **The gain is modest (−5%, not the −45% the per-epoch `PeakDepthError` suggested).**
   `PeakDepthError` is a single-worst-cell statistic over a 60-tile subset and is dominated by
   one or two tiles; it fell 2.66 m → 1.72 m, but the *distribution-wide* deep MAE only falls
   5%. Earlier reports should not have leaned on `PeakDepthError` as the headline.
2. **The ≥ 3 m tail does not improve in the way we want.** MAE/RMSE go down slightly, but the
   *bias* gets **worse** (−1.755 → −2.103 m) and the underestimated fraction **rises** 75.0% →
   84.6%. The deepest pools/channels are still pushed too shallow; the small MAE gain comes from
   fewer mid-bin blow-ups, not from fixing the tail.
3. **The cost is severe and lands on dry/near-dry ground.** Overall MAE rises 30%, driven by the
   0–0.05 m bin (8.85 M cells = 86% of all pixels) where MAE and RMSE roughly double. A loss that
   only sees cells ≥ 1 m should not be able to do this — the mechanism is gradient
   interference, and the loss table shows it plainly:

   | arm | epoch | total loss | weighted `l_deep` | share of total |
   |---|---|---|---|---|
   | D0 | 22 | 0.1315 | — | 0% |
   | D2 | 22 | 0.4415 | 0.5 × 0.5169 = 0.2585 | **59%** |

   With `w_deep=0.5`, `l_deep` is by far the single largest term in the objective, and
   `l_extreme` simultaneously *rose* (0.4516 → 0.4885) — the two tail terms are fighting.
   This is a **weighting defect, not a conceptual failure of `l_deep`**.

**Guardrail trajectory (why D2 will not be selected).** The rule is fixed in advance:
val `best_csi` must be within 0.01 of D0's `best_csi`. D0's is 0.4977, so the floor is 0.4877.

| | best `CSI_005` | epoch |
|---|---|---|
| D0 (floor reference) | **0.4977** | 18 |
| D2, so far | 0.4470 | 20 |
| D2, needed | ≥ 0.4877 | — |

Gap = **0.0407 with 18 epochs left** (~4 h at 13.2 min/epoch, ETA ~18:30). **Outcome: confirmed
rejected** — D2 ended at best CSI_005 0.4487, i.e. 0.0390 below the 0.4877 floor.

**Recommended follow-up (implemented — see below).** Two conclusions follow:
- *From scratch*, `w_deep=0.5` is far too aggressive (59% of the objective) and the 40-epoch
  budget cannot recover `CSI_005`. A gentler arm (`D1`, `w_deep=0.25`) would very likely land
  inside the guardrail.
- The frozen V0 (`best_csi.pt` ep155 / `last.pt` ep180) is **already good** (full-split val
  CSI_005 0.5392); the open question is whether `l_deep` can improve *its* deep tail. That is a
  **fine-tune**, not the from-scratch sweep, and it needs only ~20 epochs.

### Fine-tune of the frozen V0 (`scripts/run_deep_finetune.py`)

The frozen `last.pt` (ep180) is a better starting point than `best_csi.pt` (ep155) for this
purpose — at equal val subset it already has **CSI_100 0.1821 vs 0.1664** and
**PeakDepthError 2.32 m vs 2.50 m**, even though its overall CSI_005 is lower
(0.5316 vs 0.5380). So the fine-tune resumes from `last.pt`.

**Two traps found and fixed in `scripts/train_fixed.py` (both empirically verified).**

1. **`--epochs` is an absolute index and the scheduler fast-forwards.** Resuming ep180 with the
   intuitive `--epochs 175` ("20 more epochs") gives 0 epochs and **lr = 2.3e-7**; `--epochs 200`
   gives lr = 2.7e-6. Either way the fine-tune would burn 4+ h learning nothing. Fix: a new
   `--sched_epochs N` builds a fresh cosine horizon over exactly N epochs, derives
   `epochs = start + N - 1`, and skips the fast-forward.
2. **`optimizer.load_state_dict` restores the checkpoint's own lr** (3.66e-5 at ep180) and
   silently clobbers the constructed `lr`. Fix: in `--sched_epochs` mode the lr is re-applied
   *after* `load_checkpoint`, and `--lr` overrides it.

Verified end-to-end on CPU against the real checkpoint: `lr reset to 1.000e-04 ... epochs
181..200`, and every logged step prints `lr=1.00e-04`.

**Critical weighting correction (measured, not assumed).** I probed `FloodLoss` components on
the frozen ep180 checkpoint (val, 8 batches, CPU):

| component | raw | weight | weighted | share (w_deep=0) |
|---|---|---|---|---|
| `l_depth` | 0.0695 | 1.00 | 0.0695 | 34.3% |
| `l_extreme` | 0.6567 | 0.10 | 0.0657 | 32.4% |
| `l_log` | 0.1599 | 0.20 | 0.0320 | 15.8% |
| `l_boundary` | 0.3104 | 0.10 | 0.0310 | 15.3% |
| `l_wet` | 0.0155 | 0.30 | 0.0047 | 2.3% |
| **total** | | | **0.2028** | 100% |

raw `l_deep` = **1.5684** here, ~3x its value in D2 (0.5169), because the frozen model still
carries a −2 m deep bias. So the *same* `w_deep` buys far more influence than it did in the
from-scratch sweep:

| `w_deep` | weighted `l_deep` | share of objective |
|---|---|---|
| 0.05 | 0.0784 | **27.9%** (comparable to `l_depth`/`l_extreme`) |
| 0.10 | 0.1568 | 43.6% |
| 0.20 | 0.3137 | 60.7% |
| **0.25** | 0.3921 | **65.9% — more dominant than D2's 59%** |
| 0.50 | 0.7842 | 79.5% |

**This invalidates the earlier plan to fine-tune at `w_deep=0.25`.** On this checkpoint 0.25 is
*not* a gentler half — it reproduces D2's defect (the deep term dominating the objective) and
would predictably flatten overall accuracy again. Gentle values here are **0.05–0.10**, so the
armed chain uses those.

Also added `best_deep_csi.pt` (+ `best_deep_summary.json`): `best_csi.pt` is selected on
CSI_005, which is **not** what a deep-tail fine-tune is aiming at, so the deep-tail checkpoint
is now tracked separately.

Armed: `chain_after_pid.py` (PID 18516) waits for the sweep (33776), then runs
`run_deep_finetune.py --arms 0.05 0.10 --epochs 20 --lr 1e-4`. Selection: a candidate is
eligible only if its full-split val CSI_005 is within **0.005** of frozen V0 (tighter than the
from-scratch 0.01, since we start from an already-good checkpoint) and it **improves CSI_100**;
`test` is read once for the winner only. If nothing qualifies, the verdict is written as a null
result and nothing is kept.

**Timing correction (again, the same class of mistake as the 5-arm estimate).** Measured
fine-tune cost is **790 s = 13.2 min per epoch** (including the 60-batch val eval), so
20 epochs = **4.4 h per arm** and 2 arms = **~8.8 h total**, started 21:40 => ETA ~06:26.

#### Two measurement traps found, and the control arm that fixes the second one

1. **The scheduler/lr traps** (fixed earlier, verified live: `lr reset to 1.000e-04`, verified
   `epochs 181..200`).
2. **The reference-basis trap.** The per-epoch `CSI_100` that picks each arm's checkpoint comes
   from the **60-batch val subset**, whereas the frozen V0 number (0.1686) and every figure in
   the main results table come from the **full 182-tile split**. Reading a fine-tuned arm's
   per-epoch `CSI_100` against the frozen full-split value would silently mix "what `l_deep`
   did" with "which subset happened to be measured" — the same class of error as the earlier
   representative-tile averaging mistake, one level up.

   Interim numbers show how easy it would be to misread: at `w_deep=0.05` the per-epoch
   `CSI_100` sits at **0.157–0.174**, i.e. *below* the frozen **full-split** 0.1686, which
   looks like the fine-tune is harming the deep tail. It is not necessarily: on the 60-batch
   subset the from-scratch **D0 control scored only 0.1487**, so on a like-for-like basis the
   fine-tune may still be ahead. The two references are simply not comparable.

   Fix: **a `w_deep=0` fine-tuned control arm** is trained for the same 20 epochs from the same
   frozen checkpoint, and every candidate is compared against *that* control, on the full val
   split, with the same evaluator. The comparison then isolates the deep term. `select()` refuses
   to use the frozen full-split number as the reference while a control exists, and logs which
   reference it used.

   Also note the per-epoch `CSI_100` is visible in the *frozen checkpoint itself*: `last.pt`
   (ep180) has CSI_100 0.1821 while `best_csi.pt` (ep155) has 0.1664 — i.e. **fine-tuning
   already lowered CSI_100 before any training**, purely as an artefact of which checkpoint we
   start from. The control arm absorbs all of this.

Armed after the current run: `chain_after_pid.py` (PID 34972) waits for the fine-tune driver
(11840), then runs `run_deep_finetune.py --arms 0.0 0.05 0.10 --epochs 20 --lr 1e-4`. Arms
0.05/0.10 are already finished by then and are skipped (their status is `done`), so only the
control trains (~4.4 h), then the full-split comparison runs automatically.


#### First run failed instantly — and the failure was nearly misfiled as a conclusion

The armed chain fired at 18:29:52 and both arms exited with **code 1 after 7 seconds**. Root
cause was self-inflicted: while cleaning up probe scripts earlier I had run
`$env:CUDA_VISIBLE_DEVICES=""` in the agent shell, and `Start-Process` inherited that empty
variable, so PyTorch hid the GPU. `train_fixed.py`'s guard did its job and aborted loudly
("CUDA_VISIBLE_DEVICES is empty: training would silently run on CPU") rather than quietly
training on CPU at ~50x slower.

**The serious problem was in my own driver, not the trainer.** `run_deep_finetune.py` treated
exit=1 as a finished arm: it wrote `state: "done"`, skipped the (impossible) evaluation, and
emitted `VERDICT: no fine-tune improved deep-water CSI_100 ... The deep term does not help the
frozen model either`. That is a **fabricated scientific conclusion derived from a crash**. It
also consumed the arms: because the crashed dirs were marked `done`, a naive re-run would have
skipped them and reproduced the same false verdict forever.

Three fixes, all in `scripts/run_deep_finetune.py`:
- `clean_env()` strips an **empty** (not merely unset) `CUDA_VISIBLE_DEVICES` from every child
  process, so the trainer can never be starved of the GPU by an inherited shell variable.
- An arm is only skipped if it genuinely finished (clean exit **and** a `best_deep_csi.pt`).
  Failures are recorded as `state: "failed"` and are **retried**, never skipped.
- If any arm fails, the driver **aborts** with `status: "aborted_arms_failed"` instead of
  writing a verdict. A crashed run can no longer masquerade as a null result.

Side benefit of the failure: it confirmed `train_fixed.py`'s CPU-fallback guard works.

**`select()` was branch-tested before being trusted.** Because it runs unattended overnight, its
four control-flow branches were exercised with synthetic metrics on a redirected output dir:
(1) control present + no deep gain => null result; (2) deep gain but accuracy outside tolerance =>
guardrail rejects it; (3) no control arm => falls back to the frozen full-split reference and
rejects a below-frozen score; (4) control + genuine winner => correct ranking and `test` read
once for the winner only. All four pass. This caught one wrong expectation of mine rather than a
code fault, but the logging it produced confirmed the reference basis is stated explicitly in
`selection.json` (`reference: fine-tuned control (w_deep=0), full-split val`).


#### Interim fine-tune signal (2026-09-27 23:29, w_deep=0.05, epochs 181-188)

Per-epoch metrics on the 60-batch val subset (comparable to the D0 control's 0.1487 there, **not**
to the frozen full-split 0.1686 — see the reference-basis note above):

| epoch | CSI_005 | CSI_100 | PeakDepthError | `l_deep` | sec |
|---|---|---|---|---|---|
| 181 | 0.5119 | 0.1737 | 2.385 m | 0.233 | 805 |
| 182 | 0.5214 | 0.1700 | 2.663 m | 0.218 | 793 |
| 183 | 0.5087 | 0.1593 | 2.670 m | 0.196 | 791 |
| 184 | 0.5079 | **0.1686** | 2.316 m | 0.194 | 791 |
| 185 | 0.5177 | 0.1626 | 2.396 m | 0.179 | 787 |
| 186 | 0.5130 | 0.1660 | 2.612 m | 0.172 | 781 |
| 187 | 0.5091 | 0.1717 | 2.498 m | 0.161 | 784 |
| 188 | 0.5125 | 0.1573 | 2.611 m | 0.158 | 785 |

Observations that can already be made honestly:
- `l_deep` falls monotonically **0.233 → 0.158** over 8 epochs, so the term is being actively
  optimised, not starved (consistent with it being ~28% of the objective at `w_deep=0.05`).
- **CSI_005 is stable at ~0.51**, versus our earlier from-scratch finding that `w_deep=0.5` cost
  0.049 of overall CSI_005. Keeping overall accuracy while the deep term is optimised is the
  whole point of using a gentle weight, and so far that holds.
- `CSI_100` is **noisy at this subset size** (0.157–0.174, no trend over 9 epochs), and mean
  peak depth error is ~2.5 m — *not* better than the frozen checkpoint's 2.32 m. **No claim is
  made yet:** 9 of 200 arm-epochs is far too few, and the decisive comparison is the control
  arm on the full val split.

**Measured noise budget for the subset `CSI_100` (9 epochs, w_deep=0.05).** This matters because
it bounds what can honestly be concluded from `argmax CSI_100` over 20 epochs:

| metric | min | max | range | stdev |
|---|---|---|---|---|
| `CSI_100` | 0.1573 | 0.1737 | 0.0164 | **0.0052** |
| `CSI_005` | 0.5079 | 0.5214 | 0.0135 | 0.0041 |
| `PeakDepthError` | 2.316 m | 2.670 m | 0.355 m | 0.123 m |
| `l_deep` | 0.151 | 0.234 | 0.083 | 0.026 |

One-sigma epoch noise on `CSI_100` is **0.0052**, i.e. **24% of the +0.022 deep gain** achieved
by D2 from scratch, and **33% of the 0.016 gap** between the two candidate starting checkpoints
(`last.pt` 0.1821 vs `best_csi.pt` 0.1664). Consequences:
- `best_deep_csi.pt` (= argmax of this noisy signal) is **partly selected on noise**, so a
  bare "we fine-tuned and the deep score went up" claim from this subset would not be sound.
  This is exactly why per-epoch snapshots are taken and the final choice is re-made on the
  **full** val split with the control arm as reference.
- `PeakDepthError` deserves particular suspicion: its epoch noise is ±0.12 m while the D2 gain
  was 0.78 m, so it is usable but should always be reported with its spread, not as a point
  value (this is the second time this metric has been over-interpreted in this project).
- The mean trend over 9 epochs is **flat-to-slightly-negative** (CSI_100 first-3 0.1677 vs
  last-3 0.1652; CSI_005 0.5140 vs 0.5114), while `l_deep` clearly falls (0.2157 → 0.1567).
  So the optimiser is driving its own term down **without yet moving the deep-water metric** —
  a real possibility that the honest summary must allow for, and the from-scratch sweep showed
  the same pattern (loss down, distribution-level deep MAE down only ~5%).




#### Per-epoch snapshots, so the reported epoch is chosen on the FULL val split

The per-epoch `CSI_100` used for checkpointing comes from the 60-batch val subset, which has too
few deep pixels for a stable deep-water score. Choosing a headline "best deep epoch" on it would
repeat the earlier representative-tile averaging mistake. The driver therefore starts
`watch_checkpoints.py --stride 1` per arm, snapshotting every epoch (~17 MB each, 20/arm), so
the deep-water best can be re-selected afterwards on the **full** 182-tile val split.





#### Fine-tune arms done; two bugs found on review (2026-09-28 08:40)

Both fine-tune arms **completed** 20 epochs (181..200, 4.4 h each) and wrote
`best_deep_csi.pt` / `best_deep_summary.json` / 20 per-epoch snapshots. No result was reported
overnight because of two bugs — both the same family as the earlier "crash misfiled as a
conclusion" incident, now in the **opposite direction**: a *success* misfiled as a *crash*.

**Bug A — a benign exit code treated as a crash.** `train_fixed.py` exits **120** on this box
even on a fully successful run: the interpreter prints `Exception ignored on flushing sys.stdout`
at teardown and Python turns that into a non-zero status. Every previously **accepted** arm has
the identical code — `deep_sweep/D0` and `deep_sweep/D2` are both `exit_code: 120,
state: "done"`. The driver tested `if code != 0` and so wrote `state: "failed"` for both
completed arms and logged `ABORT: ... refusing to draw a scientific conclusion from a crashed
run`. Since `train_arm()` only skips `state in ("done","running")`, the live re-run would have
**retrained both finished arms from scratch (~8.8 h wasted)** and *still* produced no result.

Evidence the arms were complete (each checked independently before reclassifying):

| check | w_deep=0.05 | w_deep=0.10 |
|---|---|---|
| `stdout.log` sentinel | `done. checkpoints ... elapsed_h=4.38` | `... elapsed_h=4.35` |
| `history.jsonl` | 20 rows, ep181..200 | 20 rows, ep181..200 |
| `best_deep_csi.pt` | present (17 MB) | present (17 MB) |
| `snapshots/ep0181..0200.pt` | 20 files | 20 files |

Fixed:
- `run_deep_finetune.py` now decides completion from the trainer's **own artefacts**
  (`trainer_reached_end()`: the `done. checkpoints` sentinel **and** a non-empty `history.jsonl`
  **and** a written `last.pt`), not from the process exit code, and logs a separate message when a
  non-zero code is a benign teardown artefact. A real crash is still retried/aborted on.
- `scripts/_reclassify_arms.py` re-verified each arm's artefacts (20/20 epochs, populated
  `best_deep_summary.json`) and restored `state: "done"` for `w005`/`w01`, keeping the original
  label under `init_state`. Nothing was lost but the label.

**Bug B (the scientific one) — the reported epoch was picked on a noisy subset, and the "deep
gain" rested on that pick.** `best_deep_csi.pt` is the `argmax` of the per-epoch **60-batch val
subset** `CSI_100`. That subset is far too small for the deep tail: the `w_deep=0` control arm —
which by construction *cannot* gain from a deep term — wandered over **0.1407..0.1711** across its
own epochs on pure noise (range 0.030, sigma ~0.005). An `argmax` over ~20 epochs is therefore a
**max-of-noise** statistic, positively biased by ~+0.012 (its own best 0.1711 vs its own mean
~0.159). Measured on that same subset basis, the candidates' lead over the control is **inside**
that bias:

| quantity | value | source |
|---|---|---|
| control subset `CSI_100`, min..max over its epochs | 0.1407 .. 0.1711 | `w00/history.jsonl` |
| control subset `CSI_100`, range / sigma | 0.0304 / ~0.005 | same |
| candidate `w_deep=0.05` best subset `CSI_100` | 0.1737 | `w005/best_deep_summary.json` |
| candidate `w_deep=0.10` best subset `CSI_100` | 0.1736 | `w01/best_deep_summary.json` |
| candidate best − control best | **+0.0026 / +0.0025** (~0.5 sigma) | — |

So on the subset basis the two candidates are **not separable from the control's own noise**.
Headlining an epoch chosen by `argmax` of this signal would be unsound — the same error as the
earlier representative-tile averaging, one level up.

**The correct basis, and it is running.** `scripts/scan_epochs.py` re-scores **every** per-epoch
snapshot on the **full 182-tile val split** with the same evaluator used for the baseline table
(~2.4 min/eval => ~50 min per 20-epoch arm), re-selects each arm's best epoch on that full basis,
and compares candidates against the control **on that same basis**. Launched as PID **39964**,
chained behind the control trainer so the single 4 GB GPU is never oversubscribed. Contract:
`outputs/finetune_deep/epoch_scan.json` + `outputs/finetune_deep/scan.log`.

Note the correct denominator: the frozen full-split `CSI_100` **0.1686** is the external
reference for the paper table, but "does `l_deep` help the frozen model" is a **paired**
comparison — fine-tuned `0.05`/`0.10` vs fine-tuned `0.0` control, same epoch budget, same split,
same evaluator. The live driver still emits its own `selection.json` from a single full-val pass
over each arm's subset-selected checkpoint; that is a valid first read but its epoch choice is
provisional until the scan supersedes it.

**Paired evidence that already exists (and it is more trustworthy than the subset CSI_100).**
`visualizations/bias_evolution.json` is computed by the visualisation pipeline on a **fixed panel
of val tiles** for every checkpoint, so unlike the per-epoch subset `CSI_100` it is *paired*: the
epoch-to-epoch change is attributable to the weights, not to which tiles/deep pixels were sampled.
Extracted with `scripts/_summarize_bias_evo.py`:

| arm | epoch | bias 1.0–1.5 m | bias > 3 m | RMSE > 3 m | mean wet bias |
|---|---|---|---|---|---|
| **w_deep=0.05** | 181 (ep180 weights) | −0.6606 | −1.6743 | 2.5981 | −0.01060 |
| | 200 | −0.7239 | −1.7571 | 2.6528 | −0.01452 |
| | **Δ** | **−0.0633** | **−0.0828** | +0.0547 | −0.0039 |
| **w_deep=0.10** | 181 | −0.6829 | −1.1485 | 2.4861 | −0.01273 |
| | 200 | −0.7176 | −1.8529 | 2.6927 | −0.01374 |
| | **Δ** | **−0.0346** | **−0.7044** | +0.2066 | −0.0010 |
| **w_deep=0.00** (control) | 181 | −0.6340 | −1.5400 | 2.6797 | −0.00968 |
| | 186 | −0.6785 | −1.7068 | 2.6769 | −0.01152 |
| | **Δ** | −0.0445 | −0.1667 | −0.0028 | −0.0018 |

Reading (interim, and only as strong as a fixed panel allows):
- **The control drifts the same way as the candidates.** Its `bias 1.0–1.5 m` moves −0.0445 over
  just 5 epochs, i.e. within the range of `w_deep=0.05`'s −0.0633 over 19 epochs. So the 1–1.5 m
  bias drift is **not** evidence for `l_deep` specifically — it is what 20 epochs of this lr does
  to the model regardless of the deep term. This is exactly what the control arm was trained for,
  and it is why the subset `argmax` must not be headlined.
- `w_deep=0.10` shows a **large** `bias > 3 m` degradation (−1.15 → −1.85, and worse than its own
  epoch-181 value by −0.70) with `RMSE > 3 m` **up** +0.21. That is the mild-weight version of D2's
  deepest-bin defect: more deep pressure, but the tail bias gets *worse*, not better.
- `w_deep=0.05` moves the >3 m bias by −0.083 with RMSE +0.055 — small compared with the control's
  own excursion at 183 (−2.76 m, an outlier epoch), so it is not yet a demonstrated gain either.
- **Conclusion so far: no arm has yet been shown to improve the deep tail.** The only honest
  claim is that `l_deep` is being optimised (it fell 0.233 → 0.158 at `w_deep=0.05`) without a
  demonstrated deep-metric gain — the same pattern the from-scratch sweep showed (loss down,
  distribution-level deep MAE down only ~5%, deepest bin bias worse). The full-split scan decides
  it on the proper basis; nothing is claimed until then.

### FINAL fine-tune result (2026-09-28 15:50) — `l_deep` **does** help the frozen model

The full-val per-epoch scan finished 12:36 (`scripts/scan_epochs.py`, PID 39964) and the winner
was read once on test. **This reverses the training-time subset signal** — and the reversal is
exactly what Bug B predicted, so it is worth stating plainly: the subset `argmax` was *not*
reliable in either direction, and only the paired full-split comparison settled it.

![paired_fullval_deep_csi.png](outputs/finetune_deep/visualizations/paired_fullval_deep_csi.png)

**Paired design (why this is decision-grade).** All three arms resume from the *same* frozen
ep180 checkpoint and run 20 epochs with identical seed / data order / lr schedule; **only
`w_deep` differs**. Epoch N of each arm is therefore a matched condition, so the per-epoch
candidate-minus-control difference is a **paired** measurement (n = 20) and cannot be inflated by
picking a lucky epoch.

| paired vs `w_deep=0` control (n=20 epochs) | mean diff | t | candidate better in | verdict |
|---|---|---|---|---|
| **CSI_100** @ `w_deep=0.05` | **+0.00635** | **+4.85** | 18/20 | **significant** |
| **CSI_100** @ `w_deep=0.10` | **+0.00816** | **+5.72** | 17/20 | **significant** |
| CSI_005 @ `w_deep=0.05` | +0.00004 | +0.08 | 9/20 | no change |
| CSI_005 @ `w_deep=0.10` | −0.00058 | −0.85 | 10/20 | no change |
| VolumeRelativeError @ `w_deep=0.05` | −0.01420 | −4.56 | 19/20 | significant |
| VolumeRelativeError @ `w_deep=0.10` | −0.01994 | −8.14 | 18/20 | significant |
| PeakDepthError @ `w_deep=0.10` | −0.03260 | −2.10 | 15/20 | borderline |

(`|t| > 2.09` is p < 0.05 for n = 20.)

**The gain is not a lucky-epoch artefact.** For `w_deep=0.10` the early-epoch mean CSI_100
(ep181–185) is 0.1851 and the late-epoch mean (ep196–200) is **0.1901** — it *rises* and holds.
Even taking the final epoch with no selection at all, ep200 = 0.1903 vs the control's 0.1814
(**+0.0089**). That is the strongest form of the evidence: no epoch choice is needed.

**Winner (pre-registered rule: best full-split val CSI_100 with CSI_005 inside 0.005 of control).**

| | CSI_005 | CSI_030 | CSI_100 | RMSE_wet | PeakDepthErr | VolErr |
|---|---|---|---|---|---|---|
| frozen ep180 (start) | 0.5329 | 0.3570 | 0.1840 | 0.5133 | 2.324 m | 0.287 |
| control `w_deep=0` (mean over 20 ep) | 0.5356 | 0.3547 | 0.1802 | 0.5134 | 2.383 m | 0.288 |
| **winner `w_deep=0.10` @ep187** | **0.5333** | 0.3561 | **0.1980** | **0.5071** | **2.221 m** | **0.263** |
| candidate `w_deep=0.05` @ep181 | 0.5373 | 0.3609 | 0.1970 | 0.5176 | 2.121 m | 0.244 |

On val the winner lifts CSI_100 **0.1840 → 0.1980 (+7.6%)**, peak depth error
**2.324 → 2.221 m**, volume error **0.287 → 0.263**, while CSI_005 moves −0.0034 (inside the
0.005 guardrail) and RMSE_wet *improves* 0.5133 → 0.5071. Val selection rule and evidence:
`outputs/finetune_deep/selection.json`.

**Test (read once, winner only, 242 tiles) — the deep gain replicates:**

| | CSI_005 | CSI_030 | CSI_100 | RMSE_wet | PeakDepthErr | VolErr |
|---|---|---|---|---|---|---|
| frozen V0 | 0.5987 | 0.4666 | 0.2547 | 0.3986 | 1.614 m | 0.194 |
| **winner `w_deep=0.10` @ep187** | 0.5936 | 0.4691 | **0.2754** | **0.3913** | **1.410 m** | **0.171** |

CSI_100 **0.2547 → 0.2754 (+8.1%)**, peak depth error **1.614 → 1.410 m** (−0.204 m), RMSE_wet
and volume error both improve, CSI_005 −0.0051. The deep gain is therefore **not a val-overfit
artefact**: it competes on a split (242 tiles) never used for selection.

**Honest comparison with the from-scratch sweep.** D2 (`w_deep=0.5` from scratch) bought
+0.022 CSI_100 but cost **−0.049** CSI_005 and was guardrail-rejected. The fine-tune buys about
**+0.014** CSI_100 on val / **+0.021** on test while costing only **−0.003 / −0.005** CSI_005 —
roughly **half the deep gain for one tenth of the accuracy cost**. That is the whole argument for
fine-tuning a good checkpoint over re-weighting a from-scratch run, and it is the first time in
this project that a deep-water change has paid for itself.

**What this does and does not license.**
- It **does** license: "a gentle `l_deep` term (`w_deep=0.10`) fine-tuned onto an already-good
  checkpoint improves deep-water CSI and peak depth error on both splits at negligible
  overall-accuracy cost."
- It **does not** license: any claim that the deep-water *underestimation bias* is fixed. The
  per-epoch fixed-panel evidence earlier in this report showed the >3 m bias getting **worse**
  under `w_deep=0.10` while the aggregate improves — the same deepest-bin defect D2 had. Report
  CSI_100/peak-depth/volume together, and state the >3 m bias caveat.
- The winner (`w_deep=0.10`) edges `w_deep=0.05` on CSI_100 val/test, but `w_deep=0.05` is better
  on peak depth error and volume error; the two are close and both beat the control. If a single
  arm is to be carried forward, `0.10` is the pre-registered pick (highest CSI_100).

**Bug A recurred on the control arm and is now fixed for the running code.** `w00` also exited
120 and was logged `FAILED ... ABORT`, but the scan had already produced the full evidence, so
no result was at risk this time. `_reclassify_arms.py` restored it (artefact-verified: 20/20
epochs) and the fix in `run_deep_finetune.py` applies to any future run.



### Extension: more fine-tuning from the winner, with a matched control (2026-09-28 16:04)

The user chose to push further. `scripts/run_deep_extend.py` resumes **both** arms from the
sAME parent — the selected winner snapshot `w01/snapshots/ep0187.pt` — and runs 20 more epochs
with identical lr/patience, so **only `w_deep` differs** and per-epoch differences are paired
again. Parent choice is deliberate: `ep0187.pt`, **not** `w01/last.pt` (ep200), because ep200 is
not the selected epoch and extending from it would reintroduce exactly the subset-argmax problem
this project just removed.

| arm | dir | parent | w_deep | purpose |
|---|---|---|---|---|
| `0.0` | `outputs/finetune_deep/w0ext` | `w01/snapshots/ep0187.pt` | 0.0 | **matched control** — separates "deep term" from "20 more epochs" |
| `0.1` | `outputs/finetune_deep/w01ext` | `w01/snapshots/ep0187.pt` | 0.1 | does the deep gain persist / grow? |

Design carried over from the first fine-tune, all verified live again at launch
(`lr reset to 1.000e-04, cosine horizon over 20 epochs (epochs 188..207, no scheduler
fast-forward)`, `resume at epoch=188`), plus the artefact-based completion check (never the exit
code). On completion it auto-runs the full-val per-epoch scan (`epoch_scan_ext.json`) and writes
`extend_selection.json`.

Two specific questions this answers that the first run could not:
1. **Does the deep gain persist or fade with more training?** (First run rose slightly,
   0.1851 → 0.1901 in the mean, so it is worth extending.)
2. **Does the >3 m bias defect get fixed, or worsen further?** The fixed-panel evidence showed
   it *worsening* at `w_deep=0.10` by ep200; a longer run is the direct test.

ETA: 16:04 + 2 × 4.4 h ≈ **00:55**, then the chained scan (~40 min) → **~01:40**.

### Post-sweep automation (`scripts/post_sweep.py`)

Encodes the chosen path so nothing has to be babysat:
1. optional `--baselines-only`: full-split CUDA evaluation of the frozen V0 checkpoint
   (`best_csi.pt`, ep155) plus bilinear/nearest on both val and test -> `outputs/post_sweep/metrics/`;
2. `--wait`: block until `SWEEP_DONE`, then select the winner;
3. selection rule is fixed in advance (not chosen after seeing numbers): an arm is eligible
   only if its val CSI_005 is within `--csi-tol` (default 0.01) of control D0, then the highest
   val **CSI_100** (deep-water score) wins, tie-broken by lower PeakDepthError; if no arm beats
   D0 on CSI_100, do **not** resume — report that the deep term did not help;
4. on a real win, resume from the frozen ep180 `last.pt` with the winner's loss config into a
   **new** output dir (`outputs/v0_deepresume`) so the frozen run stays intact.

`test` is never used for selection, only reported. Selection provenance is written to
`outputs/post_sweep/selection.json`.

### The complete unattended chain (state as of 2026-09-27 21:50)

Everything runs without a babysitter, and the GPU only ever hosts one job:

```
baseline eval (DONE, ~1 h)
└─ sweep run_deep_sweep.py --only D0 D2 --epochs 40   (~17.6 h, FINISHED 18:29)
     ├─ D0 DONE  best CSI_005 0.4977
     ├─ D2 DONE  best CSI_005 0.4487 -> REJECTED by the CSI_005 guardrail
     └─ post_sweep.py (run manually, --skip-eval --skip-resume) => WINNER = D0, no resume
          └─ run_deep_finetune.py --arms 0.0 0.05 0.10 --epochs 20 --lr 1e-4
               ├─ w_deep=0.05 DONE  (20 ep, 4.4 h) -> was MISFILED failed (Bug A), now done
               ├─ w_deep=0.10 DONE  (20 ep, 4.4 h) -> was MISFILED failed (Bug A), now done
               ├─ w_deep=0.00 RUNNING (driver PID 23680, per-epoch snapshots; ETA ~10:45)
               └─ then: full-split val eval of each best_deep_csi.pt -> selection.json
                    (epoch choice provisional: picked on the noisy 60-batch subset)
└─ scan_epochs.py --arms w005 w01 w00 --stride 1   (PID 39964, chained behind 23680)
     re-scores ALL 20 snapshots/arm on the FULL 182-tile val split (~2.4 min each)
     -> re-selects the best epoch per arm on the full basis (not max-of-subset-noise)
     -> candidate vs w_deep=0 control, same split/evaluator  => epoch_scan.json
     -> review, then write the final verdict + test split for the winner only
```

Both new guards are the mirror of the 2026-09-27 incident: that one stopped a **crash** from
being read as a **null result**; these stop a **success** from being read as a **crash**
(Bug A) and a **noise-driven epoch pick** from being read as a **deep-water gain** (Bug B).

**Change from the original chain:** the `post_sweep.py --wait --epochs 300` link (PID 35072) was
**stopped** on 2026-09-27 15:35, because with D2 expected to fail the guardrail it could only
have ended in "winner = D0, nothing to resume" — or, worst case, launched an unwanted 300-epoch
resume. It was replaced by the fine-tune chain. `post_sweep.py` is unchanged and was then run
manually to record the formal verdict.

Logs: `outputs/chain_after_pid.log`, `outputs/chain_finetune.log`,
`outputs/deep_sweep/sweep.log`, `outputs/finetune_deep/finetune.log`,
`outputs/post_sweep/post_sweep.log`.


Both wait timeouts are set to **30 h** so they exceed the ~17.6 h sweep. This matters: the
first version of the chain was armed without `--timeout-h`, and `post_sweep.py`'s default is
**12 h** — it would have given up waiting before the sweep finished, silently breaking the
handoff. Caught and re-armed.

Two guardrails worth knowing:
- If the sweep finds that the deep term does **not** beat the control on CSI_100, `post_sweep`
  deliberately does **not** resume — it reports the negative result and leaves the frozen run
  untouched. A null result is a valid outcome here, not something to train around.
- The resumed run writes to a **new** directory (`outputs/v0_deepresume`), so the frozen
  ep155/ep180 checkpoints and every number in this report stay reproducible.

### Visualisation can now follow any run

`viz/common.py` and `scripts/auto_visualize.py` read `V0_OUT_DIR` (or `V0_TAG`) to choose which
run to visualise, defaulting to `outputs/v0_10m2m_hmax` exactly as before. This lets the same
figures be produced for a sweep arm or the resumed run:
`V0_OUT_DIR=outputs/deep_sweep/D2 python scripts/auto_visualize.py`. The daemon exports the
variable so its child scripts inherit it.

**Gap found and fixed (2026-09-27).** Enabling `V0_OUT_DIR` was not sufficient on its own: the
daemon's liveness check (`trainer_alive()`) reads `<out_dir>/train.pid`, and
`run_deep_sweep.py` never writes that file for an arm, so the daemon would refresh once and then
mistake "no PID file" for "training finished", run a final pass and exit. That is why the D2 arm
had no figures for its first 7 epochs even though a trainer was running. `trainer_alive()` now
falls back to discovering a live `train_fixed.py` process whose command line names this
`--out_dir` (matching either the absolute or the workspace-relative spelling, since the sweep
uses the latter), then self-heals by writing `train.pid` so later cycles stay cheap. Verified
against the live D2 trainer (discovered pid 25792).

Running now: `auto_visualize.py` on D2 (900 s cycle, full scan every 3rd) and a one-shot
`--final-only` exhaustive pass on D0; both are CPU-only and do not contend with the GPU trainer.


## Commands run and outcomes

| Command | Outcome |
|---------|---------|
| `python tests/test_*.py` | All OK after Sobel-mask fix |
| `python scripts/evaluate.py ... --model bilinear --split val --max_batches 20` | CSI_005≈0.300, RMSE_wet≈0.704 → `outputs/benchmarks/bilinear_val_partial.json` |
| `python scripts/evaluate.py ... --model nearest --max_batches 20` | CSI_005≈0.284 → `outputs/benchmarks/nearest_val_partial.json` |
| `python scripts/train_fixed.py --config configs/v0_10m2m_hmax_smoke.yaml` | 3 real steps, loss≈0.15, wrote `outputs/smoke/{last,best_csi,best_rmse_wet}.pt` |
| `python scripts/train_multiscale.py --max_steps 2` | OK (batch_size=1); `outputs/v1_multiscale/last_multiscale.pt` |
| `python scripts/train_residual_ldm.py --smoke` | Module shapes OK |

## Live V0 training (detached — survives agent exit)

| Field | Value |
|-------|-------|
| Status | **STOPPED at epoch 180** (user decision; plateaued — see below). GPU reassigned to the deep-water loss sweep |
| PID | none (stopped; was `24696` attempt2) |
| Manager | `scripts/run_v0_pipeline.ps1` PID `37568` **stopped** (so it could not auto-restart training) |
| Device | **GPU** — GeForce GTX 950M 4 GB, `torch 2.7.1+cu118`, FP32 (`amp: false`) |
| Config | `configs/v0_10m2m_hmax.yaml` (unchanged; the sweep overrides via CLI) |
| Progress | **epoch 180 / 300** — resumed to 180, then stopped deliberately |
| Last metrics (ep180) | CSI_005=0.5361, RMSE_wet=0.5099, MAE_wet=0.2672, loss=0.0579 |
| Final checkpoint | CSI_005=**0.5380** @ep155 → `best_csi.pt`; RMSE_wet=**0.5073** @ep157 → `best_rmse_wet.pt` |
| Backups | `outputs/v0_10m2m_hmax/checkpoints_backup/{best_csi_ep155_*,best_rmse_wet_*,best_summary_*,history_*}` |
| Snapshots | `snapshots/ep{0080..0170}.pt` (stride 10) + genuine-improvement `best_ep*.pt` only |
| Live progress | Prefer `outputs/v0_10m2m_hmax/history.jsonl` + `best_summary.json` |

### Trainer death at ep149 (2026-09-26 16:49)

The original trainer PID `33744` vanished after epoch 149 with **no traceback** — GPU memory
was 3850/4096 MiB immediately before, so a silent device-side abort (OOM or driver reset) is
the most likely cause. `run_v0_pipeline.ps1` detected it, logged
`Training incomplete: last_epoch=149/300. Retrying ONCE`, and restarted from `last.pt` as
attempt2 (PID `24696`). No epochs were lost; `history.jsonl` is continuous across the seam
(no duplicate/skipped epoch numbers). The pipeline retried only **once**, so a second death
would have ended training; that risk is now moot because V0 training was stopped deliberately
at ep180. The GPU was at ~3.6 GB of 4 GB throughout, which is the constraint that makes every
sweep arm a sequential, one-at-a-time job.

### Resume monitoring

```powershell
# Alive?
Get-Content outputs\v0_10m2m_hmax\train.pid | ForEach-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }
# Latest epoch / metrics
Get-Content outputs\v0_10m2m_hmax\best_summary.json
Get-Content outputs\v0_10m2m_hmax\history.jsonl -Tail 3
# After 300 epochs (or early-stop): val+test + baselines
python scripts/evaluate.py --config configs/v0_10m2m_hmax.yaml --model hydrogeo_srno --ckpt outputs/v0_10m2m_hmax/best_csi.pt --split val
python scripts/evaluate.py --config configs/v0_10m2m_hmax.yaml --model hydrogeo_srno --ckpt outputs/v0_10m2m_hmax/best_csi.pt --split test
python scripts/evaluate.py --config configs/v0_10m2m_hmax.yaml --model bilinear --split val
python scripts/evaluate.py --config configs/v0_10m2m_hmax.yaml --model nearest --split val
```

Do **not** start a second train while another trainer is alive (check `outputs/*/train.pid`
and `outputs/deep_sweep/*/`). Do **not** kill unrelated python jobs.

## Detached one-shot pipeline (STOPPED — superseded)

`scripts/run_v0_pipeline.ps1` runs the whole chain unattended: finish training → evaluate
val+test for the trained checkpoint and bilinear/nearest baselines → append a summary to
`outputs/v0_10m2m_hmax/pipeline_status.txt`. It **adopts** an already-running trainer instead
of starting a second one, retries training once from the latest checkpoint if it dies early,
and always proceeds to eval so partial results are recorded.

**Status: stopped 2026-09-26** (PID `37568` killed) so it could not auto-restart V0 training
after the deliberate stop at ep180. It never reached stage 2/3, so `pipeline_status.txt` was
never written. Its evaluation role has been taken over by `scripts/post_sweep.py`, which
already produced the full-split baseline table above.

| Field | Value |
|-------|-------|
| Pipeline PID | `37568` — **stopped** |
| Pipeline log | `outputs/v0_10m2m_hmax/pipeline.log` |
| Summary file | `outputs/v0_10m2m_hmax/pipeline_status.txt` (never written) |
| Replacement | `scripts/post_sweep.py` → `outputs/post_sweep/` |

### How to check the replacement later

```powershell
# baseline / sweep evaluations
Get-Content outputs\post_sweep\post_sweep.log -Tail 30
Get-ChildItem outputs\post_sweep\metrics\*.json
# sweep progress and per-arm status
Get-Content outputs\deep_sweep\sweep.log -Tail 20
Get-ChildItem outputs\deep_sweep\*\status.json
```

### Still NOT done

- **Deep-water >3 m *bias* is still not fixed.** The fine-tune improves CSI@1.00 m, peak depth
  error and volume error on both splits, but the fixed-panel evidence shows the >3 m bias getting
  slightly **worse** at `w_deep=0.10` while the aggregate improves — the same deepest-bin defect
  the from-scratch D2 had. Any paper claim must pair the gain with this caveat.
- The winner (`w_deep=0.10` @ep187) has **not** been carried forward into a longer run. It is a
  20-epoch fine-tune of the frozen ep180 checkpoint, evaluated; it has not been resumed further.
- **V0 training stopped at epoch 180/300** by decision, because the model had plateaued since
  ~ep98 (identical-tile evidence above) and the deep-water bias was worsening. The remaining
  epochs were not expected to change any reported number. `last.pt` (ep180) is intact and
  training can be resumed at any time with
  `python scripts/train_fixed.py --config configs/v0_10m2m_hmax.yaml`.
- The **deep-water loss sweep is running now instead** (`scripts/run_deep_sweep.py --only D0 D2`).
- The full-split evaluation **is** done (`outputs/post_sweep/metrics/`); what is still missing is
  the evaluation of the *winner* of the loss sweep, which requires the sweep to finish.
- Ablations (A0–A7) / other baselines (EDSR, RCAN, ResUNet, Swin) not evaluated.
- Residual-LDM, temporal/dynamic-hydro, cross-city transfer remain deferred.

## Blocked / remaining

| Item | Blocker |
|------|---------|
| Full V0 300-epoch train stopped at ep180 (plateaued) | Deliberate; `last.pt` resumable. Frozen run untouched |
| Deep-water (>3 m) underestimation bias not improving | Root-caused to 2 loss defects; `l_deep` fix. **From scratch (D2)**: CSI_100 +0.022 but CSI_005 −0.049 -> rejected. **Fine-tune (`w_deep=0.10`)**: CSI_100 +0.008 val / +0.021 test for CSI_005 −0.003/−0.005 -> **KEPT**. Deep *aggregates* improve on both splits; the >3 m *bias* specifically still degrades (open) |
| Full-split baseline evaluation | **DONE** 2026-09-27 (`outputs/post_sweep/metrics/`) |
| Pipeline retries training only once | attempt2 was consumed at the ep149 death; pipeline now stopped |
| Deep-term sweep result & resumed run | **DONE**: D0 best CSI_005 0.4977; D2 **rejected** (0.4487 < 0.4877 floor). `post_sweep.py` returned **D0, no resume** — as predicted. D2 still shows the deep gain: CSI_100 0.1487→**0.1707**, peak depth error 2.66→**1.88 m** |
| Deep-water fine-tune of frozen V0 | **DONE** (2026-09-28): arms `0.05` / `0.10` / control `0.0` each 20 epochs. Winner **`w_deep=0.10` @ep187** |
| Fine-tune control arm + final comparison | **DONE**: full-val paired scan (n=20 epochs/arm) -> `outputs/finetune_deep/selection.json`. CSI_100 vs control **+0.008** at `0.10` (**t=+5.7**, significant) with CSI_005 unchanged; **test replicates** (0.2547 -> 0.2754). See "FINAL fine-tune result" |
| **Bug A: benign exit=120 misfiled as a crash** | **FIXED**. Completion is now proved from the trainer's artefacts (`trainer_reached_end()`), not the exit code; `w005`/`w01`/`w00` reclassified `done` (artefact-verified). Recurred on the control arm but no result was at risk |
| **Bug B: deep gain picked on a noisy 60-batch subset** | **RESOLVED**: `scripts/scan_epochs.py` re-scored all 60 snapshots on the full 182-tile val split. The subset `argmax` was unreliable in *both* directions; the paired full-split comparison is the decision-grade basis and it is **positive** (+0.006/+0.008 CSI_100, t>4.8) |
| Full-domain bilinear/HydroGeo benchmark (all 182 val / 242 test tiles) | Time; partial (20 batches) benchmarks only |
| True Swin Residual SwinUNet (B4) | Lite ResUNet stand-in shipped; full Swin needs dependency work |
| Residual-LDM training (P11) | Explicitly deferred until V0 deterministic backbone is trained |
| Dynamic h/hux/hvy + temporal (P10) | Scaffold only (`in_channels=4` path exists; no temporal block) |
| Cross-city transfer (P12) | No second-city data in workspace |
| 5 m / 20 m / 30 m train-norm stats | Only `static_2m_*` and `static_10m_*` in `norm_stats_train.json`; normalizer falls back |

## How to run the main experiment

```bash
cd E:\Projects\20260921-JiaChenchen-Paper

# Benchmark
python scripts/evaluate.py --config configs/v0_10m2m_hmax.yaml --model bilinear --split val

# Smoke (proven)
python scripts/train_fixed.py --config configs/v0_10m2m_hmax_smoke.yaml

# Full V0 (GPU recommended)
python scripts/train_fixed.py --config configs/v0_10m2m_hmax.yaml

# Eval checkpoint
python scripts/evaluate.py --config configs/v0_10m2m_hmax.yaml --model hydrogeo_srno --ckpt outputs/v0_10m2m_hmax/best_csi.pt --split test
```

Raw data (`wellington-output-data/`, `static_geo_data/`) untouched. Patch index not reshuffled. No git commit/push.
