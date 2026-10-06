# Visualization guide

Everything here is **CPU-only** and reads files the trainer already writes, so it can run
while GPU training continues (PID in `outputs/v0_10m2m_hmax/train.pid`) without disturbing it.

Open the live gallery at:

```
outputs/v0_10m2m_hmax/visualizations/index.html
```

## What gets visualized

| Stage | Figure | Source | Script |
|-------|--------|--------|--------|
| **Training process** | `01_training_dashboard.png` — loss+LR, CSI/F1 at 0.05/0.30/1.00 m, best-so-far envelopes, depth errors, mass/peak errors, epoch time, run-state panel | `history.jsonl` | `viz/plot_training.py` |
| **Stage/milestone progress** | `01_progress_overview.png` — CSI saturation, wet-depth error, extremes, pending-epoch shading | `history.jsonl` | `viz/plot_training.py` |
| **Domain & splits** | `02_domain_splits.png` — patch grid by split, usable tiles, wet fraction per return period | `index/patches_480m.json` | `viz/plot_domain.py` |
| **Error geography** | `02_error_geography.png` — per-tile CSI/MAE maps over the study area | per-tile tables | `viz/plot_domain.py` |
| **Physical diagnostics** | `03_diagnostics_val.png` — bias vs depth, error vs slope, error/bias by land use, per-tile model-vs-bilinear, mass check, residual anchoring | per-pixel stats | `scripts/visualize_diagnostics.py` |
| **Intermediate reconstruction** | `epNNN/tiles_<split>/tile_*.png` — coarse → bilinear → model → truth, `\|error\|`, wet-extent TP/FP/FN, learned residual Δz | checkpoint | `scripts/visualize_tiles.py` |
| **Milestone sample set** | `epNNN/tiles_<split>_overview.png` — representative tiles across the inundation-coverage spectrum | checkpoint | `scripts/visualize_tiles.py` |
| **Model vs baselines** | `epNNN/metrics_vs_baseline_<split>.png` — B0/B1/model bars, per-tile CSI histogram, CSI vs coverage | checkpoint + `benchmarks/` | `scripts/visualize_tiles.py` |
| **Image-space evolution** | `evolution/<tile>/evolution_predictions.png` — prediction/error/agreement at each checkpoint epoch | `snapshots/*.pt` | `scripts/visualize_evolution.py` |
| **Bias evolution (metric-space)** | `04_bias_evolution.png` — depth-binned bias and RMSE at each checkpoint, deep-water bias trend, extreme-depth RMSE vs sample size | `snapshots/*.pt` | `scripts/visualize_bias_evolution.py` |
| **Tabular** | `epNNN/per_tile_metrics_<split>.csv/.json`, `summary_<split>.json`, `diagnostics_<split>.json`, `bias_evolution.json` | checkpoint | both |

## Commands

```powershell
# one full refresh (curves + tiles + evolution + gallery)
python scripts/visualize_all.py

# same, but scan every tile of val and test instead of a 60-tile subset
python scripts/visualize_all.py --full --splits val test --n-representative 6

# just the training curves (seconds)
python viz\plot_training.py

# tiles at a specific checkpoint
python scripts\visualize_tiles.py --ckpt outputs\v0_10m2m_hmax\last.pt --splits val test

# cheap periodic mode: representative tiles only, no full-split scan (~30 s)
# NOTE: this records only 4 representative tiles and will NOT downgrade an existing
# full-scan average in summary_*.json (see "Metric provenance" below)
python scripts\visualize_tiles.py --representatives-only --splits val

# recompute summary_*.json averages from the per-tile tables (idempotent)
python scripts\repair_summaries.py --dry-run
python scripts\repair_summaries.py

# physical diagnostics
python scripts\visualize_diagnostics.py --split val --ckpt outputs\v0_10m2m_hmax\best_csi.pt

# domain / split layout
python viz\plot_domain.py

# image-space evolution across snapshots
python scripts\visualize_evolution.py --split val

# bias-vs-depth curves across checkpoints (answers: is deep-water bias shrinking?)
python scripts\visualize_bias_evolution.py --split val --tiles 40

# exhaustive single-shot refresh (every tile, both splits, all diagnostics)
python scripts\auto_visualize.py --final-only

# rebuild the HTML gallery only
python scripts\build_gallery.py
```

## Visualising a different run

Every script here reads its run directory from `V0_OUT_DIR` (or `V0_TAG`), defaulting to
`outputs/v0_10m2m_hmax`. Use it to visualise a sweep arm or a resumed run:

```powershell
# sweep arm D2
$env:V0_TAG = "deep_sweep/D2"
python scripts\auto_visualize.py --interval 1500 --threads 3
Remove-Item Env:\V0_TAG

# or point at any run directory
$env:V0_OUT_DIR = "outputs/v0_deepresume"
python scripts\visualize_all.py
Remove-Item Env:\V0_OUT_DIR
```

`auto_visualize.py` exports the variable itself, so its child scripts (`visualize_tiles.py`,
`plot_domain.py`, `build_gallery.py`, ...) follow the same run automatically.

## Metric provenance (`avg_basis`)

`summary_{split}.json` carries an `avg_basis` field saying whether `avg_metrics` came from a
**full scan** (`full_scan_N_tiles`) or from only the **hand-picked representatives**
(`representative_only_N_tiles`). Always check it before quoting a number.

This was added after a real mis-reporting bug: because `auto_visualize.py` runs
`--representatives-only` every cycle, the 4-tile average repeatedly overwrote the real 60-tile
average, inflating val CSI_005 by roughly **+0.05** (ep098 read 0.5682 vs a true 0.5157).
A `--representatives-only` run now refuses to downgrade an existing full-scan average, and
`scripts/repair_summaries.py` can rebuild any summaries already affected.

## Continuous background jobs (survive agent exit)

| Job | PID file | Purpose |
|-----|----------|---------|
| `scripts/watch_checkpoints.py` | `snapshots/watcher.pid` | copies `last.pt` to `snapshots/epNNNN.pt` every 10 epochs, and `best_csi.pt` to `snapshots/best_epNNNN.pt` **only when the best actually improves** (state kept in `snapshots/.watcher_state.json`). `--prune` deletes redundant `best_ep*.pt` copies and exits |
| `scripts/auto_visualize.py` | `visualizations/auto_visualize.pid` | every 25 min: refresh curves + representative tiles + gallery; every 4th cycle also the full-split scan, diagnostics, evolution and bias-evolution |

On trainer exit, `auto_visualize.py` automatically runs an **exhaustive final pass**: every
tile of both `val` and `test`, both diagnostics figures, evolution, bias evolution and the
gallery — then exits. `--final-only` runs that pass immediately without waiting.

Logs: `snapshots/watcher.log`, `visualizations/auto_visualize.log`, `visualizations/refresh.log`.

Start them again after a reboot:

```powershell
$py = "E:\Miniconda3\python.exe"
# optional: drop redundant best-snapshot copies first
& $py scripts/watch_checkpoints.py --prune
Start-Process $py -ArgumentList '-u','scripts/watch_checkpoints.py','--stride','10','--interval','45' -WorkingDirectory $PWD -WindowStyle Hidden
Start-Process $py -ArgumentList '-u','scripts/auto_visualize.py','--interval','1500','--threads','3' -WorkingDirectory $PWD -WindowStyle Hidden
```

## Reading the residual panel

`Δz` is the model's correction in **log1p depth space** (`z = log1p(h / 0.1)`), added on top of
the bilinear base. Positive = the model decided the coarse simulation under-resolved a
channel/depression; negative = it removed water the coarse run had spread too thinly.
The "residual anchoring" panel in `03_diagnostics` plots the mean learned Δz against the mean
true coarse-vs-truth log-depth mismatch per tile: points near the diagonal mean the operator
is recovering the signal that interpolation loses.

## Reading `04_bias_evolution.png`

The deep-water depth bins hold relatively few pixels, so the per-checkpoint trace is noisy.
Read the **fitted slope** (dashed line), not the first-to-last comparison. The script prints an
explicit verdict per band:

| Verdict | Meaning |
|---------|---------|
| `TRENDING UP` / `TRENDING DOWN` | fitted drift over the checkpoint range exceeds the residual scatter — a real trend |
| `FLAT vs NOISE` | drift is smaller than the scatter — do not interpret as progress |

Example at epochs 80/90/100 on 30 fixed val tiles:

| Band | Drift | Scatter | Verdict |
|------|-------|---------|---------|
| bias > 3 m | +0.071 m | 0.318 m | FLAT vs NOISE (trace −1.94 → −2.45 → −1.87) |
| bias 1.0–1.5 m | +0.057 m | 0.011 m | TRENDING UP |
| mean bias, all wet | +0.002 m | 0.001 m | FLAT vs NOISE |

## Do not spawn agents to "drive training forward"

Training is a long GPU job that runs **detached** (`Start-Process` + PID file). An agent that
tries to sit in the training loop synchronously will be killed by turn timeouts and report
`no progress`, even though training is perfectly healthy. To check progress, read
`history.jsonl` / `best_summary.json`; to influence training, edit configs and let the
pipeline's single retry-from-checkpoint path pick them up. Never start a second trainer while
`outputs/v0_10m2m_hmax/train.pid` is alive.

## Notes

- Nothing here writes to `dataset/`, `wellington-output-data/` or `static_geo_data/`.
- Figures are PNG (Agg backend); pass `--also-pdf` is available on `viz.common.save` if vector
  output is ever needed.
- `viz/_probe.py` is a scratch script used to measure CPU inference cost; safe to delete.
