"""Summarise the paired deep-water bias evolution for each fine-tune arm.

bias_evolution.json is produced by the visualisation pipeline on a FIXED panel of val tiles
for every checkpoint, so unlike the per-epoch 60-batch subset CSI_100 it is a paired
measurement: the epoch-to-epoch change is attributable to the weights, not to which tiles or
how many deep pixels happened to be sampled. That makes it the cleanest per-epoch evidence
available before the full-split scan finishes.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIN = ROOT / "outputs" / "finetune_deep"

for arm in ("w005", "w01", "w00"):
    p = FIN / arm / "visualizations" / "bias_evolution.json"
    if not p.exists():
        print(f"{arm}: no bias_evolution.json")
        continue
    j = json.loads(p.read_text(encoding="utf-8"))
    curves = j["curves"]
    print(f"\n=== {arm}  (fixed {len(j['tiles'])}-tile val panel, paired across epochs) ===")
    print("  ep  | bias 1-1.5 | bias >3m | rmse >3m | mean bias wet")
    for c in curves:
        print(f"  {c['epoch']} | {c['bias_1_1.5']:+.4f}    | {c['bias_gt3']:+.4f} | "
              f"{c['rmse_gt3']:.4f}   | {c['mean_bias_wet']:+.5f}")
    if len(curves) >= 2:
        a, b = curves[0], curves[-1]
        print(f"  Δ over ep{a['epoch']}->ep{b['epoch']}: "
              f"bias_1-1.5 {b['bias_1_1.5']-a['bias_1_1.5']:+.4f}  "
              f"bias_>3 {b['bias_gt3']-a['bias_gt3']:+.4f}  "
              f"rmse_>3 {b['rmse_gt3']-a['rmse_gt3']:+.4f}  "
              f"mean_bias_wet {b['mean_bias_wet']-a['mean_bias_wet']:+.5f}")
