#!/usr/bin/env python
"""Measure FloodLoss component magnitudes and audit the extreme-term semantics.

Background: `extreme_quantile=0.95` is applied to target over ALL valid cells, but ~86% of
valid cells are dry, so the quantile lands near 0.3-0.5 m rather than in the deep-water tail.
This script quantifies (a) each loss term's magnitude and weighted share, and (b) how much of
the true deep water each extreme-term definition actually covers.

CPU-only on purpose: safe to run while the GPU is busy or idle.

Usage:
    python scripts/diagnose_loss.py --split val --batches 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.wellington_fixed_sr import WellingtonFixedSRDataset, collate_fixed
from engine.checkpoint import load_checkpoint
from engine.evaluator import build_model
from losses.flood_loss import FloodLoss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0_10m2m_hmax.yaml")
    ap.add_argument("--ckpt", default="outputs/v0_10m2m_hmax/best_csi.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batches", type=int, default=20)
    args = ap.parse_args()

    torch.set_num_threads(4)
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    lc = cfg.get("loss", {})
    dcfg = cfg["dataset"]

    ds = WellingtonFixedSRDataset(
        root=ROOT / dcfg.get("root", "dataset"),
        split=args.split,
        lr_res=int(dcfg.get("lr_res", 10)),
        hr_res=int(dcfg.get("hr_res", 2)),
        target=dcfg.get("target", "h_max"),
        geo_mode=dcfg.get("geo_mode", "all"),
        scenarios=tuple(dcfg.get("scenarios", ["20a", "100a"])),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fixed)

    model = build_model(cfg["model"]["name"], cfg)
    ck = load_checkpoint(str(ROOT / args.ckpt), model, map_location="cpu")
    model.eval()
    print(f"checkpoint epoch={ck.get('epoch')} recorded={ {k: round(float(v),4) for k,v in (ck.get('metrics') or {}).items() if isinstance(v,(int,float))} }")

    weights = {k: float(lc.get(k, d)) for k, d in
               (("w_depth", 1.0), ("w_wet", 0.30), ("w_log", 0.20),
                ("w_boundary", 0.10), ("w_extreme", 0.10))}
    crit = FloodLoss(
        w_depth=weights["w_depth"], w_wet=weights["w_wet"], w_log=weights["w_log"],
        w_boundary=weights["w_boundary"], w_extreme=weights["w_extreme"],
        wet_threshold=float(lc.get("wet_threshold", 0.05)),
        depth_ref=float(cfg["model"].get("depth_ref", 0.10)),
        extreme_quantile=float(lc.get("extreme_quantile", 0.95)),
    )
    print(f"weights={weights}  wet_threshold={crit.wet_threshold}  "
          f"depth_ref={crit.depth_ref}  extreme_quantile={crit.extreme_quantile}")

    comps: dict[str, list[float]] = {}
    cov_all_of_wet: list[float] = []   # how much of wet cells 'extreme' (all-cells quantile) covers
    cov_true_deep: list[float] = []    # how much of true>1m is covered by the all-cells quantile

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.batches:
                break
            out = model(lr=batch["lr"], lr_valid=batch["lr_valid"],
                        static_cont=batch["static_cont"], landuse=batch["landuse"],
                        lr_res=batch["lr_res"], hr_res=batch["hr_res"])
            d = crit(output=out, target=batch["hr"], mask=batch["mask"])
            for k, v in d.items():
                comps.setdefault(k, []).append(float(v))

            tgt = batch["hr"][:, 0]
            msk = batch["mask"].bool()
            if msk.ndim == 4:
                msk = msk[:, 0]
            v = tgt[msk]
            thr = torch.quantile(v, crit.extreme_quantile) if v.numel() else torch.tensor(1.0)
            wet = v > crit.wet_threshold
            cov_all_of_wet.append(float((v[wet] >= thr).float().mean()) if wet.any() else float("nan"))
            deep = v >= 1.0
            cov_true_deep.append(float((v[deep] >= thr).float().mean()) if deep.any() else float("nan"))

    print(f"\n=== component magnitudes over {args.batches} {args.split} tiles ===")
    print(f"  {'term':<14} {'mean':>10} {'weight':>8} {'weighted':>10} {'share':>7}")
    totals = {}
    for k, wkey in (("l_depth", "w_depth"), ("l_wet", "w_wet"), ("l_log", "w_log"),
                    ("l_boundary", "w_boundary"), ("l_extreme", "w_extreme")):
        if k not in comps:
            continue
        m = float(np.mean(comps[k]))
        w = weights[wkey]
        totals[k] = m * w
    s = sum(totals.values()) or 1.0
    for k in ("l_depth", "l_wet", "l_log", "l_boundary", "l_extreme"):
        if k not in totals:
            continue
        m = float(np.mean(comps[k]))
        print(f"  {k:<14} {m:>10.5f} {weights[{'l_depth':'w_depth','l_wet':'w_wet','l_log':'w_log','l_boundary':'w_boundary','l_extreme':'w_extreme'}[k]]:>8.2f} "
              f"{totals[k]:>10.5f} {totals[k]/s*100:>6.1f}%")
    print(f"  {'TOTAL':<14} {'':>10} {'':>8} {s:>10.5f}  100.0%")

    print(f"\n=== extreme-term coverage (constraint from all-cells quantile) ===")
    print(f"  mean coverage of WET cells (h>{crit.wet_threshold}): {np.nanmean(cov_all_of_wet)*100:5.1f}%")
    print(f"  mean coverage of TRUE >1 m cells             : {np.nanmean(cov_true_deep)*100:5.1f}%")


if __name__ == "__main__":
    main()
