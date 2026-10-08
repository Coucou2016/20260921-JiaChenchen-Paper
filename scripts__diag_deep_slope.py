#!/usr/bin/env python
"""Deep-water mechanism diagnostic for the Experiment A/B arms.

The prior diagnostic (``scripts/_diagnose_error_scale.py``) established two facts the
report leans on: over deep pixels the frozen V0's regression slope of prediction on the
10 m input is 0.39, whereas the bilinear base itself is 0.72 (the net SHRINKS deep water);
deep pixels need ~+1.059 m and the frozen net delivers +0.027 m. Experiment A changes the
output parameterisation (explicit metre-space residual on the bilinear/nearest base), so the
right question is whether that slope is restored.

This script computes, on the FULL validation split and for each model, the domain-pooled:
  * regression slope of prediction on the coarse (10 m) input over deep pixels (truth > 1 m),
  * the mean needed lift vs delivered lift on those pixels,
  * signed deep bias,
so the Experiment A win/loss can be attributed to a mechanism, not just an aggregate metric.

Reads only the tile npz + saved checkpoints; no training. Each model is evaluated once.

Usage
-----
    python -u scripts/diag_deep_slope.py --split val
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.wellington_fixed_sr import WellingtonFixedSRDataset, collate_fixed
from engine.checkpoint import load_checkpoint
from engine.evaluator import build_model
from engine.trainer import move_to_device

CONFIG_V0 = "configs/v0_10m2m_hmax.yaml"
CONFIG_V1 = "configs/v1_bilinear_residual.yaml"

MODELS = [
    ("bilinear", CONFIG_V0, "bilinear", None),
    ("frozen_ep180", CONFIG_V0, "hydrogeo_srno", "outputs/v0_10m2m_hmax/last.pt"),
    ("w01_ep187", CONFIG_V0, "hydrogeo_srno",
     "outputs/finetune_deep/w01/snapshots/ep0187.pt"),
    ("residual_depth", CONFIG_V1, "bilinear_residual",
     "outputs/v1_bilinear_residual/depth/best_csi.pt"),
    ("E7_continuous", CONFIG_V0, "hydrogeo_srno",
     "outputs/deep_objective/E7/best_pooled_rmse.pt"),
]


@torch.no_grad()
def collect(model, loader, device) -> dict:
    """Accumulate deep-pixel slope/bias sufficient statistics over the whole split."""
    sc = {"n_deep": 0, "sum_x": 0.0, "sum_y": 0.0, "sum_xx": 0.0, "sum_xy": 0.0,
          "sum_need": 0.0, "sum_lift": 0.0, "sum_bias": 0.0}
    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(lr=batch["lr"], lr_valid=batch["lr_valid"],
                    static_cont=batch["static_cont"], landuse=batch["landuse"],
                    lr_res=batch["lr_res"], hr_res=batch["hr_res"])
        pred = out["depth"][:, 0].detach().cpu().numpy()
        truth = batch["hr"][:, 0].detach().cpu().numpy()
        mask = batch["mask"][:, 0].detach().cpu().numpy() > 0
        # the coarse input up-sampled to HR (10 m -> 2 m nearest, the diagnostic's 'x')
        lr = batch["lr"][:, 0].detach().cpu().numpy()
        for i in range(pred.shape[0]):
            v = mask[i]
            deep = v & (truth[i] > 1.0)
            if not deep.any():
                continue
            # nearest-upsample the 10 m field to the 2 m tile grid
            sc0 = truth.shape[-1] // lr.shape[-1]
            lr_up = np.repeat(np.repeat(lr[i], sc0, 0), sc0, 1)
            lr_up = lr_up[:truth.shape[-2], :truth.shape[-1]]
            x = lr_up[deep].astype(np.float64)
            y = pred[i][deep].astype(np.float64)
            t = truth[i][deep].astype(np.float64)
            nt = x.size
            sc["n_deep"] += nt
            sc["sum_x"] += x.sum(); sc["sum_y"] += y.sum()
            sc["sum_xx"] += (x * x).sum(); sc["sum_xy"] += (x * y).sum()
            sc["sum_need"] += (t - x).sum()      # needed lift relative to the coarse input
            sc["sum_lift"] += (y - x).sum()      # delivered lift
            sc["sum_bias"] += (y - t).sum()
    n = max(sc["n_deep"], 1)
    sx, sy = sc["sum_x"], sc["sum_y"]
    denom = sc["n_deep"] * sc["sum_xx"] - sx * sx
    slope = (sc["n_deep"] * sc["sum_xy"] - sx * sy) / denom if abs(denom) > 1e-9 else float("nan")
    return {
        "n_deep_pixels": int(sc["n_deep"]),
        "slope_pred_on_10m": float(slope),
        "mean_needed_lift_m": float(sc["sum_need"] / n),
        "mean_delivered_lift_m": float(sc["sum_lift"] / n),
        "delivered_over_needed": float(sc["sum_lift"] / sc["sum_need"]) if abs(sc["sum_need"]) > 1e-9 else float("nan"),
        "deep_mean_bias_m": float(sc["sum_bias"] / n),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="use cpu to avoid contending with a running GPU training job")
    ap.add_argument("--out", default="outputs/premodel/exp_ab_deep_slope.json")
    args = ap.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device={device}")
    ds_cfg = yaml.safe_load((ROOT / CONFIG_V0).read_text(encoding="utf-8"))["dataset"]
    ds = WellingtonFixedSRDataset(
        root=ROOT / ds_cfg.get("root", "dataset"), split=args.split,
        lr_res=int(ds_cfg.get("lr_res", 10)), hr_res=int(ds_cfg.get("hr_res", 2)),
        target=ds_cfg.get("target", "h_max"), geo_mode=ds_cfg.get("geo_mode", "all"),
        scenarios=tuple(ds_cfg.get("scenarios", ["20a", "100a"])),
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fixed)

    out: dict = {"split": args.split, "models": {}}
    for label, cfg, model, ckpt in MODELS:
        if ckpt is not None and not (ROOT / ckpt).exists():
            print(f"[skip] {label}: {ckpt} missing")
            continue
        c = yaml.safe_load((ROOT / cfg).read_text(encoding="utf-8"))
        m = build_model(model, c)
        if ckpt:
            load_checkpoint(str(ROOT / ckpt), m, map_location="cpu")
        m = m.to(device).eval()
        res = collect(m, loader, device)
        out["models"][label] = res
        print(f"{label:>16}: slope={res['slope_pred_on_10m']:.3f} "
              f"need={res['mean_needed_lift_m']:+.3f} delivered={res['mean_delivered_lift_m']:+.3f} "
              f"bias={res['deep_mean_bias_m']:+.3f} n={res['n_deep_pixels']}")

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
