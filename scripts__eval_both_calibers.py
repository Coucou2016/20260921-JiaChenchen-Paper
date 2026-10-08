#!/usr/bin/env python
"""Evaluate ANY model on val/test and report BOTH aggregation calibers.

The report is explicit that tile-macro and domain-pooled metrics are different
quantities and must never be mixed silently (report §5.9).  ``evaluate.py`` only
returns the tile-macro caliber, so this script runs one full pass and emits both:

* tile-macro keys (``RMSE_wet``, ``CSI_005`` ...) via ``average_metrics``, identical
  to ``evaluate.py``;
* domain-pooled keys (``RMSE_wet_domain``, ``Bias_1m_domain`` ...) via
  ``FloodMetricAccumulator``.

Because the residual arms (Experiment A) and the deep-objective arms are wrapped in
different model classes, the model is built through the same ``build_model`` factory
the rest of the harness uses.  This is the single tool used to produce every number in
the Experiment A / B comparison tables, so calibers cannot drift between models.

Usage
-----
    python -u scripts/eval_both_calibers.py \
        --config configs/v1_bilinear_residual.yaml --model bilinear_residual \
        --ckpt outputs/v1_bilinear_residual/depth/last.pt --split val \
        --out outputs/v1_bilinear_residual/depth/eval_val.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.wellington_fixed_sr import WellingtonFixedSRDataset, collate_fixed
from engine.checkpoint import load_checkpoint
from engine.evaluator import build_model
from engine.trainer import move_to_device
from metrics.aggregation import FloodMetricAccumulator, average_metrics
from metrics.flood_metrics import compute_flood_metrics


@torch.no_grad()
def evaluate_both(model, loader, device) -> dict:
    model.eval()
    tiles: list[dict] = []
    acc = FloodMetricAccumulator()
    t0 = time.time()
    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(lr=batch["lr"], lr_valid=batch["lr_valid"],
                    static_cont=batch["static_cont"], landuse=batch["landuse"],
                    lr_res=batch["lr_res"], hr_res=batch["hr_res"])
        pred, tgt, mask = out["depth"], batch["hr"], batch["mask"]
        tiles.append(compute_flood_metrics(pred, tgt, mask))
        acc.update(pred, tgt, mask)
    merged = dict(average_metrics(tiles))
    merged.update(acc.compute())
    merged["_n_tiles"] = len(tiles)
    merged["_sec"] = time.time() - t0
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_cfg = cfg["dataset"]
    root = ROOT / ds_cfg.get("root", "dataset")
    ds = WellingtonFixedSRDataset(
        root=root, split=args.split,
        lr_res=int(ds_cfg.get("lr_res", 10)), hr_res=int(ds_cfg.get("hr_res", 2)),
        target=ds_cfg.get("target", "h_max"), geo_mode=ds_cfg.get("geo_mode", "all"),
        scenarios=tuple(ds_cfg.get("scenarios", ["20a", "100a"])),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fixed)
    model = build_model(args.model, cfg).to(device)
    if args.ckpt:
        load_checkpoint(args.ckpt, model, map_location=device)

    metrics = evaluate_both(model, loader, device) if args.max_batches is None else _capped(
        model, loader, device, args.max_batches)
    metrics["_model"] = args.model
    metrics["_ckpt"] = args.ckpt
    metrics["_split"] = args.split
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if not k.startswith("_")}, indent=2))
    print(f"wrote {out}")


@torch.no_grad()
def _capped(model, loader, device, max_batches: int) -> dict:
    model.eval()
    tiles: list[dict] = []
    acc = FloodMetricAccumulator()
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_to_device(batch, device)
        out = model(lr=batch["lr"], lr_valid=batch["lr_valid"],
                    static_cont=batch["static_cont"], landuse=batch["landuse"],
                    lr_res=batch["lr_res"], hr_res=batch["hr_res"])
        pred, tgt, mask = out["depth"], batch["hr"], batch["mask"]
        tiles.append(compute_flood_metrics(pred, tgt, mask))
        acc.update(pred, tgt, mask)
    merged = dict(average_metrics(tiles))
    merged.update(acc.compute())
    return merged


if __name__ == "__main__":
    main()
