#!/usr/bin/env python
"""Evaluate a model (trainable or baseline) on val/test with flood metrics."""

from __future__ import annotations

import argparse
import json
import sys
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
from engine.trainer import evaluate_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/v0_10m2m_hmax.yaml")
    parser.add_argument("--model", type=str, default="bilinear")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_cfg = cfg["dataset"]
    root = ROOT / ds_cfg.get("root", "dataset")
    ds = WellingtonFixedSRDataset(
        root=root,
        split=args.split,
        lr_res=int(ds_cfg.get("lr_res", 10)),
        hr_res=int(ds_cfg.get("hr_res", 2)),
        target=ds_cfg.get("target", "h_max"),
        geo_mode=ds_cfg.get("geo_mode", "all"),
        scenarios=tuple(ds_cfg.get("scenarios", ["20a", "100a"])),
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fixed)
    model = build_model(args.model, cfg).to(device)
    if args.ckpt:
        load_checkpoint(args.ckpt, model, map_location=device)
    metrics = evaluate_loader(model, loader, device, max_batches=args.max_batches)
    print(json.dumps(metrics, indent=2))
    out = Path(args.out) if args.out else ROOT / "outputs" / "benchmarks" / f"{args.model}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
