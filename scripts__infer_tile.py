#!/usr/bin/env python
"""Single-tile inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.wellington_fixed_sr import WellingtonFixedSRDataset, collate_fixed
from engine.checkpoint import load_checkpoint
from engine.evaluator import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0_10m2m_hmax.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="outputs/infer_tile.npy")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    ds = WellingtonFixedSRDataset(root=ROOT / "dataset", split=args.split)
    sample = collate_fixed([ds[args.index]])
    device = torch.device("cpu")
    model = build_model(cfg["model"]["name"], cfg).to(device)
    load_checkpoint(args.ckpt, model)
    model.eval()
    with torch.no_grad():
        out = model(
            lr=sample["lr"],
            lr_valid=sample["lr_valid"],
            static_cont=sample["static_cont"],
            landuse=sample["landuse"],
            lr_res=sample["lr_res"],
            hr_res=sample["hr_res"],
        )
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out["depth"].cpu().numpy())
    print(f"saved {path} shape={out['depth'].shape}")


if __name__ == "__main__":
    main()
