#!/usr/bin/env python
"""Train multiscale HydroGeo-SRNO (stage M1/M2)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.wellington_fixed_sr import collate_fixed
from dataset.wellington_multiscale import WellingtonMultiscaleDataset
from engine.evaluator import build_model
from engine.trainer import train_step
from losses.flood_loss import FloodLoss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/v1_multiscale.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = [tuple(p) for p in cfg["dataset"]["pairs"]]
    ds = WellingtonMultiscaleDataset(
        root=ROOT / cfg["dataset"].get("root", "dataset"),
        split="train",
        pairs=pairs,
        geo_mode=cfg["dataset"].get("geo_mode", "all"),
    )
    loader = DataLoader(
        ds,
        batch_size=1,  # LR spatial sizes differ across pairs; batch>1 needs size grouping
        shuffle=True,
        collate_fn=collate_fixed,
    )
    model = build_model("hydrogeo_srno", cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = FloodLoss()
    steps = 0
    max_steps = args.max_steps or cfg["train"].get("max_steps")
    for batch in loader:
        stats = train_step(batch, model, crit, opt, None, device, use_amp=False)
        steps += 1
        print(f"step={steps} loss={stats['total']:.4f}")
        if max_steps and steps >= int(max_steps):
            break
    out = ROOT / cfg.get("output", {}).get("dir", "outputs/v1_multiscale")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "steps": steps}, out / "last_multiscale.pt")
    print(f"saved {out / 'last_multiscale.pt'}")


if __name__ == "__main__":
    main()
