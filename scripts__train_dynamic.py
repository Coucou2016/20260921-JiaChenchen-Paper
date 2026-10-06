#!/usr/bin/env python
"""Train dynamic single-frame HydroGeo-SRNO (plan §二十一): LR [h,hux,hvy]+valid → HR h."""

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
sys.path.insert(0, str(ROOT))

from dataset.wellington_dynamic_sr import WellingtonDynamicSRDataset
from dataset.wellington_fixed_sr import collate_fixed
from engine.checkpoint import save_checkpoint
from engine.evaluator import build_model
from engine.trainer import EarlyStopper, evaluate_loader, train_step
from losses.flood_loss import FloodLoss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0_dynamic_h.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = False
    if device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 7:
        use_amp = bool(cfg.get("train", {}).get("amp", False))

    train_ds = WellingtonDynamicSRDataset(root=ROOT / "dataset", split="train")
    val_ds = WellingtonDynamicSRDataset(root=ROOT / "dataset", split="val")
    print(f"dynamic train={len(train_ds)} val={len(val_ds)}")
    bs = int(cfg.get("train", {}).get("batch_size", 1))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fixed)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fixed)

    model = build_model("hydrogeo_srno", cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    crit = FloodLoss()
    out_dir = ROOT / cfg.get("output", {}).get("dir", "outputs/v0_dynamic_h")
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(cfg.get("train", {}).get("epochs", 100))
    stopper = EarlyStopper(patience=20, mode="max")
    best_csi = -1.0
    step = 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        losses = []
        for batch in train_loader:
            stats = train_step(batch, model, crit, opt, None, device, use_amp=use_amp)
            losses.append(stats["total"])
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        metrics = evaluate_loader(model, val_loader, device, max_batches=20 if args.max_steps else None)
        csi = metrics.get("CSI_005", float("nan"))
        print(f"epoch={epoch} loss={sum(losses)/max(len(losses),1):.4f} CSI_005={csi:.4f}")
        save_checkpoint(out_dir / "last.pt", model, opt, epoch, metrics, cfg)
        if csi == csi and csi >= best_csi:
            best_csi = csi
            save_checkpoint(out_dir / "best_csi.pt", model, opt, epoch, metrics, cfg)
        if args.max_steps and step >= args.max_steps:
            break
        if stopper.step(csi if csi == csi else -1.0):
            break
    print(f"done elapsed_h={(time.time()-t0)/3600:.2f}")


if __name__ == "__main__":
    main()
