#!/usr/bin/env python
"""Train a baseline model from the plan §十五 matrix (resunet/edsr/srno_single/...)."""

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

from dataset.wellington_fixed_sr import WellingtonFixedSRDataset, collate_fixed
from engine.checkpoint import load_checkpoint, save_checkpoint
from engine.evaluator import build_model
from engine.trainer import EarlyStopper, evaluate_loader, train_step
from losses.flood_loss import FloodLoss, MaskedL1Loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0_10m2m_hmax.yaml")
    parser.add_argument("--model", required=True, help="resunet|edsr|rcan|rswinunet|srno_single|srno_dem")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--loss", default="flood", choices=["flood", "masked_l1"])
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(0)
        use_amp = bool(cfg.get("train", {}).get("amp", False)) and major >= 7
    else:
        use_amp = False

    ds_cfg = cfg["dataset"]
    geo_mode = "dem" if args.model in ("srno_dem", "rswinunet") else ds_cfg.get("geo_mode", "all")
    train_ds = WellingtonFixedSRDataset(
        root=ROOT / "dataset", split="train", geo_mode=geo_mode,
        lr_res=int(ds_cfg.get("lr_res", 10)), hr_res=int(ds_cfg.get("hr_res", 2)),
        target=ds_cfg.get("target", "h_max"),
        scenarios=tuple(ds_cfg.get("scenarios", ["20a", "100a"])),
    )
    val_ds = WellingtonFixedSRDataset(
        root=ROOT / "dataset", split="val", geo_mode=geo_mode,
        lr_res=int(ds_cfg.get("lr_res", 10)), hr_res=int(ds_cfg.get("hr_res", 2)),
        target=ds_cfg.get("target", "h_max"),
        scenarios=tuple(ds_cfg.get("scenarios", ["20a", "100a"])),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fixed)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fixed)

    # Adjust ResUNet static channels for geo_mode
    if args.model == "resunet" and geo_mode == "all":
        pass
    model = build_model(args.model, cfg).to(device)
    # Fix ResUNet in_channels if geo_mode reduced channels — rebuild for dem-only variants handled by rswinunet

    criterion = FloodLoss() if args.loss == "flood" else MaskedL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / "baselines" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_csi = -1.0
    if args.resume and Path(args.resume).exists():
        ckpt = load_checkpoint(args.resume, model, opt, map_location=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_csi = float((ckpt.get("metrics") or {}).get("CSI_005", -1.0))
    elif (out_dir / "last.pt").exists():
        ckpt = load_checkpoint(out_dir / "last.pt", model, opt, map_location=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_csi = float((ckpt.get("metrics") or {}).get("CSI_005", -1.0))

    epochs = args.epochs or int(cfg.get("train", {}).get("epochs", 300))
    stopper = EarlyStopper(patience=40, mode="max")
    if best_csi > 0:
        stopper.best = best_csi

    print(f"train baseline={args.model} device={device} epochs={start_epoch}..{epochs} → {out_dir}")
    history = out_dir / "history.jsonl"
    t0 = time.time()
    for epoch in range(start_epoch, epochs + 1):
        losses = []
        for batch in train_loader:
            stats = train_step(batch, model, criterion, opt, None, device, use_amp=use_amp)
            losses.append(stats["total"])
        metrics = evaluate_loader(model, val_loader, device)
        mean_loss = sum(losses) / max(len(losses), 1)
        csi = metrics.get("CSI_005", float("nan"))
        print(f"epoch={epoch} loss={mean_loss:.4f} CSI_005={csi:.4f} RMSE_wet={metrics.get('RMSE_wet', float('nan')):.4f}")
        with history.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "loss": mean_loss, **metrics}) + "\n")
        save_checkpoint(out_dir / "last.pt", model, opt, epoch, metrics, cfg)
        if csi == csi and csi >= best_csi:
            best_csi = csi
            save_checkpoint(out_dir / "best_csi.pt", model, opt, epoch, metrics, cfg)
        if stopper.step(csi if csi == csi else -1.0):
            print(f"early stop at {epoch}")
            break
    print(f"done {args.model} elapsed_h={(time.time()-t0)/3600:.2f}")


if __name__ == "__main__":
    main()
