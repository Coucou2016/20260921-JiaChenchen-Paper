#!/usr/bin/env python
"""Run plan §十五 baseline matrix on val/test and write metric tables.

Non-trainable: nearest, bilinear.
Trainable baselines use --ckpt if provided, else evaluated untrained only when
--allow_untrained is set (not for official tables).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.wellington_fixed_sr import WellingtonFixedSRDataset, collate_fixed
from engine.checkpoint import load_checkpoint
from engine.evaluator import build_model
from engine.trainer import evaluate_loader

# Plan table IDs
BASELINE_MODELS = [
    ("B0", "nearest"),
    ("B1", "bilinear"),
    ("B2", "resunet"),
    ("B3", "edsr"),
    ("B4", "rswinunet"),
    ("B5", "srno_single"),
    ("M1", "srno_dem"),
    ("M2", "hydrogeo_srno"),
    ("M3", "hydrogeo_srno"),  # same arch; flood-loss trained ckpt
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0_10m2m_hmax.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--out_dir", default="outputs/benchmarks")
    parser.add_argument("--models", nargs="*", default=None, help="subset of names")
    parser.add_argument("--ckpt_map", type=str, default=None, help="JSON {model: ckpt}")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # If CUDA is busy / tiny, allow CPU for classical upsamplers
    ds_cfg = cfg["dataset"]
    ds = WellingtonFixedSRDataset(
        root=ROOT / ds_cfg.get("root", "dataset"),
        split=args.split,
        lr_res=int(ds_cfg.get("lr_res", 10)),
        hr_res=int(ds_cfg.get("hr_res", 2)),
        target=ds_cfg.get("target", "h_max"),
        geo_mode=ds_cfg.get("geo_mode", "all"),
        scenarios=tuple(ds_cfg.get("scenarios", ["20a", "100a"])),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fixed)

    ckpt_map = {}
    if args.ckpt_map:
        ckpt_map = json.loads(Path(args.ckpt_map).read_text(encoding="utf-8"))
    # Defaults
    defaults = {
        "hydrogeo_srno": str(ROOT / "outputs/v0_10m2m_hmax/best_csi.pt"),
        "M3": str(ROOT / "outputs/v0_10m2m_hmax/best_csi.pt"),
        "resunet": str(ROOT / "outputs/baselines/resunet/best_csi.pt"),
        "edsr": str(ROOT / "outputs/baselines/edsr/best_csi.pt"),
        "rswinunet": str(ROOT / "outputs/baselines/rswinunet/best_csi.pt"),
        "srno_single": str(ROOT / "outputs/baselines/srno_single/best_csi.pt"),
        "srno_dem": str(ROOT / "outputs/baselines/srno_dem/best_csi.pt"),
    }
    for k, v in defaults.items():
        ckpt_map.setdefault(k, v)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    wanted = set(args.models) if args.models else None

    for bid, name in BASELINE_MODELS:
        if wanted and name not in wanted and bid not in wanted:
            continue
        # M2 vs M3: M2 may use a no-flood-loss ckpt if present
        ckpt = ckpt_map.get(bid) or ckpt_map.get(name)
        needs_ckpt = name not in ("nearest", "bilinear")
        if needs_ckpt and (not ckpt or not Path(ckpt).exists()):
            print(f"skip {bid}/{name}: missing ckpt {ckpt}")
            continue
        # Classical upsamplers: prefer CPU to free GPU for training
        dev = torch.device("cpu") if name in ("nearest", "bilinear") else device
        print(f"eval {bid} {name} on {args.split} device={dev}")
        model = build_model(name if bid != "M3" else "hydrogeo_srno", cfg).to(dev)
        if needs_ckpt:
            load_checkpoint(ckpt, model, map_location=dev)
        metrics = evaluate_loader(model, loader, dev)
        metrics["id"] = bid
        metrics["model"] = name
        metrics["split"] = args.split
        metrics["ckpt"] = ckpt if needs_ckpt else None
        rows.append(metrics)
        out_json = out_dir / f"{bid}_{name}_{args.split}.json"
        out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps({k: metrics[k] for k in ("CSI_005", "RMSE_wet", "MAE_wet", "F1_005")}, indent=2))

    # Aggregate table
    table_path = out_dir / f"table_{args.split}.csv"
    if rows:
        keys = ["id", "model", "split", "CSI_005", "F1_005", "CSI_030", "F1_030", "CSI_100", "F1_100",
                "RMSE_all", "RMSE_wet", "MAE_wet", "PeakDepthError", "FloodAreaRelativeError",
                "VolumeRelativeError", "PSNR", "SSIM", "ckpt"]
        with table_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        (out_dir / f"table_{args.split}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {table_path}")


if __name__ == "__main__":
    main()
