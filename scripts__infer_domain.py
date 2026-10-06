#!/usr/bin/env python
"""Domain tiled inference with overlap + linear feather blending (plan §二十七).

tile = 480 m, overlap = 120 m → at 2 m: 240×240 with 60 px overlap.
"""

from __future__ import annotations

import argparse
import json
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


def feather_weights(h: int, w: int, overlap: int) -> np.ndarray:
    wy = np.ones(h, dtype=np.float32)
    wx = np.ones(w, dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap, dtype=np.float32)
        wy[:overlap] = ramp
        wy[-overlap:] = ramp[::-1]
        wx[:overlap] = ramp
        wx[-overlap:] = ramp[::-1]
    return np.outer(wy, wx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0_10m2m_hmax.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--overlap_m", type=float, default=120.0)
    parser.add_argument("--out", default="outputs/infer_domain")
    parser.add_argument("--max_tiles", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    hr_res = int(cfg["dataset"].get("hr_res", 2))
    overlap_px = int(args.overlap_m / hr_res)
    ds = WellingtonFixedSRDataset(root=ROOT / "dataset", split=args.split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]["name"], cfg).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    model.eval()

    domain_h, domain_w, patch = 11520, 6480, 240
    acc = np.zeros((domain_h, domain_w), dtype=np.float64)
    wsum = np.zeros((domain_h, domain_w), dtype=np.float64)
    fw = feather_weights(patch, patch, overlap_px)

    n = len(ds) if args.max_tiles is None else min(len(ds), args.max_tiles)
    for i in range(n):
        sample = collate_fixed([ds[i]])
        iy = int(sample["iy"][0])
        ix = int(sample["ix"][0])
        with torch.no_grad():
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
            out = model(
                lr=batch["lr"],
                lr_valid=batch["lr_valid"],
                static_cont=batch["static_cont"],
                landuse=batch["landuse"],
                lr_res=batch["lr_res"],
                hr_res=batch["hr_res"],
            )
        pred = out["depth"][0, 0].cpu().numpy()
        y0, x0 = iy * patch, ix * patch
        acc[y0 : y0 + patch, x0 : x0 + patch] += pred * fw
        wsum[y0 : y0 + patch, x0 : x0 + patch] += fw
        if (i + 1) % 20 == 0:
            print(f"tiles {i + 1}/{n}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mosaic = np.divide(acc, wsum, out=np.full_like(acc, np.nan), where=wsum > 0)
    np.save(out_dir / f"mosaic_{args.split}.npy", mosaic.astype(np.float32))
    meta = {"overlap_m": args.overlap_m, "overlap_px": overlap_px, "n_tiles": n, "split": args.split}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
