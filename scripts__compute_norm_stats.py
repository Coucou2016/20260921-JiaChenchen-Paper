#!/usr/bin/env python
"""Compute missing static norm stats for 5/20/30 m from grids (train usable tiles only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.wellington_sr import STATIC_CHANNELS, WellingtonSRDataset, _clean

NODATA = -9999.0


def stats_for_res(res: int) -> dict:
    index = json.loads((ROOT / "dataset/index/patches_480m.json").read_text(encoding="utf-8"))
    patches = [p for p in index["patches"] if p["usable"] and p["split"] == "train"]
    ds = WellingtonSRDataset(root=ROOT / "dataset", lr_res=max(res, 10), hr_res=min(res, 2) if res != 5 else 2, split="train")
    # Read static directly
    import netCDF4 as nc
    path = ROOT / "dataset" / "grids" / f"{res}m" / "static.nc"
    nc_ds = nc.Dataset(path)
    py = 480 // res
    accum = {name: [] for name in STATIC_CHANNELS}
    for p in patches:
        iy, ix = p["iy"], p["ix"]
        ys = slice(iy * py, (iy + 1) * py)
        xs = slice(ix * py, (ix + 1) * py)
        for name in STATIC_CHANNELS:
            a = _clean(nc_ds[name][ys, xs])
            finite = a[np.isfinite(a)]
            if finite.size:
                accum[name].append(finite.astype(np.float64))
    nc_ds.close()
    out = {}
    for name, chunks in accum.items():
        if not chunks:
            continue
        v = np.concatenate(chunks)
        out[f"static_{res}m_{name}"] = {
            "n": int(v.size),
            "min": float(v.min()),
            "max": float(v.max()),
            "mean": float(v.mean()),
            "std": float(v.std()),
        }
    return out


def main() -> None:
    stats_path = ROOT / "dataset/index/norm_stats_train.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for res in (5, 20, 30):
        key = f"static_{res}m_DEM"
        if key in stats:
            print(f"skip {res}m (already present)")
            continue
        print(f"computing {res}m static norms...")
        part = stats_for_res(res)
        stats.update(part)
        print(f"  added {len(part)} keys")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"updated {stats_path}")


if __name__ == "__main__":
    main()
