"""Quick inventory of static.nc / flood nc variables, sampled (not full arrays)."""
from __future__ import annotations

import json
import sys
import io
from pathlib import Path

import numpy as np
import xarray as xr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

out = {}
for res in ["2m", "5m", "10m", "20m", "30m"]:
    p = ROOT / f"dataset/grids/{res}/static.nc"
    ds = xr.open_dataset(p, decode_cf=True)
    info = {"dims": dict(ds.sizes), "vars": []}
    for v in ds.variables:
        if v in ds.coords:
            info["vars"].append({"name": v, "coord": True, "dims": list(ds[v].dims),
                                 "dtype": str(ds[v].dtype)})
            continue
        da = ds[v]
        # sample every 37th cell along each axis to bound memory
        sl = tuple(slice(None, None, 37) for _ in da.dims)
        arr = da.isel({d: range(0, da.sizes[d], 37) for d in da.dims}).values.astype("float64")
        arr = np.where(arr <= -9990, np.nan, arr)
        finite = np.isfinite(arr)
        info["vars"].append({
            "name": v, "dims": list(da.dims), "shape": list(da.shape), "dtype": str(da.dtype),
            "attrs": {k: str(x) for k, x in da.attrs.items()},
            "sample_min": float(np.nanmin(arr)) if finite.any() else None,
            "sample_max": float(np.nanmax(arr)) if finite.any() else None,
            "sample_mean": float(np.nanmean(arr)) if finite.any() else None,
            "sample_finite_frac": float(finite.mean()),
            "n_unique_sample": int(np.unique(arr[finite]).size) if finite.any() else 0,
        })
    ds.close()
    out[res] = info
    print(f"=== static {res} dims {info['dims']}")
    for v in info["vars"]:
        if v.get("coord"):
            print(f"    {v['name']:22s} (coord) {v['dims']}")
        else:
            print(f"    {v['name']:22s} {str(v['dims']):18s} {v['shape']} {v['dtype']}"
                  f"  min={v['sample_min']} max={v['sample_max']} mean={v['sample_mean']}"
                  f"  fin={v['sample_finite_frac']:.3f} nuniq={v['n_unique_sample']}")

(ROOT / "outputs/premodel").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs/premodel/static_inventory.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("wrote outputs/premodel/static_inventory.json")
