"""Independent recomputation of the unit building coverage contrast in figure 12.

Rebuilds the three coverage groups from the raw two-metre flood and static files
with a separate code path, checks the aggregated truth against the stored stage-1
maps, and compares the per-group mean absolute error with cell_contrasts.json.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import netCDF4 as nc  # noqa: E402
import premodel_lib as L  # noqa: E402

SCEN = "100a"
RES = "10m"
STEP = 5                                    # 10 m / 2 m


def clean(a):
    a = np.array(a, dtype="float32", copy=True)
    a[~np.isfinite(a) | (a <= -9998.0)] = np.nan
    return a


def block_mean(a, step):
    ny, nx = a.shape
    ny = ny // step * step
    nx = nx // step * step
    v = a[:ny, :nx].reshape(ny // step, step, nx // step, step)
    return np.nanmean(v, axis=(1, 3))


with nc.Dataset(L.GRIDS / "2m" / f"flood_{SCEN}.nc") as fh:
    fine = clean(fh["h_max"][:])
with nc.Dataset(L.GRIDS / "2m" / "static.nc") as fh:
    bld2 = np.array(fh["Building_binary"][:], dtype="float32")
bld2 = np.where(bld2 >= 0, (bld2 > 0).astype("float32"), np.nan)
with nc.Dataset(L.GRIDS / RES / f"flood_{SCEN}.nc") as fh:
    native = clean(fh["h_max"][:])

truth = block_mean(fine, STEP).astype("float64")
cover = block_mean(bld2, STEP).astype("float64")
native = native.astype("float64")

d = np.load(ROOT / "outputs/premodel" / f"map_{SCEN}_{RES}.npz")
print("truth  max |mine - stored| =", np.nanmax(np.abs(truth - d["truth"])))
print("native max |mine - stored| =", np.nanmax(np.abs(native - d["native"])))
print("mask   agrees on",
      int((np.isfinite(truth) & np.isfinite(d["truth"])).sum()),
      "cells, mine-only",
      int((np.isfinite(truth) & ~np.isfinite(d["truth"])).sum()))

err = native - d["truth"].astype("float64")
ok = np.isfinite(err) & (d["truth"] > 0.05)
v = L.aggregate(bld2, 2, native.shape).astype("float64")
groups = (("无建筑", ok & (v <= 1e-9)),
          ("部分", ok & (v > 1e-9) & (v < 1 - 1e-9)),
          ("满覆盖", ok & (v >= 1 - 1e-9)))
print(f"\n{RES} {SCEN}  wet cells {int(ok.sum())}")
print("group      n         mae(m)    mean truth(m)   coverage range")
mine = {}
for name, sel in groups:
    n = int(sel.sum())
    mae = float(np.abs(err[sel]).mean())
    mt = float(d["truth"][sel].astype("float64").mean())
    mine[name] = (n, mae)
    print(f"{name:<10} {n:<9} {mae:.4f}    {mt:.4f}          "
          f"{v[sel].min() if n else float('nan'):.4f}.."
          f"{v[sel].max() if n else float('nan'):.4f}")

ref = json.loads((ROOT / "outputs/premodel/cell_contrasts.json")
                 .read_text(encoding="utf-8"))["resolutions"][RES][SCEN]["factors"]
e = ref["Building_binary"]
print("\nstored cell_contrasts.json")
stored = {}
for tag, name in (("low", "无建筑"), ("mid", "部分"), ("high", "满覆盖")):
    stored[name] = (e[tag]["n"], e[tag]["mae"])
    print(f"  {name:<10} n={e[tag]['n']:<9} mae={e[tag]['mae']:.4f} "
          f"truth={e[tag]['mean_truth_m']:.4f}")

print("\nmatch:", {k: (mine[k][0] == stored[k][0]
                       and abs(mine[k][1] - stored[k][1]) < 1e-6) for k in mine})

# what a quantile split of the raw binary flag would have produced
raw = np.array(nc.Dataset(L.GRIDS / RES / "static.nc")["Building_binary"][:],
               dtype="float32")
good = ok & np.isfinite(raw)
q = np.percentile(raw[good], [100 / 3., 200 / 3.])
print(f"\nraw binary flag on the same sample: unique {np.unique(raw[good])}")
print(f"  terciles q1={q[0]} q2={q[1]}  -> middle group n="
      f"{int((good & (raw > q[0]) & (raw < q[1])).sum())} (empty by construction)")
