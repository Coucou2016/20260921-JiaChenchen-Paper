"""Print every number the new result sections quote, in one place."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "outputs" / "report_figs" / "result_fields"


def j(n):
    return json.loads((F / n).read_text(encoding="utf-8"))


print("=" * 78)
print("PER TILE (test split, 242 tiles)")
rows = j("per_tile.json")
print(f"  tiles                  {len(rows)}")
peak = np.array([r["peak_truth"] for r in rows])
print(f"  peak truth  min/med/max {peak.min():.2f} / {np.median(peak):.2f} / {peak.max():.2f} m")
for k in ("CSI_005", "CSI_030", "CSI_100", "MAE_wet", "RMSE_wet",
          "VolumeRelativeError", "PeakDepthError"):
    a = np.array([r[f"frozen_{k}"] for r in rows], float)
    b = np.array([r[f"win_{k}"] for r in rows], float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = (b - a)[ok]
    se = d.std(ddof=1) / np.sqrt(d.size)
    print(f"  {k:<22} frozen {a[ok].mean():.5f}  win {b[ok].mean():.5f}  "
          f"d {d.mean():+.5f}  t {d.mean()/se:+.2f}  n {d.size}")
    print(f"  {'':<22} median frozen {np.median(a[ok]):.5f}  win {np.median(b[ok]):.5f}"
          f"  win>frozen {int((d>0).sum())}  win<frozen {int((d<0).sum())}")

print()
print("=" * 78)
print("MOMENTS")
mom = j("moments.json")
for k in ("frozen", "win"):
    g = mom["models"][k]["global"]
    print(f"  {k:<7} n={g['n']:,.0f} mean={g['mean']:+.5f} sd={g['sd']:.5f} "
          f"MAE={g['mae']:.5f} RMSE={g['rmse']:.5f}")
    print(f"  {'':<7} skew={g['skew']:+.3f} excess_kurtosis={g['excess_kurtosis']:.2f}")
print("  per-bin (true depth bin | n | MAE frozen | MAE win | bias frozen | bias win "
      "| skew frozen | skew win | kurt frozen | kurt win)")
for i, lab in enumerate(mom["bins"]):
    f, w = mom["models"]["frozen"], mom["models"]["win"]
    print(f"    {lab:<9} {f['n'][i]:>10,.0f}  {f['mae'][i]:.4f} {w['mae'][i]:.4f}   "
          f"{f['mean'][i]:+.4f} {w['mean'][i]:+.4f}   {f['skew'][i]:+.2f} {w['skew'][i]:+.2f}   "
          f"{f['excess_kurtosis'][i]:.1f} {w['excess_kurtosis'][i]:.1f}")

print()
print("=" * 78)
print("EXCEEDANCE (share of test pixels above a depth threshold, %)")
e = np.load(F / "exceed.npz")
thr, n = e["thr"], float(e["n"][0])
for t in (0.05, 0.10, 0.30, 0.50, 1.00, 1.50, 2.00):
    i = int(np.argmin(np.abs(thr - t)))
    tr, fr, wn = (e[k][i] / n * 100 for k in ("truth", "frozen", "win"))
    print(f"  >{t:.2f} m   truth {tr:8.4f}%   frozen {fr:8.4f}% ({fr/tr*100:6.1f}% of truth)"
          f"   win {wn:8.4f}% ({wn/tr*100:6.1f}% of truth)")

print()
print("=" * 78)
print("THRESHOLD SWEEP from the 4M-pixel subsample")
z = np.load(F / "pixels.npz")
gt, fr, wn = z["truth"].astype(np.float64), z["frozen"].astype(np.float64), z["win"].astype(np.float64)
print(f"  subsample n = {gt.size:,} of {int(z['n_total'][0]):,}")
for t in (0.05, 0.10, 0.30, 0.50, 1.00, 1.50):
    def csi(p):
        tp = np.sum((p > t) & (gt > t)); fp = np.sum((p > t) & ~(gt > t))
        fn = np.sum(~(p > t) & (gt > t))
        return tp / max(tp + fp + fn, 1)
    cf, cw = csi(fr), csi(wn)
    print(f"  CSI@{t:.2f}  frozen {cf:.4f}  win {cw:.4f}  d {cw-cf:+.4f}")
print("  deep-water (truth > 1 m) behaviour:")
sel = gt > 1.0
print(f"    n = {sel.sum():,}   mean truth {gt[sel].mean():.3f} m")
for nm, p in (("frozen", fr), ("win", wn)):
    d = p[sel] - gt[sel]
    print(f"    {nm:<7} mean {p[sel].mean():.3f} m  bias {d.mean():+.4f} m  "
          f"MAE {np.abs(d).mean():.4f} m  under-predicted share {100*(d<0).mean():.1f}%")
print(f"    depth-weighted volume shortfall: frozen {100*(1-fr[sel].sum()/gt[sel].sum()):.2f}%  "
      f"win {100*(1-wn[sel].sum()/gt[sel].sum()):.2f}%")

print()
print("=" * 78)
print("SPATIAL AUTOCORRELATION OF THE ERROR")
ac = j("autocorr.json")
a = np.array([x["moran"] for x in ac if x["model"] == "frozen"], float)
b = np.array([x["moran"] for x in ac if x["model"] == "win"], float)
ok = np.isfinite(a) & np.isfinite(b)
a, b = a[ok], b[ok]
d = b - a
print(f"  frozen   mean {a.mean():.4f}  median {np.median(a):.4f}  range {a.min():.3f}..{a.max():.3f}")
print(f"  win      mean {b.mean():.4f}  median {np.median(b):.4f}  range {b.min():.3f}..{b.max():.3f}")
print(f"  paired   d {d.mean():+.4f}  t {d.mean()/(d.std(ddof=1)/np.sqrt(d.size)):+.2f}  n {d.size}")
print(f"  tiles with Moran > 0.5 : frozen {int((a>0.5).sum())}  win {int((b>0.5).sum())}")

print()
print("=" * 78)
print("DOMAIN MAP EXTREMES")
pr = np.array([r["peak_truth"] for r in rows])
cf = np.array([r["frozen_CSI_100"] for r in rows], float)
cw = np.array([r["win_CSI_100"] for r in rows], float)
dd = cw - cf
print(f"  tiles with peak > 1 m : {int((pr>1).sum())} of {pr.size}")
print(f"  tiles with peak > 2 m : {int((pr>2).sum())}")
print(f"  CSI@1.00 = 0 tiles    : frozen {int((np.nan_to_num(cf)==0).sum())}  "
      f"win {int((np.nan_to_num(cw)==0).sum())}")
print(f"  per-tile improvement  : mean {np.nanmean(dd):+.4f}  median {np.nanmedian(dd):+.4f}  "
      f"max {np.nanmax(dd):+.4f}  min {np.nanmin(dd):+.4f}")

print()
print("=" * 78)
print("BOOTSTRAP FILE (if already written)")
p = F / "bootstrap_tiles.json"
if p.exists():
    for s in json.loads(p.read_text(encoding="utf-8")):
        print(f"  {s['metric']:<22} d {s['mean']:+.5f}  t {s['t']:+.2f}  "
              f"CI [{s['ci'][0]:+.5f}, {s['ci'][1]:+.5f}]  n {s['n']}")
print()
print("hypergeometric-style check: number of tiles where the improvement exceeds 0.01 CSI")
print(f"  {(dd > 0.01).sum()} tiles; where it is worse than -0.01: {(dd < -0.01).sum()}")
