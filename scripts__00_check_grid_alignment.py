"""Check whether the multi-resolution Wellington flood grids are strictly nested,
and quantify how far a coarse simulation is from the aggregated fine simulation.

Read-only: touches nothing under wellington-output-data/.
"""

import numpy as np
import xarray as xr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "wellington-output-data"
RES = [30, 20, 10, 5, 2]
SCENARIOS = ["20a", "100a"]


def open_hmax(scenario: str, res: int) -> xr.Dataset:
    return xr.open_dataset(ROOT / f"{scenario}_{res}m" / "flood_h_max.nc")


print("=" * 72)
print("1. GRID GEOMETRY (cell-edge extents)")
print("=" * 72)
geom = {}
for res in RES:
    ds = open_hmax("100a", res)
    x, y = ds["x"].values, ds["y"].values
    half = res / 2
    west, east = x[0] - half, x[-1] + half
    north, south = y[0] + half, y[-1] - half
    geom[res] = dict(
        nx=ds.sizes["x"], ny=ds.sizes["y"],
        west=west, east=east, north=north, south=south,
    )
    print(
        f"{res:>3}m  ny={ds.sizes['y']:>5} nx={ds.sizes['x']:>5}  "
        f"W={west:.1f} E={east:.1f} N={north:.1f} S={south:.1f}  "
        f"W-E span={east - west:.1f} N-S span={north - south:.1f}"
    )
    ds.close()

nw_x = {g["west"] for g in geom.values()}
nw_y = {g["north"] for g in geom.values()}
print(f"\nShared NW anchor?  west={nw_x}  north={nw_y}")
print(f"South edges differ: {sorted({r: geom[r]['south'] for r in RES}.items())}")

# Common nested domain, anchored at the shared NW corner.
# Height must be divisible by every resolution -> use lcm(2,5,10,20,30) = 60.
W0 = max(g["west"] for g in geom.values())
N0 = min(g["north"] for g in geom.values())
span_x = min(g["east"] for g in geom.values()) - W0
span_y = N0 - max(g["south"] for g in geom.values())
step = np.lcm.reduce(RES)
span_x_c = int(span_x // step) * step
span_y_c = int(span_y // step) * step
print(f"\nlcm of resolutions = {step} m")
print(f"Common domain: x [{W0:.0f}, {W0 + span_x_c:.0f}]  ({span_x_c} m)")
print(f"               y [{N0 - span_y_c:.0f}, {N0:.0f}]  ({span_y_c} m)")
print("Cropped shape per resolution (rows, cols), counting from index 0 at NW:")
crop = {}
for res in RES:
    crop[res] = (span_y_c // res, span_x_c // res)
    print(f"  {res:>3}m -> {crop[res]}   (drops {geom[res]['ny'] - crop[res][0]} rows, "
          f"{geom[res]['nx'] - crop[res][1]} cols)")


def load_crop(scenario: str, res: int) -> np.ndarray:
    ds = open_hmax(scenario, res)
    ny, nx = crop[res]
    a = ds["h_max"].isel(y=slice(0, ny), x=slice(0, nx)).values.astype(np.float32)
    ds.close()
    return a


def block_reduce(a: np.ndarray, f: int, how: str) -> np.ndarray:
    """Aggregate by f x f blocks, ignoring NaN."""
    ny, nx = a.shape
    b = a.reshape(ny // f, f, nx // f, f)
    with np.errstate(all="ignore"):
        if how == "mean":
            return np.nanmean(b, axis=(1, 3))
        if how == "max":
            return np.nanmax(b, axis=(1, 3))
    raise ValueError(how)


print()
print("=" * 72)
print("2. NODATA MASK CONSISTENCY  (h_max, 100a)")
print("=" * 72)
masks = {}
for res in RES:
    a = load_crop("100a", res)
    valid = np.isfinite(a)
    masks[res] = valid
    print(f"{res:>3}m  valid={valid.sum():>10,} / {valid.size:>10,}  "
          f"({100 * valid.mean():.2f}%)")

print("\nAgreement of the valid-land mask after aggregating each grid to 30 m:")
m30 = masks[30]
for res in [20, 10, 5, 2]:
    f = 30 // res
    if 30 % res:
        print(f"  {res:>3}m -> 30m: non-integer factor, skipped")
        continue
    frac = block_reduce(masks[res].astype(np.float32), f, "mean")
    agree = ((frac > 0.5) == m30).mean()
    print(f"  {res:>3}m -> 30m (factor {f:>2}): mask agreement = {100 * agree:.2f}%")

print()
print("=" * 72
      )
print("3. IS LR SIMPLY A DOWNSAMPLED HR?  (h_max)")
print("   Comparing coarse-grid simulation vs aggregated 2 m simulation")
print("=" * 72)
for scenario in SCENARIOS:
    a2 = load_crop(scenario, 2)
    print(f"\n--- {scenario} ---")
    for res in [30, 10]:
        f = res // 2
        coarse = load_crop(scenario, res)
        agg = block_reduce(a2, f, "mean")
        both = np.isfinite(coarse) & np.isfinite(agg)
        c, g = coarse[both], agg[both]
        r = np.corrcoef(c, g)[0, 1]
        print(
            f"{res:>3}m sim  vs  2m sim aggregated to {res}m   (n={both.sum():,})\n"
            f"     corr={r:.4f}  bias(coarse-fine)={np.mean(c - g):+.4f} m  "
            f"RMSE={np.sqrt(np.mean((c - g) ** 2)):.4f} m\n"
            f"     mean depth: coarse={c.mean():.4f} m  fine_agg={g.mean():.4f} m\n"
            f"     wet(>0.05m) area share: coarse={100 * (c > 0.05).mean():.1f}%  "
            f"fine_agg={100 * (g > 0.05).mean():.1f}%"
        )
    del a2

print()
print("=" * 72)
print("4. NAIVE BASELINE: nearest-upsample 30m -> 2m, error against 2m sim")
print("=" * 72)
for scenario in SCENARIOS:
    a2 = load_crop(scenario, 2)
    a30 = load_crop(scenario, 30)
    up = np.repeat(np.repeat(a30, 15, axis=0), 15, axis=1)
    both = np.isfinite(a2) & np.isfinite(up)
    d = up[both] - a2[both]
    print(
        f"{scenario}:  n={both.sum():,}  bias={d.mean():+.4f} m  "
        f"RMSE={np.sqrt((d ** 2).mean()):.4f} m  MAE={np.abs(d).mean():.4f} m"
    )
    wet_true = a2[both] > 0.05
    wet_up = up[both] > 0.05
    tp = (wet_true & wet_up).sum()
    csi = tp / (tp + (wet_up & ~wet_true).sum() + (~wet_up & wet_true).sum())
    print(f"          inundation CSI @0.05 m = {csi:.4f}  "
          f"(this is the bar an SR model must beat)")
    del a2, a30, up

print()
print("=" * 72)
print("5. PATCH CENSUS  (how many usable tiles exist)")
print("=" * 72)
print(f"{'scale':>7} {'LR res':>7} {'HR res':>7} {'HR patch':>9} {'tiles/frame':>12} "
      f"{'valid>70%':>10} {'wet>5%':>8}")
for lr_res, hr_res in [(30, 10), (20, 10), (10, 5), (30, 2), (10, 2), (20, 5)]:
    s = lr_res // hr_res
    hr_patch = 32 * s  # LR patch fixed at 32 px
    hr = load_crop("100a", hr_res)
    ny, nx = hr.shape
    ty, tx = ny // hr_patch, nx // hr_patch
    keep_valid = keep_wet = 0
    for i in range(ty):
        for j in range(tx):
            p = hr[i * hr_patch:(i + 1) * hr_patch, j * hr_patch:(j + 1) * hr_patch]
            v = np.isfinite(p)
            if v.mean() < 0.70:
                continue
            keep_valid += 1
            if np.nanmean(p > 0.05) > 0.05:
                keep_wet += 1
    print(f"{'x' + str(s):>7} {str(lr_res) + 'm':>7} {str(hr_res) + 'm':>7} "
          f"{str(hr_patch) + 'px':>9} {ty * tx:>12,} {keep_valid:>10,} {keep_wet:>8,}")
    del hr

print()
print("=" * 72)
print("6. STORAGE CHUNKING (matters a lot for patch reading)")
print("=" * 72)
for res in [30, 2]:
    ds = xr.open_dataset(ROOT / f"100a_{res}m" / "flood_h.nc")
    enc = ds["h"].encoding
    print(f"{res:>3}m flood_h.nc  shape={ds['h'].shape}  "
          f"chunksizes={enc.get('chunksizes')}  complevel={enc.get('complevel')}")
    ds.close()
