"""Shared helpers for the pre-model (coarse-grid hydrodynamic error) study.

Everything here works on the already-published Wellington grids.  Nothing in this
module touches the super-resolution model or its results.

Conventions
-----------
* node data is NaN where the source file stores -9999 (sea / outside catchment).
* "aggregation" always means exact area-weighted block averaging of the 2 m field
  onto a coarser lattice that shares the same north-west corner.  For 5 m the
  480 m tile is 2.5 fine cells wide, so the weights are fractional; the same
  routine handles the integer cases 10/20/30 m exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
GRIDS = ROOT / "dataset" / "grids"
NODATA = -9999.0
STATIC_CHANNELS = [
    "DEM", "Slope", "Aspect_sin", "Aspect_cos", "Curv_plan", "Curv_profile",
    "Tpi", "Twi", "Dist_building", "Dist_road", "Dist_water",
    "Infiltration", "Manning", "Landuse", "Building_binary",
]
# cell size (m) of every stored resolution
RES_M = {"2m": 2, "5m": 5, "10m": 10, "20m": 20, "30m": 30}
COARSE = ["5m", "10m", "20m", "30m"]
SCENARIOS = ["20a", "100a"]

OUT = ROOT / "outputs" / "premodel"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- io
def clean(a):
    """Copy to float32 and turn the stored nodata sentinel into NaN."""
    out = np.array(a, dtype=np.float32, copy=True)
    out[~np.isfinite(out) | (out <= NODATA + 1.0)] = np.nan
    return out


def load_flood(res: str, scenario: str, var: str = "h_max", times=None):
    """Read one flood variable at one resolution.  times=None keeps every frame."""
    import netCDF4 as nc

    ds = nc.Dataset(GRIDS / res / f"flood_{scenario}.nc")
    v = ds[var]
    arr = v[:] if times is None else v[times]
    ds.close()
    return clean(arr)


def load_static(res: str, name: str):
    import netCDF4 as nc

    ds = nc.Dataset(GRIDS / res / "static.nc")
    arr = ds[name][:]
    ds.close()
    return clean(arr)


def load_static_multi(res: str, names):
    import netCDF4 as nc

    ds = nc.Dataset(GRIDS / res / "static.nc")
    out = {n: clean(ds[n][:]) for n in names}
    ds.close()
    return out


# ----------------------------------------------------------------- aggregation
_WEIGHTS: dict[tuple[int, int], sp.csr_matrix] = {}


def agg_matrix(n_coarse: int, n_fine: int, fine_res: int) -> sp.csr_matrix:
    """Area-overlap weights (n_coarse, n_fine); every row sums to one.

    Cell i of the coarse lattice is [i*c, (i+1)*c) with c = n_fine*fine_res/n_coarse.
    """
    key = (n_coarse, n_fine)
    if key in _WEIGHTS:
        return _WEIGHTS[key]
    coarse_res = n_fine * fine_res / n_coarse
    rows, cols, vals = [], [], []
    for i in range(n_coarse):
        lo, hi = i * coarse_res, (i + 1) * coarse_res
        j0 = int(np.floor(lo / fine_res))
        j1 = int(np.ceil(hi / fine_res))
        for j in range(j0, min(j1, n_fine)):
            ov = min(hi, (j + 1) * fine_res) - max(lo, j * fine_res)
            if ov > 1e-9:
                rows.append(i)
                cols.append(j)
                vals.append(ov / coarse_res)
    m = sp.csr_matrix((vals, (rows, cols)), shape=(n_coarse, n_fine), dtype=np.float32)
    _WEIGHTS[key] = m
    return m


def aggregate(fine: np.ndarray, fine_res: int, coarse_cells) -> np.ndarray:
    """Area-weighted average of a 2-D fine field onto a coarser lattice.

    `coarse_cells` is either one integer (square tile lattice) or a
    (n_y, n_x) pair, which is what the full domain needs.

    NaN is skipped and the weights are renormalised on the finite support, so a
    coarse cell that is entirely sea comes out NaN instead of zero.
    """
    ny_c, nx_c = (coarse_cells, coarse_cells) if np.isscalar(coarse_cells) \
        else (int(coarse_cells[0]), int(coarse_cells[1]))
    n_y, n_x = fine.shape
    Wy = agg_matrix(ny_c, n_y, fine_res)
    Wx = agg_matrix(nx_c, n_x, fine_res)
    fin = np.isfinite(fine).astype(np.float32)
    val = np.where(fin > 0, fine, 0.0).astype(np.float32)
    num = Wy @ val @ Wx.T
    den = Wy @ fin @ Wx.T
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.asarray(num / den, dtype=np.float32)
    out[den <= 1e-6] = np.nan
    return out


def aggregate_mask(fine_mask: np.ndarray, fine_res: int, coarse_cells, how: str = "mean"):
    """Aggregate a boolean mask.  'mean' gives a fraction, 'any' flags one wet cell."""
    f = fine_mask.astype(np.float32)
    a = aggregate(f, fine_res, coarse_cells)
    if how == "any":
        return np.nan_to_num(a, nan=0.0) > 0.0
    return a


def resample_nearest(field: np.ndarray, coarse_cells, fine_res: int,
                     fine_shape=None) -> np.ndarray:
    """Nearest upsample of a coarse field back to the fine lattice.

    `fine_shape` defaults to a square 480 m tile; pass (n_y, n_x) for the domain.
    """
    ny_c, nx_c = (coarse_cells, coarse_cells) if np.isscalar(coarse_cells) \
        else (int(coarse_cells[0]), int(coarse_cells[1]))
    ny_f, nx_f = (fine_shape if fine_shape is not None
                  else (int(round(480 / fine_res)), int(round(480 / fine_res))))
    jy = np.clip(((np.arange(ny_f) + 0.5) * ny_c / ny_f).astype(int), 0, ny_c - 1)
    jx = np.clip(((np.arange(nx_f) + 0.5) * nx_c / nx_f).astype(int), 0, nx_c - 1)
    return field[np.ix_(jy, jx)]


# ------------------------------------------------------------------- metrics
def csi(pred, truth, thr):
    """Critical success index for the binary event (depth > thr)."""
    p = pred > thr
    t = truth > thr
    tp = np.count_nonzero(p & t)
    fp = np.count_nonzero(p & ~t)
    fn = np.count_nonzero(~p & t)
    d = tp + fp + fn
    return (float(tp) / d) if d else float("nan")


def f1(pred, truth, thr):
    p = pred > thr
    t = truth > thr
    tp = np.count_nonzero(p & t)
    fp = np.count_nonzero(p & ~t)
    fn = np.count_nonzero(~p & t)
    d = 2 * tp + fp + fn
    return (2.0 * tp / d) if d else float("nan")


def numeric_metrics(pred, truth):
    """MAE / RMSE / bias / volume relative error on the shared finite support."""
    ok = np.isfinite(pred) & np.isfinite(truth)
    if not np.any(ok):
        return {}
    p = pred[ok]
    t = truth[ok]
    d = p - t
    sv = float(np.sum(t))
    return {
        "n": int(ok.sum()),
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "bias": float(np.mean(d)),
        "volume_rel": float((float(np.sum(p)) - sv) / sv) if sv > 0 else float("nan"),
        "mean_truth": float(np.mean(t)),
        "mean_pred": float(np.mean(p)),
        "p95_truth": float(np.percentile(t, 95)),
        "max_truth": float(np.max(t)),
    }


def pearson(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    x, y = a[ok].astype("float64"), b[ok].astype("float64")
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    x, y = a[ok], b[ok]
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return pearson(rx.astype("float64"), ry.astype("float64"))


def ssim(a, b, data_range=None):
    """Structural similarity on a finite common support."""
    from skimage.metrics import structural_similarity

    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 100:
        return float("nan")
    aa = np.where(ok, a, 0.0)
    bb = np.where(ok, b, 0.0)
    if data_range is None:
        data_range = float(max(aa.max(), bb.max()) - min(aa.min(), bb.min()))
    if data_range <= 0:
        return float("nan")
    try:
        return float(structural_similarity(aa, bb, data_range=data_range,
                                           gaussian_weights=True, sigma=1.5,
                                           use_sample_covariance=False))
    except Exception:
        return float("nan")


def block_minmax(field: np.ndarray, coarse_cells, fine_res: int):
    """Min and max of the fine field inside each coarse cell.

    Exact (stride) for integer ratios.  For the 5 m lattice, whose cells are
    2.5 fine cells wide, a 3x3 window centred on the cell is used instead; that
    is the closest integer neighbourhood and is stated in the report.
    """
    ny_c, nx_c = (coarse_cells, coarse_cells) if np.isscalar(coarse_cells) \
        else (int(coarse_cells[0]), int(coarse_cells[1]))
    ny, nx = field.shape
    ky = ny / ny_c
    kx = nx / nx_c
    if abs(ky - round(ky)) < 1e-9 and abs(kx - round(kx)) < 1e-9:
        red = field.reshape(ny_c, int(round(ky)), nx_c, int(round(kx)))
        return np.nanmin(red, axis=(1, 3)), np.nanmax(red, axis=(1, 3))
    mn = ndimage.minimum_filter(field, size=3, mode="nearest")
    mx = ndimage.maximum_filter(field, size=3, mode="nearest")
    jy = np.clip(((np.arange(ny_c) + 0.5) * ky).astype(int), 0, ny - 1)
    jx = np.clip(((np.arange(nx_c) + 0.5) * kx).astype(int), 0, nx - 1)
    return mn[np.ix_(jy, jx)], mx[np.ix_(jy, jx)]


def local_width(mask: np.ndarray) -> np.ndarray:
    """Characteristic width of a binary feature, 2 x distance to the nearest background."""
    dt = ndimage.distance_transform_edt(mask)
    return (2.0 * dt).astype(np.float32)


def save_json(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                                       default=_json_default), encoding="utf-8")
    print("wrote", OUT / name)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def load_json(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def usable_patches(split: str):
    idx = load_json("dataset/index/patches_480m.json")["patches"]
    return [p for p in idx if p["usable"] and p["split"] == split]
