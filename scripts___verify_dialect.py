"""Independent recomputation of the dialect identity.

Deliberately does NOT call premodel_lib.aggregate / resample_nearest.
Own overlap-weight aggregation, own nearest-neighbour index, float64 throughout.

Works on a physical window (given in 2 m indices) that is known to be wet, and
derives the coarse index ranges for each resolution from it.  The identity is
per-block, so a window is enough to test it.
"""
import io
import sys

import netCDF4 as nc
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

GRIDS = r"E:\Projects\20260921-JiaChenchen-Paper\dataset\grids"
FINE_RES = 2.0
NODATA = -9999.0

# physical window, in 2 m indices, inside the dense wet band
FY0, FY1 = 8000, 9000
FX0, FX1 = 3240, 3560


def clean(a):
    a = np.array(a, dtype=np.float32, copy=True)
    a[~np.isfinite(a) | (a <= NODATA + 1.0)] = np.nan
    return a


def load(res, scen):
    with nc.Dataset(f"{GRIDS}\\{res}\\flood_{scen}.nc") as ds:
        return clean(ds["h_max"][:])


def overlap_weights(n_c, n_f, k, c_off, f_off):
    """W[c, j] = fraction of fine cell (f_off+j) inside coarse block (c_off+c)."""
    lo = (np.arange(n_c) + c_off)[:, None] * k
    j = (np.arange(n_f) + f_off)[None, :]
    ov = np.clip(np.minimum(lo + k, j + 1) - np.maximum(lo, j), 0, None)
    return ov / k


print(f"physical window: 2 m rows {FY0}..{FY1}, cols {FX0}..{FX1}")
print(f"{'scen':>5} {'res':>5} {'wet%':>6} {'my fine':>9} {'my coarse':>10} {'my within':>10} "
      f"{'coarse+within':>14} {'resid':>10} {'2*cross':>11}")
print("-" * 92)

for scen in ("20a", "100a"):
    h2 = load("2m", scen)
    for res in ("5m", "10m", "20m", "30m"):
        r = float(res[:-1])
        k = r / FINE_RES
        # coarse index ranges covering the physical window
        R0, R1 = int(FY0 // k), int(np.ceil(FY1 / k))
        C0, C1 = int(FX0 // k), int(np.ceil(FX1 / k))
        fy0, fy1 = int(round(R0 * k)), int(round(R1 * k))
        fx0, fx1 = int(round(C0 * k)), int(round(C1 * k))

        sub = np.asarray(h2[fy0:fy1, fx0:fx1], dtype=np.float64)
        native = load(res, scen)[R0:R1, C0:C1]
        wet = float(np.isfinite(sub).mean() * 100)

        Wy = overlap_weights(R1 - R0, fy1 - fy0, k, R0, fy0)
        Wx = overlap_weights(C1 - C0, fx1 - fx0, k, C0, fx0)

        m = np.isfinite(sub)
        den = Wy @ m.astype(np.float64) @ Wx.T
        A = (Wy @ np.where(m, sub, 0.0) @ Wx.T) / np.where(den > 0, den, np.nan)
        A2 = (Wy @ np.where(m, sub * sub, 0.0) @ Wx.T) / np.where(den > 0, den, np.nan)
        A[den <= 0] = np.nan
        A2[den <= 0] = np.nan

        # my own nearest-neighbour index, matching cell centres
        jy = np.clip(((np.arange(fy1 - fy0) + 0.5) * (R1 - R0) / (fy1 - fy0)
                      + R0).astype(int), R0, R1 - 1) - R0
        jx = np.clip(((np.arange(fx1 - fx0) + 0.5) * (C1 - C0) / (fx1 - fx0)
                      + C0).astype(int), C0, C1 - 1) - C0
        upn = native[np.ix_(jy, jx)]
        up = A[np.ix_(jy, jx)]

        ok = np.isfinite(sub) & np.isfinite(upn) & np.isfinite(up)
        if not ok.any():
            print(f"{scen:>5} {res:>5} {wet:>6.1f}  no overlapping finite cells")
            continue
        es = upn[ok] - up[ok]
        ea = up[ok] - sub[ok]
        fine = float(np.mean(es * es + ea * ea + 2 * es * ea))
        coarse = float(np.mean(es * es))
        within = float(np.mean(ea * ea))
        cross2 = float(2 * np.mean(es * ea))
        print(f"{scen:>5} {res:>5} {wet:>6.1f} {fine:>9.4f} {coarse:>10.4f} {within:>10.4f} "
              f"{coarse+within:>14.4f} {fine-coarse-within:>10.2e} {cross2:>11.2e}")

        # block-exact route, float64, no algebraic shortcut
        okb = np.isfinite(native) & np.isfinite(A)
        lhs = float(np.nanmean((native[okb] - A[okb]) ** 2))
        var = A2[okb] - A[okb] ** 2
        rhs_v = float(np.nanmean(var))
        neg = int(np.sum(var < 0))
        print(f"      {'':>5} block route: coarse {lhs:.4f} + within {rhs_v:.4f} "
              f"= {lhs + rhs_v:.4f}  |  fine-route within {lhs + rhs_v - lhs:+.4f} "
              f"| var<0: {neg} of {int(okb.sum())}")
    print()
