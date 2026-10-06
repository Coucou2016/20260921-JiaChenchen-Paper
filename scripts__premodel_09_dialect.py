r"""Stage 9: the two comparison dialects, and the exact identity joining them.

Fine-lattice dialect
    every 2 m cell is compared against the coarse run value carried onto it by
    nearest-neighbour upsampling.
Coarse-lattice dialect
    every coarse block is compared against the area-weighted 2 m truth of that
    same block.

For one block holding N fine cells with truths t_i, a coarse-run value c and the
block mean a = (1/N) sum_i t_i, the two dialects are joined exactly by

    (1/N) sum_i (c - t_i)^2  =  (c - a)^2  +  (1/N) sum_i (a - t_i)^2

because the cross term vanishes, since sum_i (a - t_i) = 0.  The left side is the
fine-lattice error, the first term on the right is the error the coarse run
actually makes, and the second is the within-block spread of the ground truth,
which no coarse run and no block-constant correction can represent.

Nothing here trains, changes or re-evaluates any model.  Outputs go to
outputs/premodel/scale_dialect.json.

Usage:  python scripts/premodel_09_dialect.py
"""
from __future__ import annotations

import time

import numpy as np

import premodel_lib as L

DOMAIN_Y, DOMAIN_X = 23040, 12960
FINE_SHAPE = (11520, 6480)
FINE_RES = 2


def coarse_cells(res: str):
    r = L.RES_M[res]
    return int(DOMAIN_Y / r), int(DOMAIN_X / r)


def nn_index(n_c: int, n_f: int) -> np.ndarray:
    """Which of n_c coarse indices each of n_f fine indices falls into."""
    return np.clip(((np.arange(n_f) + 0.5) * n_c / n_f).astype(int), 0, n_c - 1)


def fine_nn_accounting(h2, native, agg, n_cells, strip=768):
    """Direct fine-cell accounting of the two dialects plus the cross term.

    Walks the 2 m lattice in row strips and accumulates sums, so no full-size
    float64 temporary is ever materialised.
    """
    ny, nx = h2.shape
    jx = nn_index(n_cells[1], nx)
    acc = {"n": 0, "tot": 0.0, "sim": 0.0, "within": 0.0, "cross": 0.0}
    for y0 in range(0, ny, strip):
        y1 = min(ny, y0 + strip)
        sub = h2[y0:y1]
        jy = nn_index(n_cells[0], ny)[y0:y1]
        upn = native[np.ix_(jy, jx)]
        up = agg[np.ix_(jy, jx)]
        fin = np.isfinite(sub) & np.isfinite(upn) & np.isfinite(up)
        if not fin.any():
            continue
        es = (upn[fin] - up[fin]).astype("float64")
        ea = (up[fin] - sub[fin]).astype("float64")
        acc["sim"] += float(np.sum(es * es))
        acc["within"] += float(np.sum(ea * ea))
        acc["cross"] += float(np.sum(es * ea))
        acc["n"] += int(fin.sum())
    n = max(acc["n"], 1)
    return {
        "n_fine_cells": acc["n"],
        "mse_fine": (acc["sim"] + acc["within"] + 2 * acc["cross"]) / n,
        "mse_coarse": acc["sim"] / n,
        "mse_within": acc["within"] / n,
        "cross": acc["cross"] / n,
    }


def block_accounting_integer(h2, native, ky, kx, cstrip=48):
    """Block-by-block identity for grids whose blocks are whole sets of fine cells."""
    ny, nx = h2.shape
    ny_c, nx_c = ny // ky, nx // kx
    resid_max, resid_mean = 0.0, 0.0
    blocks = 0
    mse_coarse, var_within, mse_fine_blk = 0.0, 0.0, 0.0
    for c0 in range(0, ny_c, cstrip):
        c1 = min(ny_c, c0 + cstrip)
        block = h2[c0 * ky:c1 * ky].reshape(c1 - c0, ky, nx_c, kx)
        fin = np.isfinite(block)
        cnt = fin.sum(axis=(1, 3))
        tt = np.where(fin, block, 0.0).astype("float64")
        s2 = (tt * tt).sum(axis=(1, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            a = (tt.sum(axis=(1, 3)) / cnt)
        cc = native[c0:c1]
        good = (cnt > 0) & np.isfinite(cc) & np.isfinite(a)
        if not good.any():
            continue
        # independent route: sum of squared deviations taken one cell at a time
        dev = np.where(fin, tt - a[:, None, :, None], 0.0)
        v_sum = (dev * dev).sum(axis=(1, 3))
        v = v_sum / cnt
        # left side of the identity, again cell by cell
        dd = np.where(fin, cc[:, None, :, None] - tt, 0.0)
        lhs = (dd * dd).sum(axis=(1, 3)) / cnt
        rhs = (cc - a) ** 2 + (s2 / cnt - a ** 2)
        r = np.abs(lhs - rhs)[good]
        resid_max = max(resid_max, float(r.max()))
        resid_mean += float(r.sum())
        blocks += int(good.sum())
        mse_coarse += float(np.sum((cc - a)[good] ** 2))
        var_within += float(np.sum(v[good]))
        mse_fine_blk += float(np.sum(lhs[good]))
    b = max(blocks, 1)
    return {
        "n_blocks": blocks,
        "mse_coarse_blocks": mse_coarse / b,
        "var_within_blocks": var_within / b,
        "mse_fine_blocks": mse_fine_blk / b,
        "residual_max": resid_max,
        "residual_mean": resid_mean / b,
    }


def check_block_constant_upsample(agg, n_cells, ky, kx):
    """Nearest upsampling followed by area-weighted aggregation returns the input.

    This is the reason nearest neighbour is used rather than bilinear, whose
    weights do not preserve the block mean.
    """
    up = L.resample_nearest(agg, n_cells, FINE_RES, FINE_SHAPE)
    back = L.aggregate(up, FINE_RES, n_cells)
    ok = np.isfinite(agg) & np.isfinite(back)
    d = (back[ok] - agg[ok]).astype("float64")
    return {"n": int(ok.sum()), "max_abs": float(np.abs(d).max()) if ok.any() else None,
            "mean_abs": float(np.mean(np.abs(d))) if ok.any() else None}


def main() -> None:
    out = {"fine_resolution_m": FINE_RES, "domain_fine_shape": list(FINE_SHAPE),
           "scenarios": L.SCENARIOS, "resolutions": L.COARSE,
           "identity": ("(1/N) sum_i (c-t_i)^2 = (c-a)^2 + (1/N) sum_i (a-t_i)^2, "
                        "with a the area-weighted block mean and the cross term "
                        "vanishing because sum_i (a-t_i) = 0"),
           "results": {}, "roundtrip": {}}
    for scen in L.SCENARIOS:
        t0 = time.time()
        h2 = L.load_flood("2m", scen, "h_max")
        print(f"[{scen}] read 2 m h_max {h2.shape} in {time.time()-t0:.1f}s", flush=True)
        out["results"][scen] = {}
        for res in L.COARSE:
            t1 = time.time()
            n_cells = coarse_cells(res)
            # fine cells per coarse cell, in each direction; only the integer
            # cases 10/20/30 m let a block be read as a whole set of fine cells
            ky = FINE_SHAPE[0] / n_cells[0]
            kx = FINE_SHAPE[1] / n_cells[1]
            agg = L.aggregate(h2, FINE_RES, n_cells)
            native = L.load_flood(res, scen, "h_max")
            rec = {"ratio": L.RES_M[res] / FINE_RES,
                   "fine_cells_per_block": ky,
                   "fine_nn": fine_nn_accounting(h2, native, agg, n_cells)}
            if abs(ky - round(ky)) < 1e-9 and abs(kx - round(kx)) < 1e-9:
                rec["block_exact"] = block_accounting_integer(h2, native,
                                                              int(round(ky)), int(round(kx)))
            # area-weighted block route, valid for the fractional 5 m lattice too
            agg2 = L.aggregate(h2 * h2, FINE_RES, n_cells)
            var = agg2 - agg * agg
            var[var < 0] = 0.0
            ok = np.isfinite(native) & np.isfinite(agg)
            rec["block_weighted"] = {
                "n_blocks": int(ok.sum()),
                "mse_coarse": float(np.mean((native[ok] - agg[ok]).astype("float64") ** 2)),
                "var_within": float(np.mean(var[ok].astype("float64"))),
            }
            rec["block_weighted"]["sum"] = (rec["block_weighted"]["mse_coarse"]
                                           + rec["block_weighted"]["var_within"])
            rec["block_weighted"]["excess_over_fine"] = (
                rec["fine_nn"]["mse_fine"] - rec["block_weighted"]["sum"])
            # shares, taken on the fine-lattice dialect which is what the reader sees
            mf = rec["fine_nn"]["mse_fine"]
            rec["shares"] = {
                "coarse_of_fine": rec["fine_nn"]["mse_coarse"] / mf,
                "within_of_fine": rec["fine_nn"]["mse_within"] / mf,
                "cross_of_fine": 2 * rec["fine_nn"]["cross"] / mf,
            }
            rec["rmse_m"] = {
                "fine_dialect": float(np.sqrt(rec["fine_nn"]["mse_fine"])),
                "coarse_dialect": float(np.sqrt(rec["fine_nn"]["mse_coarse"])),
                "within_block": float(np.sqrt(rec["fine_nn"]["mse_within"])),
            }
            del agg2, var
            out["results"][scen][res] = rec
            print(f"[{scen}] {res}: fine MSE {rec['fine_nn']['mse_fine']:.4f} = "
                  f"coarse {rec['fine_nn']['mse_coarse']:.4f} + within "
                  f"{rec['fine_nn']['mse_within']:.4f}  (cross "
                  f"{rec['fine_nn']['cross']:+.2e})  {time.time()-t1:.1f}s", flush=True)
        # round-trip property of nearest upsampling, checked once per scenario
        n10 = coarse_cells("10m")
        a10 = L.aggregate(h2, FINE_RES, n10)
        out["roundtrip"][scen] = check_block_constant_upsample(a10, n10, 5, 5)
        print(f"[{scen}] NN round-trip {out['roundtrip'][scen]}", flush=True)
        del h2, a10
    L.save_json("scale_dialect.json", out)


if __name__ == "__main__":
    main()
