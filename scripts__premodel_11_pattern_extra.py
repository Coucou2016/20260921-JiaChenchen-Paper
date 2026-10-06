r"""Stage 11: the extra cross-resolution pattern analyses requested for chapter 2.

Four things the earlier stage-10 report listed as outstanding, now actually run:

  1  SABRE cross-check.  The V-measure matrix is re-derived through the published
     procedure and, when the R runtime is available, through the SABRE package
     itself, and the two are compared value by value.
  2  Variogram / scale-space.  Empirical semivariograms of the water-depth field
     at 2/5/10/20/30 m, an exponential fit per resolution, and a scale-space view
     of how much of the depth variance survives aggregation.
  3  Contiguity-constrained regionalisation.  The same K-means regionalisation is
     repeated under an explicit rook-adjacency constraint and compared with the
     unconstrained one on the identical cells.
  4  Cluster-count sensitivity.  Every metric family is recomputed for K = 2..8
     and the ordering of the resolution pairs is checked for stability.

Nothing here trains a model, touches a checkpoint or changes an existing JSON
key.  Everything lands in outputs/premodel/pattern_extra.json (plus a separate
sabre_crosscheck.json once R is available).

Usage:  python scripts/premodel_11_pattern_extra.py [k|contig|vario|scale|all]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy import ndimage

import premodel_lib as L
import premodel_10_consistency as C

sys.stdout.reconfigure(encoding="utf-8")

SEED = 20260921
WET = 0.05
CHOSEN_K = 3
K_GRID = [2, 3, 4, 5, 6, 7, 8]
ALLRES = ["2m", "5m", "10m", "20m", "30m"]
COARSE = ["5m", "10m", "20m", "30m"]
PAIRS = [(a, b) for i, a in enumerate(ALLRES) for b in ALLRES[i + 1:]]
OUTNAME = "pattern_extra.json"
TILE_M = 960.0            # pooled-variogram tile side, metres


# --------------------------------------------------------------- shared build
def build(var: str) -> dict:
    """Native fields upsampled to 2 m, the common wet support, and the standardised
    log feature per resolution, exactly as stage 10 builds them."""
    scen = "100a"
    k_frame = C.frames_from_velocity_json()[scen]
    dom2 = C.domain_mask_2m()
    ref = C._load_var("2m", var, scen, k_frame)
    ring = dom2 & np.isfinite(ref) & (ref > WET)
    up = {"2m": ref}
    for res in COARSE:
        up[res] = L.resample_nearest(C._load_var(res, var, scen, k_frame),
                                     C.coarse_cells(res), C.FINE_RES, C.FINE_SHAPE)
    m = ring.copy()
    for res in ALLRES:
        m &= np.isfinite(up[res]) & (up[res] > WET)
    zs = {}
    for res in ALLRES:
        lg = np.log1p(up[res][m])
        mu, sd = float(np.mean(lg)), max(float(np.std(lg)), 1e-12)
        zs[res] = ((lg - mu) / sd).astype("float32")
        del lg
    val2m = up["2m"][m].astype("float64")
    del up, ring
    return {"scen": scen, "k_frame": k_frame, "dom2": dom2, "m": m,
            "zs": zs, "val2m": val2m, "n": int(m.sum())}


def _ordered_kmeans(z: np.ndarray, k: int) -> np.ndarray:
    km = C.kmeans_1d(z, k)
    lab = km.predict(z.reshape(-1, 1)).astype("int64")
    cen = np.asarray(km.cluster_centers_).ravel()
    order = np.argsort(cen)
    remap = np.empty(k, dtype="int64")
    remap[order] = np.arange(k)
    return remap[lab]


def _metrics(la: np.ndarray, lb: np.ndarray) -> dict:
    from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                                 cohen_kappa_score, fowlkes_mallows_score,
                                 normalized_mutual_info_score)
    from sklearn.metrics.cluster import homogeneity_completeness_v_measure
    h, c, v = homogeneity_completeness_v_measure(la, lb)
    out = {"v": float(v), "h": float(h), "c": float(c),
           "nmi": float(normalized_mutual_info_score(la, lb)),
           "ami": float(adjusted_mutual_info_score(la, lb)),
           "ari": float(adjusted_rand_score(la, lb)),
           "fm": float(fowlkes_mallows_score(la, lb)),
           "kappa": float(cohen_kappa_score(la, lb))}
    areal = C._iou_pairs(la, lb)
    out["mean_iou"] = float(areal["mean_iou"])
    out["area_tv"] = float(areal["area_tv"])
    return out


def _pairkey(a: str, b: str) -> str:
    return f"{a}|{b}"


# --------------------------------------------------------- item 4: K sweep
def k_sensitivity(dom2: np.ndarray) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    out = {"k_grid": K_GRID, "chosen_k": CHOSEN_K, "seed": SEED,
           "order_stability_metric": "Spearman rank correlation against K = 3",
           "variables": {}}
    for var in ("h_max", "speed"):
        t0 = time.time()
        B = build(var)
        m, zs, n = B["m"], B["zs"], B["n"]
        print(f"[k] {var} support {n}", flush=True)
        rng = np.random.default_rng(SEED)
        idx = rng.choice(n, size=min(60000, n), replace=False)
        sub = zs["2m"][idx].reshape(-1, 1)
        sil = {}
        for k in K_GRID:
            km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(sub)
            sil[str(k)] = float(silhouette_score(sub, km.labels_))
        rec = {"n": n, "silhouette": sil, "silhouette_sub": int(sub.shape[0]),
               "per_k": {}}
        for k in K_GRID:
            labs = {res: _ordered_kmeans(zs[res], k) for res in ALLRES}
            pk = {}
            for a, b in PAIRS:
                pk[_pairkey(a, b)] = _metrics(labs[a], labs[b])
            rec["per_k"][str(k)] = pk
            del labs
            print(f"[k] {var} K={k} done {round(time.time()-t0,1)}s", flush=True)
        # ordering stability per metric: rank the ten pairs at each K and compare
        from scipy.stats import spearmanr
        metrics = ["v", "ari", "fm", "kappa", "ami", "mean_iou", "area_tv"]
        ref_order = {mt: [rec["per_k"][str(CHOSEN_K)][_pairkey(*p)][mt]
                          for p in PAIRS] for mt in metrics}
        stab = {}
        for mt in metrics:
            rho = {}
            top = {}
            for k in K_GRID:
                vals = [rec["per_k"][str(k)][_pairkey(*p)][mt] for p in PAIRS]
                rho[str(k)] = float(spearmanr(ref_order[mt], vals).statistic)
                order = np.argsort(vals)[::-1]
                top[str(k)] = (_pairkey(*PAIRS[int(order[0])]),)
            stab[mt] = {"spearman_vs_k3": rho,
                        "min_spearman": float(min(rho.values())),
                        "top_pair": {k: top[k][0] for k in top}}
        rec["stability"] = stab
        out["variables"][var] = rec
        print(f"[k] {var} total {round(time.time()-t0,1)}s", flush=True)
    return out


# --------------------------------------------------- item 3: contiguity
def _core(m: np.ndarray):
    """A single connected component of the common support.

    The raw common support is dust: it breaks into thousands of connected
    components, most only a few dozen cells wide, so a global contiguity
    constraint is not even well posed on it.  One 3x3 closing bridges the thin
    gaps, and the core is the largest resulting component intersected back with
    the support, so every retained cell is wet in all five fields and the rook
    graph on the core is a single connected component.
    """
    closed = ndimage.binary_closing(m, structure=np.ones((3, 3)), iterations=1)
    lab, ncomp = ndimage.label(closed, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    core = (lab == int(np.argmax(sizes[1:]) + 1)) & m
    # removing cells can split the blob again; keep its largest rook-component
    lab2, n2 = ndimage.label(core, structure=np.array([[0, 1, 0],
                                                       [1, 1, 1],
                                                       [0, 1, 0]], dtype=bool))
    if n2 > 1:
        sz2 = np.bincount(lab2.ravel())
        core = lab2 == int(np.argmax(sz2[1:]) + 1)
    return core, ncomp


def contiguity(dom2: np.ndarray) -> dict:
    from scipy.sparse.csgraph import connected_components
    from sklearn.cluster import AgglomerativeClustering
    out = {"chosen_k": CHOSEN_K, "seed": SEED,
           "core_note": ("a global contiguity constraint is not well posed on the "
                         "raw common support, which breaks into thousands of small "
                         "components; the constrained run is therefore on the "
                         "largest component of a once-closed support, a single "
                         "connected region on which the constraint genuinely binds"),
           "variables": {}}
    for var in ("h_max", "speed"):
        t0 = time.time()
        B = build(var)
        m, zs, n = B["m"], B["zs"], B["n"]
        Wfull = C.rook_adjacency(m)
        ncomp_full, labf = connected_components(Wfull, directed=False)
        sz = np.bincount(labf)
        core, ncomp_closed = _core(m)
        ncore = int(core.sum())
        print(f"[contig] {var} n={n} comps={ncomp_full} "
              f"largest_share={sz.max()/n:.3f} core={ncore}", flush=True)
        del Wfull, labf
        sel = core[m]                       # boolean over the m flat order
        zc = {res: zs[res][sel] for res in ALLRES}
        Wcore = C.rook_adjacency(core)
        nc_core, _ = connected_components(Wcore, directed=False)
        rec = {"n_support": n, "n_core": ncore, "support_share": ncore / n,
               "components_support": int(ncomp_full),
               "largest_component_share": float(sz.max() / n),
               "components_closed": ncomp_closed,
               "components_core": int(nc_core), "k": CHOSEN_K,
               "agreement_constrained_vs_unconstrained": {},
               "cluster_share": {}, "pairs": {}}
        lu, lc = {}, {}
        for res in ALLRES:
            z = zc[res].reshape(-1, 1)
            lu[res] = _ordered_kmeans(zc[res], CHOSEN_K)
            ac = AgglomerativeClustering(n_clusters=CHOSEN_K, connectivity=Wcore,
                                         linkage="ward")
            lc[res] = ac.fit_predict(z).astype("int64")
            rec["cluster_share"][res] = {
                "unconstrained": [float(v) for v in np.bincount(
                    lu[res], minlength=CHOSEN_K) / ncore],
                "constrained": [float(v) for v in np.bincount(
                    lc[res], minlength=CHOSEN_K) / ncore]}
            print(f"[contig] {var} {res} ward done "
                  f"{round(time.time()-t0,1)}s", flush=True)
        # a compact plate for the figure: the core of the 2 m lattice keeps the
        # cells' row/column indices so a scatter can redraw the two labellings
        row, col = np.nonzero(core)

        def _by_depth(lab, z):
            """Ward labels arrive without an order; renumber them by mean depth so
            the plates and the K-means plate share one colour-to-depth reading."""
            cen = [float(np.mean(z[lab == q])) for q in range(CHOSEN_K)]
            order = np.argsort(cen)
            remap = np.empty(CHOSEN_K, dtype="int64")
            remap[order] = np.arange(CHOSEN_K)
            return remap[lab]

        np.savez_compressed(
            L.OUT / f"contiguity_core_{var}.npz",
            row=row.astype("int32"), col=col.astype("int32"),
            unconstrained=lu["2m"].astype("int8"),
            constrained_2m=_by_depth(lc["2m"], zc["2m"]).astype("int8"),
            unconstrained_30m=lu["30m"].astype("int8"),
            constrained_30m=_by_depth(lc["30m"], zc["30m"]).astype("int8"))
        for res in ALLRES:
            rec["agreement_constrained_vs_unconstrained"][res] = _metrics(
                lu[res], lc[res])
        rec["pairs"] = {"unconstrained": {}, "constrained": {}}
        for mode, labs in (("unconstrained", lu), ("constrained", lc)):
            for a, b in PAIRS:
                rec["pairs"][mode][_pairkey(a, b)] = _metrics(labs[a], labs[b])
        del Wcore, zc, lu, lc
        out["variables"][var] = rec
        print(f"[contig] {var} total {round(time.time()-t0,1)}s", flush=True)
    return out


# -------------------------------------------------------- item 2: variogram
def _semivario(Z: np.ndarray, M: np.ndarray, tc: int):
    """Pooled within-tile masked semivariogram via FFT, all lags in one pass.

    Returns numerator and pair-count arrays of shape (tc, tc) accumulated over
    tiles so the omnidirectional curve can be binned by physical lag.
    """
    from scipy.fft import rfft2, irfft2
    A = (Z * M).astype(np.float32)
    B = (Z * Z * M).astype(np.float32)
    Mf = M.astype(np.float32)
    ny, nx = Z.shape
    cnt = np.zeros((tc, tc))
    num = np.zeros((tc, tc))
    ph, pw = 2 * tc, 2 * tc
    for iy in range(0, ny - tc + 1, tc):
        for ix in range(0, nx - tc + 1, tc):
            fm = rfft2(Mf[iy:iy + tc, ix:ix + tc], (ph, pw))
            fa = rfft2(A[iy:iy + tc, ix:ix + tc], (ph, pw))
            fb = rfft2(B[iy:iy + tc, ix:ix + tc], (ph, pw))
            cmm = irfft2(fm.conj() * fm, (ph, pw))
            cmb = irfft2(fm.conj() * fb, (ph, pw))
            cbm = irfft2(fb.conj() * fm, (ph, pw))
            caa = irfft2(fa.conj() * fa, (ph, pw))
            cnt += cmm[:tc, :tc]
            num += (cmb + cbm - 2.0 * caa)[:tc, :tc]
    return cnt, num


def _bin_gamma(cnt, num, cell_m, nbins=24, max_lag=480.0):
    tc = cnt.shape[0]
    dy, dx = np.mgrid[0:tc, 0:tc]
    r = np.sqrt(dy ** 2 + dx ** 2) * cell_m
    m = cnt > 0
    rmax = min(max_lag, float(r[m].max()))
    edges = np.geomspace(cell_m * 0.9, rmax, nbins + 1)
    idx = np.digitize(r[m], edges) - 1
    cc = cnt[m]
    nn = num[m]
    rr = r[m]
    lags, gam, pairs = [], [], []
    for b in range(nbins):
        s = idx == b
        csum = float(cc[s].sum())
        if csum <= 0:
            continue
        lags.append(float(np.sqrt(np.mean(rr[s] ** 2))))
        gam.append(0.5 * float(nn[s].sum()) / csum)
        pairs.append(csum)
    return np.array(lags), np.array(gam), np.array(pairs)


def _range95(lags, gam, var, pairs, min_pairs):
    """Model-free practical range: first lag where gamma reaches 95% of the variance."""
    ok = pairs >= min_pairs
    if ok.sum() < 3:
        return float("nan")
    x, y = lags[ok], gam[ok]
    tgt = 0.95 * var
    above = np.nonzero(y >= tgt)[0]
    if above.size == 0:
        return float(x[-1])
    i = int(above[0])
    if i == 0:
        return float(x[0])
    return float(np.interp(tgt, [y[i - 1], y[i]], [x[i - 1], x[i]]))


def _fit_exp(lags, gam, pairs):
    from scipy.optimize import curve_fit
    ok = pairs >= 500
    lags, gam, pairs = lags[ok], gam[ok], pairs[ok]
    if lags.size < 5:
        return {"nugget": float("nan"), "sill": float("nan"),
                "range_m": float("nan"), "r2": float("nan")}
    def model(h, c0, c, a):
        return c0 + c * (1.0 - np.exp(-h / a))
    c0_0 = float(max(gam[0], 1e-6))
    sill0 = float(np.median(gam[-min(3, gam.size):]))
    a0 = float(lags[lags.size // 3])
    p0 = [c0_0, max(sill0 - c0_0, 1e-6), max(a0, lags[1])]
    bounds = ([0.0, 0.0, lags[1] * 0.5],
              [sill0 + 1e-6, sill0 * 2 + 1e-6, lags[-1] * 2])
    try:
        popt, _ = curve_fit(model, lags, gam, p0=p0, bounds=bounds,
                            sigma=1.0 / np.sqrt(pairs), absolute_sigma=False,
                            maxfev=40000)
    except Exception as exc:            # noqa: BLE001
        return {"nugget": float("nan"), "sill": float("nan"),
                "range_m": float("nan"), "r2": float("nan"),
                "note": f"fit failed: {type(exc).__name__}"}
    resid = gam - model(lags, *popt)
    r2 = 1.0 - float(np.sum(resid ** 2) /
                     max(float(np.sum((gam - gam.mean()) ** 2)), 1e-12))
    return {"nugget": float(popt[0]), "sill": float(popt[0] + popt[1]),
            "range_m": float(3.0 * popt[2]),
            "ratio_nugget_sill": float(popt[0] / (popt[0] + popt[1])),
            "decay_m": float(popt[2]), "r2": r2}


def variogram(dom2: np.ndarray) -> dict:
    out = {"tile_m": TILE_M, "max_lag_m": 480.0, "wet": WET,
           "model": "exponential gamma(h) = c0 + c (1 - exp(-h/a)), "
                    "effective range = 3a",
           "note": ("lags are pooled within 960 m tiles so the estimator is not "
                    "swamped by the domain-scale trend; the curve is truncated at "
                    "half the tile side, 480 m, where the pooled pair count is "
                    "still large and corner pairs do not dominate the bin"),
           "variables": {}}
    for var in ("h_max",):
        B = build(var)
        rec = {}
        for res in ALLRES:
            t0 = time.time()
            f = C._load_var(res, var, B["scen"], B["k_frame"])
            cell = L.RES_M[res]
            M = np.isfinite(f) & (f > WET)
            zr = np.where(M, np.clip(f, 0, None), 0.0)
            zl = np.where(M, np.log1p(np.clip(f, 0, None)), 0.0)
            tc = int(round(TILE_M / cell))
            entry = {"cell_m": cell, "tile_cells": tc,
                     "wet_cells": int(M.sum())}
            for tag, Z in (("depth", zr), ("log_depth", zl)):
                cnt, num = _semivario(Z, M, tc)
                lags, gam, pairs = _bin_gamma(cnt, num, cell)
                svar = float(np.var(Z[M].astype("float64")))
                fit = _fit_exp(lags, gam, pairs)
                fit["range95_m"] = _range95(lags, gam, svar, pairs, 500)
                fit["nugget_over_var"] = float(gam[0] / svar) if svar > 0 else None
                fit["gamma_at_first_lag"] = float(gam[0])
                fit["fit_reaches_sill"] = bool(
                    np.isfinite(fit["range_m"]) and fit["range_m"] <= lags[-1])
                entry[tag] = {"lags": lags.tolist(), "gamma": gam.tolist(),
                              "pairs": pairs.tolist(),
                              "fit": fit,
                              "sill_sample_var": svar}
                del cnt, num
            rec[res] = entry
            del f, M, zr, zl
            print(f"[vario] {var} {res} {round(time.time()-t0,1)}s", flush=True)
        out["variables"][var] = rec
    return out


def scale_space(dom2: np.ndarray) -> dict:
    """Dispersion variance and block-mean variance against aggregation scale."""
    f = L.load_flood("2m", "100a", "h_max")
    wet = np.isfinite(f) & (f > WET)
    v2 = f[wet].astype("float64")
    total = float(np.var(v2))
    ny, nx = f.shape
    scales = [4, 8, 16, 32, 64, 128, 256, 512]
    rows = []
    for s in scales:
        nc = (int(round(ny * 2 / s)), int(round(nx * 2 / s)))
        agg = L.aggregate(f, C.FINE_RES, nc)
        ok = np.isfinite(agg)
        back = L.resample_nearest(np.where(ok, agg, 0.0).astype("float32"),
                                  nc, C.FINE_RES, f.shape)
        d = (f - back)[wet]
        band = ok & (agg > WET)
        rows.append({
            "scale_m": s,
            "dispersion_var": float(np.mean(d ** 2)),
            "dispersion_frac": float(np.mean(d ** 2) / total),
            "block_var": float(np.var(agg[ok].astype("float64"))),
            "block_var_frac": float(np.var(agg[ok].astype("float64")) / total),
            "wet_cells": int(band.sum()),
            "wet_share": float(band.sum() / max(int(ok.sum()), 1)),
        })
        del agg, back, d
    return {"total_var": total, "wet_cells_2m": int(wet.sum()),
            "scales": rows}


# ------------------------------------------------------------- item 1: SABRE
def sabre_export(dom2: np.ndarray) -> dict:
    """Write the K = 3 label rasters to GeoTIFF so the R package can read them.

    The V-measure is scale-invariant to the log base, so SABRE's base-2 entropy
    and scikit-learn's natural-log entropy give the same number.
    """
    import rasterio
    from rasterio.transform import Affine
    d = L.OUT / "sabre_input"
    d.mkdir(parents=True, exist_ok=True)
    summary = {"scenario": "100a", "seed": SEED, "k": CHOSEN_K,
               "support_note": ("labels are -1 outside the common wet support; "
                                "rasterio writes them as nodata so SABRE's "
                                "crosstab ignores those cells"),
               "variables": {}}
    for var in ("h_max", "speed"):
        B = build(var)
        m, zs = B["m"], B["zs"]
        labs = {}
        for res in ALLRES:
            lab = _ordered_kmeans(zs[res], CHOSEN_K).astype("int16")
            labs[res] = lab
            full = np.full(m.shape, -1, dtype="int16")
            full[m] = lab
            with rasterio.open(d / f"{var}_{res}.tif", "w", driver="GTiff",
                               height=full.shape[0], width=full.shape[1],
                               count=1, dtype="int16", nodata=-1,
                               compress="deflate", transform=Affine.identity()) as dst:
                dst.write(full, 1)
            print(f"[sabre] wrote {var}_{res}.tif", flush=True)
            del full
        pv = {}
        from sklearn.metrics.cluster import homogeneity_completeness_v_measure
        for a, b in PAIRS:
            h, c, v = homogeneity_completeness_v_measure(labs[a], labs[b])
            pv[_pairkey(a, b)] = {"h": float(h), "c": float(c), "v": float(v)}
        summary["variables"][var] = {"n": B["n"], "pairs": pv}
        del labs, B
    L.save_json("sabre_python_vmeasure.json", summary)
    return summary


def sabre_reference(dom2: np.ndarray) -> dict:
    """SABRE's own formula, re-implemented from the published procedure.

    SABRE builds the contingency table z of the two regionalisations, then
    H(R) = MI / H(x), C(R) = MI / H(y), V = (1 + B) H C / (B H + C) with B = 1.
    Entropy and MI are computed in bits inside the package, which does not change
    the ratio.  The number-of-regions effect is reported by sweeping K: for each
    K both regionalisations have exactly K regions, so the count is controlled and
    the residual change in V is the effect of the region count itself.
    """
    from scipy.stats import entropy as _ent

    def _v(la, lb, B=1.0):
        ka = int(la.max()) + 1
        kb = int(lb.max()) + 1
        cnt = np.zeros((ka, kb), dtype="float64")
        np.add.at(cnt, (la, lb), 1.0)
        z = cnt / cnt.sum()
        px = z.sum(axis=1)
        py = z.sum(axis=0)
        hx = float(_ent(px, base=2))
        hy = float(_ent(py, base=2))
        pxy = z[z > 0]
        hxy = float(_ent(pxy, base=2))
        mi = hx + hy - hxy
        h = mi / hx if hx > 0 else 1.0
        c = mi / hy if hy > 0 else 1.0
        v = ((1 + B) * h * c) / (B * h + c) if (h + c) > 0 else 0.0
        return float(h), float(c), float(v)

    out = {"B": 1, "note": ("re-implementation of SABRE 0.4.3 vmeasure() from its "
                            "source; B = 1 makes it the standard symmetric V-measure"),
           "variables": {}}
    for var in ("h_max", "speed"):
        B = build(var)
        zs = B["zs"]
        labs = {res: _ordered_kmeans(zs[res], CHOSEN_K) for res in ALLRES}
        rec = {}
        for a, b in PAIRS:
            h, c, v = _v(labs[a], labs[b])
            rec[_pairkey(a, b)] = {"h": h, "c": c, "v": v}
        out["variables"][var] = {"n": B["n"], "pairs": rec}
        del labs, B
    return out


def sabre_compare() -> dict:
    """Compare SABRE's R output against the Python V-measure matrix."""
    d = L.OUT
    rp = d / "sabre_r_vmeasure.csv"
    pp = d / "sabre_python_vmeasure.json"
    out = {"r_available": False, "max_abs_diff": None, "variables": {}}
    if not rp.exists() or not pp.exists():
        out["note"] = "R output or Python reference missing"
        return out
    py = json.loads(pp.read_text(encoding="utf-8"))
    import csv

    def _cells(line):
        return [c.strip().strip('"') for c in next(csv.reader([line]))]

    rows = [r for r in rp.read_text(encoding="utf-8").splitlines() if r.strip()]
    hdr = _cells(rows[0])
    idx = {n: i for i, n in enumerate(hdr)}
    got = {}
    for line in rows[1:]:
        f = _cells(line)
        key = f"{f[idx['a']]}|{f[idx['b']]}"
        got[(f[idx["var"]], key)] = {m: float(f[idx[m]])
                                     for m in ("h", "c", "v")}
    worst = 0.0
    for var in ("h_max", "speed"):
        pv = py["variables"][var]["pairs"]
        rec = {}
        for key, val in pv.items():
            sv = got.get((var, key))
            if sv is None:
                continue
            diff = abs(sv["v"] - val["v"])
            rec[key] = {"python_v": val["v"], "sabre_v": sv["v"],
                        "abs_diff": diff,
                        "python_h": val["h"], "sabre_h": sv["h"],
                        "python_c": val["c"], "sabre_c": sv["c"]}
            worst = max(worst, diff)
        out["variables"][var] = rec
    out["r_available"] = True
    out["max_abs_diff"] = worst
    return out


# --------------------------------------------------------------------- main
def _merge(key: str, val) -> None:
    """Add or replace one top-level key without clobbering the others."""
    p = L.OUT / OUTNAME
    store = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    store[key] = val
    L.save_json(OUTNAME, store)


def main() -> None:
    which = sys.argv[1:] or ["all"]
    dom2 = C.domain_mask_2m()
    if "all" in which or "k" in which:
        _merge("k_sensitivity", k_sensitivity(dom2))
    if "all" in which or "contig" in which:
        _merge("contiguity", contiguity(dom2))
    if "all" in which or "vario" in which:
        _merge("variogram", variogram(dom2))
    if "all" in which or "scale" in which:
        _merge("scale_space", scale_space(dom2))
    if "all" in which or "sabre" in which:
        _merge("sabre_reference", sabre_reference(dom2))
    if "all" in which or "sabre_export" in which:
        sabre_export(dom2)
    if "all" in which or "sabre_cmp" in which:
        _merge("sabre_crosscheck", sabre_compare())


if __name__ == "__main__":
    main()
