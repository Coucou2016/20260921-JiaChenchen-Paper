r"""Stage 10: cross-resolution spatial consistency of terrain, surface cover and
flood variables (the self-contained research part requested for chapter two).

Three steps, all on the already-published grids.  Nothing here trains, changes or
re-evaluates any model, and no downstream super-resolution artefact is touched.

Step 1  terrain and surface cover
        the 2 m static field is resampled DOWN onto each coarse lattice with
        bilinear interpolation and compared cell by cell with the coarse run.
        Area-weighted block averaging is reported alongside as a cross-check.
        Land cover and the building mask are categorical, so they are compared
        with the agreement rate, Cohen kappa and Cramer's V instead of Pearson.

Step 2  hydraulic variables
        depth and depth-averaged speed, paired at matching coordinates, with
        Pearson and Spearman, plus probability-distribution fits per resolution.

Step 3  spatial pattern similarity
        every resolution's own flood field becomes a regionalisation through a
        KMeans partition of its log values on the common 2 m support, carried
        onto that support by nearest neighbour.  Pairwise V-measure between the
        regionalisations is the similarity matrix.  A per-cell stability field
        then shows where the pattern survives coarsening and where it does not.

V-measure is computed from its definition, V = 2 h c / (h + c), with h the
homogeneity and c the completeness of one labelling against the other.  SABRE is
an R package and no R runtime exists on this machine, so its implementation
cannot be called, but the quantity is the same one.

Usage:  python scripts/premodel_10_consistency.py [cont|flood|vm|all]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import scipy.sparse as sp

import premodel_lib as L

sys.stdout.reconfigure(encoding="utf-8")

DOMAIN_Y, DOMAIN_X = 23040, 12960
FINE_SHAPE = (11520, 6480)
FINE_RES = 2
SEED = 20260921
WET = 0.05
STRIDE = 4                       # decimation used for the plotted rasters
ALLRES = ["2m", "5m", "10m", "20m", "30m"]
CONT_GROUPS = [
    ("地形", ["DEM", "Slope", "Curv_plan", "Curv_profile", "Tpi", "Twi"]),
    ("下垫面", ["Manning", "impervious_frac", "built_frac"]),
]
DERIVED_NAMES = {"impervious_frac": "不透水率", "built_frac": "建筑占比"}
FLOOD_VARS = ["h_max", "speed"]
PATTERN_SCEN = "100a"
WATER_CLASS = 7
IMPERVIOUS_CLASSES = (2, 3)      # the report's existing derived reading


def coarse_cells(res: str):
    r = L.RES_M[res]
    return int(DOMAIN_Y / r), int(DOMAIN_X / r)


def stride_view(a: np.ndarray) -> np.ndarray:
    ny, nx = a.shape
    return a[:ny // STRIDE * STRIDE:STRIDE, :nx // STRIDE * STRIDE:STRIDE]


# --------------------------------------------------------------- bilinear route
_BIL: dict[tuple[int, int], sp.csr_matrix] = {}


def bilinear_matrix(n_dst: int, n_src: int) -> sp.csr_matrix:
    """Row-stochastic (n_dst, n_src) matrix sampling src cell centres bilinearly."""
    key = (n_dst, n_src)
    if key in _BIL:
        return _BIL[key]
    u = (np.arange(n_dst) + 0.5) * (n_src / n_dst) - 0.5
    j0f = np.floor(u)
    w = u - j0f
    lo = j0f.astype(np.int64)
    hi = lo + 1
    edge = (lo < 0) | (hi > n_src - 1)
    lo = np.clip(lo, 0, n_src - 1)
    hi = np.clip(hi, 0, n_src - 1)
    w = np.where(edge, 0.0, w)
    rows = np.repeat(np.arange(n_dst), 2)
    cols = np.empty(2 * n_dst, dtype=np.int64)
    cols[0::2] = lo
    cols[1::2] = hi
    vals = np.empty(2 * n_dst, dtype=np.float32)
    vals[0::2] = 1.0 - w
    vals[1::2] = w
    m = sp.csr_matrix((vals, (rows, cols)), shape=(n_dst, n_src), dtype=np.float32)
    _BIL[key] = m
    return m


def bilinear_resample(field: np.ndarray, dst_shape, src_shape=None) -> np.ndarray:
    """Bilinear resampling between lattices sharing the north-west corner."""
    ny_s, nx_s = src_shape if src_shape is not None else field.shape
    ny_d, nx_d = dst_shape
    return np.asarray(bilinear_matrix(ny_d, ny_s) @ field
                      @ bilinear_matrix(nx_d, nx_s).T, dtype=np.float32)


def derived_static(name: str, res: str, dom: np.ndarray) -> np.ndarray:
    """The report's two derived surface-cover fractions, rebuilt at one grid size.

    impervious_frac is the share of land-use classes 2 and 3, which is the reading
    the chapter already uses for 不透水率.  built_frac is the share of cells whose
    building flag is one.  Both are fractions, so they are continuous quantities
    and can be compared with a correlation coefficient.
    """
    if name == "impervious_frac":
        a = L.load_static(res, "Landuse")
        v = np.where(dom & np.isfinite(a) & (a > 0),
                     np.isin(a, IMPERVIOUS_CLASSES).astype("float32"), np.nan)
        del a
        return v
    a = L.load_static(res, "Building_binary")
    v = np.where(dom & np.isfinite(a) & (a >= 0), (a > 0).astype("float32"), np.nan)
    del a
    return v


def load_speed(res: str, scen: str, frame: int) -> np.ndarray:
    """Depth-averaged speed |(hux,hvy)|/h clipped to 20 m/s, as the report does."""
    h = L.load_flood(res, scen, "h", times=frame)
    ux = L.load_flood(res, scen, "hux", times=frame)
    vy = L.load_flood(res, scen, "hvy", times=frame)
    num = np.sqrt(ux ** 2 + vy ** 2)
    del ux, vy
    with np.errstate(invalid="ignore", divide="ignore"):
        s = num / np.where(h > 0.02, h, np.nan)
    del num, h
    return np.clip(s, 0, 20).astype(np.float32)


def domain_mask_2m() -> np.ndarray:
    import netCDF4 as nc
    with nc.Dataset(L.GRIDS / "2m" / f"flood_{L.SCENARIOS[0]}.nc") as fh:
        return np.isfinite(L.clean(fh["h_max"][:]))


def frames_from_velocity_json() -> dict:
    vel = L.load_json("outputs/premodel/velocity_correlation.json")
    return {s: vel["scenarios"][s]["frame"] for s in L.SCENARIOS}


def majority_by_aggregation(indicator, classes, n):
    """Area-weighted majority class inside every coarse block.

    The 5 m lattice is two and a half 2 m cells wide, so an integer reshape is
    not available.  Averaging the class indicator fields and taking the largest
    average works for every ratio and matches the aggregation used elsewhere.
    """
    best = np.full(n, -1, dtype="int16")
    best_v = np.zeros(n, dtype="float32")
    for c in classes:
        a = L.aggregate((indicator == c).astype("float32"), FINE_RES, n)
        upd = a > best_v
        best_v = np.where(upd, a, best_v)
        best = np.where(upd, c, best).astype("int16")
        del a
    return best


# ------------------------------------------------------------------- step one
def continuous_part(dom2: np.ndarray) -> dict:
    t_all = time.time()
    scen = "100a"
    frames = frames_from_velocity_json()
    out = {"scenario": scen,
           "route_down": "bilinear interpolation onto the coarse cell centres",
           "route_down_crosscheck": ("area-weighted block average, the route used "
                                     "everywhere else in the report"),
           "support": "coarse cells whose centre falls inside the 2 m study domain",
           "variables": {}}
    for res in L.COARSE:
        n = coarse_cells(res)
        ratio = L.RES_M[res] / FINE_RES
        dom_c = np.nan_to_num(L.aggregate_mask(dom2, FINE_RES, n, "mean"), nan=0.0) > 0.5
        print(f"[cont] {res} ratio {ratio:g}", flush=True)
        for grp, names in CONT_GROUPS:
            for name in names:
                t0 = time.time()
                if name in DERIVED_NAMES:
                    fine = derived_static(name, "2m", dom2)
                    coarse = derived_static(name, res, dom_c)
                else:
                    fine = np.where(dom2, L.load_static("2m", name), np.nan).astype("float32")
                    coarse = np.where(dom_c, L.load_static(res, name), np.nan)
                bd = bilinear_resample(fine, n)
                aw = L.aggregate(fine, FINE_RES, n)
                ok = np.isfinite(coarse) & np.isfinite(bd)
                rec = {"group": grp, "res": res, "ratio": ratio, "n": int(ok.sum()),
                       "bilinear_pearson": L.pearson(np.where(ok, coarse, np.nan),
                                                     np.where(ok, bd, np.nan)),
                       "bilinear_spearman": L.spearman(np.where(ok, coarse, np.nan),
                                                       np.where(ok, bd, np.nan)),
                       "bilinear_rmse": float(np.sqrt(np.nanmean(
                           (coarse[ok] - bd[ok]).astype("float64") ** 2))),
                       "bilinear_bias": float(np.nanmean(
                           (coarse[ok] - bd[ok]).astype("float64"))),
                       "agg_pearson": L.pearson(np.where(ok, coarse, np.nan),
                                                np.where(ok, aw, np.nan)),
                       "agg_spearman": L.spearman(np.where(ok, coarse, np.nan),
                                                  np.where(ok, aw, np.nan)),
                       "agg_rmse": float(np.sqrt(np.nanmean(
                           (coarse[ok] - aw[ok]).astype("float64") ** 2))),
                       "fine_mean": float(np.nanmean(aw[ok])),
                       "coarse_mean": float(np.nanmean(coarse[ok]))}
                # curvature has a handful of extreme cells that alone decide the
                # product-moment coefficient, so a clipped variant is recorded too
                if name.startswith("Curv"):
                    lo, hi = np.nanpercentile(aw[ok], [0.5, 99.5])
                    ck = np.clip(coarse, lo, hi)
                    bk = np.clip(bd, lo, hi)
                    rec["clip_lo"], rec["clip_hi"] = float(lo), float(hi)
                    rec["bilinear_pearson_clipped"] = L.pearson(
                        np.where(ok, ck, np.nan), np.where(ok, bk, np.nan))
                    rec["bilinear_spearman_clipped"] = L.spearman(
                        np.where(ok, ck, np.nan), np.where(ok, bk, np.nan))
                    rec["fine_p1"], rec["fine_p99"] = (
                        float(np.nanpercentile(fine, 1)),
                        float(np.nanpercentile(fine, 99)))
                    del ck, bk
                if name == "DEM":
                    back = L.aggregate(bilinear_resample(aw, FINE_SHAPE, n),
                                       FINE_RES, n)
                    kk = np.isfinite(aw) & np.isfinite(back)
                    dd = (back[kk] - aw[kk]).astype("float64")
                    rec["bilinear_roundtrip_bias"] = float(np.mean(dd))
                    rec["bilinear_roundtrip_max"] = float(np.max(np.abs(dd)))
                    del back, dd
                out["variables"].setdefault(name, {"group": grp})[res] = rec
                print(f"[cont]   {name:<13} r={rec['bilinear_pearson']:+.5f} "
                      f"rho={rec['bilinear_spearman']:+.5f} "
                      f"agg={rec['agg_pearson']:+.5f} n={rec['n']} "
                      f"({time.time()-t0:.1f}s)", flush=True)
                del fine, coarse, bd, aw
        del dom_c

    for res in L.COARSE:
        n = coarse_cells(res)
        dom_c = np.nan_to_num(L.aggregate_mask(dom2, FINE_RES, n, "mean"), nan=0.0) > 0.5
        for name in FLOOD_VARS:
            t0 = time.time()
            if name == "h_max":
                fine = np.where(dom2, L.load_flood("2m", scen, "h_max"), np.nan)
                coarse = np.where(dom_c, L.load_flood(res, scen, "h_max"), np.nan)
            else:
                k = frames[scen]
                fine = np.where(dom2, load_speed("2m", scen, k), np.nan)
                coarse = np.where(dom_c, load_speed(res, scen, k), np.nan)
            bd = bilinear_resample(fine, n)
            aw = L.aggregate(fine, FINE_RES, n)
            ok = np.isfinite(coarse) & np.isfinite(bd) & (aw > WET)
            rec = {"group": "洪涝", "res": res, "ratio": L.RES_M[res] / FINE_RES,
                   "n": int(ok.sum()),
                   "bilinear_pearson": L.pearson(np.where(ok, coarse, np.nan),
                                                 np.where(ok, bd, np.nan)),
                   "bilinear_spearman": L.spearman(np.where(ok, coarse, np.nan),
                                                   np.where(ok, bd, np.nan)),
                   "bilinear_rmse": float(np.sqrt(np.nanmean(
                       (coarse[ok] - bd[ok]).astype("float64") ** 2))),
                   "bilinear_bias": float(np.nanmean(
                       (coarse[ok] - bd[ok]).astype("float64"))),
                   "agg_pearson": L.pearson(np.where(ok, coarse, np.nan),
                                            np.where(ok, aw, np.nan)),
                   "agg_spearman": L.spearman(np.where(ok, coarse, np.nan),
                                              np.where(ok, aw, np.nan)),
                   "fine_mean": float(np.nanmean(aw[ok])),
                   "coarse_mean": float(np.nanmean(coarse[ok]))}
            out["variables"].setdefault(name, {"group": "洪涝"})[res] = rec
            print(f"[cont]   {name:<13} r={rec['bilinear_pearson']:+.4f} "
                  f"rho={rec['bilinear_spearman']:+.4f} "
                  f"agg={rec['agg_pearson']:+.4f} n={rec['n']} "
                  f"({time.time()-t0:.1f}s)", flush=True)
            del fine, coarse, bd, aw
        del dom_c
    print(f"[cont] done in {time.time()-t_all:.0f}s", flush=True)
    return out


def categorical_part(dom2: np.ndarray) -> dict:
    from sklearn.metrics import cohen_kappa_score
    out = {"measures": {
        "agreement": ("fraction of coarse cells whose class equals the majority "
                      "class of the 2 m cells inside the same block"),
        "kappa": "Cohen kappa, agreement after removing what chance alone gives",
        "cramers_v": ("Cramer's V from the full class-by-class contingency table, "
                      "zero means no association and one means perfect association"),
        "why": ("Landuse carries a nominal code and Building_binary is a flag, so a "
                "product-moment correlation has no meaning for either of them")},
        "water_class": WATER_CLASS, "impervious_classes": list(IMPERVIOUS_CLASSES),
        "variables": {}}

    def cramers_v(a: np.ndarray, b: np.ndarray) -> float:
        aa = a - a.min()
        bb = b - b.min()
        tab = np.bincount(aa * (int(bb.max()) + 1) + bb,
                          minlength=(int(aa.max()) + 1) * (int(bb.max()) + 1))
        tab = tab.reshape(int(aa.max()) + 1, int(bb.max()) + 1).astype("float64")
        tab = tab[tab.sum(axis=1) > 0][:, tab.sum(axis=0) > 0]
        n = tab.sum()
        if n <= 0 or min(tab.shape) < 2:
            return float("nan")
        exp = np.outer(tab.sum(axis=1), tab.sum(axis=0)) / n
        chi2 = float(np.sum((tab - exp) ** 2 / np.maximum(exp, 1e-12)))
        return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))

    land_raw = L.load_static("2m", "Landuse")
    land2 = np.where(dom2 & np.isfinite(land_raw) & (land_raw > 0), land_raw, 0).astype("int16")
    del land_raw
    bld_raw = L.load_static("2m", "Building_binary")
    bld2 = np.where(dom2 & np.isfinite(bld_raw) & (bld_raw >= 0),
                    (bld_raw > 0).astype("int16"), -1)
    del bld_raw
    land_classes = [c for c in np.unique(land2) if c > 0]

    for res in L.COARSE:
        n = coarse_cells(res)
        dom_c = np.nan_to_num(L.aggregate_mask(dom2, FINE_RES, n, "mean"), nan=0.0) > 0.5
        land_c_raw = L.load_static(res, "Landuse")
        land_c = np.where(dom_c & np.isfinite(land_c_raw) & (land_c_raw > 0),
                          land_c_raw, 0).astype("int16")
        del land_c_raw
        bld_c_raw = L.load_static(res, "Building_binary")
        bld_c = np.where(dom_c & np.isfinite(bld_c_raw) & (bld_c_raw >= 0),
                         (bld_c_raw > 0).astype("int16"), -1)
        del bld_c_raw

        land_maj = majority_by_aggregation(land2, land_classes, n)
        bld_maj = majority_by_aggregation(bld2, [0, 1], n)
        for tag, cm, cmaj in (("Landuse", land_c, land_maj),
                              ("Building_binary", bld_c, bld_maj)):
            ok = (cm > 0) & (cmaj > 0) if tag == "Landuse" else ((cm >= 0) & (cmaj >= 0))
            a = cm[ok].astype("int64")
            b = cmaj[ok].astype("int64")
            rec = {"group": "下垫面", "res": res, "ratio": L.RES_M[res] / FINE_RES,
                   "n": int(ok.sum()),
                   "agreement": float(np.mean(a == b)),
                   "kappa": float(cohen_kappa_score(a, b)),
                   "cramers_v": cramers_v(a, b),
                   "majority_frac_coarse": float(np.max(np.bincount(a)) / max(a.size, 1))}
            if tag == "Building_binary":
                rec["building_frac_coarse"] = float(np.mean(a > 0))
                rec["building_frac_block"] = float(np.mean(b > 0))
            out["variables"].setdefault(tag, {"group": "下垫面"})[res] = rec
            print(f"[cat] {tag:<16} {res} agree={rec['agreement']:.4f} "
                  f"kappa={rec['kappa']:.4f} V={rec['cramers_v']:.4f} n={rec['n']}",
                  flush=True)
        del land_c, bld_c, land_maj, bld_maj, dom_c
    return out


# ------------------------------------------------------------------- step two
def flood_part() -> dict:
    from scipy import stats

    out = {"speed_definition": L.load_json(
               "outputs/premodel/velocity_correlation.json")["note"],
           "wet_threshold_m": WET,
           "fit_note": ("families are fitted by maximum likelihood on a random "
                        "subsample of at most one hundred thousand wet cells, with "
                        "the location parameter fixed at zero, seed 20260921"),
           "variables": {}}
    fl = L.load_json("outputs/premodel/flood_error.json")
    vel = L.load_json("outputs/premodel/velocity_correlation.json")
    frames = frames_from_velocity_json()

    def fit(sample: np.ndarray) -> dict:
        x = sample[np.isfinite(sample) & (sample > WET)].astype("float64")
        if x.size > 100000:
            x = np.random.default_rng(SEED).choice(x, 100000, replace=False)
        cands = {"Gamma 伽马": stats.gamma,
                 "Log-normal 对数正态": stats.lognorm,
                 "Weibull 三参数韦布尔": stats.weibull_min,
                 "Exponential 指数": stats.expon,
                 "Log-logistic 对数逻辑斯蒂": stats.fisk}
        rows = []
        for cn, dist in cands.items():
            try:
                p = dist.fit(x, floc=0)
                ll = float(np.sum(dist.logpdf(x, *p)))
                aic = 2 * (len(p) - 1) - 2 * ll      # floc is fixed, not estimated
                d, pv = stats.kstest(x, dist.cdf, args=p)
                rows.append({"family": cn, "params": [float(v) for v in p],
                             "aic": float(aic), "ks": float(d), "ks_p": float(pv)})
            except Exception as exc:                 # pragma: no cover
                rows.append({"family": cn, "error": f"{type(exc).__name__}: {exc}"})
        good = sorted([r for r in rows if "aic" in r], key=lambda r: r["aic"])
        best = good[0] if good else {}
        return {"n_fit": int(x.size), "mean": float(np.mean(x)),
                "median": float(np.median(x)), "std": float(np.std(x)),
                "p95": float(np.percentile(x, 95)),
                "skew": float(stats.skew(x)),
                "kurtosis_excess": float(stats.kurtosis(x)),
                "best_family": best.get("family"), "best_aic": best.get("aic"),
                "best_ks": best.get("ks"), "best_ks_p": best.get("ks_p"),
                "best_params": best.get("params"),
                "runner_up": good[1]["family"] if len(good) > 1 else None,
                "runner_aic": good[1]["aic"] if len(good) > 1 else None,
                "all_families": rows}

    pairs = {}
    for scen in L.SCENARIOS:
        k = frames[scen]
        h2 = L.load_flood("2m", scen, "h_max")
        s2 = load_speed("2m", scen, k)
        out["variables"].setdefault("h_max", {})[scen] = {"resolutions": {}}
        out["variables"].setdefault("speed", {})[scen] = {"resolutions": {}}
        out["variables"]["h_max"][scen]["dist_2m"] = fit(h2)
        out["variables"]["speed"][scen]["dist_2m"] = fit(s2)
        for res in L.COARSE:
            n = coarse_cells(res)
            hc = L.load_flood(res, scen, "h_max")
            sc = load_speed(res, scen, k)
            ph = fl["resolutions"][scen][res]
            pv = vel["scenarios"][scen]["resolutions"][res]
            out["variables"]["h_max"][scen]["resolutions"][res] = {
                "n": int(ph["n"]), "pearson_wet": ph["pearson_wet"],
                "spearman_wet": ph["spearman_wet"], "rmse": ph["rmse"],
                "bias": ph["bias"], "mean_truth": ph["mean_truth"],
                "mean_native": ph["mean_pred"], "dist_native": fit(hc)}
            out["variables"]["speed"][scen]["resolutions"][res] = {
                "n": int(pv["n"]), "pearson_wet": pv["pearson"],
                "spearman_wet": pv["spearman"], "mean_native": pv["mean_native"],
                "mean_truth": pv["mean_agg"], "median_native": pv["median_native"],
                "median_truth": pv["median_agg"], "dist_native": fit(sc)}
            h_agg = L.aggregate(h2, FINE_RES, n)
            s_agg = L.aggregate(s2, FINE_RES, n)
            for tag, a, b, m in (("h_max", hc, h_agg,
                                  np.isfinite(hc) & np.isfinite(h_agg)),
                                 ("speed", sc, s_agg,
                                  np.isfinite(sc) & np.isfinite(s_agg) & (s_agg > WET))):
                ai = a[m].astype("float32")
                bi = b[m].astype("float32")
                if ai.size > 200000:
                    idx = np.random.default_rng(SEED).choice(ai.size, 200000,
                                                             replace=False)
                    ai, bi = ai[idx], bi[idx]
                pairs[f"{tag}_{scen}_{res}"] = np.stack([ai, bi])
            print(f"[flood] {scen} {res}: depth r={ph['pearson_wet']:.3f} "
                  f"rho={ph['spearman_wet']:.3f} | speed r={pv['pearson']:.3f} "
                  f"rho={pv['spearman']:.3f}", flush=True)
            del hc, sc, h_agg, s_agg
        del h2, s2
    np.savez_compressed(L.OUT / "consistency_pairs.npz", **pairs)
    print(f"[flood] wrote consistency_pairs.npz ({len(pairs)} blocks)", flush=True)
    return out


# ----------------------------------------------------------------- step three
def choose_k(values: np.ndarray, ks=(3, 4, 5, 6, 7, 8), seed: int = SEED) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    rng = np.random.default_rng(seed)
    sub = rng.choice(values, min(60000, values.size), replace=False).reshape(-1, 1)
    scores, best = {}, None
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(sub)
        idx = rng.choice(sub.shape[0], min(20000, sub.shape[0]), replace=False)
        s = float(silhouette_score(sub[idx], km.labels_[idx]))
        scores[str(k)] = s
        if best is None or s > best[1]:
            best = (k, s)
    return {"chosen_k": best[0], "chosen_silhouette": best[1], "scores": scores,
            "subsample": int(sub.shape[0]), "metric": "silhouette on the 2 m reference"}


def kmeans_1d(values: np.ndarray, k: int, seed: int = SEED):
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    sub = (rng.choice(values, 300000, replace=False)
           if values.size > 300000 else values)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(sub.reshape(-1, 1))


def vmeasure_part(dom2: np.ndarray) -> dict:
    from sklearn.metrics.cluster import homogeneity_completeness_v_measure

    scen = "100a"
    k_frame = frames_from_velocity_json()[scen]
    print("[vm] reading 2 m fields", flush=True)
    h2 = L.load_flood("2m", scen, "h_max")
    s2 = load_speed("2m", scen, k_frame)
    ring = {"h_max": dom2 & np.isfinite(h2) & (h2 > WET),
            "speed": dom2 & np.isfinite(s2) & (s2 > WET)}
    print(f"[vm] support depth {int(ring['h_max'].sum())} "
          f"speed {int(ring['speed'].sum())}", flush=True)

    native = {"h_max": {"2m": h2}, "speed": {"2m": s2}}
    for res in L.COARSE:
        native["h_max"][res] = L.load_flood(res, scen, "h_max")
        native["speed"][res] = load_speed(res, scen, k_frame)

    out = {"scenario": scen, "speed_frame": k_frame, "seed": SEED,
           "regionalisation": {
               "unit": "2 m cells of the study domain",
               "feature": "log(1 + value) of that resolution's own simulated field",
               "scaling": ("each resolution's log field is centred and scaled by its "
                           "own mean and standard deviation on the common support, "
                           "so the comparison is about spatial pattern rather than "
                           "about the overall level"),
               "algorithm": "KMeans with one feature, KMeans++ initialisation",
               "k_choice": ("silhouette score over K = 3 to 8, evaluated on the 2 m "
                            "reference field"),
               "label_projection": ("a coarse label is carried to the 2 m cells "
                                    "inside its block by nearest neighbour, which "
                                    "keeps each block label intact"),
               "support": ("the common support is the 2 m cells whose own value "
                           "and the value of every coarse field at that cell all "
                           "exceed the wet threshold, so no dry coarse value is "
                           "ever set against a wet 2 m value"),
               "metric": ("V-measure, the harmonic mean of homogeneity and "
                          "completeness, computed from its definition"),
               "sabre_note": ("SABRE is an R package and this machine has no R "
                              "runtime, so its implementation cannot be called. "
                              "The V-measure definition is unique, so computing it "
                              "directly gives the same quantity.")},
           "variables": {}}
    raster = {}

    for var in FLOOD_VARS:
        print(f"[vm] === {var} ===", flush=True)
        # Every resolution is first carried to the 2 m lattice, then the common
        # support is narrowed to the cells that all five resolutions leave valid.
        up = {}
        for res in ALLRES:
            up[res] = (native[var][res] if res == "2m" else L.resample_nearest(
                native[var][res], coarse_cells(res), FINE_RES, FINE_SHAPE))
        m = ring[var].copy()
        for res in ALLRES:
            m &= np.isfinite(up[res]) & (up[res] > WET)
        print(f"[vm]   common wet support {int(m.sum())}", flush=True)
        lab2m, zs = {}, {}
        for res in ALLRES:
            lg = np.log1p(up[res][m])
            mu, sd = float(np.mean(lg)), max(float(np.std(lg)), 1e-12)
            zs[res] = (lg - mu) / sd
            del lg
        del up
        diag = choose_k(zs["2m"])
        k = diag["chosen_k"]
        rec = {"group": "洪涝", "k_diagnostics": diag, "pairwise": {}, "k": {},
               "support_cells": int(m.sum())}
        for res in ALLRES:
            km = kmeans_1d(zs[res], k)
            lab = np.full(dom2.shape, -1, dtype="int16")
            lab[m] = km.predict(zs[res].reshape(-1, 1)).astype("int16")
            lab2m[res] = lab
            rec["k"][res] = k
            rec.setdefault("cluster_share", {})[res] = [
                float(v) for v in np.bincount(lab[m].astype("int64"), minlength=k) / m.sum()]
            print(f"[vm]   {res} K={k}", flush=True)
        for a in ALLRES:
            rec["pairwise"][a] = {}
            for b in ALLRES:
                if a == b:
                    rec["pairwise"][a][b] = {"h": 1.0, "c": 1.0, "v": 1.0}
                    continue
                h, c, v = homogeneity_completeness_v_measure(
                    lab2m[a][m].astype("int64"), lab2m[b][m].astype("int64"))
                rec["pairwise"][a][b] = {"h": float(h), "c": float(c), "v": float(v)}
            print(f"[vm]   {a} vs: " + " ".join(
                f"{b}={rec['pairwise'][a][b]['v']:.3f}" for b in ALLRES), flush=True)

        ref = lab2m["2m"][m].astype("int64")
        agree = np.zeros(ref.shape, dtype="int8")
        for res in L.COARSE:
            lab = lab2m[res][m].astype("int64")
            lut = np.arange(1024, dtype="int64")
            for cl in np.unique(lab):
                lut[cl] = int(np.bincount(ref[lab == cl]).argmax())
            agree += (lut[lab] == ref).astype("int8")
        full_stab = np.full(dom2.shape, -1, dtype="int8")
        full_stab[m] = agree
        rec["stability_share"] = {str(int(q)): float(np.mean(agree == q))
                                  for q in np.unique(agree)}
        rec["mean_stability"] = float(np.mean(agree) / len(L.COARSE))
        rec["share_agree_le_one"] = float(np.mean(agree <= 1))
        rec["n_coarse_compared"] = len(L.COARSE)
        print(f"[vm]   stability {rec['stability_share']} "
              f"mean {rec['mean_stability']:.3f}", flush=True)
        raster[f"stab_{var}"] = stride_view(full_stab)
        for res in ALLRES:
            raster[f"lab_{var}_{res}"] = stride_view(lab2m[res])
        raster[f"val_{var}"] = stride_view(
            np.where(dom2, native[var]["2m"], np.nan).astype("float32"))
        out["variables"][var] = rec
        del lab2m, zs, ref, agree, full_stab
    del h2, s2
    np.savez_compressed(L.OUT / "consistency_regions.npz", **raster)
    print(f"[vm] wrote consistency_regions.npz ({len(raster)} rasters)", flush=True)
    return out


# ----------------------------------------------------------------- step four
def _load_var(res: str, var: str, scen: str, k_frame: int) -> np.ndarray:
    if var == "h_max":
        return L.load_flood(res, scen, "h_max")
    return load_speed(res, scen, k_frame)


def regionalise(dom2: np.ndarray, var: str, scen: str, k_frame: int) -> dict:
    """Rebuild the step-three regionalisation so the new families share its labels.

    The procedure is identical to the one used for the V-measure matrix: each
    resolution's own field is log-transformed, standardised on the common support,
    and cut by a one-feature KMeans whose K is chosen on the 2 m reference.
    """
    ref = _load_var("2m", var, scen, k_frame)
    ring = dom2 & np.isfinite(ref) & (ref > WET)
    up = {"2m": ref}
    for res in L.COARSE:
        up[res] = L.resample_nearest(_load_var(res, var, scen, k_frame),
                                     coarse_cells(res), FINE_RES, FINE_SHAPE)
    m = ring.copy()
    for res in ALLRES:
        m &= np.isfinite(up[res]) & (up[res] > WET)
    del ring
    zs = {}
    for res in ALLRES:
        lg = np.log1p(up[res][m])
        mu, sd = float(np.mean(lg)), max(float(np.std(lg)), 1e-12)
        zs[res] = ((lg - mu) / sd).astype("float32")
        del lg
    val2m = up["2m"][m].astype("float64")
    diag = choose_k(zs["2m"])
    k = diag["chosen_k"]
    lab = {}
    for res in ALLRES:
        km = kmeans_1d(zs[res], k)
        lab[res] = km.predict(zs[res].reshape(-1, 1)).astype("int64")
        # order the clusters by their centre so the codes read as an ordinal
        # band of the standardised log value, which makes Moran's I meaningful
        cen = np.asarray(km.cluster_centers_).ravel()
        order = np.argsort(cen)
        remap = np.empty(k, dtype="int64")
        remap[order] = np.arange(k)
        lab[res] = remap[lab[res]]
    del up, zs
    return {"m": m, "lab": lab, "k": k, "diag": diag, "val2m": val2m}


def rook_adjacency(mask: np.ndarray):
    """Sparse rook (four-neighbour) adjacency among the True cells of a 2-D mask."""
    idx = np.full(mask.shape, -1, dtype=np.int32)
    n = int(mask.sum())
    idx[mask] = np.arange(n, dtype=np.int32)
    a = mask[:, :-1] & mask[:, 1:]
    rows = idx[:, :-1][a]
    cols = idx[:, 1:][a]
    b = mask[:-1, :] & mask[1:, :]
    rows2 = idx[:-1, :][b]
    cols2 = idx[1:, :][b]
    r = np.concatenate([rows, rows2])
    c = np.concatenate([cols, cols2])
    del rows, cols, rows2, cols2, a, b, idx
    rr = np.concatenate([r, c])
    cc = np.concatenate([c, r])
    del r, c
    W = sp.csr_matrix((np.ones(rr.size, dtype=np.float64),
                       (rr.astype(np.int64), cc.astype(np.int64))),
                      shape=(n, n))
    return W


def moran_geary(vals: np.ndarray, W: sp.csr_matrix) -> dict:
    """Moran's I and Geary's C of one value vector under a spatial weight matrix.

    With a symmetric weight matrix the Geary numerator can be written as
    2 (d . z^2) - 2 z'Wz, where d is the row-sum vector, so the sparse matrix is
    never expanded to coordinate form.
    """
    v = vals.astype(np.float64)
    n = v.size
    z = v - v.mean()
    den = float(z @ z)
    Wz = W @ z
    zWz = float(z @ Wz)
    d = np.asarray(W.sum(axis=1)).ravel()
    S0 = float(d.sum())
    I = float((n / S0) * zWz / den) if den > 0 else float("nan")
    gnum = 2.0 * float(np.sum(d * z * z)) - 2.0 * zWz
    C = float(((n - 1) / (2 * S0)) * gnum / den) if den > 0 else float("nan")
    del z, Wz, d
    return {"moran": I, "geary": C, "n": int(n),
            "expected_moran": float(-1.0 / (n - 1)) if n > 1 else float("nan")}


def join_count(labels: np.ndarray, W: sp.csr_matrix, k: int,
               n_perm: int = 100, seed: int = SEED) -> dict:
    """Same-class join count for a nominal labelling, with a permutation z-score."""
    n = labels.size
    S0 = float(W.sum())
    p = np.bincount(labels, minlength=k).astype("float64") / n

    def stat(lab: np.ndarray) -> float:
        tot = 0.0
        for c in range(k):
            mask = (lab == c).astype(np.float64)
            tot += float(mask @ (W @ mask))
        return tot

    obs = stat(labels)
    exp = S0 * float(np.sum(p * p))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype="float64")
    work = labels.copy()
    for i in range(n_perm):
        rng.shuffle(work)
        null[i] = stat(work)
    sd = float(null.std(ddof=1))
    z = float((obs - null.mean()) / sd) if sd > 0 else float("nan")
    del work, null
    return {"observed": obs, "expected": exp,
            "ratio": float(obs / exp) if exp else float("nan"),
            "z": z, "n": int(n), "n_perm": n_perm}


def _front_field(res: str, var: str, scen: str, k_frame: int) -> np.ndarray:
    """The water-depth field whose wet footprint defines the front.

    An absolute speed threshold is not comparable across the five grids.  The
    derived speed on the 5 m lattice is about nine times smaller than at the
    coarser grids, so a fixed 0.05 speed threshold cuts away seventy per cent of
    its cells and leaves a footprint far smaller than the 2 m one.  That alone
    pushed the 2 m to 5 m modified Hausdorff distance of the speed front to 91 m,
    against 6.6 m for the water-depth front.  The front is therefore always read
    off a depth field in metres.  For the maximum-depth variable the field is the
    event maximum, and for the speed variable the depth at the peak-volume hour.
    """
    if var == "h_max":
        return L.load_flood(res, scen, "h_max")
    return L.load_flood(res, scen, "h", times=k_frame)


def _boundary_metrics(ref_mask: np.ndarray, res_mask: np.ndarray) -> dict:
    """Wet-mask boundary agreement on the shared 2 m lattice."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    b_ref = ref_mask & ~ndimage.binary_erosion(ref_mask)
    b_res = res_mask & ~ndimage.binary_erosion(res_mask)
    n_ref = int(b_ref.sum())
    n_res = int(b_res.sum())
    inter = int(np.count_nonzero(b_ref & b_res))
    union = int(np.count_nonzero(b_ref | b_res))
    f1 = (2.0 * inter / (n_ref + n_res)) if (n_ref + n_res) else float("nan")
    iou = (inter / union) if union else float("nan")
    pr, pc = np.nonzero(b_ref)
    qr, qc = np.nonzero(b_res)
    mhd = float("nan")
    if pr.size and qr.size:
        t_ref = cKDTree(np.column_stack([pr, pc]).astype("float32"))
        t_res = cKDTree(np.column_stack([qr, qc]).astype("float32"))
        d1 = t_ref.query(np.column_stack([qr, qc]).astype("float32"), k=1)[0]
        d2 = t_res.query(np.column_stack([pr, pc]).astype("float32"), k=1)[0]
        mhd = float(max(d1.mean(), d2.mean())) * FINE_RES
        del d1, d2, t_ref, t_res
    del b_ref, b_res, pr, pc, qr, qc
    return {"f1": f1, "iou": iou, "mhd_m": mhd,
            "n_ref_boundary": n_ref, "n_res_boundary": n_res}


def _iou_pairs(la: np.ndarray, lb: np.ndarray) -> dict:
    """Mean best-match IoU and total-variation area-share distance for two labelings."""
    from scipy.optimize import linear_sum_assignment
    ka = int(la.max()) + 1
    kb = int(lb.max()) + 1
    cnt = np.zeros((ka, kb), dtype="float64")
    np.add.at(cnt, (la, lb), 1.0)
    na = cnt.sum(axis=1)
    nb = cnt.sum(axis=0)
    inter = cnt
    union = na[:, None] + nb[None, :] - cnt
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
    r, c = linear_sum_assignment(-iou)
    best = iou[r, c]
    w = na[r]
    mean_iou = float(np.sum(best * w) / np.sum(w))
    pa = na / na.sum()
    pb = nb / nb.sum()
    tv = float(0.5 * np.sum(np.abs(pa[r] - pb[c])))
    return {"mean_iou": mean_iou, "area_tv": tv,
            "per_cluster_iou": [float(x) for x in best],
            "cluster_share_a": [float(x) for x in pa],
            "cluster_share_b": [float(x) for x in pb]}


def _separation(a: str, b: str) -> float:
    return max(L.RES_M[a], L.RES_M[b]) / min(L.RES_M[a], L.RES_M[b])


def chance_floor(la: np.ndarray, lb: np.ndarray, n_perm: int = 30,
                 seed: int = SEED) -> dict:
    """Mean of four read-outs under a random relabelling with fixed cluster sizes.

    Fowlkes-Mallows has a non-zero expectation, so its raw value flatters an
    agreement as soon as one cluster dominates.  Shuffling the second labelling
    keeps both cluster-size vectors fixed and gives the level to subtract.
    """
    from sklearn.metrics import (adjusted_rand_score, cohen_kappa_score,
                                 fowlkes_mallows_score,
                                 normalized_mutual_info_score)
    rng = np.random.default_rng(seed)
    got = {"v": [], "ari": [], "fm": [], "kappa": []}
    work = lb.copy()
    for _ in range(n_perm):
        rng.shuffle(work)
        got["v"].append(normalized_mutual_info_score(la, work))
        got["ari"].append(adjusted_rand_score(la, work))
        got["fm"].append(fowlkes_mallows_score(la, work))
        got["kappa"].append(cohen_kappa_score(la, work))
    del work
    return {key: float(np.mean(v)) for key, v in got.items()}


def boundary_part(var: str, scen: str, k_frame: int) -> tuple:
    """Family C on its own, so it can be recomputed without the slow families.

    Returns the per-pair records and, for the speed variable, one extra number
    that shows why the front is read off a depth field rather than off the speed
    field itself.
    """
    ref_mask = np.isfinite(_front_field("2m", var, scen, k_frame))
    ref_mask &= _front_field("2m", var, scen, k_frame) > WET
    out = {}
    for res in L.COARSE:
        f = _front_field(res, var, scen, k_frame)
        cm = np.isfinite(f) & (f > WET)
        del f
        up = L.resample_nearest(cm.astype("float32"), coarse_cells(res),
                                FINE_RES, FINE_SHAPE) > 0.5
        del cm
        rec = _boundary_metrics(ref_mask, up)
        rec["separation"] = _separation("2m", res)
        rec["mask_source"] = ("h_max" if var == "h_max"
                              else f"depth at frame {k_frame}")
        out[f"2m|{res}"] = rec
        print(f"[pattern]   boundary 2m|{res} "
              f"F1={rec['f1']:.3f} MHD={rec['mhd_m']:.2f} m", flush=True)
        del up
    del ref_mask

    diag = {}
    if var == "speed":
        r2 = _load_var("2m", var, scen, k_frame)
        m2 = np.isfinite(r2) & (r2 > WET)
        del r2
        f5 = _load_var("5m", var, scen, k_frame)
        m5 = np.isfinite(f5) & (f5 > WET)
        del f5
        up5 = L.resample_nearest(m5.astype("float32"), coarse_cells("5m"),
                                 FINE_RES, FINE_SHAPE) > 0.5
        diag = _boundary_metrics(m2, up5)
        # How much of the wet footprint survives one and the same absolute speed
        # threshold.  Both widths are counted on the shared 2 m lattice, and the
        # share is taken against the depth-wet cells of the same resolution.
        diag["mask_cells_2m"] = int(m2.sum())
        diag["mask_cells_5m"] = int(up5.sum())
        diag["mask_ratio_5m_to_2m"] = float(int(up5.sum()) / max(int(m2.sum()), 1))
        for tag, res, mask in (("2m", "2m", m2), ("5m", "5m", m5)):
            h = _front_field(res, var, scen, k_frame)
            wet = np.isfinite(h) & (h > WET)
            diag[f"wet_cells_{tag}"] = int(wet.sum())
            diag[f"speed_pass_share_{tag}"] = float(
                int((mask & wet).sum()) / max(int(wet.sum()), 1))
            del h, wet
        diag["note"] = ("front read off the speed field itself with the same "
                        "absolute 0.05 threshold, kept only to show why the "
                        "reported family uses a depth field")
        print(f"[pattern]   absolute-threshold speed front 2m|5m "
              f"MHD={diag['mhd_m']:.2f} m, mask ratio "
              f"{diag['mask_ratio_5m_to_2m']:.3f}, pass share "
              f"{diag['speed_pass_share_2m']:.3f}/{diag['speed_pass_share_5m']:.3f}",
              flush=True)
        del m2, m5, up5
    return out, diag


def pattern_part(dom2: np.ndarray) -> dict:
    """Step four: several distinct families of cross-resolution pattern similarity.

    Every family is computed on the same common support and the same KMeans
    regionalisation as the V-measure step, so the numbers sit beside each other.
    No model is trained and no super-resolution artefact is touched.
    """
    from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                                 cohen_kappa_score, fowlkes_mallows_score,
                                 normalized_mutual_info_score)
    from sklearn.metrics.cluster import homogeneity_completeness_v_measure

    scen = PATTERN_SCEN
    k_frame = frames_from_velocity_json()[scen]
    out = {"scenario": scen, "speed_frame": k_frame, "seed": SEED,
           "families": {
               "chance_corrected": ("Adjusted Rand Index, Fowlkes-Mallows, "
                                    "Normalized and Adjusted Mutual Information, "
                                    "and Cohen kappa beside the V-measure"),
               "areal": ("per-cluster intersection over union after best matching, "
                         "and the total-variation distance of the area shares"),
               "boundary": ("wet-mask boundary F1, boundary IoU and modified "
                            "Hausdorff distance between wet fronts, the front "
                            "being read off a depth field in metres"),
               "autocorrelation": ("Moran's I and Geary's C of the field and of the "
                                   "ordered class labels, plus a join-count statistic"),
               "distance_resolved": ("agreement against the 2 m reference resolved by "
                                     "depth band and by distance to the nearest channel")},
           "support_cells": {}, "variables": {}}

    for var in FLOOD_VARS:
        print(f"[pattern] === {var} ===", flush=True)
        R = regionalise(dom2, var, scen, k_frame)
        m, lab, k = R["m"], R["lab"], R["k"]
        n = int(m.sum())
        out["support_cells"][var] = n
        print(f"[pattern]   support {n} K={k}", flush=True)
        rec = {"k": k, "n": n, "chance_corrected": {}, "areal": {},
               "chance_floor": {}, "front_diagnostic": {},
               "autocorrelation": {}, "join_count": {},
               "distance_resolved": {"value_bands": {}, "dist_water_bands": {}}}

        # ---- family A and B over every resolution pair -------------------
        for i, a in enumerate(ALLRES):
            for b in ALLRES[i + 1:]:
                la, lb = lab[a], lab[b]
                h, c, v = homogeneity_completeness_v_measure(la, lb)
                key = f"{a}|{b}"
                rec["chance_corrected"][key] = {
                    "ari": float(adjusted_rand_score(la, lb)),
                    "ami": float(adjusted_mutual_info_score(la, lb)),
                    "nmi": float(normalized_mutual_info_score(la, lb)),
                    "fm": float(fowlkes_mallows_score(la, lb)),
                    "kappa": float(cohen_kappa_score(la, lb)),
                    "v": float(v), "h": float(h), "c": float(c),
                    "separation": _separation(a, b)}
                rec["areal"][key] = _iou_pairs(la, lb)
                rec["areal"][key]["separation"] = _separation(a, b)

        # ---- family D: autocorrelation of the field and of the labels -----
        ref_ordered = lab["2m"]
        W_ref = rook_adjacency(m)
        rec["autocorrelation"]["class_labels"] = {
            r: moran_geary(lab[r].astype("float64"), W_ref) for r in ALLRES}
        rec["join_count"] = {r: join_count(lab[r], W_ref, k) for r in ALLRES}
        del W_ref

        # per-resolution field autocorrelation on that resolution's own wet cells
        field = {}
        for res in ALLRES:
            f = _load_var(res, var, scen, k_frame)
            mask = np.isfinite(f) & (f > WET)
            Wf = rook_adjacency(mask)
            field[res] = moran_geary(np.clip(f[mask], 0, None), Wf)
            del f, mask, Wf
        rec["autocorrelation"]["field"] = field

        # ---- family C: wet-mask boundary agreement ------------------------
        rec["boundary"], rec["front_diagnostic"] = boundary_part(
            var, scen, k_frame)

        # ---- reference level of the chance-corrected read-outs -------------
        for pair in ("2m|30m", "5m|10m"):
            pa, pb = pair.split("|")
            rec["chance_floor"][pair] = chance_floor(lab[pa], lab[pb])
            print(f"[pattern]   floor {pair} "
                  f"FM={rec['chance_floor'][pair]['fm']:.3f}", flush=True)

        # ---- family E: agreement resolved by depth band and water distance -
        agree = np.zeros(n, dtype=np.int64)
        for res in L.COARSE:
            lr = lab[res]
            lut = np.arange(k, dtype="int64")
            for cl in np.unique(lr):
                lut[cl] = int(np.bincount(ref_ordered[lr == cl],
                                          minlength=k).argmax())
            agree += (lut[lr] == ref_ordered).astype(np.int64)
        stab = agree.astype("float64") / len(L.COARSE)
        edges = [WET, 0.1, 0.3, 0.5, 1.0, 2.0, np.inf]
        v2 = R["val2m"]
        bnds = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (v2 >= lo) & (v2 < hi)
            if sel.sum() > 0:
                bnds.append({"lo": lo, "hi": None if np.isinf(hi) else hi,
                             "n": int(sel.sum()),
                             "mean_stability": float(stab[sel].mean())})
        rec["distance_resolved"]["value_bands"] = bnds
        dw = L.load_static("2m", "Dist_water")
        dw = np.where(np.isfinite(dw) & (dw >= 0), dw, np.nan)
        d = dw[m]
        dedges = [0, 25, 50, 100, 200, 400, 800, np.inf]
        db = []
        for lo, hi in zip(dedges[:-1], dedges[1:]):
            sel = (d >= lo) & (d < hi)
            if sel.sum() > 0:
                db.append({"lo": lo, "hi": None if np.isinf(hi) else hi,
                           "n": int(sel.sum()),
                           "mean_stability": float(stab[sel].mean())})
        rec["distance_resolved"]["dist_water_bands"] = db
        del dw, d, agree, stab, v2

        out["variables"][var] = rec
        del R, m, lab, ref_ordered
        print(f"[pattern]   {var} done", flush=True)
    return out


# ----------------------------------------------------------------------- main
def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    store = {}
    if which in ("cont", "continuous", "all"):
        dom2 = domain_mask_2m()
        store["continuous"] = continuous_part(dom2)
        store["categorical"] = categorical_part(dom2)
        L.save_json("consistency_partial_step1.json",
                    {k: store[k] for k in ("continuous", "categorical")})
        del dom2
    if which in ("flood", "all"):
        store["flood"] = flood_part()
        L.save_json("consistency_partial_step2.json", store["flood"])
    if which in ("vm", "vmeasure", "all"):
        dom2 = domain_mask_2m()
        store["vmeasure"] = vmeasure_part(dom2)
        L.save_json("consistency_partial_step3.json", store["vmeasure"])
        del dom2
    if which in ("pattern", "spatial", "all"):
        dom2 = domain_mask_2m()
        store["spatial_pattern"] = pattern_part(dom2)
        L.save_json("consistency_partial_step4.json", store["spatial_pattern"])
        del dom2
    if which in ("bnd", "boundary"):
        k_frame = frames_from_velocity_json()[PATTERN_SCEN]
        sp = L.load_json("outputs/premodel/consistency_partial_step4.json")
        for var in FLOOD_VARS:
            sp["variables"][var]["boundary"], sp["variables"][var]["front_diagnostic"] = \
                boundary_part(var, PATTERN_SCEN, k_frame)
        L.save_json("consistency_partial_step4.json", sp)
    if which == "all":
        L.save_json("cross_resolution_consistency.json", store)
    if which == "merge":
        merged = {}
        merged.update(L.load_json("outputs/premodel/consistency_partial_step1.json"))
        merged["flood"] = L.load_json("outputs/premodel/consistency_partial_step2.json")
        merged["vmeasure"] = L.load_json("outputs/premodel/consistency_partial_step3.json")
        merged["spatial_pattern"] = L.load_json(
            "outputs/premodel/consistency_partial_step4.json")
        L.save_json("cross_resolution_consistency.json", merged)
    print("done", flush=True)


if __name__ == "__main__":
    main()
