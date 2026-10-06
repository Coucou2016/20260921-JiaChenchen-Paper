r"""Stage 1 of the pre-model study: full-domain coarse-vs-fine error accounting.

Produces, without touching the super-resolution model:

  * native coarse HiPIMS depth  vs  area-weighted 2 m truth aggregated to the
    same coarse lattice                     -> representation + numerical error
  * the aggregated truth round-tripped back to 2 m by nearest upsampling vs the
    true 2 m depth                          -> aggregation (information-loss) error
  * terrain-distortion statistics that only use channels present in static.nc
  * cross-resolution agreement of depth and of the derived speed field

Usage:  python scripts/premodel_01_fields.py [flood|terrain|velocity|all]
"""
from __future__ import annotations

import sys
import time

import numpy as np

import premodel_lib as L

THRESHOLDS = [0.05, 0.30, 1.00]
DEPTH_BINS = [(0.0, 0.05), (0.05, 0.30), (0.30, 0.50), (0.50, 1.00),
              (1.00, 1.50), (1.50, 2.00), (2.00, 3.00), (3.00, 99.0)]
SLOPE_BINS = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.60), (0.60, 9.0)]


DOMAIN_Y, DOMAIN_X = 23040, 12960
FINE_SHAPE = (11520, 6480)


def coarse_cells(res: str):
    r = L.RES_M[res]
    return int(DOMAIN_Y / r), int(DOMAIN_X / r)


def bin_stats(pred, truth, edges, labels):
    """MAE / RMSE / bias of pred-truth inside strata of the truth itself."""
    out = []
    ok = np.isfinite(pred) & np.isfinite(truth)
    for (lo, hi), lab in zip(edges, labels):
        sel = ok & (truth >= lo) & (truth < hi)
        n = int(sel.sum())
        if n == 0:
            out.append({"bin": lab, "n": 0})
            continue
        d = (pred[sel] - truth[sel]).astype("float64")
        out.append({"bin": lab, "n": n,
                    "mae": float(np.mean(np.abs(d))),
                    "rmse": float(np.sqrt(np.mean(d * d))),
                    "bias": float(np.mean(d)),
                    "mean_truth": float(np.mean(truth[sel]))})
    return out


def bin_stats_by(pred, truth, cond, edges, labels, wet=0.05):
    """MAE / RMSE / bias of pred-truth inside strata of a conditioning field.

    Strata are formed on ``cond`` while the metric stays pred minus truth, so a
    slope stratum still reports the error of the coarse run rather than the
    difference between a depth and a slope.  Both the all-cell and the wet-cell
    versions are stored, because steeper ground is mostly dry and the all-cell
    average is therefore dominated by dry cells.
    """
    out = []
    ok = np.isfinite(pred) & np.isfinite(truth) & np.isfinite(cond)
    for (lo, hi), lab in zip(edges, labels):
        sel = ok & (cond >= lo) & (cond < hi)
        n = int(sel.sum())
        if n == 0:
            out.append({"bin": lab, "n": 0})
            continue
        d = (pred[sel] - truth[sel]).astype("float64")
        sub = sel & (truth > wet)
        dw = (pred[sub] - truth[sub]).astype("float64")
        out.append({"bin": lab, "n": n,
                    "mae": float(np.mean(np.abs(d))),
                    "rmse": float(np.sqrt(np.mean(d * d))),
                    "bias": float(np.mean(d)),
                    "mean_cond": float(np.mean(cond[sel])),
                    "mean_truth": float(np.mean(truth[sel])),
                    "wet_frac": float(np.mean(truth[sel] > wet)),
                    "n_wet": int(sub.sum()),
                    "mae_wet": float(np.mean(np.abs(dw))) if sub.any() else None,
                    "rmse_wet": float(np.sqrt(np.mean(dw * dw))) if sub.any() else None,
                    "bias_wet": float(np.mean(dw)) if sub.any() else None})
    return out


def flood_part():
    summary = {"thresholds": THRESHOLDS,
               "depth_bins": [f"{a}-{b}" for a, b in DEPTH_BINS],
               "slope_bins": [f"{a}-{b}" for a, b in SLOPE_BINS],
               "resolutions": {}}
    # coarse slope comes from the *native coarse* static file, which is exactly
    # what a coarse run can see without any fine information.
    for scen in L.SCENARIOS:
        t0 = time.time()
        h2 = L.load_flood("2m", scen, "h_max")
        print(f"[flood] {scen}: read 2 m h_max {h2.shape} in {time.time()-t0:.1f}s",
              flush=True)
        summary["resolutions"][scen] = {}
        for res in L.COARSE:
            n = coarse_cells(res)
            t1 = time.time()
            agg = L.aggregate(h2, 2, n)
            native = L.load_flood(res, scen, "h_max")
            slope_c = L.load_static(res, "Slope")
            ok = np.isfinite(agg) & np.isfinite(native)
            m = L.numeric_metrics(native, agg)
            m["csi"] = {f"{t:.2f}": L.csi(native, agg, t) for t in THRESHOLDS}
            m["f1"] = {f"{t:.2f}": L.f1(native, agg, t) for t in THRESHOLDS}
            m["iou"] = m["csi"]  # identical to CSI for a binary mask
            m["pearson_wet"] = L.pearson(np.where(agg > 0.05, native, np.nan),
                                         np.where(agg > 0.05, agg, np.nan))
            m["spearman_wet"] = L.spearman(np.where(agg > 0.05, native, np.nan),
                                           np.where(agg > 0.05, agg, np.nan))
            m["ssim"] = L.ssim(native, agg)
            m["by_depth"] = bin_stats(native, agg, DEPTH_BINS,
                                      [f"{a}-{b}" for a, b in DEPTH_BINS])
            smask = np.where(np.isfinite(slope_c), slope_c, np.nan)
            m["by_slope"] = bin_stats_by(native, agg, smask, SLOPE_BINS,
                                         [f"{a}-{b}" for a, b in SLOPE_BINS])
            # inundation area at each threshold, in km^2
            cell_km2 = (L.RES_M[res] / 1000.0) ** 2
            m["area_km2"] = {f"{t:.2f}": {
                "truth": float(np.count_nonzero(agg > t) * cell_km2),
                "native": float(np.count_nonzero(native > t) * cell_km2)}
                for t in THRESHOLDS}

            # ---- error decomposition measured back on the 2 m lattice.
            #      total = NN(native) - truth = [NN(native)-NN(agg)] + [NN(agg)-truth]
            #      i.e.  simulation term  +  aggregation (information-loss) term.
            up = L.resample_nearest(agg, n, 2, FINE_SHAPE)
            upn = L.resample_nearest(native, n, 2, FINE_SHAPE)
            fin = np.isfinite(h2) & np.isfinite(upn)
            e_tot = (upn[fin] - h2[fin]).astype("float64")
            e_sim = (upn[fin] - up[fin]).astype("float64")
            e_agg = (up[fin] - h2[fin]).astype("float64")
            r_tot = float(np.sqrt(np.mean(e_tot ** 2)))
            r_sim = float(np.sqrt(np.mean(e_sim ** 2)))
            r_agg = float(np.sqrt(np.mean(e_agg ** 2)))
            cross = float(np.mean(e_sim * e_agg))
            m["error_decomp_2m"] = {
                "note": ("total = simulation term + aggregation term measured on the "
                         "2 m lattice with nearest upsampling of both coarse fields"),
                "n": int(fin.sum()),
                "rmse_total": r_tot, "mae_total": float(np.mean(np.abs(e_tot))),
                "bias_total": float(np.mean(e_tot)),
                "rmse_simulation": r_sim, "mae_simulation": float(np.mean(np.abs(e_sim))),
                "rmse_aggregation": r_agg, "mae_aggregation": float(np.mean(np.abs(e_agg))),
                "cross_cov": cross,
                "share_simulation_var": float(r_sim ** 2 / (r_tot ** 2)),
                "share_aggregation_var": float(r_agg ** 2 / (r_tot ** 2)),
                "share_cross_var": float(2 * cross / (r_tot ** 2)),
                "identity_check": float(r_sim ** 2 + r_agg ** 2 + 2 * cross - r_tot ** 2),
                "csi_total@0.05": L.csi(np.where(fin, upn, np.nan),
                                        np.where(fin, h2, np.nan), 0.05),
                "csi_agg@0.05": L.csi(np.where(fin, up, np.nan),
                                      np.where(fin, h2, np.nan), 0.05),
            }
            m["rmse_rel_mean_truth"] = float(m["rmse"] / max(m["mean_truth"], 1e-9))
            del upn, up

            m["ratio"] = L.RES_M[res] / 2.0
            m["seconds"] = round(time.time() - t1, 1)
            summary["resolutions"][scen][res] = m
            print(f"[flood] {scen} {res}: rmse={m['rmse']:.4f} bias={m['bias']:+.4f} "
                  f"csi0.05={m['csi']['0.05']:.3f} ratio={m['ratio']} "
                  f"({m['seconds']}s)", flush=True)

            np.savez_compressed(
                L.OUT / f"map_{scen}_{res}.npz",
                truth=agg.astype(np.float16), native=native.astype(np.float16),
                err=(native - agg).astype(np.float16))
            del agg, native
        del h2
    L.save_json("flood_error.json", summary)


def terrain_part():
    """Terrain distortion of the coarse lattices, restricted to static.nc channels."""
    t0 = time.time()
    dem2 = L.load_static("2m", "DEM")
    slope2 = L.load_static("2m", "Slope")
    mann2 = L.load_static("2m", "Manning")
    land2 = L.load_static("2m", "Landuse")
    distw2 = L.load_static("2m", "Dist_water")
    build2 = L.load_static("2m", "Building_binary")
    print(f"[terrain] read 2 m static in {time.time()-t0:.1f}s", flush=True)

    # Study domain: exactly the 2 m cells the hydrodynamic run resolves.  The
    # static rasters cover a larger rectangle (sea and a nodata apron), so every
    # fraction below is normalised on this mask rather than on the full grid.
    import netCDF4 as nc
    with nc.Dataset(L.GRIDS / "2m" / f"flood_{L.SCENARIOS[0]}.nc") as fh:
        dom = np.isfinite(L.clean(fh["h_max"][:]))
    print(f"[terrain] domain cells {int(dom.sum())} of {dom.size}", flush=True)

    out = {"channels_available": L.STATIC_CHANNELS,
           "water_mask_definition": "Landuse == 7",
           "domain_cells": int(dom.sum()),
           "grid_cells": int(dom.size),
           "landuse_nodata": -9999.0,
           "building_nodata": -15,
           "resolutions": {}}
    wmask = dom & np.isfinite(land2) & (land2 == 7)
    bvalid = dom & (build2 >= 0)
    out["fine"] = {
        "dem_mean": float(np.nanmean(dem2[dom])),
        "dem_std": float(np.nanstd(dem2[dom])),
        "slope_mean": float(np.nanmean(slope2[dom])),
        "slope_median": float(np.nanmedian(slope2[dom])),
        "water_frac_of_valid": float(wmask.sum() / dom.sum()),
        "building_frac_of_valid": float(np.mean(build2[bvalid] > 0)),
    }

    # channel width from the 2 m water mask (one global distance transform)
    t1 = time.time()
    width = L.local_width(wmask)
    print(f"[terrain] distance transform in {time.time()-t1:.1f}s", flush=True)
    wv = width[wmask]
    out["channel_width_2m"] = {
        "n_water_pixels": int(wmask.sum()),
        "p10": float(np.percentile(wv, 10)),
        "p25": float(np.percentile(wv, 25)),
        "median": float(np.percentile(wv, 50)),
        "p75": float(np.percentile(wv, 75)),
        "p90": float(np.percentile(wv, 90)),
        "median_below_500m": float(np.median(wv[wv < 500])) if (wv < 500).any() else float("nan"),
    }

    dem2f = np.where(dom, dem2, np.nan)
    sq = np.where(dom, dem2.astype("float64") ** 2, np.nan)
    # distribution of the 2 m channel width, shared by every resolution
    wv_all = width[wmask]
    hist_w, edges_w = np.histogram(np.clip(wv_all, 0, 200), bins=100, range=(0, 200))
    out["channel_width_hist"] = {"counts": hist_w.tolist(),
                                "edges": edges_w.tolist(),
                                "n_total": int(wmask.sum())}
    for res in L.COARSE:
        n = coarse_cells(res)
        dem_c = L.load_static(res, "DEM")
        slope_c = L.load_static(res, "Slope")
        mann_c = L.load_static(res, "Manning")
        dem_agg = L.aggregate(dem2f.astype("float32"), 2, n)
        dem2_agg = L.aggregate(sq.astype("float32"), 2, n)
        slope_agg = L.aggregate(np.where(dom, slope2, np.nan).astype("float32"), 2, n)
        mann_agg = L.aggregate(np.where(dom, mann2, np.nan).astype("float32"), 2, n)
        with np.errstate(invalid="ignore"):
            within_var = dem2_agg - dem_agg ** 2
        within_std = np.sqrt(np.clip(within_var, 0, None))
        mn, mx = L.block_minmax(dem2f, n, 2)
        okd = np.isfinite(dem_agg) & np.isfinite(dem_c)
        dom_c = L.aggregate_mask(dom, 2, n, "mean")
        okd = okd & (np.nan_to_num(dom_c, nan=0.0) > 0.5)
        sink = (dem_agg - mn)                      # depth of the deepest lost hollow
        relief = (mx - mn)
        wcell = L.aggregate(np.where(dom, wmask.astype("float32"), np.nan), 2, n)
        bcell = L.aggregate(np.where(bvalid, (build2 > 0).astype("float32"), np.nan),
                            2, n)
        # land-use composition heterogeneity: fraction of the cell taken by its
        # dominant land-use class, computed class by class and ignoring the
        # -9999 nodata code that fills the apron outside the study domain.
        classes = [int(c) for c in np.unique(land2)
                   if np.isfinite(c) and int(c) != -9999]
        frac = {c: L.aggregate(np.where(dom, (land2 == c).astype("float32"), np.nan),
                               2, n)
                for c in classes}
        stack = np.stack([np.nan_to_num(f, nan=0.0) for f in frac.values()], axis=0)
        dominant = np.nanmax(stack, axis=0)
        imperv = np.nan_to_num(frac.get(2, 0)) + np.nan_to_num(frac.get(3, 0))
        out["resolutions"][res] = {
            "ratio": L.RES_M[res] / 2.0,
            "n_cells": int(n[0] * n[1]),
            "cell_shape": [int(n[0]), int(n[1])],
            "dem_rmse_vs_agg": float(np.sqrt(np.nanmean((dem_c[okd] - dem_agg[okd]) ** 2))),
            "dem_mae_vs_agg": float(np.nanmean(np.abs(dem_c[okd] - dem_agg[okd]))),
            "dem_corr_vs_agg": L.pearson(dem_c, dem_agg),
            "within_cell_std_mean": float(np.nanmean(within_std[okd])),
            "within_cell_std_median": float(np.nanmedian(within_std[okd])),
            "within_cell_std_p90": float(np.nanpercentile(within_std[okd], 90)),
            "within_cell_var_mean": float(np.nanmean(np.clip(within_var, 0, None)[okd])),
            "relief_mean": float(np.nanmean(relief[okd])),
            "relief_p90": float(np.nanpercentile(relief[okd], 90)),
            "sink_depth_mean": float(np.nanmean(sink[okd])),
            "sink_frac_gt_0p1": float(np.nanmean((sink[okd] > 0.10))),
            "sink_frac_gt_0p5": float(np.nanmean((sink[okd] > 0.50))),
            "slope_native_mean": float(np.nanmean(slope_c[okd])),
            "slope_agg_mean": float(np.nanmean(slope_agg[okd])),
            "slope_ratio": float(np.nanmean(slope_c[okd])
                                 / np.nanmean(slope_agg[okd])),
            "slope_native_median": float(np.nanmedian(slope_c[okd])),
            "slope_agg_median": float(np.nanmedian(slope_agg[okd])),
            "manning_native_mean": float(np.nanmean(mann_c[okd])),
            "manning_agg_mean": float(np.nanmean(mann_agg[okd])),
            "water_frac_mean": float(np.nanmean(wcell[okd])),
            "channel_minus": {
                str(w): float(np.nanmean(wcell[okd] < w)) for w in (0.25, 0.5, 0.75)},
            "building_frac_mean": float(np.nanmean(bcell[okd])),
            "impervious_frac_mean": float(np.nanmean(imperv[okd])),
            "dominant_class_frac_mean": float(np.nanmean(dominant[okd])),
            "dist_water_agg_mean": float(np.nanmean(
                L.aggregate(np.where(dom, distw2, np.nan).astype("float32"), 2, n))),
            "width_ratio_median": float(np.median(wv) / L.RES_M[res]),
            "water_pixels_narrower_than_cell": float(np.mean(wv < L.RES_M[res])),
            "within_std_hist": np.histogram(
                np.clip(within_std[okd], 0, 20), bins=60, range=(0, 20))[0].tolist(),
            "sink_hist": np.histogram(
                np.clip(sink[okd], 0, 10), bins=60, range=(0, 10))[0].tolist(),
        }
        print(f"[terrain] {res}: within_std={out['resolutions'][res]['within_cell_std_mean']:.3f} "
              f"slope_ratio={out['resolutions'][res]['slope_ratio']:.3f} "
              f"sink>0.1={out['resolutions'][res]['sink_frac_gt_0p1']:.3f}", flush=True)
    L.save_json("terrain_distortion.json", out)


def velocity_part():
    """Cross-resolution agreement of the depth-averaged speed derived from hux/hvy."""
    out = {"note": "speed = |(hux,hvy)|/h, evaluated at the hour with the largest "
                   "domain water volume; these are derived quantities, not stored fields.",
           "scenarios": {}}
    for scen in L.SCENARIOS:
        h_all = L.load_flood("2m", scen, "h", times=slice(1, 7))
        vol = [float(np.nansum(h_all[i])) for i in range(h_all.shape[0])]
        k = int(np.argmax(vol)) + 1
        print(f"[velocity] {scen}: hourly volumes {['%.3g' % v for v in vol]} -> frame {k}",
              flush=True)
        del h_all
        h2 = L.load_flood("2m", scen, "h", times=k)
        ux2 = L.load_flood("2m", scen, "hux", times=k)
        vy2 = L.load_flood("2m", scen, "hvy", times=k)
        with np.errstate(invalid="ignore", divide="ignore"):
            sp2 = np.sqrt(ux2 ** 2 + vy2 ** 2) / np.where(h2 > 0.02, h2, np.nan)
        sp2 = np.clip(sp2, 0, 20).astype(np.float32)
        rec = {"frame": k, "resolutions": {}}
        for res in L.COARSE:
            n = coarse_cells(res)
            hc = L.load_flood(res, scen, "h", times=k)
            uxc = L.load_flood(res, scen, "hux", times=k)
            vyc = L.load_flood(res, scen, "hvy", times=k)
            with np.errstate(invalid="ignore", divide="ignore"):
                spc = np.sqrt(uxc ** 2 + vyc ** 2) / np.where(hc > 0.02, hc, np.nan)
            spc = np.clip(spc, 0, 20).astype(np.float32)
            sp_agg = L.aggregate(sp2, 2, n)
            wet = np.isfinite(sp_agg) & np.isfinite(spc) & (sp_agg > 0.05)
            rec["resolutions"][res] = {
                "ratio": L.RES_M[res] / 2.0,
                "n": int(wet.sum()),
                "pearson": L.pearson(np.where(wet, spc, np.nan),
                                     np.where(wet, sp_agg, np.nan)),
                "spearman": L.spearman(np.where(wet, spc, np.nan),
                                       np.where(wet, sp_agg, np.nan)),
                "mean_native": float(np.nanmean(spc[wet])) if wet.any() else float("nan"),
                "mean_agg": float(np.nanmean(sp_agg[wet])) if wet.any() else float("nan"),
                "median_native": float(np.nanmedian(spc[wet])) if wet.any() else float("nan"),
                "median_agg": float(np.nanmedian(sp_agg[wet])) if wet.any() else float("nan"),
            }
            print(f"[velocity] {scen} {res}: r={rec['resolutions'][res]['pearson']:.3f} "
                  f"mean {rec['resolutions'][res]['mean_native']:.3f} vs "
                  f"{rec['resolutions'][res]['mean_agg']:.3f}", flush=True)
            del hc, uxc, vyc, spc, sp_agg
        out["scenarios"][scen] = rec
        del h2, ux2, vy2, sp2
    L.save_json("velocity_correlation.json", out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("flood", "all"):
        flood_part()
    if which in ("terrain", "all"):
        terrain_part()
    if which in ("velocity", "all"):
        velocity_part()
