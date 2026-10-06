r"""Stage 2 of the pre-model study: per-tile fields, metrics and predictors.

For every *usable* 480 m tile on every split and every coarse resolution this
script writes

  * the coarse-grid target of the pre-model   (native coarse depth minus the
    area-weighted 2 m truth), the aggregated truth itself, and the coarse input
    stack that a coarse-only model is allowed to see;
  * a per-tile record of agreement metrics and of the terrain / land-use
    predictors used later for the variance decomposition.

Nothing here reads or writes any super-resolution artefact.
"""
from __future__ import annotations

import sys
import time

import numpy as np

import premodel_lib as L

TILE = 480
SCEN_IDX = {s: i for i, s in enumerate(L.SCENARIOS)}
# coarse-resolution channels handed to the pre-model (all from static.nc)
IN_CHANNELS = L.STATIC_CHANNELS
COARSE_CELLS = {"5m": 96, "10m": 48, "20m": 24, "30m": 16}
THRESH = [0.05, 0.30, 1.00]


def main():
    import netCDF4 as nc

    idx = L.load_json("dataset/index/patches_480m.json")["patches"]
    todo = [p for p in idx if p["usable"]]
    print(f"{len(todo)} usable tiles", flush=True)

    recs = []
    for res in L.COARSE:
        n_c = COARSE_CELLS[res]
        f2 = {s: nc.Dataset(L.GRIDS / "2m" / f"flood_{s}.nc") for s in L.SCENARIOS}
        fc = {s: nc.Dataset(L.GRIDS / res / f"flood_{s}.nc") for s in L.SCENARIOS}
        s2 = nc.Dataset(L.GRIDS / "2m" / "static.nc")
        sc = nc.Dataset(L.GRIDS / res / "static.nc")

        store = {}
        for split in ("train", "val", "test"):
            pats = [p for p in todo if p["split"] == split]
            n = len(pats) * len(L.SCENARIOS)
            store[split] = {
                "X": np.zeros((n, len(IN_CHANNELS) + 1, n_c, n_c), np.float32),
                "Y": np.zeros((n, 1, n_c, n_c), np.float32),
                "T": np.zeros((n, 1, n_c, n_c), np.float32),
                "M": np.zeros((n, 1, n_c, n_c), np.uint8),
                "meta": [],
            }
            store[split]["pats"] = pats
        cnt = {k: 0 for k in store}

        t0 = time.time()
        for p in todo:
            iy, ix = p["iy"], p["ix"]
            ys, xs = iy * 240, ix * 240
            h2 = L.clean(f2[L.SCENARIOS[0]]["h_max"][ys:ys + 240, xs:xs + 240])
            st2 = {nm: L.clean(s2[nm][ys:ys + 240, xs:xs + 240]) for nm in IN_CHANNELS}
            valid2 = np.isfinite(st2["DEM"])
            for scen in L.SCENARIOS:
                if scen != L.SCENARIOS[0]:
                    h2 = L.clean(f2[scen]["h_max"][ys:ys + 240, xs:xs + 240])
                agg = L.aggregate(h2, 2, n_c)
                hc = L.clean(fc[scen]["h_max"][iy * n_c:(iy + 1) * n_c,
                                               ix * n_c:(ix + 1) * n_c])
                stc = {nm: L.clean(sc[nm][iy * n_c:(iy + 1) * n_c,
                                          ix * n_c:(ix + 1) * n_c]) for nm in IN_CHANNELS}
                # Building_binary stores its nodata as -15, which the generic cleaner
                # does not catch.  Without this line the channel enters the network and
                # the predictor table as large negative spikes.
                stc["Building_binary"] = np.where(stc["Building_binary"] >= 0,
                                                  stc["Building_binary"], np.nan)
                mask = (np.isfinite(agg) & np.isfinite(hc) & np.isfinite(stc["DEM"]))

                # ---------- per-tile agreement metrics and predictors
                ok = mask
                d = (hc[ok] - agg[ok]).astype("float64")
                wet = ok & (agg > 0.05)
                rec = {
                    "res": res, "ratio": L.RES_M[res] / 2.0, "split": p["split"],
                    "iy": iy, "ix": ix, "scenario": scen,
                    "n_cells": int(ok.sum()),
                    "wet_frac": float(wet.sum() / max(ok.sum(), 1)),
                    "mae": float(np.mean(np.abs(d))) if d.size else float("nan"),
                    "rmse": float(np.sqrt(np.mean(d * d))) if d.size else float("nan"),
                    "bias": float(np.mean(d)) if d.size else float("nan"),
                    "volume_rel": float((np.sum(hc[ok]) - np.sum(agg[ok])) /
                                        max(np.sum(agg[ok]), 1e-9)),
                    "csi005": L.csi(hc, agg, 0.05),
                    "csi030": L.csi(hc, agg, 0.30),
                    "csi100": L.csi(hc, agg, 1.00),
                    "pearson_wet": L.pearson(np.where(wet, hc, np.nan),
                                             np.where(wet, agg, np.nan)),
                    "spearman_wet": L.spearman(np.where(wet, hc, np.nan),
                                               np.where(wet, agg, np.nan)),
                    "peak_truth": float(np.nanmax(agg[ok])) if ok.any() else float("nan"),
                    "max_depth_ratio": float(np.nanmax(hc[ok]) /
                                             max(np.nanmax(agg[ok]), 1e-6)) if ok.any() else float("nan"),
                }
                # terrain / land-use predictors, all computed from static.nc
                sl = stc["Slope"][ok]
                rec.update({
                    "mean_slope": float(np.nanmean(sl)),
                    "median_slope": float(np.nanmedian(sl)),
                    "mean_slope_2m": float(np.nanmean(st2["Slope"][valid2]))
                    if valid2.any() else float("nan"),
                    "mean_manning": float(np.nanmean(stc["Manning"][ok])),
                    "mean_dist_water": float(np.nanmean(stc["Dist_water"][ok])),
                    "min_dist_water": float(np.nanmin(stc["Dist_water"][ok])),
                    "mean_twi": float(np.nanmean(stc["Twi"][ok])),
                    "mean_tpi": float(np.nanmean(stc["Tpi"][ok])),
                    "building_frac": float(np.nanmean(
                        (stc["Building_binary"][ok] > 0).astype("float64"))),
                    "water_frac": float(np.nanmean((stc["Landuse"] == 7)[ok])),
                    "impervious_frac": float(np.nanmean(
                        np.isin(stc["Landuse"][ok], [2, 3]))),
                    "green_frac": float(np.nanmean(
                        np.isin(stc["Landuse"][ok], [4, 6]))),
                    "landuse_entropy": _entropy(stc["Landuse"], ok),
                    "wet_frac_2m": float(np.nanmean((h2 > 0.05)[valid2])),
                    "mean_depth_2m": float(np.nanmean(h2[valid2])) if valid2.any() else float("nan"),
                })
                # fine-scale terrain roughness aggregated into the cell
                dem2v = np.where(np.isfinite(st2["DEM"]), st2["DEM"].astype("float64"), np.nan)
                d2 = L.aggregate(dem2v.astype(np.float32), 2, n_c).astype("float64")
                sq = L.aggregate((dem2v ** 2).astype(np.float32), 2, n_c).astype("float64")
                within = np.sqrt(np.clip(sq - d2 ** 2, 0, None))[ok]
                rec["within_cell_dem_std"] = float(np.nanmean(within))
                wmask = np.isfinite(st2["Landuse"]) & (st2["Landuse"] == 7)
                wfrac = L.aggregate_mask(wmask, 2, n_c, "mean")[ok]
                rec["water_frac_fine"] = float(np.nanmean(wfrac))
                if not np.isfinite(rec.get("mae", np.nan)):
                    continue
                recs.append(rec)

                # ---------- store tensors
                s = store[p["split"]]
                j = cnt[p["split"]]
                s["X"][j, 0] = hc
                for ci, nm in enumerate(IN_CHANNELS):
                    s["X"][j, ci + 1] = stc[nm]
                s["X"][j] = np.nan_to_num(s["X"][j], nan=0.0)
                s["Y"][j, 0] = np.nan_to_num(hc - agg, nan=0.0)
                s["T"][j, 0] = np.nan_to_num(agg, nan=0.0)
                s["M"][j, 0] = mask.astype(np.uint8)
                s["meta"].append({"iy": iy, "ix": ix, "scenario": scen,
                                  "split": p["split"]})
                cnt[p["split"]] += 1
        print(f"[tiles] {res} built in {time.time()-t0:.1f}s", flush=True)

        for split in ("train", "val", "test"):
            s = store[split]
            k = cnt[split]
            np.savez_compressed(
                L.OUT / f"tile_{res}_{split}.npz",
                X=s["X"][:k], Y=s["Y"][:k], T=s["T"][:k], M=s["M"][:k],
                meta=np.array([(m["iy"], m["ix"], SCEN_IDX[m["scenario"]])
                               for m in s["meta"]], dtype=np.int16))
            print(f"  wrote tile_{res}_{split}.npz  X={s['X'][:k].shape}", flush=True)
        for ds in list(f2.values()) + list(fc.values()):
            ds.close()
        s2.close()
        sc.close()
    L.save_json("tile_metrics.json", recs)


def _entropy(landuse, ok):
    v = landuse[ok]
    if v.size == 0:
        return float("nan")
    _, c = np.unique(v, return_counts=True)
    p = c / c.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


if __name__ == "__main__":
    main()
