"""Stage 7 of the pre-model study: inputs for the readability rewrite of chapter 2.

Produces, from artefacts that already exist (no global re-run, no training):

  * cell_example.json      one real 10 m block, its 25 two-metre sub-cells, the
                           area-weighted aggregate and the coarse HiPIMS value
  * factor_contrasts.json  low-group vs high-group error contrast for six static
                           factors, computed from the per-tile metrics

Usage:  python scripts/premodel_07_readability.py [cell|factors|all]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "outputs" / "premodel"
GRIDS = ROOT / "dataset" / "grids"
RES = ["5m", "10m", "20m", "30m"]

FACTORS = [
    ("mean_slope", "单元平均坡度", "平缓", "陡峭"),
    ("within_cell_dem_std", "单元内二米高程标准差", "起伏小", "起伏大"),
    ("building_frac", "瓦片内建筑像元占比", "建筑少", "建筑多"),
    ("impervious_frac", "瓦片内不透水面占比", "硬化少", "硬化多"),
    ("mean_dist_water", "瓦片到水体平均距离", "靠河道", "远离河道"),
    ("landuse_entropy", "瓦片土地利用混合度", "单一", "混杂"),
]


def save_json(name, obj):
    (PRE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print("wrote", PRE / name, flush=True)


# ----------------------------------------------------------------- cell example
def cell_part():
    scen, res = "100a", "10m"
    d = np.load(PRE / f"map_{scen}_{res}.npz")
    agg = d["truth"].astype("float64")
    nat = d["native"].astype("float64")
    step = 5                                   # 10 m / 2 m
    h, w = agg.shape

    ok = np.isfinite(agg) & np.isfinite(nat) & (agg > 0.05) & (nat > 0.05)
    inner = np.ones_like(ok)
    inner[:2] = inner[-2:] = False
    inner[:, :2] = inner[:, -2:] = False
    wet = ok & inner
    med_diff = float(np.median(np.abs(nat[wet] - agg[wet])))

    # one representative block: about one metre deep, difference close to the
    # all-domain median for a wet cell
    band = wet & (agg >= 0.8) & (agg <= 1.2)
    idx = np.argwhere(band)
    if idx.size == 0:                          # defensive fallback
        band = wet & (agg >= 0.5)
        idx = np.argwhere(band)
    pick = idx[np.argmin(np.abs(np.abs(nat[band] - agg[band]) - med_diff))]
    iy, ix = int(pick[0]), int(pick[1])

    import netCDF4 as nc
    with nc.Dataset(GRIDS / "2m" / f"flood_{scen}.nc") as fh:
        sub = np.array(fh["h_max"][iy * step:(iy + 1) * step,
                                   ix * step:(ix + 1) * step], dtype="float64")
    sub_clean = np.where(np.isfinite(sub), sub, np.nan)
    area_agg = float(np.nanmean(sub_clean))     # equal-area sub-cells

    out = {
        "scenario": scen,
        "resolution_m": 10,
        "sub_cells_per_side": step,
        "selection_rule": (
            "在聚合水深介于 0.8 与 1.2 米之间、且四周留有两格边距的十米单元中，"
            "取差值最接近全域湿区差值中位数的那一块"),
        "domain_median_abs_diff_wet_m": med_diff,
        "n_wet_cells": int(wet.sum()),
        "row": iy, "col": ix,
        "row_span_2m": [iy * step, (iy + 1) * step],
        "col_span_2m": [ix * step, (ix + 1) * step],
        "sub_max_m": float(np.nanmax(sub_clean)),
        "sub_min_m": float(np.nanmin(sub_clean)),
        "sub_mean_m": float(np.nanmean(sub_clean)),
        "sub_std_m": float(np.nanstd(sub_clean)),
        "sub_range_m": float(np.nanmax(sub_clean) - np.nanmin(sub_clean)),
        "area_weighted_aggregate_m": area_agg,
        "coarse_native_m": float(nat[iy, ix]),
        "native_minus_aggregate_m": float(nat[iy, ix] - area_agg),
        "grid_aggregate_m": float(agg[iy, ix]),
        "grid_native_m": float(nat[iy, ix]),
        "grid_diff_m": float(nat[iy, ix] - agg[iy, ix]),
        "sub_values_m": [[float(v) for v in r] for r in sub_clean],
        "note": (
            "sub_values_m 的 25 个值就是这一块十米单元内部的全部两米水深，"
            "面积加权平均即聚合真值；coarse_native_m 是粗网格水动力程序在同一个位置"
            "自己算出的水深"),
    }
    save_json("cell_example.json", out)
    print(f"[cell] block ({iy},{ix}) agg={area_agg:.4f} native={nat[iy, ix]:.4f} "
          f"diff={nat[iy, ix] - area_agg:+.4f} sub [{np.nanmin(sub_clean):.3f}, "
          f"{np.nanmax(sub_clean):.3f}]", flush=True)


# -------------------------------------------------------------- factor tables
def _stats(v):
    v = np.asarray(v, dtype="float64")
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return {"n": int(v.size), "mean": float(v.mean()),
            "median": float(np.median(v)), "std": float(v.std())}


def factors_part():
    recs = json.loads((PRE / "tile_metrics.json").read_text(encoding="utf-8"))
    out = {"metric_definition": {
        "unit": "瓦片（480 米见方），每档网格与每个降雨情景各算一次",
        "mae": "逐瓦片平均绝对误差，单位米，衡量该瓦片内粗网格模拟与聚合真值的平局差别",
        "rmse": "逐瓦片均方根误差，单位米",
        "csi005": "逐瓦片五厘米阈值下的临界成功指数，取值零到一，越大越好",
        "groups": "按该因素的取值把瓦片分成三档，只比较最低的三分之一与最高的三分之一",
    }, "resolutions": {}}

    for res in RES:
        sub = [r for r in recs if r["res"] == res]
        rec = {"n_tiles": len(sub), "factors": {}}
        for key, cn, lo_lab, hi_lab in FACTORS:
            xs = np.array([r.get(key, np.nan) for r in sub], dtype="float64")
            good = np.isfinite(xs)
            if good.sum() < 30:
                rec["factors"][key] = {"insufficient": True}
                continue
            q1, q2 = np.percentile(xs[good], [100 / 3., 200 / 3.])
            low = np.array([good[i] and xs[i] <= q1 for i in range(len(sub))])
            high = np.array([good[i] and xs[i] >= q2 for i in range(len(sub))])
            entry = {"cn": cn, "low_label": lo_lab, "high_label": hi_lab,
                     "low_cut": float(q1), "high_cut": float(q2),
                     "pearson_with_mae": _pearson(xs, np.array(
                         [r.get("mae", np.nan) for r in sub], dtype="float64")),
                     "pearson_with_csi005": _pearson(xs, np.array(
                         [r.get("csi005", np.nan) for r in sub], dtype="float64")),
                     "pearson_with_depth": _pearson(xs, np.array(
                         [r.get("mean_depth_2m", np.nan) for r in sub],
                         dtype="float64"))}
            for tag, sel in (("low", low), ("mid", (~low) & (~high) & good),
                             ("high", high)):
                block = [sub[i] for i in range(len(sub)) if sel[i]]
                entry[tag] = {
                    "factor_mean": float(np.mean([b[key] for b in block])),
                    "mean_depth_2m": _stats([b.get("mean_depth_2m")
                                             for b in block]),
                    "mae": _stats([b.get("mae") for b in block]),
                    "rmse": _stats([b.get("rmse") for b in block]),
                    "csi005": _stats([b.get("csi005") for b in block]),
                }
            lm = entry["low"]["mae"]["mean"]
            hm = entry["high"]["mae"]["mean"]
            entry["mae_ratio_high_over_low"] = float(hm / lm) if lm else float("nan")
            entry["mae_gap_m"] = float(hm - lm)
            lc = entry["low"]["csi005"]["mean"]
            hc = entry["high"]["csi005"]["mean"]
            entry["csi_gap"] = float(hc - lc)
            rec["factors"][key] = entry
        out["resolutions"][res] = rec
        print(f"[factors] {res}: {len(sub)} tiles", flush=True)
    save_json("factor_contrasts.json", out)


def _pearson(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return None
    a, b = a[ok], b[ok]
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


# ------------------------------------------------- cell-level factor contrast
CELL_FACTORS = [
    ("Slope", "单元坡度", None),
    ("Building_binary", "单元建筑覆盖率", None),
    ("Dist_water", "单元到水体距离", None),
    ("Manning", "单元曼宁系数", None),
]


def cells_part():
    """Cell-level contrast: which coarse-grid conditions make the error bigger.

    Uses only artefacts that already exist.  The error field comes from the stage-1
    maps, the conditioning variables come from the native coarse static files, so
    no simulation and no two-metre pass is repeated here.
    """
    import netCDF4 as nc

    import premodel_lib as L

    out = {"note": (
        "在单元一级比较，条件变量取自粗网格静态文件本身，也就是粗网格模型能直接看到的量。"
        "误差为粗网格模拟水深减去聚合真值。每组给出样本量、平均绝对误差、"
        "该组聚合真值的平均水深，以及两者的皮尔逊相关。"),
        "coverage_note": (
            "粗网格静态文件里的 Building_binary 是零一标识，直接三等分求分位数会得到"
            "零与一两个断点，中间那一组因此为空。单元建筑覆盖率改由二米建筑掩膜按面积"
            "加权聚合到粗网格得到，是零到一之间的连续量。三个档按物理阈值划分，"
            "覆盖率等于零为无建筑，介于零与一之间为部分覆盖，等于一为满覆盖。"),
        "resolutions": {}}
    bld2 = L.load_static("2m", "Building_binary")
    bld2 = np.where(bld2 >= 0, (bld2 > 0).astype("float32"), np.nan)
    for res in RES:
        out["resolutions"][res] = {}
        st = {}
        with nc.Dataset(GRIDS / res / "static.nc") as fh:
            for name, _, _ in CELL_FACTORS:
                st[name] = np.array(fh[name][:], dtype="float32")
            land = np.array(fh["Landuse"][:], dtype="float32")
        st["Building_binary"] = L.aggregate(bld2, 2, st["Building_binary"].shape)
        for scen in ("20a", "100a"):
            d = np.load(PRE / f"map_{scen}_{res}.npz")
            agg = d["truth"].astype("float64")
            nat = d["native"].astype("float64")
            err = nat - agg
            ok = np.isfinite(err) & (agg > 0.05)
            rec = {"n_wet_cells": int(ok.sum()), "factors": {}}
            for name, cn, _ in CELL_FACTORS:
                v = st[name]
                good = ok & np.isfinite(v)
                if good.sum() < 1000:
                    rec["factors"][name] = {"insufficient": True}
                    continue
                if name == "Building_binary":
                    # the static flag is binary, so the three levels are physical
                    # coverage thresholds rather than quantile thirds
                    q1, q2 = 0.0, 1.0
                    lo_sel = good & (v <= 1e-9)
                    mid_sel = good & (v > 1e-9) & (v < 1 - 1e-9)
                    hi_sel = good & (v >= 1 - 1e-9)
                else:
                    q1, q2 = np.percentile(v[good], [100 / 3., 200 / 3.])
                    lo_sel = good & (v <= q1)
                    mid_sel = good & (v > q1) & (v < q2)
                    hi_sel = good & (v >= q2)
                entry = {"cn": cn, "low_cut": float(q1), "high_cut": float(q2),
                         "pearson_with_abs_err": _pearson(
                             v[good], np.abs(err[good])),
                         "pearson_with_bias": _pearson(v[good], err[good])}
                for tag, sel in (("low", lo_sel), ("mid", mid_sel),
                                 ("high", hi_sel)):
                    e = err[sel]
                    if e.size == 0:
                        entry[tag] = {"n": 0, "mae": float("nan"),
                                      "rmse": float("nan"), "bias": float("nan"),
                                      "mean_truth_m": float("nan")}
                        continue
                    entry[tag] = {
                        "n": int(sel.sum()),
                        "mae": float(np.mean(np.abs(e))),
                        "rmse": float(np.sqrt(np.mean(e * e))),
                        "bias": float(np.mean(e)),
                        "mean_truth_m": float(np.mean(agg[sel])),
                    }
                entry["mae_ratio_high_over_low"] = float(
                    entry["high"]["mae"] / entry["low"]["mae"])
                rec["factors"][name] = entry
            # land-use classes keep their own codes instead of terciles
            classes = {}
            for c in np.unique(land[ok]):
                c = int(c)
                if c < 0:
                    continue
                sel = ok & (land == c)
                if sel.sum() < 1000:
                    continue
                e = err[sel]
                classes[str(c)] = {
                    "n": int(sel.sum()),
                    "mae": float(np.mean(np.abs(e))),
                    "rmse": float(np.sqrt(np.mean(e * e))),
                    "bias": float(np.mean(e)),
                    "mean_truth_m": float(np.mean(agg[sel])),
                }
            rec["landuse_classes"] = classes
            out["resolutions"][res][scen] = rec
            print(f"[cells] {res} {scen}: {int(ok.sum())} wet cells, "
                  f"{len(classes)} land-use classes", flush=True)
    save_json("cell_contrasts.json", out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("cell", "all"):
        cell_part()
    if which in ("cells", "all"):
        cells_part()
    if which in ("factors", "all"):
        factors_part()
