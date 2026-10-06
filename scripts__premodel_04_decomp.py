r"""Stage 4: what explains the tile-to-tile spread of cross-resolution agreement?

The dependent variable is the per-tile agreement between the native coarse run
and the aggregated 2 m truth (CSI at 0.05 m, plus the wet-depth Pearson r as a
secondary response).  Explanatory variables are grouped exactly as the research
plan asks, and every input comes from static.nc or from the coarse depth field
itself.  Shares are Shapley values over the grouped predictors, so they add up
to the full-model R^2 without depending on the order of entry.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np

import premodel_lib as L

GROUPS = {
    "rain_scenario": ["scen_100a"],
    "slope_roughness": ["mean_slope", "within_cell_dem_std", "mean_tpi"],
    "impervious": ["impervious_frac"],
    "built_landuse": ["building_frac", "landuse_entropy", "green_frac"],
    "river_drainage": ["mean_dist_water", "min_dist_water", "water_frac_fine"],
}
EXTRA = {"wet_state": ["wet_frac"]}
ALL_GROUPS = {**GROUPS, **EXTRA}


def design(rows):
    cols = []
    names = []
    n = len(rows)
    for k, keys in ALL_GROUPS.items():
        for key in keys:
            v = np.array([r[key] if key != "scen_100a"
                          else (1.0 if r["scenario"] == "100a" else 0.0)
                          for r in rows], float)
            cols.append(v)
            names.append(key)
    X = np.column_stack(cols)
    return X, names


def r2(X, y, idx):
    if len(idx) == 0:
        return 0.0
    Z = np.column_stack([np.ones(len(y)), X[:, idx]])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ beta
    ss = np.sum(resid ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1 - ss / sst)


def shapley(X, y, group_cols):
    keys = list(group_cols)
    m = len(keys)
    cache = {}
    for r in range(m + 1):
        for combo in itertools.combinations(keys, r):
            idx = [i for k in combo for i in group_cols[k]]
            cache[tuple(sorted(combo))] = r2(X, y, idx)
    vals = {}
    for g in keys:
        others = [k for k in keys if k != g]
        s = 0.0
        for r in range(len(others) + 1):
            for combo in itertools.combinations(others, r):
                w = math.factorial(r) * math.factorial(m - r - 1) / math.factorial(m)
                s += w * (cache[tuple(sorted(combo + (g,)))] -
                          cache[tuple(sorted(combo))])
        vals[g] = s
    return vals, cache[tuple(sorted(keys))]


def vif(X, names):
    out = {}
    for i, nm in enumerate(names):
        others = [j for j in range(X.shape[1]) if j != i]
        Z = np.column_stack([np.ones(len(X)), X[:, others]])
        beta, *_ = np.linalg.lstsq(Z, X[:, i], rcond=None)
        resid = X[:, i] - Z @ beta
        r2i = 1 - np.sum(resid ** 2) / max(np.sum((X[:, i] - X[:, i].mean()) ** 2), 1e-12)
        out[nm] = float(1 / max(1 - r2i, 1e-9))
    return out


def main():
    recs = L.load_json("outputs/premodel/tile_metrics.json")
    everything = {}
    for res in L.COARSE:
        rows = [r for r in recs if r["res"] == res]
        rows = [r for r in rows if all(np.isfinite(r.get(k, np.nan))
                                       for ks in ALL_GROUPS.values() for k in ks
                                       if k != "scen_100a")]
        X, names = design(rows)
        idx = {k: [names.index(n) for n in v] for k, v in ALL_GROUPS.items()}
        rng = np.random.default_rng(0)
        entry = {"n_tiles": len(rows),
                 "scenario_split": {s: sum(1 for r in rows if r["scenario"] == s)
                                    for s in L.SCENARIOS},
                 "vif": vif(X, names),
                 "univariate_r2": {}, "responses": {}}
        for g, cols in idx.items():
            entry["univariate_r2"][g] = r2(X, np.array([r["csi005"] for r in rows]), cols)
        for resp, label in (("csi005", "csi@0.05"), ("pearson_wet", "pearson_wet")):
            y = np.array([r[resp] for r in rows], float)
            ok = np.isfinite(y)
            Xs, ys = X[ok], y[ok]
            vals, full = shapley(Xs, ys, idx)
            entry["responses"][label] = {
                "r2_full": full,
                "shapley": vals,
                "share_of_explained": {k: v / full if full > 0 else float("nan")
                                       for k, v in vals.items()},
                "share_of_total": vals,
            }
        everything[res] = entry
        b = entry["responses"]["csi@0.05"]
        print(f"{res}: n={len(rows)} R2={b['r2_full']:.3f} " +
              " ".join(f"{k}={v:.3f}" for k, v in b["shapley"].items()), flush=True)
    L.save_json("variance_decomposition.json", everything)


if __name__ == "__main__":
    main()
