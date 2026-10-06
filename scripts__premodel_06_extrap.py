r"""Stage 6: empirical growth law of the coarse-grid error.

Fits a one-parameter power law to the observed 10/20/30 m segment of the
RMSE-vs-cell-size curve for each scenario, and reports the value that the same
law would give at 50 m, 100 m and 200 m.  Points beyond 30 m are extrapolations
and are labelled as such in the report, because this dataset contains no grid
coarser than 30 m.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sp

import premodel_lib as L


def main():
    fl = L.load_json("outputs/premodel/flood_error.json")
    out = {"note": ("least-squares power law RMSE(dx) = a * dx**b, fitted on the "
                    "10, 20 and 30 m points only; the 5 m point is excluded because "
                    "the observed curve is not monotone between 5 and 10 m. "
                    "Values past 30 m are extrapolations, not measurements."),
           "fit_points_m": [10, 20, 30], "scenarios": {}}
    for scen in L.SCENARIOS:
        rec = {}
        for key, scale in (("rmse", 1.0), ("mae", 1.0)):
            dx = np.array([10.0, 20.0, 30.0])
            y = np.array([fl["resolutions"][scen][r][key] for r in ("10m", "20m", "30m")])
            b, loga = np.polyfit(np.log(dx), np.log(y), 1)
            a = float(np.exp(loga))
            pred = a * dx ** b
            r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
            rec[key] = {"a": a, "b": float(b), "r2_on_fit_points": float(r2),
                        "fitted_m": [float(v) for v in pred],
                        "observed_m": [float(v) for v in y]}
        rec["extrapolated"] = {
            f"{d}m": {k: float(rec[k]["a"] * d ** rec[k]["b"]) for k in ("rmse", "mae")}
            for d in (50, 100, 150, 200)}
        # ratio of the error at a given size to the error at 10 m
        rec["growth_factor_vs_10m"] = {
            f"{d}m": float((d / 10.0) ** rec["rmse"]["b"]) for d in (20, 30, 50, 100, 200)}
        out["scenarios"][scen] = rec
        print(f"[extrap] {scen} b_rmse={rec['rmse']['b']:.3f} "
              f"R2={rec['rmse']['r2_on_fit_points']:.4f} "
              f"rmse@200m={rec['extrapolated']['200m']['rmse']:.3f}", flush=True)

    # how many coarse cells would a 200 m lattice have on the study domain
    ny, nx = 11520, 6480
    for d in (50, 100, 200):
        out.setdefault("lattice_size", {})[f"{d}m"] = [ny * 2 // d, nx * 2 // d]

    # tile-level Spearman correlation between agreement and each driver
    tm = L.load_json("outputs/premodel/tile_metrics.json")
    corr = {}
    for res in L.COARSE:
        rows = [r for r in tm if r["res"] == res]
        rec = {}
        for key in ("wet_frac", "within_cell_dem_std", "mean_slope", "building_frac",
                    "impervious_frac", "mean_dist_water", "peak_truth"):
            x = np.array([r[key] for r in rows], float)
            y = np.array([r["csi005"] for r in rows], float)
            ok = np.isfinite(x) & np.isfinite(y)
            rec[key] = {"pearson": float(np.corrcoef(x[ok], y[ok])[0, 1]),
                        "spearman": float(sp.spearmanr(x[ok], y[ok]).statistic)}
        corr[res] = rec
    out["tile_correlations_vs_csi005"] = corr

    out["velocity_caution"] = (
        "the derived speed comparison is reported as a rank agreement only; the "
        "mean magnitude of the native coarse speed and of the aggregated speed "
        "differ by up to a factor of six at 30 m, so the two fields are not on the "
        "same scale.")
    L.save_json("extrapolation.json", out)


if __name__ == "__main__":
    main()
