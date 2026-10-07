"""Audit helper: is the fig-21 (report fig 40) error really as large as the 0-2 m signal?

The reviewer's concern is that the depth panels span 0-2 m while the |error| panels span
0-1.6/0-1.75 m, which makes the reconstruction look unusable. Part of that impression is a
plotting artefact: fig21 gives the depth panels a shared vmax = p99.5(truth) and gives each
error panel its own vmax = p98(|error|), so the two colour bars measure different quantiles of
different quantities and cannot be compared.

This script re-renders the SAME tile with all panels on one shared, comparable scale, adds a
truth-normalised relative-error panel, and prints the numbers behind the verdict.

Reads only already-saved artefacts (outputs/report_figs/result_fields/tiles_chosen.npz and
pixels.npz) plus the frozen-V0 depth-binned diagnostics JSON. No model, no GPU, no training.

Writes:
  outputs/report_figs/fig21b_tile_deep_scale.png
  outputs/premodel/error_scale_diagnostic.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

F = ROOT / "outputs" / "report_figs" / "result_fields"
OUT_FIG = ROOT / "outputs" / "report_figs" / "fig21b_tile_deep_scale.png"
OUT_JSON = ROOT / "outputs" / "premodel" / "error_scale_diagnostic.json"

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Microsoft YaHei",
                                            "SimSun", "SimHei", "DejaVu Serif"],
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix", "font.size": 9.5, "axes.labelsize": 10,
    "axes.titlesize": 10.5, "legend.fontsize": 8.5, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "figure.dpi": 200, "savefig.dpi": 200,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
})
FLOOD = LinearSegmentedColormap.from_list("flood", [
    (0.00, "#ffffff"), (0.03, "#eaf4fb"), (0.10, "#c2e2f5"), (0.25, "#7fc8ec"),
    (0.45, "#4a9fdc"), (0.65, "#2b6fbe"), (0.85, "#18408f"), (1.00, "#0b1f52")])
FLOOD.set_bad("#e6e6e6")


def _fracs(e: np.ndarray) -> dict:
    return {f"frac_lt_{t}": float((e < t).mean()) for t in (0.05, 0.1, 0.25, 0.5, 1.0)}


def main() -> None:
    z = np.load(F / "tiles_chosen.npz")
    gt = z["t0_gt"].astype(np.float64)
    fr = z["t0_frozen"].astype(np.float64)
    wn = z["t0_win"].astype(np.float64)
    v = z["t0_valid"].astype(bool)
    dem = z["t0_dem"].astype(np.float64)
    coarse = z["t0_coarse"].astype(np.float64)

    g = gt[v]
    ef_all = np.abs(fr - gt)
    ew_all = np.abs(wn - gt)
    ef = ef_all[v]
    ew = ew_all[v]

    # ---- the two scales the original figure actually used --------------------
    vmax_depth = float(np.nanpercentile(gt[v], 99.5))
    lim_fr = float(np.nanpercentile(ef_all, 98))
    lim_wn = float(np.nanpercentile(ew_all, 98))
    # a single, comparable scale for everything: the 99.5th percentile of truth depth.
    shared = vmax_depth
    # and the honest max absolute error (not a percentile)
    err_max = float(max(ef.max(), ew.max()))

    wet = g > 0.05
    deep = g > 1.0
    dry = ~wet
    wet2d = v & (gt > 0.05)  # 2-D wet mask for plotting

    stats: dict = {
        "tile_index": 165, "scenario": "100a", "iy": 42, "ix": 10,
        "source": "outputs/report_figs/result_fields/tiles_chosen.npz (t0_*)",
        "note": "tile index/scenario from tiles_chosen_meta.json",
        "n_valid": int(v.sum()),
        "n_wet": int(wet.sum()),
        "n_dry": int(dry.sum()),
        "wet_frac": float(wet.mean()),
        "peak_truth_m": float(g.max()),
        "depth_panel_vmax_p99.5_truth_m": vmax_depth,
        "error_panel_lim_p98_frozen_m": lim_fr,
        "error_panel_lim_p98_win_m": lim_wn,
        "error_panel_lim_over_depth_vmax_frozen": lim_fr / vmax_depth,
        "error_panel_lim_over_depth_vmax_win": lim_wn / vmax_depth,
        "true_max_abs_error_m": err_max,
        "models": {},
        "depth_bins": {},
    }
    for name, e in (("frozen", ef), ("win", ew)):
        e_wet, e_dry, e_deep = e[wet], e[dry], e[deep]
        stats["models"][name] = {
            "MAE_all_m": float(e.mean()),
            "MAE_wet_m": float(e_wet.mean()),
            "MAE_dry_m": float(e_dry.mean()),
            "RMSE_wet_m": float(np.sqrt((e_wet ** 2).mean())),
            "median_abs_err_wet_m": float(np.median(e_wet)),
            "p75_abs_err_wet_m": float(np.percentile(e_wet, 75)),
            "p90_abs_err_wet_m": float(np.percentile(e_wet, 90)),
            "p95_abs_err_wet_m": float(np.percentile(e_wet, 95)),
            "p98_abs_err_all_m": float(np.percentile(e, 98)),
            "p99_abs_err_all_m": float(np.percentile(e, 99)),
            "frac_valid_lt_0.05m": float((e < 0.05).mean()),
            "frac_valid_lt_0.10m": float((e < 0.10).mean()),
            "frac_valid_lt_0.25m": float((e < 0.25).mean()),
            "frac_valid_lt_0.50m": float((e < 0.50).mean()),
            "frac_valid_lt_1.00m": float((e < 1.00).mean()),
            "wet_rel_err_median": float(np.median(e_wet / g[wet])),
            "wet_rel_err_mean": float((e_wet / g[wet]).mean()),
            "wet_rel_err_p90": float(np.percentile(e_wet / g[wet], 90)),
            "wet_rel_err_gt_0.5_frac": float((e_wet / g[wet] > 0.5).mean()),
            "deep_MAE_m": float(e_deep.mean()),
            "deep_share_of_tile_MSE": float((e[deep] ** 2).sum() / (e ** 2).sum()),
        }
        signed = (fr - gt)[v] if name == "frozen" else (wn - gt)[v]
        stats["models"][name]["mean_signed_error_m"] = float(signed.mean())
        stats["models"][name]["deep_mean_signed_error_m"] = float(signed[deep].mean())
        for lo, hi in ((0.0, 0.05), (0.05, 0.3), (0.3, 1.0), (1.0, 2.0), (2.0, np.inf)):
            sel = (g >= lo) & (g < hi)
            stats["depth_bins"][f"{lo}-{hi}"] = stats["depth_bins"].get(f"{lo}-{hi}", {})
            stats["depth_bins"][f"{lo}-{hi}"][name] = {
                "n": int(sel.sum()),
                "share_of_pixels": float(sel.mean()),
                "MAE_m": float(e[sel].mean()) if sel.any() else None,
                "bias_m": float(signed[sel].mean()) if sel.any() else None,
                "share_of_tile_MSE": float((e[sel] ** 2).sum() / (e ** 2).sum()),
            }

    stats["improved_frac"] = float(((ew_all < ef_all) & v).sum() / v.sum())

    # ---- how much of the deep signal is even present in the 10 m input? ------
    sc0 = gt.shape[-1] // coarse.shape[-1]
    iy, ix = np.where(deep2d) if False else np.where(v & (gt > 1.0))
    cell10 = coarse[np.clip(iy // sc0, 0, coarse.shape[0] - 1),
                    np.clip(ix // sc0, 0, coarse.shape[1] - 1)]
    stats["coarse_input_deep_signal"] = {
        "note": "does the 10 m input itself carry the deep water on this tile?",
        "n_deep_px": int(len(iy)),
        "mean_10m_cell_at_deep_px_m": float(cell10.mean()),
        "median_10m_cell_at_deep_px_m": float(np.median(cell10)),
        "p90_10m_cell_at_deep_px_m": float(np.percentile(cell10, 90)),
        "share_of_deep_px_in_10m_cell_gt_1m": float((cell10 > 1.0).mean()),
        "share_of_deep_px_in_10m_cell_gt_0p5m": float((cell10 > 0.5).mean()),
        "mean_truth_at_those_px_m": float(gt[v & (gt > 1.0)].mean()),
    }

    # ---- the full-split context the single tile must be read against ---------
    diag = json.loads((ROOT / "outputs" / "v0_10m2m_hmax" / "visualizations"
                       / "diagnostics_test.json").read_text(encoding="utf-8"))
    bins, dep, N = diag["depth_bins"], diag["depth"], float(diag["n_valid_pixels"])
    n = np.asarray(dep["n"], float)
    mse = n * np.asarray(dep["rmse"], float) ** 2
    idx_deep = [i for i, b in enumerate(bins) if b in ("1-1.5", "1.5-2", "2-3", "3-inf")]
    idx_2 = [i for i, b in enumerate(bins) if b in ("2-3", "3-inf")]
    stats["full_test_split"] = {
        "source": "outputs/v0_10m2m_hmax/visualizations/diagnostics_test.json (frozen V0)",
        "n_valid_pixels": N,
        "RMSE_all_from_bins_m": float(np.sqrt(mse.sum() / n.sum())),
        "MAE_all_from_bins_m": float((n * np.asarray(dep["mae"], float)).sum() / n.sum()),
        "share_pixels_ge_1m": float(n[idx_deep].sum() / n.sum()),
        "share_MSE_ge_1m": float(mse[idx_deep].sum() / mse.sum()),
        "share_pixels_ge_2m": float(n[idx_2].sum() / n.sum()),
        "share_MSE_ge_2m": float(mse[idx_2].sum() / mse.sum()),
        "per_bin": {b: {"n": int(n[i]), "mae_m": float(dep["mae"][i]),
                        "rmse_m": float(dep["rmse"][i]), "bias_m": float(dep["bias"][i]),
                        "share_of_MSE": float(mse[i] / mse.sum())}
                    for i, b in enumerate(bins)},
    }

    # ---- pixels.npz: population distribution of |error| on wet pixels --------
    pz = np.load(F / "pixels.npz")
    truth, pfr, pwn = pz["truth"], pz["frozen"], pz["win"]
    w = truth > 0.05
    stats["pixels_subsample"] = {
        "source": "outputs/report_figs/result_fields/pixels.npz (4M-pixel subsample of test)",
        "n_total_valid": int(pz["n_total"][0]),
        "n_subsample": int(truth.size),
        "wet_frac_in_subsample": float(w.mean()),
    }
    for name, p in (("frozen", pfr), ("win", pwn)):
        e = np.abs(p[w] - truth[w]).astype(np.float64)
        stats["pixels_subsample"][name] = {
            "MAE_wet_m": float(e.mean()),
            "median_abs_err_wet_m": float(np.median(e)),
            "p90_abs_err_wet_m": float(np.percentile(e, 90)),
            **_fracs(e),
        }
    OUT_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(json.dumps({k: stats[k] for k in
                      ("depth_panel_vmax_p99.5_truth_m", "error_panel_lim_p98_frozen_m",
                       "error_panel_lim_p98_win_m", "true_max_abs_error_m")}, indent=2))
    print("full-test MSE shares:",
          json.dumps({k: stats["full_test_split"][k] for k in
                      ("share_pixels_ge_1m", "share_MSE_ge_1m",
                       "share_pixels_ge_2m", "share_MSE_ge_2m")}, indent=2))

    # ---- figure: every depth-like panel on ONE scale, plus relative error ----
    sc = gt.shape[-1] // coarse.shape[-1]
    cup = np.repeat(np.repeat(coarse, sc, 0), sc, 1)[:gt.shape[0], :gt.shape[1]]
    fig, axes = plt.subplots(2, 4, figsize=(12.8, 6.8))

    def panel(ax, arr, title, cmap, vmax_, label=None):
        a = np.where(v, arr, np.nan)
        im = ax.imshow(a, cmap=cmap, vmin=0, vmax=vmax_, interpolation="nearest")
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if label:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(label, fontsize=7.5)
            cb.ax.tick_params(labelsize=7)
        return im

    panel(axes[0, 0], dem, "(a) Terrain elevation on the SAME depth scale", "terrain", shared,
          "elevation (m)")
    panel(axes[0, 1], cup, "(b) Coarse input, 10 m", FLOOD, shared)
    panel(axes[0, 2], gt, "(c) Ground truth, 2 m", FLOOD, shared)
    panel(axes[0, 3], fr, "(d) Frozen V0 prediction", FLOOD, shared, "water depth (m)")
    panel(axes[1, 0], wn, "(e) Fine-tuned prediction", FLOOD, shared, "water depth (m)")

    # (f) both errors on ONE shared scale = the depth scale
    e_pair = np.where(v, np.maximum(ef_all, ew_all), np.nan)
    panel(axes[1, 1], e_pair, "(f) max(|err| frozen, |err| fine-tuned)\n"
                              "on the SAME 0-%.2f m scale as (c)-(e)" % shared,
          "magma_r", shared, "|error| (m)")

    # (g) error as a percentage of local truth depth
    rel = np.where(wet2d, 100.0 * np.maximum(ef_all, ew_all) / np.maximum(gt, 0.05), np.nan)
    rel = np.clip(rel, 0, 200)
    im = axes[1, 2].imshow(rel, cmap="magma_r", vmin=0, vmax=200, interpolation="nearest")
    axes[1, 2].set_title("(g) Relative error, % of local truth depth\n(wet pixels only)",
                         fontsize=9.5)
    axes[1, 2].set_xticks([]); axes[1, 2].set_yticks([]); axes[1, 2].grid(False)
    cb = fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.02)
    cb.set_label("|error| / truth  (%)", fontsize=7.5); cb.ax.tick_params(labelsize=7)

    # (h) fraction of |error| that the depth colour bar can even resolve
    frac_over = np.where(v, ef_all > 0.5 * shared, np.nan)
    im = axes[1, 3].imshow(frac_over, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    axes[1, 3].set_title("(h) Pixels whose frozen |error| exceeds\n50%% of the depth colour scale",
                         fontsize=9.5)
    axes[1, 3].set_xticks([]); axes[1, 3].set_yticks([]); axes[1, 3].grid(False)
    cb = fig.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.02)
    cb.set_label("flag (black = yes)", fontsize=7.5); cb.ax.tick_params(labelsize=7)

    mf, mw = stats["models"]["frozen"], stats["models"]["win"]
    fig.suptitle(
        "Same tile as report fig 40, re-drawn on ONE comparable scale\n"
        f"depth vmax (p99.5 truth) = {shared:.2f} m   |   "
        f"but MAE over the 26.6%% wet pixels = {mf['MAE_wet_m']:.3f} m (frozen) / "
        f"{mw['MAE_wet_m']:.3f} m (fine-tuned)   |   "
        f"only {100 * mf['frac_valid_lt_0.25m']:.1f}%% of valid pixels have |error| < 0.25 m\n"
        f"the 13.7%% of pixels deeper than 1 m carry "
        f"{100 * mf['deep_share_of_tile_MSE']:.0f}%% of this tile's squared error",
        fontsize=10.0)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG)
    plt.close(fig)
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
