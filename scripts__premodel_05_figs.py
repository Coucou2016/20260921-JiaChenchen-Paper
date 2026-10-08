r"""Stage 5: report figures for the pre-model chapter (SciencePlots style)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report_figs"
OUT.mkdir(parents=True, exist_ok=True)
PRE = ROOT / "outputs" / "premodel"

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    # font.family must list the CJK faces explicitly, otherwise matplotlib
    # resolves "serif" to Times New Roman alone and draws CJK glyphs as tofu.
    "font.family": ["Times New Roman", "Microsoft YaHei", "SimSun", "SimHei",
                    "DejaVu Serif"],
    "font.serif": ["Times New Roman", "Microsoft YaHei", "SimSun", "SimHei",
                   "DejaVu Serif"],
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
})
C = {"5m": "#1f4e79", "10m": "#2a7fbf", "20m": "#e08b2b", "30m": "#c0392b",
     "20a": "#2a7fbf", "100a": "#c0392b", "model": "#1a7a3a",
     "id": "#5b5b5b", "bil": "#e08b2b", "pre": "#1a7a3a"}
RES = ["5m", "10m", "20m", "30m"]
SCEN = ["20a", "100a"]
THR = ["0.05", "0.30", "1.00"]
ALLRES_FIG = ["2m", "5m", "10m", "20m", "30m"]
_PAIRS = {}


def save(fig, name):
    """Write one plate out, after checking that its panel letters are a clean run.

    Figure 19 once came out with `d` twice and no `h`, because the letters were
    read out of a two-character string.  The check reads the finished titles, so
    it catches any repeat whatever way the letter was produced.
    """
    import re as _re
    letters = []
    for ax in fig.axes:
        for loc in ("left", "center", "right"):
            t = (ax.get_title(loc=loc) or "").strip()
            m = _re.match(r"^\(([a-zA-Z])\)", t)
            if m:
                letters.append(m.group(1))
    dup = {c for c in letters if letters.count(c) > 1}
    if dup:
        raise AssertionError(f"{name}: panel letter(s) {sorted(dup)} used twice "
                             f"in {letters}")
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name, f"[panel letters {' '.join(letters) or 'none'}]",
          flush=True)


def load(p):
    return json.loads((p).read_text(encoding="utf-8"))


FL = load(PRE / "flood_error.json")
TE = load(PRE / "terrain_distortion.json")
VE = load(PRE / "velocity_correlation.json")
TM = load(PRE / "tile_metrics.json")
VD = load(PRE / "variance_decomposition.json")
PR = load(PRE / "premodel_results.json")


def cells(res):
    return {"5m": 5, "10m": 10, "20m": 20, "30m": 30}[res]


# ---------------------------------------------------------------- fig 41
def fig_task():
    fig = plt.figure(figsize=(9.6, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35], hspace=0.42, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    boxes = [
        (0.10, "Coarse hydrodynamic run\n(5/10/20/30 m)\ncoarse depth field", "#dce6f2"),
        (3.05, "Pre-model\nerror correction\n(new in this report)", "#fdf0d8"),
        (6.00, "Super-resolution model\nHydroGeo-SRNO\n(done, unchanged)", "#dcefe0"),
        (8.95, "2 m depth field\nfor emergency decisions", "#efe6f6"),
    ]
    for x, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.75), 2.5, 1.5,
                                    boxstyle="round,pad=0.05,rounding_size=0.12",
                                    linewidth=1.2, edgecolor="#3a4a5a", facecolor=col))
        ax.text(x + 1.25, 1.5, txt, ha="center", va="center", fontsize=9.2)
    for x in (2.62, 5.57, 8.52):
        ax.add_patch(FancyArrowPatch((x, 1.5), (x + 0.41, 1.5), arrowstyle="-|>",
                                     mutation_scale=13, linewidth=1.4, color="#3a4a5a"))
    ax.text(3.0, 2.62, "Corrected coarse field", ha="center", fontsize=9, color="#8a5a10")
    ax.annotate("", xy=(4.3, 2.42), xytext=(4.3, 2.30),
                arrowprops=dict(arrowstyle="-|>", color="#8a5a10", lw=1.1))
    ax.text(1.35, 0.42, "Error source 1\ncoarse simulation vs\naggregated truth", ha="center",
            fontsize=8.4, color="#1f4e79")
    ax.text(1.35, 2.62, "Error source 2\ninformation lost in coarsening\n(within-cell terrain and depth variation)",
            ha="center", fontsize=8.4, color="#c0392b")
    ax.add_patch(FancyArrowPatch((1.35, 2.42), (1.35, 0.72), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.0, color="#c0392b",
                                 linestyle=":"))
    ax.set_title("(a) Position of the pre-model in the forecast chain", fontsize=10.5, loc="left")

    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(RES))
    w = 0.36
    for k, scen in enumerate(SCEN):
        vals = [FL["resolutions"][scen][r]["rmse"] for r in RES]
        ax.bar(x + (k - 0.5) * w, vals, w, label=f"{scen} return period",
               color=C["20a"] if scen == "20a" else C["100a"], alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cells(RES[i])} m" for i in range(len(RES))])
    ax.set_ylabel("RMSE of the coarse simulation against the aggregated truth (m)")
    ax.set_title("(b) Representation and numerical error vs grid size", fontsize=10.5, loc="left")
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[1, 1])
    for scen, ls in (("20a", "--"), ("100a", "-")):
        tot, sim, agg = [], [], []
        for r in RES:
            d = FL["resolutions"][scen][r]["error_decomp_2m"]
            tot.append(d["rmse_total"])
            sim.append(d["rmse_simulation"])
            agg.append(d["rmse_aggregation"])
        ax.plot(x, tot, ls, marker="o", color=C[scen], label=f"{scen} total")
        ax.plot(x, sim, ls, marker="s", color=C[scen], alpha=0.55,
                label=f"{scen} simulation")
        ax.plot(x, agg, ls, marker="^", color="#7a4fb5", alpha=0.75,
                label=f"{scen} aggregation")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cells(RES[i])} m" for i in range(len(RES))])
    ax.set_ylabel("RMSE measured on the 2 m lattice (m)")
    ax.set_title("(c) Decomposition of the two error terms", fontsize=10.5, loc="left")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    save(fig, "fig41_premodel_task.png")


# ---------------------------------------------------------------- fig 42
def fig_error_grid():
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0))
    xs = np.array([5, 10, 20, 30], float)
    ax = axes[0, 0]
    for scen in SCEN:
        ax.plot(xs, [FL["resolutions"][scen][r]["mae"] for r in RES], "-o",
                color=C[scen], label=f"{scen} MAE")
        ax.plot(xs, [FL["resolutions"][scen][r]["rmse"] for r in RES], "--s",
                color=C[scen], alpha=0.6, label=f"{scen} RMSE")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel("Grid edge length (m)")
    ax.set_ylabel("Error (m)")
    ax.set_title("(a) MAE and RMSE", fontsize=10.5, loc="left")
    ax.legend(frameon=False, ncol=2, fontsize=8)

    ax = axes[0, 1]
    for scen in SCEN:
        ax.plot(xs, [FL["resolutions"][scen][r]["bias"] for r in RES], "-o",
                color=C[scen], label=scen)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel("Grid edge length (m)")
    ax.set_ylabel("Bias, prediction minus truth (m)")
    ax.set_title("(b) Domain-mean bias", fontsize=10.5, loc="left")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    mk = {"0.05": "o", "0.30": "s", "1.00": "^"}
    for scen, ls in (("20a", "--"), ("100a", "-")):
        for t in THR:
            ax.plot(xs, [FL["resolutions"][scen][r]["csi"][t] for r in RES],
                    ls, marker=mk[t], color=C[scen],
                    alpha={"0.05": 1.0, "0.30": 0.7, "1.00": 0.45}[t],
                    label=f"{scen} CSI@{t} m")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel("Grid edge length (m)")
    ax.set_ylabel("Critical success index")
    ax.set_title("(c) Flood-extent agreement", fontsize=10.5, loc="left")
    ax.legend(frameon=False, ncol=2, fontsize=7.6)

    ax = axes[1, 1]
    for scen, ls in (("20a", "--"), ("100a", "-")):
        ax.plot(xs, [FL["resolutions"][scen][r]["volume_rel"] * 100 for r in RES],
                ls, marker="o", color=C[scen], label=f"{scen} volume-rel. error")
    for t, mkc in (("0.05", "s"), ("0.30", "^"), ("1.00", "D")):
        ax.plot(xs, [FL["resolutions"]["100a"][r]["area_km2"][t]["native"] /
                     FL["resolutions"]["100a"][r]["area_km2"][t]["truth"] - 1
                     for r in RES], ":", marker=mkc, color="#7a4fb5", alpha=0.8,
                label=f"100a flooded-area ratio @{t} m")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel("Grid edge length (m)")
    ax.set_ylabel("Relative bias (%)")
    ax.set_title("(d) Volume and flooded-area bias", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=7.6)
    save(fig, "fig42_error_vs_grid.png")


# ---------------------------------------------------------------- fig 43
def fig_depth_bins():
    bins = [b["bin"] for b in FL["resolutions"]["100a"]["5m"]["by_depth"]]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0))
    x = np.arange(len(bins))
    for ax, scen in zip(axes[0], SCEN):
        for r in RES:
            rec = {b["bin"]: b for b in FL["resolutions"][scen][r]["by_depth"]}
            ax.plot(x, [rec[b].get("bias", np.nan) for b in bins], "-o",
                    color=C[r], ms=3.6, label=f"{r} m grid")
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=45, ha="right", fontsize=7.6)
        ax.set_xlabel("Aggregated truth depth bin (m)")
        ax.set_ylabel("Bias (m)")
        ax.set_title(f"({('a', 'b')[SCEN.index(scen)]}) {scen} return period: bias by depth bin",
                     fontsize=10.5, loc="left")
        ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 0]
    for r in RES:
        rec = {b["bin"]: b for b in FL["resolutions"]["100a"][r]["by_depth"]}
        ax.plot(x, [rec[b].get("mae", np.nan) for b in bins], "-o", color=C[r],
                ms=3.6, label=f"{r} m grid")
    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=45, ha="right", fontsize=7.6)
    ax.set_xlabel("Aggregated truth depth bin (m)")
    ax.set_ylabel("MAE (m)")
    ax.set_title("(c) 100-year event: MAE by depth bin", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    for scen in SCEN:
        rec = {b["bin"]: b for b in FL["resolutions"][scen]["10m"]["by_depth"]}
        ax.semilogy(x, [max(rec[b].get("n", 0), 1) for b in bins], "-o",
                    color=C[scen], ms=3.6, label=scen)
    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=45, ha="right", fontsize=7.6)
    ax.set_xlabel("Aggregated truth depth bin (m)")
    ax.set_ylabel("Pixel count (log scale)")
    ax.set_title("(d) Sample count per depth bin on the 10 m grid", fontsize=10.5, loc="left")
    ax.legend(frameon=False)
    save(fig, "fig43_error_depth_bins.png")


# ---------------------------------------------------------------- fig 44
def fig_slope_bins():
    bins = [b["bin"] for b in FL["resolutions"]["100a"]["5m"]["by_slope"]]
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.0))
    x = np.arange(len(bins))
    for scen, ls, al in (("20a", "--", 0.55), ("100a", "-", 1.0)):
        for r in RES:
            rec = {b["bin"]: b for b in FL["resolutions"][scen][r]["by_slope"]}
            axes[0, 0].plot(x, [rec[b].get("mae", np.nan) for b in bins], ls,
                            marker="o", color=C[r], ms=3.4, alpha=al)
            axes[0, 1].plot(x, [rec[b].get("mae_wet", np.nan) for b in bins], ls,
                            marker="o", color=C[r], ms=3.4, alpha=al)
            axes[1, 1].plot(x, [rec[b].get("bias", np.nan) for b in bins], ls,
                            marker="o", color=C[r], ms=3.4, alpha=al)
    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("Cell mean slope (dimensionless, rise over run)")
    axes[0, 0].set_ylabel("MAE (m)")
    axes[0, 0].set_title("(a) All valid cells", fontsize=10.5, loc="left")
    axes[0, 1].set_ylabel("MAE (m)")
    axes[0, 1].set_title("(b) Wet cells only", fontsize=10.5, loc="left")
    axes[1, 1].axhline(0, color="#888", lw=0.8)
    axes[1, 1].set_ylabel("Mean bias (m)")
    axes[1, 1].set_title("(d) Mean bias over all valid cells", fontsize=10.5, loc="left")

    ax = axes[1, 0]
    wf = [FL["resolutions"]["100a"]["10m"]["by_slope"][k]["wet_frac"] * 100
          for k in range(len(bins))]
    dep = [FL["resolutions"]["100a"]["10m"]["by_slope"][k]["mean_truth"]
           for k in range(len(bins))]
    ax.bar(x - 0.19, dep, 0.36, color="#2a7fbf", alpha=0.85,
           label="Mean aggregated truth per bin (m)")
    ax.set_ylabel("Mean aggregated truth depth (m)", color="#1f4e79")
    ax.tick_params(axis="y", labelcolor="#1f4e79")
    ax2 = ax.twinx()
    ax2.plot(x + 0.19, wf, "-s", color="#c0392b", ms=4.2,
             label="Wet-cell share in the bin (%)")
    ax2.set_ylabel("Share of wet cells (%)", color="#8a2b20")
    ax2.tick_params(axis="y", labelcolor="#8a2b20")
    ax2.grid(False)
    ax.set_title("(c) Water condition per slope bin on the 10 m grid", fontsize=10.5,
                 loc="left")
    hs = [Line2D([], [], color="#2a7fbf", lw=6, alpha=0.85,
                 label="Mean aggregated truth per bin (m)"),
          Line2D([], [], color="#c0392b", marker="s", ls="-",
                 label="Wet-cell share in the bin (%)")]
    ax.legend(handles=hs, frameon=False, fontsize=8, loc="upper right")

    handles = [Line2D([], [], color=C[r], marker="o", ls="-", label=f"{r} m grid")
               for r in RES]
    handles += [Line2D([], [], color="#444", ls="-", label="100a"),
                Line2D([], [], color="#444", ls="--", alpha=0.6, label="20a")]
    axes[0, 1].legend(handles=handles, frameon=False, fontsize=8, ncol=1,
                      loc="upper left")
    fig.tight_layout()
    save(fig, "fig44_error_slope_bins.png")


# ---------------------------------------------------------------- fig 45
def fig_terrain():
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
    xs = np.array([5, 10, 20, 30], float)
    ax = axes[0, 0]
    ax.plot(xs, [TE["resolutions"][r]["within_cell_std_mean"] for r in RES], "-o",
            color="#1f4e79", label="Mean within-cell DEM std")
    ax.plot(xs, [TE["resolutions"][r]["within_cell_std_median"] for r in RES], "--s",
            color="#1f4e79", alpha=0.6, label="Median")
    ax.plot(xs, [TE["resolutions"][r]["relief_mean"] for r in RES], "-^",
            color="#c0392b", label="Mean within-cell relief")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel("Grid edge length (m)")
    ax.set_ylabel("Elevation statistic (m)")
    ax.set_title("(a) Terrain relief within coarse cells", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.03, 0.95, f"2 m DEM std {TE['fine']['dem_std']:.1f} m",
            transform=ax.transAxes, fontsize=8, va="top", color="#444")

    ax = axes[0, 1]
    ax.plot(xs, [TE["resolutions"][r]["slope_native_mean"] for r in RES], "-o",
            color="#c0392b", label="Mean coarse-grid slope")
    ax.plot(xs, [TE["resolutions"][r]["slope_agg_mean"] for r in RES], "--s",
            color="#1f4e79", label="2 m slope, aggregated mean")
    ax.plot(xs, [TE["resolutions"][r]["slope_native_median"] for r in RES], ":o",
            color="#c0392b", alpha=0.6, label="Median coarse-grid slope")
    ax.plot(xs, [TE["resolutions"][r]["slope_agg_median"] for r in RES], ":s",
            color="#1f4e79", alpha=0.6, label="Median aggregated slope")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel("Grid edge length (m)")
    ax.set_ylabel("Slope (dimensionless)")
    ax.set_title("(b) Slope attenuation with grid size", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=7.6)
    for r in RES:
        ax.annotate(f"{TE['resolutions'][r]['slope_ratio']:.3f}",
                    (cells(r), TE["resolutions"][r]["slope_native_mean"]),
                    textcoords="offset points", xytext=(2, -12), fontsize=7.4,
                    color="#8a5a10")

    ax = axes[1, 0]
    ax.plot(xs, [TE["resolutions"][r]["sink_depth_mean"] for r in RES], "-o",
            color="#1f4e79", label="Mean sink depth (m)")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel("Grid edge length (m)")
    ax.set_ylabel("Cell mean elevation minus cell minimum (m)")
    ax.set_title("(c) Local sinks erased by coarsening", fontsize=10.5, loc="left")
    ax2 = ax.twinx()
    ax2.plot(xs, [TE["resolutions"][r]["sink_frac_gt_0p1"] * 100 for r in RES], "--s",
             color="#c0392b", label="Share of cells with a sink deeper than 0.1 m")
    ax2.plot(xs, [TE["resolutions"][r]["sink_frac_gt_0p5"] * 100 for r in RES], ":^",
             color="#c0392b", alpha=0.6, label="Deeper than 0.5 m")
    ax2.set_ylabel("Share of cells (%)", color="#c0392b")
    ax2.tick_params(axis="y", colors="#c0392b")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.6, loc="center right")

    ax = axes[1, 1]
    h = TE["channel_width_hist"]
    e = np.array(h["edges"])
    ax.bar(e[:-1], np.array(h["counts"]) / h["n_total"] * 100,
           width=np.diff(e), align="edge", color="#9fb8d0", edgecolor="#5b7fa6",
           linewidth=0.3)
    for r, col in zip(RES, [C["5m"], C["10m"], C["20m"], C["30m"]]):
        ax.axvline(cells(r), color=col, ls="--", lw=1.2)
        ax.text(cells(r), ax.get_ylim()[1] * 0.94, f"{cells(r)} m",
                rotation=90, fontsize=7.4, color=col, ha="right", va="top")
    ax.set_xlabel("Channel width from the Euclidean distance transform of the 2 m water mask (m)")
    ax.set_ylabel("Share of water pixels (%)")
    ax.set_title("(d) Channel-width distribution vs grid edge length", fontsize=10.5, loc="left")
    save(fig, "fig45_terrain_distortion.png")


# ---------------------------------------------------------------- fig 46
def fig_maps():
    scen = "100a"
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 8.4))
    ext = [0, 12960, 23040, 0]
    for k, r in enumerate(RES):
        d = np.load(PRE / f"map_{scen}_{r}.npz")
        truth = d["truth"].astype("float32")
        err = d["err"].astype("float32")
        im = axes[0, k].imshow(truth, extent=ext, cmap="Blues", vmin=0, vmax=3.0,
                               interpolation="nearest")
        axes[0, k].set_title(f"{cells(r)} m grid, aggregated truth", fontsize=9.6)
        cb = fig.colorbar(im, ax=axes[0, k], fraction=0.028, pad=0.02)
        cb.set_label("Depth (m)" if k == 0 else "", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        lim = 2.0
        im2 = axes[1, k].imshow(err, extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim,
                                interpolation="nearest")
        axes[1, k].set_title(f"{cells(r)} m grid, coarse simulation minus aggregated truth", fontsize=9.6)
        cb2 = fig.colorbar(im2, ax=axes[1, k], fraction=0.028, pad=0.02)
        cb2.set_label("Error (m)" if k == 0 else "", fontsize=8)
        cb2.ax.tick_params(labelsize=7)
        for ax in (axes[0, k], axes[1, k]):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
    fig.suptitle("100-year event: domain depth and coarse-grid error; north up; metres", fontsize=11, y=0.985)
    save(fig, "fig46_error_maps.png")


# ---------------------------------------------------------------- fig 64
def fig_zoom():
    """Two typical sub-regions enlarged, so the four coarse grids can be told apart.

    On the full-domain plate a ten-metre cell is far under one pixel, so all four
    resolutions print as the same blue smear.  This plate takes a 900 m window and
    blows it up, and puts the two-metre field alongside as the reference of what
    the detail was before any coarsening.
    """
    import netCDF4 as nc

    scen = "100a"
    GRIDS = ROOT / "dataset" / "grids"
    DIV = {"5m": 2.0, "10m": 1.0, "20m": 0.5, "30m": 10.0 / 30.0}
    COLS = ["5m", "10m", "20m", "30m"]
    # (label, row0, row1, col0, col1) as 10 m coarse indices, north-up.
    # Bounds are multiples of six so that every grid lands on the window exactly.
    REG = [("Region 1  dense channel network", (1188, 1230, 468, 510)),
           ("Region 2  broad water surface", (330, 372, 1068, 1110))]

    def clean(a):
        a = np.array(a, dtype="float32", copy=True)
        a[~np.isfinite(a) | (a <= -9998.0)] = np.nan
        return a

    d30 = np.load(PRE / f"map_{scen}_30m.npz")
    loc = d30["truth"].astype("float32")

    # The plate is twenty panels wide, so it is kept to eleven inches and the type
    # is set large.  At the report's column width that keeps the labels at the same
    # printed size as the rest of the chapter instead of shrinking them away.
    fig, axes = plt.subplots(4, 5, figsize=(9.4, 8.9))
    TY = 10
    for gi, (glab, (r0, r1, c0, c1)) in enumerate(REG):
        # two-metre reference for this window, read straight from the netCDF
        with nc.Dataset(GRIDS / "2m" / f"flood_{scen}.nc") as ds:
            h2 = clean(ds["h_max"][r0 * 5:r1 * 5, c0 * 5:c1 * 5])
        ext = [c0 * 10, c1 * 10, r1 * 10, r0 * 10]

        rows = {}
        for k, res in enumerate(COLS):
            d = DIV[res]
            a = np.load(PRE / f"map_{scen}_{res}.npz")
            t = a["truth"].astype("float32")[int(r0 * d):int(r1 * d), int(c0 * d):int(c1 * d)]
            e = a["err"].astype("float32")[int(r0 * d):int(r1 * d), int(c0 * d):int(c1 * d)]
            rows[res] = (t, e)
        for q in (0, 1):
            ax = axes[gi * 2 + q, 0]
            if q == 0:
                ax.imshow(h2, extent=ext, cmap="Blues", vmin=0, vmax=3.0,
                          interpolation="nearest")
                ax.set_title(f"{glab}\n2 m reference", fontsize=TY, color="#1a7a3a")
            else:
                ax.imshow(loc, extent=[0, 12960, 23040, 0], cmap="Blues",
                          vmin=0, vmax=3.0, interpolation="nearest")
                ax.add_patch(plt.Rectangle((c0 * 10, r0 * 10), (c1 - c0) * 10,
                                           (r1 - r0) * 10, fill=False,
                                           edgecolor="#c0392b", linewidth=1.8))
                ax.set_title(f"{glab}\nzoom location", fontsize=TY, color="#c0392b")
                ax.plot([ext[0] + 250, ext[0] + 350], [ext[2] - 30, ext[2] - 30],
                        color="#1a1a1a", lw=2.4)
                ax.text(ext[0] + 300, ext[2] - 52, "100 m", fontsize=9.0,
                        ha="center", va="bottom", color="#1a1a1a")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

            for k, res in enumerate(COLS):
                t, e = rows[res]
                ax = axes[gi * 2 + q, k + 1]
                if q == 0:
                    im = ax.imshow(t, extent=ext, cmap="Blues", vmin=0, vmax=3.0,
                                   interpolation="nearest")
                    ax.set_title(f"{cells(res)} m\n{t.shape[1]} × {t.shape[0]} cells",
                                 fontsize=TY)
                else:
                    im = ax.imshow(e, extent=ext, cmap="RdBu_r", vmin=-2.0, vmax=2.0,
                                   interpolation="nearest")
                    fin = np.isfinite(e)
                    lab = (f"MAE {np.mean(np.abs(e[fin])):.3f} m"
                           if fin.any() else "No valid cells")
                    ax.set_title(f"{cells(res)} m\n{lab}", fontsize=TY)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                if k == 3:
                    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
                    cb.set_label("Depth (m)" if q == 0 else "Error (m)", fontsize=9.5)
                    cb.ax.tick_params(labelsize=8)

    fig.suptitle(f"{scen} return period, zoom on two typical sub-regions, 420 m window, north up, metres",
                 fontsize=11.5, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.972])
    save(fig, "fig64_domain_zoom.png")


# ---------------------------------------------------------------- fig 47
def fig_cross_res():
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0))
    xs = np.array([2.5, 5.0, 10.0, 15.0])
    ax = axes[0, 0]
    mk = {"0.05": "o", "0.30": "s", "1.00": "^"}
    for scen, ls in (("20a", "--"), ("100a", "-")):
        for t in THR:
            ax.plot(xs, [FL["resolutions"][scen][r]["csi"][t] for r in RES], ls,
                    marker=mk[t], color=C[scen],
                    alpha={"0.05": 1.0, "0.30": 0.72, "1.00": 0.48}[t],
                    label=f"{scen} CSI@{t}")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel("Resolution ratio, coarse edge / 2 m")
    ax.set_ylabel("Critical success index")
    ax.set_title("(a) Flood-extent agreement decays with the resolution ratio", fontsize=10.5, loc="left")
    ax.legend(frameon=False, ncol=2, fontsize=7.6)

    ax = axes[0, 1]
    for scen, ls in (("20a", "--"), ("100a", "-")):
        ax.plot(xs, [FL["resolutions"][scen][r]["pearson_wet"] for r in RES], ls,
                marker="o", color=C[scen], label=f"{scen} Pearson")
        ax.plot(xs, [FL["resolutions"][scen][r]["spearman_wet"] for r in RES], ls,
                marker="s", color=C[scen], alpha=0.6, label=f"{scen} Spearman")
        ax.plot(xs, [FL["resolutions"][scen][r]["ssim"] for r in RES], ls,
                marker="^", color="#7a4fb5", alpha=0.75, label=f"{scen} SSIM")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel("Resolution ratio, coarse edge / 2 m")
    ax.set_ylabel("Correlation and structural similarity")
    ax.set_title("(b) Depth-distribution similarity", fontsize=10.5, loc="left")
    ax.legend(frameon=False, ncol=2, fontsize=7.4)

    ax = axes[1, 0]
    for scen, ls in (("20a", "--"), ("100a", "-")):
        rec = VE["scenarios"][scen]["resolutions"]
        ax.plot(xs, [rec[r]["pearson"] for r in RES], ls, marker="o", color=C[scen],
                label=f"{scen} Pearson")
        ax.plot(xs, [rec[r]["spearman"] for r in RES], ls, marker="s", color=C[scen],
                alpha=0.6, label=f"{scen} Spearman")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel("Resolution ratio")
    ax.set_ylabel("Speed-magnitude correlation")
    ax.set_title("(c) Speed-distribution similarity (only at times with a flow field)", fontsize=10.5, loc="left")
    ax.legend(frameon=False, ncol=2, fontsize=7.6)

    ax = axes[1, 1]
    for scen, ls in (("20a", "--"), ("100a", "-")):
        ax.plot(xs, [FL["resolutions"][scen][r]["rmse_rel_mean_truth"] for r in RES],
                ls, marker="o", color=C[scen], label=f"{scen} relative RMSE")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel("Resolution ratio")
    ax.set_ylabel("RMSE divided by mean truth depth")
    ax.set_title("(d) Normalised error grows with the resolution ratio", fontsize=10.5, loc="left")
    ax.legend(frameon=False)
    save(fig, "fig47_cross_resolution.png")


# ---------------------------------------------------------------- fig 48
def fig_tile_scatter():
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
    def pick(res, scen=None):
        return [r for r in TM if r["res"] == res and (scen is None or r["scenario"] == scen)]
    ax = axes[0, 0]
    d5 = {(r["iy"], r["ix"], r["scenario"]): r for r in pick("5m")}
    d30 = {(r["iy"], r["ix"], r["scenario"]): r for r in pick("30m")}
    for scen in SCEN:
        xs = [d5[k]["csi005"] for k in d5 if k[2] == scen and k in d30]
        ys = [d30[k]["csi005"] for k in d5 if k[2] == scen and k in d30]
        ax.scatter(xs, ys, s=13, alpha=0.65, color=C[scen], label=scen,
                   edgecolors="none")
    lim = [0, 1]
    ax.plot(lim, lim, color="#888", lw=0.9, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Per-tile CSI@0.05 on the 5 m grid")
    ax.set_ylabel("Per-tile CSI@0.05 on the 30 m grid")
    ax.set_title("(a) Agreement of the same tile on two grids", fontsize=10.5, loc="left")
    ax.legend(frameon=False)

    for ax, key, xlab, ttl in (
            (axes[0, 1], "wet_frac", "Tile wet fraction", "(b) Wet fraction vs agreement"),
            (axes[1, 0], "within_cell_dem_std", "Within-cell 2 m DEM std (m)",
             "(c) Terrain roughness vs agreement"),
            (axes[1, 1], "building_frac", "Tile building-pixel fraction", "(d) Building fraction vs agreement")):
        for scen in SCEN:
            rec = pick("10m", scen)
            xs = np.array([r[key] for r in rec], float)
            ys = np.array([r["csi005"] for r in rec], float)
            ok = np.isfinite(xs) & np.isfinite(ys)
            ax.scatter(xs[ok], ys[ok], s=13, alpha=0.65, color=C[scen],
                       label=f"{scen} r={np.corrcoef(xs[ok], ys[ok])[0,1]:.2f}",
                       edgecolors="none")
        ax.set_xlabel(xlab)
        ax.set_ylabel("Per-tile CSI@0.05 on the 10 m grid")
        ax.set_title(ttl, fontsize=10.5, loc="left")
        ax.legend(frameon=False, fontsize=8)
    save(fig, "fig48_tile_scatter.png")


# ---------------------------------------------------------------- fig 49
def fig_decomp():
    fig = plt.figure(figsize=(9.8, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.45, wspace=0.3)
    ax = fig.add_subplot(gs[0, :])
    groups = ["rain_scenario", "slope_roughness", "impervious", "built_landuse",
              "river_drainage", "wet_state"]
    cn = {"rain_scenario": "Rain scenario", "slope_roughness": "Slope and roughness",
          "impervious": "Impervious fraction", "built_landuse": "Buildings and land use",
          "river_drainage": "River proximity", "wet_state": "Wet fraction (state variable)"}
    cols = ["#1f4e79", "#2a7fbf", "#5aa469", "#e08b2b", "#c0392b", "#8c8c8c"]
    bottom = np.zeros(len(RES))
    x = np.arange(len(RES))
    for g, col in zip(groups, cols):
        vals = np.array([VD[r]["responses"]["csi@0.05"]["shapley"][g] for r in RES])
        ax.bar(x, vals, 0.62, bottom=bottom, color=col, label=cn[g],
               edgecolor="white", linewidth=0.5)
        bottom += vals
    for i, r in enumerate(RES):
        ax.text(i, bottom[i] + 0.012, f"R²={VD[r]['responses']['csi@0.05']['r2_full']:.2f}",
                ha="center", fontsize=8.4, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cells(r)} m grid" for r in RES])
    ax.set_ylabel("Share of per-tile CSI@0.05 variance explained (Shapley)")
    ax.set_title("(a) Contribution decomposition of cross-resolution agreement", fontsize=10.5, loc="left")
    ax.legend(frameon=False, ncol=3, fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    names = ["mean_slope", "within_cell_dem_std", "mean_tpi", "impervious_frac",
             "building_frac", "landuse_entropy", "green_frac", "mean_dist_water",
             "min_dist_water", "water_frac_fine", "wet_frac"]
    cn2 = {"mean_slope": "Slope", "within_cell_dem_std": "Within-cell DEM std",
           "mean_tpi": "Topographic position index", "impervious_frac": "Impervious fraction",
           "building_frac": "Building fraction", "landuse_entropy": "Land-use entropy",
           "green_frac": "Green-space fraction", "mean_dist_water": "Mean distance to water",
           "min_dist_water": "Minimum distance to water", "water_frac_fine": "Fine-grid water fraction",
           "wet_frac": "Wet fraction"}
    vifs = np.array([VD["10m"]["vif"][n] for n in names])
    ax.barh(np.arange(len(names)), vifs, color="#5b7fa6")
    ax.axvline(5, color="#c0392b", ls="--", lw=1.0)
    ax.text(5.1, len(names) - 0.6, "Collinearity reference 5", color="#c0392b", fontsize=7.8)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([cn2[n] for n in names], fontsize=7.8)
    ax.set_xlabel("Variance inflation factor (10 m grid)")
    ax.set_title("(b) Collinearity among the predictors", fontsize=10.5, loc="left")
    ax.grid(axis="y", alpha=0.0)

    ax = fig.add_subplot(gs[1, 1])
    gs_names = groups
    mat = np.zeros((len(gs_names), len(RES)))
    for j, r in enumerate(RES):
        for i, g in enumerate(gs_names):
            mat[i, j] = VD[r]["responses"]["csi@0.05"]["share_of_explained"][g]
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(RES)))
    ax.set_xticklabels([f"{cells(r)} m" for r in RES])
    ax.set_yticks(range(len(gs_names)))
    ax.set_yticklabels([cn[g] for g in gs_names], fontsize=8)
    for i in range(len(gs_names)):
        for j in range(len(RES)):
            ax.text(j, i, f"{mat[i, j]*100:.0f}%", ha="center", va="center",
                    fontsize=7.6, color="#111" if mat[i, j] < 0.42 else "white")
    ax.set_title("(c) Share of explained variance by factor", fontsize=10.5, loc="left")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    save(fig, "fig49_decomposition.png")


# ---------------------------------------------------------------- fig 50
def fig_premodel():
    import torch
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import premodel_03_train as T

    res = "10m"
    R = PR[res]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))

    ax = axes[0, 0]
    d = np.load(PRE / f"tile_{res}_test.npz")
    X, M = d["X"], d["M"][:, 0].astype(bool)
    ck = torch.load(PRE / f"ckpt_{res}.pt", map_location="cpu", weights_only=False)
    model = T.PreModel()
    model.load_state_dict(ck["state"])
    model.eval()
    x = torch.tensor(X)
    x = torch.nan_to_num((x - torch.tensor(ck["mu"])[None, :, None, None])
                         / torch.tensor(ck["sd"])[None, :, None, None])
    with torch.no_grad():
        corr = ((x[:, :1] - model(T.build_input(x))) * float(ck["sd"][0])
                + float(ck["mu"][0])).numpy()[:, 0]
    err_before, err_after = [], []
    for i in range(X.shape[0]):
        ok = M[i]
        if ok.sum() < 4:
            continue
        err_before.append(float(np.mean(np.abs(X[i, 0][ok] - d["T"][i, 0][ok]))))
        err_after.append(float(np.mean(np.abs(corr[i][ok] - d["T"][i, 0][ok]))))
    ax.scatter(err_before, err_after, s=15, alpha=0.7, color="#1f4e79",
               edgecolors="none")
    hi = max(max(err_before), max(err_after)) * 1.05
    ax.plot([0, hi], [0, hi], "--", color="#c0392b", lw=1.0, label="1:1 line")
    ax.set_xlabel("Per-tile MAE before correction (m)")
    ax.set_ylabel("Per-tile MAE after correction (m)")
    ax.set_title("(a) Paired per-tile pre-model effect (10 m test split)", fontsize=10.5, loc="left")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    g = R["global"]
    names = ["identity", "bilinear", "premodel"]
    cn = {"identity": "Identity", "bilinear": "Bilinear resampling", "premodel": "Pre-model"}
    x = np.arange(len(THR))
    w = 0.26
    for k, nm in enumerate(names):
        vals = [g[nm][f"csi@{t}"] for t in THR]
        ax.bar(x + (k - 1) * w, vals, w, label=cn[nm],
               color=[C["id"], C["bil"], C["pre"]][k])
    ax.set_xticks(x)
    ax.set_xticklabels([f"CSI@{t}" for t in THR])
    ax.set_ylabel("Critical success index")
    ax.set_title("(b) Agreement of the three fields on the coarse grid", fontsize=10.5, loc="left")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    names2 = ["rmse", "mae"]
    cn2 = {"rmse": "RMSE", "mae": "MAE"}
    x2 = np.arange(len(names2))
    for k, nm in enumerate(names):
        vals = [g[nm][q] for q in names2]
        ax.bar(x2 + (k - 1) * w, vals, w, label=cn[nm],
               color=[C["id"], C["bil"], C["pre"]][k])
    ax.set_xticks(x2)
    ax.set_xticklabels([cn2[q] for q in names2])
    ax.set_ylabel("Error (m)")
    ax.set_title("(c) Numerical-error comparison", fontsize=10.5, loc="left")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    fg = R["fine_grid_check"]
    x3 = np.arange(len(THR))
    for k, nm in enumerate(names):
        vals = [fg[nm][f"csi@{t}"] for t in THR]
        ax.bar(x3 + (k - 1) * w, vals, w, label=cn[nm],
               color=[C["id"], C["bil"], C["pre"]][k])
    ax.set_xticks(x3)
    ax.set_xticklabels([f"CSI@{t}" for t in THR])
    ax.set_ylabel("Critical success index")
    ax.set_title("(d) Agreement after nearest-neighbour upsampling to 2 m", fontsize=10.5, loc="left")
    ax.legend(frameon=False)
    save(fig, "fig50_premodel_result.png")


# ---------------------------------------------------------------- fig 51
def fig_dialect():
    """The two comparison dialects and the identity that joins them."""
    SD = load(PRE / "scale_dialect.json")
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
    x = np.arange(len(RES))
    lab = [f"{cells(r)} m" for r in RES]
    c_coarse, c_within = "#1f4e79", "#e08b2b"

    for k, scen in enumerate(SCEN):
        ax = axes[0, k]
        coarse = np.array([SD["results"][scen][r]["fine_nn"]["mse_coarse"] for r in RES])
        within = np.array([SD["results"][scen][r]["fine_nn"]["mse_within"] for r in RES])
        fine = np.array([SD["results"][scen][r]["fine_nn"]["mse_fine"] for r in RES])
        ax.bar(x, coarse, 0.58, color=c_coarse, label="Coarse-lattice caliber, reducible error",
               edgecolor="white", linewidth=0.5)
        ax.bar(x, within, 0.58, bottom=coarse, color=c_within,
               label="Within-block variance, irreducible detail", edgecolor="white", linewidth=0.5)
        for i in range(len(RES)):
            ax.text(i, fine[i] + 0.006, f"{within[i]/fine[i]*100:.0f}%",
                    ha="center", fontsize=8.2, color="#8a5a10")
        ax.plot(x, fine, "k_", markersize=14, markeredgewidth=1.6,
                label="Fine-lattice caliber, total")
        ax.set_xticks(x)
        ax.set_xticklabels(lab)
        ax.set_ylabel("Mean squared error (m^2)")
        ax.set_title(f"({('a', 'b')[k]}) {scen} return period, two calibers", fontsize=10.5,
                     loc="left")
        ax.legend(frameon=False, fontsize=7.8)

    ax = axes[1, 0]
    for scen, ls in (("20a", "--"), ("100a", "-")):
        ax.plot(x, [SD["results"][scen][r]["rmse_m"]["fine_dialect"] for r in RES],
                ls, marker="o", color=C[scen], label=f"{scen} fine-lattice")
        ax.plot(x, [SD["results"][scen][r]["rmse_m"]["coarse_dialect"] for r in RES],
                ls, marker="s", color=C[scen], alpha=0.55, label=f"{scen} coarse-lattice")
    ax.set_xticks(x)
    ax.set_xticklabels(lab)
    ax.set_ylabel("RMSE (m)")
    ax.set_title("(c) Error level under the two calibers", fontsize=10.5, loc="left")
    ax.legend(frameon=False, ncol=2, fontsize=8)

    ax = axes[1, 1]
    for scen, ls, dx in (("20a", "--", -17), ("100a", "-", 17)):
        sh = [SD["results"][scen][r]["shares"]["within_of_fine"] * 100 for r in RES]
        ax.plot(x, sh, ls, marker="^", color=C[scen], label=f"{scen} return period")
        for i in range(len(RES)):
            ax.annotate(f"{sh[i]:.1f}%", (x[i], sh[i]), textcoords="offset points",
                        xytext=(dx, 5), ha="center", fontsize=7.8, color=C[scen])
    ax.set_xticks(x)
    ax.set_xticklabels(lab)
    ax.set_ylim(0, 30)
    ax.set_ylabel("Within-block variance as a share of the fine-lattice caliber (%)")
    ax.set_title("(d) How much of the fine-lattice caliber is intrinsic ground variability", fontsize=10.5, loc="left")
    ax.legend(frameon=False)
    save(fig, "fig51_dialects.png")


# ---------------------------------------------------------------- fig 71
def _cs():
    """The cross-resolution consistency payload, loaded on demand."""
    return load(PRE / "cross_resolution_consistency.json")


def _clip(a):
    a = np.array(a, dtype="float32", copy=True)
    a[~np.isfinite(a) | (a <= -9998.0)] = np.nan
    return a


def fig_terrain_flood():
    """The same terrain and flood fields seen at four grid sizes.

    Column by column the four grids are drawn on a common colour scale, so the
    eye can judge how much of the pattern actually survives coarsening.  The
    study domain is taller than it is wide, so each field is turned on its side
    before it is drawn.  That keeps four grids abreast inside the report's
    column width instead of squeezing them into four narrow slivers, and it
    leaves every spatial relation in the plate untouched.
    """
    import premodel_lib as L

    scen = "100a"
    fig = plt.figure(figsize=(9.8, 6.4))
    gs = fig.add_gridspec(4, 5, width_ratios=[1, 1, 1, 1, 0.05],
                          hspace=0.13, wspace=0.05)
    rows = [("Elevation (m)", "terrain", None, None),
            ("Slope", "YlOrBr", None, None),
            ("Depth (m)", "Blues", 0.0, 3.0),
            ("Speed (m/s)", "PuBuGn", 0.0, 3.0)]
    vel = load(PRE / "velocity_correlation.json")
    k = vel["scenarios"][scen]["frame"]
    for i, (lab, cmap, vmin, vmax) in enumerate(rows):
        im = None
        for j, res in enumerate(RES):
            ax = fig.add_subplot(gs[i, j])
            if i == 0:
                a = L.load_static(res, "DEM")
            elif i == 1:
                a = L.load_static(res, "Slope")
            elif i == 2:
                a = L.load_flood(res, scen, "h_max")
            else:
                h = L.load_flood(res, scen, "h", times=k)
                ux = L.load_flood(res, scen, "hux", times=k)
                vy = L.load_flood(res, scen, "hvy", times=k)
                with np.errstate(invalid="ignore", divide="ignore"):
                    a = np.sqrt(ux ** 2 + vy ** 2) / np.where(h > 0.02, h, np.nan)
                del h, ux, vy
            a = np.rot90(_clip(a))
            if vmin is None:
                v = a[np.isfinite(a)]
                lo, hi = np.percentile(v, 1), np.percentile(v, 99)
            else:
                lo, hi = vmin, vmax
            im = ax.imshow(a, cmap=cmap, vmin=lo, vmax=hi, aspect="equal",
                           interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if i == 0:
                ax.set_title(f"{cells(res)} m grid", fontsize=9.5, pad=3)
            for s in ax.spines.values():
                s.set_linewidth(0.5)
            if j == 0:
                ax.set_ylabel(lab, fontsize=9)
        cax = fig.add_subplot(gs[i, 4])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=9)
    save(fig, "fig71_pattern_by_grid.png")


# ---------------------------------------------------------------- fig 72 / 73
def _scatter_panel(ax, cs, var, scen, res, thr=0.05):
    P = _pairs()
    key = f"{var}_{scen}_{res}"
    if key not in P:
        return
    d = P[key]
    x, y = d[1], d[0]          # x aggregated truth, y coarse simulation
    ok = (x > thr) & (y > thr)
    x, y = x[ok], y[ok].astype("float64")
    ax.hexbin(np.log10(x), np.log10(y), gridsize=48, bins="log",
              cmap="Blues", mincnt=1, linewidths=0)
    lim = [np.log10(thr), max(np.log10(np.percentile(x, 99.9)),
                              np.log10(np.percentile(y, 99.9)))]
    ax.plot(lim, lim, "k--", lw=0.9)
    r = cs["flood"]["variables"][var][scen]["resolutions"][res]
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Aggregated truth (log10)", fontsize=9)
    ax.set_ylabel("Coarse simulation (log10)", fontsize=9)
    ax.set_title(f"({('a', 'b')[0 if res == '10m' else 1]}) {cells(res)} m grid, "
                 f"Pearson {r['pearson_wet']:.3f}, Spearman {r['spearman_wet']:.3f}",
                 fontsize=9.5, loc="left")
    ax.tick_params(labelsize=9)


def _pairs():
    if not _PAIRS:
        z = np.load(PRE / "consistency_pairs.npz")
        for k in z.files:
            _PAIRS[k] = z[k]
    return _PAIRS


def _wet_sample(var, scen, res):
    y = _pairs()[f"{var}_{scen}_{res}"][0].astype("float64")
    return y[np.isfinite(y) & (y > 0.05)]


def _density_panel(ax, cs, var, scen, ref):
    ref = ref[np.isfinite(ref) & (ref > 0.05)]
    bins = np.linspace(0, np.percentile(ref, 99.5), 70)
    ax.hist(ref, bins=bins, density=True, histtype="step", lw=1.2,
            color="#5b5b5b", label="2 m reference")
    for res in RES:
        ax.hist(_wet_sample(var, scen, res), bins=bins, density=True,
                histtype="step", lw=1.1, color=C[res], label=f"{cells(res)} m")
    ax.set_xlabel("Depth (m)" if var == "h_max" else "Speed (m/s)", fontsize=9)
    ax.set_ylabel("Probability density", fontsize=9)
    ax.set_title("(c) Shape of the depth distribution at each resolution" if var == "h_max"
                 else "(c) Shape of the speed distribution at each resolution", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=2, columnspacing=0.9)
    ax.tick_params(labelsize=9)


def _fit_panel(ax, cs, var, scen, res):
    from scipy import stats

    fam = {"Gamma": stats.gamma, "Log-normal": stats.lognorm,
           "3-parameter Weibull": stats.weibull_min, "Exponential": stats.expon,
           "Log-logistic": stats.fisk}

    def _dist(name):
        """Stored names may carry a Chinese suffix; match on the English token."""
        for token, dist in (("Log-normal", stats.lognorm), ("Log-logistic", stats.fisk),
                            ("Weibull", stats.weibull_min), ("Gamma", stats.gamma),
                            ("Exponential", stats.expon)):
            if token in name:
                return dist
        raise KeyError(name)
    d = cs["flood"]["variables"][var][scen]["resolutions"][res]["dist_native"]
    y = _wet_sample(var, scen, res)
    top = np.percentile(y, 99)
    xs = np.linspace(1e-4, top, 400)
    ax.hist(y, bins=np.linspace(0, top, 60), density=True, histtype="step",
            lw=1.0, color="#9a9a9a", label="Sample")
    order = sorted([f for f in d["all_families"] if "aic" in f],
                   key=lambda f: f["aic"])[:3]
    for k, f in enumerate(order):
        pdf = _dist(f["family"]).pdf(xs, *f["params"])
        short = f["family"].split(" ")[0]
        ax.plot(xs, pdf, lw=1.3, color=["#1f4e79", "#e08b2b", "#c0392b"][k],
                label=f"{short}\nAIC {f['aic']:.0f}  KS {f['ks']:.3f}")
    ax.set_xlabel("Depth (m)" if var == "h_max" else "Speed (m/s)", fontsize=9)
    ax.set_ylabel("Probability density", fontsize=9)
    ax.set_title(f"(d) {cells(res)} m grid distribution fit", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right",
              labelspacing=0.35, handlelength=1.3, borderaxespad=0.35)
    ax.tick_params(labelsize=9)


def _quant_panel(ax, cs, var, scen):
    for q, col, mk in zip((50, 75, 95, 99),
                          ["#1f4e79", "#2a7fbf", "#e08b2b", "#c0392b"], "os^D"):
        vals = [float(np.percentile(_wet_sample(var, scen, r), q)) for r in RES]
        ax.plot([cells(r) for r in RES], vals, marker=mk, color=col, lw=1.2,
                label=f"{q}th percentile")
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 30])
    ax.set_xticklabels(["5", "10", "20", "30"])
    ax.set_xlabel("Grid edge length (m)", fontsize=9)
    ax.set_ylabel("Depth (m)" if var == "h_max" else "Speed (m/s)", fontsize=9)
    ax.set_title("(e) Quantiles vs grid size", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=2, columnspacing=0.9)
    ax.tick_params(labelsize=9)


def _moment_panel(ax, cs, var, scen):
    """Skew and excess kurtosis against grid size, on two scales since they differ."""
    ax2 = ax.twinx()
    for tag, key, col, target in (("Skewness", "skew", "#1f4e79", ax),
                                  ("Excess kurtosis", "kurtosis_excess", "#e08b2b", ax2)):
        vals = [cs["flood"]["variables"][var][scen]["resolutions"][r]
                ["dist_native"][key] for r in RES]
        target.plot([cells(r) for r in RES], vals, marker="o", color=col, lw=1.2,
                    label=tag)
    for target, col in ((ax, "#1f4e79"), (ax2, "#e08b2b")):
        target.tick_params(axis="y", labelsize=9, colors=col)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 30])
    ax.set_xticklabels([f"{c}" for c in [5, 10, 20, 30]])
    ax.set_xlabel("Grid edge length (m)", fontsize=9)
    ax.set_ylabel("Skewness", fontsize=9, color="#1f4e79")
    ax2.set_ylabel("Excess kurtosis", fontsize=9, color="#e08b2b")
    ax.set_title("(f) Stability of the distribution shape", fontsize=9.5, loc="left")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="best")
    ax.tick_params(labelsize=9)


def fig_depth_consistency():
    import premodel_lib as L

    cs = _cs()
    scen = "100a"
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 6.6))
    _scatter_panel(axes[0, 0], cs, "h_max", scen, "10m")
    _scatter_panel(axes[0, 1], cs, "h_max", scen, "30m")
    ref = _clip(L.load_flood("2m", scen, "h_max"))
    _density_panel(axes[0, 2], cs, "h_max", scen, ref)
    del ref
    _fit_panel(axes[1, 0], cs, "h_max", scen, "30m")
    _quant_panel(axes[1, 1], cs, "h_max", scen)
    _moment_panel(axes[1, 2], cs, "h_max", scen)
    fig.tight_layout()
    save(fig, "fig72_depth_consistency.png")


def fig_speed_consistency():
    import premodel_lib as L

    cs = _cs()
    scen = "100a"
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 6.6))
    _scatter_panel(axes[0, 0], cs, "speed", scen, "10m")
    _scatter_panel(axes[0, 1], cs, "speed", scen, "30m")
    vel = load(PRE / "velocity_correlation.json")
    k = vel["scenarios"][scen]["frame"]
    h = L.load_flood("2m", scen, "h", times=k)
    ux = L.load_flood("2m", scen, "hux", times=k)
    vy = L.load_flood("2m", scen, "hvy", times=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = np.clip(np.sqrt(ux ** 2 + vy ** 2) / np.where(h > 0.02, h, np.nan),
                      0, 20).astype("float32")
    del h, ux, vy
    _density_panel(axes[0, 2], cs, "speed", scen, ref)
    del ref
    _fit_panel(axes[1, 0], cs, "speed", scen, "30m")
    _quant_panel(axes[1, 1], cs, "speed", scen)
    _moment_panel(axes[1, 2], cs, "speed", scen)
    fig.tight_layout()
    save(fig, "fig73_speed_consistency.png")


# ---------------------------------------------------------------- fig 74
def fig_vmeasure():
    cs = _cs()
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.7))
    for ax, var, lab in ((axes[0], "h_max", "Depth"),
                         (axes[1], "speed", "Speed")):
        M = np.array([[cs["vmeasure"]["variables"][var]["pairwise"][a][b]["v"]
                       for b in ALLRES_FIG] for a in ALLRES_FIG])
        im = ax.imshow(M, cmap="YlGnBu", vmin=0.3, vmax=1.0)
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"{int(a[:-1])}" for a in ALLRES_FIG], fontsize=9)
        ax.set_yticks(range(5))
        ax.set_yticklabels([f"{int(a[:-1])}" for a in ALLRES_FIG], fontsize=9)
        ax.set_xlabel("Grid edge length (m)", fontsize=9)
        ax.set_ylabel("Grid edge length (m)", fontsize=9)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=9,
                        color="#111" if M[i, j] < 0.82 else "white")
        ax.set_title(f"({('a', 'b')[0 if var == 'h_max' else 1]}) {lab} regionalisation agreement",
                     fontsize=9.5, loc="left")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=9)
    ax = axes[2]
    for var, col, mk in (("h_max", "#1f4e79", "o"), ("speed", "#e08b2b", "s")):
        sc = cs["vmeasure"]["variables"][var]["k_diagnostics"]
        ks = sorted(int(k) for k in sc["scores"])
        ax.plot(ks, [sc["scores"][str(k)] for k in ks], marker=mk, color=col,
                lw=1.2, label="Depth" if var == "h_max" else "Speed")
        ax.axvline(sc["chosen_k"], color=col, ls=":", lw=1.0, alpha=0.7)
    ax.set_xlabel("Cluster count K", fontsize=9)
    ax.set_ylabel("Silhouette score (2 m reference field)", fontsize=9)
    ax.set_title("(c) Cluster-count selection for the regionalisation", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    save(fig, "fig74_vmeasure.png")


# ---------------------------------------------------------------- fig 75
def _match_labels(lab, ref):
    """Relabel `lab` so that each of its clusters takes the reference cluster it
    overlaps most, which puts the two plates of a row on one colour code."""
    out = np.full(lab.shape, np.nan, dtype="float32")
    ok = np.isfinite(lab) & np.isfinite(ref)
    for cl in np.unique(lab[ok]):
        sel = ok & (lab == cl)
        vals, cnt = np.unique(ref[sel], return_counts=True)
        out[sel] = float(vals[cnt.argmax()])
    return out


def _support_bbox(mask):
    """Smallest window that holds every support cell, so a plate can be cropped
    to the area its own variable occupies instead of the whole canvas."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0 = int(np.argmax(rows))
    r1 = int(len(rows) - np.argmax(rows[::-1]))
    c0 = int(np.argmax(cols))
    c1 = int(len(cols) - np.argmax(cols[::-1]))
    return r0, r1, c0, c1


def fig_pattern_map():
    """Regionalisation plates, the stability plate and the stability histogram.

    Two things had to be repaired here.  The panel letters came from string
    indexing, which gave `d, e, f, g` to the velocity row and therefore printed
    `d` twice while dropping `h`.  Each row now carries its own four letters.

    The plates were also nearly empty.  The stored rasters are a stride-4
    decimation of the two-metre grid, and on that lattice the common wet support
    is 53 568 cells for depth and only 6 026 for speed, so the speed maps were a
    grey field with a sprinkle of colour.  Each support cell is now drawn as a
    square marker rather than left to `imshow`, the domain is turned to
    landscape so the portrait shape stops shrinking the plates, and each row is
    cropped to the bounding box of its own support.
    """
    cs = _cs()
    Z = np.load(PRE / "consistency_regions.npz")
    groups = (("h_max", "Depth", ("a", "b", "c", "d")),
              ("speed", "Speed", ("e", "f", "g", "h")))
    fig = plt.figure(figsize=(12.0, 6.1))
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 0.10],
                          width_ratios=[1, 1, 1, 0.62, 0.05],
                          left=0.072, right=0.952, top=0.925, bottom=0.065,
                          hspace=0.24, wspace=0.07)
    stab_cm = plt.get_cmap("RdYlGn").copy()
    stab_cm.set_bad("#f4f4f4")
    dom_styles = plt.matplotlib.colors.ListedColormap(["#e6e6e6"])
    dom_full = np.rot90(np.isfinite(Z["val_h_max"].astype("float32")))
    notes = []
    for i, (var, lab, L) in enumerate(groups):
        ref = np.rot90(Z[f"lab_{var}_2m"].astype("float32"))
        ref[ref < 0] = np.nan
        coast = np.rot90(Z[f"lab_{var}_30m"].astype("float32"))
        coast[coast < 0] = np.nan
        coast = _match_labels(coast, ref)
        stab = np.rot90(Z[f"stab_{var}"].astype("float32"))
        stab[stab < 0] = np.nan
        k = int(cs["vmeasure"]["variables"][var]["k_diagnostics"]["chosen_k"])
        cms = plt.matplotlib.colors.ListedColormap(
            plt.get_cmap("tab10").colors[:k]).copy()
        cms.set_bad("#f4f4f4")
        r0, r1, c0, c1 = _support_bbox(np.isfinite(ref))
        n = int(np.isfinite(ref).sum())
        notes.append((lab, L, n, r0, r1, c0, c1))
        dom = dom_full[r0:r1, c0:c1]
        H, W = dom.shape
        sim = None
        row_ax = None
        for j, (arr, cm, vm, vx, ttl) in enumerate((
                (ref, cms, -0.5, k - 0.5, f"({L[0]}) {lab}2 m reference regionalisation"),
                (coast, cms, -0.5, k - 0.5, f"({L[1]}) {lab}30 m regionalisation"),
                (stab, stab_cm, -0.5, 4.5, f"({L[2]}) {lab}cross-resolution stability"))):
            ax = fig.add_subplot(gs[i, j])
            row_ax = ax if j == 0 else row_ax
            ax.imshow(dom, cmap=dom_styles, vmin=0, vmax=1, aspect="equal",
                      interpolation="nearest")
            a = arr[r0:r1, c0:c1]
            yy, xx = np.nonzero(np.isfinite(a))
            im = ax.scatter(xx, yy, c=a[yy, xx], cmap=cm, vmin=vm, vmax=vx,
                            s=2.6, marker="s", linewidths=0)
            sim = im if j == 2 else sim
            ax.contour(dom.astype("float32"), levels=[0.5], colors="#9c9c9c",
                       linewidths=0.4)
            ax.set_xlim(-0.5, W - 0.5)
            ax.set_ylim(H - 0.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(ttl, fontsize=9.5, loc="left")
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
        row_ax.set_ylabel(f"{lab}\n{n} cells", fontsize=9, labelpad=2)
        del ref, coast, stab
        ax = fig.add_subplot(gs[i, 3])
        sh = cs["vmeasure"]["variables"][var]["stability_share"]
        ks = sorted(int(q) for q in sh)
        ax.bar(ks, [sh[str(q)] * 100 for q in ks],
               color=["#b2182b", "#e08214", "#fdb863", "#a6dba0", "#1b7837"])
        for q in ks:
            ax.text(q, sh[str(q)] * 100 + 1.2, f"{sh[str(q)]*100:.1f}",
                    ha="center", fontsize=9)
        ax.set_xticks(ks)
        ax.set_xticklabels(ks, fontsize=9)
        ax.set_ylim(0, max(sh[str(q)] * 100 for q in ks) * 1.28)
        ax.set_xlabel("Coarse-grid classes matching the 2 m reference", fontsize=9)
        ax.set_ylabel("Area share (%)", fontsize=9)
        ax.set_title(f"({L[3]}) {lab} stability distribution", fontsize=9.5, loc="left")
        ax.tick_params(labelsize=9)
    cax = fig.add_subplot(gs[0:2, 4])
    cb = fig.colorbar(sim, cax=cax, ticks=[0, 1, 2, 3, 4])
    cb.ax.tick_params(labelsize=8.5, length=2)
    cb.set_label("Classes matching the 2 m reference", fontsize=9)
    cax = fig.add_subplot(gs[2, 0:3])
    kd = int(cs["vmeasure"]["variables"]["h_max"]["k_diagnostics"]["chosen_k"])
    cmsd = plt.matplotlib.colors.ListedColormap(
        plt.get_cmap("tab10").colors[:kd])
    cb2 = fig.colorbar(plt.cm.ScalarMappable(
        norm=plt.matplotlib.colors.BoundaryNorm(np.arange(-0.5, kd), kd),
        cmap=cmsd), cax=cax, orientation="horizontal",
        ticks=list(range(kd)))
    cb2.ax.set_xticklabels([str(q + 1) for q in range(kd)])
    cb2.ax.tick_params(labelsize=8.5, length=2)
    cb2.set_label("Region id, shared by both rows", fontsize=9)
    save(fig, "fig75_pattern_map.png")
    for lab, L, n, r0, r1, c0, c1 in notes:
        print(f"  {L[0]}-{L[3]} {lab}: {n} support cells, "
              f"crop rows {r0}-{r1} cols {c0}-{c1} of 1620 x 2880",
              flush=True)


# ---------------------------------------------------------------- fig 76 77 78
PAT_METRICS = [("v", "V-measure = NMI (identical)", "#1f4e79", "o"),
               ("ari", "Adjusted Rand index", "#c0392b", "s"),
               ("fm", "Fowlkes-Mallows", "#1a7a3a", "^"),
               ("kappa", "Cohen's kappa", "#8e5ea2", "P")]


def _pat():
    return _cs()["spatial_pattern"]


def _pair(d, a, b):
    return d.get(f"{a}|{b}") or d[f"{b}|{a}"]


def _by_sep(d, field):
    acc = {}
    for v in d.values():
        acc.setdefault(v["separation"], []).append(v[field])
    xs = sorted(acc)
    return xs, [float(np.mean(acc[s])) for s in xs]


def _band_label(b):
    lo, hi = b["lo"], b["hi"]
    return f"{lo:g}~" if hi is None else f"{lo:g}~{hi:g}"


def fig_pattern_agreement():
    sp = _pat()
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.4))
    for j, (var, lab) in enumerate((("h_max", "Depth"), ("speed", "Speed"))):
        cc = sp["variables"][var]["chance_corrected"]
        M = np.ones((5, 5))
        for i, a in enumerate(ALLRES_FIG):
            for k2, b in enumerate(ALLRES_FIG):
                if i != k2:
                    M[i, k2] = _pair(cc, a, b)["ari"]
        ax = axes[0][j]
        im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=1)
        for i in range(5):
            for k2 in range(5):
                ax.text(k2, i, f"{M[i, k2]:.2f}", ha="center", va="center",
                        fontsize=8.2, color="#111" if M[i, k2] < 0.72 else "white")
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"{int(a[:-1])}" for a in ALLRES_FIG], fontsize=8.5)
        ax.set_yticks(range(5))
        ax.set_yticklabels([f"{int(a[:-1])}" for a in ALLRES_FIG], fontsize=8.5)
        ax.set_xlabel("Grid edge length (m)", fontsize=9)
        ax.set_ylabel("Grid edge length (m)", fontsize=9)
        ax.grid(False)
        ax.set_title(f"({('a', 'b')[j]}) {lab} regionalisation adjusted Rand index",
                     fontsize=9.6, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=8.5)
    for j, (var, lab) in enumerate((("h_max", "Depth"), ("speed", "Speed"))):
        cc = sp["variables"][var]["chance_corrected"]
        ax = axes[1][j]
        for key, name, col, mk in PAT_METRICS:
            xs, ys = _by_sep(cc, key)
            ax.plot(xs, ys, marker=mk, color=col, lw=1.25, ms=4.2, label=name)
        ax.set_xscale("log")
        ax.set_xticks([1.5, 2, 2.5, 3, 4, 5, 6, 10, 15])
        ax.set_xticklabels(["1.5", "2", "2.5", "3", "4", "5", "6", "10", "15"],
                           fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Ratio of the two grid edge lengths", fontsize=9)
        ax.set_ylabel("Agreement index value", fontsize=9)
        ax.set_title(f"({('c', 'd')[j]}) {lab} decay of six indices with resolution separation",
                     fontsize=9.6, loc="left")
        ax.legend(frameon=False, fontsize=7.4, ncol=2, columnspacing=0.8)
    fig.tight_layout()
    save(fig, "fig76_pattern_agreement.png")


def fig_pattern_areal():
    sp = _pat()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))
    for ax, field, ylab, title, mk in (
            (axes[0], "mean_iou", "Mean IoU of the best match",
             "(a) Decay of areal overlap with resolution separation", "o"),
            (axes[1], "area_tv", "Total-variation distance of area shares",
             "(b) Difference in regional area shares", "s")):
        for var, lab, col in (("h_max", "Depth", "#1f4e79"),
                              ("speed", "Speed", "#e08b2b")):
            xs, ys = _by_sep(sp["variables"][var]["areal"], field)
            ax.plot(xs, ys, mk + "-", color=col, lw=1.3, ms=4.4, label=lab)
        ax.set_xscale("log")
        ax.set_xticks([1.5, 2, 2.5, 3, 4, 5, 6, 10, 15])
        ax.set_xticklabels(["1.5", "2", "2.5", "3", "4", "5", "6", "10", "15"],
                           fontsize=8)
        ax.set_xlabel("Ratio of the two grid edge lengths", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=9.6, loc="left")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylim(0, 1.02)
    ax = axes[2]
    k = sp["variables"]["h_max"]["k"]
    for j, r in enumerate(RES):
        d = _pair(sp["variables"]["h_max"]["areal"], "2m", r)
        pci = d["per_cluster_iou"]
        xs = np.arange(len(pci)) + (j - 1.5) * 0.2
        ax.bar(xs, pci, 0.2, color=["#1f4e79", "#2a7fbf", "#e08b2b", "#c0392b"][j],
               label=f"{cells(r)} m")
        for x, v in zip(xs, pci):
            ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.4)
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels([f"cluster {i + 1}" for i in range(k)], fontsize=8.5)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("Cluster index of the depth regionalisation", fontsize=9)
    ax.set_ylabel("Best IoU against the 2 m reference", fontsize=9)
    ax.set_title("(c) IoU of each cluster against the 2 m reference", fontsize=9.6, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, "fig77_pattern_areal.png")


def fig_pattern_boundary():
    sp = _pat()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))
    for ax, field, ylab, title in (
            (axes[0], "f1", "F1 of the wet-front boundary", "(a) Overlap of the wet-front boundary"),
            (axes[1], "mhd_m", "Mean wet-front displacement (m)", "(b) Mean displacement of the wet front"),
            (axes[2], "iou", "IoU of the wet-front boundary", "(c) IoU of the wet-front boundary")):
        for var, lab, col, mk in (("h_max", "Depth", "#1f4e79", "o"),
                                  ("speed", "Speed", "#e08b2b", "s")):
            xs, ys = _by_sep(sp["variables"][var]["boundary"], field)
            ax.plot(xs, ys, mk + "-", color=col, lw=1.3, ms=4.4, label=lab)
        ax.set_xticks([2.5, 5, 10, 15])
        ax.set_xticklabels(["2.5", "5", "10", "15"], fontsize=8.5)
        ax.set_xlabel("Ratio of the coarse edge to 2 m", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=9.6, loc="left")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylim(0, 0.6)
    fig.tight_layout()
    save(fig, "fig78_pattern_boundary.png")


def fig_pattern_autocorr():
    sp = _pat()
    xs = [2, 5, 10, 20, 30]
    series = [("Depth field", "h_max", "field", "#1f4e79", "o"),
              ("Speed field", "speed", "field", "#e08b2b", "s"),
              ("Depth labels", "h_max", "class_labels", "#2a7fbf", "^"),
              ("Speed labels", "speed", "class_labels", "#c0392b", "D")]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))
    for ax, field, ylab, title in (
            (axes[0], "moran", "Moran's I", "(a) Spatial autocorrelation vs grid size"),
            (axes[1], "geary", "Geary's C", "(b) Neighbour-difference ratio vs grid size")):
        for lab, var, kind, col, mk in series:
            ys = [sp["variables"][var]["autocorrelation"][kind][r][field]
                  for r in ALLRES_FIG]
            ax.plot(xs, ys, mk + "-", color=col, lw=1.25, ms=4.2, label=lab)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs], fontsize=8.5)
        ax.set_xlabel("Grid edge length (m)", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=9.6, loc="left")
        ax.legend(frameon=False, fontsize=7.6)
    ax = axes[2]
    for var, lab, col, mk in (("h_max", "Depth", "#1f4e79", "o"),
                              ("speed", "Speed", "#e08b2b", "s")):
        zs = [sp["variables"][var]["join_count"][r]["z"] for r in ALLRES_FIG]
        rs = [sp["variables"][var]["join_count"][r]["ratio"] for r in ALLRES_FIG]
        ax.plot(xs, zs, mk + "-", color=col, lw=1.3, ms=4.4, label=f"{lab} z-score")
        for x, zz, rr in zip(xs, zs, rs):
            ax.text(x, zz + 0.8, f"{rr:.2f}", ha="center", fontsize=7, color=col)
    ax.axhline(0, color="#888888", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], fontsize=8.5)
    ax.set_xlabel("Grid edge length (m)", fontsize=9)
    ax.set_ylabel("z-score of same-label adjacency", fontsize=9)
    ax.set_title("(c) Clustering of label adjacency", fontsize=9.6, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "fig79_pattern_autocorr.png")


def fig_pattern_distance():
    sp = _pat()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))
    for j, (var, vlab, unit) in enumerate((("h_max", "Depth", "Depth (m)"),
                                           ("speed", "Speed", "Speed (m/s)"))):
        ax = axes[j]
        b = sp["variables"][var]["distance_resolved"]["value_bands"]
        xs = np.arange(len(b))
        ax.bar(xs, [q["mean_stability"] for q in b], 0.62,
               color="#e08b2b", alpha=0.9)
        for x, q in zip(xs, b):
            ax.text(x, q["mean_stability"] + 0.02, f"{q['mean_stability']:.2f}",
                    ha="center", fontsize=7.4)
        ax.set_xticks(xs)
        ax.set_xticklabels([_band_label(q) for q in b], fontsize=7.6, rotation=30)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel(f"2 m reference {unit} bin", fontsize=9)
        ax.set_ylabel("Mean share of classes matching the 2 m reference", fontsize=9)
        ax.set_title(f"({('a', 'b')[j]}) {vlab} stability by value bin",
                     fontsize=9.6, loc="left")
    ax = axes[2]
    for var, lab, col, mk in (("h_max", "Depth", "#1f4e79", "o"),
                              ("speed", "Speed", "#e08b2b", "s")):
        b = sp["variables"][var]["distance_resolved"]["dist_water_bands"]
        ax.plot(range(len(b)), [q["mean_stability"] for q in b], mk + "-",
                color=col, lw=1.3, ms=4.4, label=lab)
        ax.set_xticks(range(len(b)))
        ax.set_xticklabels([_band_label(q) for q in b], fontsize=7.6, rotation=30)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Distance to water at 2 m (m), binned", fontsize=9)
    ax.set_ylabel("Mean share of classes matching the 2 m reference", fontsize=9)
    ax.set_title("(c) Stability by distance to water", fontsize=9.6, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "fig80_pattern_distance.png")


# --------------------------------------------------- fig 81..85  extra tests
def _extra():
    """The stage-11 payload: variogram, scale space, contiguity, K sweep, SABRE."""
    return load(PRE / "pattern_extra.json")


RES_COL = {"2m": "#111111", "5m": "#1f4e79", "10m": "#2a7fbf",
           "20m": "#e08b2b", "30m": "#c0392b"}
RES_MK = {"2m": "o", "5m": "s", "10m": "^", "20m": "D", "30m": "v"}


def fig_pattern_variogram():
    ex = _extra()
    V = ex["variogram"]["variables"]["h_max"]
    S = ex["scale_space"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))

    ax = axes[0]
    for r in ALLRES_FIG:
        e = V[r]["log_depth"]
        lags = np.array(e["lags"], dtype="float64")
        gam = np.array(e["gamma"], dtype="float64")
        ax.plot(lags, gam, RES_MK[r] + "-", color=RES_COL[r], lw=1.1, ms=3.4,
                label=f"{int(r[:-1])} m grid")
        f = e["fit"]
        if np.isfinite(f["nugget"]) and np.isfinite(f["decay_m"]):
            hh = np.geomspace(lags[0], lags[-1], 200)
            ax.plot(hh, f["nugget"] + (f["sill"] - f["nugget"])
                    * (1.0 - np.exp(-hh / f["decay_m"])),
                    color=RES_COL[r], lw=0.8, ls="--", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("Spatial lag h (m)", fontsize=9)
    ax.set_ylabel("Semivariogram gamma(h) (variance of log depth)", fontsize=9)
    ax.set_title("(a) Empirical semivariogram of log depth and the exponential fit", fontsize=9.6, loc="left")
    ax.legend(frameon=False, fontsize=7.0, ncol=2, columnspacing=0.7)

    ax = axes[1]
    xs = [int(r[:-1]) for r in ALLRES_FIG]
    rho = [V[r]["log_depth"]["fit"]["nugget_over_var"] for r in ALLRES_FIG]
    r95 = [V[r]["log_depth"]["fit"]["range95_m"] for r in ALLRES_FIG]
    ax.plot(xs, rho, "o-", color="#1f4e79", lw=1.3, ms=4.4, label="First-lag relative roughness")
    for x, v in zip(xs, rho):
        ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=7.4, color="#1f4e79")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Grid edge length (m)", fontsize=9)
    ax.set_ylabel("First-lag semivariance / total variance", fontsize=9, color="#1f4e79")
    ax.tick_params(axis="y", labelcolor="#1f4e79")
    ax2 = ax.twinx()
    ax2.plot(xs, r95, "s--", color="#c0392b", lw=1.3, ms=4.4,
             label="Lag reaching 95% of the total variance")
    ax2.set_ylabel("Lag reaching 95% of the total variance (m)", fontsize=9, color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.4, loc="lower right")
    ax.set_title("(b) First-lag roughness and effective range vs grid size", fontsize=9.6, loc="left")

    ax = axes[2]
    xs2 = [q["scale_m"] for q in S["scales"]]
    dp = [q["dispersion_frac"] for q in S["scales"]]
    bv = [q["block_var_frac"] for q in S["scales"]]
    ax.plot(xs2, dp, "o-", color="#c0392b", lw=1.3, ms=4.4,
            label="Variance share not explained by block means")
    ax.axhline(1.0, color="#888888", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(xs2)
    ax.set_xticklabels([str(x) for x in xs2], fontsize=7.4, rotation=45)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Aggregation block edge (m)", fontsize=9)
    ax.set_ylabel("Variance share not explained by block means", fontsize=9, color="#c0392b")
    ax.tick_params(axis="y", labelcolor="#c0392b")
    ax2 = ax.twinx()
    ax2.plot(xs2, bv, "s--", color="#1f4e79", lw=1.3, ms=4.4,
             label="Variance share retained by block means")
    ax2.set_ylabel("Variance share retained by block means", fontsize=9, color="#1f4e79")
    ax2.tick_params(axis="y", labelcolor="#1f4e79")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.4, loc="center left")
    ax.set_title("(c) Scale space: how far block means explain local depth", fontsize=9.6, loc="left")
    fig.tight_layout()
    save(fig, "fig82_variogram_scale.png")


def fig_pattern_contiguity_maps():
    ex = _extra()
    cm = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 5.4))
    for i, (var, lab, L) in enumerate((("h_max", "Depth", ("a", "b", "c")),
                                       ("speed", "Speed", ("d", "e", "f")))):
        Z = np.load(PRE / f"contiguity_core_{var}.npz")
        row = Z["row"].astype("int64")
        col = Z["col"].astype("int64")
        r0, c0 = int(row.min()), int(col.min())
        yy, xx = row - r0, col - c0
        k = int(ex["contiguity"]["variables"][var]["k"])
        cols = plt.matplotlib.colors.ListedColormap(cm.colors[:k])
        for j, (key, ttl, cma, vmx) in enumerate((
                ("unconstrained", f"({L[0]}) {lab}2 m clusters, unconstrained k-means", cols, k - 0.5),
                ("constrained_2m", f"({L[1]}) {lab}2 m clusters, adjacency-constrained Ward", cols, k - 0.5),
                (None, f"({L[2]}) {lab}label agreement between the two partitions", None, None))):
            ax = axes[i][j]
            if key is None:
                same = (Z["unconstrained"] == Z["constrained_2m"]).astype("float64")
                ax.scatter(xx, yy, c=same, cmap="RdYlGn", vmin=0, vmax=1,
                           s=1.1, marker="s", linewidths=0)
            else:
                ax.scatter(xx, yy, c=Z[key].astype("float64"), cmap=cma,
                           vmin=-0.5, vmax=vmx, s=1.1, marker="s", linewidths=0)
            ax.set_xlim(xx.min() - 4, xx.max() + 4)
            ax.set_ylim(yy.max() + 4, yy.min() - 4)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(ttl, fontsize=8.6, loc="left")
    fig.suptitle("Regionalisation on the largest connected core: unconstrained vs adjacency-constrained", fontsize=10.2, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, "fig83_contiguity_map.png")


def fig_pattern_contiguity_metrics():
    ex = _extra()
    C = ex["contiguity"]["variables"]
    pairs = ["2m|5m", "2m|10m", "2m|20m", "2m|30m", "5m|10m",
             "5m|20m", "5m|30m", "10m|20m", "10m|30m", "20m|30m"]
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.4))
    for i, (var, lab) in enumerate((("h_max", "Depth"), ("speed", "Speed"))):
        rec = C[var]
        for j, (field, ylab, ttl) in enumerate((
                ("v", "V-measure", "Resolution pair"),
                ("mean_iou", "Best-match mean IoU", "Resolution pair"))):
            ax = axes[i][j]
            xs = np.arange(len(pairs))
            u = [rec["pairs"]["unconstrained"][p][field] for p in pairs]
            c = [rec["pairs"]["constrained"][p][field] for p in pairs]
            ax.bar(xs - 0.19, u, 0.38, color="#9fb6cd", label="Unconstrained k-means")
            ax.bar(xs + 0.19, c, 0.38, color="#1f4e79", label="Adjacency-constrained Ward")
            ax.set_xticks(xs)
            ax.set_xticklabels([p.replace("|", "\nvs ") for p in pairs], fontsize=6.2)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel(ylab, fontsize=9)
            ax.set_title(f"({('a', 'b', 'c', 'd', 'e', 'f')[i * 3 + j]}) {lab} {ylab}",
                         fontsize=9.2, loc="left")
            ax.legend(frameon=False, fontsize=7.4)
        ax = axes[i][2]
        xs = np.arange(3)
        us = rec["cluster_share"]["2m"]["unconstrained"]
        cs = rec["cluster_share"]["2m"]["constrained"]
        ax.bar(xs - 0.19, us, 0.38, color="#9fb6cd", label="Unconstrained k-means")
        ax.bar(xs + 0.19, cs, 0.38, color="#1f4e79", label="Adjacency-constrained Ward")
        for x, (a, b) in enumerate(zip(us, cs)):
            ax.text(x - 0.19, a + 0.015, f"{a:.2f}", ha="center", fontsize=7)
            ax.text(x + 0.19, b + 0.015, f"{b:.2f}", ha="center", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"cluster {q + 1}" for q in xs], fontsize=8.5)
        ax.set_ylim(0, 0.9)
        ax.set_ylabel("Area share of the 2 m clusters", fontsize=9)
        ax.set_title(f"({('a', 'b', 'c', 'd', 'e', 'f')[i * 3 + 2]}) {lab}2 m cluster share",
                     fontsize=9.2, loc="left")
        ax.legend(frameon=False, fontsize=7.4)
    fig.tight_layout()
    save(fig, "fig84_contiguity_metrics.png")


K_PAIRS = ["5m|10m", "10m|20m", "2m|30m"]


def fig_pattern_ksweep():
    ex = _extra()
    K = ex["k_sensitivity"]
    ks = K["k_grid"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    ax = axes[0][0]
    for var, lab, col, mk in (("h_max", "Depth", "#1f4e79", "o"),
                              ("speed", "Speed", "#e08b2b", "s")):
        sil = K["variables"][var]["silhouette"]
        ax.plot(ks, [sil[str(k)] for k in ks], mk + "-", color=col, lw=1.3,
                ms=4.4, label=lab)
    ax.axvline(K["chosen_k"], color="#888888", lw=0.9, ls=":")
    ax.text(K["chosen_k"] + 0.08, ax.get_ylim()[0], f"chosen K = {K['chosen_k']}",
            fontsize=7.6, color="#555555", va="bottom")
    ax.set_xticks(ks)
    ax.set_xlabel("Cluster count K", fontsize=9)
    ax.set_ylabel("Silhouette score on the 2 m reference field", fontsize=9)
    ax.set_title("(a) Basis for the cluster count: silhouette score", fontsize=9.6, loc="left")
    ax.legend(frameon=False, fontsize=8)
    for j, (var, lab) in enumerate((("h_max", "Depth"), ("speed", "Speed"))):
        ax = axes[0][1] if j == 0 else axes[1][0]
        rec = K["variables"][var]["per_k"]
        allp = list(rec[str(K["chosen_k"])].keys())
        for p in allp:
            ys = [rec[str(k)][p]["v"] for k in ks]
            if p in K_PAIRS:
                col = {"5m|10m": "#1f4e79", "10m|20m": "#e08b2b",
                       "2m|30m": "#c0392b"}[p]
                ax.plot(ks, ys, "o-", color=col, lw=1.4, ms=4.0,
                        label=p.replace("|", " m vs ") + " m")
            else:
                ax.plot(ks, ys, "-", color="#c9c9c9", lw=0.8, zorder=1)
        ax.set_xticks(ks)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Cluster count K", fontsize=9)
        ax.set_ylabel("V-measure", fontsize=9)
        ax.set_title(f"({'bc'[j]}) {lab} V-measure vs cluster count for each resolution pair",
                     fontsize=9.6, loc="left")
        ax.legend(frameon=False, fontsize=7.6)
    ax = axes[1][1]
    fam = ["v", "ari", "fm", "kappa", "ami", "mean_iou", "area_tv"]
    cmap = plt.get_cmap("tab10")
    for q, mt in enumerate(fam):
        st = K["variables"]["h_max"]["stability"][mt]["spearman_vs_k3"]
        ax.plot(ks, [st[str(k)] for k in ks], "o-", color=cmap.colors[q],
                lw=1.2, ms=3.6, label=mt)
    ax.axhline(1.0, color="#888888", lw=0.8, ls=":")
    ax.set_xticks(ks)
    ax.set_xlabel("Cluster count K", fontsize=9)
    ax.set_ylabel("Spearman rank correlation vs K = 3", fontsize=9)
    ax.set_title("(d) Rank stability of the ten depth resolution pairs", fontsize=9.6, loc="left")
    ax.legend(frameon=False, fontsize=7.2, ncol=2, columnspacing=0.8)
    fig.tight_layout()
    save(fig, "fig85_k_sensitivity.png")


def fig_pattern_sabre():
    ex = _extra()
    sc = ex["sabre_crosscheck"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax = axes[0]
    lo = 1.0
    hi = 0.0
    for var, lab, col, mk in (("h_max", "Depth", "#1f4e79", "o"),
                              ("speed", "Speed", "#e08b2b", "s")):
        rec = sc["variables"][var]
        x = [q["python_v"] for q in rec.values()]
        y = [q["sabre_v"] for q in rec.values()]
        lo = min(lo, min(x))
        hi = max(hi, max(x))
        ax.plot(x, y, mk, color=col, ms=5.0, label=f"{lab} ({len(x)} pairs)")
    ax.plot([lo, hi], [lo, hi], color="#888888", lw=0.9, ls="--",
            label="1:1 reference line")
    ax.set_xlabel("V-measure computed directly from the definition", fontsize=9)
    ax.set_ylabel("V-measure from SABRE 0.4.3", fontsize=9)
    ax.set_title("(a) Pairwise comparison of the two implementations", fontsize=9.6, loc="left")
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    ax.text(0.98, 0.06, f"max abs diff {sc['max_abs_diff']:.2e}",
            transform=ax.transAxes, ha="right", fontsize=8.2, color="#c0392b")
    ax = axes[1]
    xs = []
    ys = []
    cs = []
    for var, col in (("h_max", "#1f4e79"), ("speed", "#e08b2b")):
        for p, q in sc["variables"][var].items():
            xs.append(f"{var[0]}{p.replace('|', '/')}")
            ys.append(max(q["abs_diff"], 1e-18))
            cs.append(col)
    ax.bar(range(len(xs)), ys, 0.7, color=cs)
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e-12)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, fontsize=5.6, rotation=90)
    ax.axhline(np.finfo(np.float64).eps, color="#c0392b", lw=0.9, ls="--")
    ax.text(0.02, np.finfo(np.float64).eps * 1.6,
            f"double-precision eps {np.finfo(np.float64).eps:.1e}",
            transform=ax.get_yaxis_transform(), fontsize=7.4, color="#c0392b")
    ax.set_xlabel("Variable and resolution pair", fontsize=9)
    ax.set_ylabel("Absolute V-measure difference (log scale)", fontsize=9)
    ax.set_title("(b) Absolute difference of every pair", fontsize=9.6, loc="left")
    fig.tight_layout()
    save(fig, "fig81_sabre_crosscheck.png")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["all"]
    fns = {"task": fig_task, "grid": fig_error_grid, "depth": fig_depth_bins,
           "slope": fig_slope_bins, "terrain": fig_terrain, "maps": fig_maps,
           "cross": fig_cross_res, "scatter": fig_tile_scatter, "decomp": fig_decomp,
           "premodel": fig_premodel, "dialect": fig_dialect, "zoom": fig_zoom,
           "pattern": fig_terrain_flood, "depth2": fig_depth_consistency,
           "speed2": fig_speed_consistency, "vmeasure": fig_vmeasure,
           "patternmap": fig_pattern_map, "patagree": fig_pattern_agreement,
           "patareal": fig_pattern_areal, "patboundary": fig_pattern_boundary,
           "patautocorr": fig_pattern_autocorr,
           "patdistance": fig_pattern_distance,
           "patvario": fig_pattern_variogram,
           "patcontigmap": fig_pattern_contiguity_maps,
           "patcontigmet": fig_pattern_contiguity_metrics,
           "patksweep": fig_pattern_ksweep,
           "patsabre": fig_pattern_sabre}
    for w in which:
        if w == "all":
            for f in fns.values():
                f()
        else:
            fns[w]()
