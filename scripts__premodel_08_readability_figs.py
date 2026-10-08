r"""Stage 8: readability figures for chapter 2 (SciencePlots style).

  fig61_two_fields.png      the two parallel fields and one real ten-metre block
  fig62_error_accounts.png  the three-part error accounting and the variance shares
  fig63_error_sources.png   which conditions make the error larger

Reads only artefacts that already exist.  No model is trained and no simulation
is re-run.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report_figs"
OUT.mkdir(parents=True, exist_ok=True)
PRE = ROOT / "outputs" / "premodel"

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
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
     "20a": "#2a7fbf", "100a": "#c0392b", "fine": "#2a7fbf", "coarse": "#c0392b",
     "sim": "#8e5ea2", "agg": "#1a7a3a", "low": "#2a7fbf", "mid": "#9aa0a6",
     "high": "#c0392b", "gray": "#6b6b6b"}
RES = ["5m", "10m", "20m", "30m"]
SCEN = ["20a", "100a"]


def load(name):
    return json.loads((PRE / name).read_text(encoding="utf-8"))


CELL = load("cell_example.json")
CELF = load("cell_contrasts.json")
TILF = load("factor_contrasts.json")
FL = load("flood_error.json")


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name, flush=True)


def _box(ax, x, y, w, h, text, fc="#ffffff", ec="#333333", fs=9.5, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=lw, edgecolor=ec, facecolor=fc,
                                zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3, linespacing=1.45)


def _arrow(ax, xy1, xy2, text="", fs=8.6, color="#333333", rad=0.0,
           dx=0.0, dy=0.03, ls="-"):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.1, color=color, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}",
                                 linestyle=ls))
    if text:
        ax.text((xy1[0] + xy2[0]) / 2 + dx, (xy1[1] + xy2[1]) / 2 + dy, text,
                ha="center", va="bottom", fontsize=fs, color=color, zorder=3)


def _grid(ax, x, y, w, h, n, fc, ec, lw=0.7):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=ec,
                           linewidth=lw * 1.5, zorder=2))
    for k in range(1, n):
        ax.plot([x + w * k / n] * 2, [y, y + h], color=ec, lw=lw, zorder=2)
        ax.plot([x, x + w], [y + h * k / n] * 2, color=ec, lw=lw, zorder=2)
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="none",
                           alpha=0.35, zorder=1))


# ------------------------------------------------------------------- fig 61
def fig_two_fields():
    fig = plt.figure(figsize=(12.4, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.0, 1.0], wspace=0.32)

    # (a) the two parallel fields on one lattice
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _box(ax, 0.01, 0.60, 0.20, 0.30, "2 m reference\n(fine grid)", fc="#eaf2fb")
    _grid(ax, 0.03, 0.635, 0.16, 0.17, 4, C["fine"], "#4a6b8a")
    _box(ax, 0.01, 0.10, 0.20, 0.30, "Coarse hydrodynamic solver\n(solved directly)", fc="#fdeceb")
    _grid(ax, 0.032, 0.135, 0.158, 0.15, 2, C["coarse"], "#8a4a4a")
    _arrow(ax, (0.21, 0.75), (0.44, 0.75), "Area-weighted mean\n(aggregation)", dy=0.02)
    _arrow(ax, (0.21, 0.25), (0.44, 0.25), "Solved directly\non the coarse grid", dy=0.02)
    _box(ax, 0.45, 0.57, 0.22, 0.36,
         "Aggregated truth\n\narea-average 25 two-metre\ncells into one value", fc="#eaf7ee")
    _grid(ax, 0.495, 0.60, 0.13, 0.13, 2, C["agg"], "#3f7a55")
    _box(ax, 0.45, 0.07, 0.22, 0.36,
         "Coarse simulation\n\none depth per\n10 m block", fc="#f3ecf7")
    _grid(ax, 0.495, 0.10, 0.13, 0.13, 2, C["sim"], "#6b4a7a")
    _arrow(ax, (0.67, 0.75), (0.78, 0.60), "")
    _arrow(ax, (0.67, 0.25), (0.78, 0.40), "")
    _box(ax, 0.79, 0.35, 0.20, 0.30,
         "Same blocks\n\nsame location\nso the two values can be subtracted", fc="#f2f2f2",
         ec="#555555")
    ax.set_title("(a) The two fields share the same ground and the same rain event", fontsize=10.5,
                 loc="left")

    # (b) the real ten-metre block
    ax = fig.add_subplot(gs[0, 1])
    sub = np.array(CELL["sub_values_m"], dtype=float)
    im = ax.imshow(sub, cmap="Blues", vmin=0)
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = sub[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.6,
                    color="#111111" if v < 1.6 else "#ffffff")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f"{CELL['col_span_2m'][0]+k}" for k in range(5)],
                       fontsize=7)
    ax.set_yticklabels([f"{CELL['row_span_2m'][0]+k}" for k in range(5)],
                       fontsize=7)
    ax.set_xlabel("2 m grid column index")
    ax.set_ylabel("2 m grid row index")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("2 m depth (m)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title("(b) The 25 two-metre depths inside one real 10 m block", fontsize=10.5,
                 loc="left")
    ax.text(0.0, -0.32, f"10 m row {CELL['row']}, col {CELL['col']}, "
                        f"100-year event", transform=ax.transAxes, fontsize=8.2,
            color="#444444")

    # (c) the three numbers on that same block
    ax = fig.add_subplot(gs[0, 2])
    y = np.arange(sub.size)
    ax.plot(sub.ravel(), y, "o", ms=4.2, color=C["fine"], alpha=0.85,
            label="25 two-metre cells")
    ax.axvline(CELL["area_weighted_aggregate_m"], color=C["agg"], lw=1.6,
               label=f"aggregated truth {CELL['area_weighted_aggregate_m']:.3f} m")
    ax.axvline(CELL["coarse_native_m"], color=C["sim"], lw=1.6, ls="--",
               label=f"coarse simulation {CELL['coarse_native_m']:.3f} m")
    lo = CELL["coarse_native_m"]
    hi = CELL["area_weighted_aggregate_m"]
    ax.axvspan(min(lo, hi), max(lo, hi), color="#c0392b", alpha=0.13, lw=0)
    ax.annotate(f"difference\n{abs(hi-lo):.3f} m",
                xy=((lo + hi) / 2, sub.size * 0.62),
                xytext=((lo + hi) / 2 + 0.75, sub.size * 0.78),
                fontsize=8.6, color="#8a2b20", ha="left",
                arrowprops=dict(arrowstyle="-", color="#8a2b20", lw=0.9))
    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("Two-metre cells in the block (sorted by depth)")
    ax.set_xlim(-0.05, sub.max() * 1.22)
    ax.set_ylim(-1, sub.size)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("(c) Aggregated value, simulated value and their difference on one block", fontsize=10.5,
                 loc="left")

    save(fig, "fig61_two_fields.png")


# ------------------------------------------------------------------- fig 62
def fig_error_accounts():
    fig = plt.figure(figsize=(12.4, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.22)

    d = FL["resolutions"]["100a"]["30m"]["error_decomp_2m"]
    sim, agg, tot = d["rmse_simulation"], d["rmse_aggregation"], d["rmse_total"]

    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rows = [
        (0.68, "Coarse simulation - aggregated truth", "Simulation term: numerical solve and block shape",
         f"at 30 m, 100-year: {sim:.4f} m", "#f3ecf7", C["sim"]),
        (0.40, "Aggregated truth - 2 m truth", "Aggregation term: coarsening itself",
         f"same condition: {agg:.4f} m", "#eaf7ee", C["agg"]),
    ]
    for y, expr, note, num, fc, ec in rows:
        _box(ax, 0.02, y, 0.46, 0.22, f"{expr}\n\n{note}\n{num}", fc=fc, ec=ec,
             fs=9.0)
    _box(ax, 0.52, 0.54, 0.46, 0.36,
         "Add the two lines\n\nthe middle terms cancel\nleaving the total error\n"
         "Coarse simulation - 2 m truth",
         fc="#f2f2f2", ec="#555555", fs=9.4)
    ax.text(0.75, 0.42, f"at 30 m, 100-year: {tot:.4f} m", ha="center",
            va="top", fontsize=9.0, color="#333333")
    _arrow(ax, (0.48, 0.79), (0.52, 0.76), "")
    _arrow(ax, (0.48, 0.51), (0.52, 0.54), "")
    _box(ax, 0.02, 0.06, 0.96, 0.26,
         "The two terms are about 76:24, but they are two legs of one difference and must not be added directly\n"
         f"quadrature sum {np.sqrt(sim**2 + agg**2):.4f} m vs total {tot:.4f} m",
         fc="#fdf6e3", ec="#b08800", fs=9.0)
    ax.set_title("(a) The error splits into two terms that sum to the total", fontsize=10.5,
                 loc="left")

    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(RES))
    w = 0.36
    for k, scen in enumerate(SCEN):
        share_s, share_a, share_c = [], [], []
        for r in RES:
            m = FL["resolutions"][scen][r]["error_decomp_2m"]
            share_s.append(m["share_simulation_var"] * 100)
            share_a.append(m["share_aggregation_var"] * 100)
            share_c.append(m["share_cross_var"] * 100)
        pos = x + (k - 0.5) * w
        ax.bar(pos, share_s, w * 0.94, color=C["sim"],
               label="Simulation-term variance share" if k == 0 else None)
        ax.bar(pos, share_a, w * 0.94, bottom=share_s, color=C["agg"],
               label="Aggregation-term variance share" if k == 0 else None)
        alpha = 1.0 if scen == "100a" else 0.6
        for p, s, a in zip(pos, share_s, share_a):
            ax.text(p, s + a + 2.4, f"{a:.1f}", ha="center", fontsize=7.4,
                    color="#1a7a3a", alpha=alpha)
        ax.text(pos.mean() + 0.0, 108, "20a" if scen == "20a" else "100a",
                ha="center", fontsize=8.2, color=C[scen])
        for b in ax.patches[-8:]:
            b.set_alpha(alpha)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r[:-1]}" for r in RES])
    ax.set_xlabel("Grid edge length (m)")
    ax.set_ylabel("Share of the total squared error (%)")
    ax.set_ylim(0, 116)
    ax.legend(frameon=False, ncol=2, loc="lower center")
    ax.text(0.02, 0.96, "The cross term stays below 6e-4 on all four grids and is not drawn separately",
            transform=ax.transAxes, fontsize=8.0, color="#555555", va="top")
    ax.set_title("(b) Share of each term as the grid coarsens", fontsize=10.5,
                 loc="left")

    save(fig, "fig62_error_accounts.png")


# ------------------------------------------------------------------- fig 63
PAIRS_CELL = [
    ("Slope", "Cell slope (dimensionless)", ["Gentle", "Medium", "Steep"]),
    ("Building_binary", "Cell building coverage", ["No building", "Partial", "Full coverage"]),
    ("Dist_water", "Cell distance to water (m)", ["Near", "Medium", "Far"]),
]
PAIRS_TILE = [
    ("building_frac", "Tile building fraction", ["Low", "Medium", "High"]),
    ("impervious_frac", "Tile impervious fraction", ["Low", "Medium", "High"]),
    ("landuse_entropy", "Tile land-use mixing", ["Uniform", "Medium", "Mixed"]),
]

# The contrast JSONs carry a Chinese condition name in their "cn" field.  In-figure
# text must be English, so map the condition keys to short English labels here
# rather than editing the data artefacts.
EN_FACTOR = {
    "Slope": "Cell slope",
    "Building_binary": "Building coverage",
    "Dist_water": "Distance to water",
    "building_frac": "Building fraction",
    "impervious_frac": "Impervious fraction",
    "landuse_entropy": "Land-use mixing",
}


def factor_label(entry, key):
    return EN_FACTOR.get(key, entry.get("cn", key))


def _tercile(v, q1, q2):
    return np.where(v <= q1, 0, np.where(v >= q2, 1, 2))


def fig_error_sources():
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.2))
    colors = [C["low"], C["mid"], C["high"]]

    def panel(ax, tag, vals, depths, labels, xlab, title, ylab, dep_lab):
        xs = np.arange(len(vals))
        ax.bar(xs, vals, 0.62, color=colors[:len(vals)], alpha=0.9)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.025, f"{v:.3f}", ha="center", fontsize=8.4)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, max(vals) * 1.34)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10.0, loc="left")
        ax2 = ax.twinx()
        ax2.plot(xs, depths, "--o", color="#6b6b6b", ms=4.6, lw=1.1)
        ax2.set_ylabel(dep_lab, color="#4a4a4a", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#4a4a4a")
        ax2.set_ylim(0, max(depths) * 2.8)
        ax2.grid(False)
        return ax2

    rec = CELF["resolutions"]["10m"]["100a"]
    letters = "abcdef"
    for k, (key, xlab, labels) in enumerate(PAIRS_CELL):
        e = rec["factors"][key]
        vals = [e[t]["mae"] for t in ("low", "mid", "high")]
        dep = [e[t]["mean_truth_m"] for t in ("low", "mid", "high")]
        a2 = panel(axes[0][k], key, vals, dep, labels, xlab,
                   f"({letters[k]}) {factor_label(e, key)} high group / low group "
                   f"{e['mae_ratio_high_over_low']:.2f}x",
                   "MAE (m)", "Group-mean depth (m)")
        if k == 2:
            a2.legend(handles=[Line2D([], [], color="#6b6b6b", ls="--", marker="o",
                                      label="Group-mean depth (right axis)")],
                      frameon=False, fontsize=8, loc="upper left")

    tf = TILF["resolutions"]["10m"]["factors"]
    for k, (key, xlab, labels) in enumerate(PAIRS_TILE):
        e = tf[key]
        vals = [e[t]["mae"]["mean"] for t in ("low", "mid", "high")]
        dep = [e[t]["mean_depth_2m"]["mean"] for t in ("low", "mid", "high")]
        a2 = panel(axes[1][k], key, vals, dep, labels, xlab,
                   f"({letters[k+3]}) {factor_label(e, key)} high group / low group "
                   f"{e['mae_ratio_high_over_low']:.2f}x",
                   "Per-tile MAE (m)", "Group-mean depth (m)")
        if k == 0:
            a2.legend(handles=[Line2D([], [], color="#6b6b6b", ls="--", marker="o",
                                      label="Group-mean depth (right axis)")],
                      frameon=False, fontsize=8, loc="upper left")

    fig.text(0.012, 0.762, "Cell level (wet cells only)", rotation=90,
             va="center", fontsize=10, color="#333333")
    fig.text(0.012, 0.285, "Tile level (480 m square)", rotation=90,
             va="center", fontsize=10, color="#333333")
    fig.suptitle("10 m grid, 100-year event: high-group vs low-group error per condition",
                 fontsize=11, y=0.985)
    fig.tight_layout(rect=(0.028, 0, 1, 0.962))
    save(fig, "fig63_error_sources.png")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["all"]
    fns = {"fields": fig_two_fields, "accounts": fig_error_accounts,
           "sources": fig_error_sources}
    for w in which:
        if w == "all":
            for f in fns.values():
                f()
        else:
            fns[w]()
