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
    _box(ax, 0.01, 0.60, 0.20, 0.30, "两米参考结果\n（精细网格）", fc="#eaf2fb")
    _grid(ax, 0.03, 0.635, 0.16, 0.17, 4, C["fine"], "#4a6b8a")
    _box(ax, 0.01, 0.10, 0.20, 0.30, "粗网格水动力程序\n（直接求解）", fc="#fdeceb")
    _grid(ax, 0.032, 0.135, 0.158, 0.15, 2, C["coarse"], "#8a4a4a")
    _arrow(ax, (0.21, 0.75), (0.44, 0.75), "面积加权平均\n（聚合）", dy=0.02)
    _arrow(ax, (0.21, 0.25), (0.44, 0.25), "在粗网格上\n直接求解", dy=0.02)
    _box(ax, 0.45, 0.57, 0.22, 0.36,
         "聚合真值\n\n把 25 个两米小格\n按面积平均成一个值", fc="#eaf7ee")
    _grid(ax, 0.495, 0.60, 0.13, 0.13, 2, C["agg"], "#3f7a55")
    _box(ax, 0.45, 0.07, 0.22, 0.36,
         "粗网格模拟值\n\n每个十米方块\n算出一个水深", fc="#f3ecf7")
    _grid(ax, 0.495, 0.10, 0.13, 0.13, 2, C["sim"], "#6b4a7a")
    _arrow(ax, (0.67, 0.75), (0.78, 0.60), "")
    _arrow(ax, (0.67, 0.25), (0.78, 0.40), "")
    _box(ax, 0.79, 0.35, 0.20, 0.30,
         "同一套方块\n\n同一个位置\n两份值可以相减", fc="#f2f2f2",
         ec="#555555")
    ax.set_title("(a) 两份资料来自同一片地面与同一个降雨情景", fontsize=10.5,
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
    ax.set_xlabel("两米网格列号")
    ax.set_ylabel("两米网格行号")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("两米水深（米）", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title("(b) 一块真实十米方块内部的 25 个两米水深", fontsize=10.5,
                 loc="left")
    ax.text(0.0, -0.32, f"十米行号 {CELL['row']}，列号 {CELL['col']}，"
                        f"一百年一遇情景", transform=ax.transAxes, fontsize=8.2,
            color="#444444")

    # (c) the three numbers on that same block
    ax = fig.add_subplot(gs[0, 2])
    y = np.arange(sub.size)
    ax.plot(sub.ravel(), y, "o", ms=4.2, color=C["fine"], alpha=0.85,
            label="25 个两米小格")
    ax.axvline(CELL["area_weighted_aggregate_m"], color=C["agg"], lw=1.6,
               label=f"聚合真值 {CELL['area_weighted_aggregate_m']:.3f} 米")
    ax.axvline(CELL["coarse_native_m"], color=C["sim"], lw=1.6, ls="--",
               label=f"粗网格模拟 {CELL['coarse_native_m']:.3f} 米")
    lo = CELL["coarse_native_m"]
    hi = CELL["area_weighted_aggregate_m"]
    ax.axvspan(min(lo, hi), max(lo, hi), color="#c0392b", alpha=0.13, lw=0)
    ax.annotate(f"两者相差\n{abs(hi-lo):.3f} 米",
                xy=((lo + hi) / 2, sub.size * 0.62),
                xytext=((lo + hi) / 2 + 0.75, sub.size * 0.78),
                fontsize=8.6, color="#8a2b20", ha="left",
                arrowprops=dict(arrowstyle="-", color="#8a2b20", lw=0.9))
    ax.set_xlabel("水深（米）")
    ax.set_ylabel("方块内的两米小格（按水深排序）")
    ax.set_xlim(-0.05, sub.max() * 1.22)
    ax.set_ylim(-1, sub.size)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("(c) 同一方块上的聚合值、模拟值与两者之差", fontsize=10.5,
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
        (0.68, "粗网格模拟值　−　聚合真值", "模拟项：数值求解与方块形状造成的差",
         f"三十米、一百年一遇下 {sim:.4f} 米", "#f3ecf7", C["sim"]),
        (0.40, "聚合真值　−　两米真值", "聚合项：粗化本身造成的差",
         f"同一条件下 {agg:.4f} 米", "#eaf7ee", C["agg"]),
    ]
    for y, expr, note, num, fc, ec in rows:
        _box(ax, 0.02, y, 0.46, 0.22, f"{expr}\n\n{note}\n{num}", fc=fc, ec=ec,
             fs=9.0)
    _box(ax, 0.52, 0.54, 0.46, 0.36,
         "两式相加\n\n中间两项相互抵消\n得到总误差\n"
         "粗网格模拟值　−　两米真值",
         fc="#f2f2f2", ec="#555555", fs=9.4)
    ax.text(0.75, 0.42, f"三十米、一百年一遇下 {tot:.4f} 米", ha="center",
            va="top", fontsize=9.0, color="#333333")
    _arrow(ax, (0.48, 0.79), (0.52, 0.76), "")
    _arrow(ax, (0.48, 0.51), (0.52, 0.54), "")
    _box(ax, 0.02, 0.06, 0.96, 0.26,
         "两项之比约为 76 比 24，但它们出自同一个差值的两段，相加时不能直接相加\n"
         f"按平方相加得到 {np.sqrt(sim**2 + agg**2):.4f} 米，与总误差 {tot:.4f} 米一致",
         fc="#fdf6e3", ec="#b08800", fs=9.0)
    ax.set_title("(a) 误差可以拆成两笔，两笔相加等于总差", fontsize=10.5,
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
               label="模拟项方差占比" if k == 0 else None)
        ax.bar(pos, share_a, w * 0.94, bottom=share_s, color=C["agg"],
               label="聚合项方差占比" if k == 0 else None)
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
    ax.set_xlabel("网格边长（米）")
    ax.set_ylabel("占总平方误差的比例（百分数）")
    ax.set_ylim(0, 116)
    ax.legend(frameon=False, ncol=2, loc="lower center")
    ax.text(0.02, 0.96, "交叉项在四档网格上都不超过万分之六，因此图中未单独画出",
            transform=ax.transAxes, fontsize=8.0, color="#555555", va="top")
    ax.set_title("(b) 两笔账各占多少，随网格变粗而变化", fontsize=10.5,
                 loc="left")

    save(fig, "fig62_error_accounts.png")


# ------------------------------------------------------------------- fig 63
PAIRS_CELL = [
    ("Slope", "单元坡度（无量纲）", ["平缓", "中等", "陡峭"]),
    ("Building_binary", "单元建筑覆盖率", ["无建筑", "部分", "满覆盖"]),
    ("Dist_water", "单元到水体的距离（米）", ["靠近", "中等", "远离"]),
]
PAIRS_TILE = [
    ("building_frac", "瓦片内建筑占比", ["少", "中", "多"]),
    ("impervious_frac", "瓦片内不透水面占比", ["少", "中", "多"]),
    ("landuse_entropy", "瓦片土地利用混合度", ["单一", "中等", "混杂"]),
]


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
                   f"({letters[k]}) {e['cn']}　高值组是低值组的 "
                   f"{e['mae_ratio_high_over_low']:.2f} 倍",
                   "平均绝对误差（米）", "组内平均水深（米）")
        if k == 2:
            a2.legend(handles=[Line2D([], [], color="#6b6b6b", ls="--", marker="o",
                                      label="组内平均水深（右轴）")],
                      frameon=False, fontsize=8, loc="upper left")

    tf = TILF["resolutions"]["10m"]["factors"]
    for k, (key, xlab, labels) in enumerate(PAIRS_TILE):
        e = tf[key]
        vals = [e[t]["mae"]["mean"] for t in ("low", "mid", "high")]
        dep = [e[t]["mean_depth_2m"]["mean"] for t in ("low", "mid", "high")]
        a2 = panel(axes[1][k], key, vals, dep, labels, xlab,
                   f"({letters[k+3]}) {e['cn']}　高值组是低值组的 "
                   f"{e['mae_ratio_high_over_low']:.2f} 倍",
                   "逐瓦片平均绝对误差（米）", "组内平均水深（米）")
        if k == 0:
            a2.legend(handles=[Line2D([], [], color="#6b6b6b", ls="--", marker="o",
                                      label="组内平均水深（右轴）")],
                      frameon=False, fontsize=8, loc="upper left")

    fig.text(0.012, 0.762, "单元一级（只统计有水单元）", rotation=90,
             va="center", fontsize=10, color="#333333")
    fig.text(0.012, 0.285, "瓦片一级（480 米见方）", rotation=90,
             va="center", fontsize=10, color="#333333")
    fig.suptitle("十米网格、一百年一遇情景下，各条件取值高组与低组的误差对照",
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
