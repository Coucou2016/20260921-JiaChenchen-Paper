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
        (0.10, "粗网格水动力模拟\n（5/10/20/30 米）\n粗网格模拟水深场", "#dce6f2"),
        (3.05, "前置模型\n误差校正\n（本报告新增）", "#fdf0d8"),
        (6.00, "超分辨率模型\nHydroGeo-SRNO\n（已完成，不修改）", "#dcefe0"),
        (8.95, "两米水深场\n应急决策使用", "#efe6f6"),
    ]
    for x, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.75), 2.5, 1.5,
                                    boxstyle="round,pad=0.05,rounding_size=0.12",
                                    linewidth=1.2, edgecolor="#3a4a5a", facecolor=col))
        ax.text(x + 1.25, 1.5, txt, ha="center", va="center", fontsize=9.2)
    for x in (2.62, 5.57, 8.52):
        ax.add_patch(FancyArrowPatch((x, 1.5), (x + 0.41, 1.5), arrowstyle="-|>",
                                     mutation_scale=13, linewidth=1.4, color="#3a4a5a"))
    ax.text(3.0, 2.62, "校正后的粗网格场", ha="center", fontsize=9, color="#8a5a10")
    ax.annotate("", xy=(4.3, 2.42), xytext=(4.3, 2.30),
                arrowprops=dict(arrowstyle="-|>", color="#8a5a10", lw=1.1))
    ax.text(1.35, 0.42, "误差来源一\n粗网格模拟值\n对聚合真值的偏差", ha="center",
            fontsize=8.4, color="#1f4e79")
    ax.text(1.35, 2.62, "误差来源二\n粗化造成的信息损失\n（单元内地形与水深变化）",
            ha="center", fontsize=8.4, color="#c0392b")
    ax.add_patch(FancyArrowPatch((1.35, 2.42), (1.35, 0.72), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.0, color="#c0392b",
                                 linestyle=":"))
    ax.set_title("(a) 前置模型在预报链中的位置", fontsize=10.5, loc="left")

    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(RES))
    w = 0.36
    for k, scen in enumerate(SCEN):
        vals = [FL["resolutions"][scen][r]["rmse"] for r in RES]
        ax.bar(x + (k - 0.5) * w, vals, w, label=f"{scen} 重现期",
               color=C["20a"] if scen == "20a" else C["100a"], alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cells(RES[i])} 米" for i in range(len(RES))])
    ax.set_ylabel("粗网格模拟值相对聚合真值的 RMSE（米）")
    ax.set_title("(b) 表示与数值误差随网格尺寸", fontsize=10.5, loc="left")
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[1, 1])
    for scen, ls in (("20a", "--"), ("100a", "-")):
        tot, sim, agg = [], [], []
        for r in RES:
            d = FL["resolutions"][scen][r]["error_decomp_2m"]
            tot.append(d["rmse_total"])
            sim.append(d["rmse_simulation"])
            agg.append(d["rmse_aggregation"])
        ax.plot(x, tot, ls, marker="o", color=C[scen], label=f"{scen} 总误差")
        ax.plot(x, sim, ls, marker="s", color=C[scen], alpha=0.55,
                label=f"{scen} 模拟项")
        ax.plot(x, agg, ls, marker="^", color="#7a4fb5", alpha=0.75,
                label=f"{scen} 聚合项")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cells(RES[i])} 米" for i in range(len(RES))])
    ax.set_ylabel("在 2 米格网上量得的 RMSE（米）")
    ax.set_title("(c) 两类误差的分解", fontsize=10.5, loc="left")
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
    ax.set_xlabel("网格边长（米）")
    ax.set_ylabel("误差（米）")
    ax.set_title("(a) 平均绝对误差与均方根误差", fontsize=10.5, loc="left")
    ax.legend(frameon=False, ncol=2, fontsize=8)

    ax = axes[0, 1]
    for scen in SCEN:
        ax.plot(xs, [FL["resolutions"][scen][r]["bias"] for r in RES], "-o",
                color=C[scen], label=scen)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel("网格边长（米）")
    ax.set_ylabel("偏差，预测减真值（米）")
    ax.set_title("(b) 全域平均偏差", fontsize=10.5, loc="left")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    mk = {"0.05": "o", "0.30": "s", "1.00": "^"}
    for scen, ls in (("20a", "--"), ("100a", "-")):
        for t in THR:
            ax.plot(xs, [FL["resolutions"][scen][r]["csi"][t] for r in RES],
                    ls, marker=mk[t], color=C[scen],
                    alpha={"0.05": 1.0, "0.30": 0.7, "1.00": 0.45}[t],
                    label=f"{scen} CSI@{t} 米")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel("网格边长（米）")
    ax.set_ylabel("临界成功指数")
    ax.set_title("(c) 淹没范围匹配度", fontsize=10.5, loc="left")
    ax.legend(frameon=False, ncol=2, fontsize=7.6)

    ax = axes[1, 1]
    for scen, ls in (("20a", "--"), ("100a", "-")):
        ax.plot(xs, [FL["resolutions"][scen][r]["volume_rel"] * 100 for r in RES],
                ls, marker="o", color=C[scen], label=f"{scen} 体积相对误差")
    for t, mkc in (("0.05", "s"), ("0.30", "^"), ("1.00", "D")):
        ax.plot(xs, [FL["resolutions"]["100a"][r]["area_km2"][t]["native"] /
                     FL["resolutions"]["100a"][r]["area_km2"][t]["truth"] - 1
                     for r in RES], ":", marker=mkc, color="#7a4fb5", alpha=0.8,
                label=f"100a 淹没面积比@{t} 米")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel("网格边长（米）")
    ax.set_ylabel("相对偏差（%）")
    ax.set_title("(d) 水量与淹没面积偏差", fontsize=10.5, loc="left")
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
                    color=C[r], ms=3.6, label=f"{r} 网格")
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=45, ha="right", fontsize=7.6)
        ax.set_xlabel("聚合真值水深区间（米）")
        ax.set_ylabel("偏差（米）")
        ax.set_title(f"({('a', 'b')[SCEN.index(scen)]}) {scen} 重现期按水深分层的偏差",
                     fontsize=10.5, loc="left")
        ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 0]
    for r in RES:
        rec = {b["bin"]: b for b in FL["resolutions"]["100a"][r]["by_depth"]}
        ax.plot(x, [rec[b].get("mae", np.nan) for b in bins], "-o", color=C[r],
                ms=3.6, label=f"{r} 网格")
    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=45, ha="right", fontsize=7.6)
    ax.set_xlabel("聚合真值水深区间（米）")
    ax.set_ylabel("平均绝对误差（米）")
    ax.set_title("(c) 100a 重现期按水深分层的平均绝对误差", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    for scen in SCEN:
        rec = {b["bin"]: b for b in FL["resolutions"][scen]["10m"]["by_depth"]}
        ax.semilogy(x, [max(rec[b].get("n", 0), 1) for b in bins], "-o",
                    color=C[scen], ms=3.6, label=scen)
    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=45, ha="right", fontsize=7.6)
    ax.set_xlabel("聚合真值水深区间（米）")
    ax.set_ylabel("像元数（对数刻度）")
    ax.set_title("(d) 十米网格各水深区间的样本量", fontsize=10.5, loc="left")
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
        ax.set_xlabel("单元平均坡度（无量纲，升除以水平距离）")
    axes[0, 0].set_ylabel("平均绝对误差（米）")
    axes[0, 0].set_title("(a) 全部有效单元", fontsize=10.5, loc="left")
    axes[0, 1].set_ylabel("平均绝对误差（米）")
    axes[0, 1].set_title("(b) 只统计有水单元", fontsize=10.5, loc="left")
    axes[1, 1].axhline(0, color="#888", lw=0.8)
    axes[1, 1].set_ylabel("平均偏差（米）")
    axes[1, 1].set_title("(d) 全部有效单元的平均偏差", fontsize=10.5, loc="left")

    ax = axes[1, 0]
    wf = [FL["resolutions"]["100a"]["10m"]["by_slope"][k]["wet_frac"] * 100
          for k in range(len(bins))]
    dep = [FL["resolutions"]["100a"]["10m"]["by_slope"][k]["mean_truth"]
           for k in range(len(bins))]
    ax.bar(x - 0.19, dep, 0.36, color="#2a7fbf", alpha=0.85,
           label="各区间聚合真值平均水深（米）")
    ax.set_ylabel("聚合真值平均水深（米）", color="#1f4e79")
    ax.tick_params(axis="y", labelcolor="#1f4e79")
    ax2 = ax.twinx()
    ax2.plot(x + 0.19, wf, "-s", color="#c0392b", ms=4.2,
             label="该区间有水的单元占比（百分数）")
    ax2.set_ylabel("有水单元占比（百分数）", color="#8a2b20")
    ax2.tick_params(axis="y", labelcolor="#8a2b20")
    ax2.grid(False)
    ax.set_title("(c) 十米网格上每个坡度区间的水量条件", fontsize=10.5,
                 loc="left")
    hs = [Line2D([], [], color="#2a7fbf", lw=6, alpha=0.85,
                 label="各区间聚合真值平均水深（米）"),
          Line2D([], [], color="#c0392b", marker="s", ls="-",
                 label="该区间有水的单元占比（百分数）")]
    ax.legend(handles=hs, frameon=False, fontsize=8, loc="upper right")

    handles = [Line2D([], [], color=C[r], marker="o", ls="-", label=f"{r} 网格")
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
            color="#1f4e79", label="单元内高程标准差均值")
    ax.plot(xs, [TE["resolutions"][r]["within_cell_std_median"] for r in RES], "--s",
            color="#1f4e79", alpha=0.6, label="中位数")
    ax.plot(xs, [TE["resolutions"][r]["relief_mean"] for r in RES], "-^",
            color="#c0392b", label="单元内高差均值")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel("网格边长（米）")
    ax.set_ylabel("高程统计量（米）")
    ax.set_title("(a) 单元内地形起伏", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.03, 0.95, f"2 米高程标准差 {TE['fine']['dem_std']:.1f} 米",
            transform=ax.transAxes, fontsize=8, va="top", color="#444")

    ax = axes[0, 1]
    ax.plot(xs, [TE["resolutions"][r]["slope_native_mean"] for r in RES], "-o",
            color="#c0392b", label="粗网格自身坡度均值")
    ax.plot(xs, [TE["resolutions"][r]["slope_agg_mean"] for r in RES], "--s",
            color="#1f4e79", label="两米坡度聚合后均值")
    ax.plot(xs, [TE["resolutions"][r]["slope_native_median"] for r in RES], ":o",
            color="#c0392b", alpha=0.6, label="粗网格坡度中位数")
    ax.plot(xs, [TE["resolutions"][r]["slope_agg_median"] for r in RES], ":s",
            color="#1f4e79", alpha=0.6, label="聚合坡度中位数")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel("网格边长（米）")
    ax.set_ylabel("坡度（无量纲）")
    ax.set_title("(b) 坡度随网格尺寸的衰减", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=7.6)
    for r in RES:
        ax.annotate(f"{TE['resolutions'][r]['slope_ratio']:.3f}",
                    (cells(r), TE["resolutions"][r]["slope_native_mean"]),
                    textcoords="offset points", xytext=(2, -12), fontsize=7.4,
                    color="#8a5a10")

    ax = axes[1, 0]
    ax.plot(xs, [TE["resolutions"][r]["sink_depth_mean"] for r in RES], "-o",
            color="#1f4e79", label="洼地深度均值（米）")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel("网格边长（米）")
    ax.set_ylabel("单元均值高程减单元最低高程（米）")
    ax.set_title("(c) 被粗化抹掉的局部洼地", fontsize=10.5, loc="left")
    ax2 = ax.twinx()
    ax2.plot(xs, [TE["resolutions"][r]["sink_frac_gt_0p1"] * 100 for r in RES], "--s",
             color="#c0392b", label="洼地深于 0.1 米的单元占比")
    ax2.plot(xs, [TE["resolutions"][r]["sink_frac_gt_0p5"] * 100 for r in RES], ":^",
             color="#c0392b", alpha=0.6, label="深于 0.5 米")
    ax2.set_ylabel("单元占比（%）", color="#c0392b")
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
        ax.text(cells(r), ax.get_ylim()[1] * 0.94, f"{cells(r)} 米",
                rotation=90, fontsize=7.4, color=col, ha="right", va="top")
    ax.set_xlabel("水道宽度，由两米水体掩膜的欧氏距离变换得到（米）")
    ax.set_ylabel("水体像元占比（%）")
    ax.set_title("(d) 水道宽度分布与网格边长", fontsize=10.5, loc="left")
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
        axes[0, k].set_title(f"{cells(r)} 米网格，聚合真值", fontsize=9.6)
        cb = fig.colorbar(im, ax=axes[0, k], fraction=0.028, pad=0.02)
        cb.set_label("水深（米）" if k == 0 else "", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        lim = 2.0
        im2 = axes[1, k].imshow(err, extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim,
                                interpolation="nearest")
        axes[1, k].set_title(f"{cells(r)} 米网格，粗网格模拟减聚合", fontsize=9.6)
        cb2 = fig.colorbar(im2, ax=axes[1, k], fraction=0.028, pad=0.02)
        cb2.set_label("误差（米）" if k == 0 else "", fontsize=8)
        cb2.ax.tick_params(labelsize=7)
        for ax in (axes[0, k], axes[1, k]):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
    fig.suptitle("100a 重现期全域水深与粗网格误差，北在上，单位米", fontsize=11, y=0.985)
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
    REG = [("区域一  河网密集", (1188, 1230, 468, 510)),
           ("区域二  水面宽阔", (330, 372, 1068, 1110))]

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
                ax.set_title(f"{glab}\n两米参考", fontsize=TY, color="#1a7a3a")
            else:
                ax.imshow(loc, extent=[0, 12960, 23040, 0], cmap="Blues",
                          vmin=0, vmax=3.0, interpolation="nearest")
                ax.add_patch(plt.Rectangle((c0 * 10, r0 * 10), (c1 - c0) * 10,
                                           (r1 - r0) * 10, fill=False,
                                           edgecolor="#c0392b", linewidth=1.8))
                ax.set_title(f"{glab}\n放大位置", fontsize=TY, color="#c0392b")
                ax.plot([ext[0] + 250, ext[0] + 350], [ext[2] - 30, ext[2] - 30],
                        color="#1a1a1a", lw=2.4)
                ax.text(ext[0] + 300, ext[2] - 52, "100 米", fontsize=9.0,
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
                    ax.set_title(f"{cells(res)} 米\n{t.shape[1]} × {t.shape[0]} 格",
                                 fontsize=TY)
                else:
                    im = ax.imshow(e, extent=ext, cmap="RdBu_r", vmin=-2.0, vmax=2.0,
                                   interpolation="nearest")
                    fin = np.isfinite(e)
                    lab = (f"平均绝对误差 {np.mean(np.abs(e[fin])):.3f} 米"
                           if fin.any() else "无有效单元")
                    ax.set_title(f"{cells(res)} 米\n{lab}", fontsize=TY)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                if k == 3:
                    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
                    cb.set_label("水深（米）" if q == 0 else "误差（米）", fontsize=9.5)
                    cb.ax.tick_params(labelsize=8)

    fig.suptitle(f"{scen} 重现期两个典型小区间的放大对比，窗口 420 米见方，北在上，单位米",
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
    ax.set_xlabel("分辨率比，粗网格边长除以 2 米")
    ax.set_ylabel("临界成功指数")
    ax.set_title("(a) 淹没范围匹配度随分辨率比衰减", fontsize=10.5, loc="left")
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
    ax.set_xlabel("分辨率比，粗网格边长除以 2 米")
    ax.set_ylabel("相关系数与结构相似度")
    ax.set_title("(b) 水深分布相似性", fontsize=10.5, loc="left")
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
    ax.set_xlabel("分辨率比")
    ax.set_ylabel("流速幅值相关系数")
    ax.set_title("(c) 流速分布相似性（仅在数据含流量场的时刻）", fontsize=10.5, loc="left")
    ax.legend(frameon=False, ncol=2, fontsize=7.6)

    ax = axes[1, 1]
    for scen, ls in (("20a", "--"), ("100a", "-")):
        ax.plot(xs, [FL["resolutions"][scen][r]["rmse_rel_mean_truth"] for r in RES],
                ls, marker="o", color=C[scen], label=f"{scen} 相对均方根误差")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel("分辨率比")
    ax.set_ylabel("均方根误差除以平均真值水深")
    ax.set_title("(d) 归一化误差随分辨率比的增长", fontsize=10.5, loc="left")
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
    ax.set_xlabel("5 米网格逐瓦片 CSI@0.05")
    ax.set_ylabel("30 米网格逐瓦片 CSI@0.05")
    ax.set_title("(a) 同一瓦片在两个网格上的匹配度", fontsize=10.5, loc="left")
    ax.legend(frameon=False)

    for ax, key, xlab, ttl in (
            (axes[0, 1], "wet_frac", "瓦片内湿区占比", "(b) 湿区占比与匹配度"),
            (axes[1, 0], "within_cell_dem_std", "单元内两米高程标准差（米）",
             "(c) 地形粗糙度与匹配度"),
            (axes[1, 1], "building_frac", "瓦片内建筑像元占比", "(d) 建筑占比与匹配度")):
        for scen in SCEN:
            rec = pick("10m", scen)
            xs = np.array([r[key] for r in rec], float)
            ys = np.array([r["csi005"] for r in rec], float)
            ok = np.isfinite(xs) & np.isfinite(ys)
            ax.scatter(xs[ok], ys[ok], s=13, alpha=0.65, color=C[scen],
                       label=f"{scen} r={np.corrcoef(xs[ok], ys[ok])[0,1]:.2f}",
                       edgecolors="none")
        ax.set_xlabel(xlab)
        ax.set_ylabel("10 米网格逐瓦片 CSI@0.05")
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
    cn = {"rain_scenario": "降雨情景", "slope_roughness": "坡度与粗糙度",
          "impervious": "不透水率", "built_landuse": "建筑与土地利用",
          "river_drainage": "河道邻近度", "wet_state": "湿区占比（状态量）"}
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
    ax.set_xticklabels([f"{cells(r)} 米网格" for r in RES])
    ax.set_ylabel("对逐瓦片 CSI@0.05 方差的解释份额（Shapley 值）")
    ax.set_title("(a) 跨分辨率相关性的贡献度分解", fontsize=10.5, loc="left")
    ax.legend(frameon=False, ncol=3, fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    names = ["mean_slope", "within_cell_dem_std", "mean_tpi", "impervious_frac",
             "building_frac", "landuse_entropy", "green_frac", "mean_dist_water",
             "min_dist_water", "water_frac_fine", "wet_frac"]
    cn2 = {"mean_slope": "坡度", "within_cell_dem_std": "单元内高程标准差",
           "mean_tpi": "地形位置指数", "impervious_frac": "不透水率",
           "building_frac": "建筑占比", "landuse_entropy": "土地利用熵",
           "green_frac": "绿地占比", "mean_dist_water": "平均到水体距离",
           "min_dist_water": "最小到水体距离", "water_frac_fine": "细网格水体占比",
           "wet_frac": "湿区占比"}
    vifs = np.array([VD["10m"]["vif"][n] for n in names])
    ax.barh(np.arange(len(names)), vifs, color="#5b7fa6")
    ax.axvline(5, color="#c0392b", ls="--", lw=1.0)
    ax.text(5.1, len(names) - 0.6, "共线性参考线 5", color="#c0392b", fontsize=7.8)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([cn2[n] for n in names], fontsize=7.8)
    ax.set_xlabel("方差膨胀因子（十米网格）")
    ax.set_title("(b) 解释变量之间的共线性", fontsize=10.5, loc="left")
    ax.grid(axis="y", alpha=0.0)

    ax = fig.add_subplot(gs[1, 1])
    gs_names = groups
    mat = np.zeros((len(gs_names), len(RES)))
    for j, r in enumerate(RES):
        for i, g in enumerate(gs_names):
            mat[i, j] = VD[r]["responses"]["csi@0.05"]["share_of_explained"][g]
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(RES)))
    ax.set_xticklabels([f"{cells(r)} 米" for r in RES])
    ax.set_yticks(range(len(gs_names)))
    ax.set_yticklabels([cn[g] for g in gs_names], fontsize=8)
    for i in range(len(gs_names)):
        for j in range(len(RES)):
            ax.text(j, i, f"{mat[i, j]*100:.0f}%", ha="center", va="center",
                    fontsize=7.6, color="#111" if mat[i, j] < 0.42 else "white")
    ax.set_title("(c) 各因素占已解释方差的比例", fontsize=10.5, loc="left")
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
    ax.plot([0, hi], [0, hi], "--", color="#c0392b", lw=1.0, label="等值线")
    ax.set_xlabel("校正前逐瓦片平均绝对误差（米）")
    ax.set_ylabel("校正后逐瓦片平均绝对误差（米）")
    ax.set_title("(a) 前置模型逐瓦片配对效果（十米网格测试集）", fontsize=10.5, loc="left")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    g = R["global"]
    names = ["identity", "bilinear", "premodel"]
    cn = {"identity": "恒等映射", "bilinear": "双线性重采样", "premodel": "前置模型"}
    x = np.arange(len(THR))
    w = 0.26
    for k, nm in enumerate(names):
        vals = [g[nm][f"csi@{t}"] for t in THR]
        ax.bar(x + (k - 1) * w, vals, w, label=cn[nm],
               color=[C["id"], C["bil"], C["pre"]][k])
    ax.set_xticks(x)
    ax.set_xticklabels([f"CSI@{t}" for t in THR])
    ax.set_ylabel("临界成功指数")
    ax.set_title("(b) 三种场在粗网格上的匹配度", fontsize=10.5, loc="left")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    names2 = ["rmse", "mae"]
    cn2 = {"rmse": "均方根误差", "mae": "平均绝对误差"}
    x2 = np.arange(len(names2))
    for k, nm in enumerate(names):
        vals = [g[nm][q] for q in names2]
        ax.bar(x2 + (k - 1) * w, vals, w, label=cn[nm],
               color=[C["id"], C["bil"], C["pre"]][k])
    ax.set_xticks(x2)
    ax.set_xticklabels([cn2[q] for q in names2])
    ax.set_ylabel("误差（米）")
    ax.set_title("(c) 数值误差对比", fontsize=10.5, loc="left")
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
    ax.set_ylabel("临界成功指数")
    ax.set_title("(d) 最近邻升采样到两米后的匹配度", fontsize=10.5, loc="left")
    ax.legend(frameon=False)
    save(fig, "fig50_premodel_result.png")


# ---------------------------------------------------------------- fig 51
def fig_dialect():
    """The two comparison dialects and the identity that joins them."""
    SD = load(PRE / "scale_dialect.json")
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
    x = np.arange(len(RES))
    lab = [f"{cells(r)} 米" for r in RES]
    c_coarse, c_within = "#1f4e79", "#e08b2b"

    for k, scen in enumerate(SCEN):
        ax = axes[0, k]
        coarse = np.array([SD["results"][scen][r]["fine_nn"]["mse_coarse"] for r in RES])
        within = np.array([SD["results"][scen][r]["fine_nn"]["mse_within"] for r in RES])
        fine = np.array([SD["results"][scen][r]["fine_nn"]["mse_fine"] for r in RES])
        ax.bar(x, coarse, 0.58, color=c_coarse, label="粗网格口径，可减误差",
               edgecolor="white", linewidth=0.5)
        ax.bar(x, within, 0.58, bottom=coarse, color=c_within,
               label="块内方差，不可减细节", edgecolor="white", linewidth=0.5)
        for i in range(len(RES)):
            ax.text(i, fine[i] + 0.006, f"{within[i]/fine[i]*100:.0f}%",
                    ha="center", fontsize=8.2, color="#8a5a10")
        ax.plot(x, fine, "k_", markersize=14, markeredgewidth=1.6,
                label="细网格口径合计")
        ax.set_xticks(x)
        ax.set_xticklabels(lab)
        ax.set_ylabel("均方误差（平方米）")
        ax.set_title(f"({('a', 'b')[k]}) {scen} 重现期，粗细两个口径的对照", fontsize=10.5,
                     loc="left")
        ax.legend(frameon=False, fontsize=7.8)

    ax = axes[1, 0]
    for scen, ls in (("20a", "--"), ("100a", "-")):
        ax.plot(x, [SD["results"][scen][r]["rmse_m"]["fine_dialect"] for r in RES],
                ls, marker="o", color=C[scen], label=f"{scen} 细网格口径")
        ax.plot(x, [SD["results"][scen][r]["rmse_m"]["coarse_dialect"] for r in RES],
                ls, marker="s", color=C[scen], alpha=0.55, label=f"{scen} 粗网格口径")
    ax.set_xticks(x)
    ax.set_xticklabels(lab)
    ax.set_ylabel("均方根误差（米）")
    ax.set_title("(c) 两个口径给出的误差水平", fontsize=10.5, loc="left")
    ax.legend(frameon=False, ncol=2, fontsize=8)

    ax = axes[1, 1]
    for scen, ls, dx in (("20a", "--", -17), ("100a", "-", 17)):
        sh = [SD["results"][scen][r]["shares"]["within_of_fine"] * 100 for r in RES]
        ax.plot(x, sh, ls, marker="^", color=C[scen], label=f"{scen} 重现期")
        for i in range(len(RES)):
            ax.annotate(f"{sh[i]:.1f}%", (x[i], sh[i]), textcoords="offset points",
                        xytext=(dx, 5), ha="center", fontsize=7.8, color=C[scen])
    ax.set_xticks(x)
    ax.set_xticklabels(lab)
    ax.set_ylim(0, 30)
    ax.set_ylabel("块内方差占细网格口径的比例（%）")
    ax.set_title("(d) 细网格口径里有多少是地面本来就不均匀", fontsize=10.5, loc="left")
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
    rows = [("高程（米）", "terrain", None, None),
            ("坡度", "YlOrBr", None, None),
            ("水深（米）", "Blues", 0.0, 3.0),
            ("流速（米每秒）", "PuBuGn", 0.0, 3.0)]
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
                ax.set_title(f"{cells(res)} 米网格", fontsize=9.5, pad=3)
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
    ax.set_xlabel("聚合真值（十进对数）", fontsize=9)
    ax.set_ylabel("粗网格模拟值（十进对数）", fontsize=9)
    ax.set_title(f"({('a', 'b')[0 if res == '10m' else 1]}) {cells(res)} 米网格，"
                 f"皮尔逊 {r['pearson_wet']:.3f}，斯皮尔曼 {r['spearman_wet']:.3f}",
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
            color="#5b5b5b", label="二米参考")
    for res in RES:
        ax.hist(_wet_sample(var, scen, res), bins=bins, density=True,
                histtype="step", lw=1.1, color=C[res], label=f"{cells(res)} 米")
    ax.set_xlabel("水深（米）" if var == "h_max" else "流速（米每秒）", fontsize=9)
    ax.set_ylabel("概率密度", fontsize=9)
    ax.set_title("(c) 各分辨率下水深分布的形状" if var == "h_max"
                 else "(c) 各分辨率下流速分布的形状", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=2, columnspacing=0.9)
    ax.tick_params(labelsize=9)


def _fit_panel(ax, cs, var, scen, res):
    from scipy import stats

    fam = {"Gamma 伽马": stats.gamma, "Log-normal 对数正态": stats.lognorm,
           "Weibull 三参数韦布尔": stats.weibull_min, "Exponential 指数": stats.expon,
           "Log-logistic 对数逻辑斯蒂": stats.fisk}
    d = cs["flood"]["variables"][var][scen]["resolutions"][res]["dist_native"]
    y = _wet_sample(var, scen, res)
    top = np.percentile(y, 99)
    xs = np.linspace(1e-4, top, 400)
    ax.hist(y, bins=np.linspace(0, top, 60), density=True, histtype="step",
            lw=1.0, color="#9a9a9a", label="样本")
    order = sorted([f for f in d["all_families"] if "aic" in f],
                   key=lambda f: f["aic"])[:3]
    for k, f in enumerate(order):
        pdf = fam[f["family"]].pdf(xs, *f["params"])
        short = f["family"].split(" ")[0]
        ax.plot(xs, pdf, lw=1.3, color=["#1f4e79", "#e08b2b", "#c0392b"][k],
                label=f"{short}\nAIC {f['aic']:.0f}  KS {f['ks']:.3f}")
    ax.set_xlabel("水深（米）" if var == "h_max" else "流速（米每秒）", fontsize=9)
    ax.set_ylabel("概率密度", fontsize=9)
    ax.set_title(f"(d) {cells(res)} 米网格的分布拟合", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right",
              labelspacing=0.35, handlelength=1.3, borderaxespad=0.35)
    ax.tick_params(labelsize=9)


def _quant_panel(ax, cs, var, scen):
    for q, col, mk in zip((50, 75, 95, 99),
                          ["#1f4e79", "#2a7fbf", "#e08b2b", "#c0392b"], "os^D"):
        vals = [float(np.percentile(_wet_sample(var, scen, r), q)) for r in RES]
        ax.plot([cells(r) for r in RES], vals, marker=mk, color=col, lw=1.2,
                label=f"{q} 分位")
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 30])
    ax.set_xticklabels(["5", "10", "20", "30"])
    ax.set_xlabel("网格边长（米）", fontsize=9)
    ax.set_ylabel("水深（米）" if var == "h_max" else "流速（米每秒）", fontsize=9)
    ax.set_title("(e) 分位数随网格的变化", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=2, columnspacing=0.9)
    ax.tick_params(labelsize=9)


def _moment_panel(ax, cs, var, scen):
    """Skew and excess kurtosis against grid size, on two scales since they differ."""
    ax2 = ax.twinx()
    for tag, key, col, target in (("偏度", "skew", "#1f4e79", ax),
                                  ("超额峰度", "kurtosis_excess", "#e08b2b", ax2)):
        vals = [cs["flood"]["variables"][var][scen]["resolutions"][r]
                ["dist_native"][key] for r in RES]
        target.plot([cells(r) for r in RES], vals, marker="o", color=col, lw=1.2,
                    label=tag)
    for target, col in ((ax, "#1f4e79"), (ax2, "#e08b2b")):
        target.tick_params(axis="y", labelsize=9, colors=col)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 30])
    ax.set_xticklabels([f"{c}" for c in [5, 10, 20, 30]])
    ax.set_xlabel("网格边长（米）", fontsize=9)
    ax.set_ylabel("偏度", fontsize=9, color="#1f4e79")
    ax2.set_ylabel("超额峰度", fontsize=9, color="#e08b2b")
    ax.set_title("(f) 分布形状的稳定性", fontsize=9.5, loc="left")
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
    for ax, var, lab in ((axes[0], "h_max", "水深"),
                         (axes[1], "speed", "流速")):
        M = np.array([[cs["vmeasure"]["variables"][var]["pairwise"][a][b]["v"]
                       for b in ALLRES_FIG] for a in ALLRES_FIG])
        im = ax.imshow(M, cmap="YlGnBu", vmin=0.3, vmax=1.0)
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"{int(a[:-1])}" for a in ALLRES_FIG], fontsize=9)
        ax.set_yticks(range(5))
        ax.set_yticklabels([f"{int(a[:-1])}" for a in ALLRES_FIG], fontsize=9)
        ax.set_xlabel("网格边长（米）", fontsize=9)
        ax.set_ylabel("网格边长（米）", fontsize=9)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=9,
                        color="#111" if M[i, j] < 0.82 else "white")
        ax.set_title(f"({('a', 'b')[0 if var == 'h_max' else 1]}) {lab}的区域划分一致性",
                     fontsize=9.5, loc="left")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=9)
    ax = axes[2]
    for var, col, mk in (("h_max", "#1f4e79", "o"), ("speed", "#e08b2b", "s")):
        sc = cs["vmeasure"]["variables"][var]["k_diagnostics"]
        ks = sorted(int(k) for k in sc["scores"])
        ax.plot(ks, [sc["scores"][str(k)] for k in ks], marker=mk, color=col,
                lw=1.2, label="水深" if var == "h_max" else "流速")
        ax.axvline(sc["chosen_k"], color=col, ls=":", lw=1.0, alpha=0.7)
    ax.set_xlabel("簇数 K", fontsize=9)
    ax.set_ylabel("轮廓系数（二米参考场）", fontsize=9)
    ax.set_title("(c) 区域划分的簇数选择", fontsize=9.5, loc="left")
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
    groups = (("h_max", "水深", ("a", "b", "c", "d")),
              ("speed", "流速", ("e", "f", "g", "h")))
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
                (ref, cms, -0.5, k - 0.5, f"({L[0]}) {lab}二米参考区域划分"),
                (coast, cms, -0.5, k - 0.5, f"({L[1]}) {lab}三十米区域划分"),
                (stab, stab_cm, -0.5, 4.5, f"({L[2]}) {lab}跨分辨率稳定性"))):
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
        row_ax.set_ylabel(f"{lab}\n{n} 单元", fontsize=9, labelpad=2)
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
        ax.set_xlabel("与二米参考一致的粗网格档数", fontsize=9)
        ax.set_ylabel("面积占比（%）", fontsize=9)
        ax.set_title(f"({L[3]}) {lab}稳定性的分布", fontsize=9.5, loc="left")
        ax.tick_params(labelsize=9)
    cax = fig.add_subplot(gs[0:2, 4])
    cb = fig.colorbar(sim, cax=cax, ticks=[0, 1, 2, 3, 4])
    cb.ax.tick_params(labelsize=8.5, length=2)
    cb.set_label("与二米参考一致的档数", fontsize=9)
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
    cb2.set_label("区域编号，两行共用", fontsize=9)
    save(fig, "fig75_pattern_map.png")
    for lab, L, n, r0, r1, c0, c1 in notes:
        print(f"  {L[0]}-{L[3]} {lab}: {n} support cells, "
              f"crop rows {r0}-{r1} cols {c0}-{c1} of 1620 x 2880",
              flush=True)


# ---------------------------------------------------------------- fig 76 77 78
PAT_METRICS = [("v", "V-measure 与归一化互信息（重合）", "#1f4e79", "o"),
               ("ari", "调整兰德指数", "#c0392b", "s"),
               ("fm", "Fowlkes-Mallows", "#1a7a3a", "^"),
               ("kappa", "科恩卡帕", "#8e5ea2", "P")]


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
    for j, (var, lab) in enumerate((("h_max", "水深"), ("speed", "流速"))):
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
        ax.set_xlabel("网格边长（米）", fontsize=9)
        ax.set_ylabel("网格边长（米）", fontsize=9)
        ax.grid(False)
        ax.set_title(f"({('a', 'b')[j]}) {lab}区域划分的调整兰德指数",
                     fontsize=9.6, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=8.5)
    for j, (var, lab) in enumerate((("h_max", "水深"), ("speed", "流速"))):
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
        ax.set_xlabel("两档网格边长之比", fontsize=9)
        ax.set_ylabel("一致性指标取值", fontsize=9)
        ax.set_title(f"({('c', 'd')[j]}) {lab}六类指标随分辨间隔的衰减",
                     fontsize=9.6, loc="left")
        ax.legend(frameon=False, fontsize=7.4, ncol=2, columnspacing=0.8)
    fig.tight_layout()
    save(fig, "fig76_pattern_agreement.png")


def fig_pattern_areal():
    sp = _pat()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))
    for ax, field, ylab, title, mk in (
            (axes[0], "mean_iou", "最优匹配的平均交并比",
             "(a) 区域面积重叠随分辨间隔的衰减", "o"),
            (axes[1], "area_tv", "面积占比的总变差距离",
             "(b) 区域面积份额的差异", "s")):
        for var, lab, col in (("h_max", "水深", "#1f4e79"),
                              ("speed", "流速", "#e08b2b")):
            xs, ys = _by_sep(sp["variables"][var]["areal"], field)
            ax.plot(xs, ys, mk + "-", color=col, lw=1.3, ms=4.4, label=lab)
        ax.set_xscale("log")
        ax.set_xticks([1.5, 2, 2.5, 3, 4, 5, 6, 10, 15])
        ax.set_xticklabels(["1.5", "2", "2.5", "3", "4", "5", "6", "10", "15"],
                           fontsize=8)
        ax.set_xlabel("两档网格边长之比", fontsize=9)
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
               label=f"{cells(r)} 米")
        for x, v in zip(xs, pci):
            ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.4)
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels([f"第 {i + 1} 档" for i in range(k)], fontsize=8.5)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("水深区域划分的簇编号", fontsize=9)
    ax.set_ylabel("与二米对应的最优交并比", fontsize=9)
    ax.set_title("(c) 各簇与二米参考的交并比", fontsize=9.6, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, "fig77_pattern_areal.png")


def fig_pattern_boundary():
    sp = _pat()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))
    for ax, field, ylab, title in (
            (axes[0], "f1", "湿区边界的 F1", "(a) 湿区边界的重合程度"),
            (axes[1], "mhd_m", "湿区前缘的平均位移（米）", "(b) 湿区前缘的平均位移"),
            (axes[2], "iou", "湿区边界的交并比", "(c) 湿区边界的交并比")):
        for var, lab, col, mk in (("h_max", "水深", "#1f4e79", "o"),
                                  ("speed", "流速", "#e08b2b", "s")):
            xs, ys = _by_sep(sp["variables"][var]["boundary"], field)
            ax.plot(xs, ys, mk + "-", color=col, lw=1.3, ms=4.4, label=lab)
        ax.set_xticks([2.5, 5, 10, 15])
        ax.set_xticklabels(["2.5", "5", "10", "15"], fontsize=8.5)
        ax.set_xlabel("粗网格与二米的边长之比", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=9.6, loc="left")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylim(0, 0.6)
    fig.tight_layout()
    save(fig, "fig78_pattern_boundary.png")


def fig_pattern_autocorr():
    sp = _pat()
    xs = [2, 5, 10, 20, 30]
    series = [("水深场", "h_max", "field", "#1f4e79", "o"),
              ("流速场", "speed", "field", "#e08b2b", "s"),
              ("水深标号", "h_max", "class_labels", "#2a7fbf", "^"),
              ("流速标号", "speed", "class_labels", "#c0392b", "D")]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))
    for ax, field, ylab, title in (
            (axes[0], "moran", "莫兰指数 I", "(a) 空间自相关随网格的变化"),
            (axes[1], "geary", "吉尔里系数 C", "(b) 相邻差异比随网格的变化")):
        for lab, var, kind, col, mk in series:
            ys = [sp["variables"][var]["autocorrelation"][kind][r][field]
                  for r in ALLRES_FIG]
            ax.plot(xs, ys, mk + "-", color=col, lw=1.25, ms=4.2, label=lab)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs], fontsize=8.5)
        ax.set_xlabel("网格边长（米）", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=9.6, loc="left")
        ax.legend(frameon=False, fontsize=7.6)
    ax = axes[2]
    for var, lab, col, mk in (("h_max", "水深", "#1f4e79", "o"),
                              ("speed", "流速", "#e08b2b", "s")):
        zs = [sp["variables"][var]["join_count"][r]["z"] for r in ALLRES_FIG]
        rs = [sp["variables"][var]["join_count"][r]["ratio"] for r in ALLRES_FIG]
        ax.plot(xs, zs, mk + "-", color=col, lw=1.3, ms=4.4, label=f"{lab}的 z 值")
        for x, zz, rr in zip(xs, zs, rs):
            ax.text(x, zz + 0.8, f"{rr:.2f}", ha="center", fontsize=7, color=col)
    ax.axhline(0, color="#888888", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], fontsize=8.5)
    ax.set_xlabel("网格边长（米）", fontsize=9)
    ax.set_ylabel("同标号邻接的 z 值", fontsize=9)
    ax.set_title("(c) 标号邻接的聚集程度", fontsize=9.6, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "fig79_pattern_autocorr.png")


def fig_pattern_distance():
    sp = _pat()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))
    for j, (var, vlab, unit) in enumerate((("h_max", "水深", "水深（米）"),
                                           ("speed", "流速", "流速（米每秒）"))):
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
        ax.set_xlabel(f"二米参考的{unit}分档", fontsize=9)
        ax.set_ylabel("与二米一致的平均档数占比", fontsize=9)
        ax.set_title(f"({('a', 'b')[j]}) {vlab}的稳定性按取值分档",
                     fontsize=9.6, loc="left")
    ax = axes[2]
    for var, lab, col, mk in (("h_max", "水深", "#1f4e79", "o"),
                              ("speed", "流速", "#e08b2b", "s")):
        b = sp["variables"][var]["distance_resolved"]["dist_water_bands"]
        ax.plot(range(len(b)), [q["mean_stability"] for q in b], mk + "-",
                color=col, lw=1.3, ms=4.4, label=lab)
        ax.set_xticks(range(len(b)))
        ax.set_xticklabels([_band_label(q) for q in b], fontsize=7.6, rotation=30)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("二米到水体的距离分档（米）", fontsize=9)
    ax.set_ylabel("与二米一致的平均档数占比", fontsize=9)
    ax.set_title("(c) 稳定性按到水体距离分档", fontsize=9.6, loc="left")
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
                label=f"{int(r[:-1])} 米网格")
        f = e["fit"]
        if np.isfinite(f["nugget"]) and np.isfinite(f["decay_m"]):
            hh = np.geomspace(lags[0], lags[-1], 200)
            ax.plot(hh, f["nugget"] + (f["sill"] - f["nugget"])
                    * (1.0 - np.exp(-hh / f["decay_m"])),
                    color=RES_COL[r], lw=0.8, ls="--", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("空间滞后 h（米）", fontsize=9)
    ax.set_ylabel("半变异函数 γ(h)（对数水深方差）", fontsize=9)
    ax.set_title("(a) 对数水深场的实验半变异函数与指数拟合", fontsize=9.6, loc="left")
    ax.legend(frameon=False, fontsize=7.0, ncol=2, columnspacing=0.7)

    ax = axes[1]
    xs = [int(r[:-1]) for r in ALLRES_FIG]
    rho = [V[r]["log_depth"]["fit"]["nugget_over_var"] for r in ALLRES_FIG]
    r95 = [V[r]["log_depth"]["fit"]["range95_m"] for r in ALLRES_FIG]
    ax.plot(xs, rho, "o-", color="#1f4e79", lw=1.3, ms=4.4, label="首档相对粗糙度")
    for x, v in zip(xs, rho):
        ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=7.4, color="#1f4e79")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("网格边长（米）", fontsize=9)
    ax.set_ylabel("首档半方差与总方差之比", fontsize=9, color="#1f4e79")
    ax.tick_params(axis="y", labelcolor="#1f4e79")
    ax2 = ax.twinx()
    ax2.plot(xs, r95, "s--", color="#c0392b", lw=1.3, ms=4.4,
             label="达到总方差 95% 的滞后")
    ax2.set_ylabel("达到总方差 95% 的滞后（米）", fontsize=9, color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.4, loc="lower right")
    ax.set_title("(b) 首档粗糙度与有效变程随网格的变化", fontsize=9.6, loc="left")

    ax = axes[2]
    xs2 = [q["scale_m"] for q in S["scales"]]
    dp = [q["dispersion_frac"] for q in S["scales"]]
    bv = [q["block_var_frac"] for q in S["scales"]]
    ax.plot(xs2, dp, "o-", color="#c0392b", lw=1.3, ms=4.4,
            label="块均值未解释的方差份额")
    ax.axhline(1.0, color="#888888", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(xs2)
    ax.set_xticklabels([str(x) for x in xs2], fontsize=7.4, rotation=45)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("聚合方块边长（米）", fontsize=9)
    ax.set_ylabel("块均值未解释的方差份额", fontsize=9, color="#c0392b")
    ax.tick_params(axis="y", labelcolor="#c0392b")
    ax2 = ax.twinx()
    ax2.plot(xs2, bv, "s--", color="#1f4e79", lw=1.3, ms=4.4,
             label="块均值自身保留的方差份额")
    ax2.set_ylabel("块均值自身保留的方差份额", fontsize=9, color="#1f4e79")
    ax2.tick_params(axis="y", labelcolor="#1f4e79")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.4, loc="center left")
    ax.set_title("(c) 尺度空间：块均值对局部水深的解释力", fontsize=9.6, loc="left")
    fig.tight_layout()
    save(fig, "fig82_variogram_scale.png")


def fig_pattern_contiguity_maps():
    ex = _extra()
    cm = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 5.4))
    for i, (var, lab, L) in enumerate((("h_max", "水深", ("a", "b", "c")),
                                       ("speed", "流速", ("d", "e", "f")))):
        Z = np.load(PRE / f"contiguity_core_{var}.npz")
        row = Z["row"].astype("int64")
        col = Z["col"].astype("int64")
        r0, c0 = int(row.min()), int(col.min())
        yy, xx = row - r0, col - c0
        k = int(ex["contiguity"]["variables"][var]["k"])
        cols = plt.matplotlib.colors.ListedColormap(cm.colors[:k])
        for j, (key, ttl, cma, vmx) in enumerate((
                ("unconstrained", f"({L[0]}) {lab}二米簇，无约束 K 均值", cols, k - 0.5),
                ("constrained_2m", f"({L[1]}) {lab}二米簇，邻接约束层次聚类", cols, k - 0.5),
                (None, f"({L[2]}) {lab}两种划分标号是否相同", None, None))):
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
    fig.suptitle("最大连通核上的区域划分：无约束与邻接约束的对照", fontsize=10.2, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, "fig83_contiguity_map.png")


def fig_pattern_contiguity_metrics():
    ex = _extra()
    C = ex["contiguity"]["variables"]
    pairs = ["2m|5m", "2m|10m", "2m|20m", "2m|30m", "5m|10m",
             "5m|20m", "5m|30m", "10m|20m", "10m|30m", "20m|30m"]
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.4))
    for i, (var, lab) in enumerate((("h_max", "水深"), ("speed", "流速"))):
        rec = C[var]
        for j, (field, ylab, ttl) in enumerate((
                ("v", "V-measure", "分辨率组合"),
                ("mean_iou", "最优匹配平均交并比", "分辨率组合"))):
            ax = axes[i][j]
            xs = np.arange(len(pairs))
            u = [rec["pairs"]["unconstrained"][p][field] for p in pairs]
            c = [rec["pairs"]["constrained"][p][field] for p in pairs]
            ax.bar(xs - 0.19, u, 0.38, color="#9fb6cd", label="无约束 K 均值")
            ax.bar(xs + 0.19, c, 0.38, color="#1f4e79", label="邻接约束层次聚类")
            ax.set_xticks(xs)
            ax.set_xticklabels([p.replace("|", "\n与") for p in pairs], fontsize=6.2)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel(ylab, fontsize=9)
            ax.set_title(f"({('a', 'b', 'c', 'd', 'e', 'f')[i * 3 + j]}) {lab}的{ylab}",
                         fontsize=9.2, loc="left")
            ax.legend(frameon=False, fontsize=7.4)
        ax = axes[i][2]
        xs = np.arange(3)
        us = rec["cluster_share"]["2m"]["unconstrained"]
        cs = rec["cluster_share"]["2m"]["constrained"]
        ax.bar(xs - 0.19, us, 0.38, color="#9fb6cd", label="无约束 K 均值")
        ax.bar(xs + 0.19, cs, 0.38, color="#1f4e79", label="邻接约束层次聚类")
        for x, (a, b) in enumerate(zip(us, cs)):
            ax.text(x - 0.19, a + 0.015, f"{a:.2f}", ha="center", fontsize=7)
            ax.text(x + 0.19, b + 0.015, f"{b:.2f}", ha="center", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"第 {q + 1} 档" for q in xs], fontsize=8.5)
        ax.set_ylim(0, 0.9)
        ax.set_ylabel("二米簇的面积份额", fontsize=9)
        ax.set_title(f"({('a', 'b', 'c', 'd', 'e', 'f')[i * 3 + 2]}) {lab}二米簇份额",
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
    for var, lab, col, mk in (("h_max", "水深", "#1f4e79", "o"),
                              ("speed", "流速", "#e08b2b", "s")):
        sil = K["variables"][var]["silhouette"]
        ax.plot(ks, [sil[str(k)] for k in ks], mk + "-", color=col, lw=1.3,
                ms=4.4, label=lab)
    ax.axvline(K["chosen_k"], color="#888888", lw=0.9, ls=":")
    ax.text(K["chosen_k"] + 0.08, ax.get_ylim()[0], f"选定 K = {K['chosen_k']}",
            fontsize=7.6, color="#555555", va="bottom")
    ax.set_xticks(ks)
    ax.set_xlabel("簇数 K", fontsize=9)
    ax.set_ylabel("二米参考场上的轮廓系数", fontsize=9)
    ax.set_title("(a) 簇数选择依据：轮廓系数", fontsize=9.6, loc="left")
    ax.legend(frameon=False, fontsize=8)
    for j, (var, lab) in enumerate((("h_max", "水深"), ("speed", "流速"))):
        ax = axes[0][1] if j == 0 else axes[1][0]
        rec = K["variables"][var]["per_k"]
        allp = list(rec[str(K["chosen_k"])].keys())
        for p in allp:
            ys = [rec[str(k)][p]["v"] for k in ks]
            if p in K_PAIRS:
                col = {"5m|10m": "#1f4e79", "10m|20m": "#e08b2b",
                       "2m|30m": "#c0392b"}[p]
                ax.plot(ks, ys, "o-", color=col, lw=1.4, ms=4.0,
                        label=p.replace("|", " 米与 ") + " 米")
            else:
                ax.plot(ks, ys, "-", color="#c9c9c9", lw=0.8, zorder=1)
        ax.set_xticks(ks)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("簇数 K", fontsize=9)
        ax.set_ylabel("V-measure", fontsize=9)
        ax.set_title(f"({'bc'[j]}) {lab}各分辨率组合的 V-measure 随簇数变化",
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
    ax.set_xlabel("簇数 K", fontsize=9)
    ax.set_ylabel("与 K = 3 排序的斯皮尔曼相关", fontsize=9)
    ax.set_title("(d) 水深十个分辨率组合排序的稳定性", fontsize=9.6, loc="left")
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
    for var, lab, col, mk in (("h_max", "水深", "#1f4e79", "o"),
                              ("speed", "流速", "#e08b2b", "s")):
        rec = sc["variables"][var]
        x = [q["python_v"] for q in rec.values()]
        y = [q["sabre_v"] for q in rec.values()]
        lo = min(lo, min(x))
        hi = max(hi, max(x))
        ax.plot(x, y, mk, color=col, ms=5.0, label=f"{lab}（{len(x)} 对）")
    ax.plot([lo, hi], [lo, hi], color="#888888", lw=0.9, ls="--",
            label="一 比 一 参照线")
    ax.set_xlabel("按定义直接计算得到的 V-measure", fontsize=9)
    ax.set_ylabel("SABRE 0.4.3 给出的 V-measure", fontsize=9)
    ax.set_title("(a) 两种实现的逐对比较", fontsize=9.6, loc="left")
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    ax.text(0.98, 0.06, f"最大绝对差 {sc['max_abs_diff']:.2e}",
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
            f"双精度机器精度 {np.finfo(np.float64).eps:.1e}",
            transform=ax.get_yaxis_transform(), fontsize=7.4, color="#c0392b")
    ax.set_xlabel("变量与分辨率组合", fontsize=9)
    ax.set_ylabel("V-measure 的绝对差（对数刻度）", fontsize=9)
    ax.set_title("(b) 每一对的绝对偏差", fontsize=9.6, loc="left")
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
