"""Generate all report figures in SciencePlots style with Times New Roman."""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  (registers the 'science' style)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report_figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.family": "serif",
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
C = {"model": "#1f4e79", "bilin": "#e08b2b", "near": "#7a7a7a",
     "frozen": "#5b5b5b", "w005": "#2a7fbf", "w01": "#c0392b", "ctl": "#8c8c8c",
     "deep": "#c0392b", "shallow": "#2a7fbf"}


def save(fig, name):
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print(f"wrote {p.name}")


# ---------------------------------------------------------------- data
def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


scan = load("outputs/finetune_deep/epoch_scan.json")
scans = scan["scans"]
refs = scan["refs"]
sel = load("outputs/finetune_deep/selection.json")
winner_test = load("outputs/finetune_deep/metrics/winner_w01_ep187_test.json")
ev = load("outputs/post_sweep/eval_results.json")
d0 = load("outputs/deep_sweep/D0/visualizations/diagnostics_val.json")
d2 = load("outputs/deep_sweep/D2/visualizations/diagnostics_val.json")
sweep = {t: load(f"outputs/deep_sweep/{t}/status.json")["metrics"] for t in ("D0", "D2")}
EPS = list(range(181, 201))


# ---------------------------------------------------------------- Fig 1: task schematic
def fig_task():
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.4)

    boxes = [
        (0.15, "Low-resolution depth\n10 m grid  (h_max)", "#dce6f2"),
        (2.65, "HydroGeo-SRNO\nneural operator", "#f6e3d5"),
        (5.15, "High-resolution depth\n2 m grid  (h_max)", "#dcefe0"),
        (7.65, "Ground truth\n2 m simulation", "#efe6f6"),
    ]
    for x, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.25), 2.05, 1.1, boxstyle="round,pad=0.06",
                                    fc=col, ec="#444", lw=1.0))
        ax.text(x + 1.02, 1.80, txt, ha="center", va="center", fontsize=9.5)

    for x0, x1 in [(2.20, 2.65), (4.70, 5.15)]:
        ax.add_patch(FancyArrowPatch((x0, 1.80), (x1, 1.80), arrowstyle="-|>",
                                     mutation_scale=14, lw=1.3, color="#333"))
    ax.add_patch(FancyArrowPatch((7.65, 1.80), (7.20, 1.80), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.3, ls="--", color="#333"))
    ax.text(7.42, 1.55, "loss", ha="center", fontsize=8.5, style="italic")

    ax.text(1.18, 2.60, "input", ha="center", fontsize=9, color="#1f4e79")
    ax.text(6.18, 2.60, "prediction", ha="center", fontsize=9, color="#1f4e79")
    ax.text(5.0, 0.62, "static geography: terrain, slope, land use, "
                       "distance to channel, imperviousness",
            ha="center", fontsize=8.5, color="#555")
    ax.add_patch(FancyArrowPatch((5.0, 1.22), (5.0, 0.82), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.0, color="#777"))
    ax.text(5.0, 3.05, "Guided super-resolution of urban flood depth (factor 5)",
            ha="center", fontsize=11, fontweight="bold")
    save(fig, "fig01_task.png")


# ---------------------------------------------------------------- Fig 2: baselines
def fig_baselines():
    mets = [("CSI_005", "CSI@0.05 m"), ("CSI_030", "CSI@0.30 m"), ("CSI_100", "CSI@1.00 m")]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    for ax, split in zip(axes, ("val", "test")):
        x = np.arange(len(mets)); wd = 0.26
        series = [("HydroGeo-SRNO", [ev[f"baseline_v0_{split}"][k] for k, _ in mets], C["model"]),
                  ("Bilinear", [ev[f"bilinear_{split}"][k] for k, _ in mets], C["bilin"]),
                  ("Nearest", [ev[f"nearest_{split}"][k] for k, _ in mets], C["near"])]
        for i, (lab, vals, col) in enumerate(series):
            b = ax.bar(x + (i - 1) * wd, vals, wd, label=lab, color=col, ec="black", lw=0.6)
            ax.bar_label(b, fmt="%.2f", fontsize=7.0, padding=1.5)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in mets])
        ax.set_ylim(0, 0.75)
        ax.set_ylabel("Critical Success Index")
        ax.set_title(f"{'Validation' if split == 'val' else 'Test'} split")
        if split == "val":
            ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    save(fig, "fig02_baselines.png")


# ---------------------------------------------------------------- Fig 3: depth-binned
def fig_depth_bins():
    bins = d0["depth_bins"]
    x = np.arange(len(bins))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    a1.bar(x - 0.2, d0["depth"]["bias"], 0.4, label="Control (w=0)", color=C["ctl"],
           ec="black", lw=0.5)
    a1.bar(x + 0.2, d2["depth"]["bias"], 0.4, label="Deep term (w=0.5)", color=C["deep"],
           ec="black", lw=0.5)
    a1.axhline(0, color="black", lw=0.8)
    a1.set_xticks(x); a1.set_xticklabels(bins, rotation=45, ha="right")
    a1.set_ylabel("Mean bias (m)"); a1.set_xlabel("True depth bin (m)")
    a1.set_title("(a) Depth-binned bias"); a1.legend(frameon=False, fontsize=8)
    a2.plot(x, d0["depth"]["mae"], "o-", label="Control (w=0)", color=C["ctl"], ms=4)
    a2.plot(x, d2["depth"]["mae"], "s-", label="Deep term (w=0.5)", color=C["deep"], ms=4)
    a2.set_xticks(x); a2.set_xticklabels(bins, rotation=45, ha="right")
    a2.set_ylabel("Mean absolute error (m)"); a2.set_xlabel("True depth bin (m)")
    a2.set_title("(b) Depth-binned error"); a2.legend(frameon=False, fontsize=8)
    save(fig, "fig03_depth_bins.png")


# ---------------------------------------------------------------- Fig 4: paired fine-tune
def fig_paired():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    panels = [("CSI_100", "CSI@1.00 m (deep water)", False),
              ("CSI_005", "CSI@0.05 m (overall)", False),
              ("VolumeRelativeError", "Volume relative error", True)]
    arms = [("w00", "w_deep = 0 (control)", C["ctl"], "o"),
            ("w005", "w_deep = 0.05", C["w005"], "s"),
            ("w01", "w_deep = 0.10", C["w01"], "^")]
    for ax, (m, title, lower) in zip(axes, panels):
        for arm, lab, col, mk in arms:
            y = [scans[arm][f"ep{e:04d}"][m] for e in EPS]
            ax.plot(EPS, y, marker=mk, ms=3.2, lw=1.4, color=col, label=lab)
        ax.axhline(refs["frozen_ep180"][m], color="k", ls="--", lw=1.0)
        ax.set_title(title, fontsize=10); ax.set_xlabel("Fine-tune epoch")
        if m == "CSI_100":
            ax.legend(frameon=False, fontsize=8, loc="lower right")
            ax.text(0.02, 0.97, "dashed = frozen ep180", transform=ax.transAxes,
                    fontsize=7.5, va="top", color="#333")
    axes[0].set_ylabel("score")
    axes[2].set_ylabel("relative error")
    save(fig, "fig04_paired_finetune.png")


# ---------------------------------------------------------------- Fig 5: selection noise
def fig_selection_noise():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.5))
    # (a) subset CSI_100 for the control (from history) vs full-val
    h00 = [json.loads(l) for l in
           (ROOT / "outputs/finetune_deep/w00/history.jsonl").read_text(encoding="utf-8").splitlines()
           if l.strip()]
    hep = [int(r["epoch"]) for r in h00]
    hsub = [r["CSI_100"] for r in h00]
    a1.plot(hep, hsub, "o-", color=C["ctl"], ms=3.5, lw=1.3, label="60-batch subset")
    full = [scans["w00"][f"ep{e:04d}"]["CSI_100"] for e in hep]
    a1.plot(hep, full, "s-", color=C["model"], ms=3.5, lw=1.3, label="full 182-tile val")
    a1.fill_between([min(hep), max(hep)], min(hsub), max(hsub), color=C["ctl"], alpha=0.13)
    a1.annotate(f"subset spread = {max(hsub)-min(hsub):.3f}\n(pure noise: no deep term)",
                xy=(hep[-1], max(hsub)), xytext=(hep[0], max(hsub) + 0.012), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    a1.set_xlabel("Fine-tune epoch"); a1.set_ylabel("CSI@1.00 m")
    a1.set_title("(a) Same arm, two measurement bases")
    a1.legend(frameon=False, fontsize=8)
    # (b) candidate minus control, subset vs full
    h5 = [json.loads(l) for l in
          (ROOT / "outputs/finetune_deep/w005/history.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip()]
    e5 = [int(r["epoch"]) for r in h5]
    dsub = [r["CSI_100"] - s for r, s in zip(h5, hsub) if int(r["epoch"]) in hep]
    dfull = [scans["w005"][f"ep{e:04d}"]["CSI_100"] - scans["w00"][f"ep{e:04d}"]["CSI_100"] for e in e5]
    a2.axhline(0, color="black", lw=0.8)
    a2.plot(e5, dsub, "o-", color=C["ctl"], ms=3.5, lw=1.3, label="subset basis")
    a2.plot(e5, dfull, "s-", color=C["w01"], ms=3.5, lw=1.3, label="full-val basis (paired)")
    a2.set_xlabel("Fine-tune epoch"); a2.set_ylabel("CSI@1.00 m gain over control")
    a2.set_title("(b) The effect, measured two ways")
    a2.legend(frameon=False, fontsize=8)
    save(fig, "fig05_selection_noise.png")


# ---------------------------------------------------------------- Fig 6: before/after
def fig_before_after():
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6))
    grp = [("CSI_100", "CSI@1.00 m"), ("PeakDepthError", "Peak depth error (m)"),
           ("VolumeRelativeError", "Volume rel. error"), ("RMSE_wet", "RMSE wet (m)")]
    for ax, split in zip(axes, ("val", "test")):
        x = np.arange(len(grp)); wd = 0.36
        fz = [refs["frozen_ep180"][k] if split == "val" else ev["baseline_v0_test"][k] for k, _ in grp]
        wn = [sel["winner_val_metrics"][k] if split == "val" else winner_test[k] for k, _ in grp]
        b1 = ax.bar(x - wd/2, fz, wd, label="Frozen V0", color=C["frozen"], ec="black", lw=0.6)
        b2 = ax.bar(x + wd/2, wn, wd, label="Fine-tuned (w=0.10)", color=C["w01"], ec="black", lw=0.6)
        ax.bar_label(b1, fmt="%.3f", fontsize=6.8, padding=1.2)
        ax.bar_label(b2, fmt="%.3f", fontsize=6.8, padding=1.2)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in grp], rotation=20, ha="right")
        ax.set_title(f"{'Validation' if split == 'val' else 'Test'} split")
        ax.set_ylim(0, max(max(fz), max(wn)) * 1.28)
        if split == "val":
            ax.legend(frameon=False, fontsize=8.5)
        for i, (k, _) in enumerate(grp):
            lo = k in ("PeakDepthError", "VolumeRelativeError", "RMSE_wet")
            good = (wn[i] < fz[i]) if lo else (wn[i] > fz[i])
            ax.text(i, max(fz[i], wn[i]) * 1.10, "better" if good else "worse",
                    ha="center", fontsize=7, color="#1a7a3a" if good else "#b03030")
    save(fig, "fig06_before_after.png")


# ---------------------------------------------------------------- Fig 7: forest plot
def fig_forest():
    pairs = [("CSI@1.00 m", "w005", "CSI_100", False),
             ("CSI@1.00 m", "w01", "CSI_100", False),
             ("CSI@0.05 m", "w005", "CSI_005", False),
             ("CSI@0.05 m", "w01", "CSI_005", False),
             ("Volume error", "w005", "VolumeRelativeError", True),
             ("Volume error", "w01", "VolumeRelativeError", True),
             ("Peak depth err.", "w005", "PeakDepthError", True),
             ("Peak depth err.", "w01", "PeakDepthError", True)]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    labels, means, los, his, cols = [], [], [], [], []
    for name, arm, metric, lower in pairs:
        ctl = [scans["w00"][f"ep{e:04d}"][metric] for e in EPS]
        cand = [scans[arm][f"ep{e:04d}"][metric] for e in EPS]
        d = np.array(cand) - np.array(ctl)
        if lower:
            d = -d  # flip so positive always means improvement
        md = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
        labels.append(f"{name}  |  w={0.05 if arm=='w005' else 0.10}")
        means.append(md); los.append(md - 1.96 * se); his.append(md + 1.96 * se)
        cols.append(C["w005"] if arm == "w005" else C["w01"])
    y = np.arange(len(labels))[::-1]
    ax.axvline(0, color="black", lw=1.0)
    for yi, m, lo, hi, c in zip(y, means, los, his, cols):
        ax.plot([lo, hi], [yi, yi], color=c, lw=1.8)
        ax.plot([m], [yi], "o", color=c, ms=6)
        sig = (lo > 0) or (hi < 0)
        ax.text(hi + 0.0025, yi, ("improved" if m > 0 else "worse") + (" *" if sig else ""),
                va="center", fontsize=7.5)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Paired change vs control, positive = better  (* 95% CI excludes 0)")
    save(fig, "fig07_forest.png")


# ---------------------------------------------------------------- Fig 8: bias evolution
def fig_bias_evo():
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.4))
    for ax, arm, name, col in ((axes[0], "w005", "w_deep = 0.05", C["w005"]),
                               (axes[1], "w01", "w_deep = 0.10", C["w01"])):
        p = ROOT / f"outputs/finetune_deep/{arm}/visualizations/bias_evolution.json"
        if not p.exists():
            continue
        j = json.loads(p.read_text(encoding="utf-8"))
        cs = j["curves"]
        ep = [c["epoch"] for c in cs]
        ax.plot(ep, [c["bias_1_1.5"] for c in cs], "o-", ms=3.5, color="#2a7fbf",
                label="bias, 1.0-1.5 m")
        ax.plot(ep, [c["bias_gt3"] for c in cs], "s-", ms=3.5, color="#c0392b",
                label="bias, > 3 m")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Fine-tune epoch"); ax.set_ylabel("Mean bias (m)")
        ax.set_title(f"{name}  (fixed val panel)")
        ax.legend(frameon=False, fontsize=8)
    save(fig, "fig08_bias_evolution.png")


# ---------------------------------------------------------------- Fig 9: sweep
def fig_sweep():
    """From-scratch sweep: the deep term helps the tail but costs overall accuracy.

    Numbers are the full 182-tile validation metrics of each arm's final checkpoint
    (outputs/deep_sweep/*/metrics_last.json), so both bars sit on the same basis.
    """
    d0 = json.loads((ROOT / "outputs/deep_sweep/D0/metrics_last.json").read_text(encoding="utf-8"))
    d2 = json.loads((ROOT / "outputs/deep_sweep/D2/metrics_last.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    groups = [("CSI@1.00 m\n(deep water)", "CSI_100"), ("CSI@0.05 m\n(overall)", "CSI_005")]
    x = np.arange(len(groups)); wd = 0.36
    v0 = [d0[k] for _, k in groups]
    v2 = [d2[k] for _, k in groups]
    b1 = ax.bar(x - wd/2, v0, wd, label="Control (w = 0)", color=C["ctl"], ec="black", lw=0.6)
    b2 = ax.bar(x + wd/2, v2, wd, label="Deep term (w = 0.5)", color=C["deep"], ec="black", lw=0.6)
    ax.bar_label(b1, fmt="%.3f", fontsize=8.5, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=8.5, padding=2)
    ax.annotate("", xy=(0 + wd/2, v2[0] + 0.030), xytext=(0 - wd/2, v0[0] + 0.030),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#1a7a3a"))
    ax.text(0, max(v0[0], v2[0]) + 0.045, "+0.022", ha="center", fontsize=8, color="#1a7a3a")
    ax.annotate("", xy=(1 + wd/2, v2[1] + 0.030), xytext=(1 - wd/2, v0[1] + 0.030),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#b03030"))
    ax.text(1, max(v0[1], v2[1]) + 0.045, "-0.043", ha="center", fontsize=8, color="#b03030")
    ax.set_xticks(x); ax.set_xticklabels([g for g, _ in groups])
    ax.set_ylabel("Critical Success Index")
    ax.set_ylim(0, 0.62)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Training from scratch (40 epochs each)")
    save(fig, "fig09_sweep.png")


# ---------------------------------------------------------------- Fig 10: metrics heatmap
def fig_heatmap():
    """Normalised score card: how each model performs on each metric (val)."""
    models = ["Bilinear", "Nearest", "Frozen V0", "Fine-tuned w=0.10"]
    mets = [("CSI_005", True), ("CSI_030", True), ("CSI_100", True),
            ("RMSE_wet", False), ("PeakDepthError", False), ("VolumeRelativeError", False)]
    rows = [
        [ev["bilinear_val"][k] for k, _ in mets],
        [ev["nearest_val"][k] for k, _ in mets],
        [refs["frozen_ep180"][k] for k, _ in mets],
        [sel["winner_val_metrics"][k] for k, _ in mets],
    ]
    arr = np.array(rows, dtype=float)
    norm = np.zeros_like(arr)
    for j, (k, higher) in enumerate(mets):
        col = arr[:, j]
        lo, hi = col.min(), col.max()
        norm[:, j] = (col - lo) / (hi - lo) if hi > lo else 0.5
        if not higher:
            norm[:, j] = 1 - norm[:, j]
    fig, ax = plt.subplots(figsize=(9.6, 3.0))
    im = ax.imshow(norm, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(mets)))
    ax.set_xticklabels([k.replace("_", "@") for k, _ in mets], rotation=25, ha="right")
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=9)
    for i in range(len(models)):
        for j in range(len(mets)):
            val = arr[i, j]
            txt = f"{val:.3f}" if val < 10 else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black")
    ax.set_title("Validation score card (green = relatively better in that column)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label("relative rank", fontsize=8); cb.ax.tick_params(labelsize=7)
    save(fig, "fig10_scorecard.png")


if __name__ == "__main__":
    fig_task(); fig_baselines(); fig_depth_bins(); fig_paired(); fig_selection_noise()
    fig_before_after(); fig_forest(); fig_bias_evo(); fig_sweep(); fig_heatmap()
    print("all figures done")
