#!/usr/bin/env python
"""Training-process visualisation from history.jsonl (live, safe to re-run).

Produces:
  visualizations/01_training_dashboard.png   multi-metric training curves
  visualizations/01_progress_overview.png    one-glance milestone/progress strip

Reads only files the trainer already writes, so it is safe to run at any time.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from viz.common import (
    METRIC_LABELS,
    OUT_DIR,
    VIZ_DIR,
    apply_style,
    best_summary,
    load_history,
    mark_best,
    save,
    series,
    shade_future,
    stamp,
)


def reconstruct_lr(epoch: int, base: float, warmup: int, total: int, start: float = 0.1) -> float:
    if epoch <= warmup:
        return base * (start + (1 - start) * epoch / max(warmup, 1))
    import math
    p = (epoch - warmup) / max(total - warmup, 1)
    p = min(max(p, 0.0), 1.0)
    return base * 0.5 * (1 + math.cos(math.pi * p))


def best_so_far(rows: list[dict], key: str, mode: str = "max") -> tuple[np.ndarray, np.ndarray]:
    ep, vals = series(rows, key)
    if vals.size == 0:
        return ep, vals
    if mode == "max":
        return ep, np.maximum.accumulate(vals)
    return ep, np.minimum.accumulate(vals)


def improvements(rows: list[dict], key: str, mode: str = "max") -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    best = None
    for r in sorted(rows, key=lambda r: int(r["epoch"])):
        v = r.get(key)
        if v is None or (isinstance(v, float) and v != v):
            continue
        v = float(v)
        if best is None or (v > best if mode == "max" else v < best):
            best = v
            out.append((int(r["epoch"]), v))
    return out


def panel(ax, rows, key, *, mode="min", logy=False, color="#1f77b4", lw=1.7, marker="", ms=3,
          label=None, mark=True, alpha=1.0) -> None:
    ep, vals = series(rows, key)
    if vals.size == 0:
        ax.text(0.5, 0.5, f"{key}: no data yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#8a8a8a", style="italic")
        return
    ax.plot(ep, vals, color=color, lw=lw, marker=marker, ms=ms, alpha=alpha,
            label=label or METRIC_LABELS.get(key, key))
    if mark:
        mark_best(ax, rows, key, mode)
    ax.set_ylabel(METRIC_LABELS.get(key, key), fontsize=8)
    if logy:
        ax.set_yscale("log")
    ax.legend(loc="best")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default=str(OUT_DIR / "history.jsonl"))
    ap.add_argument("--total-epochs", type=int, default=300)
    ap.add_argument("--base-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    apply_style()
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_history(args.history)
    if not rows:
        print("no history rows yet; nothing to plot")
        return
    last_epoch = int(rows[-1]["epoch"])
    bs = best_summary() or {}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    env = f"epochs logged: {len(rows)}  |  last epoch: {last_epoch}/{args.total_epochs}  |  {now}"

    # ------------------------------------------------------------------ dashboard
    fig = plt.figure(figsize=(15.5, 12.0))
    gs = fig.add_gridspec(4, 4, hspace=0.42, wspace=0.30)

    # row 1: loss (with lr) + CSI trio
    ax_loss = fig.add_subplot(gs[0, 0])
    panel(ax_loss, rows, "loss", mode="min", logy=True, color="#c0392b")
    lr_ep = np.arange(1, last_epoch + 1)
    lr_vals = [reconstruct_lr(int(e), args.base_lr, args.warmup, args.total_epochs) for e in lr_ep]
    ax_lr = ax_loss.twinx()
    ax_lr.plot(lr_ep, lr_vals, color="#7f8c8d", lw=1.2, ls="--", label="LR (reconstructed)")
    ax_lr.set_ylabel("learning rate", fontsize=8, color="#7f8c8d")
    ax_lr.tick_params(axis="y", labelcolor="#7f8c8d", labelsize=7)
    ax_lr.grid(False)
    ax_lr.legend(loc="upper right")

    ax_csi = fig.add_subplot(gs[0, 1])
    for k, c in (("CSI_005", "#2e7d32"), ("CSI_030", "#1565c0"), ("CSI_100", "#8e24aa")):
        panel(ax_csi, rows, k, mode="max", color=c, mark=(k == "CSI_005"))

    ax_csibest = fig.add_subplot(gs[0, 2])
    for k, c in (("CSI_005", "#2e7d32"), ("CSI_030", "#1565c0"), ("CSI_100", "#8e24aa")):
        ep, vals = best_so_far(rows, k, "max")
        if vals.size:
            ax_csibest.plot(ep, vals, color=c, lw=2.0, label=f"{k} (best-so-far)")
    ax_csibest.set_ylabel("CSI (monotone envelope)")
    ax_csibest.legend(loc="lower right")
    ax_csibest.set_ylim(bottom=0)

    ax_f1 = fig.add_subplot(gs[0, 3])
    for k, c in (("F1_005", "#2e7d32"), ("F1_030", "#1565c0"), ("F1_100", "#8e24aa")):
        panel(ax_f1, rows, k, mode="max", color=c, mark=False)

    # row 2: depth errors + agreement + volume + extremes
    ax_e = fig.add_subplot(gs[1, 0])
    for k, c in (("RMSE_all", "#616161"), ("RMSE_wet", "#e65100"), ("MAE_wet", "#f9a825")):
        panel(ax_e, rows, k, mode="min", color=c, mark=False)

    ax_area = fig.add_subplot(gs[1, 1])
    for k, c in (("VolumeRelativeError", "#00695c"), ("FloodAreaRelativeError", "#6a1b9a")):
        panel(ax_area, rows, k, mode="min", color=c, mark=False)

    ax_peak = fig.add_subplot(gs[1, 2])
    panel(ax_peak, rows, "PeakDepthError", mode="min", color="#ad1457")

    ax_psnr = fig.add_subplot(gs[1, 3])
    ax_psnr.plot(*series(rows, "PSNR"), color="#0277bd", lw=1.7, label="PSNR (dB)")
    mark_best(ax_psnr, rows, "PSNR", "max")
    ax2 = ax_psnr.twinx()
    ax2.plot(*series(rows, "SSIM"), color="#ef6c00", lw=1.7, label="SSIM")
    ax2.set_ylabel("SSIM", fontsize=8, color="#ef6c00")
    ax2.tick_params(axis="y", labelcolor="#ef6c00", labelsize=7)
    ax2.grid(False)
    ax_psnr.set_ylabel("PSNR (dB)")
    h1, l1 = ax_psnr.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax_psnr.legend(h1 + h2, l1 + l2, loc="lower right")

    # row 3: milestone markers + epoch time + loss decomposition proxy
    ax_mile = fig.add_subplot(gs[2, :2])
    ep_c, v_c = series(rows, "CSI_005")
    if ep_c.size:
        ax_mile.plot(ep_c, v_c, color="#455a64", lw=1.6, label="val CSI@0.05")
    imp = improvements(rows, "CSI_005", "max")
    if imp:
        ie = [i[0] for i in imp]
        iv = [i[1] for i in imp]
        ax_mile.scatter(ie, iv, s=52, color="#e67e22", zorder=6, edgecolor="white",
                        linewidth=0.8, label=f"checkpoint improvement ({len(imp)}x)")
        for e, v in imp[-3:]:
            ax_mile.annotate(f"ep{e}\n{v:.3f}", (e, v), textcoords="offset points",
                             xytext=(6, -18), fontsize=7, color="#b8620f")
    ax_mile.axhline(0.5235, color="#2e7d32", ls=":", lw=1.2, label="current best CSI = 0.5235")
    ax_mile.set_title("Milestones: validation CSI@0.05 improvements (checkpoint saves)")
    ax_mile.set_xlabel("epoch")
    ax_mile.set_ylabel("CSI @ 0.05 m")
    ax_mile.legend(loc="lower right")

    ax_time = fig.add_subplot(gs[2, 2])
    ep_t, v_t = series(rows, "sec")
    if v_t.size:
        ax_time.bar(ep_t, v_t, color="#90a4ae", width=0.85)
        ax_time.axhline(v_t.mean(), color="#c0392b", ls="--", lw=1.2,
                        label=f"mean {v_t.mean():.0f} s")
        ax_time.set_ylabel("epoch time (s)")
        ax_time.legend(loc="best")
        ax_time.text(0.02, 0.95, f"total {v_t.sum()/3600:.1f} h", transform=ax_time.transAxes,
                     va="top", fontsize=8, color="#37474f")

    ax_eta = fig.add_subplot(gs[2, 3])
    ax_eta.axis("off")
    mean_sec = float(v_t.mean()) if v_t.size else float("nan")
    remain = max(args.total_epochs - last_epoch, 0)
    lines = [
        "RUN STATE",
        "",
        f"epoch           {last_epoch} / {args.total_epochs}",
        f"logged epochs   {len(rows)}",
        f"elapsed         {v_t.sum()/3600:.2f} h" if v_t.size else "elapsed         n/a",
        f"epoch time      {mean_sec:.0f} s (mean)",
        f"ETA remaining   {remain*mean_sec/3600:.1f} h" if mean_sec == mean_sec else "ETA",
        "",
        f"best CSI_005    {bs.get('best_csi', float('nan')):.4f}" if bs else "best CSI_005    n/a",
        f"best RMSE_wet   {bs.get('best_rmse_wet', float('nan')):.4f}" if bs else "best RMSE_wet   n/a",
        "",
        "learning-rate   warmup 10 -> cosine",
        "checkpoints     last / best_csi / best_rmse_wet",
    ]
    ax_eta.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9,
                family="DejaVu Sans Mono", transform=ax_eta.transAxes)

    # row 4: everything vs epoch, compact
    compact = ["RMSE_wet", "CSI_005", "CSI_030", "CSI_100"]
    for j, k in enumerate(compact):
        ax = fig.add_subplot(gs[3, j])
        panel(ax, rows, k, mode="max" if k.startswith(("CSI", "F1")) else "min",
              color=["#e65100", "#2e7d32", "#1565c0", "#8e24aa"][j], mark=False)
        ax.set_xlabel("epoch")

    fig.suptitle(
        "HydroGeo-SRNO V0 training dynamics — 10 m → 2 m, h_max (log-depth residual + wet head)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    for ax in fig.axes:
        if ax.get_xlabel() == "" and ax.get_ylabel() != "learning rate":
            ax.set_xlabel("epoch", fontsize=8)
        if ax.get_xlim()[1] <= args.total_epochs * 1.02:
            ax.set_xlim(0, args.total_epochs)
    stamp(fig, env)
    p = save(fig, VIZ_DIR / "01_training_dashboard.png")
    print(f"wrote {p}")

    # ------------------------------------------------------------------ overview
    fig2 = plt.figure(figsize=(13, 6.2))
    gs2 = fig2.add_gridspec(2, 2, hspace=0.40, wspace=0.25)
    a1 = fig2.add_subplot(gs2[0, :])
    for k, c in (("CSI_005", "#2e7d32"), ("CSI_030", "#1565c0"), ("CSI_100", "#8e24aa")):
        ep, v = series(rows, k)
        if v.size:
            a1.plot(ep, v, color=c, lw=2, label=k)
    shade_future(a1, last_epoch, args.total_epochs)
    a1.set_title("Validation flood agreement — progress toward CSI saturation")
    a1.set_ylabel("CSI")
    a1.set_xlabel("epoch")
    a1.legend(loc="lower right")
    a1.set_ylim(0, max(0.7, a1.get_ylim()[1]))

    a2 = fig2.add_subplot(gs2[1, 0])
    ep, v = series(rows, "RMSE_wet")
    if v.size:
        a2.plot(ep, v, color="#e65100", lw=2, label="RMSE wet (m)")
    ep2, v2 = series(rows, "MAE_wet")
    if v2.size:
        a2.plot(ep2, v2, color="#f9a825", lw=2, label="MAE wet (m)")
    a2.set_title("Wet-cell depth error")
    a2.set_xlabel("epoch")
    a2.legend(loc="best")

    a3 = fig2.add_subplot(gs2[1, 1])
    ep3, v3 = series(rows, "VolumeRelativeError")
    if v3.size:
        a3.plot(ep3, v3, color="#00695c", lw=2, label="volume rel. err.")
    ep4, v4 = series(rows, "PeakDepthError")
    if v4.size:
        a3.plot(ep4, v4, color="#ad1457", lw=2, label="peak depth err. (m)")
    a3.set_title("Mass / peak extremes")
    a3.set_xlabel("epoch")
    a3.legend(loc="best")
    fig2.suptitle("Stage progress overview", fontsize=12, fontweight="bold")
    stamp(fig2, env)
    p2 = save(fig2, VIZ_DIR / "01_progress_overview.png")
    print(f"wrote {p2}")

    summary = {
        "generated_at": now,
        "last_epoch": last_epoch,
        "logged_epochs": len(rows),
        "total_epochs": args.total_epochs,
        "best_summary": bs,
        "csi005_improvements": [{"epoch": e, "value": v} for e, v in imp],
    }
    (VIZ_DIR / "training_dashboard_state.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
