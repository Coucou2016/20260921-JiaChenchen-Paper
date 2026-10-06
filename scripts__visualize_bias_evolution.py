#!/usr/bin/env python
"""Track the depth-binned bias curve across training checkpoints.

Motivation: the aggregate metrics hide *where* the error lives. At epoch 78 the
model still under-predicted deep water badly (bias -0.79 m at 1-1.5 m, -2.17 m
above 3 m). CSI@1.00 m is the weakest metric. This figure answers a single
decision-relevant question: **is the deep-water bias actually shrinking as
training proceeds, or is it frozen?**

Uses the `snapshots/epNNNN.pt` files written by scripts/watch_checkpoints.py and
runs on CPU only.

Outputs:
  visualizations/04_bias_evolution.png
  visualizations/bias_evolution.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from dataset.wellington_fixed_sr import WellingtonFixedSRDataset, collate_fixed
from engine.evaluator import build_model
from viz.common import OUT_DIR, VIZ_DIR, apply_style, save, stamp

DEPTH_BINS = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 1e3])
EP_RE = re.compile(r"ep(\d{3,4})")


def bin_labels(bins: np.ndarray) -> list[str]:
    return [f"{bins[i]:g}-{'inf' if bins[i+1] >= 1e3 else format(bins[i+1], 'g')}"
            for i in range(len(bins) - 1)]


def discover(snap_dir: Path, out_dir: Path) -> list[tuple[int, Path]]:
    found: dict[int, Path] = {}
    for p in sorted(snap_dir.glob("ep*.pt")):
        m = EP_RE.search(p.stem)
        if m:
            found[int(m.group(1))] = p
    if not found:
        p = out_dir / "best_csi.pt"
        if p.exists():
            found[0] = p
    return sorted(found.items())


@torch.no_grad()
def bias_curve(model, ds, indices, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nb = len(bins) - 1
    n = np.zeros(nb); s = np.zeros(nb); sq = np.zeros(nb)
    for i in indices:
        batch = collate_fixed([ds[i]])
        out = model(
            lr=batch["lr"], lr_valid=batch["lr_valid"],
            static_cont=batch["static_cont"], landuse=batch["landuse"],
            lr_res=batch["lr_res"], hr_res=batch["hr_res"],
        )
        pred = out["depth"][0, 0].numpy()
        truth = batch["hr"][0, 0].numpy()
        valid = batch["mask"][0].numpy().astype(bool)
        if valid.sum() == 0:
            continue
        idx = np.clip(np.digitize(truth[valid], bins) - 1, 0, nb - 1)
        d = pred[valid] - truth[valid]
        np.add.at(n, idx, 1.0)
        np.add.at(s, idx, d)
        np.add.at(sq, idx, d * d)
    nn = np.maximum(n, 1)
    return s / nn, np.sqrt(sq / nn), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0_10m2m_hmax.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--tiles", type=int, default=40, help="val tiles sampled per checkpoint")
    ap.add_argument("--max-checkpoints", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    apply_style()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    dc = cfg["dataset"]
    ds = WellingtonFixedSRDataset(
        root=ROOT / dc.get("root", "dataset"), split=args.split,
        lr_res=int(dc.get("lr_res", 10)), hr_res=int(dc.get("hr_res", 2)),
        target=dc.get("target", "h_max"), geo_mode=dc.get("geo_mode", "all"),
        scenarios=tuple(dc.get("scenarios", ["20a", "100a"])),
    )

    ckpts = discover(OUT_DIR / "snapshots", OUT_DIR)
    if not ckpts:
        print("no checkpoints found; run training first")
        return
    if len(ckpts) > args.max_checkpoints:
        sel = np.linspace(0, len(ckpts) - 1, args.max_checkpoints).round().astype(int)
        ckpts = [ckpts[i] for i in sorted(set(sel))]
    print("checkpoints: " + ", ".join(f"ep{e}" for e, _ in ckpts))

    # fixed tile sample so curves are comparable across epochs
    rng = np.random.default_rng(42)
    indices = np.sort(rng.choice(len(ds), size=min(args.tiles, len(ds)), replace=False))
    print(f"splits={args.split}  tiles sampled={len(indices)}")

    curves: list[dict] = []
    t0 = time.time()
    for epoch, path in ckpts:
        model = build_model(cfg["model"]["name"], cfg)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        bias, rmse, n = bias_curve(model, ds, indices, DEPTH_BINS)
        skew, kurt = None, None
        peaks = []
        deep = n[-1]
        print(f"  ep{epoch:>4}  deep>3m n={int(deep):>7}  bias={bias[-1]:+.3f}  rmse={rmse[-1]:.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        curves.append({
            "epoch": epoch, "path": str(path),
            "bias": bias.tolist(), "rmse": rmse.tolist(), "n": n.tolist(),
            "bias_1_1.5": float(bias[7]), "bias_gt3": float(bias[-1]),
            "rmse_gt3": float(rmse[-1]),
            "mean_bias_wet": float(np.average(bias[:-1], weights=np.maximum(n[:-1], 1))),
        })

    # ------------------------------------------------------------------ figure
    epochs = [c["epoch"] for c in curves]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(curves) - 1, 1)) for i in range(len(curves))]
    x = np.arange(len(DEPTH_BINS) - 1)

    fig = plt.figure(figsize=(14.5, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.24)

    ax = fig.add_subplot(gs[0, 0])
    for c, col in zip(curves, colors):
        ax.plot(x, c["bias"], "o-", color=col, lw=1.8, ms=4, label=f"ep {c['epoch']}")
    ax.axhline(0, color="#333", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(bin_labels(DEPTH_BINS), rotation=45, ha="right", fontsize=7)
    ax.set_title("Bias vs true depth, across training checkpoints")
    ax.set_ylabel("mean (pred - truth)  [m]"); ax.set_xlabel("true depth bin (m)")
    ax.legend(fontsize=7, ncol=2)

    ax = fig.add_subplot(gs[0, 1])
    for c, col in zip(curves, colors):
        ax.plot(x, c["rmse"], "s-", color=col, lw=1.8, ms=4, label=f"ep {c['epoch']}")
    ax.set_xticks(x); ax.set_xticklabels(bin_labels(DEPTH_BINS), rotation=45, ha="right", fontsize=7)
    ax.set_title("RMSE vs true depth, across checkpoints")
    ax.set_ylabel("RMSE (m)"); ax.set_xlabel("true depth bin (m)")
    ax.legend(fontsize=7, ncol=2)

    ax = fig.add_subplot(gs[1, 0])
    for key, col, lbl in (("bias_1_1.5", "#e65100", "bias, 1.0-1.5 m"),
                          ("bias_gt3", "#c62828", "bias, > 3 m"),
                          ("mean_bias_wet", "#1565c0", "mean bias, all wet bins")):
        ys = [c[key] for c in curves]
        ax.plot(epochs, ys, "o-", color=col, lw=2, ms=5, label=lbl)
        if len(curves) >= 3:
            slope, intercept = np.polyfit(np.asarray(epochs, float), np.asarray(ys, float), 1)
            xs = np.linspace(min(epochs), max(epochs), 50)
            ax.plot(xs, slope * xs + intercept, "--", color=col, lw=1.1, alpha=0.75)
    ax.axhline(0, color="#333", lw=0.9)
    ax.set_title("Deep-water bias trend (solid = checkpoints, dashed = fitted slope)")
    ax.set_xlabel("checkpoint epoch"); ax.set_ylabel("mean bias (m)")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(epochs, [c["rmse_gt3"] for c in curves], "s-", color="#6a1b9a", lw=2, ms=5,
            label="RMSE, > 3 m")
    ax2 = ax.twinx()
    ax2.plot(epochs, [c["n"][-1] for c in curves], "^--", color="#00695c", lw=1.4, ms=5,
             label="pixels > 3 m")
    ax2.set_ylabel("# pixels > 3 m", color="#00695c", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#00695c", labelsize=7)
    ax2.grid(False)
    ax.set_title("Extreme-depth error and its sample size")
    ax.set_xlabel("checkpoint epoch"); ax.set_ylabel("RMSE (m)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")

    fig.suptitle(
        f"Deep-water bias evolution - {args.split} split, {len(indices)} fixed tiles, "
        f"CPU inference across {len(curves)} checkpoints",
        fontsize=13, fontweight="bold",
    )
    stamp(fig, "source: outputs/v0_10m2m_hmax/snapshots/*.pt (watch_checkpoints.py)")
    p = save(fig, VIZ_DIR / "04_bias_evolution.png")
    print(f"wrote {p}")

    (VIZ_DIR / "bias_evolution.json").write_text(
        json.dumps({"split": args.split, "tiles": [int(i) for i in indices],
                    "depth_bins": bin_labels(DEPTH_BINS), "curves": curves}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {VIZ_DIR / 'bias_evolution.json'}")

    # ------------------------------------------------------------------ verdict
    # Comparing only the first and last checkpoint is misleading: the per-checkpoint
    # trace is noisy, so a small net change can be pure run-to-run scatter. Use the
    # least-squares slope over all checkpoints and compare it with the residual
    # spread; only call it IMPROVING/WORSENING when the trend exceeds the noise.
    if len(curves) >= 3:
        e = np.asarray(epochs, dtype=float)
        for key, label in (("bias_gt3", "deep water (>3 m)"),
                           ("bias_1_1.5", "1.0-1.5 m"),
                           ("mean_bias_wet", "all wet bins")):
            y = np.asarray([c[key] for c in curves], dtype=float)
            slope, intercept = np.polyfit(e, y, 1)
            resid = y - (slope * e + intercept)
            # noise floor: residual scatter of the fit, plus half the end-to-end swing
            scatter = float(np.std(resid, ddof=1)) if y.size > 2 else float(np.std(y))
            span = float(np.max(y) - np.min(y))
            drift = abs(float(slope) * (e[-1] - e[0]))       # total fitted change
            verdict = ("TRENDING UP" if slope > 0 else "TRENDING DOWN")
            if drift < max(scatter, 0.05):
                verdict = "FLAT vs NOISE"
            print(f"  {label:18s} slope={slope:+.5f} m/epoch  drift={drift:+.3f} m  "
                  f"scatter={scatter:.3f}  span={span:.3f}  => {verdict}")
        print(f"  per-checkpoint deep>3m bias: "
              + "  ".join(f"ep{c['epoch']}={c['bias_gt3']:+.3f}" for c in curves))
        print("  NOTE: on this 30-tile sample the trace is noisy; the fitted slope is the")
        print("        reliable signal, and its sign is what to read.")

    # concise single-line summary for the log
    if len(curves) >= 2:
        first, last = curves[0], curves[-1]
        d3 = last["bias_gt3"] - first["bias_gt3"]
        print(f"deep-water (>3 m) bias: first={first['bias_gt3']:+.3f} last={last['bias_gt3']:+.3f} "
              f"({d3:+.3f} m end-to-end)")


if __name__ == "__main__":
    main()
