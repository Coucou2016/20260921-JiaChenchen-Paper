#!/usr/bin/env python
"""Image-space evolution: how predictions change across training checkpoints.

Given a set of checkpoint snapshots (see scripts/watch_checkpoints.py), this
renders, for a fixed tile, how the predicted depth evolves epoch by epoch:

  visualizations/evolution/<tile>/evolution_<tile>.png       rows = epochs
  visualizations/evolution/<tile>/evolution_metrics_<tile>.png  metric vs epoch

If no snapshots exist yet, it falls back to whatever checkpoints are available
(best_csi / best_rmse_wet / last) so it still produces something useful.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
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
from metrics.flood_metrics import compute_flood_metrics
from viz.common import OUT_DIR, VIZ_DIR, apply_style, depth_vmax, save, show_depth
from viz.reconstruct import TilePrediction, agreement_map, predict_tile
from viz.common import AGREE_CMAP

EP_RE = re.compile(r"(?:^|[^0-9])ep(\d{3,4})(?!\d)")


def discover_checkpoints(snap_dir: Path, out_dir: Path) -> list[tuple[int, Path]]:
    found: dict[int, Path] = {}
    # best-first ranking so a later snapshot of the same epoch wins
    for p in sorted(snap_dir.glob("ep*.pt")):
        m = EP_RE.search(p.stem)
        if m:
            found[int(m.group(1))] = p
    for p in sorted(snap_dir.glob("best_ep*.pt")):
        m = EP_RE.search(p.stem)
        if m:
            found.setdefault(int(m.group(1)), p)
    if not found:
        fallbacks = {
            0: out_dir / "best_csi.pt",
        }
        for e, p in fallbacks.items():
            if p.exists():
                found[e] = p
    return sorted(found.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0_10m2m_hmax.yaml")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--split", default="val")
    ap.add_argument("--index", type=int, default=None, help="dataset index; default = wettest tile")
    ap.add_argument("--max-rows", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    apply_style()
    out_dir = Path(args.out_dir)
    snap_dir = out_dir / "snapshots"

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    dc = cfg["dataset"]
    ds = WellingtonFixedSRDataset(
        root=ROOT / dc.get("root", "dataset"), split=args.split,
        lr_res=int(dc.get("lr_res", 10)), hr_res=int(dc.get("hr_res", 2)),
        target=dc.get("target", "h_max"), geo_mode=dc.get("geo_mode", "all"),
        scenarios=tuple(dc.get("scenarios", ["20a", "100a"])),
    )

    # Choose a tile that actually floods: requires one cheap metric pass per candidate
    idx = args.index
    if idx is None:
        best_wet, idx = -1.0, 0
        for i in range(min(len(ds), 120)):
            hr = ds[i]["hr"][0]
            mask = ds[i]["mask"]
            wf = float(((hr > 0.05) & mask).sum() / max(mask.sum(), 1))
            if wf > best_wet:
                best_wet, idx = wf, i
        print(f"selected index={idx} wet_frac={best_wet:.3f}")
    batch = collate_fixed([ds[idx]])
    tile_name = f"{batch['scenario'][0]}_iy{int(batch['iy'][0]):02d}_ix{int(batch['ix'][0]):02d}"

    ckpts = discover_checkpoints(snap_dir, out_dir)
    if not ckpts:
        print("no checkpoints found; run training first")
        return
    if len(ckpts) > args.max_rows:
        sel = np.linspace(0, len(ckpts) - 1, args.max_rows).round().astype(int)
        ckpts = [ckpts[i] for i in sorted(set(sel))]
    print("checkpoints: " + ", ".join(f"ep{e}" for e, _ in ckpts))

    gt = batch["hr"][0, 0].numpy()
    valid = batch["mask"][0].numpy().astype(bool)
    coarse = batch["lr"][0, 0].numpy()
    vmax = depth_vmax(gt, q=99.5)

    results: list[tuple[int, np.ndarray, dict]] = []
    for epoch, path in ckpts:
        model = build_model(cfg["model"]["name"], cfg)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        tp = predict_tile(model, batch, device=torch.device("cpu"))
        results.append((epoch, tp.pred, tp.metrics))
        m = tp.metrics
        print(f"  ep{epoch:>4}  CSI005={m['CSI_005']:.4f}  MAE_wet={m['MAE_wet']:.4f}  "
              f"RMSE_wet={m['RMSE_wet']:.4f}")

    n = len(results)
    fig = plt.figure(figsize=(13.0, 2.35 * n + 1.6))
    gs = fig.add_gridspec(n, 4, hspace=0.22, wspace=0.10)
    scale = max(1, round(gt.shape[-1] / max(coarse.shape[-1], 1)))
    coarse_up = np.repeat(np.repeat(coarse, scale, axis=-2), scale, axis=-1)[
        : gt.shape[0], : gt.shape[1]
    ]
    for r, (epoch, pred, m) in enumerate(results):
        for c, (title, arr) in enumerate(
            [("Prediction", pred), ("Ground truth", gt)]
        ):
            ax = fig.add_subplot(gs[r, c])
            show_depth(ax, arr, valid, vmax=vmax, title=(title if r == 0 else ""))
            if c == 0:
                ax.set_ylabel(f"ep {epoch}\nCSI={m['CSI_005']:.3f}", fontsize=7.5)
        ax = fig.add_subplot(gs[r, 2])
        err = np.where(valid, np.abs(pred - gt), np.nan)
        im = ax.imshow(err, cmap="inferno_r", vmin=0, vmax=max(np.nanpercentile(err, 98), 1e-3))
        if r == 0:
            ax.set_title("|error|")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        from viz.common import side_colorbar

        side_colorbar(fig, ax, im, fmt="%.2f")
        ax = fig.add_subplot(gs[r, 3])
        ax.imshow(agreement_map(pred, gt, valid), cmap=AGREE_CMAP, vmin=0, vmax=1)
        if r == 0:
            ax.set_title("wet extent @0.05 m")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    fig.suptitle(
        f"Training evolution of the prediction — tile {tile_name} ({args.split}), 10 m → 2 m h_max\n"
        f"first row: coarse block-mean of the input · vmax = {vmax:.2f} m",
        fontsize=12, fontweight="bold", y=1.0,
    )
    d = VIZ_DIR / "evolution" / tile_name
    d.mkdir(parents=True, exist_ok=True)
    p1 = save(fig, d / "evolution_predictions.png")
    print(f"wrote {p1}")

    fig2, ax2 = plt.subplots(figsize=(8.5, 4.6))
    eps = [e for e, _, _ in results]
    for key, color, tag in (("CSI_005", "#2e7d32", "CSI @0.05"), ("CSI_030", "#1565c0", "CSI @0.30"),
                            ("CSI_100", "#8e24aa", "CSI @1.00")):
        ax2.plot(eps, [m.get(key, np.nan) for _, _, m in results], "o-", color=color, lw=2, label=tag)
    ax2.set_xlabel("checkpoint epoch")
    ax2.set_ylabel("CSI (this tile)")
    ax2.set_title(f"Tile-level metrics vs training epoch — {tile_name}")
    ax2.legend()
    ax2b = ax2.twinx()
    ax2b.plot(eps, [m.get("MAE_wet", np.nan) for _, _, m in results], "s--", color="#e65100",
              lw=1.6, label="MAE wet (m)")
    ax2b.set_ylabel("MAE wet (m)", color="#e65100")
    ax2b.tick_params(axis="y", labelcolor="#e65100")
    ax2b.grid(False)
    ax2b.legend(loc="center right")
    p2 = save(fig2, d / "evolution_metrics.png")
    print(f"wrote {p2}")

    (d / "evolution_metrics.json").write_text(
        json.dumps(
            {"tile": tile_name, "split": args.split, "index": idx,
             "checkpoints": [{"epoch": e, "path": str(p), "metrics": m} for (e, _, m), (_, p) in zip(results, ckpts)]},
            indent=2, default=float,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
