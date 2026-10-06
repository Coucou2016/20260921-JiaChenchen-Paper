#!/usr/bin/env python
"""Visualise reconstruction quality at a given checkpoint (CPU only).

For a split it computes per-tile flood metrics for every patch, writes a full
per-tile metric table, then renders:

  visualizations/<tag>/per_tile_metrics_<split>.csv / .json
  visualizations/<tag>/tiles_<split>_overview.png     4 representative tiles
  visualizations/<tag>/tiles_<split>_<iy>_<ix>_<scn>.png  detailed per-tile
  visualizations/<tag>/metrics_vs_baseline_<split>.png  model vs B0/B1

Safe to run while training occupies the GPU: everything is on CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
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
from engine.checkpoint import load_checkpoint
from engine.evaluator import build_model
from metrics.flood_metrics import compute_flood_metrics
from viz.common import (
    AGREE_CMAP,
    OUT_DIR,
    VIZ_DIR,
    add_colorbar,
    apply_style,
    depth_vmax,
    load_json,
    save,
    show_depth,
    side_colorbar,
    stamp,
)
from viz.reconstruct import LEGEND_TEXT, TilePrediction, predict_tile

SET_NAMES = [
    "RMSE_all", "RMSE_wet", "MAE_wet", "CSI_005", "F1_005", "CSI_030",
    "F1_030", "CSI_100", "F1_100", "PeakDepthError",
    "FloodAreaRelativeError", "VolumeRelativeError", "PSNR", "SSIM",
]


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_split_dataset(cfg: dict, split: str) -> WellingtonFixedSRDataset:
    dc = cfg["dataset"]
    return WellingtonFixedSRDataset(
        root=ROOT / dc.get("root", "dataset"),
        split=split,
        lr_res=int(dc.get("lr_res", 10)),
        hr_res=int(dc.get("hr_res", 2)),
        target=dc.get("target", "h_max"),
        geo_mode=dc.get("geo_mode", "all"),
        scenarios=tuple(dc.get("scenarios", ["20a", "100a"])),
    )


def describe_tiles(ds, *, max_tiles: int | None = None, log_every: int = 100) -> dict[int, dict]:
    """Cheap pass: tile metadata (wet fraction, valid fraction, peak truth) with no model forward."""
    n = len(ds) if max_tiles is None else min(len(ds), max_tiles)
    out: dict[int, dict] = {}
    t0 = time.time()
    for i in range(n):
        sample = ds[i]
        hr_np = np.asarray(sample["hr"])[0]
        mask_np = np.asarray(sample["mask"]).astype(bool)
        out[i] = {
            "index": i,
            "scenario": sample["scenario"],
            "iy": int(sample["iy"]),
            "ix": int(sample["ix"]),
            "wet_frac": float(((hr_np > 0.05) & mask_np).sum() / max(mask_np.sum(), 1)),
            "valid_frac": float(mask_np.mean()),
            "peak_truth": float(hr_np[mask_np].max()) if mask_np.any() else 0.0,
        }
        if log_every and (i + 1) % log_every == 0:
            print(f"  describe {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)
    return out


@torch.no_grad()
def metric_for_tile(model, ds, i: int, meta: dict, device) -> dict:
    batch = collate_fixed([ds[i]])
    out = model(
        lr=batch["lr"], lr_valid=batch["lr_valid"],
        static_cont=batch["static_cont"], landuse=batch["landuse"],
        lr_res=batch["lr_res"], hr_res=batch["hr_res"],
    )
    m = compute_flood_metrics(out["depth"], batch["hr"], batch["mask"])
    return {**meta, **{k: float(v) for k, v in m.items()}}


@torch.no_grad()
def scan_split(model, ds, device, *, max_tiles: int | None, log_every: int = 25,
               meta: dict[int, dict] | None = None) -> list[dict]:
    """Per-tile metrics for the whole split (or the first max_tiles)."""
    meta = meta or describe_tiles(ds, max_tiles=max_tiles, log_every=0)
    rows: list[dict] = []
    n = len(ds) if max_tiles is None else min(len(ds), max_tiles)
    t0 = time.time()
    for i in range(n):
        rows.append(metric_for_tile(model, ds, i, meta[i], device))
        if log_every and (i + 1) % log_every == 0:
            print(f"  scan {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)
    return rows


def pick_representatives(rows: list[dict], k: int = 4) -> list[int]:
    """Pick tiles across the inundation-coverage spectrum plus best/worst CSI."""
    if not rows:
        return []
    order = np.argsort([r["wet_frac"] for r in rows])
    picks: list[int] = []
    qs = np.linspace(0.06, 0.94, max(k - 2, 1))
    for q in qs:
        picks.append(int(order[min(int(q * len(order)), len(order) - 1)]))
    csi = np.array([r.get("CSI_005", np.nan) for r in rows], dtype=float)
    ok = np.isfinite(csi)
    if ok.any():
        picks.append(int(np.argmax(np.where(ok, csi, -np.inf))))
        picks.append(int(np.argmin(np.where(ok, csi, np.inf))))
    seen, out = set(), []
    for p in picks:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:k]


def write_table(rows: list[dict], path_csv: Path, path_json: Path) -> None:
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["index", "scenario", "iy", "ix", "wet_frac", "valid_frac", "peak_truth", *SET_NAMES]
    with path_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    path_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def detailed_tile_figure(tp: TilePrediction, *, out_dir: Path, name: str) -> Path:
    fig = plt.figure(figsize=(15.5, 3.4))
    gs = fig.add_gridspec(1, 7, wspace=0.14)
    vmax = depth_vmax(tp.gt, tp.pred, q=99.5)
    coarse_up = np.repeat(np.repeat(tp.coarse, tp.scale, axis=-2), tp.scale, axis=-1)[
        : tp.gt.shape[0], : tp.gt.shape[1]
    ]
    panels = [
        ("Coarse input 10 m", coarse_up),
        ("Bilinear baseline", tp.base),
        ("HydroGeo-SRNO (pred)", tp.pred),
        ("Ground truth 2 m", tp.gt),
    ]
    for j, (title, arr) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        show_depth(ax, arr, tp.valid, vmax=vmax, title=title)

    ax = fig.add_subplot(gs[0, 4])
    err = np.where(tp.valid, np.abs(tp.pred - tp.gt), np.nan)
    lim = float(np.nanpercentile(err, 98)) if np.isfinite(err).any() else 1.0
    im = ax.imshow(err, cmap="inferno_r", vmin=0, vmax=max(lim, 1e-3))
    ax.set_title("|pred - truth|", pad=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    side_colorbar(fig, ax, im, fmt="%.2f")

    ax = fig.add_subplot(gs[0, 5])
    from viz.reconstruct import agreement_map

    ax.imshow(agreement_map(tp.pred, tp.gt, tp.valid), cmap=AGREE_CMAP, vmin=0, vmax=1)
    ax.set_title(f"Wet extent @0.05 m\nCSI={tp.metrics.get('CSI_005', float('nan')):.3f}", pad=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    ax = fig.add_subplot(gs[0, 6])
    d = np.where(tp.valid, tp.residual, np.nan)
    lim2 = max(float(np.nanpercentile(np.abs(d), 98)), 1e-3)
    im = ax.imshow(d, cmap="RdBu_r", vmin=-lim2, vmax=lim2)
    ax.set_title("Learned residual Δz", pad=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    side_colorbar(fig, ax, im, fmt="%.2f")

    sub = (
        f"MAE_wet={tp.metrics.get('MAE_wet', float('nan')):.3f} m   "
        f"RMSE_wet={tp.metrics.get('RMSE_wet', float('nan')):.3f} m   "
        f"CSI@0.30={tp.metrics.get('CSI_030', float('nan')):.3f}   "
        f"CSI@1.00={tp.metrics.get('CSI_100', float('nan')):.3f}   "
        f"vol.err={tp.metrics.get('VolumeRelativeError', float('nan')):.3f}"
    )
    fig.suptitle(f"{tp.label}   |   {sub}", fontsize=11, fontweight="bold", y=1.06)
    fig.text(0.5, -0.10, LEGEND_TEXT, ha="center", fontsize=7.5, color="#555555")
    return save(fig, out_dir / name)


def overview_figure(preds: list[TilePrediction], split: str, tag_dir: Path) -> Path:
    n = len(preds)
    fig = plt.figure(figsize=(15.0, 3.15 * n))
    gs = fig.add_gridspec(n, 5, wspace=0.10, hspace=0.16)
    for i, tp in enumerate(preds):
        vmax = depth_vmax(tp.gt, tp.pred, q=99.5)
        coarse_up = np.repeat(np.repeat(tp.coarse, tp.scale, axis=-2), tp.scale, axis=-1)[
            : tp.gt.shape[0], : tp.gt.shape[1]
        ]
        cols = [("Coarse 10 m", coarse_up), ("Bilinear", tp.base), ("Model", tp.pred), ("Truth", tp.gt)]
        for j, (title, arr) in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            show_depth(ax, arr, tp.valid, vmax=vmax, title=title if i == 0 else "")
            if j == 0:
                ax.set_ylabel(f"{tp.label}\nwet={tp.metrics.get('CSI_005', float('nan')):.2f}", fontsize=7)
        ax = fig.add_subplot(gs[i, 4])
        err = np.where(tp.valid, np.abs(tp.pred - tp.gt), np.nan)
        im = ax.imshow(err, cmap="inferno_r", vmin=0, vmax=max(np.nanpercentile(err, 98), 1e-3))
        if i == 0:
            ax.set_title("|pred - truth|")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        side_colorbar(fig, ax, im, fmt="%.2f")

    fig.suptitle(
        f"HydroGeo-SRNO {tag_dir.name} — representative tiles on {split} split (10 m → 2 m, h_max)",
        fontsize=12.5, fontweight="bold",
    )
    return save(fig, tag_dir / f"tiles_{split}_overview.png")


def comparison_figure(rows: list[dict], split: str, tag_dir: Path, model_metrics: dict | None) -> Path:
    """Model vs B0/B1 baseline metrics + per-tile CSI histogram."""
    b0 = load_json(OUT_DIR.parent / "benchmarks" / f"nearest_{split}.json")
    b1 = load_json(OUT_DIR.parent / "benchmarks" / f"bilinear_{split}.json")
    keys = ["CSI_005", "CSI_030", "CSI_100", "RMSE_wet", "MAE_wet"]
    labels = ["CSI\n0.05 m", "CSI\n0.30 m", "CSI\n1.00 m", "RMSE wet\n(m)", "MAE wet\n(m)"]
    model_avg = model_metrics or {}
    vals = {
        "B0 nearest": [b0.get(k) if b0 else None for k in keys],
        "B1 bilinear": [b1.get(k) if b1 else None for k in keys],
        "HydroGeo-SRNO": [model_avg.get(k) for k in keys],
    }
    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 3, wspace=0.30, width_ratios=[1.7, 1.0, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(keys))
    w = 0.26
    colors = ["#9e9e9e", "#78909c", "#2e7d32"]
    for i, (name, v) in enumerate(vals.items()):
        yy = [np.nan if s is None else s for s in v]
        ax.bar(x + (i - 1) * w, yy, w, label=name, color=colors[i],
               edgecolor="white", linewidth=0.6)
        for xi, yv in zip(x + (i - 1) * w, yy):
            if yv == yv:
                ax.text(xi, yv, f"{yv:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title(f"Aggregate metrics on {split}")
    ax.legend(loc="upper left")
    ax.set_ylim(0, max([v for vs in vals.values() for v in vs if v and v == v] + [0.1]) * 1.35)

    ax2 = fig.add_subplot(gs[0, 1])
    csi = np.array([r.get("CSI_005", np.nan) for r in rows], dtype=float)
    csi = csi[np.isfinite(csi)]
    if csi.size:
        ax2.hist(csi, bins=24, color="#2e7d32", alpha=0.85, edgecolor="white")
        ax2.axvline(csi.mean(), color="#c0392b", ls="--", lw=1.4, label=f"mean {csi.mean():.3f}")
        ax2.axvline(float(np.median(csi)), color="#1565c0", ls=":", lw=1.4, label=f"median {np.median(csi):.3f}")
        ax2.legend()
    ax2.set_title("Per-tile CSI@0.05 distribution")
    ax2.set_xlabel("CSI @ 0.05 m"); ax2.set_ylabel("# tiles")

    ax3 = fig.add_subplot(gs[0, 2])
    csi03 = np.array([r.get("CSI_030", np.nan) for r in rows], dtype=float)
    wetf = np.array([r.get("wet_frac", np.nan) for r in rows], dtype=float)
    ok = np.isfinite(csi) & np.isfinite(wetf)
    if ok.any():
        sc = ax3.scatter(wetf[ok], csi[ok], c=np.array([r.get("peak_truth", 0) for r in rows])[ok],
                         s=22, cmap="viridis", alpha=0.85)
        fig.colorbar(sc, ax=ax3, fraction=0.046, pad=0.02, label="peak truth (m)")
    ax3.set_title("CSI@0.05 vs inundation coverage")
    ax3.set_xlabel("wet fraction of tile"); ax3.set_ylabel("CSI @ 0.05 m")

    fig.suptitle(f"Stage evaluation — {split} ({tag_dir.name})", fontsize=12, fontweight="bold")
    return save(fig, tag_dir / f"metrics_vs_baseline_{split}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0_10m2m_hmax.yaml")
    ap.add_argument("--ckpt", default=None, help="default: outputs/v0_10m2m_hmax/best_csi.pt")
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default=None, help="output subfolder, default ep<epoch>")
    ap.add_argument("--n-representative", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--representatives-only", action="store_true",
                    help="skip the full-split metric scan; only render representative tiles")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    apply_style()

    ckpt_path = Path(args.ckpt) if args.ckpt else OUT_DIR / "best_csi.pt"
    cfg = load_cfg(ROOT / args.config)
    device = torch.device(args.device)

    model = build_model(cfg["model"]["name"], cfg)
    ckpt = load_checkpoint(str(ckpt_path), model, map_location="cpu")
    epoch = int(ckpt.get("epoch", 0))
    tag = args.tag or f"ep{epoch:03d}"
    tag_dir = VIZ_DIR / tag
    tag_dir.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint={ckpt_path} epoch={epoch} tag={tag} device={device}")
    print(f"metrics recorded at save: " + json.dumps(
        {k: round(float(v), 4) for k, v in (ckpt.get("metrics") or {}).items() if isinstance(v, (int, float))}
    ))

    for split in args.splits:
        ds = build_split_dataset(cfg, split)
        print(f"\n[{split}] tiles={len(ds)} — describing tiles (no forward passes)")
        t0 = time.time()
        meta = describe_tiles(ds, max_tiles=args.max_tiles)
        print(f"[{split}] described {len(meta)} tiles in {time.time()-t0:.0f}s")

        rows: list[dict] = []
        if not args.representatives_only:
            print(f"[{split}] scanning per-tile metrics (CPU)")
            t0 = time.time()
            rows = scan_split(model, ds, device, max_tiles=args.max_tiles, meta=meta)
            print(f"[{split}] scan done in {time.time()-t0:.0f}s")
            write_table(rows, tag_dir / f"per_tile_metrics_{split}.csv",
                        tag_dir / f"per_tile_metrics_{split}.json")

        picks = pick_representatives(rows, k=args.n_representative) if rows else None
        if picks is None:
            # rank by inundation coverage only (no model metrics available)
            order = sorted(meta.values(), key=lambda r: r["wet_frac"])
            qs = np.linspace(0.10, 0.90, args.n_representative)
            picks = sorted({order[min(int(q * (len(order) - 1)), len(order) - 1)]["index"] for q in qs})
        print(f"[{split}] representative tiles: {picks}")

        tag_dir.mkdir(parents=True, exist_ok=True)
        tile_dir = tag_dir / f"tiles_{split}"
        preds: list[TilePrediction] = []
        tile_rows: list[dict] = []
        for i in picks:
            batch = collate_fixed([ds[i]])
            tp = predict_tile(model, batch, device=device)
            r = {**meta[i], **{k: float(v) for k, v in tp.metrics.items()}}
            tile_rows.append(r)
            tp.label = f"{r['scenario']} iy={r['iy']} ix={r['ix']} wet={r['wet_frac']:.2f}"
            preds.append(tp)
            fname = f"tile_{split}_{r['iy']:02d}_{r['ix']:02d}_{r['scenario']}.png"
            out = detailed_tile_figure(tp, out_dir=tile_dir, name=fname)
            print(f"  wrote {out}")

        p = overview_figure(preds, split, tag_dir)
        print(f"  wrote {p}")

        if rows:
            avg = {k: float(np.nanmean([r[k] for r in rows])) for k in SET_NAMES if any(k in r for r in rows)}
            avg_basis = f"full_scan_{len(rows)}_tiles"
            p = comparison_figure(rows, split, tag_dir, avg)
            print(f"  wrote {p}")
        else:
            # No scan this run: the only numbers available come from the handful of
            # hand-picked representative tiles. Reporting those as avg_metrics made a
            # later --representatives-only refresh overwrite a real 60-tile scan with a
            # 4-tile average (ep155 val read 0.567 while the true 60-tile mean was 0.515).
            # Label it explicitly and keep the previously scanned average if one exists.
            avg = {k: float(np.nanmean([r[k] for r in tile_rows])) for k in SET_NAMES
                   if any(k in r for r in tile_rows)}
            avg_basis = f"representative_only_{len(tile_rows)}_tiles"
            prev_path = tag_dir / f"summary_{split}.json"
            if prev_path.exists():
                try:
                    prev = json.loads(prev_path.read_text(encoding="utf-8"))
                    if str(prev.get("avg_basis", "")).startswith("full_scan") and prev.get("avg_metrics"):
                        print(f"  keeping previously scanned avg_metrics "
                              f"({prev.get('avg_basis')}) instead of {avg_basis}")
                        avg = prev["avg_metrics"]
                        avg_basis = prev["avg_basis"]
                except (json.JSONDecodeError, OSError):
                    pass

        (tag_dir / f"summary_{split}.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(ckpt_path),
                    "epoch": epoch,
                    "split": split,
                    "n_tiles_scanned": len(rows),
                    "n_tiles_described": len(meta),
                    "representative_tiles": [int(i) for i in picks],
                    "avg_basis": avg_basis,
                    "avg_metrics": avg,
                    "representative_metrics": tile_rows,
                    "ckpt_recorded_metrics": ckpt.get("metrics"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[{split}] avg: " + " ".join(f"{k}={v:.4f}" for k, v in avg.items()
                                            if k in ("CSI_005", "CSI_030", "RMSE_wet", "MAE_wet")))


if __name__ == "__main__":
    main()
