#!/usr/bin/env python
"""Domain-level spatial context: patch layout, splits, and error geography.

Produces:
  visualizations/02_domain_splits.png       patch grid coloured by split + coverage
  visualizations/02_error_geography.png     per-tile metric maps (needs a scan)
                                            stitched per-tile metric heatmaps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from viz.common import OUT_DIR, VIZ_DIR, apply_style, save, stamp

INDEX = ROOT / "dataset" / "index" / "patches_480m.json"

SPLIT_COLORS = {
    "train": "#2e7d32",
    "val": "#1565c0",
    "buffer": "#f9a825",
    "test": "#c62828",
}


def load_index() -> dict:
    return json.loads(INDEX.read_text(encoding="utf-8"))


def to_grid(patches: list[dict], key) -> np.ndarray:
    ny = max(p["iy"] for p in patches) + 1
    nx = max(p["ix"] for p in patches) + 1
    grid = np.full((ny, nx), np.nan, dtype=float)
    for p in patches:
        v = key(p)
        grid[p["iy"], p["ix"]] = np.nan if v is None else float(v)
    return grid


def domain_extent(meta: dict) -> list[float]:
    west = meta["origin_west"]
    north = meta["origin_north"]
    w = meta["width_m"]
    h = meta["height_m"]
    return [west, west + w, north - h, north]  # [xmin,xmax,ymin,ymax] in metres


def figure_splits(meta: dict) -> Path:
    patches = meta["patches"]
    ny, nx = meta["n_patch_y"], meta["n_patch_x"]

    split_id = {"train": 0, "buffer": 1, "val": 2, "test": 3}
    split_grid = np.full((ny, nx), np.nan)
    usable_grid = np.full((ny, nx), np.nan)
    wet20 = np.full((ny, nx), np.nan)
    wet100 = np.full((ny, nx), np.nan)
    for p in patches:
        split_grid[p["iy"], p["ix"]] = split_id[p["split"]]
        usable_grid[p["iy"], p["ix"]] = 1.0 if p["usable"] else 0.0
        wet20[p["iy"], p["ix"]] = p["wet_frac"].get("20a", np.nan)
        wet100[p["iy"], p["ix"]] = p["wet_frac"].get("100a", np.nan)

    fig = plt.figure(figsize=(15.0, 9.0))
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22, width_ratios=[1, 1, 1.15])

    cmap_split = ListedColormap([SPLIT_COLORS["train"], SPLIT_COLORS["buffer"],
                                 SPLIT_COLORS["val"], SPLIT_COLORS["test"]])
    cmap_split.set_bad("#f2f2f2")

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(split_grid, cmap=cmap_split, vmin=-0.5, vmax=3.5, interpolation="nearest")
    ax.set_title("Geographic split (north→south bands)")
    ax.set_xlabel("patch ix (west→east)"); ax.set_ylabel("patch iy (north→south)")
    ax.legend(handles=[Patch(facecolor=c, label=s) for s, c in SPLIT_COLORS.items()],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    ax.grid(None, alpha=0.15)

    ax = fig.add_subplot(gs[0, 1])
    cmap_use = ListedColormap(["#e8e8e8", "#4db6ac"])
    ax.imshow(usable_grid, cmap=cmap_use, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"Usable tiles (valid≥0.70, no buffer)\n"
                 f"{meta['counts_usable']}")
    ax.set_xlabel("patch ix"); ax.set_ylabel("patch iy")
    ax.legend(handles=[Patch(facecolor="#4db6ac", label="usable"),
                       Patch(facecolor="#e8e8e8", label="excluded")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.grid(None, alpha=0.15)

    for j, (wet, name) in enumerate([(wet20, "20-year"), (wet100, "100-year")]):
        ax = fig.add_subplot(gs[1, j])
        im = ax.imshow(np.clip(wet, 0, 1), cmap="YlGnBu", vmin=0, vmax=max(0.3, np.nanpercentile(wet, 98)))
        ax.set_title(f"Flood wet fraction — {name} event (usable tiles)")
        ax.set_xlabel("patch ix"); ax.set_ylabel("patch iy")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="wet fraction (>0.05 m)")
        ax.grid(None, alpha=0.15)

    ax = fig.add_subplot(gs[1, 2])
    u = usable_grid.astype(bool)
    a = wet20[u & np.isfinite(wet20)]
    b = wet100[u & np.isfinite(wet100)]
    ax.hist([a, b], bins=26, color=["#1565c0", "#c62828"], label=["20a", "100a"], alpha=0.85)
    ax.set_title("Inundation coverage distribution (usable tiles)")
    ax.set_xlabel("wet fraction of tile"); ax.set_ylabel("# tiles")
    ax.legend()
    ax.text(0.98, 0.95, f"20a: {len(a)} tiles, mean {a.mean():.3f}\n100a: {len(b)} tiles, mean {b.mean():.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)

    ext = domain_extent(meta)
    fig.suptitle(
        "Wellington flood-SR dataset — patch domain 480 m, "
        f"grid {nx}×{ny}, extent {meta['width_m']/1000:.1f}×{meta['height_m']/1000:.1f} km "
        f"(EPSG:2193; W {ext[0]:.0f}, N {ext[3]:.0f})",
        fontsize=12.5, fontweight="bold",
    )
    stamp(fig, f"split_rule: {meta['split_rule']}")
    return save(fig, VIZ_DIR / "02_domain_splits.png")


def find_tables(tag: str | None) -> dict[str, Path]:
    base = VIZ_DIR / tag if tag else OUT_DIR / "visualizations"
    cands = sorted(VIZ_DIR.glob("ep*/per_tile_metrics_*.json"))
    if tag:
        cands = sorted((VIZ_DIR / tag).glob("per_tile_metrics_*.json")) or cands
    out: dict[str, Path] = {}
    for p in cands:
        split = p.stem.replace("per_tile_metrics_", "")
        if split not in out:
            out[split] = p
    return out


def figure_error_geography(meta: dict, tables: dict[str, Path]) -> Path | None:
    if not tables:
        print("no per-tile metric tables found; skipping error geography")
        return None
    splits = sorted(tables)
    fig = plt.figure(figsize=(13.5, 4.6 * len(splits)))
    gs = fig.add_gridspec(len(splits), 3, hspace=0.30, wspace=0.22)
    for r, split in enumerate(splits):
        rows = json.loads(tables[split].read_text(encoding="utf-8"))
        for c, key in enumerate(["CSI_005", "CSI_030", "MAE_wet"]):
            ax = fig.add_subplot(gs[r, c])
            grid = np.full((meta["n_patch_y"], meta["n_patch_x"]), np.nan)
            for row in rows:
                v = row.get(key)
                if v is not None and np.isfinite(float(v)):
                    grid[int(row["iy"]), int(row["ix"])] = float(v)
            cmap = "RdYlGn" if key.startswith("CSI") else "RdYlGn_r"
            finite = grid[np.isfinite(grid)]
            if finite.size == 0:
                ax.axis("off")
                continue
            vmin, vmax = float(np.min(finite)), float(np.max(finite))
            im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(f"{split} — {key}  (mean {finite.mean():.4f})")
            ax.set_xlabel("ix"); ax.set_ylabel("iy")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            ax.grid(None, alpha=0.15)
    fig.suptitle("Per-tile metric geography (blank = tile not scanned / unusable)",
                 fontsize=12.5, fontweight="bold")
    stamp(fig, f"source: {', '.join(str(p.name) for p in tables.values())}")
    return save(fig, VIZ_DIR / "02_error_geography.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    apply_style()
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_index()
    p = figure_splits(meta)
    print(f"wrote {p}")
    tables = find_tables(args.tag)
    p = figure_error_geography(meta, tables)
    if p:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
