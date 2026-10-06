"""Tile-level reconstruction figures: coarse input → baselines → model → truth.

Runs on CPU only so it can execute while training owns the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from metrics.flood_metrics import compute_flood_metrics
from viz.common import (
    AGREE_CMAP,
    FLOOD_CMAP,
    TERRAIN_CMAP,
    add_colorbar,
    depth_vmax,
    show_depth,
)


@dataclass
class TilePrediction:
    """Everything needed to draw one tile comparison."""

    tile: dict[str, Any]
    gt: np.ndarray                 # [H,W] high-res truth (m)
    pred: np.ndarray               # [H,W] model depth (m)
    base: np.ndarray               # [H,W] bilinear base (m)
    coarse: np.ndarray             # [h,w] coarse LR depth (m)
    dem: np.ndarray                # [H,W] terrain elevation (m)
    residual: np.ndarray           # [H,W] log-space residual Δz
    wet_prob: np.ndarray | None    # [H,W] sigmoid of wet head
    valid: np.ndarray              # [H,W] bool
    metrics: dict[str, float] = field(default_factory=dict)
    label: str = ""

    @property
    def scale(self) -> int:
        return max(1, int(round(self.gt.shape[-1] / max(self.coarse.shape[-1], 1))))


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


@torch.no_grad()
def predict_tile(
    model: torch.nn.Module,
    sample: dict[str, Any],
    *,
    device: torch.device | None = None,
) -> TilePrediction:
    """Run the model on one collated sample (batch size 1) and unpack outputs."""
    device = device or torch.device("cpu")
    model.eval()
    model.to(device)

    def mv(v):
        return v.to(device) if torch.is_tensor(v) else v

    out = model(
        lr=mv(sample["lr"]),
        lr_valid=mv(sample["lr_valid"]),
        static_cont=mv(sample["static_cont"]),
        landuse=mv(sample["landuse"]),
        lr_res=mv(sample["lr_res"]),
        hr_res=mv(sample["hr_res"]),
    )

    gt = _to_np(sample["hr"])[0, 0]
    pred = _to_np(out["depth"])[0, 0]
    base = _to_np(out["base"])[0, 0]
    coarse = _to_np(sample["lr"])[0, 0]
    valid = _to_np(sample["mask"])[0].astype(bool)
    residual = _to_np(out["residual"])[0, 0]
    wet_prob = None
    if out.get("wet_logits") is not None:
        wet_prob = 1.0 / (1.0 + np.exp(-_to_np(out["wet_logits"])[0, 0]))

    # static_cont is z-scored per channel; channel 0 is DEM -> un-standardise for context
    from dataset.normalization import StaticNormalizer

    norm = StaticNormalizer(hr_res=int(sample["hr_res"][0].item()))
    raw_dem = _to_np(sample["static_cont"])[0, 0] * float(norm.std[0]) + float(norm.mean[0])

    tx = mv(sample["lr_res"]); hx = mv(sample["hr_res"])
    assert tx is not None and hx is not None
    metrics = compute_flood_metrics(mv(out["depth"]), mv(sample["hr"]), mv(sample["mask"]))
    label = (
        f"{sample['scenario'][0]} · iy={int(sample['iy'][0])} ix={int(sample['ix'][0])}"
        f" · 10 m → 2 m"
    )

    return TilePrediction(
        tile={k: (int(v[0]) if torch.is_tensor(v) and v.numel() == 1 else v) for k, v in sample.items()
              if k in ("iy", "ix", "scenario")},
        gt=gt, pred=pred, base=base, coarse=coarse, dem=raw_dem,
        residual=residual, wet_prob=wet_prob, valid=valid,
        metrics={k: float(v) for k, v in metrics.items()}, label=label,
    )


def agreement_map(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, thr: float = 0.05) -> np.ndarray:
    """0 = TN, 1 = FP, 2 = FN, 3 = TP (encoded on a 0..1 scale for AGREE_CMAP)."""
    pb = pred > thr
    tb = gt > thr
    code = np.full(pred.shape, np.nan, dtype=np.float32)
    code[valid & ~pb & ~tb] = 0.0
    code[valid & pb & ~tb] = 1 / 3
    code[valid & ~pb & tb] = 2 / 3
    code[valid & pb & tb] = 1.0
    return code


def draw_tile_row(fig, gs_row, tp: TilePrediction, *, show_residual: bool = True) -> None:
    """Draw one tile as a row: coarse, bilinear, model, truth, error, agreement, DEM."""
    ncol = 7 if show_residual else 6
    vmax = depth_vmax(tp.gt, tp.pred, tp.base, q=99.5)
    coarse_up = np.repeat(np.repeat(tp.coarse, tp.scale, axis=-2), tp.scale, axis=-1)[
        : tp.gt.shape[0], : tp.gt.shape[1]
    ]
    views = [
        ("Coarse input (10 m)", coarse_up),
        ("Bilinear baseline (2 m)", tp.base),
        ("HydroGeo-SRNO (2 m)", tp.pred),
        ("Ground truth (2 m)", tp.gt),
    ]
    axes = []
    for j, (title, arr) in enumerate(views):
        ax = fig.add_subplot(gs_row[j])
        show_depth(ax, arr, tp.valid, vmax=vmax, title=title)
        axes.append(ax)

    # absolute error
    ax = fig.add_subplot(gs_row[4])
    err = np.where(tp.valid, np.abs(tp.pred - tp.gt), np.nan)
    im = ax.imshow(err, cmap="inferno_r", vmin=0, vmax=min(vmax, np.nanpercentile(err, 99) if np.isfinite(err).any() else 1.0))
    ax.set_title(f"|Error|  MAE_all={tp.metrics.get('RMSE_all', float('nan')):.3f} m", pad=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=6)

    # agreement at 0.05 m
    ax = fig.add_subplot(gs_row[5])
    ax.imshow(agreement_map(tp.pred, tp.gt, tp.valid), cmap=AGREE_CMAP, vmin=0, vmax=1, interpolation="nearest")
    csi = tp.metrics.get("CSI_005", float("nan"))
    ax.set_title(f"Wet extent vs truth @0.05 m\nCSI={csi:.3f}  TP/FP/FN", pad=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    if show_residual:
        ax = fig.add_subplot(gs_row[6])
        r = np.where(tp.valid, tp.residual, np.nan)
        lim = float(np.nanpercentile(np.abs(r), 99)) if np.isfinite(r).any() else 1.0
        lim = max(lim, 1e-3)
        im = ax.imshow(r, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title("Learned residual Δz (log-depth)", pad=3)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6)


LEGEND_TEXT = (
    "agreement: ■ green = correct wet (TP)   ■ red = false wet (FP)   "
    "■ amber = missed wet (FN)   ■ grey = correct dry"
)
