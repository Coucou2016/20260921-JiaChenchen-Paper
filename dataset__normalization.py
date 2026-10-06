"""Train-split normalization helpers for continuous static channels.

Landuse (channel 13) is never z-scored. Stats come only from
index/norm_stats_train.json (usable training tiles).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

# Indices into the 15-channel static stack (DATASET.md).
CONTINUOUS_CHANNELS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14)
LANDUSE_CHANNEL = 13

STATIC_NAMES = [
    "DEM", "Slope", "Aspect_sin", "Aspect_cos", "Curv_plan", "Curv_profile",
    "Tpi", "Twi", "Dist_building", "Dist_road", "Dist_water",
    "Infiltration", "Manning", "Landuse", "Building_binary",
]


def load_norm_stats(root: str | Path | None = None) -> dict:
    root = Path(root) if root is not None else HERE
    path = root / "index" / "norm_stats_train.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _static_key(name: str, res: int) -> str:
    return f"static_{res}m_{name}"


class StaticNormalizer:
    """Z-score continuous / binary static channels; leave landuse as integer codes."""

    def __init__(
        self,
        stats: dict | None = None,
        hr_res: int = 2,
        root: str | Path | None = None,
        continuous_channels: Sequence[int] = CONTINUOUS_CHANNELS,
    ) -> None:
        self.stats = stats if stats is not None else load_norm_stats(root)
        self.hr_res = hr_res
        self.continuous_channels = tuple(continuous_channels)
        # Prefer exact resolution stats; fall back to 2 m then 10 m (same physical units).
        stat_res = hr_res
        probe = _static_key("DEM", hr_res)
        if probe not in self.stats:
            for cand in (2, 10, 5, 20, 30):
                if _static_key("DEM", cand) in self.stats:
                    stat_res = cand
                    break
            else:
                raise KeyError(f"no static norm stats for hr_res={hr_res}")
        self.stat_res = stat_res
        means, stds = [], []
        for i in self.continuous_channels:
            key = _static_key(STATIC_NAMES[i], stat_res)
            entry = self.stats[key]
            means.append(float(entry["mean"]))
            std = float(entry["std"])
            stds.append(std if std > 1e-8 else 1.0)
        self.mean = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1)

    def split_static(self, static: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (static_cont [C,H,W], landuse [H,W] long)."""
        if static.ndim == 4:
            # Batched
            cont = static[:, list(self.continuous_channels)]
            landuse = static[:, LANDUSE_CHANNEL]
            cont = (cont - self.mean.to(cont.device)) / self.std.to(cont.device)
            cont = torch.nan_to_num(cont, nan=0.0)
            landuse = torch.nan_to_num(landuse, nan=0.0).long().clamp(0, 7)
            return cont, landuse
        cont = static[list(self.continuous_channels)]
        landuse = static[LANDUSE_CHANNEL]
        cont = (cont - self.mean) / self.std
        cont = torch.nan_to_num(cont, nan=0.0)
        landuse = torch.nan_to_num(landuse, nan=0.0).long().clamp(0, 7)
        return cont, landuse


def depth_encode(h: torch.Tensor, h0: float = 0.1) -> torch.Tensor:
    return torch.log1p(h / h0)


def depth_decode(z: torch.Tensor, h0: float = 0.1) -> torch.Tensor:
    return h0 * torch.expm1(z)


def select_static_channels(
    static_cont: torch.Tensor,
    landuse: torch.Tensor,
    geo_mode: str = "all",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Ablation channel selection on already-normalized continuous stack.

    continuous order matches CONTINUOUS_CHANNELS:
      0 DEM, 1 Slope, 2–3 Aspect, 4–5 Curv, 6 TPI, 7 TWI,
      8 Dist_building, 9 Dist_road, 10 Dist_water,
      11 Infiltration, 12 Manning, 13 Building_binary
    """
    if geo_mode in ("none", "lr_only"):
        return static_cont[:, :0] if static_cont.ndim == 4 else static_cont[:0], None
    if geo_mode == "dem":
        # DEM only
        idx = [0]
        lu = None
    elif geo_mode == "dem_building":
        idx = [0, 13]
        lu = None
    elif geo_mode == "topography":
        idx = [0, 1, 2, 3, 4, 5, 6, 7]
        lu = None
    elif geo_mode == "all":
        idx = list(range(static_cont.shape[-3] if static_cont.ndim >= 3 else len(CONTINUOUS_CHANNELS)))
        # Prefer full continuous count from tensor
        if static_cont.ndim == 4:
            idx = list(range(static_cont.shape[1]))
        else:
            idx = list(range(static_cont.shape[0]))
        lu = landuse
    else:
        raise ValueError(f"unknown geo_mode={geo_mode}")

    if static_cont.ndim == 4:
        cont = static_cont[:, idx]
    else:
        cont = static_cont[idx]
    return cont, lu
