"""Lightweight Residual SwinUNet-style baseline (flood + DEM fusion).

Full Swin Transformer is heavy on CPU; this is a compact windowed residual UNet
stand-in for B4 comparisons until a GPU Swin port is warranted.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.normalization import depth_decode, depth_encode
from models.baselines.resunet import ResUNet


class ResidualSwinUNetLite(ResUNet):
    """Alias with DEM-focused static input (channel 0 only) for ablation parity."""

    def __init__(self, depth_ref: float = 0.1, base: int = 32) -> None:
        super().__init__(static_channels=1, base=base, depth_ref=depth_ref)

    def forward(self, lr, lr_valid=None, static_cont=None, landuse=None, **kwargs):
        dem = static_cont[:, :1] if static_cont is not None else None
        return super().forward(lr=lr, lr_valid=lr_valid, static_cont=dem, landuse=landuse, **kwargs)
