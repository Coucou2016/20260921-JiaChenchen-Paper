"""EDSR-style single-image SR baseline (upsample LR then residual CNN)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.normalization import depth_decode, depth_encode
from models.hydro_encoder import ResidualBlock


class EDSRBaseline(nn.Module):
    def __init__(self, width: int = 64, n_blocks: int = 8, depth_ref: float = 0.1) -> None:
        super().__init__()
        self.depth_ref = depth_ref
        self.head = nn.Conv2d(1, width, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(width) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(width, 1, 3, padding=1)
        self.wet = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, lr: torch.Tensor, static_cont: torch.Tensor | None = None, landuse=None, **kwargs):
        if static_cont is not None and static_cont.numel():
            hr_size = (static_cont.shape[-2], static_cont.shape[-1])
        elif landuse is not None:
            hr_size = (landuse.shape[-2], landuse.shape[-1])
        else:
            raise ValueError("need HR size")
        base = F.interpolate(lr[:, :1], size=hr_size, mode="bilinear", align_corners=False)
        feat = self.head(base)
        feat = feat + self.body(feat)
        delta_z = self.tail(feat)
        wet_logits = self.wet(feat)
        z = depth_encode(base, self.depth_ref) + delta_z
        h = torch.clamp(depth_decode(z, self.depth_ref), min=0.0)
        return {"depth": h, "wet_logits": wet_logits, "residual": delta_z, "base": base}
