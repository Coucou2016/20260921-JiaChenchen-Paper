"""RCAN-style channel attention residual baseline (compact)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.normalization import depth_decode, depth_encode


class CALayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.avg(x))


class RCAB(nn.Module):
    def __init__(self, channel: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channel, channel, 3, padding=1),
            CALayer(channel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class RCANBaseline(nn.Module):
    def __init__(self, width: int = 64, n_blocks: int = 8, depth_ref: float = 0.1) -> None:
        super().__init__()
        self.depth_ref = depth_ref
        self.head = nn.Conv2d(1, width, 3, padding=1)
        self.body = nn.Sequential(*[RCAB(width) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(width, 1, 3, padding=1)
        self.wet = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, lr, static_cont=None, landuse=None, **kwargs):
        hr_size = (static_cont.shape[-2], static_cont.shape[-1]) if static_cont is not None and static_cont.numel() else (landuse.shape[-2], landuse.shape[-1])
        base = F.interpolate(lr[:, :1], size=hr_size, mode="bilinear", align_corners=False)
        feat = self.body(self.head(base))
        delta_z = self.tail(feat)
        h = torch.clamp(depth_decode(depth_encode(base, self.depth_ref) + delta_z, self.depth_ref), min=0.0)
        return {"depth": h, "wet_logits": self.wet(feat), "residual": delta_z, "base": base}
