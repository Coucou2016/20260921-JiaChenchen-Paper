"""Residual latent diffusion stub (plan §二十二–二十四). Train after V0."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualLDM(nn.Module):
    """
    Learns p(R_z | H_det, G) in latent space.
    Condition channels (12): noisy residual(4) + det flood(4) + geo(4).
    """

    def __init__(self, in_channel: int = 12, out_channel: int = 4, base: int = 64) -> None:
        super().__init__()
        self.unet = nn.Sequential(
            nn.Conv2d(in_channel, base, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base, out_channel, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        return self.unet(x)
