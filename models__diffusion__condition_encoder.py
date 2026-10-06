"""Geography condition encoder → 4-channel latent (Flood-LDM compatible). Stub."""

from __future__ import annotations

import torch
import torch.nn as nn


class GeoConditionEncoder(nn.Module):
    def __init__(self, in_ch: int = 15, latent_ch: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, latent_ch, 1),
        )

    def forward(self, static: torch.Tensor) -> torch.Tensor:
        return self.net(static)
