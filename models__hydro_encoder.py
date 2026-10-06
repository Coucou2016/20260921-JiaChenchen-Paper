"""HydroEncoder: EDSR-style residual stack for coarse flood (+ validity) inputs."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class HydroEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        width: int = 64,
        n_blocks: int = 8,
    ) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels, width, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(width) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.head(x)
        x1 = self.tail(self.body(x0))
        return x0 + x1
