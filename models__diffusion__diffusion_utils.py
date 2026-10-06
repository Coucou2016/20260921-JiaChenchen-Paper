"""Diffusion schedule helpers (stub)."""

from __future__ import annotations

import torch


def linear_beta_schedule(timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def pad_240_to_256(x: torch.Tensor) -> torch.Tensor:
    # pad bottom/right
    return torch.nn.functional.pad(x, (0, 16, 0, 16))


def crop_256_to_240(x: torch.Tensor) -> torch.Tensor:
    return x[..., :240, :240]
