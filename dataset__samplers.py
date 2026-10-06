"""Optional samplers. Default DataLoader shuffle is fine; split is geographic."""

from __future__ import annotations

import torch
from torch.utils.data import Sampler, WeightedRandomSampler


def wet_fraction_weights(dataset, floor: float = 0.05) -> torch.Tensor:
    """Upweight wetter tiles using index wet_frac (100a) when available."""
    weights = []
    for s in dataset.base.samples:
        p = s["patch"]
        wf = float(p.get("wet_frac", {}).get("100a", floor))
        weights.append(max(wf, floor))
    return torch.tensor(weights, dtype=torch.double)


def make_wet_sampler(dataset, num_samples: int | None = None) -> Sampler:
    w = wet_fraction_weights(dataset)
    n = num_samples if num_samples is not None else len(dataset)
    return WeightedRandomSampler(w, num_samples=n, replacement=True)
