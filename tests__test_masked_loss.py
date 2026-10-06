"""Masked flood loss tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from losses.flood_loss import FloodLoss


def test_mask_zeros_out_nodata():
    loss_fn = FloodLoss()
    pred = {"depth": torch.ones(1, 1, 8, 8), "wet_logits": torch.zeros(1, 1, 8, 8)}
    target = torch.ones(1, 1, 8, 8) * 0.2
    mask = torch.zeros(1, 8, 8, dtype=torch.bool)
    mask[0, :4, :4] = True
    # Corrupt nodata region in pred/target — must not affect loss if masked
    pred["depth"] = pred["depth"].clone()
    pred["depth"][:, :, 4:, 4:] = 1e6
    out = loss_fn(pred, target, mask)
    assert torch.isfinite(out["total"])
    assert out["total"].item() < 1e3


def test_no_conservation_term():
    # Ensure FloodLoss module does not reference avg_pool consistency
    import inspect
    src = inspect.getsource(FloodLoss.forward)
    assert "avg_pool" not in src
    assert "conservation" not in src.lower()


if __name__ == "__main__":
    test_mask_zeros_out_nodata()
    test_no_conservation_term()
    print("test_masked_loss OK")
