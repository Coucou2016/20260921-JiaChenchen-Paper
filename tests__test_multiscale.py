"""Multiscale / fractional loader tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.wellington_fixed_sr import WellingtonFixedSRDataset
from dataset.wellington_multiscale import WellingtonMultiscaleDataset


def test_multiscale_len():
    ds = WellingtonMultiscaleDataset(
        root=ROOT / "dataset",
        split="train",
        pairs=((20, 10), (10, 2)),
    )
    assert len(ds) == 2 * 550  # two pairs × train h_max samples


def test_fractional_fixed():
    ds = WellingtonFixedSRDataset(
        root=ROOT / "dataset",
        split="train",
        lr_res=5,
        hr_res=2,
        allow_fractional_scale=True,
    )
    x = ds[0]
    assert x["lr"].shape[-2:] == (96, 96)
    assert abs(x["scale"] - 2.5) < 1e-6


if __name__ == "__main__":
    test_multiscale_len()
    test_fractional_fixed()
    print("test_multiscale OK")
