"""Dataset / alignment sanity tests (P0)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.wellington_fixed_sr import WellingtonFixedSRDataset
from dataset.wellington_sr import WellingtonSRDataset


def test_shapes_and_mask():
    ds = WellingtonFixedSRDataset(root=ROOT / "dataset", split="train")
    x = ds[0]
    assert x["lr"].shape == (1, 48, 48)
    assert x["hr"].shape == (1, 240, 240)
    assert x["static_cont"].shape[0] == 14
    assert x["landuse"].shape == (240, 240)
    assert x["mask"].shape == (240, 240)
    assert x["lr_valid"].shape == (1, 48, 48)
    assert torch.isfinite(x["lr"]).all()
    assert torch.isfinite(x["hr"]).all()
    assert x["landuse"].min() >= 0 and x["landuse"].max() <= 7


def test_geographic_split_untouched():
    base = WellingtonSRDataset(root=ROOT / "dataset", split="train")
    iys = {s["patch"]["iy"] for s in base.samples}
    assert max(iys) <= 29
    assert min(iys) >= 0
    val = WellingtonSRDataset(root=ROOT / "dataset", split="val")
    viys = {s["patch"]["iy"] for s in val.samples}
    assert min(viys) >= 31 and max(viys) <= 36


def test_fractional_scale_gate():
    try:
        WellingtonSRDataset(root=ROOT / "dataset", lr_res=5, hr_res=2, split="train")
        raised = False
    except ValueError:
        raised = True
    assert raised
    ds = WellingtonSRDataset(
        root=ROOT / "dataset", lr_res=5, hr_res=2, split="train", allow_fractional_scale=True
    )
    assert abs(ds.scale - 2.5) < 1e-6
    s = ds[0]
    assert s["lr"].shape[-2:] == (96, 96)
    assert s["hr"].shape[-2:] == (240, 240)


if __name__ == "__main__":
    test_shapes_and_mask()
    test_geographic_split_untouched()
    test_fractional_scale_gate()
    print("test_dataset OK")
