"""DEM nesting / physical extent alignment checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.wellington_sr import WellingtonSRDataset


def test_patch_extent_matches():
    ds10 = WellingtonSRDataset(root=ROOT / "dataset", lr_res=10, hr_res=2, split="train")
    s = ds10[0]
    # 480 m / res
    assert s["lr"].shape[-1] == 48
    assert s["hr"].shape[-1] == 240
    assert s["static"].shape[0] == 15


def test_mask_excludes_nan():
    ds = WellingtonSRDataset(root=ROOT / "dataset", split="train")
    s = ds[0]
    # where mask==0, hr or dem was nan in raw (now nan_to_num only in fixed wrapper)
    assert s["mask"].dtype == np.float32
    assert s["mask"].min() >= 0 and s["mask"].max() <= 1


if __name__ == "__main__":
    test_patch_extent_matches()
    test_mask_excludes_nan()
    print("test_alignment OK")
