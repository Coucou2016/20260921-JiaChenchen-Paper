"""Model forward shape tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.hydrogeo_srno import HydroGeoSRNO


def test_hydrogeo_forward():
    model = HydroGeoSRNO(
        hydro_width=32,
        hydro_blocks=2,
        geo_width=16,
        landuse_embedding=4,
        operator_width=64,
        heads=4,
        operator_layers=1,
    )
    b = 1
    lr = torch.rand(b, 1, 48, 48)
    lr_valid = torch.ones(b, 1, 48, 48)
    static = torch.randn(b, 14, 240, 240)
    landuse = torch.randint(0, 8, (b, 240, 240))
    out = model(
        lr=lr,
        lr_valid=lr_valid,
        static_cont=static,
        landuse=landuse,
        lr_res=torch.tensor([10.0]),
        hr_res=torch.tensor([2.0]),
    )
    assert out["depth"].shape == (1, 1, 240, 240)
    assert out["wet_logits"].shape == (1, 1, 240, 240)
    assert out["residual"].shape == (1, 1, 240, 240)
    assert (out["depth"] >= 0).all()


if __name__ == "__main__":
    test_hydrogeo_forward()
    print("test_model_shapes OK")
