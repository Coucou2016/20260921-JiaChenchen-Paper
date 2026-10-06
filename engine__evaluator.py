"""Evaluation helpers and model factory."""

from __future__ import annotations

from typing import Any

import torch

from models.baselines.bilinear import BilinearBaseline, NearestBaseline
from models.baselines.edsr import EDSRBaseline
from models.baselines.rcan import RCANBaseline
from models.baselines.resunet import ResUNet
from models.baselines.rswinunet import ResidualSwinUNetLite
from models.baselines.srno_single import SRNOSingle
from models.hydrogeo_srno import build_hydrogeo_srno


def build_model(name: str, cfg: dict | None = None) -> torch.nn.Module:
    cfg = cfg or {}
    name = name.lower()
    if name in ("hydrogeo_srno", "m2", "m3"):
        return build_hydrogeo_srno(cfg)
    if name in ("bilinear", "b1"):
        return BilinearBaseline()
    if name in ("nearest", "b0"):
        return NearestBaseline()
    if name in ("resunet", "b2"):
        return ResUNet(static_channels=14, depth_ref=cfg.get("model", {}).get("depth_ref", 0.1))
    if name in ("edsr", "b3"):
        return EDSRBaseline(depth_ref=cfg.get("model", {}).get("depth_ref", 0.1))
    if name in ("rswinunet", "b4"):
        return ResidualSwinUNetLite(depth_ref=cfg.get("model", {}).get("depth_ref", 0.1))
    if name in ("rcan",):
        return RCANBaseline(depth_ref=cfg.get("model", {}).get("depth_ref", 0.1))
    if name in ("srno_single", "b5", "a0"):
        return SRNOSingle(use_dem=False, depth_ref=cfg.get("model", {}).get("depth_ref", 0.1))
    if name in ("srno_dem", "m1", "a1"):
        return SRNOSingle(use_dem=True, depth_ref=cfg.get("model", {}).get("depth_ref", 0.1))
    raise ValueError(f"unknown model name: {name}")
