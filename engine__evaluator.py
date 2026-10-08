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
from models.bilinear_residual_srno import BilinearResidualSRNO
from models.hydrogeo_srno import build_hydrogeo_srno


def build_model(name: str, cfg: dict | None = None) -> torch.nn.Module:
    cfg = cfg or {}
    name = name.lower()
    if name in ("hydrogeo_srno", "m2", "m3"):
        return build_hydrogeo_srno(cfg)
    if name in ("bilinear_residual", "br"):
        return _build_residual(cfg, "bilinear")
    if name in ("nearest_residual", "nr"):
        return _build_residual(cfg, "nearest")
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


def _build_residual(cfg: dict, base_mode: str) -> torch.nn.Module:
    """Shared builder for the Experiment-A residual arms.

    The residual variants use the SAME encoder/operator/head architecture as the
    direct V0 model, so the only difference is the parameterisation of the output.
    """
    m = cfg.get("model", cfg)
    hydro = m.get("hydro_encoder", {})
    geo = m.get("geo_encoder", {})
    op = m.get("operator", {})
    return BilinearResidualSRNO(
        base_mode=base_mode,
        residual_space=m.get("residual_space", "depth"),
        hydro_width=hydro.get("width", 64),
        hydro_blocks=hydro.get("blocks", 8),
        hydro_in_channels=hydro.get("in_channels", 2),
        geo_width=geo.get("width", 48),
        geo_in_cont=geo.get("in_cont", 14),
        landuse_embedding=geo.get("landuse_embedding", 8),
        use_landuse=geo.get("use_landuse", True),
        operator_width=op.get("width", 192),
        heads=op.get("heads", 8),
        operator_layers=op.get("layers", 2),
        scale_dim=m.get("scale_dim", 16),
        depth_ref=m.get("depth_ref", 0.10),
        predict_residual=True,
        predict_wet=m.get("predict_wet", True),
        mask_aware_base=m.get("mask_aware_base", True),
    )
