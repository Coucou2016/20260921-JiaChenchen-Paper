"""Bilinear-residual cross-resolution learner (Experiment A).

The user's idea: take a plain interpolation of the coarse input as the base,
learn the residual ``truth - base`` with the cross-resolution model, then add the
predicted residual back onto that base::

    base   = interp(lr_depth)          # bilinear (default) or nearest
    h_pred = base + delta              # delta is what the network predicts

This is deliberately NOT the latent-diffusion residual model in
``models/diffusion`` and NOT ``scripts/train_residual_ldm.py``. It reuses the
HydroGeo-SRNO encoder + Galerkin operator as the delta predictor.

Relationship to the frozen V0 model (important for honesty)
-----------------------------------------------------------
V0 is *already* a residual learner, but in **log1p depth space** and only on the
mask-aware bilinear base. Writing ``z = log1p(h/h0)``, V0 computes
``z_pred = z_base(bilinear) + delta_z``. That log-space residual is exactly what
the diagnostic blamed for shrinking deep water: over deep pixels the frozen net's
regression slope on the 10 m input is 0.39 against 0.72 for the bilinear base,
so it does not amplify what the interpolation already carries.

Two residual spaces are therefore supported here:

``residual_space="log"``
    ``z_pred = z_base + head(features)``. Algebraically identical to V0, so a
    fresh run of this arm is a like-for-like retrain of V0 and answers only the
    "is the residual target cheaper to fit" question.

``residual_space="depth"`` (the arm the experiment actually reports)
    ``h_pred = clamp(base + head(features), min=0)``. The network now predicts a
    correction in **metres**, so the interpolation's deep signal is preserved by
    construction and the head only has to recover the slope shrinkage and the
    systematic deep bias. This is the genuinely different parameterisation.
"""

from __future__ import annotations

import torch

from dataset.normalization import depth_decode, depth_encode
from models.hydrogeo_srno import HydroGeoSRNO


class BilinearResidualSRNO(HydroGeoSRNO):
    """HydroGeo-SRNO with an explicit interpolation base (bilinear or nearest)."""

    def __init__(self, base_mode: str = "bilinear", residual_space: str = "depth",
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if base_mode not in ("bilinear", "nearest"):
            raise ValueError(f"base_mode must be bilinear|nearest, got {base_mode}")
        if residual_space not in ("log", "depth"):
            raise ValueError(f"residual_space must be log|depth, got {residual_space}")
        self.base_mode = base_mode
        self.residual_space = residual_space

    def forward(
        self,
        lr: torch.Tensor,
        lr_valid: torch.Tensor,
        static_cont: torch.Tensor,
        landuse: torch.Tensor,
        lr_res: torch.Tensor,
        hr_res: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        feat, hr_h, hr_w = self.encode_features(
            lr, lr_valid, static_cont, landuse, lr_res, hr_res)

        base_h = self.compute_base(lr, lr_valid, hr_h, hr_w, mode=self.base_mode)
        raw = self.residual_head(feat)

        if self.residual_space == "log":
            # Exactly the V0 parameterisation; `raw` is the log-space correction.
            z_pred = depth_encode(base_h, self.depth_ref) + raw
            h_pred = depth_decode(z_pred, self.depth_ref)
            residual = raw
        else:
            # Depth-space residual: the base carries the coarse deep signal and the
            # head only adds a metre-scale correction to it.
            residual = raw
            h_pred = base_h + residual

        h_pred = torch.clamp(h_pred, min=0.0)

        wet_logit = self.wet_head(feat) if self.wet_head is not None else None
        out: dict[str, torch.Tensor] = {
            "depth": h_pred,
            "residual": residual,
            "base": base_h,
        }
        if wet_logit is not None:
            out["wet_logits"] = wet_logit
        return out
