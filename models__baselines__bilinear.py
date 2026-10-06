"""Classical upsampling baselines (no trainable params)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleBaseline(nn.Module):
    def __init__(self, mode: str = "bilinear") -> None:
        super().__init__()
        self.mode = mode

    def forward(
        self,
        lr: torch.Tensor,
        lr_valid: torch.Tensor | None = None,
        static_cont: torch.Tensor | None = None,
        landuse: torch.Tensor | None = None,
        lr_res: torch.Tensor | None = None,
        hr_res: torch.Tensor | None = None,
        hr_size: tuple[int, int] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if hr_size is None:
            if static_cont is not None and static_cont.numel() > 0:
                hr_size = (static_cont.shape[-2], static_cont.shape[-1])
            elif landuse is not None:
                hr_size = (landuse.shape[-2], landuse.shape[-1])
            else:
                raise ValueError("hr_size or static/landuse required")
        depth = lr[:, :1]
        if self.mode == "nearest":
            up = F.interpolate(depth, size=hr_size, mode="nearest")
        else:
            up = F.interpolate(depth, size=hr_size, mode="bilinear", align_corners=False)
        up = torch.clamp(up, min=0.0)
        # Pseudo wet logits from depth
        wet_logits = (up - 0.05) * 10.0
        return {"depth": up, "wet_logits": wet_logits, "residual": torch.zeros_like(up), "base": up}


class NearestBaseline(UpsampleBaseline):
    def __init__(self) -> None:
        super().__init__(mode="nearest")


class BilinearBaseline(UpsampleBaseline):
    def __init__(self) -> None:
        super().__init__(mode="bilinear")
