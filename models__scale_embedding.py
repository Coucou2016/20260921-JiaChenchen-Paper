"""Resolution conditioning for arbitrary-scale operator mapping."""

from __future__ import annotations

import torch
import torch.nn as nn


class ScaleEmbedding(nn.Module):
    def __init__(self, out_dim: int = 16) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, out_dim),
        )
        self.out_dim = out_dim

    def forward(
        self,
        lr_res: torch.Tensor,
        hr_res: torch.Tensor,
    ) -> torch.Tensor:
        """
        lr_res, hr_res: shape [B] or [B,1] in metres.
        returns [B, out_dim]
        """
        lr_res = lr_res.reshape(-1).float()
        hr_res = hr_res.reshape(-1).float()
        ratio = lr_res / hr_res.clamp_min(1e-6)
        x = torch.stack(
            [lr_res / 30.0, hr_res / 30.0, torch.log(ratio.clamp_min(1e-6))],
            dim=-1,
        )
        return self.mlp(x)
