"""GeoEncoder: continuous static channels + landuse embedding (never z-score landuse)."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.hydro_encoder import ResidualBlock


class GeoEncoder(nn.Module):
    def __init__(
        self,
        in_cont: int = 14,
        embed_dim: int = 8,
        width: int = 48,
        use_landuse: bool = True,
        n_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.use_landuse = use_landuse
        self.embed_dim = embed_dim
        self.landuse_embed = nn.Embedding(8, embed_dim, padding_idx=0)
        in_ch = in_cont + (embed_dim if use_landuse else 0)
        if in_ch <= 0:
            # lr-only ablation: emit zeros later; keep a dummy param for DDP safety
            self.stem = None
            self.width = width
            return
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.GELU(),
        ]
        for _ in range(n_blocks):
            layers.append(ResidualBlock(width))
        self.stem = nn.Sequential(*layers)
        self.width = width

    def forward(
        self,
        static_cont: torch.Tensor,
        landuse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.stem is None:
            b = static_cont.shape[0] if static_cont.ndim == 4 else 1
            # Infer spatial size from landuse if cont empty
            if static_cont.numel() == 0 and landuse is not None:
                _, h, w = landuse.shape if landuse.ndim == 3 else landuse.shape[1:]
                b = landuse.shape[0]
            elif static_cont.ndim == 4:
                b, _, h, w = static_cont.shape
            else:
                raise RuntimeError("cannot infer geo feature shape")
            return static_cont.new_zeros(b, self.width, h, w)

        feats = [static_cont]
        if self.use_landuse:
            if landuse is None:
                raise ValueError("landuse required when use_landuse=True")
            emb = self.landuse_embed(landuse.long())  # B,H,W,E
            emb = emb.permute(0, 3, 1, 2).contiguous()
            feats.append(emb)
        x = torch.cat(feats, dim=1)
        return self.stem(x)
