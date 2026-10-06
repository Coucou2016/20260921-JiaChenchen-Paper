"""Galerkin-style attention adapted from SRNO (cleaned; heads renamed).

Original SRNO passed `blocks` into simple_attn but that argument is the
number of attention heads, not stacked operator blocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,C,H,W
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)


class GalerkinAttention(nn.Module):
    """Channel-wise Galerkin attention over spatial tokens (SRNO-style)."""

    def __init__(self, width: int, heads: int = 8) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width {width} must be divisible by heads {heads}")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Conv2d(width, width * 3, kernel_size=1)
        self.proj = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=1),
        )
        self.norm_k = nn.LayerNorm(self.head_dim)
        self.norm_v = nn.LayerNorm(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        # B, heads, head_dim, HW
        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(b, self.heads, self.head_dim, h * w)

        q, k, v = reshape(q), reshape(k), reshape(v)
        # Normalize K,V along channel (Galerkin)
        k = self.norm_k(k.transpose(-1, -2)).transpose(-1, -2)
        v = self.norm_v(v.transpose(-1, -2)).transpose(-1, -2)
        # Attention: (Q K^T) V  with channel-as-feature Galerkin product
        # k: B,H,D,N ; q: B,H,D,N -> attn over channel dim
        attn = torch.matmul(q, k.transpose(-1, -2)) / (h * w)  # B,heads,D,D
        out = torch.matmul(attn, v)  # B,heads,D,N
        out = out.reshape(b, c, h, w)
        return x + self.proj(out)


class OperatorBlock(nn.Module):
    def __init__(self, width: int, heads: int = 8) -> None:
        super().__init__()
        self.attn = GalerkinAttention(width, heads=heads)
        self.ff = nn.Sequential(
            LayerNorm2d(width),
            nn.Conv2d(width, width * 2, 1),
            nn.GELU(),
            nn.Conv2d(width * 2, width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        return x + self.ff(x)
