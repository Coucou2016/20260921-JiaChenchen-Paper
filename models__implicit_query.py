"""Coordinate helpers for continuous / local-ensemble query (SRNO-style)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_coord(shape: tuple[int, int], flatten: bool = True) -> torch.Tensor:
    """Normalized coords in [-1, 1], row=y, col=x. Returns [H*W, 2] or [H, W, 2]."""
    h, w = shape
    # align_corners=False style cell centers
    gy = torch.linspace(-1 + 1 / h, 1 - 1 / h, h)
    gx = torch.linspace(-1 + 1 / w, 1 - 1 / w, w)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
    coord = torch.stack([grid_x, grid_y], dim=-1)  # H,W,2 with (x,y)
    if flatten:
        return coord.reshape(-1, 2)
    return coord


def make_cell(shape: tuple[int, int]) -> torch.Tensor:
    h, w = shape
    cy = torch.full((h, w), 2.0 / h)
    cx = torch.full((h, w), 2.0 / w)
    return torch.stack([cx, cy], dim=-1)  # H,W,2


def local_ensemble_sample(
    feat: torch.Tensor,
    coord: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample 4 LR neighbors around each query coordinate.

    feat: [B, C, H, W]
    coord: [B, Q, 2] in [-1, 1] (x, y)
    returns:
      sampled: [B, Q, 4, C]
      rel: [B, Q, 4, 2] relative offset in normalized coords
    """
    b, c, h, w = feat.shape
    q = coord.shape[1]
    device = feat.device
    dtype = feat.dtype

    # Pixel centers in continuous coords matching make_coord
    # Inverse of make_coord mapping
    vx = ((coord[..., 0] + 1) * w - 1) / 2  # continuous x in pixel units
    vy = ((coord[..., 1] + 1) * h - 1) / 2

    vx0 = vx.floor()
    vy0 = vy.floor()
    vx1 = vx0 + 1
    vy1 = vy0 + 1

    xs = [vx0, vx0, vx1, vx1]
    ys = [vy0, vy1, vy0, vy1]

    samples = []
    rels = []
    for xi, yi in zip(xs, ys):
        # clamp for gathering
        x_c = xi.clamp(0, w - 1)
        y_c = yi.clamp(0, h - 1)
        # grid_sample expects [B, H_out, W_out, 2] with x,y in [-1,1]
        gx = (2 * (x_c + 0.5) / w) - 1
        gy = (2 * (y_c + 0.5) / h) - 1
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # B,Q,1,2
        s = F.grid_sample(feat, grid, mode="nearest", align_corners=False)  # B,C,Q,1
        s = s.squeeze(-1).permute(0, 2, 1)  # B,Q,C
        samples.append(s)
        # relative coord from neighbor center to query
        # neighbor center in normalized space
        nx = (2 * (xi + 0.5) / w) - 1
        ny = (2 * (yi + 0.5) / h) - 1
        rel = torch.stack([coord[..., 0] - nx, coord[..., 1] - ny], dim=-1)
        rels.append(rel)

    sampled = torch.stack(samples, dim=2)  # B,Q,4,C
    rel = torch.stack(rels, dim=2)  # B,Q,4,2
    return sampled.to(dtype), rel.to(dtype)


def coords_for_hr(batch: int, hr_h: int, hr_w: int, device: torch.device) -> torch.Tensor:
    coord = make_coord((hr_h, hr_w), flatten=True).to(device)  # Q,2
    return coord.unsqueeze(0).expand(batch, -1, -1).contiguous()
