"""Vanilla SRNO: LR flood only (no geography), residual in log1p space."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.normalization import depth_decode, depth_encode
from models.galerkin import OperatorBlock
from models.heads import ResidualDepthHead, WetDryHead
from models.hydro_encoder import HydroEncoder
from models.implicit_query import coords_for_hr, local_ensemble_sample, make_cell
from models.scale_embedding import ScaleEmbedding


class SRNOSingle(nn.Module):
    def __init__(
        self,
        hydro_width: int = 64,
        hydro_blocks: int = 8,
        operator_width: int = 192,
        heads: int = 8,
        operator_layers: int = 2,
        depth_ref: float = 0.1,
        use_dem: bool = False,
        dem_width: int = 16,
    ) -> None:
        super().__init__()
        self.depth_ref = depth_ref
        self.use_dem = use_dem
        self.hydro = HydroEncoder(2, hydro_width, hydro_blocks)
        self.scale_embedding = ScaleEmbedding(16)
        extra = dem_width if use_dem else 0
        if use_dem:
            self.dem_stem = nn.Sequential(
                nn.Conv2d(1, dem_width, 3, padding=1),
                nn.GELU(),
            )
        in_dim = 4 * hydro_width + 8 + 16 + 2 + extra
        self.imnet = nn.Sequential(nn.Conv2d(in_dim, operator_width, 1), nn.GELU())
        self.operator = nn.Sequential(*[OperatorBlock(operator_width, heads) for _ in range(operator_layers)])
        self.residual_head = ResidualDepthHead(operator_width)
        self.wet_head = WetDryHead(operator_width)

    def forward(self, lr, lr_valid, static_cont=None, landuse=None, lr_res=None, hr_res=None, **kwargs):
        if static_cont is not None and static_cont.numel():
            hr_h, hr_w = static_cont.shape[-2:]
        else:
            hr_h, hr_w = landuse.shape[-2:]
        b = lr.shape[0]
        x = torch.cat([lr[:, :1], lr_valid[:, :1]], dim=1)
        feat_lr = self.hydro(x)
        coord = coords_for_hr(b, hr_h, hr_w, lr.device)
        sampled, rel = local_ensemble_sample(feat_lr, coord)
        hydro_flat = sampled.reshape(b, hr_h, hr_w, -1).permute(0, 3, 1, 2)
        rel_flat = rel.reshape(b, hr_h, hr_w, -1).permute(0, 3, 1, 2)
        if lr_res is None:
            lr_res = torch.full((b,), 10.0, device=lr.device)
            hr_res = torch.full((b,), 2.0, device=lr.device)
        scale = self.scale_embedding(lr_res, hr_res)[:, :, None, None].expand(-1, -1, hr_h, hr_w)
        cell = make_cell((hr_h, hr_w)).to(lr.device).permute(2, 0, 1).unsqueeze(0).expand(b, -1, -1, -1)
        parts = [hydro_flat, rel_flat, scale, cell]
        if self.use_dem:
            dem = static_cont[:, :1]
            parts.append(self.dem_stem(dem))
        feat = self.operator(self.imnet(torch.cat(parts, dim=1)))
        delta_z = self.residual_head(feat)
        base = F.interpolate(lr[:, :1], size=(hr_h, hr_w), mode="bilinear", align_corners=False)
        h = torch.clamp(depth_decode(depth_encode(base, self.depth_ref) + delta_z, self.depth_ref), min=0.0)
        return {"depth": h, "wet_logits": self.wet_head(feat), "residual": delta_z, "base": base}
