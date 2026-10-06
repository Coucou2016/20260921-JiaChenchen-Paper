"""HydroGeo-SRNO: geography-guided arbitrary-scale neural operator for flood SR."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.normalization import depth_decode, depth_encode
from models.galerkin import OperatorBlock
from models.geo_encoder import GeoEncoder
from models.heads import ResidualDepthHead, WetDryHead
from models.hydro_encoder import HydroEncoder
from models.implicit_query import coords_for_hr, local_ensemble_sample, make_cell
from models.scale_embedding import ScaleEmbedding


class HydroGeoSRNO(nn.Module):
    """
    Dual-branch residual SRNO:
      HydroEncoder(LR flood + valid) + GeoEncoder(HR static) → Galerkin operator
      → Δz residual on top of bilinear base in log1p depth space + wet head.
    """

    def __init__(
        self,
        hydro_width: int = 64,
        hydro_blocks: int = 8,
        hydro_in_channels: int = 2,
        geo_width: int = 48,
        geo_in_cont: int = 14,
        landuse_embedding: int = 8,
        use_landuse: bool = True,
        operator_width: int = 192,
        heads: int = 8,
        operator_layers: int = 2,
        scale_dim: int = 16,
        depth_ref: float = 0.10,
        predict_residual: bool = True,
        predict_wet: bool = True,
    ) -> None:
        super().__init__()
        self.depth_ref = depth_ref
        self.predict_residual = predict_residual
        self.predict_wet = predict_wet
        self.hydro_width = hydro_width
        self.geo_width = geo_width
        self.scale_dim = scale_dim

        self.hydro_encoder = HydroEncoder(
            in_channels=hydro_in_channels,
            width=hydro_width,
            n_blocks=hydro_blocks,
        )
        self.geo_encoder = GeoEncoder(
            in_cont=geo_in_cont,
            embed_dim=landuse_embedding,
            width=geo_width,
            use_landuse=use_landuse,
        )
        self.scale_embedding = ScaleEmbedding(out_dim=scale_dim)

        # four LR neighbors: hydro 4*W + rel 4*2 + geo W_g + scale + cell 2
        in_dim = 4 * hydro_width + 4 * 2 + geo_width + scale_dim + 2
        self.imnet_in = nn.Sequential(
            nn.Conv2d(in_dim, operator_width, 1),
            nn.GELU(),
        )
        self.operator = nn.Sequential(
            *[OperatorBlock(operator_width, heads=heads) for _ in range(operator_layers)]
        )
        self.residual_head = ResidualDepthHead(operator_width)
        self.wet_head = WetDryHead(operator_width) if predict_wet else None

    def encode_lr(self, lr: torch.Tensor, lr_valid: torch.Tensor) -> torch.Tensor:
        if lr_valid.shape[1] != lr.shape[1]:
            # lr_valid may be [B,1,H,W] while lr has more channels
            if lr_valid.shape[1] == 1 and lr.shape[1] > 1:
                lr_valid = lr_valid.expand(-1, lr.shape[1], -1, -1)
        x = torch.cat([lr, lr_valid[:, :1]], dim=1)
        # If hydro expects 2 channels (depth+valid) but lr has flow, take depth only + valid
        if x.shape[1] != self.hydro_encoder.head.in_channels:
            # pack: first channel depth (+ optional flow) truncated/padded to expected-1, then valid
            need = self.hydro_encoder.head.in_channels
            depth_part = lr[:, : max(need - 1, 1)]
            if depth_part.shape[1] < need - 1:
                pad = need - 1 - depth_part.shape[1]
                depth_part = F.pad(depth_part, (0, 0, 0, 0, 0, pad))
            x = torch.cat([depth_part[:, : need - 1], lr_valid[:, :1]], dim=1)
        return self.hydro_encoder(x)

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
        b, _, hr_h, hr_w = (
            lr.shape[0],
            None,
            static_cont.shape[-2] if static_cont.numel() else landuse.shape[-2],
            static_cont.shape[-1] if static_cont.numel() else landuse.shape[-1],
        )
        if static_cont.numel() == 0:
            # lr-only: synthesize empty cont with correct spatial size from landuse zeros
            hr_h, hr_w = landuse.shape[-2], landuse.shape[-1]

        feat_lr = self.encode_lr(lr, lr_valid)  # B,Ch,hl,wl
        feat_geo = self.geo_encoder(static_cont, landuse)  # B,Cg,H,W

        # Query grid at HR
        coord = coords_for_hr(b, hr_h, hr_w, lr.device)  # B,Q,2
        sampled, rel = local_ensemble_sample(feat_lr, coord)  # B,Q,4,C and B,Q,4,2
        hydro_flat = sampled.reshape(b, hr_h, hr_w, -1).permute(0, 3, 1, 2)  # B,4C,H,W
        rel_flat = rel.reshape(b, hr_h, hr_w, -1).permute(0, 3, 1, 2)  # B,8,H,W

        scale = self.scale_embedding(lr_res, hr_res)  # B,S
        scale_map = scale[:, :, None, None].expand(-1, -1, hr_h, hr_w)

        cell = make_cell((hr_h, hr_w)).to(lr.device)  # H,W,2
        cell_map = cell.permute(2, 0, 1).unsqueeze(0).expand(b, -1, -1, -1)

        inp = torch.cat([hydro_flat, rel_flat, feat_geo, scale_map, cell_map], dim=1)
        feat = self.imnet_in(inp)
        feat = self.operator(feat)

        delta_z = self.residual_head(feat)
        if not self.predict_residual:
            delta_z = torch.zeros_like(delta_z)

        # Bilinear base in physical depth, then log1p residual
        lr_depth = lr[:, :1]
        base_h = F.interpolate(lr_depth, size=(hr_h, hr_w), mode="bilinear", align_corners=False)
        base_h = torch.clamp(base_h, min=0.0)
        z_base = depth_encode(base_h, self.depth_ref)
        z_pred = z_base + delta_z
        h_pred = depth_decode(z_pred, self.depth_ref)
        h_pred = torch.clamp(h_pred, min=0.0)

        wet_logit = self.wet_head(feat) if self.wet_head is not None else None

        out: dict[str, torch.Tensor] = {
            "depth": h_pred,
            "residual": delta_z,
            "base": base_h,
        }
        if wet_logit is not None:
            out["wet_logits"] = wet_logit
        return out


def build_hydrogeo_srno(cfg: dict) -> HydroGeoSRNO:
    m = cfg.get("model", cfg)
    hydro = m.get("hydro_encoder", {})
    geo = m.get("geo_encoder", {})
    op = m.get("operator", {})
    return HydroGeoSRNO(
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
        depth_ref=m.get("depth_ref", 0.10),
        predict_residual=m.get("predict_residual", True),
        predict_wet=m.get("predict_wet", True),
    )
