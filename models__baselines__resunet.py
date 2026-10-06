"""ResUNet baseline: LR flood upsampled + HR DEM/static fusion (FLO-SR style)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.normalization import depth_decode, depth_encode


class ConvBNAct(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResUNet(nn.Module):
    def __init__(self, static_channels: int = 14, base: int = 32, depth_ref: float = 0.1) -> None:
        super().__init__()
        self.depth_ref = depth_ref
        in_ch = 1 + static_channels  # upsampled LR depth + static_cont (no landuse embed for simplicity)
        self.enc1 = ConvBNAct(in_ch, base)
        self.enc2 = ConvBNAct(base, base * 2)
        self.enc3 = ConvBNAct(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBNAct(base * 4, base * 8)
        self.up2 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec2 = ConvBNAct(base * 8, base * 4)
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec1 = ConvBNAct(base * 4, base * 2)
        self.up0 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec0 = ConvBNAct(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)
        self.wet = nn.Conv2d(base, 1, 1)

    def forward(
        self,
        lr: torch.Tensor,
        lr_valid: torch.Tensor | None = None,
        static_cont: torch.Tensor | None = None,
        landuse: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        assert static_cont is not None
        hr_h, hr_w = static_cont.shape[-2:]
        base_h = F.interpolate(lr[:, :1], size=(hr_h, hr_w), mode="bilinear", align_corners=False)
        x = torch.cat([base_h, static_cont], dim=1)
        # pad to multiple of 8
        ph = (8 - hr_h % 8) % 8
        pw = (8 - hr_w % 8) % 8
        x = F.pad(x, (0, pw, 0, ph))
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d2 = self.dec2(torch.cat([self.up2(b), e3], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e2], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), e1], dim=1))
        delta_z = self.head(d0)[:, :, :hr_h, :hr_w]
        wet_logits = self.wet(d0)[:, :, :hr_h, :hr_w]
        z_base = depth_encode(base_h, self.depth_ref)
        h = torch.clamp(depth_decode(z_base + delta_z, self.depth_ref), min=0.0)
        return {"depth": h, "wet_logits": wet_logits, "residual": delta_z, "base": base_h}
