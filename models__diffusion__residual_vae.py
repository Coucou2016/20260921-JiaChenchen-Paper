"""Residual VAE stub for stage P11 (pad 240→256 → latent 32×32). Not trained in V0."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualVAE(nn.Module):
    def __init__(self, latent_ch: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, latent_ch * 2, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_ch, 128, 1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )
        self.latent_ch = latent_ch

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z), mu, logvar
