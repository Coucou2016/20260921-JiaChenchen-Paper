#!/usr/bin/env python
"""Train residual latent diffusion on top of a frozen HydroGeo-SRNO (plan §§二十二–二十四).

Pads 240→256, VAE latent 32×32×4, condition = noisy residual + det latent + geo latent (12 ch).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.normalization import depth_encode
from dataset.wellington_fixed_sr import WellingtonFixedSRDataset, collate_fixed
from engine.checkpoint import load_checkpoint, save_checkpoint
from engine.evaluator import build_model
from models.diffusion.condition_encoder import GeoConditionEncoder
from models.diffusion.diffusion_utils import crop_256_to_240, linear_beta_schedule, pad_240_to_256
from models.diffusion.residual_ldm import ResidualLDM
from models.diffusion.residual_vae import ResidualVAE


def q_sample(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    # t: [B]
    b = x0.shape[0]
    sa = sqrt_alphas_cumprod[t].view(b, 1, 1, 1)
    so = sqrt_one_minus_alphas_cumprod[t].view(b, 1, 1, 1)
    return sa * x0 + so * noise, noise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v2_residual_ldm.yaml")
    parser.add_argument("--det_ckpt", default="outputs/v0_10m2m_hmax/best_csi.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    det_ckpt = ROOT / args.det_ckpt
    if not det_ckpt.exists():
        raise FileNotFoundError(f"deterministic ckpt required: {det_ckpt}")

    # Frozen deterministic backbone
    v0_cfg = yaml.safe_load((ROOT / "configs/v0_10m2m_hmax.yaml").read_text(encoding="utf-8"))
    det = build_model("hydrogeo_srno", v0_cfg).to(device)
    load_checkpoint(det_ckpt, det, map_location=device)
    det.eval()
    for p in det.parameters():
        p.requires_grad_(False)

    vae = ResidualVAE(latent_ch=4).to(device)
    geo_enc = GeoConditionEncoder(in_ch=14, latent_ch=4).to(device)
    # For geo at 240, pad then encode — simple downsample path inside encoder expects stride-8 → 30 for 240
    # Use pad to 256 → 32 latent
    ldm = ResidualLDM(in_channel=12, out_channel=4).to(device)

    params = list(vae.parameters()) + list(geo_enc.parameters()) + list(ldm.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)

    timesteps = int(cfg.get("diffusion", {}).get("timesteps", 1000))
    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_ac = torch.sqrt(alphas_cumprod)
    sqrt_om = torch.sqrt(1.0 - alphas_cumprod)

    ds = WellingtonFixedSRDataset(root=ROOT / "dataset", split="train")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fixed)
    out_dir = ROOT / cfg.get("output", {}).get("dir", "outputs/v2_residual_ldm")
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.no_grad():
                out = det(
                    lr=batch["lr"],
                    lr_valid=batch["lr_valid"],
                    static_cont=batch["static_cont"],
                    landuse=batch["landuse"],
                    lr_res=batch["lr_res"],
                    hr_res=batch["hr_res"],
                )
                h_det = out["depth"]
                # residual in log-depth space
                z_true = depth_encode(batch["hr"].clamp_min(0))
                z_det = depth_encode(h_det.clamp_min(0))
                rz = (z_true - z_det) * batch["mask"].unsqueeze(1).float()

            rz_p = pad_240_to_256(rz)
            det_p = pad_240_to_256(h_det)
            # rebuild geo continuous padded
            geo_p = pad_240_to_256(batch["static_cont"])

            recon, mu, logvar = vae(rz_p)
            # VAE loss
            recon_loss = F.mse_loss(recon, rz_p)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # Det / geo latents via VAE encoder mean / geo encoder
            with torch.no_grad():
                mu_det, _ = vae.encode(det_p)
            # GeoConditionEncoder: 256→32
            geo_lat = geo_enc(geo_p)

            b = rz_p.shape[0]
            t = torch.randint(0, timesteps, (b,), device=device)
            noise = torch.randn_like(mu)
            x_noisy, noise = q_sample(mu.detach(), t, sqrt_ac, sqrt_om, noise)
            cond = torch.cat([x_noisy, mu_det.detach(), geo_lat], dim=1)
            eps_pred = ldm(cond)
            diff_loss = F.mse_loss(eps_pred, noise)

            loss = recon_loss + 0.001 * kl + diff_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        mean_loss = sum(losses) / max(len(losses), 1)
        print(f"epoch={epoch} loss={mean_loss:.4f} steps={step}")
        torch.save(
            {
                "epoch": epoch,
                "vae": vae.state_dict(),
                "geo_enc": geo_enc.state_dict(),
                "ldm": ldm.state_dict(),
                "optimizer": opt.state_dict(),
            },
            out_dir / "last_ldm.pt",
        )
        if args.max_steps and step >= args.max_steps:
            break
    (out_dir / "train_summary.json").write_text(
        json.dumps({"epochs": epoch, "steps": step, "elapsed_h": (time.time() - t0) / 3600}, indent=2),
        encoding="utf-8",
    )
    print(f"done → {out_dir}")


if __name__ == "__main__":
    main()
