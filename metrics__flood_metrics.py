"""Flood-centric evaluation metrics (plan §十二). PSNR/SSIM secondary."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


def _as_mask(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    m = mask.bool()
    if m.ndim == 4:
        m = m[:, 0]
    if ref.ndim == 4:
        ref = ref[:, 0]
    return m, ref


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    m, t = _as_mask(mask, target)
    p = pred[:, 0] if pred.ndim == 4 else pred
    if m.sum() == 0:
        return float("nan")
    err = (p[m] - t[m]) ** 2
    return float(torch.sqrt(err.mean()).item())


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    m, t = _as_mask(mask, target)
    p = pred[:, 0] if pred.ndim == 4 else pred
    if m.sum() == 0:
        return float("nan")
    return float((p[m] - t[m]).abs().mean().item())


def binary_csi_f1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, thr: float) -> tuple[float, float]:
    m, t = _as_mask(mask, target)
    p = pred[:, 0] if pred.ndim == 4 else pred
    if m.sum() == 0:
        return float("nan"), float("nan")
    pb = p[m] > thr
    tb = t[m] > thr
    tp = (pb & tb).sum().float()
    fp = (pb & ~tb).sum().float()
    fn = (~pb & tb).sum().float()
    csi = tp / (tp + fp + fn).clamp_min(1.0)
    prec = tp / (tp + fp).clamp_min(1.0)
    rec = tp / (tp + fn).clamp_min(1.0)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-6)
    return float(csi.item()), float(f1.item())


def peak_depth_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    m, t = _as_mask(mask, target)
    p = pred[:, 0] if pred.ndim == 4 else pred
    if m.sum() == 0:
        return float("nan")
    # per-sample peak then mean absolute error
    b = p.shape[0]
    errs = []
    for i in range(b):
        mi = m[i]
        if mi.sum() == 0:
            continue
        errs.append((p[i][mi].max() - t[i][mi].max()).abs())
    if not errs:
        return float("nan")
    return float(torch.stack(errs).mean().item())


def flood_area_relative_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, thr: float = 0.05) -> float:
    m, t = _as_mask(mask, target)
    p = pred[:, 0] if pred.ndim == 4 else pred
    if m.sum() == 0:
        return float("nan")
    pa = (p[m] > thr).float().sum()
    ta = (t[m] > thr).float().sum()
    return float(((pa - ta).abs() / ta.clamp_min(1.0)).item())


def volume_relative_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    m, t = _as_mask(mask, target)
    p = pred[:, 0] if pred.ndim == 4 else pred
    if m.sum() == 0:
        return float("nan")
    pv = p[m].clamp_min(0).sum()
    tv = t[m].clamp_min(0).sum()
    return float(((pv - tv).abs() / tv.clamp_min(1e-6)).item())


def masked_psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, data_range: float | None = None) -> float:
    m, t = _as_mask(mask, target)
    p = pred[:, 0] if pred.ndim == 4 else pred
    if m.sum() == 0:
        return float("nan")
    mse = ((p[m] - t[m]) ** 2).mean().item()
    if mse <= 0:
        return 99.0
    if data_range is None:
        data_range = float(t[m].max().item() - t[m].min().item())
        data_range = max(data_range, 1e-3)
    return float(10.0 * math.log10((data_range ** 2) / mse))


def masked_ssim_approx(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Lightweight masked SSIM on valid region (global stats; secondary metric)."""
    m, t = _as_mask(mask, target)
    p = pred[:, 0] if pred.ndim == 4 else pred
    if m.sum() < 2:
        return float("nan")
    x = p[m]
    y = t[m]
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(unbiased=False), y.var(unbiased=False)
    cov = ((x - mx) * (y - my)).mean()
    c1, c2 = 1e-4, 9e-4
    ssim = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    return float(ssim.item())


@torch.no_grad()
def compute_flood_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    wet_thr: float = 0.05,
) -> Dict[str, float]:
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if target.ndim == 3:
        target = target.unsqueeze(1)
    m = mask.bool()
    if m.ndim == 4:
        m = m[:, 0]
    wet = m & (target[:, 0] > wet_thr)

    csi005, f1005 = binary_csi_f1(pred, target, m, 0.05)
    csi030, f1030 = binary_csi_f1(pred, target, m, 0.30)
    csi100, f1100 = binary_csi_f1(pred, target, m, 1.00)

    return {
        "RMSE_all": masked_rmse(pred, target, m),
        "RMSE_wet": masked_rmse(pred, target, wet),
        "MAE_wet": masked_mae(pred, target, wet),
        "CSI_005": csi005,
        "F1_005": f1005,
        "CSI_030": csi030,
        "F1_030": f1030,
        "CSI_100": csi100,
        "F1_100": f1100,
        "PeakDepthError": peak_depth_error(pred, target, m),
        "FloodAreaRelativeError": flood_area_relative_error(pred, target, m, 0.05),
        "VolumeRelativeError": volume_relative_error(pred, target, m),
        "PSNR": masked_psnr(pred, target, m),
        "SSIM": masked_ssim_approx(pred, target, m),
    }
