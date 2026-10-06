"""Flood-centric evaluation metrics (plan §十二).

Naming contract, so the report and the code cannot drift apart:

* ``*_domain`` metrics are produced by ``metrics.aggregation.FloodMetricAccumulator``
  and pool all pixels of a split before forming the metric. Use these for
  physical performance claims.
* The plain names here (``RMSE_all``, ``CSI_005`` ...) are per-tile values. They
  are meant to be consumed one tile at a time and then either reported as a
  macro average or used for spatial statistics.

Semantics fixed relative to the earlier version:

* ``binary_csi_f1`` returns NaN, not 0, when the denominator is empty (no water
  in either truth or prediction). CSI of an event that never happened is
  undefined, not a score of zero.
* ``flood_area_relative_error`` / ``volume_relative_error`` return NaN when the
  truth quantity is below a physical epsilon, instead of dividing by a floor
  that silently turns the metric into an absolute count.
* ``masked_psnr`` defaults to a fixed data range (``fixed_data_range``), so PSNR
  is comparable across tiles. The previous per-tile range made the scale depend
  on the tile.
* ``masked_ssim`` is a standard windowed SSIM over the valid region;
  ``masked_ssim_global_proxy`` is the old global-statistics approximation and is
  kept under an honest name for continuity only.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F

# Physical cap used as the default fixed data range for PSNR (metres). Chosen
# from the tail of the target distribution rather than per tile.
FIXED_DATA_RANGE = 5.0
EPS_AREA = 1e-9      # metres^2, flooded-area floor
EPS_VOLUME = 1e-6    # metres^3, water-volume floor


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
    """Per-tile CSI and F1 at one threshold.

    Returns ``(nan, nan)`` when there is nothing to score, i.e. truth and
    prediction both have no pixel above the threshold. An empty denominator is
    undefined; it is not zero skill.
    """
    m, t = _as_mask(mask, target)
    p = pred[:, 0] if pred.ndim == 4 else pred
    if m.sum() == 0:
        return float("nan"), float("nan")
    pb = p[m] > thr
    tb = t[m] > thr
    tp = (pb & tb).sum().float()
    fp = (pb & ~tb).sum().float()
    fn = (~pb & tb).sum().float()
    den = tp + fp + fn
    if float(den) == 0.0:
        return float("nan"), float("nan")
    csi = tp / den
    prec = tp / (tp + fp).clamp_min(1e-9)
    rec = tp / (tp + fn).clamp_min(1e-9)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-9)
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
    """Relative error of flooded area at ``thr``.

    NaN when the truth flooded area is below the physical epsilon: a relative
    error against no water is undefined, and reporting the predicted pixel count
    under this name was misleading.
    """
    m, t = _as_mask(mask, target)
    p = pred[:, 0] if pred.ndim == 4 else pred
    if m.sum() == 0:
        return float("nan")
    pa = (p[m] > thr).float().sum()
    ta = (t[m] > thr).float().sum()
    if float(ta) <= EPS_AREA:
        return float("nan")
    return float(((pa - ta).abs() / ta).item())


def volume_relative_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Relative error of total water volume.

    NaN when the truth volume is below the physical epsilon, so a near-dry tile
    cannot produce a spuriously large ratio.
    """
    m, t = _as_mask(mask, target)
    p = pred[:, 0] if pred.ndim == 4 else pred
    if m.sum() == 0:
        return float("nan")
    pv = p[m].clamp_min(0).sum()
    tv = t[m].clamp_min(0).sum()
    if float(tv) <= EPS_VOLUME:
        return float("nan")
    return float(((pv - tv).abs() / tv).item())


def masked_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    data_range: float | None = None,
) -> float:
    """PSNR on the valid region with a FIXED data range by default.

    The default is ``FIXED_DATA_RANGE`` (metres). Passing ``data_range=None``
    previously fell back to the per-tile target range, which made the same MSE
    score differently on a shallow and a deep tile and is not comparable across
    tiles. Keep it fixed unless you deliberately want the legacy behaviour.
    """
    m, t = _as_mask(mask, target)
    p = pred[:, 0] if pred.ndim == 4 else pred
    if m.sum() == 0:
        return float("nan")
    mse = ((p[m] - t[m]) ** 2).mean().item()
    if mse <= 0:
        return 99.0
    dr = FIXED_DATA_RANGE if data_range is None else float(data_range)
    dr = max(dr, 1e-3)
    return float(10.0 * math.log10((dr ** 2) / mse))


def _gaussian_window(win: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(win, device=device, dtype=dtype) - (win - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, win, win)


def masked_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = FIXED_DATA_RANGE,
    win: int = 11,
    sigma: float = 1.5,
) -> float:
    """Standard windowed SSIM on the valid region.

    A window contributes only when ALL of its pixels are valid, so nodata is
    never interpolated into the statistic. This is the metric that may be called
    ``SSIM`` in the paper.
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if target.ndim == 3:
        target = target.unsqueeze(1)
    m4 = mask.bool()
    if m4.ndim == 3:
        m4 = m4.unsqueeze(1)
    x = pred.float()
    y = target.float()
    m = m4.float()
    if x.shape != y.shape:
        y = y.expand_as(x)
    if m.shape != x.shape:
        m = m.expand_as(x)
    # window counts only make sense for a single channel
    if x.shape[1] != 1:
        x = x[:, :1]
        y = y[:, :1]
        m = m[:, :1]

    k = _gaussian_window(win, sigma, x.device, x.dtype)
    pad = win // 2
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu_x = F.conv2d(x, k, padding=pad)
    mu_y = F.conv2d(y, k, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x = F.conv2d(x * x, k, padding=pad) - mu_x2
    sig_y = F.conv2d(y * y, k, padding=pad) - mu_y2
    sig_xy = F.conv2d(x * y, k, padding=pad) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2) + 1e-12
    )
    # a window is usable only if every pixel inside it is valid
    valid_win = F.conv2d(m, torch.ones_like(k), padding=pad)
    usable = valid_win >= (win * win) - 1e-6
    if not bool(usable.any()):
        return float("nan")
    return float(ssim_map[usable].mean().item())


def masked_ssim_global_proxy(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Legacy global-statistics approximation (NOT standard SSIM).

    Kept under an explicit name so existing figures can be reproduced. Do not
    label this as SSIM in the manuscript.
    """
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
    """Per-tile metrics. Feed to ``average_metrics`` for a macro summary.

    For domain-pooled metrics use ``FloodMetricAccumulator`` instead.
    """
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
        "SSIM": masked_ssim(pred, target, m),
        "SSIM_global_proxy": masked_ssim_global_proxy(pred, target, m),
    }
