"""Masked flood-aware losses (plan §十). No mass-conservation vs LR."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_reduce(val: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask = mask.float()
    while mask.ndim < val.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(val)
    denom = mask.sum().clamp_min(eps)
    return (val * mask).sum() / denom


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    err = F.huber_loss(pred, target, reduction="none", delta=delta)
    return _masked_reduce(err, mask)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_reduce((pred - target).abs(), mask)


def masked_focal_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    while mask.ndim < logits.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(logits).float()
    target = target.float()
    if target.ndim == logits.ndim - 1:
        target = target.unsqueeze(1)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, p, 1 - p)
    loss = alpha * (1 - pt).pow(gamma) * bce
    return (loss * mask).sum() / mask.sum().clamp_min(1e-6)


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Simple depth edge magnitude. x: B,1,H,W"""
    kx = x.new_tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).view(1, 1, 3, 3)
    ky = x.new_tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


class FloodLoss(nn.Module):
    """
    L = L_depth + 0.30 L_wet + 0.20 L_log + 0.10 L_boundary + 0.10 L_extreme
    """

    def __init__(
        self,
        w_depth: float = 1.0,
        w_wet: float = 0.30,
        w_log: float = 0.20,
        w_boundary: float = 0.10,
        w_extreme: float = 0.10,
        wet_threshold: float = 0.05,
        depth_ref: float = 0.10,
        extreme_quantile: float = 0.95,
        w_deep: float = 0.0,
        deep_threshold: float = 1.0,
        deep_quantile: float | None = None,
        deep_delta: float = 5.0,
    ) -> None:
        super().__init__()
        self.w_depth = w_depth
        self.w_wet = w_wet
        self.w_log = w_log
        self.w_boundary = w_boundary
        self.w_extreme = w_extreme
        self.wet_threshold = wet_threshold
        self.depth_ref = depth_ref
        self.extreme_quantile = extreme_quantile
        # Deep-water term. Motivation (measured on this dataset): l_depth and l_log act on the
        # LOG1P residual, where a 2 m error at 5 m depth costs ~6x less than in metres, so
        # ~45% of the loss is nearly blind to deep-water magnitude. The l_extreme mask uses a
        # per-tile quantile whose threshold is only ~0.19 m (median) because most tiles are
        # mostly dry, so it does not reliably cover the deep tail either. l_deep adds a
        # LINEAR (metre-space) penalty on genuine deep water, with an absolute threshold and
        # an optional wet-only quantile so the mask cannot collapse onto shallow cells.
        self.w_deep = w_deep
        self.deep_threshold = deep_threshold
        self.deep_quantile = deep_quantile
        self.deep_delta = deep_delta

    def forward(
        self,
        output: dict,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred_h = output["depth"]
        if pred_h.ndim == 3:
            pred_h = pred_h.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)
        mask_b = mask.bool()
        if mask_b.ndim == 4:
            mask_b = mask_b[:, 0]

        wet = target[:, 0] > self.wet_threshold
        pred_z = torch.log1p(pred_h / self.depth_ref)
        true_z = torch.log1p(target / self.depth_ref)

        l_depth = masked_huber(pred_z, true_z, mask_b)
        l_log = masked_l1(pred_z, true_z, mask_b)

        if "wet_logits" in output and output["wet_logits"] is not None:
            l_wet = masked_focal_bce(output["wet_logits"], wet.float(), mask_b)
        else:
            l_wet = pred_h.new_tensor(0.0)

        wet_mask = mask_b & wet
        l_metric = masked_l1(pred_h, target, wet_mask)

        # Boundary: zero nodata before Sobel so convolution cannot leak
        mask4 = mask_b.unsqueeze(1).float()
        e_pred = sobel_edges(pred_h * mask4)
        e_true = sobel_edges(target * mask4)
        l_boundary = masked_l1(e_pred, e_true, mask_b)

        # Extreme: emphasize high depths among valid cells
        with torch.no_grad():
            valid_vals = target[:, 0][mask_b]
            if valid_vals.numel() > 0:
                thr = torch.quantile(valid_vals, self.extreme_quantile)
            else:
                thr = target.new_tensor(1.0)
        extreme_mask = mask_b & (target[:, 0] >= thr)
        l_extreme = masked_l1(pred_h, target, extreme_mask) if extreme_mask.any() else pred_h.new_tensor(0.0)

        # Deep-water term in METRE space, restricted to the deep tail. The source of truth
        # is the target mask only; no fabricated depth is ever built, so no gradient flows
        # through a constant. The threshold is fixed, or the deep_quantile of the WET cells
        # so the mask cannot collapse onto shallow water.
        l_deep = pred_h.new_tensor(0.0)
        if self.w_deep > 0:
            with torch.no_grad():
                if valid_vals.numel() > 0:
                    wet_vals = valid_vals[valid_vals > self.wet_threshold]
                    if self.deep_quantile is not None and wet_vals.numel() > 1:
                        deep_thr = torch.quantile(wet_vals, self.deep_quantile)
                        deep_thr = torch.maximum(deep_thr, deep_thr.new_tensor(self.deep_threshold))
                    else:
                        deep_thr = target.new_tensor(self.deep_threshold)
                else:
                    deep_thr = target.new_tensor(self.deep_threshold)
            deep_mask = mask_b & (target[:, 0] >= deep_thr)
            if deep_mask.any():
                # Huber on the metre-space error: outliers cannot dominate once they are
                # already grossly wrong, unlike a plain L1/L2 term.
                deep_err = F.huber_loss(pred_h, target, reduction="none", delta=self.deep_delta)
                l_deep = _masked_reduce(deep_err, deep_mask)

        # Plain linear wet L1: this is our OBJECTIVE (it is what RMSE/MAE_wet measure) but it
        # is NOT optimized, because it is dominated by abundant shallow pixels. Recorded so
        # progress on the metric can be told apart from progress on the training loss.
        l_wet_mae_linear = masked_l1(pred_h, target, wet_mask)

        total = (
            self.w_depth * l_depth
            + self.w_wet * l_wet
            + self.w_log * l_log
            + self.w_boundary * l_boundary
            + self.w_extreme * l_extreme
            + self.w_deep * l_deep
            + 0.0 * l_metric  # tracked only
        )
        return {
            "total": total,
            "l_depth": l_depth.detach(),
            "l_wet": l_wet.detach(),
            "l_log": l_log.detach(),
            "l_boundary": l_boundary.detach(),
            "l_extreme": l_extreme.detach(),
            "l_deep": l_deep.detach(),
            "l_wet_mae": l_metric.detach(),
            "l_wet_mae_linear": l_wet_mae_linear.detach(),
        }


class MaskedL1Loss(nn.Module):
    def forward(self, output: dict, target: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = output["depth"]
        loss = masked_l1(pred, target, mask)
        return {"total": loss, "l_depth": loss.detach()}
