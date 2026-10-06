"""Masked flood-aware losses (plan §十). No mass-conservation vs LR.

Fixes relative to the earlier version, each of which changed what an experiment
actually measured:

* ``masked_focal_bce`` now applies the CLASS-SPECIFIC focal weighting
  ``alpha_t = alpha`` on positives and ``1 - alpha`` on negatives. The previous
  version multiplied every element by the same scalar ``alpha``, which is only a
  global rescale and is not the focal loss that the method section describes.
  Pass ``alpha_mode="scale"`` to reproduce the legacy behaviour.
* The boundary term is evaluated only where the whole Sobel stencil is valid.
  Zeroing nodata before the convolution creates a strong artificial edge at the
  valid/nodata border; the old code then averaged that fake edge into the loss
  on valid pixels. A 3x3 erosion of the valid mask removes those stencils.
* The extreme and deep quantile thresholds are computed PER SAMPLE, so the
  threshold a sample sees does not depend on which other samples happened to
  share its batch when ``batch_size > 1``.
"""

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
    alpha_mode: str = "class",
) -> torch.Tensor:
    """Focal BCE with class-specific alpha balancing.

    ``alpha_mode="class"`` (default) weights positives by ``alpha`` and
    negatives by ``1 - alpha``, which is the standard focal loss.
    ``alpha_mode="scale"`` multiplies everything by ``alpha`` (legacy behaviour,
    equivalent to a global rescale) and exists only for reproducibility.
    """
    while mask.ndim < logits.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(logits).float()
    target = target.float()
    if target.ndim == logits.ndim - 1:
        target = target.unsqueeze(1)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, p, 1 - p)
    if alpha_mode == "class":
        alpha_t = torch.where(
            target > 0.5,
            logits.new_tensor(alpha),
            logits.new_tensor(1.0 - alpha),
        )
    elif alpha_mode == "scale":
        alpha_t = logits.new_tensor(alpha)
    else:
        raise ValueError(f"unknown alpha_mode: {alpha_mode}")
    loss = alpha_t * (1 - pt).pow(gamma) * bce
    return (loss * mask).sum() / mask.sum().clamp_min(1e-6)


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Simple depth edge magnitude. x: B,1,H,W"""
    kx = x.new_tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).view(1, 1, 3, 3)
    ky = x.new_tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def valid_stencil_mask(mask_b: torch.Tensor, exclude_image_border: bool = True) -> torch.Tensor:
    """Pixels whose full 3x3 neighbourhood is valid.

    Used so a Sobel stencil never straddles the valid/nodata border, where
    zero-filled nodata would inject a fake gradient. When
    ``exclude_image_border`` is set, the outermost one-pixel ring is also
    dropped because the convolution pads there and would see implicit zeros.
    """
    mask4 = mask_b[:, None].float()
    k = torch.ones((1, 1, 3, 3), device=mask4.device, dtype=mask4.dtype)
    valid_count = F.conv2d(mask4, k, padding=1)
    ok = valid_count[:, 0].eq(9.0)
    if exclude_image_border:
        ok[:, 0, :] = False
        ok[:, -1, :] = False
        ok[:, :, 0] = False
        ok[:, :, -1] = False
    return ok


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
        focal_alpha: float = 0.25,
        focal_alpha_mode: str = "class",
        quantile_per_sample: bool = True,
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
        self.focal_alpha = focal_alpha
        self.focal_alpha_mode = focal_alpha_mode
        self.quantile_per_sample = quantile_per_sample
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

    def _quantile_threshold(self, values: torch.Tensor, q: float, fallback: float) -> torch.Tensor:
        if values.numel() == 0:
            return values.new_tensor(fallback)
        return torch.quantile(values, q)

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
            l_wet = masked_focal_bce(
                output["wet_logits"], wet.float(), mask_b,
                alpha=self.focal_alpha, alpha_mode=self.focal_alpha_mode,
            )
        else:
            l_wet = pred_h.new_tensor(0.0)

        wet_mask = mask_b & wet
        l_metric = masked_l1(pred_h, target, wet_mask)

        # Boundary: evaluated only where the full Sobel stencil is valid, so a
        # nodata border cannot masquerade as a flood edge.
        mask4 = mask_b.unsqueeze(1).float()
        e_pred = sobel_edges(pred_h * mask4)
        e_true = sobel_edges(target * mask4)
        boundary_ok = valid_stencil_mask(mask_b)
        if boundary_ok.any():
            l_boundary = masked_l1(e_pred, e_true, boundary_ok)
        else:
            l_boundary = pred_h.new_tensor(0.0)

        # Extreme: emphasize high depths among valid cells. Threshold is
        # per-sample so batching cannot mix tiles into one threshold.
        if self.quantile_per_sample:
            ext_terms = []
            for i in range(target.shape[0]):
                mi = mask_b[i]
                vals = target[i, 0][mi]
                thr = self._quantile_threshold(vals, self.extreme_quantile, 1.0)
                em = mi & (target[i, 0] >= thr)
                if em.any():
                    ext_terms.append(masked_l1(pred_h[i:i + 1], target[i:i + 1], em[None]))
            l_extreme = torch.stack(ext_terms).mean() if ext_terms else pred_h.new_tensor(0.0)
        else:
            with torch.no_grad():
                valid_vals = target[:, 0][mask_b]
                thr = self._quantile_threshold(valid_vals, self.extreme_quantile, 1.0)
            extreme_mask = mask_b & (target[:, 0] >= thr)
            l_extreme = masked_l1(pred_h, target, extreme_mask) if extreme_mask.any() else pred_h.new_tensor(0.0)

        # Deep-water term in METRE space, restricted to the deep tail. The source of truth
        # is the target mask only; no fabricated depth is ever built, so no gradient flows
        # through a constant. The threshold is fixed, or the deep_quantile of the WET cells
        # so the mask cannot collapse onto shallow water.
        l_deep = pred_h.new_tensor(0.0)
        if self.w_deep > 0:
            if self.quantile_per_sample:
                deep_terms = []
                for i in range(target.shape[0]):
                    mi = mask_b[i]
                    vals = target[i, 0][mi]
                    wet_vals = vals[vals > self.wet_threshold]
                    if self.deep_quantile is not None and wet_vals.numel() > 1:
                        thr = torch.quantile(wet_vals, self.deep_quantile)
                        thr = torch.maximum(thr, thr.new_tensor(self.deep_threshold))
                    else:
                        thr = vals.new_tensor(self.deep_threshold)
                    dm = mi & (target[i, 0] >= thr)
                    if dm.any():
                        de = F.huber_loss(pred_h[i:i + 1], target[i:i + 1],
                                          reduction="none", delta=self.deep_delta)
                        deep_terms.append(_masked_reduce(de, dm[None]))
                l_deep = torch.stack(deep_terms).mean() if deep_terms else pred_h.new_tensor(0.0)
            else:
                with torch.no_grad():
                    valid_vals = target[:, 0][mask_b]
                    wet_vals = valid_vals[valid_vals > self.wet_threshold]
                    if self.deep_quantile is not None and wet_vals.numel() > 1:
                        deep_thr = torch.quantile(wet_vals, self.deep_quantile)
                        deep_thr = torch.maximum(deep_thr, deep_thr.new_tensor(self.deep_threshold))
                    else:
                        deep_thr = target.new_tensor(self.deep_threshold)
                deep_mask = mask_b & (target[:, 0] >= deep_thr)
                if deep_mask.any():
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
    """Masked L1 on depth, with an OPTIONAL supervised wet term.

    The wet head is only trained when ``w_wet > 0``. The earlier version ignored
    ``wet_logits`` entirely, so any ablation that enabled the wet head while
    using this loss left that head with no gradient, which made the comparison
    meaningless. ``w_wet`` defaults to 0 so existing depth-only runs and their
    logs remain valid; ablations that intend to test the wet head must set it.
    """

    def __init__(
        self,
        wet_threshold: float = 0.05,
        w_wet: float = 0.0,
        focal_alpha: float = 0.25,
        focal_alpha_mode: str = "class",
    ) -> None:
        super().__init__()
        self.wet_threshold = wet_threshold
        self.w_wet = w_wet
        self.focal_alpha = focal_alpha
        self.focal_alpha_mode = focal_alpha_mode

    def forward(self, output: dict, target: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = output["depth"]
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)
        mask_b = mask.bool()
        if mask_b.ndim == 4:
            mask_b = mask_b[:, 0]

        loss = masked_l1(pred, target, mask_b)

        l_wet = pred.new_tensor(0.0)
        if self.w_wet > 0 and output.get("wet_logits") is not None:
            wet = (target[:, 0] > self.wet_threshold).float()
            l_wet = masked_focal_bce(
                output["wet_logits"], wet, mask_b,
                alpha=self.focal_alpha, alpha_mode=self.focal_alpha_mode,
            )
            loss = loss + self.w_wet * l_wet

        return {"total": loss, "l_depth": loss.detach(), "l_wet": l_wet.detach()}
