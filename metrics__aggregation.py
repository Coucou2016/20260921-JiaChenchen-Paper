"""Metric aggregation across batches / tiles.

Two DIFFERENT estimands live in this module and they must never be conflated:

``FloodMetricAccumulator``  -> domain-pooled metrics.
    Pixels are pooled across the whole split before the metric is formed, e.g.
    ``RMSE = sqrt(sum(err^2) / n_valid)`` and ``CSI = TP / (TP+FP+FN)`` over all
    valid pixels. This is the headline physical performance number.

``average_metrics``         -> tile-macro metrics.
    The metric is formed per tile and then averaged over tiles, giving every
    tile equal weight regardless of its wet area. Useful for spatial
    distribution and per-tile statistics, but it is NOT the same quantity as a
    domain-pooled metric and large/ small wet tiles are weighted differently.

With ``batch_size = 1`` the previous code silently reported tile-macro values
under the names used for domain metrics. Callers should now name which of the
two they want.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable

import torch


# --------------------------------------------------------------------------- #
# tile-macro
# --------------------------------------------------------------------------- #
def average_metrics(dicts: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """Tile-macro average: mean of per-tile metrics, skipping NaN entries.

    A NaN is dropped rather than treated as zero, so an undefined metric (for
    example CSI on a tile with no water in truth or prediction) does not drag
    the macro average down. The number of contributing tiles is returned under
    ``<name>__n`` so that a metric averaged over few tiles can be told apart.
    """
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, float) and (v != v):  # NaN
                continue
            sums[k] += float(v)
            counts[k] += 1
    out = {k: sums[k] / counts[k] for k in sums if counts[k] > 0}
    for k in sums:
        out[f"{k}__n"] = float(counts[k])
    return out


# --------------------------------------------------------------------------- #
# domain-pooled
# --------------------------------------------------------------------------- #
@dataclass
class FloodMetricAccumulator:
    """Accumulate flood metrics over the whole split (domain-pooled).

    All quantities are summed as raw pixel counts so the metric is identical to
    concatenating every batch and computing it once. ``test_global_rmse_accumulator_matches_concat``
    asserts this equivalence.

    Thresholds are configurable; the defaults match the rest of the project
    (CSI at 0.05, 0.30 and 1.00 m of depth).
    """

    wet_thr: float = 0.05
    thresholds: tuple[float, ...] = (0.05, 0.30, 1.00)
    deep_bins: tuple[float, ...] = (1.0, 2.0, 3.0)

    # accumulator state (pixel sums)
    sse_all: float = 0.0
    n_all: int = 0
    sse_wet: float = 0.0
    sae_wet: float = 0.0
    n_wet: int = 0
    pred_volume: float = 0.0
    true_volume: float = 0.0
    pred_area: float = 0.0
    true_area: float = 0.0
    _tp: dict = field(default_factory=dict)
    _fp: dict = field(default_factory=dict)
    _fn: dict = field(default_factory=dict)
    _peak_err_sum: float = 0.0
    _peak_err_n: int = 0
    _bias_sum: dict = field(default_factory=dict)
    _bias_n: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._tp = {t: 0.0 for t in self.thresholds}
        self._fp = {t: 0.0 for t in self.thresholds}
        self._fn = {t: 0.0 for t in self.thresholds}
        self._bias_sum = {b: 0.0 for b in self.deep_bins}
        self._bias_n = {b: 0 for b in self.deep_bins}

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _split(pred, target, mask):
        p = pred[:, 0] if pred.ndim == 4 else pred
        t = target[:, 0] if target.ndim == 4 else target
        m = mask.bool()
        if m.ndim == 4:
            m = m[:, 0]
        return p, t, m

    # -- update ------------------------------------------------------------ #
    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        p, t, m = self._split(pred, target, mask)
        if m.sum() == 0:
            return
        err = p[m] - t[m]
        self.sse_all += float((err * err).sum())
        self.n_all += int(m.sum())

        wet = m & (t > self.wet_thr)
        ew = p[wet] - t[wet]
        self.sse_wet += float((ew * ew).sum())
        self.sae_wet += float(ew.abs().sum())
        self.n_wet += int(wet.sum())

        for thr in self.thresholds:
            pb = p > thr
            tb = t > thr
            self._tp[thr] += float((pb & tb & m).sum())
            self._fp[thr] += float((pb & ~tb & m).sum())
            self._fn[thr] += float((~pb & tb & m).sum())

        pv = p[m].clamp_min(0.0)
        tv = t[m].clamp_min(0.0)
        self.pred_volume += float(pv.sum())
        self.true_volume += float(tv.sum())
        self.pred_area += float((p[m] > self.wet_thr).sum())
        self.true_area += float((t[m] > self.wet_thr).sum())

        # per-sample peak depth error, so a batched call equals per-sample calls
        for i in range(p.shape[0]):
            mi = m[i]
            if mi.sum() == 0:
                continue
            self._peak_err_sum += float((p[i][mi].max() - t[i][mi].max()).abs())
            self._peak_err_n += 1

        # mean signed bias in deep bins, to expose the deep-water problem directly
        for b in self.deep_bins:
            db = m & (t >= b)
            if db.any():
                self._bias_sum[b] += float((p[db] - t[db]).sum())
                self._bias_n[b] += int(db.sum())

    # -- read out ---------------------------------------------------------- #
    def compute(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            "RMSE_all_domain": math.sqrt(self.sse_all / max(self.n_all, 1)),
            "RMSE_wet_domain": math.sqrt(self.sse_wet / max(self.n_wet, 1)),
            "MAE_wet_domain": self.sae_wet / max(self.n_wet, 1),
            "n_valid_pixels": float(self.n_all),
            "n_wet_pixels": float(self.n_wet),
            # Bias keeps its sign; relative error is a magnitude.
            "VolumeBias_domain": (self.pred_volume - self.true_volume)
            / max(self.true_volume, 1e-12),
            "VolumeRelativeError_domain": abs(self.pred_volume - self.true_volume)
            / max(self.true_volume, 1e-12),
            "AreaBias_domain": (self.pred_area - self.true_area)
            / max(self.true_area, 1e-12),
            "PeakDepthError_domain": self._peak_err_sum / max(self._peak_err_n, 1),
        }
        for thr in self.thresholds:
            tp, fp, fn = self._tp[thr], self._fp[thr], self._fn[thr]
            den = tp + fp + fn
            # Undefined (no water in truth or prediction), not "zero skill".
            csi = float("nan") if den == 0 else tp / den
            f1 = float("nan") if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
            out[f"CSI_{thr:.2f}_domain"] = csi
            out[f"F1_{thr:.2f}_domain"] = f1
        for b in self.deep_bins:
            out[f"Bias_{b:.0f}m_domain"] = (
                self._bias_sum[b] / self._bias_n[b] if self._bias_n[b] > 0 else float("nan")
            )
        return out


def pooled_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                   accumulator: FloodMetricAccumulator | None = None) -> Dict[str, float]:
    """Convenience: domain-pooled metrics for one already-whole tensor."""
    acc = accumulator or FloodMetricAccumulator()
    acc.update(pred, target, mask)
    return acc.compute()
