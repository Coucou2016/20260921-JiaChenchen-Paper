"""Metric aggregation across batches / tiles."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable


def average_metrics(dicts: Iterable[Dict[str, float]]) -> Dict[str, float]:
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
    return {k: sums[k] / counts[k] for k in sums if counts[k] > 0}
