"""Seed-level inference for the fine-tune comparison (fixes pseudoreplication).

The report previously built its paired test from 20 consecutive fine-tune epochs
of a SINGLE optimisation trajectory. Epoch 1, 2, ... 20 are strongly dependent:
each parameter state conditions the next, and they are not independent
experimental units. A paired t-test with df=19 on those numbers cannot support a
claim of replication, and Cohen's d computed the same way is a standardised
trajectory difference, not a cross-replicate effect size.

The correct unit is the INDEPENDENT RANDOM SEED. This module implements that
analysis. It requires one control and one treatment number PER SEED, produced by
runs that share an initial checkpoint and differ only in the fine-tune objective.

If only one seed exists, the honest output is "no seed-level inference
available"; this module says so rather than silently reusing the epoch trick.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass
class SeedInference:
    metric: str
    n_seeds: int
    mean_delta: float
    sd_delta: float
    ci95_low: float
    ci95_high: float
    t: float
    p: float
    cohen_dz: float
    permutation_p: float
    estimand: str = "paired-seed difference (treatment - control)"

    def to_dict(self) -> dict:
        return asdict(self)


def paired_seed_stats(
    control: Sequence[float],
    treatment: Sequence[float],
    metric: str = "metric",
    n_boot: int = 20000,
    n_perm: int = 20000,
    seed: int = 20261006,
) -> SeedInference:
    """Paired statistics over independent seeds.

    ``control[i]`` and ``treatment[i]`` must be the SAME seed, so the pair shares
    initialisation and data order and the difference isolates the treatment.
    """
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    if c.shape != t.shape:
        raise ValueError("paired arrays must have the same shape")
    if c.ndim != 1 or c.size < 3:
        raise ValueError("need at least 3 independent seeds for a paired test")

    d = t - c
    rng = np.random.default_rng(seed)

    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])

    tt = stats.ttest_rel(t, c)

    # sign-flip permutation test: exact under the sharp null, no normality needed
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    perm = (signs * d).mean(axis=1)
    observed = abs(d.mean())
    perm_p = float((np.sum(np.abs(perm) >= observed) + 1) / (n_perm + 1))

    sd = d.std(ddof=1)
    dz = float(d.mean() / sd) if sd > 0 else float("nan")

    return SeedInference(
        metric=metric,
        n_seeds=int(len(d)),
        mean_delta=float(d.mean()),
        sd_delta=float(sd),
        ci95_low=float(lo),
        ci95_high=float(hi),
        t=float(tt.statistic),
        p=float(tt.pvalue),
        cohen_dz=dz,
        permutation_p=perm_p,
    )


def run_from_manifest(manifest: str | Path, metric: str) -> SeedInference:
    """Manifest format: {"control": {"11": v, "22": v, ...},
                          "treatment": {"11": v, "22": v, ...}}"""
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    ctl, trt = data["control"], data["treatment"]
    seeds = sorted(set(ctl) & set(trt), key=lambda s: int(s))
    if len(seeds) < 3:
        raise ValueError(
            f"only {len(seeds)} shared seed(s) found; seed-level inference needs >= 3. "
            "Epoch-wise trajectory analysis is a diagnostic, not a replicate design."
        )
    return paired_seed_stats(
        [ctl[s] for s in seeds], [trt[s] for s in seeds], metric=metric,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--metric", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run_from_manifest(a.manifest, a.metric)
    print(json.dumps(res.to_dict(), indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
