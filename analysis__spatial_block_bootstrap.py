"""Spatial block bootstrap for the 242 held-out test tiles (fixes iid bootstrap).

The test set is a spatial band held out within Wellington, and the report itself
measures substantial spatial autocorrelation in the residual field
(see the Moran's I section). Resampling tiles with replacement as if they were
exchangeable independent units therefore understates the uncertainty of any
test-set average, and the resulting confidence intervals are too narrow.

This module resamples BLOCKS of adjacent tiles instead. Blocks are formed on the
tile grid (iy, ix) and, because the two rainfall scenarios are separate
simulations, a scenario is never mixed inside one block. Block size is a
sensitivity parameter: a conclusion is only credible if it holds across a range
of plausible block sizes, so ``block_size_sensitivity`` sweeps 1x1 (the old iid
behaviour) up to 4x4.

Input is the per-tile metric table the project already writes, e.g.
``outputs/.../per_tile_metrics_test.json`` or the per-tile block inside
``outputs/v0_10m2m_hmax/visualizations/diagnostics_test.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class BlockBootstrapResult:
    value_col: str
    block_rows: int
    block_cols: int
    n_tiles: int
    n_blocks: int
    mean: float
    ci95_low: float
    ci95_median: float
    ci95_high: float
    ci_width: float
    design: str = "spatial block bootstrap; scenario never crossed within a block"

    def to_dict(self) -> dict:
        return asdict(self)


def add_spatial_blocks(df: pd.DataFrame, block_rows: int, block_cols: int) -> pd.DataFrame:
    """Attach a cluster id = scenario + block, keeping scenarios separate."""
    out = df.copy()
    out["block_y"] = out["iy"] // block_rows
    out["block_x"] = out["ix"] // block_cols
    out["cluster"] = (
        out["scenario"].astype(str) + "_"
        + out["block_y"].astype(str) + "_"
        + out["block_x"].astype(str)
    )
    return out


def spatial_block_bootstrap(
    df: pd.DataFrame,
    value_col: str,
    block_rows: int = 2,
    block_cols: int = 2,
    n_boot: int = 10000,
    seed: int = 42,
) -> BlockBootstrapResult:
    dfb = add_spatial_blocks(df, block_rows, block_cols)
    vals = dfb[value_col].to_numpy(dtype=float)
    keep = np.isfinite(vals)
    dfb, vals = dfb[keep], vals[keep]
    if len(vals) == 0:
        raise ValueError(f"no finite values in {value_col}")

    clusters = dfb["cluster"].unique()
    grouped = {c: dfb.loc[dfb.cluster == c, value_col].to_numpy(dtype=float)
               for c in clusters}

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        x = np.concatenate([grouped[c] for c in chosen])
        draws[b] = np.nanmean(x)

    lo, med, hi = np.quantile(draws, [0.025, 0.5, 0.975])
    return BlockBootstrapResult(
        value_col=value_col,
        block_rows=block_rows,
        block_cols=block_cols,
        n_tiles=int(len(vals)),
        n_blocks=int(len(clusters)),
        mean=float(np.nanmean(vals)),
        ci95_low=float(lo),
        ci95_median=float(med),
        ci95_high=float(hi),
        ci_width=float(hi - lo),
    )


def block_size_sensitivity(
    df: pd.DataFrame,
    value_col: str,
    sizes: Sequence[tuple[int, int]] = ((1, 1), (2, 2), (3, 3), (4, 4)),
    n_boot: int = 10000,
    seed: int = 42,
) -> list[BlockBootstrapResult]:
    """Sweep block sizes; 1x1 reproduces the iid bootstrap for reference."""
    return [spatial_block_bootstrap(df, value_col, br, bc, n_boot=n_boot, seed=seed)
            for br, bc in sizes]


def load_per_tile_table(path: str | Path, source: str = "auto") -> pd.DataFrame:
    """Load a per-tile metric table into a DataFrame with iy/ix/scenario.

    Accepts either a flat list of per-tile dicts (``per_tile_metrics_test.json``)
    or the ``diagnostics_test.json`` form, whose entries nest metrics under
    ``model`` / ``bilinear``. For the nested form, pass ``source="model"`` or
    ``source="bilinear"`` to choose which block to flatten.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "per_tile" in data:
        rows = data["per_tile"]
        if source == "auto":
            source = "model"
        flat = []
        for r in rows:
            base = {k: r[k] for k in ("index", "scenario", "iy", "ix") if k in r}
            metrics = r.get(source, {}) if source in r else r
            base.update({k: v for k, v in metrics.items()
                         if isinstance(v, (int, float)) and k not in base})
            flat.append(base)
        return pd.DataFrame(flat)

    rows = data["per_tile"] if isinstance(data, dict) else data
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--source", default="auto", choices=["auto", "model", "bilinear"])
    ap.add_argument("--metrics", nargs="+",
                    default=["MAE_wet", "RMSE_wet", "CSI_005", "CSI_100"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    df = load_per_tile_table(a.table, a.source)
    print(f"loaded {len(df)} tiles; columns include iy/ix/scenario: "
          f"{set(['iy','ix','scenario']).issubset(df.columns)}")
    report = {}
    for m in a.metrics:
        if m not in df.columns:
            print(f"  skip {m}: not present")
            continue
        res = block_size_sensitivity(df, m)
        for r in res:
            print(f"{m:12s} block {r.block_rows}x{r.block_cols} "
                  f"n_blocks={r.n_blocks:4d} mean={r.mean:+.5f} "
                  f"CI=[{r.ci95_low:+.5f},{r.ci95_high:+.5f}] width={r.ci_width:.5f}")
        report[m] = [r.to_dict() for r in res]
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("wrote", a.out)
