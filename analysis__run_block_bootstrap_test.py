"""Re-estimate the test-set uncertainty with a spatial block bootstrap.

Data: the per-tile TEST metrics of two fine-tune arms that differ ONLY in the
deep-water loss weight, evaluated on the same 242 held-out tiles
(w0ext: w_deep=0.00 control; w01ext: w_deep=0.10 treatment). Because both arms
were scored on the same tiles, the per-tile difference is a genuine paired
quantity and the ONLY remaining question is how much of the average difference
is sampling noise given that adjacent tiles are spatially correlated.

Outputs a JSON plus a console table showing how the CI width inflates as the
resampling block grows from 1x1 (the old iid assumption) to 4x4.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(r"E:\Projects\20260921-JiaChenchen-Paper")
sys.path.insert(0, str(ROOT))

from analysis.spatial_block_bootstrap import (  # noqa: E402
    add_spatial_blocks, block_size_sensitivity, load_per_tile_table,
)

CTL = ROOT / "outputs/finetune_deep/w0ext/visualizations/ep189/per_tile_metrics_test.json"
TRT = ROOT / "outputs/finetune_deep/w01ext/visualizations/ep189/per_tile_metrics_test.json"
OUT = ROOT / "outputs/premodel/spatial_block_bootstrap.json"

METRICS = ["MAE_wet", "RMSE_wet", "CSI_005", "CSI_030", "CSI_100",
           "VolumeRelativeError", "PeakDepthError"]


def load(path: Path) -> pd.DataFrame:
    df = load_per_tile_table(path)
    df = df.rename(columns={"index": "tile_index"})
    return df


def main() -> None:
    ctl = load(CTL)
    trt = load(TRT)
    key = ["scenario", "iy", "ix"]
    for df, name in ((ctl, "control"), (trt, "treatment")):
        if not set(key).issubset(df.columns):
            raise SystemExit(f"{name} lacks {key}: {list(df.columns)}")

    m = ctl[key + METRICS].merge(trt[key + METRICS], on=key,
                                 suffixes=("_ctl", "_trt"))
    print(f"paired test tiles: {len(m)}  (control {len(ctl)}, treatment {len(trt)})")
    print("tiles per scenario:")
    print(m.groupby("scenario").size().to_string())

    # tile grid extent, to sanity-check the block geometry
    print(f"\niy range {m.iy.min()}..{m.iy.max()}  ix range {m.ix.min()}..{m.ix.max()}")
    nb = add_spatial_blocks(m, 2, 2)
    print(f"2x2 blocks: {nb.cluster.nunique()} distinct clusters")

    report: dict = {
        "source": {
            "control": str(CTL.relative_to(ROOT)).replace("\\", "/"),
            "treatment": str(TRT.relative_to(ROOT)).replace("\\", "/"),
            "note": ("w_deep=0.00 vs w_deep=0.10; same held-out tiles; "
                     "difference is paired by tile"),
        },
        "n_paired_tiles": int(len(m)),
        "metrics": {},
    }

    for metric in METRICS:
        d = (m[f"{metric}_trt"] - m[f"{metric}_ctl"]).to_numpy(dtype=float)
        finite = np.isfinite(d)
        rows = block_size_sensitivity(
            m.assign(_diff=np.where(finite, d, np.nan)), "_diff",
            n_boot=8000, seed=20261006,
        )
        report["metrics"][metric] = {
            "n_finite": int(finite.sum()),
            "raw_mean_diff": float(np.nanmean(d)),
            "block_sensitivity": [r.to_dict() for r in rows],
        }
        print(f"\n== {metric} ==  mean diff = {np.nanmean(d):+.5f}  "
              f"(n={int(finite.sum())})")
        base_w = rows[0].ci_width
        for r in rows:
            infl = r.ci_width / base_w if base_w > 0 else float("nan")
            print(f"   block {r.block_rows}x{r.block_cols}  n_blocks={r.n_blocks:4d}  "
                  f"CI=[{r.ci95_low:+.5f},{r.ci95_high:+.5f}]  "
                  f"width={r.ci_width:.5f}  x{infl:.2f} vs iid")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")

    # headline: median CI inflation from iid to 4x4
    infl_all = []
    for metric, blk in report["metrics"].items():
        rows = blk["block_sensitivity"]
        if rows[0]["ci_width"] > 0:
            infl_all.append(rows[-1]["ci_width"] / rows[0]["ci_width"])
    if infl_all:
        print(f"\nCI width inflation iid -> 4x4 blocks: "
              f"median x{np.median(infl_all):.2f}, "
              f"range x{min(infl_all):.2f}..x{max(infl_all):.2f}")


if __name__ == "__main__":
    main()
