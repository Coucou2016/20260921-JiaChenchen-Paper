"""Repair summary_{split}.json files that were overwritten by --representatives-only runs.

For each visualizations/epNNN/summary_{split}.json whose avg_basis is a full scan (or
missing), recompute avg_metrics from the per-tile table on disk and record the provenance
in a new `avg_basis` field. Idempotent.

Usage:
    python scripts/repair_summaries.py            # fix all epNNN folders
    python scripts/repair_summaries.py --dry-run  # report only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIZ = ROOT / "outputs" / "v0_10m2m_hmax" / "visualizations"

SET_NAMES = [
    "RMSE_all", "RMSE_wet", "MAE_wet",
    "CSI_005", "F1_005", "CSI_030", "F1_030", "CSI_100", "F1_100",
    "PeakDepthError", "FloodAreaRelativeError", "VolumeRelativeError",
    "PSNR", "SSIM",
]


def recompute(table_path: Path) -> dict[str, float]:
    rows = json.loads(table_path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for k in SET_NAMES:
        v = [float(r[k]) for r in rows if k in r and r[k] == r[k]]
        if v:
            out[k] = float(np.nanmean(v))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fixed = skipped = 0
    for tag_dir in sorted(p for p in VIZ.glob("ep*") if p.is_dir()):
        for split in ("val", "test"):
            table = tag_dir / f"per_tile_metrics_{split}.json"
            summary = tag_dir / f"summary_{split}.json"
            if not table.exists() or not summary.exists():
                continue
            try:
                s = json.loads(summary.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            basis = str(s.get("avg_basis", ""))
            if basis.startswith("full_scan"):
                skipped += 1
                continue

            scan_avg = recompute(table)
            if not scan_avg:
                continue
            old = s.get("avg_metrics", {})
            note = ""
            if "CSI_005" in old and "CSI_005" in scan_avg and abs(old["CSI_005"] - scan_avg["CSI_005"]) > 1e-3:
                note = f"  (was CSI_005={old['CSI_005']:.4f})"
            print(f"{tag_dir.name}/{split}: n_tiles_scanned={s.get('n_tiles_scanned')} "
                  f"-> recomputed from {len(json.loads(table.read_text(encoding='utf-8')))} scanned tiles{note}")
            fixed += 1
            if args.dry_run:
                continue
            s["avg_metrics"] = scan_avg
            s["avg_basis"] = f"full_scan_{s.get('n_tiles_scanned') or len(scan_avg)}_tiles"
            summary.write_text(json.dumps(s, indent=2), encoding="utf-8")

    print(f"\n{'would fix' if args.dry_run else 'fixed'}: {fixed}  already-correct: {skipped}")


if __name__ == "__main__":
    main()
