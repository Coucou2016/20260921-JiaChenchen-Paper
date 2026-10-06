#!/usr/bin/env python
"""Post-V0 orchestration: baselines → tables → dynamic → multiscale → residual-LDM.

Run after outputs/v0_10m2m_hmax/best_csi.pt exists.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--skip_dynamic", action="store_true")
    parser.add_argument("--skip_multiscale", action="store_true")
    parser.add_argument("--skip_ldm", action="store_true")
    parser.add_argument("--baseline_epochs", type=int, default=100)
    args = parser.parse_args()

    ckpt = ROOT / "outputs/v0_10m2m_hmax/best_csi.pt"
    if not ckpt.exists():
        raise SystemExit(f"missing {ckpt}; finish V0 training first")

    py = sys.executable
    # Eval M3 on val/test
    run([py, "scripts/run_baseline_table.py", "--split", "val", "--models", "bilinear", "nearest", "hydrogeo_srno", "M3"])
    run([py, "scripts/run_baseline_table.py", "--split", "test", "--models", "bilinear", "nearest", "hydrogeo_srno", "M3"])

    if not args.skip_baselines:
        for m in ("resunet", "edsr", "rswinunet", "srno_single", "srno_dem"):
            run([
                py, "scripts/train_baseline.py", "--model", m,
                "--epochs", str(args.baseline_epochs), "--batch_size", "1",
            ])
            run([py, "scripts/run_baseline_table.py", "--split", "val", "--models", m])
            run([py, "scripts/run_baseline_table.py", "--split", "test", "--models", m])

    if not args.skip_dynamic:
        run([py, "scripts/train_dynamic.py", "--config", "configs/v0_dynamic_h.yaml"])

    if not args.skip_multiscale:
        run([py, "scripts/train_multiscale.py", "--config", "configs/v1_multiscale.yaml"])
        run([py, "scripts/train_multiscale.py", "--config", "configs/v1_arbitrary_scale.yaml"])

    if not args.skip_ldm:
        run([py, "scripts/train_residual_ldm.py", "--det_ckpt", str(ckpt), "--epochs", "50"])

    print("post-V0 pipeline complete")


if __name__ == "__main__":
    main()
