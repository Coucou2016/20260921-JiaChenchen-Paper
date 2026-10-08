#!/usr/bin/env python
"""Run the Experiment A/B headline evaluation, one model at a time, GPU-serialized.

Every model is scored on the SAME validation split with the SAME two calibers
(tile-macro and domain-pooled) via ``scripts/eval_both_calibers.py``, so the rows of
the comparison table are produced by one code path.

Reference rows (nearest, bilinear, frozen ep180, w00 ep200, w01 ep187) are evaluated
fresh rather than copied from older JSONs, so all rows come from one protocol.

Usage
-----
    python -u scripts/run_exp_ab_evals.py            # val only (default)
    python -u scripts/run_exp_ab_evals.py --split test
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT_DIR = ROOT / "outputs" / "premodel" / "exp_ab"
CONFIG_V0 = "configs/v0_10m2m_hmax.yaml"
CONFIG_V1 = "configs/v1_bilinear_residual.yaml"

# (label, config, model, ckpt or None)
MODELS = [
    ("nearest", CONFIG_V0, "nearest", None),
    ("bilinear", CONFIG_V0, "bilinear", None),
    ("frozen_ep180", CONFIG_V0, "hydrogeo_srno", "outputs/v0_10m2m_hmax/last.pt"),
    ("w00_ep200", CONFIG_V0, "hydrogeo_srno", "outputs/finetune_deep/w00/last.pt"),
    ("w01_ep187", CONFIG_V0, "hydrogeo_srno", "outputs/finetune_deep/w01/snapshots/ep0187.pt"),
    ("residual_depth", CONFIG_V1, "bilinear_residual",
     "outputs/v1_bilinear_residual/depth/best_csi.pt"),
    ("residual_nearest", CONFIG_V1, "nearest_residual",
     "outputs/v1_bilinear_residual/nearest/best_csi.pt"),
    ("E7_continuous", CONFIG_V0, "hydrogeo_srno",
     "outputs/deep_objective/E7/best_pooled_rmse.pt"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="cpu lets the reference rows be scored while GPU training runs")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for label, cfg, model, ckpt in MODELS:
        out = OUT_DIR / f"{label}_{args.split}.json"
        if out.exists():
            print(f"[skip] {label} already has {out.name}")
            continue
        if ckpt is not None and not (ROOT / ckpt).exists():
            print(f"[MISSING CKPT] {label}: {ckpt} not found -> cannot evaluate")
            continue
        cmd = [PY, "-X", "utf8", "-u", "scripts/eval_both_calibers.py",
               "--config", cfg, "--model", model, "--split", args.split,
               "--device", args.device, "--out", str(out.relative_to(ROOT))]
        if ckpt is not None:
            cmd += ["--ckpt", ckpt]
        print(f"\n=== {time.strftime('%H:%M:%S')} {label}: {' '.join(cmd)}", flush=True)
        log = OUT_DIR / f"{label}_{args.split}.log"
        with log.open("w", encoding="utf-8", errors="ignore") as lf:
            rc = subprocess.run(cmd, cwd=str(ROOT), stdout=lf,
                                stderr=subprocess.STDOUT).returncode
        ok = out.exists()
        print(f"    rc={rc} artefact={'yes' if ok else 'NO'}", flush=True)
    print("\nall evaluations done")


if __name__ == "__main__":
    main()
