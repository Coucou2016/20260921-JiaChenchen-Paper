#!/usr/bin/env python
"""One-command visual refresh: training curves + tile reconstructions + evolution.

Designed to be run repeatedly (by hand or from a loop) while training proceeds.
Every stage is independent and failure-tolerant: if one stage fails or has no
data yet, the others still run.

  python scripts/visualize_all.py                 # auto: uses best_csi.pt
  python scripts/visualize_all.py --ckpt last.pt  # use the newest weights
  python scripts/visualize_all.py --full          # also scan the whole split
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "v0_10m2m_hmax"
LOG = OUT_DIR / "visualizations" / "refresh.log"


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(name: str, cmd: list[str]) -> bool:
    log(f"--- {name} ---")
    log(" ".join(cmd))
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    except Exception as exc:  # noqa: BLE001
        log(f"{name} FAILED to launch: {exc}")
        return False
    for stream, tag in ((proc.stdout, "  | "), (proc.stderr, "  ! ")):
        for line in (stream or "").splitlines():
            if line.strip():
                log(tag + line)
    ok = proc.returncode == 0
    log(f"{name} {'OK' if ok else 'FAILED'} in {time.time()-t0:.0f}s (exit={proc.returncode})")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint path or name inside outputs/v0_10m2m_hmax")
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--max-tiles", type=int, default=60)
    ap.add_argument("--n-representative", type=int, default=4)
    ap.add_argument("--full", action="store_true", help="scan every tile instead of the first --max-tiles")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--skip-evolution", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    py = sys.executable

    # 1) training dynamics (always cheap and always available)
    run("training curve", [py, "-u", str(Path("viz") / "plot_training.py")])

    # 2) tile reconstruction at the current checkpoint
    ckpt = args.ckpt
    if ckpt is None:
        ckpt = str(OUT_DIR / "best_csi.pt")
    elif not Path(ckpt).is_absolute():
        ckpt = str(OUT_DIR / ckpt)

    tile_cmd = [
        py, "-u", "scripts/visualize_tiles.py",
        "--ckpt", ckpt,
        "--splits", *args.splits,
        "--n-representative", str(args.n_representative),
        "--threads", str(args.threads),
    ]
    if not args.full:
        tile_cmd += ["--max-tiles", str(args.max_tiles)]
    if args.tag:
        tile_cmd += ["--tag", args.tag]
    run("tile reconstruction", tile_cmd)

    # 3) evolution across snapshots (needs >= 2 checkpoints; fails soft otherwise)
    if not args.skip_evolution:
        run("training evolution", [
            py, "-u", "scripts/visualize_evolution.py",
            "--split", args.splits[0],
            "--threads", str(args.threads),
        ])

    log(f"done; figures under {OUT_DIR / 'visualizations'}")


if __name__ == "__main__":
    main()
