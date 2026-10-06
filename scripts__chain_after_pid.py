"""Run a command once another process has exited.

Used to serialise GPU jobs on the single 4 GB GTX 950M: wait for the baseline evaluation to
finish, then start the loss sweep, without any manual polling.

Usage:
    python scripts/chain_after_pid.py --pid 38228 -- poll-seconds 20 -- \
        python -u scripts/run_deep_sweep.py --only D0 D2 --epochs 40 --patience 40
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def alive(pid: int) -> bool:
    """Windows-safe liveness check (tasklist avoids needing psutil)."""
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                         capture_output=True, text=True, errors="ignore").stdout
    return str(pid) in out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True, help="wait for this PID to exit")
    ap.add_argument("--poll-seconds", type=int, default=20)
    ap.add_argument("--timeout-h", type=float, default=4.0)
    ap.add_argument("--log", default="outputs/chain_after_pid.log")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="command to run after the PID exits (use -- to separate)")
    args = ap.parse_args()

    cmd = [c for c in args.cmd if c != "--"]
    if not cmd:
        raise SystemExit("no command given; put it after --")

    log_path = ROOT / args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def note(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] chain: {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    note(f"waiting for pid={args.pid} to exit")
    t0 = time.time()
    while alive(args.pid):
        if (time.time() - t0) / 3600 > args.timeout_h:
            note(f"timeout after {args.timeout_h} h; not starting the command")
            return
        time.sleep(max(args.poll_seconds, 5))

    note(f"pid={args.pid} gone after {(time.time()-t0)/60:.1f} min; starting: {' '.join(cmd)}")
    with (ROOT / "outputs" / "chained_stdout.log").open("a", encoding="utf-8",
                                                        errors="ignore") as lf:
        lf.write(f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} chained run =====\n")
        lf.flush()
        code = subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf,
                                stderr=subprocess.STDOUT).wait()
    note(f"chained command exited with code {code}")


if __name__ == "__main__":
    main()
