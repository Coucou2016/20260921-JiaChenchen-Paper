#!/usr/bin/env python
"""Finalize chain for Experiment A / B, GPU-serialised behind the training jobs.

Experiment B is already launched with ``--wait-pid`` against Experiment A's PID, so
this script only has to wait for B to exit, then run the FINAL evaluations and the
figure assembly IN ORDER on the single GPU.  Nothing here starts a second GPU job:
the eval / diagnostic stages run strictly after both trainings have exited.

Why a chain and not three commands: the residual checkpoint keeps improving while
training runs, so any evaluation done mid-training is stale.  This script re-scores
the residual arm from the finished ``best_csi.pt`` and then scores the E7 arm.

Usage
-----
    python -u scripts/chain_exp_ab_finalize.py --wait-pid 83844
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG = ROOT / "outputs" / "premodel" / "finalize_chain.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                         capture_output=True, text=True, errors="ignore").stdout
    return str(pid) in out


def run(cmd: list[str], tag: str) -> int:
    log(f"RUN {tag}: {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    log(f"END {tag}: rc={rc} sec={time.time()-t0:.0f}")
    return rc


MIRROR = Path(r"E:\Projects\20260921-JiaChenchen-Paper-github")


def _git(*args: str, cwd: Path = MIRROR) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                          text=True, errors="ignore")


def publish(attempts: int = 14) -> None:
    """Commit the refreshed mirror and push with linear backoff.

    GitHub connectivity from this box is flaky (port 443 refused, empty reply,
    remote 500).  A push here has previously needed ~8 tries, so retry in a loop
    and never force-push.
    """
    if not MIRROR.exists():
        log(f"mirror {MIRROR} missing; skipping publish")
        return
    _git("config", "http.postBuffer", "524288000")
    _git("config", "http.version", "HTTP/1.1")
    _git("add", "-A")
    st = _git("status", "--porcelain")
    if not st.stdout.strip():
        log("mirror already clean; nothing to commit")
    else:
        msg = ("Experiments A/B: bilinear-residual SR + E7 deep-objective rework; "
               "report sections 4.7/5.12/5.13/8.5, figs 86-88, RERUN checklist")
        c = _git("commit", "-m", msg)
        log(f"git commit rc={c.returncode} {(c.stdout or c.stderr).strip()[:200]}")

    for i in range(1, attempts + 1):
        p = _git("push", "origin", "HEAD")
        if p.returncode == 0:
            head = _git("rev-parse", "HEAD").stdout.strip()
            log(f"git push OK on attempt {i}; HEAD={head}")
            return
        err = (p.stderr or p.stdout).strip().replace("\n", " ")[:200]
        wait = min(20 * i, 120)
        log(f"push attempt {i} failed: {err} ; retry in {wait}s")
        time.sleep(wait)
    log("git push FAILED after all attempts (see above errors)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, default=83844,
                    help="PID of the Experiment B training job")
    ap.add_argument("--timeout-h", type=float, default=8.0)
    args = ap.parse_args()

    if args.wait_pid:
        log(f"waiting for Experiment B pid={args.wait_pid} to exit (single GPU)")
        t0 = time.time()
        while alive(args.wait_pid):
            if (time.time() - t0) / 3600 > args.timeout_h:
                log(f"timeout waiting for pid={args.wait_pid}; aborting chain")
                return
            time.sleep(30)
        log(f"pid={args.wait_pid} gone after {(time.time()-t0)/60:.1f} min")

    # A stale mid-training evaluation must not satisfy run_exp_ab_evals' skip rule.
    for stale in ("outputs/premodel/exp_ab/residual_depth_val.json",
                  "outputs/premodel/exp_ab/E7_continuous_val.json"):
        p = ROOT / stale
        if p.exists():
            p.unlink()
            log(f"removed stale {stale}")

    # Final, GPU-serialised evaluations (all arms, same protocol).
    run([PY, "-X", "utf8", "-u", "scripts/run_exp_ab_evals.py",
         "--split", "val", "--device", "cuda"], "evals_val")

    # Mechanism diagnostic on the finished checkpoints.
    run([PY, "-X", "utf8", "-u", "scripts/diag_deep_slope.py",
         "--split", "val", "--device", "cuda"], "deep_slope")

    # Comparison table + the three figures.
    run([PY, "-X", "utf8", "-u", "scripts/report_exp_ab.py"], "report_ab")

    # Report renders, in the documented order, then the xref gate.
    for script, tag in (("scripts/build_report.py", "build_html"),
                        ("scripts/build_markdown.py", "build_md"),
                        ("scripts/build_brief_report.py", "build_brief_html"),
                        ("scripts/build_brief_markdown.py", "build_brief_md"),
                        ("scripts/_verify_xref.py", "xref")):
        run([PY, "-X", "utf8", script], tag)

    for script, tag in (("scripts/build_pdf.py", "build_pdf"),
                        ("scripts/build_brief_pdf.py", "build_brief_pdf"),
                        ("scripts/build_flat_mirror.py", "flat_mirror")):
        run([PY, "-X", "utf8", script], tag)

    publish()
    log("CHAIN DONE (build + mirror + git publish finished)")


if __name__ == "__main__":
    main()
