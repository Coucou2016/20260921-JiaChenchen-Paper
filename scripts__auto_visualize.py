#!/usr/bin/env python
"""Periodically refresh all visualizations while training runs.

Each cycle:
  1. regenerate training curves (cheap, every cycle)
  2. render representative tiles at the current best checkpoint
  3. every N cycles, re-render the full-split metric scan and evolution figures

Runs entirely on CPU and never touches the trainer.

  python scripts/auto_visualize.py --interval 1200
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Same V0_OUT_DIR / V0_TAG override as viz/common.py, so the daemon can watch a sweep arm or a
# resumed run. Child scripts inherit the env var because we set it below.
if os.environ.get("V0_OUT_DIR"):
    OUT_DIR = Path(os.environ["V0_OUT_DIR"])
    if not OUT_DIR.is_absolute():
        OUT_DIR = ROOT / OUT_DIR
elif os.environ.get("V0_TAG"):
    OUT_DIR = ROOT / "outputs" / os.environ["V0_TAG"]
else:
    OUT_DIR = ROOT / "outputs" / "v0_10m2m_hmax"
os.environ["V0_OUT_DIR"] = str(OUT_DIR)  # children (visualize_*.py, build_gallery.py) reuse it
VIZ = OUT_DIR / "visualizations"
LOG = VIZ / "auto_visualize.log"
PID = VIZ / "auto_visualize.pid"


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    VIZ.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def best_epoch() -> int:
    p = OUT_DIR / "best_summary.json"
    if not p.exists():
        return 0
    try:
        import json
        return int(json.loads(p.read_text(encoding="utf-8")).get("last_epoch", 0) or 0)
    except Exception:  # noqa: BLE001
        return 0


def _discover_trainer_pid() -> int | None:
    """Find a live train_fixed.py process whose --out_dir matches OUT_DIR.

    Sweep arms launched by run_deep_sweep.py (and some resumed runs) never write
    `<out_dir>/train.pid`, which used to make this daemon exit after a single cycle
    while training was still running. Fall back to inspecting live command lines.
    """
    if os.name != "nt":
        return None
    # The trainer may be invoked with an absolute --out_dir or a workspace-relative
    # one, so accept either spelling when matching the live command line.
    needles = {str(OUT_DIR).replace("/", "\\").lower()}
    try:
        needles.add(str(OUT_DIR.relative_to(ROOT)).replace("/", "\\").lower())
    except ValueError:
        pass
    try:
        raw = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get",
             "ProcessId,CommandLine", "/format:csv"],
            capture_output=True, text=True, errors="ignore",
        ).stdout
    except Exception:  # noqa: BLE001
        return None
    import csv as _csv
    import io as _io
    for row in _csv.reader(_io.StringIO(raw)):
        if len(row) < 3:
            continue
        cmd, pid_s = row[-2], row[-1]  # wmic csv columns: Node,CommandLine,ProcessId
        if "train_fixed.py" not in cmd:
            continue
        hay = cmd.replace("/", "\\").lower()
        if not any(n in hay for n in needles):
            continue
        try:
            return int(pid_s.strip())
        except ValueError:
            continue
    return None


def trainer_alive() -> bool:
    """True only if the PID file points at a live process running train_fixed.py."""
    p = OUT_DIR / "train.pid"
    pid = None
    if p.exists():
        try:
            pid = int(p.read_text(encoding="ascii").strip())
        except Exception:  # noqa: BLE001
            pid = None
    if pid is None:
        pid = _discover_trainer_pid()
        if pid is None:
            return False
        try:  # self-heal so subsequent cycles stay cheap
            p.write_text(str(pid), encoding="ascii")
            log(f"train.pid missing; discovered live trainer pid={pid} in {OUT_DIR.name}")
        except OSError:
            pass
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, errors="ignore",
        ).stdout
        if str(pid) not in out:
            return False
        # guard against PID reuse: verify the command line is the trainer
        try:
            import csv as _csv
            import io as _io
            raw = subprocess.run(
                ["wmic", "process", "where", f"ProcessId={pid}", "get", "CommandLine", "/format:csv"],
                capture_output=True, text=True, errors="ignore",
            ).stdout
            rows = [r for r in _csv.reader(_io.StringIO(raw)) if len(r) > 1 and r[-1].strip()]
            return any("train_fixed.py" in r[-1] for r in rows)
        except Exception:  # noqa: BLE001
            return True  # PID alive and wmic unavailable: accept
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def cycle(n: int, args) -> None:
    py = sys.executable
    subprocess.run([py, "-u", "viz/plot_training.py"], cwd=ROOT, capture_output=True, text=True,
                   encoding="utf-8", errors="ignore")
    log(f"cycle {n}: training curves refreshed")

    # representative tiles at the current best checkpoint — cheap enough each cycle
    subprocess.run(
        [py, "-u", "scripts/visualize_tiles.py", "--representatives-only",
         "--splits", "val", "--n-representative", str(args.n_representative),
         "--threads", str(args.threads)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="ignore",
    )
    log(f"cycle {n}: representative tiles rendered (best epoch ~{best_epoch()})")

    if n % args.full_every == 0:
        subprocess.run(
            [py, "-u", "scripts/visualize_tiles.py", "--splits", "val", "test",
             "--max-tiles", str(args.max_tiles), "--threads", str(args.threads)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="ignore",
        )
        log(f"cycle {n}: full split scan rendered")
        subprocess.run(
            [py, "-u", "viz/plot_domain.py"], cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="ignore",
        )
        log(f"cycle {n}: domain maps refreshed")
        subprocess.run(
            [py, "-u", "scripts/visualize_diagnostics.py", "--split", "val",
             "--threads", str(args.threads)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="ignore",
        )
        log(f"cycle {n}: physical diagnostics refreshed")
        subprocess.run(
            [py, "-u", "scripts/visualize_evolution.py", "--split", "val",
             "--threads", str(args.threads)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="ignore",
        )
        log(f"cycle {n}: evolution montage rendered")
        subprocess.run(
            [py, "-u", "scripts/visualize_bias_evolution.py", "--split", "val",
             "--tiles", "30", "--threads", str(args.threads)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="ignore",
        )
        log(f"cycle {n}: deep-water bias evolution rendered")

    subprocess.run([py, "-u", "scripts/build_gallery.py"], cwd=ROOT, capture_output=True,
                   text=True, encoding="utf-8", errors="ignore")
    log(f"cycle {n}: HTML gallery rebuilt")


def final_pass(args) -> None:
    """Exhaustive refresh: every tile of every split, both diagnostics, evolution, gallery."""
    py = sys.executable
    log("FINAL PASS: exhaustive refresh starting")
    subprocess.run([py, "-u", "viz/plot_training.py"], cwd=ROOT, capture_output=True,
                   text=True, encoding="utf-8", errors="ignore")
    log("final: training curves refreshed")

    for split in ("val", "test"):
        subprocess.run(
            [py, "-u", "scripts/visualize_tiles.py", "--splits", split,
             "--n-representative", str(args.n_representative),
             "--threads", str(args.threads)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="ignore",
        )
        log(f"final: full {split} tile scan rendered")
        subprocess.run(
            [py, "-u", "scripts/visualize_diagnostics.py", "--split", split,
             "--threads", str(args.threads)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="ignore",
        )
        log(f"final: {split} physical diagnostics refreshed")

    subprocess.run([py, "-u", "viz/plot_domain.py"], cwd=ROOT, capture_output=True,
                   text=True, encoding="utf-8", errors="ignore")
    log("final: domain maps refreshed")
    subprocess.run([py, "-u", "scripts/visualize_evolution.py", "--split", "val",
                    "--threads", str(args.threads)],
                   cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    log("final: evolution montage rendered")
    subprocess.run([py, "-u", "scripts/visualize_bias_evolution.py", "--split", "val",
                    "--tiles", "50", "--threads", str(args.threads)],
                   cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    log("final: deep-water bias evolution rendered")
    subprocess.run([py, "-u", "scripts/build_gallery.py"], cwd=ROOT, capture_output=True,
                   text=True, encoding="utf-8", errors="ignore")
    log("FINAL PASS: HTML gallery rebuilt; exhaustive refresh complete")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=1200, help="seconds between cycles")
    ap.add_argument("--full-every", type=int, default=6, help="do the full scan every N cycles")
    ap.add_argument("--max-tiles", type=int, default=60)
    ap.add_argument("--n-representative", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-cycles", type=int, default=1000)
    ap.add_argument("--final-only", action="store_true",
                    help="run the exhaustive pass once and exit (ignores --interval)")
    args = ap.parse_args()

    VIZ.mkdir(parents=True, exist_ok=True)
    PID.write_text(str(os.getpid()), encoding="ascii")

    if args.final_only:
        log(f"final_only start pid={os.getpid()}")
        try:
            final_pass(args)
        finally:
            try:
                PID.unlink(missing_ok=True)
            except OSError:
                pass
        log("auto_visualize stop")
        return

    log(f"auto_visualize start pid={os.getpid()} interval={args.interval}s full_every={args.full_every}")
    try:
        for n in range(1, args.max_cycles + 1):
            try:
                cycle(n, args)
            except Exception as exc:  # noqa: BLE001
                log(f"cycle {n} errored: {exc!r}")
            if not trainer_alive():
                log("trainer no longer alive; running the exhaustive final pass and exiting")
                try:
                    final_pass(args)
                except Exception as exc:  # noqa: BLE001
                    log(f"final pass errored: {exc!r}")
                break
            time.sleep(max(args.interval, 60))
    except KeyboardInterrupt:
        log("interrupted")
    finally:
        try:
            PID.unlink(missing_ok=True)
        except OSError:
            pass
        log("auto_visualize stop")


if __name__ == "__main__":
    main()
