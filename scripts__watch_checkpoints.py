#!/usr/bin/env python
"""Snapshot intermediate checkpoints so image-space training progress can be visualised.

Training only keeps `last.pt` / `best_csi.pt` / `best_rmse_wet.pt`, which are
overwritten every epoch. This watcher runs beside the trainer (read-only with
respect to the trainer) and copies the current weights to
`<out_dir>/snapshots/ep%04d.pt` on a stride, plus on every new best.

It never touches the training process, GPU or dataset; it only copies files.
Stop it with Ctrl+C or by killing the PID in `snapshots/watcher.pid`.

Usage:
  python scripts/watch_checkpoints.py --stride 10 --interval 45
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def log(msg: str, path: Path | None = None) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    if path is not None:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_best_summary(out_dir: Path) -> dict:
    p = out_dir / "best_summary.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_state(snap_dir: Path) -> dict:
    """Persisted watcher state so restarts do not re-capture identical best weights."""
    p = snap_dir / ".watcher_state.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(snap_dir: Path, state: dict) -> None:
    p = snap_dir / ".watcher_state.json"
    try:
        p.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass


def prune_redundant(out_dir: Path, snap_dir: Path, watcher_log: Path) -> None:
    """Delete best_ep*.pt files whose weights never became the new best.

    Reads history.jsonl to find the epochs where best-so-far CSI_005 actually
    increased; any best_epNNNN.pt outside that set is a byte-identical copy of an
    earlier best and can be removed safely.
    """
    hist = out_dir / "history.jsonl"
    if not hist.exists():
        log("prune: no history.jsonl; nothing to do", watcher_log)
        return
    rows = []
    for line in hist.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows = [r for r in rows if "CSI_005" in r]
    rows.sort(key=lambda r: r["epoch"])

    keep: set[str] = set()
    best = -1.0
    for r in rows:
        v = r["CSI_005"]
        if v == v and v > best:
            best = v
            keep.add(f"best_ep{int(r['epoch']):04d}.pt")

    removed, freed = 0, 0
    for p in sorted(snap_dir.glob("best_ep*.pt")):
        if p.name not in keep:
            try:
                freed += p.stat().st_size
                p.unlink()
                removed += 1
            except OSError as exc:
                log(f"prune: could not delete {p.name}: {exc}", watcher_log)
    log(f"prune: removed {removed} redundant best_ep*.pt, reclaimed {freed/1e9:.2f} GB, "
        f"kept {len(keep)} genuine improvements", watcher_log)
    print(f"pruned {removed} files ({freed/1e9:.2f} GB); kept {len(keep)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "v0_10m2m_hmax"))
    ap.add_argument("--stride", type=int, default=10, help="keep a snapshot every N epochs")
    ap.add_argument("--interval", type=int, default=45, help="poll interval (seconds)")
    ap.add_argument("--also-best", action="store_true", default=True)
    ap.add_argument("--total-epochs", type=int, default=300)
    ap.add_argument("--prune", action="store_true",
                    help="delete redundant best_ep*.pt snapshots and exit")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    snap_dir = out_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    watcher_log = snap_dir / "watcher.log"

    if args.prune:
        prune_redundant(out_dir, snap_dir, watcher_log)
        return

    pid_file = snap_dir / "watcher.pid"
    pid_file.write_text(str(os.getpid()), encoding="ascii")

    log(f"watcher start pid={os.getpid()} stride={args.stride} interval={args.interval}s", watcher_log)
    log(f"watching {out_dir}", watcher_log)

    saved_epochs: set[int] = set()
    saved_best: set[int] = set()
    state = load_state(snap_dir)
    last_best_csi: float | None = state.get("last_best_csi")
    log(f"resumed state: last_best_csi={last_best_csi}", watcher_log)
    idle = 0

    try:
        while True:
            summary = read_best_summary(out_dir)
            epoch = int(summary.get("last_epoch", 0) or 0)
            best_csi = summary.get("best_csi")

            # periodic stride snapshot
            if epoch >= args.stride and epoch % args.stride == 0 and epoch not in saved_epochs:
                src = out_dir / "last.pt"
                dst = snap_dir / f"ep{epoch:04d}.pt"
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
                    saved_epochs.add(epoch)
                    log(f"snapshot epoch={epoch} -> {dst.name} ({dst.stat().st_size/1e6:.1f} MB)", watcher_log)

            # best-CSI snapshot, ONLY when the best actually improved.
            # Copying best_csi.pt every epoch produced many identical 17 MB files
            # (the best only moves on a minority of epochs) and made the evolution
            # figure re-run near-duplicate weights.
            if (
                args.also_best
                and epoch
                and epoch not in saved_best
                and best_csi is not None
                and best_csi != last_best_csi
            ):
                src = out_dir / "best_csi.pt"
                dst = snap_dir / f"best_ep{epoch:04d}.pt"
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
                    saved_best.add(epoch)
                    last_best_csi = best_csi
                    save_state(snap_dir, {"last_best_csi": best_csi, "last_epoch": epoch})
                    log(f"best snapshot epoch={epoch} csi={best_csi} (improvement)", watcher_log)

            if epoch >= args.total_epochs:
                log(f"target epochs reached (last_epoch={epoch}); watcher exiting", watcher_log)
                break

            time.sleep(max(args.interval, 5))
            idle += 1
    except KeyboardInterrupt:
        log("watcher interrupted by user", watcher_log)
    finally:
        try:
            pid_file.unlink(missing_ok=True)
        except OSError:
            pass
        log(f"watcher stop (saved {len(saved_epochs)} stride, {len(saved_best)} best)", watcher_log)


if __name__ == "__main__":
    main()
