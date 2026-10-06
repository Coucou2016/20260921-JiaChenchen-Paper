#!/usr/bin/env python
"""Full-val per-epoch scan for the deep-water fine-tune (the decisive comparison).

Why this exists
---------------
During training, `train_fixed.py` picks `best_deep_csi.pt` from the per-epoch VAL SUBSET
(`--eval_max_batches 60`), and the driver then reports that arm's number. That is a
max-over-epochs of a NOISY statistic: the w_deep=0 control arm's subset CSI_100 wandered over
0.1407..0.1711 (range 0.030) across 20 epochs of pure noise, i.e. sigma ~ 0.0052 and a
max-of-20 bias of roughly +1.5 sigma. So a candidate's "best_deep_csi=0.1737" is not
comparable to another arm's 0.1711, nor to the frozen model's FULL-split 0.1686.

This script re-scores EVERY per-epoch snapshot on the FULL val split with the same evaluator
used for the baselines, so
  * checkpoint selection is max-of-full-val (the same basis as the reported numbers), and
  * the candidate-vs-control deep difference is measured without the subset's selection bias.

The snapshots it consumes are written by `watch_checkpoints.py --stride 1` into
`outputs/finetune_deep/<arm>/snapshots/epNNNN.pt` (one per training epoch, untouched by
training afterwards, so the scan is reproducible offline and the GPU is only needed here).

Usage
-----
    python scripts/scan_epochs.py --arms w005 w01 w00 --stride 1
    python scripts/scan_epochs.py --arms w005 --stride 4 --include-maxes   # quick probe
    python scripts/scan_epochs.py --wait-pid 23680 --arms w005 w01 w00    # chain after training
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIN = ROOT / "outputs" / "finetune_deep"
FROZEN = ROOT / "outputs" / "v0_10m2m_hmax" / "last.pt"  # ep180, the fine-tune's parent
CONFIG = "configs/v0_10m2m_hmax.yaml"
MODEL = "hydrogeo_srno"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (FIN / "scan.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                         capture_output=True, text=True, errors="ignore").stdout
    return str(pid) in out


def wait_for_pid(pid: int, timeout_h: float) -> bool:
    log(f"waiting for pid={pid} to exit before scanning (single GPU: do not starve training)")
    t0 = time.time()
    while alive(pid):
        if (time.time() - t0) / 3600 > timeout_h:
            log(f"timeout after {timeout_h} h waiting for pid={pid}; starting anyway")
            return True
        time.sleep(30)
    log(f"pid={pid} gone after {(time.time()-t0)/60:.1f} min; starting scan")
    return True


def snapshots(arm: str, stride: int) -> list[tuple[int, Path]]:
    d = FIN / arm / "snapshots"
    out = []
    for p in sorted(d.glob("ep*.pt")):
        m = re.match(r"ep(\d+)\.pt$", p.name)
        if not m:
            continue
        ep = int(m.group(1))
        out.append((ep, p))
    out.sort()
    if stride > 1:
        out = out[::stride] + ([out[-1]] if out and out[-1] not in out[::stride] else [])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=["w005", "w01", "w00"])
    ap.add_argument("--stride", type=int, default=1, help="evaluate every Nth snapshot")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out", default=str(FIN / "epoch_scan.json"))
    ap.add_argument("--wait-pid", type=int, default=None)
    ap.add_argument("--timeout-h", type=float, default=10.0)
    ap.add_argument("--include-maxes", action="store_true",
                    help="always include each arm's subset-selected best_deep epoch")
    ap.add_argument("--max-batches", type=int, default=None, help="default: full split")
    args = ap.parse_args()

    FIN.mkdir(parents=True, exist_ok=True)
    if args.wait_pid and not wait_for_pid(args.wait_pid, args.timeout_h):
        return

    out_p = Path(args.out)
    results: dict = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    results.setdefault("split", args.split)
    results.setdefault("scans", {})

    # Reference points, evaluated on the same full split.
    refs = [
        ("frozen_ep180", FROZEN),
        ("bilinear", None),
        ("nearest", None),
    ]
    last_metric = 0.0
    for arm in args.arms:
        ckpts = snapshots(arm, args.stride)
        if args.include_maxes:
            bs = FIN / arm / "best_deep_summary.json"
            if bs.exists():
                ep = json.loads(bs.read_text(encoding="utf-8")).get("epoch")
                extra = FIN / arm / "snapshots" / f"ep{int(ep):04d}.pt"
                if extra.exists() and all(extra != c for _, c in ckpts):
                    ckpts.append((int(ep), extra))
                    ckpts.sort()
        if not ckpts:
            log(f"{arm}: no snapshots found under {FIN / arm / 'snapshots'}")
            continue
        log(f"{arm}: {len(ckpts)} checkpoints (ep{ckpts[0][0]}..ep{ckpts[-1][0]}) on {args.split}")
        results["scans"].setdefault(arm, {})
        for i, (ep, ck) in enumerate(ckpts, 1):
            tag = f"ep{ep:04d}"
            if results["scans"][arm].get(tag):
                continue
            out = FIN / "scan_metrics" / f"{arm}_{tag}_{args.split}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-u", "scripts/evaluate.py", "--config", CONFIG,
                   "--model", MODEL, "--ckpt", str(ck), "--split", args.split,
                   "--out", str(out)]
            if args.max_batches:
                cmd += ["--max_batches", str(args.max_batches)]
            t0 = time.time()
            code = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True).returncode
            dt = time.time() - t0
            try:
                m = json.loads(out.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                log(f"{arm}/{tag}: eval FAILED (exit={code}, {dt:.0f}s)")
                continue
            results["scans"][arm][tag] = {"epoch": ep, **m}
            out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
            last_metric = m.get("CSI_100", float("nan"))
            log(f"{arm}/{tag} [{i}/{len(ckpts)}] CSI_005={m.get('CSI_005'):.4f} "
                f"CSI_030={m.get('CSI_030'):.4f} CSI_100={m.get('CSI_100'):.4f} "
                f"PeakDepthErr={m.get('PeakDepthError'):.4f} "
                f"VolRelErr={m.get('VolumeRelativeError'):.4f} ({dt:.0f}s)")

    # Reference evaluations (frozen model + pixel baselines) on the same split.
    results.setdefault("refs", {})
    for tag, ck in refs:
        if tag in results["refs"]:
            continue
        out = FIN / "scan_metrics" / f"ref_{tag}_{args.split}.json"
        cmd = [sys.executable, "-u", "scripts/evaluate.py", "--config", CONFIG,
               "--model", MODEL if ck else tag, "--split", args.split, "--out", str(out)]
        if ck:
            cmd += ["--ckpt", str(ck)]
        subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        try:
            results["refs"][tag] = json.loads(out.read_text(encoding="utf-8"))
            log(f"ref {tag}: CSI_100={results['refs'][tag].get('CSI_100'):.4f}")
        except (OSError, json.JSONDecodeError):
            log(f"ref {tag}: eval failed")
    out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"scan complete -> {out_p}")


if __name__ == "__main__":
    main()
