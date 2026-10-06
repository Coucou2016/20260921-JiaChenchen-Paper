#!/usr/bin/env python
"""Extension fine-tune: continue from the selected deep-tuned checkpoint, with a MATCHED
w_deep=0 control.

Why a matched control is mandatory here
--------------------------------------
The first fine-tune established that w_deep=0.10 beats w_deep=0 at 20 epochs. It did NOT
establish that *more* fine-tuning helps: any further change could come from 20 more epochs of
this lr rather than from the deep term. So this driver resumes BOTH arms from the SAME parent
(the selected winner snapshot, `w01/snapshots/ep0187.pt`) and runs the same horizon / lr /
patience; only `w_deep` differs. Per-epoch differences are then paired again.

Parent choice matters: it is the winner's snapshot (`ep0187.pt`), NOT `w01/last.pt` (ep200),
because ep200 is not the selected epoch -- extending from it would silently reintroduce the
subset-argmax problem this project just removed.

Design inherited from the first fine-tune (all verified live there):
  * `--sched_epochs N` rebuilds a fresh cosine over N epochs and sets epochs = start+N-1, so the
    scheduler does not fast-forward (the trap that gave lr 2.3e-7 on a naive resume);
  * `--lr` re-applies the lr AFTER load_checkpoint (load_state_dict otherwise restores the
    checkpoint's own lr);
  * per-epoch snapshots via `watch_checkpoints.py --stride 1`, so the epoch is re-chosen on the
    FULL val split afterwards, not on the noisy 60-batch subset;
  * completion is judged from the trainer's ARTEFACTS, never from the exit code -- a complete
    run exits 120 on this box (benign stdout-flush warning), which twice misfiled finished arms
    as crashes.

Usage
-----
    python scripts/run_deep_extend.py --arms 0.0 0.1 --epochs 20 --lr 1e-4
    python scripts/run_deep_extend.py --list
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
FIN = ROOT / "outputs" / "finetune_deep"
PARENT = FIN / "w01" / "snapshots" / "ep0187.pt"  # w_deep=0.10 winner (full-val best CSI_100)
CONFIG = "configs/v0_10m2m_hmax.yaml"
EVAL_MAX_BATCHES = 60


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (FIN / "extend.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def arm_dir(w: float) -> Path:
    # names chosen so the existing FIN/<arm>/snapshots layout (and scan_epochs.py) resolve
    return FIN / ("w0ext" if float(w) == 0.0 else f"w{str(w).replace('.', '')}ext")


def clean_env() -> dict:
    env = dict(os.environ)
    if "CUDA_VISIBLE_DEVICES" in env and not env["CUDA_VISIBLE_DEVICES"].strip():
        env.pop("CUDA_VISIBLE_DEVICES", None)  # empty value hides the GPU entirely
    return env


def spawn(cmd: list[str], log_path: Path, env: dict) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="ignore") as lf:
        return subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT,
                                env=env).pid


def trainer_reached_end(d: Path) -> bool:
    """Completion from artefacts, not the exit code (a full run exits 120 here)."""
    try:
        if "done. checkpoints" not in (d / "stdout.log").read_text(encoding="utf-8",
                                                                 errors="ignore"):
            return False
    except OSError:
        return False
    if not (d / "last.pt").exists():
        return False
    try:
        return any(l.strip() for l in
                   (d / "history.jsonl").read_text(encoding="utf-8").splitlines())
    except OSError:
        return False


def eval_ckpt(tag: str, split: str, ck: Path) -> dict:
    out = FIN / "metrics" / f"{tag}_{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with (FIN / "eval.log").open("a", encoding="utf-8") as lf:
        subprocess.run([PY, "-u", "scripts/evaluate.py", "--config", CONFIG,
                        "--model", "hydrogeo_srno", "--ckpt", str(ck), "--split", split,
                        "--out", str(out)], cwd=str(ROOT), stdout=lf,
                       stderr=subprocess.STDOUT, env=clean_env())
    try:
        return json.loads(out.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", type=float, default=[0.0, 0.1])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--parent", default=str(PARENT))
    ap.add_argument("--with-control", action="store_true", default=True)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    patience = args.patience or (args.epochs + 5)
    parent = Path(args.parent)

    if args.list:
        print(f"parent : {parent}")
        print(f"arms   : {args.arms}  (0.0 is the matched control)")
        print(f"horizon: {args.epochs} epochs/arm, lr={args.lr}, patience={patience}")
        return

    if not parent.exists():
        raise SystemExit(f"parent checkpoint missing: {parent}")
    if not trainer_reached_end(FIN / "w01") and not (FIN / "w01_ext_done").exists():
        log("NOTE: parent is the winner snapshot ep0187.pt, not last.pt (deliberate)")

    log(f"extension start: arms={args.arms} epochs={args.epochs} lr={args.lr} from {parent}")

    results: dict[str, dict] = {}
    failed: list[str] = []
    if not args.skip_train:
        for w in args.arms:
            d = arm_dir(w)
            d.mkdir(parents=True, exist_ok=True)
            st_p = d / "status.json"
            if trainer_reached_end(d):
                log(f"w_deep={w}: already complete, skipping")
                continue
            st_p.write_text(json.dumps({"state": "running", "w_deep": w, "parent": str(parent),
                                        "started": time.strftime("%Y-%m-%d %H:%M:%S")},
                                       indent=2), encoding="utf-8")
            env = dict(os.environ)
            env["V0_OUT_DIR"] = str(d)
            wpid = spawn([PY, "-u", "scripts/watch_checkpoints.py", "--out-dir", str(d),
                          "--stride", "1", "--interval", "60", "--total-epochs", "100000"],
                         d / "watch_stdout.log", clean_env())
            vpid = spawn([PY, "-u", "scripts/auto_visualize.py", "--interval", "900",
                          "--full-every", "3"], d / "auto_viz_stdout.log", env)
            log(f"w_deep={w}: watcher pid={wpid} auto_visualize pid={vpid} -> {d.name}")
            cmd = [PY, "-u", "scripts/train_fixed.py", "--config", CONFIG,
                   "--out_dir", str(d.relative_to(ROOT)).replace("\\", "/"),
                   "--resume", str(parent), "--sched_epochs", str(args.epochs),
                   "--lr", str(args.lr), "--w_deep", str(w), "--deep_delta", "5.0",
                   "--deep_threshold", "1.0", "--patience", str(patience),
                   "--eval_max_batches", str(EVAL_MAX_BATCHES)]
            with (d / "stdout.log").open("a", encoding="utf-8", errors="ignore") as lf:
                lf.write(f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} extend w={w} =====\n")
                subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT,
                                 env=clean_env()).wait()
            ok = trainer_reached_end(d)
            bs = {}
            try:
                bs = json.loads((d / "best_deep_summary.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
            st_p.write_text(json.dumps({"state": "done" if ok else "failed",
                                        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "trainer_reached_end": ok, "metrics": bs},
                                       indent=2), encoding="utf-8")
            results[str(w)] = {"completed": ok, **bs}
            if not ok:
                failed.append(str(w))
            log(f"w_deep={w}: {'DONE' if ok else 'FAILED'} best_deep_csi={bs.get('best_deep_csi')} "
                f"@ep{bs.get('epoch')}")

    if failed:
        log(f"ABORT: arm(s) {failed} did not complete; not drawing a conclusion from a partial run")
        (FIN / "extend_selection.json").write_text(json.dumps(
            {"status": "aborted_arms_failed", "failed": failed, "arms": results}, indent=2),
            encoding="utf-8")
        return

    # Paired full-val rescore of both extended arms (reuses the first scan's tooling).
    log("running full-val per-epoch scan over the extended arms")
    with (FIN / "scan_ext.log").open("a", encoding="utf-8") as lf:
        subprocess.run([PY, "-u", "scripts/scan_epochs.py", "--arms", "w0ext", "w01ext",
                        "--stride", "1", "--out", str(FIN / "epoch_scan_ext.json")],
                       cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT, env=clean_env())
    log(f"extension scan done -> {FIN / 'epoch_scan_ext.json'}")
    (FIN / "w01_ext_done").write_text("ok", encoding="utf-8")


if __name__ == "__main__":
    main()
