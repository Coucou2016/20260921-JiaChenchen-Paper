"""Deep-water FINE-TUNE of the frozen V0 checkpoint (single 4 GB GPU).

Why a fine-tune rather than another from-scratch arm
----------------------------------------------------
The from-scratch sweep (D0 control vs D2 w_deep=0.5) showed l_deep *does* move deep-water
error (>=1 m MAE -5.3%, underestimation fraction down at every bin < 3 m), but at w_deep=0.5
the term was 59% of the objective (measured), cost ~30% overall MAE, and the arm could not
recover val CSI_005 inside the guardrail. The frozen V0 checkpoint is already strong
(full-split val CSI_005 0.5392); the open question is whether l_deep can improve *its* deep
tail in a few epochs, which is what a fine-tune tests cheaply.

Two traps this script exists to avoid (both verified empirically)
-----------------------------------------------------------------
1. `--epochs` is an ABSOLUTE epoch index, and resuming fast-forwards the cosine scheduler
   once per elapsed epoch. Resuming ep180 with `--epochs 175` leaves lr ~2.3e-7 and 0 epochs;
   with `--epochs 200` lr is ~2.7e-6. Either way the model barely moves.
   -> pass the horizon via `--sched_epochs N`, which sets epochs = start+N-1, rebuilds the
      cosine over exactly N epochs and skips the fast-forward.
2. `optimizer.load_state_dict` restores the checkpoint's saved lr (3.7e-5 at ep180) and
   clobbers the constructed lr.
   -> `train_fixed.py --sched_epochs` re-applies lr AFTER load_checkpoint; `--lr` sets it.

Weighting note (measured at the frozen ep180 checkpoint, val, 8 batches)
-----------------------------------------------------------------------
raw components: l_depth 0.0695, l_extreme 0.0657, l_log 0.0320, l_boundary 0.0310,
l_wet 0.0047 -> total 0.2028 at w_deep=0.  raw l_deep = 1.5684.
Because a from-scratch model already has small deep error while the frozen one still has a
-2 m deep bias, raw l_deep here is ~3x larger than D2's, so the SAME w_deep buys far more
influence:

    w_deep   weighted l_deep   share of objective
      0.05        0.0784              27.9%
      0.10        0.1568              43.6%
      0.20        0.3137              60.7%
      0.25        0.3921              65.9%   <- MORE dominant than D2's 59%
      0.50        0.7842              79.5%

So 0.25 is NOT a "gentler half" when fine-tuning; it repeats D2's defect. Gentle arms here
are 0.05-0.10.

Usage
-----
    python scripts/run_deep_finetune.py --list
    python scripts/run_deep_finetune.py --wait-pid 33776 --arms 0.05 0.1
    python scripts/run_deep_finetune.py --arms 0.1 --epochs 20 --lr 5e-5 --dry-run
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
FIN_DIR = ROOT / "outputs" / "finetune_deep"
FROZEN = ROOT / "outputs" / "v0_10m2m_hmax" / "last.pt"  # ep180
CONFIG = "configs/v0_10m2m_hmax.yaml"
EVAL_MAX_BATCHES = 60  # same subset as the D0/D2 arms, so numbers stay comparable


def log(msg: str) -> None:
    FIN_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (FIN_DIR / "finetune.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                         capture_output=True, text=True, errors="ignore").stdout
    return str(pid) in out


def wait_for_pid(pid: int, timeout_h: float) -> bool:
    log(f"waiting for pid={pid} to exit before starting the fine-tune")
    t0 = time.time()
    while alive(pid):
        if (time.time() - t0) / 3600 > timeout_h:
            log(f"timeout after {timeout_h} h waiting for pid={pid}; not starting")
            return False
        time.sleep(30)
    log(f"pid={pid} gone after {(time.time()-t0)/60:.1f} min")
    return True


def clean_env(base: dict | None = None) -> dict:
    """Environment for child processes.

    An EMPTY (rather than unset) CUDA_VISIBLE_DEVICES makes PyTorch hide the GPU entirely.
    train_fixed.py deliberately aborts on it, but it must never be inherited from a parent
    shell that happened to set it: that is exactly how the first fine-tune run died in 7 s
    (both arms exit=1, no training at all).
    """
    env = dict(os.environ if base is None else base)
    if "CUDA_VISIBLE_DEVICES" in env and not env["CUDA_VISIBLE_DEVICES"].strip():
        env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def run(cmd: list[str], log_path: Path, label: str, env: dict | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"{label}: {' '.join(cmd)}")
    with log_path.open("a", encoding="utf-8", errors="ignore") as lf:
        lf.write(f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {label} =====\n")
        lf.flush()
        # stdout.log, never train.log: train_fixed.py itself opens train.log in append mode
        # and tees into it, and sharing the path caused a PermissionError crash earlier.
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT,
                             env=clean_env(env))
        return p.wait()


def spawn_detached(cmd: list[str], log_path: Path, env: dict) -> int:
    """Start a CPU-only helper (auto_visualize) that outlives this driver."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="ignore") as lf:
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT,
                             env=clean_env(env))
    return p.pid


def last_lines(p: Path, n: int = 3) -> str:
    """Trailing non-empty lines of a log, for failure reporting."""
    try:
        lines = [l.strip() for l in p.read_text(encoding="utf-8", errors="ignore").splitlines()
                 if l.strip()]
    except OSError:
        return ""
    return " | ".join(lines[-n:])


def trainer_reached_end(d: Path) -> bool:
    """Did train_fixed.py actually finish, regardless of the process exit code?

    On this Windows box a COMPLETE run still exits 120: the interpreter emits
    "Exception ignored on flushing sys.stdout" during teardown and Python turns that into a
    non-zero status. Every accepted from-scratch arm (D0, D2) exited 120 too, so `code != 0`
    alone is not evidence of a crash -- treating it as one is what misfiled two finished
    fine-tune arms as FAILED and would have burned ~8.8 h retraining them.

    Completion is therefore proved from the trainer's own artefacts, not from the exit code:
    its "done. checkpoints" sentinel plus a per-epoch history and a written checkpoint.
    """
    try:
        out = (d / "stdout.log").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    if "done. checkpoints" not in out:
        return False
    if not (d / "last.pt").exists():
        return False
    try:
        rows = [l for l in (d / "history.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()]
    except OSError:
        return False
    return len(rows) > 0


def arm_dir_for(w_deep: float) -> Path:
    return FIN_DIR / f"w{str(w_deep).replace('.', '')}"


def train_arm(w_deep: float, epochs: int, lr: float, patience: int,
              eval_test: bool = False) -> dict:
    d = arm_dir_for(w_deep)
    d.mkdir(parents=True, exist_ok=True)
    status_p = d / "status.json"
    status = read_json(status_p)
    # Only a genuinely finished arm may be skipped. A crashed arm is recorded as
    # state="failed" below and MUST be retried, otherwise a transient error would be
    # silently reinterpreted as "the deep term does not help".
    if status.get("state") in ("done", "running") and (status.get("metrics") or {}).get("best_deep_csi"):
        log(f"w_deep={w_deep}: already done, skipping")
        return status.get("metrics", {})

    cmd = [PY, "-u", "scripts/train_fixed.py",
           "--config", CONFIG,
           "--out_dir", str(d.relative_to(ROOT)).replace("\\", "/"),
           "--resume", str(FROZEN),
           "--sched_epochs", str(epochs),
           "--lr", str(lr),
           "--w_deep", str(w_deep),
           "--deep_delta", "5.0",
           "--deep_threshold", "1.0",
           "--patience", str(patience),
           "--eval_max_batches", str(EVAL_MAX_BATCHES)]

    status_p.write_text(json.dumps(
        {"state": "running", "w_deep": w_deep, "epochs": epochs, "lr": lr,
         "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")

    # CPU-only visualisation for this arm, so progress is visible while it runs.
    env = dict(os.environ)
    env["V0_OUT_DIR"] = str(d)
    viz_pid = spawn_detached(
        [PY, "-u", "scripts/auto_visualize.py", "--interval", "900", "--full-every", "3"],
        d / "auto_viz_stdout.log", env)
    log(f"w_deep={w_deep}: auto_visualize pid={viz_pid}")

    # Checkpoint selection for the reported number must not rest on the 60-batch per-epoch
    # subset: that subset has too few deep pixels for CSI_100 to be stable. Snapshot EVERY
    # epoch (stride 1) so the best deep-water epoch can be chosen afterwards on the FULL val
    # split. ~17 MB/epoch, 20 epochs/arm.
    watch_pid = spawn_detached(
        [PY, "-u", "scripts/watch_checkpoints.py", "--out-dir", str(d),
         "--stride", "1", "--interval", "60", "--total-epochs", "100000"],
        d / "watch_stdout.log", env)
    log(f"w_deep={w_deep}: per-epoch watcher pid={watch_pid}")

    code = run(cmd, d / "stdout.log", f"finetune w_deep={w_deep}")

    m = read_json(d / "best_deep_summary.json") or {}
    bs = read_json(d / "best_summary.json")
    out = {"exit_code": code, "w_deep": w_deep, "epochs": epochs, "lr": lr,
           "best_csi": bs.get("best_csi"), "best_deep_csi": m.get("best_deep_csi"),
           "best_deep_epoch": m.get("epoch"), "CSI_005_at_best_deep": m.get("CSI_005"),
           "CSI_030_at_best_deep": m.get("CSI_030"),
           "PeakDepthError_at_best_deep": m.get("PeakDepthError"),
           "RMSE_wet_at_best_deep": m.get("RMSE_wet"), "viz_pid": viz_pid}

    # A crash is NOT a scientific result. Require a real run: a completed trainer (proved from
    # its artefacts, since a full run still exits 120 here) AND a deep-tail checkpoint (the
    # latter also proves the LR/trap fixes took effect, because a scheduler-collapsed run
    # would never produce a new best CSI_100).
    completed = trainer_reached_end(d)
    if not completed or out["best_deep_csi"] is None:
        tail = last_lines(d / "stdout.log", 3)
        log(f"w_deep={w_deep}: FAILED exit={code} completed={completed} "
            f"best_deep_csi={out['best_deep_csi']}; last output: {tail}")
        status_p.write_text(json.dumps(
            {"state": "failed", "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
             "failure": {"exit_code": code, "completed": completed, "tail": tail},
             "metrics": out},
            indent=2), encoding="utf-8")
        out["failed"] = True
        return out

    if code != 0:
        log(f"w_deep={w_deep}: exit={code} but the trainer printed its done sentinel and wrote "
            f"all checkpoints -> benign teardown flush artifact; treating as done")
    status_p.write_text(json.dumps({"state": "done", "finished":
                                    time.strftime("%Y-%m-%d %H:%M:%S"), "metrics": out},
                                   indent=2), encoding="utf-8")
    log(f"w_deep={w_deep}: DONE exit={code} best_csi={out['best_csi']} "
        f"best_deep_csi={out['best_deep_csi']} @ep{out['best_deep_epoch']}")
    return out


def eval_ckpt(tag: str, split: str, ckpt: Path) -> dict:
    out = FIN_DIR / "metrics" / f"{tag}_{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    run([PY, "-u", "scripts/evaluate.py", "--config", CONFIG, "--model", "hydrogeo_srno",
         "--ckpt", str(ckpt), "--split", split, "--out", str(out)],
        FIN_DIR / "eval.log", f"eval {tag} {split}")
    m = read_json(out)
    log(f"eval {tag}/{split}: CSI_005={m.get('CSI_005')} CSI_030={m.get('CSI_030')} "
        f"CSI_100={m.get('CSI_100')} RMSE_wet={m.get('RMSE_wet')} "
        f"PeakDepthError={m.get('PeakDepthError')}")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", type=float, default=[0.05, 0.10],
                    help="w_deep values to fine-tune (gentle: 0.05-0.10; see module docstring)")
    ap.add_argument("--epochs", type=int, default=20, help="fine-tune horizon per arm")
    ap.add_argument("--lr", type=float, default=1e-4, help="cosine start lr for the fine-tune")
    ap.add_argument("--patience", type=int, default=None, help="defaults to --epochs")
    ap.add_argument("--wait-pid", type=int, default=None,
                    help="wait for this PID to exit first (serialise on the single GPU)")
    ap.add_argument("--timeout-h", type=float, default=8.0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-selection", action="store_true",
                    help="train the arms and stop; run --select-only later (lets a control arm "
                         "be appended without disturbing a run already in flight)")
    ap.add_argument("--select-only", action="store_true",
                    help="skip training and re-run the comparison over whatever arms exist")
    ap.add_argument("--with-control", action="store_true", default=True,
                    help="(selection) use the w_deep=0 fine-tuned arm as the reference; "
                         "required for an apples-to-apples comparison (see select())")
    args = ap.parse_args()
    patience = args.patience or args.epochs

    if not FROZEN.exists():
        raise SystemExit(f"frozen checkpoint missing: {FROZEN}")
    if args.list:
        print(f"frozen checkpoint : {FROZEN}")
        print(f"horizon           : {args.epochs} epochs/arm, lr={args.lr}, patience={patience}")
        print("arms (w_deep)     :", args.arms)
        return

    log(f"fine-tune start: arms={args.arms} epochs={args.epochs} lr={args.lr} "
        f"from {FROZEN.name} (ep180)")

    if args.wait_pid and not args.dry_run:
        if not wait_for_pid(args.wait_pid, args.timeout_h):
            return

    if args.dry_run:
        for w in args.arms:
            d = arm_dir_for(w)
            print(f"DRY-RUN w_deep={w} -> {d}")
            print(f"  {PY} -u scripts/train_fixed.py --config {CONFIG} "
                  f"--out_dir {d.relative_to(ROOT)} --resume {FROZEN} "
                  f"--sched_epochs {args.epochs} --lr {args.lr} --w_deep {w} "
                  f"--deep_delta 5.0 --deep_threshold 1.0 --patience {patience} "
                  f"--eval_max_batches {EVAL_MAX_BATCHES}")
        return

    results: dict[str, dict] = {}
    failed: list[str] = []
    if not args.select_only:
        for w in args.arms:
            r = train_arm(w, args.epochs, args.lr, patience)
            results[str(w)] = r
            if r.get("failed"):
                failed.append(str(w))

        (FIN_DIR / "finetune_summary.json").write_text(json.dumps(results, indent=2),
                                                       encoding="utf-8")

        # Never turn a crash into a conclusion. If any arm failed to actually run, stop and say
        # so instead of reporting "the deep term does not help" (which is what a 7-second exit=1
        # produced before this guard existed).
        if failed:
            log(f"ABORT: arm(s) {failed} failed to run; refusing to draw a scientific conclusion "
                f"from a crashed run. Inspect outputs/finetune_deep/*/stdout.log, fix, then re-run "
                f"the same command (failed arms are retried, not skipped).")
            (FIN_DIR / "selection.json").write_text(json.dumps(
                {"winner": None, "status": "aborted_arms_failed", "failed_arms": failed,
                 "arms": results}, indent=2), encoding="utf-8")
            return

    if args.skip_selection:
        log("skip-selection set; training finished, selection deferred (use --select-only)")
        return

    # ---- deep-tail comparison (val only; test never used to select) ----
    # Reference choice matters. The per-epoch CSI_100 that picks each arm's checkpoint comes
    # from the 60-batch val SUBSET, while `post_sweep`'s frozen number is the FULL 182-tile
    # split. Those are different measurement conditions, so comparing a fine-tuned arm against
    # the frozen full-split number would conflate "what l_deep did" with "which subset was
    # measured". The training arms already log per-epoch metrics on the *same* 60-batch subset,
    # so the clean reference is the w_deep=0 arm measured the same way.
    base = read_json(ROOT / "outputs" / "deep_sweep" / "D0" / "status.json").get("metrics", {})
    log(f"reference (from-scratch D0, 60-batch subset): best_csi={base.get('best_csi')} "
        f"CSI_100={base.get('CSI_100')}")

    frozen_val = read_json(ROOT / "outputs" / "post_sweep" / "metrics" / "baseline_v0_val.json")
    log(f"frozen V0 full-split val: CSI_005={frozen_val.get('CSI_005')} "
        f"CSI_100={frozen_val.get('CSI_100')} RMSE_wet={frozen_val.get('RMSE_wet')}")

    if args.skip_eval:
        log("skip-eval set; stopping before evaluation")
        return

    # Evaluate every fine-tuned deep-tail checkpoint on val (full split).
    metrics: dict[str, dict] = {}
    arms_to_eval = [w for w in args.arms]
    if args.with_control and 0.0 not in arms_to_eval:
        arms_to_eval.append(0.0)
    for w in arms_to_eval:
        d = arm_dir_for(w)
        ck = d / "best_deep_csi.pt"
        if not ck.exists():
            log(f"w_deep={w}: no best_deep_csi.pt; cannot evaluate this arm")
            continue
        metrics[str(w)] = eval_ckpt(f"finetune_w{w}_bestdeep", "val", ck)

    if not metrics:
        log("ABORT: no arm produced an evaluable checkpoint; nothing to compare.")
        (FIN_DIR / "selection.json").write_text(json.dumps(
            {"winner": None, "status": "no_evaluable_checkpoint", "arms": results},
            indent=2), encoding="utf-8")
        return

    select(metrics, frozen_val, base)

def select(metrics: dict[str, dict], frozen_val: dict, base: dict) -> None:
    """Compare fine-tuned arms against the w_deep=0 fine-tuned control.

    Both sides are measured on the FULL val split with the same evaluator, so the difference
    isolates the deep term's effect instead of mixing in "subset vs full split".
    """
    # A fine-tune is only worth keeping if it does not give back overall accuracy.
    # CSI_005 tolerance is 0.005 (tighter than the from-scratch 0.01, because we start from an
    # already-good checkpoint).
    TOL = 0.005
    control = metrics.get("0.0") or metrics.get("0")
    if control is not None:
        ref_c005 = float(control.get("CSI_005") or float("nan"))
        ref_c100 = float(control.get("CSI_100") or float("nan"))
        ref_src = "fine-tuned control (w_deep=0), full-split val"
    else:
        ref_c005 = float(frozen_val.get("CSI_005") or float("nan"))
        ref_c100 = float(frozen_val.get("CSI_100") or float("nan"))
        ref_src = "FROZEN V0 full-split val (no control arm trained)"
    log(f"reference: {ref_src}: CSI_005={ref_c005:.4f} CSI_100={ref_c100:.4f}")

    eligible = []
    for k, m in metrics.items():
        if k in ("0.0", "0"):
            continue  # the control defines the reference, it cannot also be the candidate
        c005, c100 = float(m.get("CSI_005") or 0), float(m.get("CSI_100") or 0)
        ok = c005 >= ref_c005 - TOL
        gain = c100 - ref_c100
        log(f"candidate w_deep={k}: CSI_005={c005:.4f} (ref {ref_c005:.4f}, "
            f"{'PASS' if ok else 'REJECT'}) CSI_100={c100:.4f} (ref {ref_c100:.4f}, "
            f"delta {gain:+.4f}){'  -> improves deep' if gain > 0 else '  -> no deep gain'}")
        if ok and gain > 0:
            eligible.append((k, c100, c005, gain))

    if not eligible:
        log("VERDICT (null result): no fine-tuned arm improved deep-water CSI_100 by more than "
            "the control allows while holding val CSI_005 within tolerance. The deep term does "
            "not help the frozen model either; nothing is kept.")
        (FIN_DIR / "selection.json").write_text(json.dumps(
            {"winner": None, "reason": "no arm beat the fine-tuned control on val CSI_100 "
                                       "within tolerance",
             "tolerance": TOL, "reference": ref_src, "control": control,
             "frozen_val": frozen_val, "candidates": metrics}, indent=2), encoding="utf-8")
        return

    eligible.sort(key=lambda t: -t[1])
    win = eligible[0][0]
    log(f"VERDICT: winner w_deep={win} (CSI_100={eligible[0][1]:.4f}, "
        f"delta {eligible[0][3]:+.4f}); eligible={[e[0] for e in eligible]}")

    # test is read once, for the winner only.
    win_ck = arm_dir_for(float(win)) / "best_deep_csi.pt"
    test_m = eval_ckpt(f"finetune_w{win}_bestdeep", "test", win_ck)

    (FIN_DIR / "selection.json").write_text(json.dumps(
        {"winner": win, "tolerance": TOL, "reference": ref_src, "control": control,
         "frozen_val": frozen_val, "frozen_test":
         read_json(ROOT / "outputs" / "post_sweep" / "metrics" / "baseline_v0_test.json"),
         "candidates_val": metrics, "winner_test": test_m, "eligible": [e[0] for e in eligible]},
        indent=2), encoding="utf-8")
    log("fine-tune finished; see outputs/finetune_deep/selection.json")


if __name__ == "__main__":
    main()
