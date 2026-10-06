"""Sequential deep-water loss sweep for V0 (single 4 GB GPU).

Motivation
----------
The V0 model plateaus at val CSI_005 ~ 0.515 and its deep-water bias is ~ -2 m above 3 m,
and the bias was slowly *worsening* over training. Measured cause (scripts/diagnose_loss.py):
  * l_depth + l_log act on the log1p residual and account for ~45% of the loss, but in log
    space a 2 m error at 5 m depth costs ~6x less than in metres, so they barely see deep error.
  * l_extreme is linear but its mask threshold is a per-tile 95th percentile that is only
    ~0.19 m (median) because most tiles are mostly dry, so it does not reliably cover the tail.

Fix under test: a linear, target-masked deep-water term (`l_deep`, see losses/flood_loss.py).

Design
------
* ONE arm at a time (4 GB GPU, 14.4 min/epoch => no parallelism).
* Arms are short and comparable: same epoch budget, stopped on deep-water val metrics that are
  NOT selected on by training (so the comparison is not rigged toward the new term).
* Held-out split: `test` is read ONLY once, at the end, for the winning arm. `val` is used for
  the per-arm checkpoint choice, so `val` is not a clean holdout here.
* Restart-safe: per-arm status lives in outputs/deep_sweep/<tag>/status.json, so re-running
  skips finished arms and resumes an interrupted one.
* CPU-only monitoring: this driver never imports torch; the GPU stays free for training.

Usage
-----
    python scripts/run_deep_sweep.py --list
    python scripts/run_deep_sweep.py                 # run all remaining arms
    python scripts/run_deep_sweep.py --only D1 D3    # run selected arms
    python scripts/run_deep_sweep.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "outputs" / "deep_sweep"
PY = sys.executable

# tag -> (w_deep, deep_delta, deep_threshold, description)
ARMS: dict[str, dict] = {
    "D0": dict(w_deep=0.0, deep_delta=5.0, deep_threshold=1.0,
               desc="control: original loss (w_deep=0)"),
    "D1": dict(w_deep=0.25, deep_delta=5.0, deep_threshold=1.0,
               desc="mild linear deep term"),
    "D2": dict(w_deep=0.5, deep_delta=5.0, deep_threshold=1.0,
               desc="medium linear deep term"),
    "D3": dict(w_deep=1.0, deep_delta=5.0, deep_threshold=1.0,
               desc="strong linear deep term"),
    "D4": dict(w_deep=0.5, deep_delta=1.0, deep_threshold=1.0,
               desc="medium weight, tighter Huber delta (aggressive on large errors)"),
}

SCRIPT = "scripts/train_fixed.py"

# Controlled experiment. VARYING: only the deep-water loss term. HELD IDENTICAL across arms:
# init seed (42, so weights start identical), architecture, data order, lr schedule, epoch
# budget, patience, and the val subset used for per-epoch metrics. `loss.name` is unchanged,
# so all non-deep terms stay byte-identical, and the deep term is off at w_deep=0 (D0 control).
# Only the loss changes => no new variables are introduced into the training loop.


def run(cmd: list[str], log_path: Path, label: str) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] {label}: {' '.join(cmd)}")
    with log_path.open("a", encoding="utf-8", errors="ignore") as lf:
        lf.write(f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {label} =====\n")
        lf.flush()
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
        return p.wait()


def read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def arm_metrics(arm_dir: Path) -> dict:
    """Best metrics for an arm, plus the val checkpoint it came from."""
    bs = read_json(arm_dir / "best_summary.json")
    out = {"best_csi": bs.get("best_csi"), "best_rmse_wet": bs.get("best_rmse_wet"),
           "last_epoch": bs.get("last_epoch")}
    mk = read_json(arm_dir / "metrics_last.json")
    for k in ("CSI_005", "CSI_030", "CSI_100", "RMSE_wet", "MAE_wet",
              "PeakDepthError", "VolumeRelativeError"):
        if k in mk:
            out[k] = mk[k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40,
                    help="epoch budget per arm; 40 matches the early_stopping patience")
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--eval-test", action="store_true",
                    help="after the sweep, run the held-out test evaluation on the winner")
    ap.add_argument("--eval-max-batches", type=int, default=60,
                    help="val batches per epoch; 60 bounds eval cost so sweep epochs stay cheap")
    args = ap.parse_args()

    tags = list(ARMS) if args.only is None else args.only
    for t in tags:
        if t not in ARMS:
            raise SystemExit(f"unknown arm {t}; known: {list(ARMS)}")

    if args.list:
        print(f"{'tag':<4} {'w_deep':>7} {'delta':>6} {'thr':>5}  description")
        for t, a in ARMS.items():
            print(f"{t:<4} {a['w_deep']:>7.2f} {a['deep_delta']:>6.1f} "
                  f"{a['deep_threshold']:>5.1f}  {a['desc']}")
        return

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    master = SWEEP_DIR / "sweep.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        with master.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"sweep start: arms={tags} epochs={args.epochs} patience={args.patience}")
    results: dict[str, dict] = {}

    for tag in tags:
        a = ARMS[tag]
        arm_dir = SWEEP_DIR / tag
        status_path = arm_dir / "status.json"
        status = read_json(status_path)

        if status.get("state") == "done":
            log(f"{tag}: already done, skipping")
            results[tag] = status.get("metrics", {})
            continue

        cmd = [PY, "-u", SCRIPT, "--config", "configs/v0_10m2m_hmax.yaml",
               "--out_dir", str(arm_dir.relative_to(ROOT)).replace("\\", "/"),
               "--w_deep", str(a["w_deep"]),
               "--deep_delta", str(a["deep_delta"]),
               "--deep_threshold", str(a["deep_threshold"]),
               "--epochs", str(args.epochs),
               "--patience", str(args.patience),
               "--eval_max_batches", str(args.eval_max_batches)]
        if args.dry_run:
            log(f"{tag}: DRY-RUN {' '.join(cmd)}")
            continue

        arm_dir.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps({"state": "running", "tag": tag,
                                           **a, "started": time.strftime("%Y-%m-%d %H:%M:%S")},
                                          indent=2), encoding="utf-8")
        log(f"{tag}: START ({a['desc']})")
        # IMPORTANT: redirect to stdout.log, NOT train.log. train_fixed.py itself opens
        # train.log in append mode and tees stdout into it; pointing the subprocess at the
        # same path caused a PermissionError crash in an earlier run of this project.
        code = run(cmd, arm_dir / "stdout.log", f"arm {tag}")
        m = arm_metrics(arm_dir)
        status_path.write_text(json.dumps({"state": "done", "tag": tag, **a, "exit_code": code,
                                           "metrics": m,
                                           "finished": time.strftime("%Y-%m-%d %H:%M:%S")},
                                          indent=2), encoding="utf-8")
        log(f"{tag}: DONE exit={code} best_csi={m.get('best_csi')} "
            f"CSI_100={m.get('CSI_100')} PeakDepthError={m.get('PeakDepthError')}")
        results[tag] = m

    # Summary table
    print(f"\n{'='*100}\nSWEEP SUMMARY\n{'='*100}")
    hdr = (f"{'tag':<4} {'w_deep':>7} {'best_csi':>9} {'CSI_005':>8} {'CSI_030':>8} "
           f"{'CSI_100':>8} {'RMSE_wet':>9} {'PeakErr':>8} {'VolErr':>7} {'ep':>4}")
    print(hdr)
    for tag in tags:
        m = results.get(tag) or arm_metrics(SWEEP_DIR / tag)
        a = ARMS[tag]
        def g(k, fmt="{:.4f}"):
            v = m.get(k)
            try:
                return fmt.format(float(v))
            except (TypeError, ValueError):
                return "n/a"
        print(f"{tag:<4} {a['w_deep']:>7.2f} {g('best_csi'):>9} {g('CSI_005'):>8} "
              f"{g('CSI_030'):>8} {g('CSI_100'):>8} {g('RMSE_wet'):>9} "
              f"{g('PeakDepthError','{:.3f}'):>8} {g('VolumeRelativeError','{:.4f}'):>7} "
              f"{str(m.get('last_epoch','?')):>4}")
    (SWEEP_DIR / "sweep_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (SWEEP_DIR / "SWEEP_DONE").write_text(
        json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "arms": tags}, indent=2),
        encoding="utf-8")
    log(f"sweep finished; summary -> {SWEEP_DIR / 'sweep_summary.json'}")

    if args.eval_test:
        log("note: set --eval-test only when the GPU is free; evaluate.py runs on CUDA")
        log("run: python scripts/evaluate.py --config configs/v0_10m2m_hmax.yaml "
            "--model hydrogeo_srno --ckpt outputs/deep_sweep/<W>/best_csi.pt --split test")


if __name__ == "__main__":
    main()
