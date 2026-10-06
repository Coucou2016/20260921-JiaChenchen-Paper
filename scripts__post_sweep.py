"""Post-sweep automation: pick the winning deep-water arm, get a clean baseline, then resume.

Sequence (unattended; the GPU is used by exactly one job at a time):
  1. wait for outputs/deep_sweep/SWEEP_DONE
  2. choose the winning arm
  3. full-split evaluation (CUDA) of: the frozen V0 baseline, the winner's own checkpoint, and
     the bilinear/nearest baselines  -> outputs/post_sweep/metrics/
  4. if (and only if) the deep term actually helped, resume training from the frozen ep180
     `last.pt` with the winner's loss config, into a NEW output dir so the old run is preserved.

Selection rule (documented so the choice is not a judgement call made after seeing numbers):
  * guardrail: an arm is eligible only if its val CSI_005 is within `--csi-tol` of the control
    arm D0 (default 0.01), i.e. the deep term must not cost overall accuracy;
  * among eligible arms, pick the highest val CSI_100 (the deep-water score);
  * tie-break on lower PeakDepthError;
  * if no arm improves CSI_100 over D0, do NOT resume — report that the deep term did not help.
`test` is never used for selection, only reported.

Usage:
    python scripts/post_sweep.py --wait
    python scripts/post_sweep.py --winner D2          # skip selection
    python scripts/post_sweep.py --wait --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "outputs" / "deep_sweep"
POST = ROOT / "outputs" / "post_sweep"
FROZEN = ROOT / "outputs" / "v0_10m2m_hmax"          # frozen V0 run (ep180 last.pt, ep155 best)
RESUME_TAG = "v0_deepresume"
PY = sys.executable


def log(msg: str) -> None:
    POST.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (POST / "post_sweep.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def wait_for_sweep(timeout_h: float) -> bool:
    done = SWEEP / "SWEEP_DONE"
    t0 = time.time()
    while not done.exists():
        if (time.time() - t0) / 3600 > timeout_h:
            log(f"timed out after {timeout_h} h waiting for {done}")
            return False
        time.sleep(30)
    return True


def load_arms() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for status in sorted(SWEEP.glob("*/status.json")):
        s = read_json(status)
        if s.get("state") == "done":
            out[status.parent.name] = s
    return out


def pick_winner(arms: dict[str, dict], csi_tol: float) -> tuple[str | None, str]:
    if "D0" not in arms:
        return None, "control arm D0 missing"
    ref_csi = float(arms["D0"]["metrics"].get("best_csi") or float("nan"))
    ref_c100 = float(arms["D0"]["metrics"].get("CSI_100") or float("nan"))
    if ref_csi != ref_csi:
        return "D0", "control has no usable best_csi; defaulting to control"

    eligible, rejected = [], []
    for tag, s in arms.items():
        m = s["metrics"]
        c005 = m.get("best_csi")
        if c005 is None:
            continue
        if tag != "D0" and float(c005) < ref_csi - csi_tol:
            rejected.append(f"{tag}(csi005={float(c005):.4f} < {ref_csi - csi_tol:.4f})")
            continue
        eligible.append((tag, float(m.get("CSI_100") or float("-inf")),
                         float(m.get("PeakDepthError") or float("inf"))))

    log(f"selection: ref D0 best_csi={ref_csi:.4f} CSI_100={ref_c100:.4f} "
        f"tolerance={csi_tol}")
    if rejected:
        log(f"selection: rejected by CSI_005 guardrail: {', '.join(rejected)}")
    if not eligible:
        return "D0", "no arm passed the guardrail"

    eligible.sort(key=lambda t: (-t[1], t[2]))
    best_tag, best_c100, _ = eligible[0]
    log("selection: eligible -> " + ", ".join(f"{t}(CSI_100={c:.4f})" for t, c, _ in eligible))

    if best_tag != "D0" and best_c100 > ref_c100:
        return best_tag, f"{best_tag} improves CSI_100 {ref_c100:.4f} -> {best_c100:.4f}"
    return "D0", (f"no arm beat the control on CSI_100 "
                  f"(best={best_tag} {best_c100:.4f} vs {ref_c100:.4f})")


def run_eval(model: str, split: str, out: Path, ckpt: Path | None, log_name: str) -> dict:
    cmd = [PY, "-u", "scripts/evaluate.py", "--config", "configs/v0_10m2m_hmax.yaml",
           "--model", model, "--split", split, "--out", str(out)]
    if ckpt is not None:
        cmd += ["--ckpt", str(ckpt)]
    out.parent.mkdir(parents=True, exist_ok=True)
    log(f"eval start: model={model} split={split} ckpt={ckpt if ckpt else '-'}")
    with (POST / log_name).open("a", encoding="utf-8", errors="ignore") as lf:
        code = subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf,
                                stderr=subprocess.STDOUT).wait()
    m = read_json(out)
    log(f"eval done : model={model} split={split} exit={code} "
        f"CSI_005={m.get('CSI_005')} CSI_100={m.get('CSI_100')} RMSE_wet={m.get('RMSE_wet')}")
    return {"exit": code, **m}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="store_true", help="block until SWEEP_DONE appears")
    ap.add_argument("--timeout-h", type=float, default=12.0)
    ap.add_argument("--winner", default=None, help="force a winner tag (skip selection)")
    ap.add_argument("--csi-tol", type=float, default=0.01,
                    help="allowed val CSI_005 drop vs control for an arm to stay eligible")
    ap.add_argument("--epochs", type=int, default=300, help="target epochs for the resumed run")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-resume", action="store_true")
    ap.add_argument("--baselines-only", action="store_true",
                    help="run only the 6 baseline evals (frozen V0 + bilinear + nearest, "
                         "val and test) and exit; no selection and no resume")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    POST.mkdir(parents=True, exist_ok=True)

    if args.baselines_only:
        log("post_sweep: baselines-only evaluation (frozen V0 + bilinear + nearest, val+test)")
        metrics = POST / "metrics"
        results: dict[str, dict] = {}
        plan = [
            ("baseline_v0", "hydrogeo_srno", "val", FROZEN / "best_csi.pt"),
            ("baseline_v0", "hydrogeo_srno", "test", FROZEN / "best_csi.pt"),
            ("bilinear", "bilinear", "val", None),
            ("bilinear", "bilinear", "test", None),
            ("nearest", "nearest", "val", None),
            ("nearest", "nearest", "test", None),
        ]
        for name, model, split, ck in plan:
            key = f"{name}_{split}"
            results[key] = run_eval(model, split, metrics / f"{key}.json", ck, f"eval_{key}.log")
        (POST / "eval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        log("post_sweep: baselines-only evaluation finished")
        return

    log("post_sweep start")

    if args.wait and not args.dry_run:
        log(f"waiting for {SWEEP / 'SWEEP_DONE'}")
        if not wait_for_sweep(args.timeout_h):
            return

    arms = load_arms()
    log(f"arms done: {sorted(arms)}")

    reason = "forced by --winner"
    if args.winner:
        winner = args.winner
    else:
        winner, reason = pick_winner(arms, args.csi_tol)
    if winner is None:
        log(f"no winner ({reason}); stopping")
        return
    log(f"WINNER = {winner}  ({reason})")

    (POST / "selection.json").write_text(json.dumps(
        {"winner": winner, "reason": reason,
         "arms": {k: v.get("metrics") for k, v in arms.items()}}, indent=2), encoding="utf-8")

    if args.dry_run:
        log("dry-run: stopping before evaluation/resume")
        return

    # ---------------------------------------------------------------- clean baseline eval
    metrics = POST / "metrics"
    results: dict[str, dict] = {}
    if not args.skip_eval:
        plan = [
            ("baseline_v0", "hydrogeo_srno", "val", FROZEN / "best_csi.pt"),
            ("baseline_v0", "hydrogeo_srno", "test", FROZEN / "best_csi.pt"),
            ("bilinear", "bilinear", "val", None),
            ("bilinear", "bilinear", "test", None),
            ("nearest", "nearest", "val", None),
            ("nearest", "nearest", "test", None),
        ]
        for name, model, split, ck in plan:
            key = f"{name}_{split}"
            results[key] = run_eval(model, split, metrics / f"{key}.json", ck,
                                    f"eval_{key}.log")

        wck = SWEEP / winner / "best_csi.pt"
        if wck.exists():
            for split in ("val", "test"):
                key = f"winner_{winner}_{split}"
                results[key] = run_eval("hydrogeo_srno", split,
                                        metrics / f"{key}.json", wck,
                                        f"eval_{key}.log")
        (POST / "eval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ resumed training
    if not args.skip_resume:
        if winner == "D0":
            log("winning arm is the control: the deep term did not help, so nothing to resume.")
            log("frozen run left untouched; resume manually if a longer budget is wanted.")
            return

        arm = read_json(SWEEP / winner / "status.json")
        out_dir = ROOT / "outputs" / RESUME_TAG
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [PY, "-u", "scripts/train_fixed.py",
               "--config", "configs/v0_10m2m_hmax.yaml",
               "--out_dir", f"outputs/{RESUME_TAG}",
               "--resume", str(FROZEN / "last.pt"),
               "--w_deep", str(arm.get("w_deep", 0.0)),
               "--deep_delta", str(arm.get("deep_delta", 5.0)),
               "--deep_threshold", str(arm.get("deep_threshold", 1.0)),
               "--epochs", str(args.epochs)]
        log(f"resume: {' '.join(cmd)}")
        log(f"resume: from ep180 last.pt, target {args.epochs} epochs, into outputs/{RESUME_TAG}")
        with (out_dir / "stdout.log").open("a", encoding="utf-8", errors="ignore") as lf:
            p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
        (out_dir / "train.pid").write_text(str(p.pid), encoding="ascii")
        log(f"resume: launched detached pid={p.pid}; log=outputs/{RESUME_TAG}/stdout.log")

    log("post_sweep finished")


if __name__ == "__main__":
    main()
