"""One-off: reclassify benign exit=120 fine-tune arms from 'failed' back to 'done'.

exit=120 is produced on this Windows box by the interpreter's stdout-flush warning at
teardown ("Exception ignored on flushing sys.stdout"). It is NOT a trainer failure:
  * the trainer printed "done. checkpoints in ... elapsed_h=..."
  * history.jsonl holds all 20 epochs (181..200)
  * best_deep_csi.pt / best_rmse_wet.pt / last.pt all exist (17 MB each)
  * best_deep_summary.json is fully populated
  * the accepted D0/D2 from-scratch arms also exited 120

Without this, run_deep_finetune.train_arm() reads state="failed" and RETRAINS an arm that has
already finished (~4.4 h each of wasted GPU).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIN = ROOT / "outputs" / "finetune_deep"

NOTE = ("exit=120 is a benign Windows stdout-flush artifact at interpreter teardown, the same "
        "code produced by the accepted D0/D2 arms. Verified complete: 20/20 epochs in "
        "history.jsonl, all checkpoints written, best_deep_summary.json populated.")

# An arm is only reclassified if the artefacts independently prove a full run.
REQUIRED = ["best_deep_csi.pt", "best_rmse_wet.pt", "last.pt",
            "best_deep_summary.json", "history.jsonl"]


def verify(d: Path) -> tuple[bool, str]:
    missing = [f for f in REQUIRED if not (d / f).exists()]
    if missing:
        return False, f"missing artefacts: {missing}"
    hist = [json.loads(l) for l in (d / "history.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if not hist:
        return False, "empty history.jsonl"
    epochs = sorted(int(h["epoch"]) for h in hist)
    if len(hist) < 20 or epochs[-1] - epochs[0] + 1 != len(hist):
        return False, f"incomplete history ({len(hist)} rows, {epochs[0]}..{epochs[-1]})"
    summary = json.loads((d / "best_deep_summary.json").read_text(encoding="utf-8"))
    if not summary.get("best_deep_csi"):
        return False, "best_deep_summary.json has no best_deep_csi"
    return True, f"{len(hist)} epochs {epochs[0]}..{epochs[-1]}, best_deep_csi={summary['best_deep_csi']:.5f}"


def main() -> None:
    for arm in sorted(FIN.glob("w*")):
        sp = arm / "status.json"
        if not sp.exists():
            continue
        st = json.loads(sp.read_text(encoding="utf-8"))
        if st.get("state") != "failed":
            print(f"{arm.name}: state={st.get('state')} -> left alone")
            continue
        ok, why = verify(arm)
        if not ok:
            print(f"{arm.name}: NOT reclassified ({why})")
            continue
        st["init_state"] = "failed"
        st["state"] = "done"
        st["reclassified"] = NOTE
        st.pop("failure", None)
        sp.write_text(json.dumps(st, indent=2), encoding="utf-8")
        print(f"{arm.name}: failed -> done  [{why}]")


if __name__ == "__main__":
    main()
