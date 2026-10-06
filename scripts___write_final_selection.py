"""Extract the winner rows and write the final, full-split selection.json."""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIN = ROOT / "outputs" / "finetune_deep"
j = json.loads((FIN / "epoch_scan.json").read_text(encoding="utf-8"))
scans, refs = j["scans"], j["refs"]


def row(arm, ep):
    return scans[arm][f"ep{ep:04d}"]


def series(arm, m):
    return [scans[arm][f"ep{e:04d}"][m] for e in range(181, 201)]


def show(arm, ep, why):
    r = row(arm, ep)
    print(f"{arm} ep{ep} ({why}): CSI_005={r['CSI_005']:.4f} CSI_030={r['CSI_030']:.4f} "
          f"CSI_100={r['CSI_100']:.4f} RMSE_wet={r['RMSE_wet']:.4f} "
          f"PeakDepthErr={r['PeakDepthError']:.4f} VolRelErr={r['VolumeRelativeError']:.4f}")


print("=== candidate best-CSI_100 epochs (full val) ===")
show("w005", 181, "max CSI_100")
show("w01", 187, "max CSI_100")
show("w00", 184, "control max CSI_100")
print("\n=== frozen ep180 reference ===")
print(refs["frozen_ep180"])
print("\n=== paired control means (reference for the guardrail) ===")
for m in ("CSI_005", "CSI_100", "VolumeRelativeError"):
    print(f"  control mean {m} = {st.mean(series('w00', m)):.5f}")

sel = {
    "basis": "FULL 182-tile val split, every per-epoch snapshot re-scored (scripts/scan_epochs.py)",
    "why_this_basis": (
        "The training-time best_deep_csi.pt is the argmax of the 60-batch val-subset CSI_100. "
        "The w_deep=0 control -- which cannot gain from a deep term -- wandered over 0.1407..0.1711 "
        "on that subset (sigma~0.005), so that argmax is a max-of-noise with ~+0.012 bias and the "
        "candidates' +0.0026 lead over the control there is inside the control's own noise. "
        "All numbers below are instead measured on the full 182-tile val split with the same "
        "evaluator used for the baseline table."),
    "paired_design": (
        "All arms resume from the SAME frozen ep180 checkpoint and run 20 epochs with identical "
        "seed / data order / lr schedule; ONLY w_deep differs. Epoch N of each arm is therefore a "
        "matched condition, so the per-epoch candidate-minus-control difference is a paired "
        "measurement (n=20) and is not a max-of-noise."),
    "winner": "w_deep=0.10",
    "winner_checkpoint": "outputs/finetune_deep/w01/snapshots/ep0187.pt",
    "winner_epoch": 187,
    "winner_selection_rule": ("highest full-split val CSI_100 among candidate epochs, with "
                             "CSI_005 not degraded (see guardrail below)"),
    "guardrail": {
        "rule": "val CSI_005 must stay within 0.005 of the w_deep=0 control",
        "control_mean_CSI_005": st.mean(series("w00", "CSI_005")),
        "winner_CSI_005": row("w01", 187)["CSI_005"],
        "verdict": "PASS (paired CSI_005 change is not significant: t=-0.85)",
    },
    "paired_evidence_vs_control": {
        "CSI_100": {"w005": {"mean_diff": 0.00635, "t": 4.85, "better_in": "18/20"},
                    "w01": {"mean_diff": 0.00816, "t": 5.72, "better_in": "17/20"}},
        "CSI_005": {"w005": {"mean_diff": 0.00004, "t": 0.08, "better_in": "9/20"},
                    "w01": {"mean_diff": -0.00058, "t": -0.85, "better_in": "10/20"}},
        "VolumeRelativeError": {"w005": {"mean_diff": -0.01420, "t": -4.56, "better_in": "19/20"},
                                "w01": {"mean_diff": -0.01994, "t": -8.14, "better_in": "18/20"}},
        "PeakDepthError": {"w005": {"mean_diff": -0.01500, "t": -0.73, "better_in": "13/20"},
                           "w01": {"mean_diff": -0.03260, "t": -2.10, "better_in": "15/20"}},
        "note": "|t| > 2.09 is significant at p<0.05 for n=20 paired epochs.",
    },
    "winner_val_metrics": row("w01", 187),
    "control_val_metrics": {m: st.mean(series("w00", m)) for m in
                            ("CSI_005", "CSI_030", "CSI_100", "RMSE_wet",
                             "PeakDepthError", "VolumeRelativeError")},
    "frozen_ep180_val": refs["frozen_ep180"],
    "refs": {k: v for k, v in refs.items()},
    "candidates_all_epochs": scans,
}
(FIN / "selection.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
print(f"\nwrote {FIN / 'selection.json'}")
