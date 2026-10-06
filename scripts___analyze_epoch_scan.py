"""Paired full-val analysis of the deep-water fine-tune (decision-grade numbers).

All three arms resume from the SAME frozen ep180 checkpoint and run 20 epochs with identical
seed / data order / lr schedule; the ONLY difference is w_deep. So epoch N of each arm is a
matched condition and the per-epoch difference between a candidate and the w_deep=0 control is
a PAIRED measurement -- the right way to ask "did l_deep do anything?" without the
max-of-noisy-subset bias that the training-time checkpointing suffered from.

Also reports the naive "pick the best epoch per arm" answer, because that is what the driver's
selection does, and shows how much of it is selection bias by doing the selfsame pick on the
control (which cannot gain from a deep term).
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "outputs" / "finetune_deep" / "epoch_scan.json"

j = json.loads(SCAN.read_text(encoding="utf-8"))
scans = j["scans"]
refs = j["refs"]

arms = ["w00", "w005", "w01"]
label = {"w00": "w_deep=0.00 (control)", "w005": "w_deep=0.05", "w01": "w_deep=0.10"}
METRICS = ["CSI_005", "CSI_030", "CSI_100", "RMSE_wet", "PeakDepthError", "VolumeRelativeError"]


def series(arm: str, m: str) -> list[float]:
    return [scans[arm][f"ep{e:04d}"][m] for e in range(181, 201)]


print("=== per-arm summary over the 20 fine-tune epochs (full 182-tile val split) ===")
print(f"{'arm':22s} {'metric':8s} {'mean':>9s} {'stdev':>8s} {'min':>9s} {'max':>9s} "
      f"{'best_ep':>8s} {'at_best':>9s}")
for arm in arms:
    for m in METRICS:
        s = series(arm, m)
        better = min if m in ("RMSE_wet", "PeakDepthError", "VolumeRelativeError") else max
        best_v = better(s)
        best_ep = 181 + s.index(best_v)
        print(f"{label[arm]:22s} {m:8s} {st.mean(s):9.4f} {st.stdev(s):8.4f} "
              f"{min(s):9.4f} {max(s):9.4f} {best_ep:8d} {best_v:9.4f}")

print("\n=== paired per-epoch difference vs the w_deep=0 control (same epoch = matched) ===")
for m in ["CSI_100", "CSI_005", "PeakDepthError", "VolumeRelativeError"]:
    ctl = series("w00", m)
    for arm in ["w005", "w01"]:
        cand = series(arm, m)
        d = [c - b for c, b in zip(cand, ctl)]
        wins = sum(1 for x in d if (x < 0 if m in ("PeakDepthError", "VolumeRelativeError")
                                    else x > 0))
        print(f"{m:22s} {label[arm]:20s} meandiff={st.mean(d):+.5f} "
              f"stdev={st.stdev(d):.5f} median={st.median(d):+.5f} "
              f"range=[{min(d):+.5f},{max(d):+.5f}] cand_better_in {wins}/20 epochs")

print("\n=== reference points (same evaluator, full val split) ===")
for k, v in refs.items():
    print(f"  {k:14s} CSI_005={v.get('CSI_005'):.4f} CSI_030={v.get('CSI_030'):.4f} "
          f"CSI_100={v.get('CSI_100'):.4f} RMSE_wet={v.get('RMSE_wet'):.4f} "
          f"PeakDepthErr={v.get('PeakDepthError'):.4f} VolRelErr={v.get('VolumeRelativeError'):.4f}")

print("\n=== naive 'max CSI_100 epoch' pick per arm (what the driver's selection does) ===")
for arm in arms:
    s = series(arm, "CSI_100")
    bep = 181 + s.index(max(s))
    row = scans[arm][f"ep{bep:04d}"]
    print(f"  {label[arm]:22s} best CSI_100={max(s):.4f} @ep{bep} "
          f"(CSI_005 there={row['CSI_005']:.4f}, PeakDepthErr={row['PeakDepthError']:.4f})")

print("\n=== the decisive paired statement ===")
for m, higher_better in [("CSI_100", True), ("CSI_005", True),
                         ("PeakDepthError", False), ("VolumeRelativeError", False)]:
    ctl = series("w00", m)
    for arm in ["w005", "w01"]:
        cand = series(arm, m)
        d = [c - b for c, b in zip(cand, ctl)]
        md, sd = st.mean(d), st.stdev(d)
        # paired t-like effect size; n=20
        t = md / (sd / 20 ** 0.5) if sd else float("inf")
        verdict = "SIGNIFICANT" if abs(t) > 2.09 else "not significant (|t|<2.09, n=20)"
        sign = "higher" if md > 0 else "lower"
        print(f"  {m:20s} {label[arm]:20s} mean {sign} by {abs(md):.5f} "
              f"(t={t:+.2f}) -> {verdict}")
