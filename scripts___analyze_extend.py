"""Paired analysis of the EXTENSION fine-tune (from the ep187 winner).

Answers two questions the first fine-tune could not:
  1. Does the deep-term gain persist/shrink when training continues past ep187?
  2. Does the >3 m bias defect get fixed, or worsen further?

The extension runs BOTH arms from the same parent (w01/snapshots/ep0187.pt) for 20 epochs, so
per-epoch differences are paired again -- the only difference is w_deep. It also compares each
extended arm against the PARENT (ep187) to answer "did extending help at all, in absolute terms".
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIN = ROOT / "outputs" / "finetune_deep"

EXT = FIN / "epoch_scan_ext.json"
FIRST = FIN / "epoch_scan.json"

EXT_ARMS = {"w0ext": "w_deep=0.00 (ext control)", "w01ext": "w_deep=0.10 (ext)"}
METRICS = ["CSI_005", "CSI_030", "CSI_100", "RMSE_wet", "PeakDepthError", "VolumeRelativeError"]


def main() -> None:
    if not EXT.exists():
        print(f"not ready: {EXT} does not exist yet (scan still running)")
        return
    j = json.loads(EXT.read_text(encoding="utf-8"))
    scans = j["scans"]

    def have(arm: str) -> bool:
        return arm in scans and bool(scans[arm])

    eps = sorted({int(k[2:]) for v in scans.values() for k in v})
    print(f"=== extension epochs present: {eps[0]}..{eps[-1]} (n={len(eps)}) ===")

    print("\n=== per-arm summary over the extension epochs (full 182-tile val) ===")
    print(f"{'arm':24s} {'metric':8s} {'mean':>9s} {'stdev':>8s} {'best_ep':>8s} {'at_best':>9s}")
    for arm, name in EXT_ARMS.items():
        if not have(arm):
            continue
        for m in METRICS:
            s = [scans[arm][f"ep{e:04d}"][m] for e in eps if f"ep{e:04d}" in scans[arm]]
            better = min if m in ("RMSE_wet", "PeakDepthError", "VolumeRelativeError") else max
            bv = better(s)
            bep = eps[s.index(bv)]
            print(f"{name:24s} {m:8s} {st.mean(s):9.4f} {st.stdev(s):8.4f} {bep:8d} {bv:9.4f}")

    print("\n=== paired: w_deep=0.10(ext) minus extension control, same epoch ===")
    if have("w0ext") and have("w01ext"):
        for m in ["CSI_100", "CSI_005", "PeakDepthError", "VolumeRelativeError"]:
            ctl = [scans["w0ext"][f"ep{e:04d}"][m] for e in eps if f"ep{e:04d}" in scans["w0ext"]]
            cand = [scans["w01ext"][f"ep{e:04d}"][m] for e in eps if f"ep{e:04d}" in scans["w01ext"]]
            n = min(len(ctl), len(cand))
            d = [c - b for c, b in zip(cand[:n], ctl[:n])]
            wins = sum(1 for x in d if (x < 0 if m in ("PeakDepthError", "VolumeRelativeError")
                                        else x > 0))
            md, sd = st.mean(d), st.stdev(d) if n > 1 else 0
            t = md / (sd / n ** 0.5) if sd else float("inf")
            verdict = "SIGNIFICANT" if abs(t) > 2.09 else "not significant"
            print(f"  {m:20s} meandiff={md:+.5f} t={t:+.2f} better_in {wins}/{n}  -> {verdict}")

    # Absolute comparison against the parent the extension started from.
    if FIRST.exists():
        fj = json.loads(FIRST.read_text(encoding="utf-8"))
        parent = fj["scans"]["w01"]["ep0187"]
        print("\n=== vs the PARENT (first fine-tune winner, w_deep=0.10 ep187) ===")
        print(f"  parent: CSI_100={parent['CSI_100']:.4f} CSI_005={parent['CSI_005']:.4f} "
              f"PeakDepthErr={parent['PeakDepthError']:.4f} VolErr={parent['VolumeRelativeError']:.4f}")
        for arm, name in EXT_ARMS.items():
            if not have(arm):
                continue
            v = [scans[arm][f"ep{e:04d}"] for e in eps if f"ep{e:04d}" in scans[arm]]
            best = max(v, key=lambda r: r["CSI_100"])
            bep = eps[[scans[arm][f"ep{e:04d}"]["CSI_100"] for e in eps
                       if f"ep{e:04d}" in scans[arm]].index(best["CSI_100"])]
            print(f"  {name:24s} best CSI_100={best['CSI_100']:.4f} @ep{bep} "
                  f"(vs parent {best['CSI_100']-parent['CSI_100']:+.4f}); "
                  f"CSI_005={best['CSI_005']:.4f}; PeakDepthErr={best['PeakDepthError']:.4f}; "
                  f"VolErr={best['VolumeRelativeError']:.4f}")


if __name__ == "__main__":
    main()
