"""Compute every statistic quoted in the report and dump to stats.json."""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIN = ROOT / "outputs" / "finetune_deep"
scan = json.loads((FIN / "epoch_scan.json").read_text(encoding="utf-8"))
scans, refs = scan["scans"], scan["refs"]
EPS = list(range(181, 201))
# Student t, two-sided 95%, df=19
T975 = 2.093

LOWER_BETTER = {"PeakDepthError", "VolumeRelativeError", "RMSE_wet", "RMSE_all"}
out: dict = {"paired": {}, "winner": {}, "refs": refs, "relative": {}}


def series(arm, m):
    return np.array([scans[arm][f"ep{e:04d}"][m] for e in EPS], dtype=float)


def cohens_d(d: np.ndarray) -> float:
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd else float("inf")


def boot_ci(d: np.ndarray, n=20000, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


for metric in ["CSI_005", "CSI_030", "CSI_100", "RMSE_wet", "PeakDepthError",
               "VolumeRelativeError"]:
    ctl = series("w00", metric)
    for arm in ("w005", "w01"):
        cand = series(arm, metric)
        d = cand - ctl
        md = float(d.mean())
        sd = float(d.std(ddof=1))
        se = sd / math.sqrt(len(d))
        t = md / se if se else float("inf")
        lo, hi = boot_ci(d)
        n_better = int(np.sum(d < 0)) if metric in LOWER_BETTER else int(np.sum(d > 0))
        out["paired"].setdefault(metric, {})[arm] = {
            "mean_diff": md,
            "sd_diff": sd,
            "se": se,
            "t": float(t),
            "ci95_normal": [md - T975 * se, md + T975 * se],
            "ci95_bootstrap": [lo, hi],
            "cohens_d": cohens_d(d),
            "n_better_of_20": n_better,
            "significant": bool(abs(t) > T975),
            "improves": bool(md < 0) if metric in LOWER_BETTER else bool(md > 0),
        }


def round_rel(a: float, b: float, lower_better=False) -> float:
    """Relative change from a to b, signed so positive always means better."""
    if a == 0:
        return float("nan")
    r = (b - a) / a
    return -r * 100 if lower_better else r * 100


val = {
    "frozen": refs["frozen_ep180"],
    "winner": json.loads(json.dumps(scan["scans"]["w01"]["ep0187"])),
    "control_mean": {m: float(series("w00", m).mean()) for m in
                     ["CSI_005", "CSI_030", "CSI_100", "RMSE_wet", "PeakDepthError",
                      "VolumeRelativeError"]},
}
out["winner"]["val"] = val
out["winner"]["test"] = {
    "frozen": json.loads((ROOT / "outputs/post_sweep/metrics/baseline_v0_test.json")
                         .read_text(encoding="utf-8")),
    "winner": json.loads((FIN / "metrics/winner_w01_ep187_test.json").read_text(encoding="utf-8")),
}

for split in ("val", "test"):
    f, w = out["winner"][split]["frozen"], out["winner"][split]["winner"]
    out["relative"][split] = {m: round_rel(f[m], w[m], m in LOWER_BETTER)
                              for m in ["CSI_005", "CSI_030", "CSI_100", "RMSE_wet",
                                        "MAE_wet", "PeakDepthError", "VolumeRelativeError"]}

# from-scratch sweep
sweep = {t: json.loads((ROOT / f"outputs/deep_sweep/{t}/status.json").read_text(encoding="utf-8"))
         ["metrics"] for t in ("D0", "D2")}
out["sweep"] = sweep
out["sweep_relative"] = {m: round_rel(sweep["D0"][m], sweep["D2"][m], m in LOWER_BETTER)
                         for m in ["CSI_005", "CSI_030", "CSI_100", "RMSE_wet", "PeakDepthError",
                                   "VolumeRelativeError"]}

(FIN / "report_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

print("=== paired t-tests (n=20 epochs), 95% CI ===")
for m, arms in out["paired"].items():
    for a, r in arms.items():
        print(f"{m:20s} {a:5s} mean={r['mean_diff']:+.5f} t={r['t']:+.2f} "
              f"CI=[{r['ci95_bootstrap'][0]:+.5f},{r['ci95_bootstrap'][1]:+.5f}] "
              f"d={r['cohens_d']:+.2f} better {r['n_better_of_20']}/20 "
              f"{'SIG' if r['significant'] else ''}")
print("\n=== relative change frozen -> winner (%) ===")
for split, d in out["relative"].items():
    print(f"{split}: " + "  ".join(f"{k}={v:+.2f}" for k, v in d.items()))
print("\n=== sweep relative (%) ===")
print("  ".join(f"{k}={v:+.2f}" for k, v in out["sweep_relative"].items()))
print("\nwrote", FIN / "report_stats.json")
