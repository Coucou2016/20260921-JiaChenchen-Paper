"""Figure: paired per-epoch deep-water CSI_100 on the full val split (the decisive evidence)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIN = ROOT / "outputs" / "finetune_deep"
j = json.loads((FIN / "epoch_scan.json").read_text(encoding="utf-8"))
s = j["scans"]
froz = j["refs"]["frozen_ep180"]

eps = list(range(181, 201))
arms = [("w00", "w_deep=0.00 (control)", "#7f7f7f", "o"),
        ("w005", "w_deep=0.05", "#1f77b4", "s"),
        ("w01", "w_deep=0.10", "#d62728", "^")]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
fig.suptitle("Deep-water fine-tune of the frozen V0 (ep180) — full 182-tile val split, "
             "paired across arms", fontsize=12, fontweight="bold")

panels = [("CSI_100", "CSI@1.00 m  (deep-water score)", False),
          ("CSI_005", "CSI@0.05 m  (overall accuracy)", False),
          ("VolumeRelativeError", "Volume relative error (lower better)", True)]

for ax, (metric, title, lower) in zip(axes, panels):
    for arm, name, color, mk in arms:
        y = [s[arm][f"ep{e:04d}"][metric] for e in eps]
        ax.plot(eps, y, marker=mk, ms=3.5, lw=1.6, color=color, label=name)
    ref = froz[metric]
    ax.axhline(ref, color="k", ls="--", lw=1.2, alpha=.7)
    ax.annotate(f"frozen ep180 = {ref:.4f}", xy=(181, ref), xytext=(2, 6),
                textcoords="offset points", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("fine-tune epoch")
    ax.grid(alpha=.25, lw=.5)
    if metric == "CSI_100":
        ax.legend(fontsize=8, loc="lower right")

# shade the deep-water panel to mark the significant paired effect
axes[0].set_facecolor("#fdf3f3")
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = FIN / "visualizations" / "paired_fullval_deep_csi.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160)
print(f"wrote {out}")
