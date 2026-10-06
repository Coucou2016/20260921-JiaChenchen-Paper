"""Two extra process figures: V0 training history and the loss decomposition."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report_figs"
plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 10, "axes.labelsize": 11,
    "axes.titlesize": 11, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
})

hist = [json.loads(l) for l in
        (ROOT / "outputs/v0_10m2m_hmax/history.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip() and "CSI_005" in l]
ep = [int(r["epoch"]) for r in hist]
csi = [r["CSI_005"] for r in hist]
loss = [r.get("loss") or r.get("train_loss") for r in hist]

fig, ax1 = plt.subplots(figsize=(7.6, 3.4))
ax1.plot(ep, csi, lw=1.2, color="#1f4e79", label="CSI@0.05 m")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("CSI@0.05 m (validation)", color="#1f4e79")
ax1.tick_params(axis="y", labelcolor="#1f4e79")
ax1.axvline(98, color="#888", ls=":", lw=1.0)
ax1.axvline(155, color="#c0392b", ls="--", lw=1.0)
ax1.annotate("best checkpoint", xy=(155, max(csi)), xytext=(120, max(csi) - 0.045),
             fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
ax1.text(100, min(csi) + 0.01, "improvement stops\nnear epoch 98-130", fontsize=8, color="#555")
if loss[0] is not None:
    ax2 = ax1.twinx()
    ax2.plot(ep, loss, lw=0.9, color="#e08b2b", alpha=0.75, label="training loss")
    ax2.set_ylabel("training loss", color="#e08b2b")
    ax2.tick_params(axis="y", labelcolor="#e08b2b")
    ax2.grid(False)
ax1.set_title("Baseline V0 training and its plateau")
save = OUT / "fig11_training_history.png"
fig.savefig(save); plt.close(fig); print("wrote", save.name)

# ---- loss decomposition -------------------------------------------------
comp = [("Depth (Huber)", 0.0695, 1.00), ("Extreme tail", 0.6567, 0.10),
        ("Log residual", 0.1599, 0.20), ("Boundary", 0.3104, 0.10),
        ("Wet L1", 0.0155, 0.30)]
names = [c[0] for c in comp]
raw = np.array([c[1] for c in comp])
wt = np.array([c[2] for c in comp])
weighted = raw * wt
share = weighted / weighted.sum() * 100

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.4))
x = np.arange(len(names))
b = a1.bar(x, weighted, color=["#4a7ab5", "#c0392b", "#7aa6c2", "#9db8cc", "#c9d6e0"],
           ec="black", lw=0.6)
a1.bar_label(b, fmt="%.4f", fontsize=7.5, padding=1.5)
a1.set_xticks(x); a1.set_xticklabels(names, rotation=22, ha="right")
a1.set_ylabel("Weighted contribution to loss")
a1.set_title("(a) What the objective actually contains")
b2 = a2.bar(x, share, color=["#4a7ab5", "#c0392b", "#7aa6c2", "#9db8cc", "#c9d6e0"],
            ec="black", lw=0.6)
a2.bar_label(b2, fmt="%.1f%%", fontsize=7.5, padding=1.5)
a2.set_xticks(x); a2.set_xticklabels(names, rotation=22, ha="right")
a2.set_ylabel("Share of the objective (%)")
a2.set_ylim(0, max(share) * 1.25)
a2.set_title("(b) Log-space terms dominate the supervision")
fig.savefig(OUT / "fig12_loss_parts.png"); plt.close(fig)
print("wrote fig12_loss_parts.png")
