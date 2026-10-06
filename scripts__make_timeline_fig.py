"""Render the research-timeline figure as a PNG (for the Markdown and PDF twins)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "savefig.dpi": 200, "savefig.bbox": "tight",
})

stages = [
    ("Baseline", "train until\nplateau", "#1f4e79"),
    ("Diagnosis", "two loss\ndefects", "#1f4e79"),
    ("From scratch", "strong weight\nrejected", "#b03030"),
    ("Fine-tune", "gentle weight\nkept", "#1a7a3a"),
    ("Extend", "does the gain\npersist?", "#1f4e79"),
]
fig, ax = plt.subplots(figsize=(10.0, 2.3))
ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 2.3)
ax.plot([0.6, 9.4], [1.1, 1.1], color="#d8dde3", lw=4, solid_capstyle="round", zorder=1)
xs = [1.2, 3.1, 5.0, 6.9, 8.8]
for x, (title, sub, col) in zip(xs, stages):
    ax.scatter([x], [1.1], s=520, color=col, zorder=3, edgecolors="white", linewidths=1.6)
    ax.text(x, 1.62, title, ha="center", va="bottom", fontsize=11, fontweight="bold", color=col)
    ax.text(x, 0.62, sub, ha="center", va="top", fontsize=9, color="#333")
fig.savefig(ROOT / "outputs/report_figs/fig00_timeline.png")
print("wrote fig00_timeline.png")
