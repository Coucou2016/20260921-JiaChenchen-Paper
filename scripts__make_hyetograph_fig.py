"""Rainfall hyetograph (the dataset's genuine time series) in SciencePlots style.

The stored series is a per-second depth increment over a 6-hour design storm. Summing it into
10-minute blocks recovers the piecewise-constant design profile, which is the form hydrologists
read; the raw per-second peak is a single transitional step and is not meaningful on its own.
"""
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

BLOCK = 600  # 10 minutes
scen = [("20a", "#2a7fbf", "20-year, 73.2 mm"),
        ("100a", "#c0392b", "100-year, 96.6 mm"),
        ("50a", "#8c8c8c", "50-year, 86.4 mm (rain only)")]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.5))

for name, col, lab in scen:
    inc = np.load(ROOT / f"dataset/rain/{name}.npy").astype(np.float64)
    nb = inc.size // BLOCK
    inten = inc[:nb * BLOCK].reshape(nb, BLOCK).sum(axis=1) * 1000.0 / (BLOCK / 3600.0)
    t = (np.arange(nb) + 0.5) * BLOCK / 3600.0
    ls = "--" if name == "50a" else "-"
    a1.step(t, inten, where="mid", color=col, ls=ls, lw=1.4, label=lab)
    if name != "50a":
        a1.fill_between(t, 0, inten, step="mid", color=col, alpha=0.10)
    a2.plot(np.arange(inc.size) / 3600.0, np.cumsum(inc) * 1000.0, color=col, ls=ls, lw=1.6,
            label=lab)

a1.set_xlabel("Time (h)"); a1.set_ylabel("Rainfall intensity (mm/h)")
a1.set_title("(a) Design hyetograph, 10-minute blocks")
a1.set_xlim(0, 6); a1.set_ylim(0, 165)
a1.annotate("peak burst\n136.7 mm/h", xy=(3.0, 137), xytext=(3.5, 150),
            fontsize=8, color="#c0392b",
            arrowprops=dict(arrowstyle="->", lw=0.9, color="#c0392b"))

a2.set_xlabel("Time (h)"); a2.set_ylabel("Cumulative rainfall (mm)")
a2.set_title("(b) Cumulative depth")
a2.set_xlim(0, 6); a2.set_ylim(0, 105)
a2.legend(frameon=False, fontsize=8, loc="upper left")

fig.savefig(OUT / "fig20_hyetograph.png")
plt.close(fig)
print("wrote fig20_hyetograph.png")

inc = np.load(ROOT / "dataset/rain/100a.npy").astype(np.float64)
nb = inc.size // BLOCK
inten = inc[:nb * BLOCK].reshape(nb, BLOCK).sum(axis=1) * 1000.0 / (BLOCK / 3600.0)
print("100a peak 10-min intensity: %.1f mm/h at t=%.1f h" % (inten.max(), (np.argmax(inten) + 0.5) / 6))
print("100a baseline intensity: %.1f mm/h" % np.median(inten[inten > 0]))
print("zero-run at start: %.1f min" % (np.argmax(inc > 0) / 60.0))
summary = json.loads((ROOT / "dataset/rain/summary.json").read_text(encoding="utf-8"))
for k, v in summary.items():
    print(k, "total=%.1f mm" % v["total_mm"], "flood output:", v["has_flood_output"])
