"""Shared plotting/style/data helpers for flood-SR visual diagnostics.

All figures are headless (Agg) and written as PNG so they can be produced
while training keeps the GPU busy. Nothing here touches the training process.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
# Which run to visualise. Defaults to the V0 run; override with V0_OUT_DIR (or V0_TAG) so the
# same scripts can visualise the deep-water sweep arms or a resumed run without code changes.
if os.environ.get("V0_OUT_DIR"):
    OUT_DIR = Path(os.environ["V0_OUT_DIR"])
    if not OUT_DIR.is_absolute():
        OUT_DIR = ROOT / OUT_DIR
elif os.environ.get("V0_TAG"):
    OUT_DIR = ROOT / "outputs" / os.environ["V0_TAG"]
else:
    OUT_DIR = ROOT / "outputs" / "v0_10m2m_hmax"
VIZ_DIR = OUT_DIR / "visualizations"

# --------------------------------------------------------------------------
# Colormaps
# --------------------------------------------------------------------------

# Water depth: bone-dry land -> shallow sheet -> deep blue pools
FLOOD_CMAP = LinearSegmentedColormap.from_list(
    "flooddepth",
    [
        (0.00, "#ffffff"),
        (0.03, "#e8f4fb"),
        (0.10, "#bfe3f5"),
        (0.25, "#7fc8ec"),
        (0.45, "#4a9fdc"),
        (0.65, "#2b6fbe"),
        (0.85, "#18408f"),
        (1.00, "#0b1f52"),
    ],
)
FLOOD_CMAP.set_bad("#d9d9d9")

# Agreement map colours: TP / FP / FN / TN
AGREE_CMAP = LinearSegmentedColormap.from_list(
    "agreement",
    [(0.00, "#e0e0e0"), (0.33, "#c0392b"), (0.66, "#f1c40f"), (1.00, "#2e7d32")],
)

TERRAIN_CMAP = "terrain"
DIVERGING = "RdBu_r"

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 170,
            "savefig.bbox": "tight",
            "font.size": 9,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


# --------------------------------------------------------------------------
# History / metric IO
# --------------------------------------------------------------------------

METRIC_LABELS = {
    "loss": "Train loss (FloodLoss)",
    "RMSE_all": "RMSE all (m)",
    "RMSE_wet": "RMSE wet (m)",
    "MAE_wet": "MAE wet (m)",
    "CSI_005": "CSI @ 0.05 m",
    "CSI_030": "CSI @ 0.30 m",
    "CSI_100": "CSI @ 1.00 m",
    "F1_005": "F1 @ 0.05 m",
    "F1_030": "F1 @ 0.30 m",
    "F1_100": "F1 @ 1.00 m",
    "PeakDepthError": "Peak depth error (m)",
    "FloodAreaRelativeError": "Flood area rel. err.",
    "VolumeRelativeError": "Volume rel. err.",
    "PSNR": "PSNR (dB)",
    "SSIM": "SSIM (masked)",
    "sec": "Epoch wall time (s)",
}


def load_history(path: Path | str = OUT_DIR / "history.jsonl") -> list[dict[str, Any]]:
    """Read history.jsonl, tolerating a partially written final line."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # partial tail while the trainer is writing
    rows.sort(key=lambda r: int(r.get("epoch", 0)))
    return rows


def series(rows: Iterable[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    ep, vals = [], []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv != fv:  # NaN
            continue
        ep.append(int(r["epoch"]))
        vals.append(fv)
    return np.asarray(ep, dtype=float), np.asarray(vals, dtype=float)


def load_json(path: Path | str) -> dict | list | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def best_summary() -> dict | None:
    return load_json(OUT_DIR / "best_summary.json")  # type: ignore[return-value]


# --------------------------------------------------------------------------
# Plot primitives
# --------------------------------------------------------------------------


def mark_best(ax, rows: list[dict], key: str, mode: str = "max") -> float | None:
    ep, vals = series(rows, key)
    if vals.size == 0:
        return None
    idx = int(np.argmax(vals) if mode == "max" else np.argmin(vals))
    ax.scatter(
        [ep[idx]], [vals[idx]], s=46, marker="*", zorder=6,
        color="#e67e22", edgecolor="white", linewidth=0.7,
        label=f"best ep{int(ep[idx])} = {vals[idx]:.4f}",
    )
    return float(vals[idx])


def shade_future(ax, last_epoch: int, total_epochs: int) -> None:
    if last_epoch and total_epochs > last_epoch:
        ax.axvspan(last_epoch, total_epochs, color="#9aa0a6", alpha=0.10, lw=0)
        ax.text(
            (last_epoch + total_epochs) / 2, 0.5, "pending",
            transform=ax.get_xaxis_transform(), ha="center", va="center",
            fontsize=7, color="#6b7076", style="italic",
        )


def finish(epochs_total: int = 300) -> None:
    for ax in plt.gcf().axes:
        ax.set_xlim(0, epochs_total)


def save(fig, path: Path, *, also_pdf: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def stamp(fig, text: str) -> None:
    fig.text(0.995, 0.003, text, ha="right", va="bottom", fontsize=6.5, color="#7a7a7a")


# --------------------------------------------------------------------------
# Depth display helpers
# --------------------------------------------------------------------------


def depth_vmax(*arrays: np.ndarray, q: float = 99.8) -> float:
    vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays if a is not None and a.size])
    if vals.size == 0:
        return 1.0
    v = float(np.percentile(vals, q))
    return max(v, 0.25)


def masked_depth(arr: np.ndarray, valid: np.ndarray) -> np.ma.MaskedArray:
    out = np.ma.array(arr.astype(np.float32), mask=~(arr > 0) if valid is None else ~valid.astype(bool))
    return out


def show_depth(
    ax,
    arr: np.ndarray,
    valid: np.ndarray,
    *,
    vmax: float,
    title: str = "",
    cmap=FLOOD_CMAP,
) -> None:
    disp = np.where(valid.astype(bool), arr, np.nan)
    ax.imshow(disp, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
    if title:
        ax.set_title(title, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def add_colorbar(fig, axes, vmax: float, label: str, cmap=FLOOD_CMAP) -> None:
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=list(np.ravel(axes)), fraction=0.018, pad=0.012)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def side_colorbar(fig, ax, im, *, label: str = "", fmt: str = "%.2f", fontsize: int = 6):
    """Attach a colorbar without shrinking the parent axes (uses an inset axes)."""
    cax = ax.inset_axes([1.03, 0.0, 0.035, 1.0])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=fontsize)
    cb.ax.yaxis.set_major_formatter(plt.FormatStrFormatter(fmt))
    if label:
        cb.set_label(label, fontsize=fontsize + 1)
    return cb
