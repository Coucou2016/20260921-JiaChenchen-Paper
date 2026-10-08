#!/usr/bin/env python
"""Assemble the Experiment A / B comparison table and figures from real artefacts.

Every number is read from a JSON/JSONL a run produced; nothing is recomputed or
fabricated.  Produces:

* ``outputs/premodel/exp_ab_comparison.json``  machine-readable headline rows
* ``outputs/premodel/exp_ab_table.md``         short takeaway table for the report
* ``outputs/report_figs/fig86_exp_ab.png``     Experiment A: same-protocol comparison
* ``outputs/report_figs/fig87_exp_b.png``      Experiment B: objective decay + deep bias
* ``outputs/report_figs/fig88_deep_slope.png`` deep-water slope mechanism

Usage
-----
    python -u scripts/make_exp_ab_figs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_JSON = ROOT / "outputs" / "premodel" / "exp_ab_comparison.json"
FIG_A = ROOT / "outputs" / "report_figs" / "fig86_exp_ab.png"
FIG_B = ROOT / "outputs" / "report_figs" / "fig87_exp_b.png"
FIG_C = ROOT / "outputs" / "report_figs" / "fig88_deep_slope.png"
E7_HIST = ROOT / "outputs" / "deep_objective" / "E7" / "history.jsonl"
SLOPE_JSON = ROOT / "outputs" / "premodel" / "exp_ab_deep_slope.json"

SOURCES = {
    "nearest": "outputs/premodel/exp_ab/nearest_val.json",
    "bilinear": "outputs/premodel/exp_ab/bilinear_val.json",
    "frozen_ep180": "outputs/premodel/exp_ab/frozen_ep180_val.json",
    "w00_ep200": "outputs/premodel/exp_ab/w00_ep200_val.json",
    "w01_ep187": "outputs/premodel/exp_ab/w01_ep187_val.json",
    "residual_depth": "outputs/premodel/exp_ab/residual_depth_val.json",
    "residual_nearest": "outputs/premodel/exp_ab/residual_nearest_val.json",
    "E7_continuous": "outputs/premodel/exp_ab/E7_continuous_val.json",
}
CN = {"nearest": "Nearest", "bilinear": "Bilinear", "frozen_ep180": "Frozen (ep180)",
      "w00_ep200": "Control w00 (ep200)", "w01_ep187": "Best fine-tune w01 (ep187)",
      "residual_depth": "Residual (Exp A)", "residual_nearest": "Residual-nearest (Exp A)",
      "E7_continuous": "E7 continuous (Exp B)"}
KEYS = ["RMSE_wet", "MAE_wet", "CSI_005", "CSI_100", "PeakDepthError",
        "VolumeRelativeError", "RMSE_wet_domain", "MAE_wet_domain",
        "Bias_1m_domain", "Bias_3m_domain", "CSI_1.00_domain"]


def load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def build_rows() -> dict:
    rows = {}
    for label, rel in SOURCES.items():
        d = load(ROOT / rel)
        if d is not None:
            rows[label] = {k: d.get(k) for k in KEYS}
    if rows:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def emit_table(rows: dict) -> None:
    lines = ["| Model | Wet RMSE (tile-macro) | Wet RMSE (domain) | Deep bias >=1 m | CSI@1.0 m |",
             "|---|---|---|---|---|"]
    for k in SOURCES:
        r = rows.get(k)
        if not r:
            continue
        def f(key):
            v = r.get(key)
            return f"{v:.4f}" if isinstance(v, (int, float)) else "—"
        lines.append(f"| {CN.get(k,k)} | {f('RMSE_wet')} | {f('RMSE_wet_domain')} | "
                     f"{f('Bias_1m_domain')} | {f('CSI_100')} |")
    md = "\n".join(lines)
    (ROOT / "outputs" / "premodel" / "exp_ab_table.md").write_text(md, encoding="utf-8")
    print(md)


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.style.use(["science", "no-latex"])
    except Exception:
        pass
    plt.rcParams.update({
        # List the CJK faces explicitly so any residual CJK text cannot fall
        # through to a Latin-only face and render as empty boxes.
        "font.family": ["Times New Roman", "Microsoft YaHei", "SimSun", "SimHei",
                        "DejaVu Serif"],
        "font.serif": ["Times New Roman", "Microsoft YaHei", "SimSun", "SimHei",
                       "DejaVu Serif"],
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
        "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    return plt


def fig_exp_a(rows: dict) -> None:
    if not rows:
        _placeholder(FIG_A, "Experiment A has not produced exp_ab_comparison.json yet")
        return
    plt = _style()
    order = [k for k in SOURCES if k in rows]
    panels = [("RMSE_wet_domain", "Domain-pooled wet RMSE (m), lower is better"),
              ("Bias_1m_domain", "Deep (>= 1 m) mean bias (m), closer to 0 is better"),
              ("CSI_100", "Deep-water CSI@1.0 m, higher is better")]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
    for ax, (key, title) in zip(axes, panels):
        vals, labs, cols = [], [], []
        for k in order:
            v = rows[k].get(key)
            if isinstance(v, (int, float)):
                vals.append(v)
                labs.append(CN.get(k, k))
                cols.append("#c0392b" if "residual" in k else
                            ("#1f4e79" if k in ("w01_ep187",) else
                             ("#2e7d32" if k == "E7_continuous" else "#98a2b3")))
        y = np.arange(len(vals))
        ax.barh(y, vals, color=cols, alpha=0.9)
        ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=8)
        ax.invert_yaxis()
        if key == "Bias_1m_domain":
            ax.axvline(0, color="k", lw=0.7)
            ax.set_xlim(min(vals) * 1.12, 0.05)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.3f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=7.2)
        ax.set_title(title, fontsize=9)
    fig.suptitle("Experiments A (bilinear residual) and B (E7 objective): same validation split, same protocol",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    FIG_A.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_A, dpi=300); plt.close(fig)
    print(f"wrote {FIG_A}")


def _placeholder(path: Path, msg: str) -> None:
    """Write a minimal valid PNG so the build pipeline never fails on a missing artefact."""
    plt = _style()
    fig, ax = plt.subplots(figsize=(6, 2.4))
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11, wrap=True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote placeholder {path}")


def fig_exp_b() -> None:
    if not E7_HIST.exists():
        _placeholder(FIG_B, "Experiment B (E7) has not produced history.jsonl yet")
        return
    rows = [json.loads(l) for l in E7_HIST.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        _placeholder(FIG_B, "Experiment B (E7) history.jsonl is empty")
        return
    plt = _style()
    ep = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
    axes[0].plot(ep, [r.get("RMSE_wet_domain") for r in rows], "-o", color="#1f4e79", ms=4)
    axes[0].set_title("Domain-pooled wet RMSE (m), lower is better", fontsize=9)
    axes[0].set_xlabel("epoch")
    ax = axes[1]
    pen = [i for i, r in enumerate(rows) if r.get("Bias_1m_domain") is not None
           and r["Bias_1m_domain"] == r["Bias_1m_domain"]]
    ax.plot([ep[i] for i in pen], [rows[i]["Bias_1m_domain"] for i in pen], "-o",
            color="#c0392b", ms=4, label="≥1 m")
    pen3 = [i for i, r in enumerate(rows) if r.get("Bias_3m_domain") is not None
            and r["Bias_3m_domain"] == r["Bias_3m_domain"]]
    if pen3:
        ax.plot([ep[i] for i in pen3], [rows[i]["Bias_3m_domain"] for i in pen3], "-s",
                color="#2e7d32", ms=4, label="≥3 m")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_title("Deep mean bias (m), closer to 0 is better", fontsize=9)
    ax.set_xlabel("epoch"); ax.legend(fontsize=7.5)
    shares = [r.get("w_deep", np.nan) /
              max(r.get("w_deep", 0) + r.get("w_depth", 0) + r.get("w_log", 0)
                  + 0.30 + 0.10 + 0.10, 1e-9) for r in rows]
    axes[2].plot(ep, shares, "-o", color="#6a1b9a", ms=4)
    axes[2].set_title("Share of the deep term in the objective (nominal)", fontsize=9)
    axes[2].set_xlabel("epoch")
    fig.suptitle("Experiment B (E7): cosine decay of the shallow weights plus a continuous deep weight raise the deep term's share",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_B, dpi=300); plt.close(fig)
    print(f"wrote {FIG_B}")


def fig_deep_slope() -> None:
    d = load(SLOPE_JSON)
    if not d or not d.get("models"):
        _placeholder(FIG_C, "The deep-slope diagnostic has not produced exp_ab_deep_slope.json yet")
        return
    plt = _style()
    ms = d["models"]
    keys = list(ms.keys())
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    labs = [CN.get(k, k) for k in keys]
    ax = axes[0]
    ax.barh(np.arange(len(keys)), [ms[k]["slope_pred_on_10m"] for k in keys],
            color="#1f4e79", alpha=0.9)
    ax.axvline(1.0, color="k", ls="--", lw=0.8, label="1.00 (identity)")
    ax.set_yticks(np.arange(len(keys))); ax.set_yticklabels(labs, fontsize=8)
    ax.invert_yaxis(); ax.set_title("Regression slope of the prediction on the 10 m input over deep pixels", fontsize=9)
    ax.legend(fontsize=7.5)
    ax = axes[1]
    w = 0.38
    y = np.arange(len(keys))
    ax.barh(y - w/2, [ms[k]["mean_needed_lift_m"] for k in keys], height=w,
            color="#c0392b", alpha=0.9, label="Required lift")
    ax.barh(y + w/2, [ms[k]["mean_delivered_lift_m"] for k in keys], height=w,
            color="#2e7d32", alpha=0.9, label="Delivered lift")
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=8)
    ax.invert_yaxis(); ax.set_title("Deep pixels: required vs delivered lift (m)", fontsize=9)
    ax.legend(fontsize=7.5)
    fig.suptitle("Mechanism of deep-water underestimation: does the network amplify the deep signal the interpolation already carries?", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_C, dpi=300); plt.close(fig)
    print(f"wrote {FIG_C}")


def main() -> None:
    rows = build_rows()
    emit_table(rows)
    fig_exp_a(rows)
    fig_exp_b()
    fig_deep_slope()
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
