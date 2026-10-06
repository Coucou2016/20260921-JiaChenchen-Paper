#!/usr/bin/env python
"""Per-pixel physical diagnostics of the learned SR operator (CPU only).

Where does the model gain or lose relative to bilinear interpolation?

Accumulates, over a split:
  * signed bias vs true depth (is deep water consistently under-predicted?)
  * error vs terrain slope (does relief break the operator?)
  * error by land-use class (buildings / roads / green / water)
  * per-tile CSI: model vs bilinear scatter + win rate
  * per-tile volume ratio (mass anchoring check)
  * residual anchoring: mean learned dz vs coarse-vs-truth mismatch

Outputs visualizations/03_diagnostics_<split>.png and diagnostics_<split>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from dataset.normalization import StaticNormalizer
from dataset.wellington_fixed_sr import WellingtonFixedSRDataset, collate_fixed
from engine.checkpoint import load_checkpoint
from engine.evaluator import build_model
from metrics.flood_metrics import compute_flood_metrics
from viz.common import OUT_DIR, VIZ_DIR, apply_style, save, stamp

LANDUSE_NAMES = {0: "nodata/0", 1: "building", 2: "road", 3: "impervious",
                 4: "pervious", 5: "missing", 6: "green", 7: "water"}
DEPTH_BINS = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 1e3])
SLOPE_BINS = np.array([0, 0.5, 1, 2, 3, 5, 8, 12, 20, 1e3])


def bin_index(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, bins) - 1, 0, len(bins) - 2)


def bin_labels(bins: np.ndarray) -> list[str]:
    out = []
    for i in range(len(bins) - 1):
        hi = bins[i + 1]
        out.append(f"{bins[i]:g}-{'inf' if hi >= 1e3 else format(hi, 'g')}")
    return out


class Accumulator:
    """Sufficient statistics for binned error analysis."""

    def __init__(self, nbins: int) -> None:
        self.n = np.zeros(nbins)
        self.abs_err = np.zeros(nbins)
        self.sq_err = np.zeros(nbins)
        self.signed = np.zeros(nbins)
        self.under = np.zeros(nbins)
        self.over = np.zeros(nbins)

    def add(self, idx: np.ndarray, pred: np.ndarray, truth: np.ndarray) -> None:
        d = pred - truth
        np.add.at(self.n, idx, 1.0)
        np.add.at(self.abs_err, idx, np.abs(d))
        np.add.at(self.sq_err, idx, d * d)
        np.add.at(self.signed, idx, d)
        np.add.at(self.under, idx, (d < -0.01).astype(float))
        np.add.at(self.over, idx, (d > 0.01).astype(float))

    def summary(self) -> dict[str, np.ndarray]:
        n = np.maximum(self.n, 1)
        return {
            "n": self.n,
            "mae": self.abs_err / n,
            "rmse": np.sqrt(self.sq_err / n),
            "bias": self.signed / n,
            "under_frac": self.under / n,
            "over_frac": self.over / n,
        }


def build_ds(cfg: dict, split: str) -> WellingtonFixedSRDataset:
    dc = cfg["dataset"]
    return WellingtonFixedSRDataset(
        root=ROOT / dc.get("root", "dataset"), split=split,
        lr_res=int(dc.get("lr_res", 10)), hr_res=int(dc.get("hr_res", 2)),
        target=dc.get("target", "h_max"), geo_mode=dc.get("geo_mode", "all"),
        scenarios=tuple(dc.get("scenarios", ["20a", "100a"])),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0_10m2m_hmax.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    apply_style()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    ckpt = Path(args.ckpt) if args.ckpt else OUT_DIR / "best_csi.pt"

    model = build_model(cfg["model"]["name"], cfg)
    load_checkpoint(str(ckpt), model, map_location="cpu")
    model.eval()

    ds = build_ds(cfg, args.split)
    norm = StaticNormalizer(hr_res=int(cfg["dataset"].get("hr_res", 2)))
    n = len(ds) if args.max_tiles is None else min(len(ds), args.max_tiles)
    print(f"diagnostics on {args.split}: {n} tiles, ckpt={ckpt.name}")

    acc_depth = Accumulator(len(DEPTH_BINS) - 1)
    acc_slope = Accumulator(len(SLOPE_BINS) - 1)
    acc_lu = Accumulator(8)

    per_tile: list[dict] = []
    anchor_resid, anchor_dc = [], []
    n_pix = 0
    t0 = time.time()

    with torch.no_grad():
        for i in range(n):
            s = ds[i]
            batch = collate_fixed([s])
            out = model(
                lr=batch["lr"], lr_valid=batch["lr_valid"],
                static_cont=batch["static_cont"], landuse=batch["landuse"],
                lr_res=batch["lr_res"], hr_res=batch["hr_res"],
            )
            pred = out["depth"][0, 0].numpy()
            base = out["base"][0, 0].numpy()
            truth = batch["hr"][0, 0].numpy()
            valid = batch["mask"][0].numpy().astype(bool)
            residual = out["residual"][0, 0].numpy()

            # raw physical fields for conditioning
            slope = batch["static_cont"][0, 1].numpy() * float(norm.std[1]) + float(norm.mean[1])
            lu = np.clip(batch["landuse"][0].numpy().astype(int), 0, 7)

            if valid.sum() == 0:
                continue
            n_pix += int(valid.sum())
            acc_depth.add(bin_index(truth[valid], DEPTH_BINS), pred[valid], truth[valid])
            acc_slope.add(bin_index(np.abs(slope[valid]), SLOPE_BINS), pred[valid], truth[valid])
            acc_lu.add(lu[valid], pred[valid], truth[valid])

            m = compute_flood_metrics(out["depth"], batch["hr"], batch["mask"])
            mb = compute_flood_metrics(torch.from_numpy(base)[None, None], batch["hr"], batch["mask"])
            per_tile.append({
                "index": i, "scenario": s["scenario"], "iy": int(s["iy"]), "ix": int(s["ix"]),
                "model": {k: float(val) for k, val in m.items()},
                "bilinear": {k: float(val) for k, val in mb.items()},
            })

            # ---- residual anchoring check -------------------------------
            lr = batch["lr"][0, 0].numpy()
            scale = max(1, round(truth.shape[-1] / max(lr.shape[-1], 1)))
            lr_up = np.repeat(np.repeat(lr, scale, axis=-2), scale, axis=-1)[
                : truth.shape[0], : truth.shape[1]]
            pos = valid & (truth > 0.05)
            if pos.sum() > 0:
                anchor_dc.append(float((truth[pos] - lr_up[pos]).mean()))
                anchor_resid.append(float(residual[pos].mean()))

            if (i + 1) % 40 == 0:
                print(f"  {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(15.0, 9.6))
    gs = fig.add_gridspec(3, 3, hspace=0.50, wspace=0.28)

    d = acc_depth.summary()
    x = np.arange(len(DEPTH_BINS) - 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(x, d["bias"], color=np.where(d["bias"] < 0, "#c62828", "#2e7d32"), alpha=0.9)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(bin_labels(DEPTH_BINS), rotation=45, ha="right", fontsize=6.5)
    ax.set_title("Signed bias vs true depth (pred - truth)")
    ax.set_ylabel("mean bias (m)"); ax.set_xlabel("true depth bin (m)")
    ax2 = ax.twinx()
    ax2.plot(x, d["n"] / max(d["n"].sum(), 1) * 100, color="#1565c0", marker="o", ms=3, lw=1.2,
             label="% of pixels")
    ax2.set_ylabel("% of valid pixels", color="#1565c0", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#1565c0", labelsize=7)
    ax2.grid(False); ax2.legend(loc="upper right")

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(x, d["rmse"], color="#e65100", alpha=0.85, label="RMSE")
    ax.plot(x, d["mae"], "k-o", ms=3, lw=1.4, label="MAE")
    ax.set_xticks(x); ax.set_xticklabels(bin_labels(DEPTH_BINS), rotation=45, ha="right", fontsize=6.5)
    ax.set_title("Error magnitude by depth bin")
    ax.set_ylabel("m"); ax.set_xlabel("true depth bin (m)")
    ax.legend()

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(x, d["under_frac"] * 100, color="#c62828", alpha=0.8, label="under-pred (< -1 cm)")
    ax.bar(x, -d["over_frac"] * 100, color="#1565c0", alpha=0.8, label="over-pred (> +1 cm)")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(bin_labels(DEPTH_BINS), rotation=45, ha="right", fontsize=6.5)
    ax.set_title("Directional error share")
    ax.set_ylabel("% of pixels in bin"); ax.set_xlabel("true depth bin (m)")
    ax.legend(fontsize=7)

    sd = acc_slope.summary()
    xs = np.arange(len(SLOPE_BINS) - 1)
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(xs, sd["rmse"], color="#6a1b9a", alpha=0.85)
    ax2 = ax.twinx()
    ax2.plot(xs, sd["n"] / max(sd["n"].sum(), 1) * 100, color="#00695c", marker="s", ms=3, lw=1.2)
    ax2.set_ylabel("% of pixels", color="#00695c", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#00695c", labelsize=7)
    ax2.grid(False)
    ax.set_xticks(xs); ax.set_xticklabels(bin_labels(SLOPE_BINS), rotation=45, ha="right", fontsize=6.5)
    ax.set_title("Error vs terrain slope")
    ax.set_ylabel("RMSE (m)"); ax.set_xlabel("slope bin (deg)")

    ld = acc_lu.summary()
    present = ld["n"] > 0
    xl = np.arange(int(present.sum()))
    labels = [LANDUSE_NAMES[c] for c in np.arange(8)[present]]
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(xl, ld["rmse"][present], color="#1565c0", alpha=0.85)
    for xi, (m_, n_) in enumerate(zip(ld["rmse"][present], ld["n"][present])):
        ax.text(xi, m_, f"{n_/1000:.0f}k", ha="center", va="bottom", fontsize=6.5, color="#37474f")
    ax.set_xticks(xl); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_title("Error by land-use class (labels = pixel count)")
    ax.set_ylabel("RMSE (m)")

    ax = fig.add_subplot(gs[1, 2])
    ax.bar(xl, ld["bias"][present], color=np.where(ld["bias"][present] < 0, "#c62828", "#2e7d32"))
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(xl); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_title("Bias by land-use class")

    mc = np.array([t["model"]["CSI_005"] for t in per_tile])
    bc = np.array([t["bilinear"]["CSI_005"] for t in per_tile])
    ax = fig.add_subplot(gs[2, 0])
    if mc.size:
        lo = min(mc.min(), bc.min()) - 0.02
        hi = max(mc.max(), bc.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.scatter(bc, mc, s=18, c=np.where(mc > bc, "#2e7d32", "#c62828"), alpha=0.8)
        ax.set_title(f"Per-tile CSI@0.05: model vs bilinear\n"
                     f"model better on {float((mc > bc).mean())*100:.0f}% of tiles "
                     f"(mean delta {np.mean(mc-bc):+.3f})")
        ax.set_xlabel("bilinear CSI"); ax.set_ylabel("model CSI")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    ax = fig.add_subplot(gs[2, 1])
    ratio = [(t["bilinear"].get("VolumeRelativeError", np.nan),
              t["model"].get("VolumeRelativeError", np.nan)) for t in per_tile]
    ratio = [r for r in ratio if np.isfinite(r[0]) and np.isfinite(r[1])]
    if ratio:
        r = np.array(ratio)
        ax.scatter(r[:, 0], r[:, 1], s=16,
                   c=np.where(r[:, 1] < r[:, 0], "#2e7d32", "#c62828"), alpha=0.8)
        m = max(float(r.max()), 0.2)
        ax.plot([0, m], [0, m], "k--", lw=1)
        ax.set_xlabel("bilinear volume rel. err.")
        ax.set_ylabel("model volume rel. err.")
        ax.set_title(f"Mass accuracy (model better on "
                     f"{float((r[:, 1] < r[:, 0]).mean())*100:.0f}% of tiles)")

    ax = fig.add_subplot(gs[2, 2])
    if anchor_resid:
        ar = np.array(anchor_resid); ad = np.array(anchor_dc)
        lim = max(float(np.abs(np.concatenate([ar, ad])).max()), 0.1)
        ax.scatter(ad, ar, s=18, alpha=0.8, c="#6a1b9a")
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_xlabel("mean (truth - coarse) over wet cells  [dz)")
        ax.set_ylabel("mean learned residual dz")
        if ad.size >= 2 and float(np.std(ad)) > 1e-9:
            r_ = float(np.corrcoef(ad, ar)[0, 1])
            slope = float(np.polyfit(ad, ar, 1)[0])
            title = f"Residual anchoring (r = {r_:.3f}, slope = {slope:.2f} vs 1.00)"
        else:
            title = "Residual anchoring"
        ax.set_title(title)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    fig.suptitle(
        f"HydroGeo-SRNO physical diagnostics - {args.split} split, {len(per_tile)} tiles, "
        f"checkpoint {ckpt.name}  ({n_pix/1e6:.2f}M valid pixels)",
        fontsize=13, fontweight="bold",
    )
    stamp(fig, f"per-pixel sufficient statistics, CPU only, ckpt={ckpt}")
    p = save(fig, VIZ_DIR / f"03_diagnostics_{args.split}.png")
    print(f"wrote {p}")

    out = {
        "checkpoint": str(ckpt), "split": args.split, "n_tiles": len(per_tile),
        "n_valid_pixels": n_pix,
        "depth_bins": bin_labels(DEPTH_BINS),
        "depth": {k: v.tolist() for k, v in d.items()},
        "slope_bins": bin_labels(SLOPE_BINS),
        "slope": {k: v.tolist() for k, v in sd.items()},
        "landuse": {LANDUSE_NAMES[c]: {k: float(v[c]) for k, v in ld.items()}
                    for c in np.arange(8) if present[c]},
        "per_tile": per_tile,
    }
    (VIZ_DIR / f"diagnostics_{args.split}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {VIZ_DIR / ('diagnostics_' + args.split + '.json')}")


if __name__ == "__main__":
    main()
