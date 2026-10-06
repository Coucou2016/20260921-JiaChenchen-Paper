"""Pick informative test tiles and render the spatial comparison figures.

Selection is driven by the ground truth alone: tiles are ranked by how many pixels exceed 1 m of
water, which is what a reader means by "a deep urban flood", rather than by the single deepest
pixel (that tends to land on the harbour, where the field is almost constant and uninformative).
Only the chosen tiles get a model forward pass, so this is cheap.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import scienceplots  # noqa: F401
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.wellington_fixed_sr import WellingtonFixedSRDataset, collate_fixed  # noqa: E402
from engine.checkpoint import load_checkpoint  # noqa: E402
from engine.evaluator import build_model  # noqa: E402
from viz.reconstruct import predict_tile  # noqa: E402

OUT = ROOT / "outputs" / "report_figs"
F = OUT / "result_fields"

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Microsoft YaHei",
                                            "SimSun", "SimHei", "DejaVu Serif"],
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix", "font.size": 9.5, "axes.labelsize": 10,
    "axes.titlesize": 10.5, "legend.fontsize": 8.5, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "figure.dpi": 200, "savefig.dpi": 200,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
})
FLOOD = LinearSegmentedColormap.from_list("flood", [
    (0.00, "#ffffff"), (0.03, "#eaf4fb"), (0.10, "#c2e2f5"), (0.25, "#7fc8ec"),
    (0.45, "#4a9fdc"), (0.65, "#2b6fbe"), (0.85, "#18408f"), (1.00, "#0b1f52")])
FLOOD.set_bad("#e6e6e6")

CKPTS = {"frozen": ROOT / "outputs/v0_10m2m_hmax/last.pt",
         "win": ROOT / "outputs/finetune_deep/w01/snapshots/ep0187.pt"}


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/v0_10m2m_hmax.yaml").read_text(encoding="utf-8"))
    dc = cfg["dataset"]
    ds = WellingtonFixedSRDataset(
        root=ROOT / dc.get("root", "dataset"), split="test",
        lr_res=int(dc["lr_res"]), hr_res=int(dc["hr_res"]), target=dc["target"],
        geo_mode=dc["geo_mode"], scenarios=tuple(dc["scenarios"]))

    meta = []
    for i in range(len(ds)):
        s = ds[i]
        hr = np.asarray(s["hr"])[0]
        m = np.asarray(s["mask"]).astype(bool)
        deep = int(((hr > 1.0) & m).sum())
        mid = int(((hr > 0.30) & m).sum())
        meta.append({"index": i, "scenario": s["scenario"], "iy": int(s["iy"]), "ix": int(s["ix"]),
                     "deep_px": deep, "mid_px": mid,
                     "peak": float(hr[m].max()) if m.any() else 0.0,
                     "valid": float(m.mean())})
        if (i + 1) % 60 == 0:
            print(f"  scanned {i+1}/{len(ds)}", flush=True)
    (F / "tile_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    by_mid = sorted([r for r in meta if r["mid_px"] > 0], key=lambda r: r["mid_px"])
    def q(p):
        return by_mid[min(int(p * (len(by_mid) - 1)), len(by_mid) - 1)]
    deep_t = max(meta, key=lambda r: r["deep_px"])
    picks, seen = [deep_t], {deep_t["index"]}
    for p in (0.62, 0.26):
        r = q(p)
        while r["index"] in seen and r is not by_mid[0]:
            r = q(p - 0.02)
        seen.add(r["index"])
        picks.append(r)
    if len(picks) < 3:  # pathological fallback
        picks = [by_mid[-1], by_mid[len(by_mid) // 2], by_mid[0]]
    print("picked tiles:")
    for r in picks:
        print(f"  idx {r['index']:>3} {r['scenario']} iy={r['iy']} ix={r['ix']} "
              f"peak={r['peak']:.2f} m  deep_px={r['deep_px']:,}  mid_px={r['mid_px']:,}")

    models = {}
    for k, ck in CKPTS.items():
        m = build_model(cfg["model"]["name"], cfg)
        load_checkpoint(str(ck), m, map_location="cpu")
        m.eval()
        models[k] = m

    tiles = []
    for r in picks:
        batch = collate_fixed([ds[r["index"]]])
        tp = {k: predict_tile(models[k], batch, device=torch.device("cpu")) for k in models}
        tiles.append({**r, "gt": tp["frozen"].gt, "coarse": tp["frozen"].coarse,
                      "base": tp["frozen"].base, "frozen": tp["frozen"].pred,
                      "win": tp["win"].pred, "valid": tp["frozen"].valid,
                      "dem": tp["frozen"].dem,
                      "metrics": {k: tp[k].metrics for k in tp}})
        print(f"  infer idx {r['index']}: frozen CSI_100={tp['frozen'].metrics['CSI_100']:.4f} "
              f"win CSI_100={tp['win'].metrics['CSI_100']:.4f}")

    np.savez_compressed(F / "tiles_chosen.npz", **{
        f"t{k}_{f}": t[f] for k, t in enumerate(tiles)
        for f in ("gt", "coarse", "base", "frozen", "win", "valid", "dem")})
    (F / "tiles_chosen_meta.json").write_text(json.dumps(
        [{k: (v if not isinstance(v, np.ndarray) else None) for k, v in t.items()
          if k in ("index", "scenario", "iy", "ix", "peak", "deep_px", "mid_px")}
         for t in tiles], indent=2), encoding="utf-8")

    # ---------------- fig 21: the deep urban tile -------------------------
    t = tiles[0]
    gt, fr, wn = t["gt"], t["frozen"], t["win"]
    v, base, coarse, dem = t["valid"], t["base"], t["coarse"], t["dem"]
    sc = gt.shape[-1] // coarse.shape[-1]
    cup = np.repeat(np.repeat(coarse, sc, 0), sc, 1)[:gt.shape[0], :gt.shape[1]]
    vmax = float(np.nanpercentile(gt[v], 99.5))

    fig, axes = plt.subplots(2, 4, figsize=(12.6, 6.6))
    def dp(ax, arr, title, cmap=FLOOD, vmax_=vmax):
        a = np.where(v, arr, np.nan)
        im = ax.imshow(a, cmap=cmap, vmin=0, vmax=vmax_, interpolation="nearest")
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        return im
    imt = dp(axes[0, 0], dem, "(a) Terrain elevation", "terrain", None)
    im = dp(axes[0, 1], cup, "(b) Coarse input, 10 m")
    dp(axes[0, 2], gt, "(c) Ground truth, 2 m")
    dp(axes[0, 3], fr, "(d) Frozen V0 prediction")
    dp(axes[1, 0], wn, "(e) Fine-tuned prediction")
    e_fr = np.where(v, np.abs(fr - gt), np.nan)
    e_wn = np.where(v, np.abs(wn - gt), np.nan)
    for ax, e, ttl in ((axes[1, 1], e_fr, "(f) |error|, frozen V0"),
                       (axes[1, 2], e_wn, "(g) |error|, fine-tuned")):
        lim = float(np.nanpercentile(e, 98))
        im2 = ax.imshow(e, cmap="magma_r", vmin=0, vmax=max(lim, 1e-3))
        ax.set_title(ttl, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        cb = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("|error| (m)", fontsize=7.5); cb.ax.tick_params(labelsize=7)
    d = np.where(v, e_fr - e_wn, np.nan)
    lim = float(np.nanpercentile(np.abs(d), 98))
    im3 = axes[1, 3].imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    axes[1, 3].set_title("(h) Error reduction by the fine-tune", fontsize=9.5)
    axes[1, 3].set_xticks([]); axes[1, 3].set_yticks([]); axes[1, 3].grid(False)
    cb = fig.colorbar(im3, ax=axes[1, 3], fraction=0.046, pad=0.02)
    cb.set_label("m, red = improvement", fontsize=7.5); cb.ax.tick_params(labelsize=7)
    cb = fig.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.02)
    cb.set_label("water depth (m)", fontsize=7.5); cb.ax.tick_params(labelsize=7)
    cb = fig.colorbar(imt, ax=axes[0, 0], fraction=0.046, pad=0.02)
    cb.set_label("elevation (m)", fontsize=7.5); cb.ax.tick_params(labelsize=7)
    mm = t["metrics"]
    share_better = float(np.nansum((d > 0) & v) / max(np.nansum(v), 1) * 100)
    fig.suptitle(
        f"Deepest urban-flood test tile  {t['scenario']}  iy={t['iy']} ix={t['ix']}   "
        f"peak truth {np.nanmax(np.where(v, gt, np.nan)):.2f} m   "
        f"{t['deep_px']:,} pixels above 1 m\n"
        f"MAE over wet pixels  frozen {mm['frozen']['MAE_wet']:.3f} m  "
        f"fine-tuned {mm['win']['MAE_wet']:.3f} m   |   "
        f"CSI@1.00 m  frozen {mm['frozen']['CSI_100']:.3f}  "
        f"fine-tuned {mm['win']['CSI_100']:.3f}   |   "
        f"{share_better:.1f}% of pixels improved", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fig21_tile_deep.png")
    plt.close(fig)
    print("wrote fig21_tile_deep.png")

    # ---------------- fig 22: three tiles --------------------------------
    fig, axes = plt.subplots(3, 5, figsize=(12.8, 7.6))
    for r, t in enumerate(tiles):
        gt, fr, wn = t["gt"], t["frozen"], t["win"]
        v, base, coarse = t["valid"], t["base"], t["coarse"]
        sc = gt.shape[-1] // coarse.shape[-1]
        cup = np.repeat(np.repeat(coarse, sc, 0), sc, 1)[:gt.shape[0], :gt.shape[1]]
        vmax = max(float(np.nanpercentile(gt[v], 99.5)), 0.05)
        for c, (ttl, arr) in enumerate([("Coarse 10 m", cup), ("Bilinear", base),
                                        ("Frozen V0", fr), ("Fine-tuned", wn),
                                        ("Ground truth", gt)]):
            ax = axes[r, c]
            a = np.where(v, arr, np.nan)
            ax.imshow(a, cmap=FLOOD, vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if r == 0:
                ax.set_title(ttl, fontsize=9.5)
            if c == 4:
                ax.set_xlabel(f"{t['scenario']}  iy={t['iy']} ix={t['ix']}\n"
                              f"peak {t['peak']:.2f} m, {t['deep_px']:,} px > 1 m",
                              fontsize=7.5)
        mm = t["metrics"]
        axes[r, 0].set_ylabel(
            f"MAE wet\n{mm['frozen']['MAE_wet']:.3f}→{mm['win']['MAE_wet']:.3f}", fontsize=8)
    fig.suptitle("Three test tiles spanning the inundation spectrum "
                 "(rows: deep urban, intermediate, shallow)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "fig22_tiles_spectrum.png")
    plt.close(fig)
    print("wrote fig22_tiles_spectrum.png")

    # summary numbers for the narrative
    print("\nnarrative numbers")
    for r, t in enumerate(tiles):
        mm = t["metrics"]
        print(f"  tile {r}: idx {t['index']} {t['scenario']} iy={t['iy']} ix={t['ix']}")
        for k in ("frozen", "win"):
            m = mm[k]
            print(f"    {k:<7} MAE_wet {m['MAE_wet']:.4f} RMSE_wet {m['RMSE_wet']:.4f} "
                  f"CSI@0.05 {m['CSI_005']:.4f} CSI@0.30 {m['CSI_030']:.4f} "
                  f"CSI@1.00 {m['CSI_100']:.4f} peak_err {m['PeakDepthError']:.4f} "
                  f"vol_err {m['VolumeRelativeError']:.4f}")
        gt, v = t["gt"], t["valid"]
        sel = v & (gt > 1.0)
        if sel.any():
            print(f"    deep pixels {sel.sum():,}  mean truth {gt[sel].mean():.3f} m  "
                  f"frozen bias {(t['frozen'][sel]-gt[sel]).mean():+.3f}  "
                  f"win bias {(t['win'][sel]-gt[sel]).mean():+.3f}")


if __name__ == "__main__":
    main()
