"""Figures and spatial statistics for the standalone residual cross-resolution chapter.

This script only *uses* artefacts that already exist.  No model is trained and no
simulation is re-run:

  * the frozen parent       ``outputs/v0_10m2m_hmax/last.pt``            (ep180)
  * the best fine-tune      ``outputs/finetune_deep/w01/snapshots/ep0187.pt``
  * the residual arm        ``outputs/v1_bilinear_residual/depth/best_csi.pt``

Two products are written.

Figures (``outputs/report_figs/``)
----------------------------------
``fig89_res_tiles_deep.png``
    One deep test tile, 3 x 3: coarse 10 m / bilinear base / 2 m truth on a shared
    depth scale, then frozen / fine-tuned / residual predictions on the same scale,
    then one absolute-error panel per learned model.
``fig90_res_tiles_spectrum.png``
    Three test tiles spanning the inundation spectrum x six columns
    (coarse, bilinear, frozen, fine-tuned, residual, truth).
``fig91_res_error_structure.png``
    Error maps on the deep tile plus the error structure against truth depth and
    distance to water, pooled over the full validation split.
``fig92_res_residual_field.png``
    What each arm adds on top of the interpolation: the ideal residual
    (truth - bilinear), the frozen net's implied residual, and the residual arm's
    metre-space prediction.

Statistics (``outputs/premodel/res_chapter_spatial.json``)
    Pooled error by truth-depth bin and by distance-to-water bin for the three
    learned arms, plus deep-pixel lift/slope statistics, all on the full
    validation split (182 tiles) so the numbers line up with
    ``exp_ab_comparison.json``.

Usage
-----
    python -X utf8 scripts/make_residual_chapter_figs.py            # everything
    python -X utf8 scripts/make_residual_chapter_figs.py --stage spatial
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.normalization import StaticNormalizer  # noqa: E402
from dataset.wellington_fixed_sr import (  # noqa: E402
    WellingtonFixedSRDataset,
    collate_fixed,
)
from engine.checkpoint import load_checkpoint  # noqa: E402
from engine.evaluator import build_model  # noqa: E402
from viz.common import FLOOD_CMAP  # noqa: E402
from viz.reconstruct import predict_tile  # noqa: E402

OUT = ROOT / "outputs" / "report_figs"
STATS = ROOT / "outputs" / "premodel" / "res_chapter_spatial.json"

CFG_V0 = ROOT / "configs/v0_10m2m_hmax.yaml"
CFG_V1 = ROOT / "configs/v1_bilinear_residual.yaml"

ARMS = {
    "frozen": ("hydrogeo_srno", CFG_V0, ROOT / "outputs/v0_10m2m_hmax/last.pt"),
    "win": ("hydrogeo_srno", CFG_V0,
            ROOT / "outputs/finetune_deep/w01/snapshots/ep0187.pt"),
    "res": ("bilinear_residual", CFG_V1,
            ROOT / "outputs/v1_bilinear_residual/depth/best_csi.pt"),
}
ARM_LABEL = {"frozen": "Frozen ep180", "win": "Best fine-tune w01 ep187",
             "res": "Residual (Exp A)"}
ARM_COLOR = {"frozen": "#98a2b3", "win": "#1f4e79", "res": "#c0392b"}

# the three test tiles the report already uses for its spatial figures
TILES = [{"index": 165, "role": "deep urban"},
         {"index": 48, "role": "intermediate"},
         {"index": 229, "role": "shallow"}]

DEPTH_BINS = [0.05, 0.25, 0.50, 1.00, 2.00, 3.00, np.inf]
DEPTH_LAB = ["0.05–0.25", "0.25–0.50", "0.50–1.00", "1.00–2.00", "2.00–3.00", ">3.00"]
DIST_BINS = [0.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, np.inf]
DIST_LAB = ["0–25", "25–50", "50–100", "100–200", "200–400", "400–800", ">800"]


def style() -> None:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({
        # English is the primary fix for in-figure text; the CJK faces are listed so
        # that any residual Chinese cannot fall through to a Latin-only face.
        "font.family": ["Times New Roman", "Microsoft YaHei", "SimSun", "SimHei",
                        "DejaVu Serif"],
        "font.serif": ["Times New Roman", "Microsoft YaHei", "SimSun", "SimHei",
                       "DejaVu Serif"],
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
        "font.size": 9.5, "axes.labelsize": 10, "axes.titlesize": 10.5,
        "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    })


def dataset(split: str) -> WellingtonFixedSRDataset:
    cfg = yaml.safe_load(CFG_V0.read_text(encoding="utf-8"))
    dc = cfg["dataset"]
    return WellingtonFixedSRDataset(
        root=ROOT / dc.get("root", "dataset"), split=split,
        lr_res=int(dc["lr_res"]), hr_res=int(dc["hr_res"]), target=dc["target"],
        geo_mode=dc["geo_mode"], scenarios=tuple(dc["scenarios"]))


def load_models() -> dict:
    models = {}
    for key, (name, cfg_p, ckpt) in ARMS.items():
        cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))
        m = build_model(name, cfg)
        load_checkpoint(str(ckpt), m, map_location="cpu")
        models[key] = m.eval().to("cpu")
        print(f"  loaded {key:>6}  {ckpt.relative_to(ROOT)}", flush=True)
    return models


def up(coarse: np.ndarray, like: np.ndarray) -> np.ndarray:
    sc = like.shape[-1] // coarse.shape[-1]
    a = np.repeat(np.repeat(coarse, sc, 0), sc, 1)
    return a[: like.shape[0], : like.shape[1]]


def metro(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


# --------------------------------------------------------------------------- tiles
def tile_payloads(models: dict) -> list[dict]:
    ds = dataset("test")
    out = []
    for t in TILES:
        i = t["index"]
        batch = collate_fixed([ds[i]])
        tp = {k: predict_tile(m, batch, device=torch.device("cpu"))
              for k, m in models.items()}
        f = tp["frozen"]
        rec = {
            "index": i, "role": t["role"],
            "scenario": str(batch["scenario"][0]),
            "iy": int(batch["iy"][0]), "ix": int(batch["ix"][0]),
            "gt": f.gt, "valid": f.valid, "dem": f.dem,
            "coarse": f.coarse, "base": f.base,
            "frozen": tp["frozen"].pred, "win": tp["win"].pred, "res": tp["res"].pred,
            "base_res": tp["res"].base,
            "metrics": {k: tp[k].metrics for k in tp},
        }
        print(f"  tile {i:>3} {rec['scenario']} iy={rec['iy']} ix={rec['ix']}  "
              + "  ".join(f"{k}:MAE_wet={rec['metrics'][k]['MAE_wet']:.4f}/"
                          f"CSI100={rec['metrics'][k]['CSI_100']:.4f}"
                          for k in ("frozen", "win", "res")), flush=True)
        out.append(rec)
    return out


def fig_tile_deep(tiles: list[dict]) -> None:
    t = tiles[0]
    gt, v = t["gt"], t["valid"]
    base = t["base"]
    cup = up(t["coarse"], gt)
    vmax = float(np.nanpercentile(gt[v], 99.5))

    def depth_panel(ax, arr, title):
        im = ax.imshow(np.where(v, arr, np.nan), cmap=FLOOD_CMAP, vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(title, fontsize=9.5)
        metro(ax)
        return im

    fig, axes = plt.subplots(3, 3, figsize=(11.4, 11.2))
    im = depth_panel(axes[0, 0], cup, "(a) Coarse input, 10 m")
    depth_panel(axes[0, 1], base, "(b) Bilinear base, 2 m")
    depth_panel(axes[0, 2], gt, "(c) Ground truth, 2 m")
    depth_panel(axes[1, 0], np.where(v, t["frozen"], np.nan).astype(np.float32),
                "(d) Frozen ep180")
    depth_panel(axes[1, 1], np.where(v, t["win"], np.nan).astype(np.float32),
                "(e) Best fine-tune w01 ep187")
    depth_panel(axes[1, 2], np.where(v, t["res"], np.nan).astype(np.float32),
                "(f) Residual (Exp A)")
    cb = fig.colorbar(im, ax=list(axes[0, :]) + list(axes[1, :]),
                      fraction=0.024, pad=0.012)
    cb.set_label("water depth (m), shared scale", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)

    errs = {k: np.where(v, np.abs(t[k] - gt), np.nan)
            for k in ("frozen", "win", "res")}
    lim = float(np.nanpercentile(np.concatenate([e[np.isfinite(e)] for e in errs.values()]),
                                 98))
    for ax, key, tag in zip(axes[2, :], ("frozen", "win", "res"), "ghi"):
        im2 = ax.imshow(errs[key], cmap="inferno_r", vmin=0, vmax=max(lim, 1e-3),
                        interpolation="nearest")
        m = t["metrics"][key]
        ax.set_title(f"({tag}) |error|, {ARM_LABEL[key].split(' (')[0]}\n"
                     f"MAE wet {m['MAE_wet']:.3f} m", fontsize=9.5)
        metro(ax)
        c = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.02)
        c.set_label("|error| (m)", fontsize=7.5)
        c.ax.tick_params(labelsize=7)

    mm = t["metrics"]
    de = mm["res"]["MAE_wet"] - mm["win"]["MAE_wet"]
    dg = mm["res"]["MAE_wet"] - mm["frozen"]["MAE_wet"]
    fig.suptitle(
        f"Deepest urban test tile  {t['scenario']}  iy={t['iy']} ix={t['ix']}   "
        f"peak truth {np.nanmax(np.where(v, gt, np.nan)):.2f} m   "
        f"wet MAE frozen {mm['frozen']['MAE_wet']:.3f} m, fine-tuned "
        f"{mm['win']['MAE_wet']:.3f} m, residual {mm['res']['MAE_wet']:.3f} m\n"
        f"residual vs fine-tune {de:+.4f} m, vs frozen {dg:+.4f} m   "
        f"|   deep-water CSI@1.0 m  frozen {mm['frozen']['CSI_100']:.3f}, "
        f"fine-tuned {mm['win']['CSI_100']:.3f}, residual {mm['res']['CSI_100']:.3f}",
        fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    p = OUT / "fig89_res_tiles_deep.png"
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p.name, flush=True)


def fig_tile_spectrum(tiles: list[dict]) -> None:
    cols = ["Coarse 10 m", "Bilinear base", "Frozen ep180",
            "Best fine-tune w01", "Residual (Exp A)", "Ground truth"]
    fig, axes = plt.subplots(len(tiles), len(cols),
                             figsize=(13.4, 7.9))
    for r, t in enumerate(tiles):
        gt, v, base = t["gt"], t["valid"], t["base"]
        arrs = [up(t["coarse"], gt), base, t["frozen"], t["win"], t["res"], gt]
        vmax = max(float(np.nanpercentile(gt[v], 99.5)), 0.05)
        for c, (ttl, arr) in enumerate(zip(cols, arrs)):
            ax = axes[r, c]
            ax.imshow(np.where(v, arr, np.nan), cmap=FLOOD_CMAP, vmin=0, vmax=vmax,
                      interpolation="nearest")
            metro(ax)
            if r == 0:
                ax.set_title(ttl, fontsize=9.5)
            if c == len(cols) - 1:
                ax.set_xlabel(f"{t['scenario']}  iy={t['iy']} ix={t['ix']}\n"
                              f"{t['role']}, peak {np.nanmax(np.where(v, gt, np.nan)):.2f} m",
                              fontsize=7.5)
        mm = t["metrics"]
        axes[r, 0].set_ylabel(
            "wet MAE (m)\nfrozen " + f"{mm['frozen']['MAE_wet']:.3f}\n"
            "fine-tuned " + f"{mm['win']['MAE_wet']:.3f}\n"
            "residual " + f"{mm['res']['MAE_wet']:.3f}", fontsize=8)
    fig.suptitle("Three test tiles spanning the inundation spectrum: what the residual "
                 "parameterisation changes on top of the interpolation", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    p = OUT / "fig90_res_tiles_spectrum.png"
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p.name, flush=True)


def fig_residual_field(tiles: list[dict]) -> dict:
    t = tiles[0]
    gt, v, base = t["gt"], t["valid"], t["base"]
    ideal = np.where(v, gt - base, np.nan)
    d_res = np.where(v, t["res"] - base, np.nan)
    d_frz = np.where(v, t["frozen"] - base, np.nan)
    deep = v & (gt > 1.0)
    lim = float(np.nanpercentile(np.abs(ideal[np.isfinite(ideal)]), 98))

    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 10:
            return float("nan")
        return float(np.corrcoef(a[m], b[m])[0, 1])

    stats = {
        "tile": t["index"],
        "corr_ideal_vs_res": corr(ideal, d_res),
        "corr_ideal_vs_frozen": corr(ideal, d_frz),
        "corr_base_vs_truth": corr(base, gt),
        "deep_mean_ideal_residual_m": float(np.nanmean(ideal[deep])) if deep.any() else float("nan"),
        "deep_mean_residual_m": float(np.nanmean(d_res[deep])) if deep.any() else float("nan"),
        "deep_mean_frozen_residual_m": float(np.nanmean(d_frz[deep])) if deep.any() else float("nan"),
        "deep_n": int(deep.sum()),
    }

    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.1))
    panels = [("(a) Bilinear base", base, FLOOD_CMAP, 0.0,
               max(float(np.nanpercentile(gt[v], 99.5)), 0.05), "m"),
              ("(b) Ideal residual = truth − bilinear", ideal, "RdBu_r", -lim, lim, "m"),
              ("(c) Frozen ep180 added above base", d_frz, "RdBu_r", -lim, lim, "m"),
              ("(d) Residual arm added above base", d_res, "RdBu_r", -lim, lim, "m")]
    for ax, (ttl, arr, cmap, lo, hi, unit) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(ttl, fontsize=9.5)
        metro(ax)
        c = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        c.set_label(unit, fontsize=7.5)
        c.ax.tick_params(labelsize=7)
    fig.suptitle(
        "What each arm adds on top of the interpolation, deep test tile "
        f"iy={t['iy']} ix={t['ix']}:  correlation with the ideal residual  "
        f"frozen {stats['corr_ideal_vs_frozen']:.3f}, residual "
        f"{stats['corr_ideal_vs_res']:.3f}", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = OUT / "fig92_res_residual_field.png"
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p.name, flush=True)
    return stats


# ------------------------------------------------------------------------ spatial
def spatial_pass(models: dict) -> dict:
    ds = dataset("val")
    norm = StaticNormalizer(hr_res=2, root=ROOT / "dataset")
    dw = 10  # Dist_water inside the continuous static stack
    dw_mean = float(norm.mean[dw]); dw_std = float(norm.std[dw])
    acc = {k: {"n_d": np.zeros(len(DEPTH_LAB)), "s_d": np.zeros(len(DEPTH_LAB)),
               "ss_d": np.zeros(len(DEPTH_LAB)),
               "n_r": np.zeros(len(DIST_LAB)), "s_r": np.zeros(len(DIST_LAB)),
               "ss_r": np.zeros(len(DIST_LAB)),
               "deep_n": 0.0, "deep_need": 0.0, "deep_lift": 0.0,
               "deep_ss": np.zeros(4), "d_sum": 0.0, "d_abs": 0.0, "d_n": 0.0}
           for k in models}
    n_tiles = len(ds)
    t0 = time.time()
    for bi in range(n_tiles):
        batch = collate_fixed([ds[bi]])
        dist = (batch["static_cont"][0, dw] * dw_std + dw_mean).numpy()
        truth = batch["hr"][0, 0].numpy()
        mask = batch["mask"][0, 0].numpy() > 0
        lr = batch["lr"][0, 0].numpy()
        for key, m in models.items():
            with torch.no_grad():
                out = m(lr=batch["lr"], lr_valid=batch["lr_valid"],
                        static_cont=batch["static_cont"], landuse=batch["landuse"],
                        lr_res=batch["lr_res"], hr_res=batch["hr_res"])
            pred = out["depth"][0, 0].numpy()
            base = out["base"][0, 0].numpy()
            A = acc[key]
            wet = mask & (truth > 0.05)
            e = (pred - truth)[wet]
            tval = truth[wet]
            dval = dist[wet]
            for j in range(len(DEPTH_LAB)):
                lo, hi = DEPTH_BINS[j], DEPTH_BINS[j + 1]
                sel = (tval >= lo) & (tval < hi)
                if sel.any():
                    A["n_d"][j] += sel.sum()
                    A["s_d"][j] += e[sel].sum()
                    A["ss_d"][j] += (e[sel] ** 2).sum()
            for j in range(len(DIST_LAB)):
                lo, hi = DIST_BINS[j], DIST_BINS[j + 1]
                sel = (dval >= lo) & (dval < hi)
                if sel.any():
                    A["n_r"][j] += sel.sum()
                    A["s_r"][j] += e[sel].sum()
                    A["ss_r"][j] += (e[sel] ** 2).sum()
            deep = mask & (truth > 1.0)
            if deep.any():
                y = pred[deep].astype("float64"); x = up(lr, truth)[deep].astype("float64")
                t = truth[deep].astype("float64")
                A["deep_n"] += x.size
                A["deep_need"] += (t - x).sum()
                A["deep_lift"] += (y - x).sum()
                A["deep_ss"] += np.array([x.sum(), y.sum(), (x * x).sum(), (x * y).sum()])
            A["d_sum"] += (pred - base)[wet].sum()
            A["d_abs"] += np.abs(pred - base)[wet].sum()
            A["d_n"] += wet.sum()
        if (bi + 1) % 20 == 0:
            print(f"  val {bi+1}/{n_tiles}  {(time.time()-t0)/60:.1f} min", flush=True)

    out = {"split": "val", "n_tiles": n_tiles, "models": {}}
    for key, A in acc.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            rmse_d = np.sqrt(A["ss_d"] / A["n_d"])
            bias_d = A["s_d"] / A["n_d"]
            rmse_r = np.sqrt(A["ss_r"] / A["n_r"])
            bias_r = A["s_r"] / A["n_r"]
        n = max(A["deep_n"], 1.0)
        sx, sy, sxx, sxy = A["deep_ss"]
        den = A["deep_n"] * sxx - sx * sx
        slope = ((A["deep_n"] * sxy - sx * sy) / den) if abs(den) > 1e-9 else float("nan")
        out["models"][key] = {
            "n_wet": int(A["n_d"].sum()),
            "depth_bins": DEPTH_LAB, "dist_bins": DIST_LAB,
            "rmse_by_depth": rmse_d.tolist(), "bias_by_depth": bias_d.tolist(),
            "n_by_depth": A["n_d"].tolist(),
            "rmse_by_dist": rmse_r.tolist(), "bias_by_dist": bias_r.tolist(),
            "n_by_dist": A["n_r"].tolist(),
            "deep_n": int(A["deep_n"]),
            "deep_slope": float(slope),
            "deep_needed_lift_m": float(A["deep_need"] / n),
            "deep_delivered_lift_m": float(A["deep_lift"] / n),
            # signed deep bias = mean(pred - truth) over deep pixels; since
            # lift = mean(pred - coarse) and need = mean(truth - coarse), the
            # difference is exactly the signed deep bias.
            "deep_bias_m": float((A["deep_lift"] - A["deep_need"]) / n),
            "mean_added_over_base_m": float(A["d_sum"] / max(A["d_n"], 1)),
            "mean_abs_added_over_base_m": float(A["d_abs"] / max(A["d_n"], 1)),
        }
    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {STATS.relative_to(ROOT)}  ({time.time()-t0:.0f} s)", flush=True)
    return out


def fig_error_structure(tiles: list[dict], sp: dict) -> None:
    t = tiles[0]
    gt, v = t["gt"], t["valid"]
    base = t["base"]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.2))

    # --- row 1: where the residual error lives on the deep tile
    err = {k: np.where(v, np.abs(t[k] - gt), np.nan) for k in ("frozen", "win", "res")}
    lim = float(np.nanpercentile(err["res"][np.isfinite(err["res"])], 98))
    for ax, key, tag in zip(axes[0, :], ("frozen", "win", "res"), "abc"):
        im = ax.imshow(err[key], cmap="inferno_r", vmin=0, vmax=max(lim, 1e-3),
                       interpolation="nearest")
        ax.set_title(f"({tag}) |error|, {ARM_LABEL[key].split(' (')[0]}", fontsize=9.5)
        metro(ax)
        c = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        c.set_label("|error| (m)", fontsize=7.5); c.ax.tick_params(labelsize=7)

    # --- row 2: pooled error structure over the full validation split
    ax = axes[1, 0]
    x = np.arange(len(DEPTH_LAB))
    for key in ("frozen", "win", "res"):
        d = sp["models"][key]
        ax.plot(x, d["rmse_by_depth"], "-o", ms=4.5, lw=1.3,
                color=ARM_COLOR[key], label=ARM_LABEL[key])
    ax.set_xticks(x); ax.set_xticklabels(DEPTH_LAB, fontsize=7.6, rotation=25)
    ax.set_xlabel("Truth depth bin (m)")
    ax.set_ylabel("Wet-pixel RMSE (m)")
    ax.set_title("(d) Error by depth bin", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=7.8)

    ax = axes[1, 1]
    for key in ("frozen", "win", "res"):
        d = sp["models"][key]
        ax.plot(x, d["bias_by_depth"], "-o", ms=4.5, lw=1.3,
                color=ARM_COLOR[key], label=ARM_LABEL[key])
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(DEPTH_LAB, fontsize=7.6, rotation=25)
    ax.set_xlabel("Truth depth bin (m)")
    ax.set_ylabel("Signed bias, prediction − truth (m)")
    ax.set_title("(e) Bias by depth bin", fontsize=9.5, loc="left")

    ax = axes[1, 2]
    yd = np.arange(len(DIST_LAB))
    for key in ("frozen", "win", "res"):
        d = sp["models"][key]
        ax.plot(yd, d["rmse_by_dist"], "-o", ms=4.5, lw=1.3,
                color=ARM_COLOR[key], label=ARM_LABEL[key])
    ax.set_xticks(yd); ax.set_xticklabels(DIST_LAB, fontsize=7.6, rotation=25)
    ax.set_xlabel("Distance to water at 2 m (m), binned")
    ax.set_ylabel("Wet-pixel RMSE (m)")
    ax.set_title("(f) Error by distance to water", fontsize=9.5, loc="left")

    fig.suptitle("Where the residual model's error lives: deep test tile iy=%d ix=%d, "
                 "and the pooled structure over the full validation split"
                 % (t["iy"], t["ix"]), fontsize=10.8)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    p = OUT / "fig91_res_error_structure.png"
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p.name, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "tiles", "spatial"])
    args = ap.parse_args()
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading checkpoints")
    models = load_models()

    print("inference on the three test tiles")
    payloads = tile_payloads(models)
    fig_tile_deep(payloads)
    fig_tile_spectrum(payloads)
    fstats = fig_residual_field(payloads)
    (OUT / "_res_chapter_tile_stats.json").write_text(
        json.dumps(fstats, indent=2), encoding="utf-8")
    tile_metrics = {str(t["index"]): {"role": t["role"], "scenario": t["scenario"],
                                      "iy": t["iy"], "ix": t["ix"],
                                      "metrics": t["metrics"]} for t in payloads}
    (OUT / "_res_chapter_tile_metrics.json").write_text(
        json.dumps(tile_metrics, indent=2), encoding="utf-8")

    if args.stage in ("all", "spatial"):
        print("pooled spatial statistics over the full validation split")
        sp = spatial_pass(models)
        fig_error_structure(payloads, sp)
    print("done")


if __name__ == "__main__":
    main()
