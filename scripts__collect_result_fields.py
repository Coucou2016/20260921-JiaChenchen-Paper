"""Collect the direct-result evidence for the report.

Runs both checkpoints (frozen V0 ep180 and the selected fine-tune w=0.10 ep187) over the whole
test split on CPU, then dumps everything the result figures need:

  pixels.npz   flat subsample of (truth, frozen, fine-tuned) + depth bins for distribution stats
  moments.json streaming moments of the error per model and per depth bin
  tiles.npz    full 2-D fields for a handful of representative tiles
  per_tile.json per-tile metrics for both models
  autocorr.json lag-1 Moran statistics of the error field per tile
  exceed.json  inundation-exceedance curves

CPU only, so it can run while the GPU is busy with the epoch scan.
"""
from __future__ import annotations

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

from dataset.wellington_fixed_sr import WellingtonFixedSRDataset, collate_fixed  # noqa: E402
from engine.checkpoint import load_checkpoint  # noqa: E402
from engine.evaluator import build_model  # noqa: E402
from viz.reconstruct import predict_tile  # noqa: E402

OUT = ROOT / "outputs" / "report_figs"
OUT.mkdir(parents=True, exist_ok=True)
OUT_FIELDS = OUT / "result_fields"
OUT_FIELDS.mkdir(exist_ok=True)

CKPTS = {
    "frozen": ROOT / "outputs/v0_10m2m_hmax/last.pt",
    "win": ROOT / "outputs/finetune_deep/w01/snapshots/ep0187.pt",
}
EDGES = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf])
BIN_LABELS = ["0-0.05", "0.05-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", "0.5-0.75",
              "0.75-1", "1-1.5", "1.5-2", "2-3", "3-inf"]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (OUT_FIELDS / "collect.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/v0_10m2m_hmax.yaml").read_text(encoding="utf-8"))
    dc = cfg["dataset"]
    ds = WellingtonFixedSRDataset(
        root=ROOT / dc.get("root", "dataset"), split="test",
        lr_res=int(dc["lr_res"]), hr_res=int(dc["hr_res"]), target=dc["target"],
        geo_mode=dc["geo_mode"], scenarios=tuple(dc["scenarios"]),
    )
    n_tiles = len(ds)
    log(f"test split: {n_tiles} tiles")

    models = {}
    for k, ck in CKPTS.items():
        m = build_model(cfg["model"]["name"], cfg)
        load_checkpoint(str(ck), m, map_location="cpu")
        m.eval()
        models[k] = m
        log(f"loaded {k} <- {ck.name}")

    rng = np.random.default_rng(7)
    SUBSAMPLE = 4_000_000
    buf = {k: [] for k in ("truth", "frozen", "win")}

    # streaming moments, per model, globally and per true-depth bin
    def blank() -> dict:
        return {"n": np.zeros(len(BIN_LABELS)), "s1": np.zeros(len(BIN_LABELS)),
                "s2": np.zeros(len(BIN_LABELS)), "s3": np.zeros(len(BIN_LABELS)),
                "s4": np.zeros(len(BIN_LABELS)),
                "abs1": np.zeros(len(BIN_LABELS)),
                "und": np.zeros(len(BIN_LABELS)),
                "g_n": 0.0, "g_s1": 0.0, "g_s2": 0.0, "g_s3": 0.0, "g_s4": 0.0,
                "g_abs": 0.0}

    mom = {k: blank() for k in ("frozen", "win")}
    # error histogram on a signed-log grid (fine near zero, coarse in the tails)
    HEDGES = np.concatenate([
        -np.logspace(np.log10(2.0), np.log10(0.004), 40),   # -2.0 .. -0.004, ascending
        [0.0],
        np.logspace(np.log10(0.004), np.log10(2.0), 40),    # 0.004 .. 2.0
    ])
    hist = {k: np.zeros(len(HEDGES) - 1) for k in ("frozen", "win")}
    # exceedance thresholds
    THR = np.concatenate([np.linspace(0.01, 0.5, 40), np.linspace(0.55, 5.0, 60)])
    exceed = {k: np.zeros(len(THR)) for k in ("truth", "frozen", "win")}
    exceed_n = 0.0

    per_tile: list[dict] = []
    autocorr: list[dict] = []
    keep_full: list[dict] = []
    keep_idx = {0, 40, 90, 150, 200, 241}

    t0 = time.time()
    for i in range(n_tiles):
        sample = ds[i]
        batch = collate_fixed([sample])
        preds = {}
        for k in ("frozen", "win"):
            preds[k] = predict_tile(models[k], batch, device=torch.device("cpu"))

        gt = preds["frozen"].gt
        valid = preds["frozen"].valid
        row = {"index": i, "scenario": str(preds["frozen"].tile.get("scenario")),
               "iy": int(preds["frozen"].tile.get("iy", -1)),
               "ix": int(preds["frozen"].tile.get("ix", -1)),
               "peak_truth": float(gt[valid].max()) if valid.any() else 0.0,
               "wet_frac": float(((gt > 0.05) & valid).sum() / max(valid.sum(), 1))}
        for k in ("frozen", "win"):
            m = preds[k].metrics
            row[f"{k}_CSI_005"] = m["CSI_005"]
            row[f"{k}_CSI_030"] = m["CSI_030"]
            row[f"{k}_CSI_100"] = m["CSI_100"]
            row[f"{k}_MAE_wet"] = m["MAE_wet"]
            row[f"{k}_RMSE_wet"] = m["RMSE_wet"]
            row[f"{k}_PeakDepthError"] = m["PeakDepthError"]
            row[f"{k}_VolumeRelativeError"] = m["VolumeRelativeError"]
        per_tile.append(row)

        if not valid.any():
            continue

        g = gt[valid].astype(np.float64)
        b = np.digitize(g, EDGES) - 1
        b = np.clip(b, 0, len(BIN_LABELS) - 1)

        for k in ("frozen", "win"):
            p = preds[k].pred[valid].astype(np.float64)
            e = p - g
            d = mom[k]
            for j in range(len(BIN_LABELS)):
                sel = b == j
                if not sel.any():
                    continue
                ej = e[sel]
                d["n"][j] += ej.size
                d["s1"][j] += ej.sum()
                d["s2"][j] += (ej ** 2).sum()
                d["s3"][j] += (ej ** 3).sum()
                d["s4"][j] += (ej ** 4).sum()
                d["abs1"][j] += np.abs(ej).sum()
                d["und"][j] += (ej < 0).sum()
            d["g_n"] += e.size
            d["g_s1"] += e.sum()
            d["g_s2"] += (e ** 2).sum()
            d["g_s3"] += (e ** 3).sum()
            d["g_s4"] += (e ** 4).sum()
            d["g_abs"] += np.abs(e).sum()
            h, _ = np.histogram(e, bins=HEDGES)
            hist[k] += h
            for t_i, t in enumerate(THR):
                exceed[k][t_i] += (p > t).sum()
        for t_i, t in enumerate(THR):
            exceed["truth"][t_i] += (g > t).sum()
        exceed_n += g.size

        # lag-1 Moran's I of the error field (4-neighbour, valid cells only)
        for k in ("frozen", "win"):
            err = np.where(valid, preds[k].pred - gt, np.nan)
            center = np.nanmean(err)
            z = err - center
            num = 0.0
            den = np.nansum(z ** 2)
            pairs = 0
            for axis in (0, 1):
                a = np.take(z, range(0, z.shape[axis] - 1), axis=axis)
                c = np.take(z, range(1, z.shape[axis]), axis=axis)
                m = np.isfinite(a) & np.isfinite(c)
                num += float(np.nansum(a[m] * c[m]))
                pairs += int(m.sum())
            W = max(pairs, 1)
            autocorr.append({"index": i, "model": k,
                             "moran": float((len(g) / W) * num / den) if den else float("nan")})

        # full 2-D fields for selected tiles
        if i in keep_idx:
            keep_full.append({
                "index": i, "scenario": str(preds["frozen"].tile.get("scenario")),
                "iy": int(preds["frozen"].tile.get("iy", -1)),
                "ix": int(preds["frozen"].tile.get("ix", -1)),
                "gt": gt.astype(np.float32),
                "coarse": preds["frozen"].coarse.astype(np.float32),
                "base": preds["frozen"].base.astype(np.float32),
                "frozen": preds["frozen"].pred.astype(np.float32),
                "win": preds["win"].pred.astype(np.float32),
                "valid": valid,
                "dem": preds["frozen"].dem.astype(np.float32),
            })

        # reservoir subsample for scatter / QQ
        for k, src in (("truth", g), ("frozen", preds["frozen"].pred[valid].astype(np.float32)),
                       ("win", preds["win"].pred[valid].astype(np.float32))):
            buf[k].append(src.astype(np.float32))

        if (i + 1) % 10 == 0 or i == n_tiles - 1:
            log(f"{i+1}/{n_tiles} tiles  ({(time.time()-t0)/(i+1):.2f}s/tile)")

    # ---- write everything -------------------------------------------------
    all_truth = np.concatenate(buf["truth"])
    all_frozen = np.concatenate(buf["frozen"])
    all_win = np.concatenate(buf["win"])
    n = all_truth.size
    take = np.sort(rng.choice(n, size=min(SUBSAMPLE, n), replace=False))

    np.savez_compressed(
        OUT_FIELDS / "pixels.npz",
        truth=all_truth[take], frozen=all_frozen[take], win=all_win[take],
        n_total=np.array([n]),
    )
    log(f"saved pixels.npz  ({n:,} valid pixels, subsample {take.size:,})")

    np.savez_compressed(OUT_FIELDS / "tiles.npz", **{
        f"t{t_['index']}_{f}": (t_[f] if f == "valid" else t_[f])
        for t_ in keep_full for f in ("gt", "coarse", "base", "frozen", "win", "valid", "dem")
    })
    (OUT_FIELDS / "tiles_index.json").write_text(
        json.dumps([{k: v for k, v in t_.items()
                     if k in ("index", "scenario", "iy", "ix")} for t_ in keep_full], indent=2),
        encoding="utf-8")
    log(f"saved tiles.npz  ({len(keep_full)} tiles)")

    moments_out: dict = {"bins": BIN_LABELS, "models": {}}
    for k in ("frozen", "win"):
        d = mom[k]
        nn = np.maximum(d["n"], 1)
        mean = d["s1"] / nn
        var = np.maximum(d["s2"] / nn - mean ** 2, 1e-12)
        sd = np.sqrt(var)
        skew = (d["s3"] / nn - 3 * mean * d["s2"] / nn + 2 * mean ** 3) / sd ** 3
        kurt = (d["s4"] / nn - 4 * mean * d["s3"] / nn + 6 * mean ** 2 * d["s2"] / nn
                - 3 * mean ** 4) / var ** 2
        gn = max(d["g_n"], 1)
        gm = d["g_s1"] / gn
        gv = max(d["g_s2"] / gn - gm ** 2, 1e-12)
        gsd = np.sqrt(gv)
        moments_out["models"][k] = {
            "n": d["n"].tolist(), "mean": mean.tolist(), "sd": sd.tolist(),
            "mae": (d["abs1"] / nn).tolist(), "rmse": np.sqrt(d["s2"] / nn).tolist(),
            "under_frac": (d["und"] / nn).tolist(),
            "skew": skew.tolist(), "excess_kurtosis": kurt.tolist(),
            "global": {
                "n": gn, "mean": gm, "sd": gsd, "mae": d["g_abs"] / gn,
                "rmse": float(np.sqrt(d["g_s2"] / gn)),
                "skew": float((d["g_s3"] / gn - 3 * gm * d["g_s2"] / gn + 2 * gm ** 3) / gsd ** 3),
                "excess_kurtosis": float((d["g_s4"] / gn - 4 * gm * d["g_s3"] / gn
                                          + 6 * gm ** 2 * d["g_s2"] / gn - 3 * gm ** 4) / gv ** 2),
            },
        }
    (OUT_FIELDS / "moments.json").write_text(json.dumps(moments_out, indent=2), encoding="utf-8")
    log("saved moments.json")

    np.savez_compressed(OUT_FIELDS / "hist.npz",
                        edges=HEDGES, frozen=hist["frozen"], win=hist["win"])
    np.savez_compressed(OUT_FIELDS / "exceed.npz", thr=THR,
                        truth=exceed["truth"], frozen=exceed["frozen"],
                        win=exceed["win"], n=np.array([exceed_n]))
    log("saved hist.npz + exceed.npz")

    (OUT_FIELDS / "per_tile.json").write_text(json.dumps(per_tile, indent=2), encoding="utf-8")
    (OUT_FIELDS / "autocorr.json").write_text(json.dumps(autocorr, indent=2), encoding="utf-8")
    log("saved per_tile.json + autocorr.json")
    log("DONE")


if __name__ == "__main__":
    main()
