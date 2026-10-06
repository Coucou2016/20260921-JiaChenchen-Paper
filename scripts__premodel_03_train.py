r"""Stage 3: train the pre-model that predicts the coarse-grid hydrodynamic error.

Target  = native coarse depth minus area-weighted 2 m truth, on the coarse lattice.
Inputs  = the coarse depth field plus the coarse static stack only.  No fine grid
          value is ever an input.

Split discipline is the one already fixed in dataset/index/patches_480m.json:
fit on the northern training bands, pick the epoch on the validation bands,
report once on the southern test bands.

Two non-learned baselines are reported:
  identity  - leave the coarse run untouched
  bilinear  - bilinearly resample the coarse field onto the 2 m lattice and
              average it back; a resampling filter with no parameters.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import premodel_lib as L
from scipy import stats

RES_SET = {"5m": 96, "10m": 48, "20m": 24, "30m": 16}
N_CH = len(L.STATIC_CHANNELS) + 1


# ------------------------------------------------------------------ operators
def bilinear_roundtrip(hc, n_c, cell_factor=5):
    """Bilinear resample the coarse field to the 2 m lattice, then average back."""
    # bilinear upsample hc[ny,nx] -> [ny*cell_factor, nx*cell_factor]
    f = cell_factor
    ny = nx = n_c
    t = torch.from_numpy(np.asarray(hc, dtype=np.float32))[None, None]
    big = torch.nn.functional.interpolate(t, size=(ny * f, nx * f), mode="bilinear",
                                          align_corners=False)
    back = torch.nn.functional.avg_pool2d(big, f, stride=f)
    return back[0, 0].numpy()


class PreModel(nn.Module):
    """Small residual CNN on the coarse lattice.  Output is a depth correction."""

    def __init__(self, cin=N_CH + 1, width=32, depth=3):
        super().__init__()
        layers = [nn.Conv2d(cin, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.body(x))


def build_input(X):
    """X: (N,C,H,W) raw coarse stack -> model input with log-depth appended."""
    h = X[:, :1]
    extra = torch.log1p(torch.clamp(h, min=0.0))
    return torch.cat([X, extra], dim=1)


# --------------------------------------------------------------------- helpers
def load_split(res, split):
    d = np.load(L.OUT / f"tile_{res}_{split}.npz")
    return d["X"], d["Y"], d["T"], d["M"], d["meta"]


def metrics(pred, truth, mask, thr=(0.05, 0.30, 1.00)):
    ok = mask.astype(bool)
    p = pred[ok].astype("float64")
    t = truth[ok].astype("float64")
    d = p - t
    sv = float(np.sum(t))
    out = {"n": int(ok.sum()),
           "mae": float(np.mean(np.abs(d))),
           "rmse": float(np.sqrt(np.mean(d * d))),
           "bias": float(np.mean(d)),
           "volume_rel": float((np.sum(p) - sv) / sv) if sv else float("nan")}
    pm = np.where(ok, pred, np.nan)
    tm = np.where(ok, truth, np.nan)
    for q in thr:
        out[f"csi@{q:.2f}"] = L.csi(pm, tm, q)
    return out


def paired_stats(a, b):
    """a, b are per-tile metric vectors; smaller is better.  a=model, b=baseline."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    d = a - b
    if d.size < 5:
        return {}
    t = stats.ttest_rel(a, b)
    w = stats.wilcoxon(a, b)
    rng = np.random.default_rng(0)
    boot = [float(np.mean(rng.choice(d, d.size, replace=True))) for _ in range(4000)]
    sd = float(np.std(d, ddof=1))
    return {"n": int(d.size),
            "mean_delta": float(np.mean(d)),
            "median_delta": float(np.median(d)),
            "stderr": float(sd / np.sqrt(d.size)),
            "t": float(t.statistic), "t_p": float(t.pvalue),
            "wilcoxon_p": float(w.pvalue),
            "cohen_dz": float(np.mean(d) / sd) if sd > 0 else float("nan"),
            "boot_ci_lo": float(np.percentile(boot, 2.5)),
            "boot_ci_hi": float(np.percentile(boot, 97.5)),
            "frac_improved": float(np.mean(d < 0))}


# ---------------------------------------------------------------------- train
def run_resolution(res, epochs=150, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, Ttr, Mtr, metr = load_split(res, "train")
    Xva, Yva, Tva, Mva, mva = load_split(res, "val")
    Xte, Yte, Tte, Mte, mte = load_split(res, "test")
    print(f"[{res}] train {Xtr.shape} val {Xva.shape} test {Xte.shape}", flush=True)

    mu = Xtr.reshape(Xtr.shape[0], Xtr.shape[1], -1).mean(axis=(0, 2))
    sd = Xtr.reshape(Xtr.shape[0], Xtr.shape[1], -1).std(axis=(0, 2))
    sd[sd < 1e-6] = 1.0
    mu_t = torch.tensor(mu)[None, :, None, None]
    sd_t = torch.tensor(sd)[None, :, None, None]

    def prep(X, M):
        x = torch.tensor(X)
        x = (x - mu_t) / sd_t
        x = torch.nan_to_num(x)
        return x, torch.tensor(M.astype(np.float32))

    xtr, mtr = prep(Xtr, Mtr)
    # the target lives on the same normalised scale as the depth channel, so the
    # residual the network predicts can be subtracted from the input directly
    ytr = torch.tensor(Ytr / sd[0])
    xva, mva_ = prep(Xva, Mva)
    xte, mte_ = prep(Xte, Mte)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    # the whole tile stack fits in 4 GB, so it is staged on the device once and
    # the per-step host-to-device copies disappear
    xtr, ytr, mtr = xtr.to(dev), ytr.to(dev), mtr.to(dev)
    xva, mva_ = xva.to(dev), mva_.to(dev)
    xte, mte_ = xte.to(dev), mte_.to(dev)
    model = PreModel().to(dev)

    def corrected(x):
        """Corrected coarse depth in metres for a normalised input stack.

        The network predicts the error divided by the depth scale, so the raw
        metres come back after multiplying by that scale.  The denormalisation of
        the depth channel itself has to be undone as well, otherwise the result
        would sit on the standardised scale rather than in metres.
        """
        resid = x[:, :1] - model(build_input(x))
        return resid * float(sd[0]) + float(mu[0])

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bs = 16
    n = xtr.shape[0]
    hist = []
    best = (1e9, -1, None)
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        tot = 0.0
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            xb = build_input(xtr[j])
            yb = ytr[j]
            mb = mtr[j]
            pred = model(xb)
            loss = (torch.abs(pred - yb) * mb).sum() / mb.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
        sched.step()
        # ---- validation on the corrected coarse field
        model.eval()
        with torch.no_grad():
            corr = []
            for i in range(0, xva.shape[0], bs):
                corr.append(corrected(xva[i:i + bs]).cpu().numpy()[:, 0])
            corr = np.concatenate(corr)
            m = metrics(corr, Tva[:, 0], Mva[:, 0])
        hist.append({"epoch": ep, "train_l1": tot / n, "val_mae": m["mae"],
                     "val_rmse": m["rmse"], "val_csi005": m["csi@0.05"]})
        if m["mae"] < best[0]:
            best = (m["mae"], ep, {k: v.clone() for k, v in model.state_dict().items()})
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  ep {ep:3d} trainL1 {tot/n:.4f} valMAE {m['mae']:.4f} "
                  f"valRMSE {m['rmse']:.4f} csi {m['csi@0.05']:.4f}", flush=True)
    model.load_state_dict(best[2])
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd,
                "best_epoch": best[1], "val_mae": best[0]},
               L.OUT / f"ckpt_{res}.pt")

    # -------------------------------------------------------------- test once
    model.eval()
    with torch.no_grad():
        corr = []
        for i in range(0, xte.shape[0], bs):
            corr.append(corrected(xte[i:i + bs]).cpu().numpy()[:, 0])
        corr = np.concatenate(corr)
    identity = Xte[:, 0]
    bilin = np.stack([bilinear_roundtrip(identity[k], identity.shape[-1])
                      for k in range(identity.shape[0])])
    Tt, Mt = Tte[:, 0], Mte[:, 0]

    arms = {"identity": identity, "bilinear": bilin, "premodel": corr}
    glob = {k: metrics(v, Tt, Mt) for k, v in arms.items()}

    # per-tile paired vectors (MAE, RMSE and CSI@0.05 per tile)
    per = {}
    for k, v in arms.items():
        per[k] = {"mae": [], "rmse": [], "csi005": []}
        for i in range(v.shape[0]):
            ok = Mt[i].astype(bool)
            if ok.sum() < 4:
                per[k]["mae"].append(np.nan)
                per[k]["rmse"].append(np.nan)
                per[k]["csi005"].append(np.nan)
                continue
            d = (v[i][ok] - Tt[i][ok]).astype("float64")
            per[k]["mae"].append(float(np.mean(np.abs(d))))
            per[k]["rmse"].append(float(np.sqrt(np.mean(d * d))))
            per[k]["csi005"].append(L.csi(np.where(ok, v[i], np.nan),
                                          np.where(ok, Tt[i], np.nan), 0.05))
    pairs = {
        "premodel_vs_identity": {m: paired_stats(per["premodel"][m], per["identity"][m])
                                 for m in ("mae", "rmse", "csi005")},
        "premodel_vs_bilinear": {m: paired_stats(per["premodel"][m], per["bilinear"][m])
                                 for m in ("mae", "rmse", "csi005")},
        "bilinear_vs_identity": {m: paired_stats(per["bilinear"][m], per["identity"][m])
                                 for m in ("mae", "rmse", "csi005")},
    }
    # error-field skill: how much of the true coarse error the model explains
    yt = Yte[:, 0]
    ok = Mt.astype(bool)
    applied = identity - corr            # the correction the model actually applies
    skill = {
        "rmse_of_true_error": float(np.sqrt(np.mean(yt[ok] ** 2))),
        "rmse_of_applied_correction": float(np.sqrt(np.mean(applied[ok] ** 2))),
        "residual_after_correction": float(np.sqrt(np.mean(
            (identity[ok] - corr[ok] - yt[ok]) ** 2))),
        "corr_correction_true_error": L.pearson(np.where(ok, applied, np.nan),
                                                np.where(ok, yt, np.nan)),
        "var_reduction": float(1 - np.var((identity[ok] - corr[ok] - yt[ok]))
                               / np.var(yt[ok])),
    }
    return {"resolution": res, "best_epoch": best[1], "n_train": int(Xtr.shape[0]),
            "n_val": int(Xva.shape[0]), "n_test": int(Xte.shape[0]),
            "global": glob, "paired": pairs, "skill": skill,
            "history": hist,
            "baseline_note": ("identity = no correction; bilinear = coarse field "
                              "bilinearly resampled to the 2 m lattice and averaged "
                              "back to the coarse lattice")}


def fine_grid_check(res, result):
    """Does the corrected coarse field help once nearest-upsampled to 2 m?"""
    import netCDF4 as nc
    from collections import defaultdict

    n_c = RES_SET[res]
    d = np.load(L.OUT / f"tile_{res}_test.npz", allow_pickle=True)
    meta = d["meta"]
    X, T, M = d["X"], d["T"][:, 0], d["M"][:, 0]
    ck = torch.load(L.OUT / f"ckpt_{res}.pt", map_location="cpu", weights_only=False)
    model = PreModel()
    model.load_state_dict(ck["state"])
    model.eval()
    mu, sd = ck["mu"], ck["sd"]
    x = torch.tensor(X)
    x = torch.nan_to_num((x - torch.tensor(mu)[None, :, None, None])
                         / torch.tensor(sd)[None, :, None, None])
    with torch.no_grad():
        corr = ((x[:, :1] - model(build_input(x))) * float(sd[0])
                + float(mu[0])).numpy()[:, 0]

    f2 = {s: nc.Dataset(L.GRIDS / "2m" / f"flood_{s}.nc") for s in L.SCENARIOS}
    acc = defaultdict(lambda: [0, 0.0])          # name -> [n_cells, sum sq err]
    mae = defaultdict(lambda: [0, 0.0])
    hits = defaultdict(lambda: np.zeros((3, 3)))
    truth_hits = np.zeros(3)
    for i in range(X.shape[0]):
        iy, ix, si = int(meta[i][0]), int(meta[i][1]), int(meta[i][2])
        scen = L.SCENARIOS[si]
        h2 = L.clean(f2[scen]["h_max"][iy * 240:(iy + 1) * 240,
                                      ix * 240:(ix + 1) * 240])
        finite = np.isfinite(h2)
        if finite.sum() < 100:
            continue
        arms = {"identity": L.resample_nearest(X[i, 0], n_c, 2),
                "premodel": L.resample_nearest(corr[i], n_c, 2),
                "bilinear": L.resample_nearest(
                    bilinear_roundtrip(X[i, 0], n_c), n_c, 2)}
        for k, v in arms.items():
            e = (v - h2)[finite]
            acc[k][0] += int(finite.sum())
            acc[k][1] += float(np.sum(e * e))
            mae[k][0] += int(finite.sum())
            mae[k][1] += float(np.sum(np.abs(e)))
            for ti, thr in enumerate((0.05, 0.30, 1.00)):
                p = v > thr
                t = h2 > thr
                hits[k][ti][0] += np.count_nonzero(p & t)
                hits[k][ti][1] += np.count_nonzero(p & ~t)
                hits[k][ti][2] += np.count_nonzero(~p & t)
        for ti, thr in enumerate((0.05, 0.30, 1.00)):
            truth_hits[ti] += np.count_nonzero((h2 > thr) & finite)
    for s in f2.values():
        s.close()
    out = {}
    for k in acc:
        tp, fp, fn = hits[k]
        out[k] = {"rmse_2m": float(np.sqrt(acc[k][1] / acc[k][0])),
                  "mae_2m": float(mae[k][1] / mae[k][0]),
                  "csi@0.05": float(tp[0] / max(tp[0] + fp[0] + fn[0], 1e-9)),
                  "csi@0.30": float(tp[1] / max(tp[1] + fp[1] + fn[1], 1e-9)),
                  "csi@1.00": float(tp[2] / max(tp[2] + fp[2] + fn[2], 1e-9)),
                  "n_cells": int(acc[k][0])}
    return out


if __name__ == "__main__":
    targets = sys.argv[1:] or L.COARSE
    out_path = L.OUT / "premodel_results.json"
    all_res = {}
    if out_path.exists():
        try:
            all_res = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            all_res = {}
    for res in targets:
        t0 = time.time()
        r = run_resolution(res)
        r["fine_grid_check"] = fine_grid_check(res, r)
        all_res[res] = r
        L.save_json("premodel_results.json", all_res)
        print(f"[{res}] done in {time.time()-t0:.1f}s  "
              f"test MAE id={r['global']['identity']['mae']:.4f} "
              f"pre={r['global']['premodel']['mae']:.4f} "
              f"bil={r['global']['bilinear']['mae']:.4f}", flush=True)
