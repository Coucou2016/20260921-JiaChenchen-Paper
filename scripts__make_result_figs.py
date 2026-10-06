"""Direct-result figures: spatial fields, density comparisons, higher-order statistics.

Consumes the arrays written by scripts/collect_result_fields.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report_figs"
F = OUT / "result_fields"

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Microsoft YaHei",
                                            "SimSun", "SimHei", "DejaVu Serif"],
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix", "font.size": 9.5, "axes.labelsize": 10.5,
    "axes.titlesize": 10.5, "legend.fontsize": 8.5, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "figure.dpi": 200, "savefig.dpi": 200,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
})

FLOOD = LinearSegmentedColormap.from_list("flood", [
    (0.00, "#ffffff"), (0.03, "#eaf4fb"), (0.10, "#c2e2f5"), (0.25, "#7fc8ec"),
    (0.45, "#4a9fdc"), (0.65, "#2b6fbe"), (0.85, "#18408f"), (1.00, "#0b1f52")])
FLOOD.set_bad("#e6e6e6")
TERRAIN = "terrain"

C = {"frozen": "#8c8c8c", "win": "#c0392b", "truth": "#1f4e79", "bilin": "#e08b2b"}
LAB = {"frozen": "Frozen V0", "win": "Fine-tuned (w = 0.10)", "truth": "Ground truth"}


def save(fig, name):
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print(f"wrote {p.name}")


def load_json(n):
    return json.loads((F / n).read_text(encoding="utf-8"))


def depth_panel(ax, arr, valid, title, vmax, cmap=FLOOD):
    a = np.where(valid, arr, np.nan)
    im = ax.imshow(a, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    return im


# ---------------------------------------------------------------- 21 / 22
def fig_tiles():
    z = np.load(F / "tiles.npz")
    idx = load_json("tiles_index.json")
    order = sorted(range(len(idx)), key=lambda k: -idx[k]["index"])
    # choose the deepest-water tile and two others
    picks = []
    for k in range(len(idx)):
        picks.append(k)
    deep = max(picks, key=lambda k: float(np.nanmax(z[f"t{idx[k]['index']}_gt"])))
    shallow = min(picks, key=lambda k: float(np.nanmax(z[f"t{idx[k]['index']}_gt"])))
    mid = sorted(picks, key=lambda k: float(np.nanmax(z[f"t{idx[k]['index']}_gt"])))[len(picks)//2]
    sel = [deep, mid, shallow]

    # --- Fig 21: the deepest tile, 8 panels -----------------------------
    k = deep
    i = idx[k]["index"]
    gt = z[f"t{i}_gt"]; fr = z[f"t{i}_frozen"]; wn = z[f"t{i}_win"]
    v = z[f"t{i}_valid"].astype(bool); base = z[f"t{i}_base"]
    coarse = z[f"t{i}_coarse"]; dem = z[f"t{i}_dem"]
    sc = gt.shape[-1] // coarse.shape[-1]
    cup = np.repeat(np.repeat(coarse, sc, 0), sc, 1)[:gt.shape[0], :gt.shape[1]]
    vmax = float(np.nanpercentile(np.where(v, gt, np.nan), 99.5))

    fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.4))
    depth_panel(axes[0, 0], dem, v, "(a) Terrain elevation", None, TERRAIN)
    im = depth_panel(axes[0, 1], cup, v, "(b) Coarse input, 10 m", vmax)
    depth_panel(axes[0, 2], base, v, "(c) Bilinear interpolation", vmax)
    depth_panel(axes[0, 3], gt, v, "(d) Ground truth, 2 m", vmax)
    depth_panel(axes[1, 0], fr, v, "(e) Frozen V0", vmax)
    depth_panel(axes[1, 1], wn, v, "(f) Fine-tuned, w = 0.10", vmax)
    for ax, arr, ttl in ((axes[1, 2], fr, "(g) |error|, frozen V0"),
                         (axes[1, 3], wn, "(h) |error|, fine-tuned")):
        e = np.where(v, np.abs(arr - gt), np.nan)
        lim = float(np.nanpercentile(e, 98))
        im2 = ax.imshow(e, cmap="magma_r", vmin=0, vmax=max(lim, 1e-3))
        ax.set_title(ttl, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        cb = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7)
    c1 = fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.02)
    c1.set_label("water depth (m)", fontsize=8); c1.ax.tick_params(labelsize=7)
    ct = fig.colorbar(axes[0, 0].images[0], ax=axes[0, 0], fraction=0.046, pad=0.02)
    ct.set_label("elevation (m)", fontsize=8); ct.ax.tick_params(labelsize=7)
    fig.suptitle(f"Deepest-water test tile  {idx[k]['scenario']}  iy={idx[k]['iy']} "
                 f"ix={idx[k]['ix']}   peak truth "
                 f"{float(np.nanmax(np.where(v, gt, np.nan))):.2f} m", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig21_tile_deep.png")

    # --- Fig 22: three tiles across the wet spectrum ---------------------
    fig, axes = plt.subplots(3, 5, figsize=(12.6, 7.4))
    for r, kk in enumerate(sel):
        i = idx[kk]["index"]
        gt = z[f"t{i}_gt"]; fr = z[f"t{i}_frozen"]; wn = z[f"t{i}_win"]
        v = z[f"t{i}_valid"].astype(bool); base = z[f"t{i}_base"]
        coarse = z[f"t{i}_coarse"]
        sc = gt.shape[-1] // coarse.shape[-1]
        cup = np.repeat(np.repeat(coarse, sc, 0), sc, 1)[:gt.shape[0], :gt.shape[1]]
        vmax = max(float(np.nanpercentile(np.where(v, gt, np.nan), 99.5)), 0.05)
        cols = [("Coarse 10 m", cup), ("Bilinear", base), ("Frozen V0", fr),
                ("Fine-tuned", wn), ("Ground truth", gt)]
        for c, (ttl, arr) in enumerate(cols):
            ax = axes[r, c]
            depth_panel(ax, arr, v, ttl if r == 0 else "", vmax)
            if c == 0:
                ax.set_ylabel(f"peak {np.nanmax(np.where(v, gt, np.nan)):.1f} m", fontsize=8.5)
        peak = float(np.nanmax(np.where(v, gt, np.nan)))
        ef = float(np.nanmean(np.abs(fr[v] - gt[v])))
        ew = float(np.nanmean(np.abs(wn[v] - gt[v])))
        axes[r, 4].set_xlabel(f"MAE  frozen {ef:.3f} m   fine-tuned {ew:.3f} m", fontsize=8)
    fig.suptitle("Three test tiles spanning the inundation spectrum "
                 "(rows: deep, intermediate, shallow)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig22_tiles_spectrum.png")


# ---------------------------------------------------------------- 23
def fig_domain_maps():
    rows = load_json("per_tile.json")
    meta = load_json("tile_meta.json")
    NY, NX = 48, 27
    peak = np.full((NY, NX), np.nan)
    deep = np.full((NY, NX), np.nan)
    f_csi = np.full((NY, NX), np.nan)
    w_csi = np.full((NY, NX), np.nan)
    for r in meta:
        if r["iy"] >= 0 and r["ix"] >= 0:
            deep[r["iy"], r["ix"]] = r["deep_px"]
    for r in rows:
        iy, ix = r["iy"], r["ix"]
        if iy < 0 or ix < 0:
            continue
        peak[iy, ix] = r["peak_truth"]
        f_csi[iy, ix] = r["frozen_CSI_100"]
        w_csi[iy, ix] = r["win_CSI_100"]

    # The test split covers one southern band of the 48 x 27 tile grid.  Keeping
    # the full canvas would leave about nine tenths of every panel empty, which
    # reads as blank space, so tighten every panel to the band that carries data.
    occ = np.isfinite(peak)
    rr = np.where(occ.any(axis=1))[0]
    cc = np.where(occ.any(axis=0))[0]
    r0, r1, c0, c1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())

    def crop(arr):
        return arr[r0:r1 + 1, c0:c1 + 1]

    panels = [
        (crop(peak), "(a) Peak ground-truth depth\nper tile (m)", "turbo", 0.0,
         float(np.nanpercentile(peak, 98)), "m"),
        (crop(deep), "(b) Pixels deeper than 1 m\nper tile", "YlGnBu", 0.0,
         float(np.nanpercentile(deep, 98)), "count"),
        (crop(f_csi), "(c) Frozen V0\nCSI@1.00 m per tile", "RdYlGn", 0.0, 1.0, None),
        (crop(w_csi), "(d) Fine-tuned\nCSI@1.00 m per tile", "RdYlGn", 0.0, 1.0, None),
    ]
    # The band is 22 tiles wide and 9 tall, so the panels are laid out on a short
    # wide canvas with the axes boxes sized to that aspect.  Every box is placed by
    # hand so the colourbar keeps the image height and no empty space is left
    # inside a panel, which the default grid slot would produce.
    nrow, ncol = r1 - r0 + 1, c1 - c0 + 1
    W, H = 13.8, 1.70
    gap, cw, cp, left, right, bottom = 0.24, 0.075, 0.05, 0.14, 0.05, 0.08
    w = (W - left - right - 4 * gap - 5 * (cw + cp)) / 5
    ih = w * nrow / ncol
    fig = plt.figure(figsize=(W, H))

    def _panel(i, arr, ttl, cmap, vmin, vmax, clab, norm=None):
        x = left + i * (w + gap + cw + cp)
        ax = fig.add_axes([x / W, bottom / H, w / W, ih / H])
        im = ax.imshow(arr, cmap=cmap, vmin=None if norm else vmin,
                       vmax=None if norm else vmax, norm=norm,
                       interpolation="nearest", aspect="auto")
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        cb = fig.colorbar(im, cax=fig.add_axes([(x + w + cp) / W, bottom / H,
                                                cw / W, ih / H]))
        cb.ax.tick_params(labelsize=7)
        if clab:
            cb.set_label(clab, fontsize=7.5)
        return ax

    for i, (arr, ttl, cmap, vmin, vmax, clab) in enumerate(panels):
        _panel(i, arr, ttl, cmap, vmin, vmax, clab)
    d = crop(w_csi) - crop(f_csi)
    lim = max(float(np.nanpercentile(np.abs(d), 88)), 1e-3)
    _panel(4, d, "(e) Fine-tuned minus frozen\nCSI@1.00 m (red = improvement)",
           "RdBu_r", None, None, None,
           norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim))
    fig.suptitle(f"Spatial pattern of the test split band, tile rows {r0} to {r1} "
                 f"north to south and columns {c0} to {c1} west to east", fontsize=11,
                 y=0.975)
    save(fig, "fig23_domain_maps.png")
    np.savez_compressed(F / "domain_fields.npz", peak=peak, deep=deep, f_csi=f_csi, w_csi=w_csi)


# ---------------------------------------------------------------- 24
def fig_density():
    z = np.load(F / "pixels.npz")
    gt = z["truth"].astype(np.float64)
    fr = z["frozen"].astype(np.float64)
    wn = z["win"].astype(np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.0))

    hi = float(np.percentile(gt, 99.95))
    for ax, pred, name in ((axes[0, 0], fr, "Frozen V0"),
                           (axes[0, 1], wn, "Fine-tuned (w = 0.10)")):
        hb = ax.hexbin(gt, pred, gridsize=90, bins="log", cmap="viridis",
                       extent=(0, hi, 0, hi), mincnt=1)
        ax.plot([0, hi], [0, hi], "w--", lw=1.1)
        ax.axhline(1.0, color="#c0392b", lw=0.8, ls=":")
        ax.axvline(1.0, color="#c0392b", lw=0.8, ls=":")
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_xlabel("Ground-truth depth (m)"); ax.set_ylabel("Predicted depth (m)")
        ax.set_title(name, fontsize=10)
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("pixel count (log)", fontsize=7.5); cb.ax.tick_params(labelsize=7)
        # deep-water zoom stats
        sel = gt > 1.0
        ax.text(0.03, 0.97, f"n(> 1 m) = {sel.sum():,}\n"
                            f"mean pred = {pred[sel].mean():.3f} m\n"
                            f"mean truth = {gt[sel].mean():.3f} m",
                transform=ax.transAxes, va="top", fontsize=7.5,
                bbox=dict(fc="white", ec="#999", alpha=0.85, pad=2.5))

    # Bland-Altman style
    for ax, pred, name in ((axes[1, 0], fr, "Frozen V0"),
                           (axes[1, 1], wn, "Fine-tuned (w = 0.10)")):
        mean_d = 0.5 * (pred + gt)
        diff = pred - gt
        m = (mean_d <= np.percentile(mean_d, 99.9))
        ax.hexbin(mean_d[m], diff[m], gridsize=80, bins="log", cmap="viridis", mincnt=1)
        mu = float(np.mean(diff[m])); sd = float(np.std(diff[m]))
        ax.axhline(mu, color="#c0392b", lw=1.2)
        ax.axhline(mu + 1.96 * sd, color="#333", lw=0.9, ls="--")
        ax.axhline(mu - 1.96 * sd, color="#333", lw=0.9, ls="--")
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xlabel("Mean of truth and prediction (m)")
        ax.set_ylabel("Prediction minus truth (m)")
        ax.set_title(f"{name}: bias {mu:+.4f} m, limits {mu-1.96*sd:+.2f} to {mu+1.96*sd:+.2f}",
                     fontsize=9)
        ax.set_ylim(-3.0, 3.0)
    fig.suptitle("Point-wise agreement on the full test split", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig24_density.png")


# ---------------------------------------------------------------- 25
def fig_distributions():
    h = np.load(F / "hist.npz")
    edges = h["edges"]; ctr = 0.5 * (edges[:-1] + edges[1:])
    mom = load_json("moments.json")
    z = np.load(F / "pixels.npz")
    gt = z["truth"].astype(np.float64); fr = z["frozen"].astype(np.float64)
    wn = z["win"].astype(np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.6))

    # (a) error PDF, signed-log axis
    for ax, key, col, name in ((axes[0, 0], "frozen", C["frozen"], "Frozen V0"),
                               (axes[0, 0], "win", C["win"], "Fine-tuned")):
        ax.step(ctr, h[key], where="mid", color=col, lw=1.4, label=name)
    axes[0, 0].set_xscale("symlog", linthresh=0.01)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Error, prediction minus truth (m)")
    axes[0, 0].set_ylabel("Pixel count")
    axes[0, 0].set_title("(a) Error distribution, signed-log")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].set_xlim(-3, 3)

    # (b) empirical CDF of the negative tail
    for ax, arr, col, name in ((axes[0, 1], fr, C["frozen"], "Frozen V0"),
                               (axes[0, 1], wn, C["win"], "Fine-tuned")):
        e = np.sort(arr - gt)
        q = np.linspace(0, 1, 4000)
        ax.plot(np.quantile(e, q), q * 100, color=col, lw=1.5, label=name)
    axes[0, 1].set_xlim(-3, 1)
    axes[0, 1].set_xlabel("Error (m)"); axes[0, 1].set_ylabel("Cumulative share of pixels (%)")
    axes[0, 1].set_title("(b) Empirical CDF of the error")
    axes[0, 1].axvline(0, color="black", lw=0.7)
    axes[0, 1].legend(frameon=False, fontsize=8)

    # (c) QQ plot against a normal with the same mean and sd
    rng = np.random.default_rng(0)
    n = 400_000
    idx = rng.choice(gt.size, size=min(n, gt.size), replace=False)
    for ax, arr, col, name in ((axes[1, 0], fr, C["frozen"], "Frozen V0"),
                               (axes[1, 0], wn, C["win"], "Fine-tuned")):
        e = np.sort(arr[idx] - gt[idx])
        qn = np.sort(rng.standard_normal(e.size))
        ax.plot(qn[::25], e[::25], ".", ms=1.6, color=col, label=name)
    sd_all = float(np.std(fr - gt))
    axes[1, 0].plot([-6, 6], [-6 * sd_all, 6 * sd_all], "k--", lw=1.0,
                    label=f"normal, sd = {sd_all:.3f} m")
    axes[1, 0].set_xlim(-5, 5); axes[1, 0].set_ylim(-1.2, 1.2)
    axes[1, 0].set_xlabel("Standard-normal quantile")
    axes[1, 0].set_ylabel("Sample error quantile (m)")
    axes[1, 0].set_title("(c) Q-Q plot, heavier tails than normal")
    axes[1, 0].legend(frameon=False, fontsize=7.5, markerscale=5, loc="upper left")

    # (d) higher-order moments per depth bin. The two shallowest bins are dominated by
    # near-zero ground truth and compress the scale, so they are plotted separately.
    bins = mom["bins"][2:]
    x = np.arange(len(bins))
    ax = axes[1, 1]
    for k, col, name in (("frozen", C["frozen"], "Frozen V0"), ("win", C["win"], "Fine-tuned")):
        ax.semilogy(x, mom["models"][k]["excess_kurtosis"][2:], "o-", color=col, ms=4,
                    lw=1.4, label=f"excess kurtosis, {name}")
    ax.set_xticks(x); ax.set_xticklabels(bins, rotation=45, ha="right")
    ax.set_ylabel("Excess kurtosis (log scale)")
    ax.set_xlabel("True depth bin (m)")
    ax2 = ax.twinx()
    for k, col, ls in (("frozen", C["frozen"], "--"), ("win", C["win"], ":")):
        ax2.plot(x, mom["models"][k]["skew"][2:], ls, color=col, lw=1.3,
                 label=f"skewness, {'Frozen V0' if k == 'frozen' else 'Fine-tuned'}")
    ax2.set_ylabel("Skewness"); ax2.grid(False)
    ax2.set_ylim(0, None)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="upper right")
    ax.set_title("(d) Third and fourth moments, depth bins > 0.1 m")
    fig.suptitle("Higher-order statistics of the reconstruction error", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig25_distributions.png")


# ---------------------------------------------------------------- 26
def fig_exceedance():
    e = np.load(F / "exceed.npz")
    thr = e["thr"]; n = float(e["n"][0])
    z = np.load(F / "pixels.npz")
    gt = z["truth"]; fr = z["frozen"]; wn = z["win"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.8))
    for key, col, name in (("truth", C["truth"], "Ground truth"),
                           ("frozen", C["frozen"], "Frozen V0"),
                           ("win", C["win"], "Fine-tuned")):
        a1.plot(thr, e[key] / n * 100, color=col, lw=1.6, label=name)
    a1.set_yscale("log")
    a1.set_xlabel("Depth threshold (m)")
    a1.set_ylabel("Share of test pixels above threshold (%)")
    a1.set_title("(a) Inundation-exceedance curve")
    a1.legend(frameon=False, fontsize=8)
    a1.axvline(1.0, color="#c0392b", lw=0.7, ls=":")
    a1.text(1.05, 0.006, "1 m", fontsize=7.5, color="#c0392b")

    ts = np.unique(np.concatenate([np.linspace(0.02, 0.5, 25), np.linspace(0.55, 2.5, 25)]))
    for arr, col, name in ((fr, C["frozen"], "Frozen V0"), (wn, C["win"], "Fine-tuned")):
        csi = []
        for t in ts:
            tp = float(np.sum((arr > t) & (gt > t)))
            fp = float(np.sum((arr > t) & ~(gt > t)))
            fn = float(np.sum(~(arr > t) & (gt > t)))
            csi.append(tp / max(tp + fp + fn, 1))
        a2.plot(ts, csi, color=col, lw=1.6, label=name)
    a2.set_xlabel("Depth threshold (m)"); a2.set_ylabel("Critical Success Index")
    a2.set_title("(b) CSI as a function of the threshold")
    a2.legend(frameon=False, fontsize=8)
    a2.set_ylim(0, 1)
    for t, lab in ((0.05, "0.05"), (0.30, "0.30"), (1.00, "1.00")):
        a2.axvline(t, color="#999", lw=0.7, ls=":")
    fig.suptitle("How the two models partition the depth range", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig26_exceedance.png")


# ---------------------------------------------------------------- 27
def fig_bootstrap():
    rows = load_json("per_tile.json")
    mets = [("CSI_100", "CSI@1.00 m"), ("CSI_005", "CSI@0.05 m"),
            ("VolumeRelativeError", "Volume relative error"),
            ("PeakDepthError", "Peak depth error (m)")]
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4))
    summary = []
    for ax, (key, title) in zip(axes.ravel(), mets):
        a = np.array([r[f"frozen_{key}"] for r in rows], dtype=float)
        b = np.array([r[f"win_{key}"] for r in rows], dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        d = (b - a)[ok]
        lower_better = key in ("VolumeRelativeError", "PeakDepthError")
        dplot = -d if lower_better else d
        n = dplot.size
        boot = dplot[rng.integers(0, n, size=(20000, n))].mean(axis=1)
        md = float(dplot.mean()); sd = float(dplot.std(ddof=1))
        se = sd / np.sqrt(n)
        t = md / se if se else float("nan")
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ax.hist(boot, bins=60, density=True, color="#8ab0d6", alpha=0.85,
                ec="white", lw=0.3, label="bootstrap replicates")
        xs = np.linspace(boot.min(), boot.max(), 400)
        ax.plot(xs, np.exp(-0.5 * ((xs - md) / se) ** 2) / (se * np.sqrt(2 * np.pi)),
                color="#1f4e79", lw=1.4, label="normal approximation")
        ax.axvline(0, color="black", lw=1.0)
        ax.axvline(md, color=C["win"], lw=1.6, label=f"observed $\\bar d$ = {md:+.4f}")
        ax.axvline(lo, color="#333", lw=0.9, ls="--")
        ax.axvline(hi, color="#333", lw=0.9, ls="--")
        ax.set_xlabel("Mean paired change per tile, positive = better")
        ax.set_ylabel("Density")
        ax.set_title(title, fontsize=10.5)
        ax.text(0.02, 0.97,
                f"$t$ = {t:+.2f}\n95% CI [{lo:+.4f}, {hi:+.4f}]\nn = {n} tiles",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(fc="white", ec="#999", alpha=0.9, pad=3))
        if key not in ("CSI_100", "CSI_005"):
            ax.legend(frameon=False, fontsize=7)
        summary.append({"metric": key, "mean": md, "t": t, "ci": [float(lo), float(hi)],
                        "n": int(n)})
    fig.suptitle("Bootstrap sampling distributions of the per-tile paired difference\n"
                 "20 000 resamples of the 242 test tiles", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "fig27_bootstrap.png")
    (F / "bootstrap_tiles.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for s in summary:
        print("  ", s["metric"], "d=%+.4f t=%+.2f CI=[%+.4f,%+.4f]" % (
            s["mean"], s["t"], s["ci"][0], s["ci"][1]))


# ---------------------------------------------------------------- 28
def fig_autocorr():
    ac = load_json("autocorr.json")
    fr = np.array([a["moran"] for a in ac if a["model"] == "frozen"], dtype=float)
    wn = np.array([a["moran"] for a in ac if a["model"] == "win"], dtype=float)
    ok = np.isfinite(fr) & np.isfinite(wn)
    fr, wn = fr[ok], wn[ok]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 3.7))
    a1.hist(fr, bins=35, alpha=0.6, color=C["frozen"], ec="white", lw=0.4, label="Frozen V0")
    a1.hist(wn, bins=35, alpha=0.6, color=C["win"], ec="white", lw=0.4, label="Fine-tuned")
    a1.axvline(fr.mean(), color=C["frozen"], lw=1.6, ls="--")
    a1.axvline(wn.mean(), color=C["win"], lw=1.6, ls="--")
    a1.set_xlabel("Lag-1 Moran's I of the error field")
    a1.set_ylabel("Number of tiles")
    a1.set_title(f"(a) Spatial autocorrelation of the error\n"
                 f"mean  frozen {fr.mean():.3f}   fine-tuned {wn.mean():.3f}", fontsize=9.5)
    a1.legend(frameon=False, fontsize=8)

    a2.plot([0, 1], [0, 1], "k--", lw=0.9)
    a2.scatter(fr, wn, s=11, alpha=0.6, color="#4a7ab5", ec="none")
    a2.set_xlabel("Frozen V0, Moran's I")
    a2.set_ylabel("Fine-tuned, Moran's I")
    a2.set_title(f"(b) Per-tile pairing\nmean change {np.mean(wn-fr):+.4f}", fontsize=9.5)
    a2.set_xlim(0, 1); a2.set_ylim(0, 1)
    fig.suptitle("Spatial structure of the residuals: errors are locally clustered, "
                 "not white noise", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "fig28_autocorr.png")
    print(f"  Moran mean: frozen {fr.mean():.4f} win {wn.mean():.4f} "
          f"paired mean change {np.mean(wn-fr):+.4f} "
          f"t={np.mean(wn-fr)/(np.std(wn-fr, ddof=1)/np.sqrt(len(fr))):+.2f}")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["hyet", "tiles", "domain", "density", "dist", "exceed",
                             "boot", "auto"]
    if "tiles" in which:
        fig_tiles()
    if "domain" in which:
        fig_domain_maps()
    if "density" in which:
        fig_density()
    if "dist" in which:
        fig_distributions()
    if "exceed" in which:
        fig_exceedance()
    if "boot" in which:
        fig_bootstrap()
    if "auto" in which:
        fig_autocorr()
    print("done")
