#!/usr/bin/env python
"""Build a self-contained HTML gallery of every visualisation produced so far.

Scans outputs/v0_10m2m_hmax/visualizations/ and emits index.html with:
  * a live run-state banner (epoch, best metrics, ETA)
  * every figure grouped by category with captions
  * metric tables (per-split averages, checkpoint header metrics)
  * CSV links for the per-tile tables

The page uses relative <img> paths only; nothing is embedded, so it stays small
and always reflects the files currently on disk.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "v0_10m2m_hmax"
VIZ = OUT_DIR / "visualizations"

CATEGORIES: list[tuple[str, str, str]] = [
    ("01_", "Training dynamics & progress", "Live curves from history.jsonl, updated every refresh cycle."),
    ("02_", "Dataset domain, splits & error geography", "Where the model does well/poorly across the study area."),
    ("03_", "Physical diagnostics", "Depth/slope/land-use conditioned error and residual anchoring."),
]

METRIC_ORDER = ["CSI_005", "F1_005", "CSI_030", "F1_030", "CSI_100", "F1_100",
                "RMSE_wet", "MAE_wet", "RMSE_all", "PeakDepthError",
                "VolumeRelativeError", "FloodAreaRelativeError", "PSNR", "SSIM"]


def rel(p: Path) -> str:
    return p.relative_to(VIZ).as_posix()


def read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def trainer_alive() -> bool:
    pf = OUT_DIR / "train.pid"
    if not pf.exists():
        return False
    try:
        pid = pf.read_text(encoding="ascii").strip()
    except OSError:
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                 capture_output=True, text=True, errors="ignore").stdout
            return pid in out
        os.kill(int(pid), 0)
        return True
    except Exception:  # noqa: BLE001
        return False


def run_state() -> dict:
    bs = read_json(OUT_DIR / "best_summary.json") or {}
    hist = OUT_DIR / "history.jsonl"
    last: dict = {}
    if hist.exists():
        lines = [l for l in hist.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
        for line in reversed(lines):
            try:
                last = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    return {"best": bs, "last": last, "alive": trainer_alive()}


def metric_table(rows: list[dict], keys: list[str]) -> str:
    if not rows:
        return ""
    head = "".join(f"<th>{html.escape(k)}</th>" for k in keys)
    body = []
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k)
            if v is None or (isinstance(v, float) and v != v):
                cells.append("<td class='na'>-</td>")
            elif isinstance(v, (int, float)):
                cells.append(f"<td>{v:.4f}</td>")
            else:
                cells.append(f"<td>{html.escape(str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def collect() -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {prefix: [] for prefix, _, _ in CATEGORIES}
    groups["tiles"] = []
    groups["evolution"] = []
    for p in sorted(VIZ.rglob("*.png")):
        name = p.name
        placed = False
        for prefix, _, _ in CATEGORIES:
            if name.startswith(prefix):
                groups[prefix].append(p)
                placed = True
                break
        if not placed:
            if "evolution" in p.parts:
                groups["evolution"].append(p)
            else:
                groups["tiles"].append(p)
    return groups


CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0;
       background:#f6f7f9; color:#1c1e21; }
@media (prefers-color-scheme: dark) {
  body { background:#15171a; color:#e6e6e6; }
  .card { background:#1e2126 !important; border-color:#2c3037 !important; }
  table { background:#1e2126; }
  th { background:#262a30 !important; }
  td, th { border-color:#2c3037 !important; }
  .banner { background:#1b2a1e !important; border-color:#2e7d32 !important; }
  code { background:#262a30 !important; }
  h2 { border-color:#2c3037 !important; }
}
header { padding: 24px 32px 12px; }
h1 { margin: 0 0 4px; font-size: 24px; }
h2 { font-size: 18px; margin: 28px 0 6px; border-bottom:2px solid #dfe1e5; padding-bottom:6px; }
h3 { font-size: 14px; margin: 20px 0 6px; }
.sub { color:#6b7076; font-size: 13px; }
.banner { background:#eaf5ec; border:1px solid #b7dfc0; border-radius:10px;
          padding:14px 18px; margin:16px 0; display:flex; flex-wrap:wrap; gap:28px; }
.banner div { font-size:13px; }
.banner b { display:block; font-size:17px; font-weight:600; }
main { padding: 0 32px 48px; }
.grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(430px, 1fr)); gap:18px; }
.card { background:#fff; border:1px solid #e2e4e8; border-radius:10px; overflow:hidden; }
.card a { display:block; }
.card img { width:100%; display:block; }
.cap { padding:8px 12px; font-size:12px; color:#55595e; word-break:break-all; }
table { border-collapse:collapse; font-size:12px; margin:10px 0 18px; background:#fff; }
th, td { border:1px solid #e2e4e8; padding:4px 8px; text-align:right; }
th { background:#eceff3; font-weight:600; }
td.na { color:#b0b0b0; }
code { background:#eceff3; padding:1px 5px; border-radius:4px; font-size:12px; }
.pill { display:inline-block; padding:2px 9px; border-radius:999px; font-size:11px;
        background:#2e7d32; color:#fff; }
.pill.off { background:#9e9e9e; }
a { color:#1565c0; }
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(VIZ / "index.html"))
    args = ap.parse_args()

    state = run_state()
    best = state["best"]
    last = state["last"]
    groups = collect()

    summaries = []
    for d in sorted([d for d in VIZ.glob("ep*") if d.is_dir()], key=lambda d: d.name):
        for s in sorted(d.glob("summary_*.json")):
            j = read_json(s)
            if j:
                summaries.append(j)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alive = state["alive"]

    def fmt(v, spec=".4f"):
        try:
            return format(float(v), spec)
        except (TypeError, ValueError):
            return "n/a"

    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>HydroGeo-SRNO visualisations</title>",
        f"<style>{CSS}</style>",
        "<header>",
        "<h1>HydroGeo-SRNO V0 - visual progress report</h1>",
        f"<div class='sub'>10 m to 2 m, h_max | generated {now} | "
        f"<span class='pill{'' if alive else ' off'}'>trainer {'running' if alive else 'stopped'}</span></div>",
        "<div class='banner'>",
        f"<div>last epoch<b>{last.get('epoch', 'n/a')} / 300</b></div>",
        f"<div>best CSI@0.05<b>{fmt(best.get('best_csi'))}</b></div>",
        f"<div>best RMSE wet<b>{fmt(best.get('best_rmse_wet'))} m</b></div>",
        f"<div>val loss (last)<b>{fmt(last.get('loss'))}</b></div>",
        f"<div>val MAE wet (last)<b>{fmt(last.get('MAE_wet'))} m</b></div>",
        f"<div>epoch time<b>{fmt(last.get('sec'), '.0f')} s</b></div>",
        "</div></header><main>",
    ]

    parts.append("<h2>Run state</h2>")
    if last:
        keys = ["epoch", "global_step", "loss", "RMSE_all", "RMSE_wet", "MAE_wet",
                "CSI_005", "F1_005", "CSI_030", "CSI_100", "PeakDepthError",
                "FloodAreaRelativeError", "VolumeRelativeError", "PSNR", "SSIM", "sec"]
        parts.append(metric_table([{k: last.get(k) for k in keys}], keys))
    parts.append("<div class='sub'>Live curves: <code>outputs/v0_10m2m_hmax/history.jsonl</code> - "
                 "<code>best_summary.json</code></div>")

    if summaries:
        parts.append("<h2>Checkpoint evaluation summaries</h2>")
        agg = [{"epoch": s.get("epoch"), "split": s.get("split"),
                "tiles": s.get("n_tiles_scanned") or s.get("n_tiles_described"),
                **(s.get("avg_metrics") or {})} for s in summaries]
        parts.append(metric_table(agg, ["epoch", "split", "tiles", *METRIC_ORDER]))

    for prefix, title, blurb in CATEGORIES:
        figs = groups.get(prefix, [])
        if not figs:
            continue
        parts.append(f"<h2>{html.escape(title)}</h2><div class='sub'>{html.escape(blurb)}</div>")
        parts.append("<div class='grid'>")
        for p in figs:
            parts.append(
                f"<div class='card'><a href='{rel(p)}' target='_blank'>"
                f"<img loading='lazy' src='{rel(p)}' alt='{html.escape(p.name)}'></a>"
                f"<div class='cap'>{html.escape(p.name)}</div></div>"
            )
        parts.append("</div>")

    tiles = groups.get("tiles", [])
    if tiles:
        parts.append("<h2>Tile reconstructions & milestone comparisons</h2>")
        parts.append("<div class='sub'>Coarse input, bilinear, model, truth, error maps, "
                     "wet-extent agreement and the learned residual, grouped by checkpoint tag.</div>")
        by_dir: dict[str, list[Path]] = {}
        for p in tiles:
            by_dir.setdefault(str(p.parent.relative_to(VIZ).as_posix()), []).append(p)
        for d in sorted(by_dir):
            parts.append(f"<h3><code>{html.escape(d)}</code></h3><div class='grid'>")
            for p in by_dir[d]:
                parts.append(
                    f"<div class='card'><a href='{rel(p)}' target='_blank'>"
                    f"<img loading='lazy' src='{rel(p)}' alt='{html.escape(p.name)}'></a>"
                    f"<div class='cap'>{html.escape(p.name)}</div></div>"
                )
            parts.append("</div>")

    evo = groups.get("evolution", [])
    if evo:
        parts.append("<h2>Training evolution (image space)</h2>")
        parts.append("<div class='sub'>How one fixed tile's prediction changes as the checkpoint improves. "
                     "Driven by <code>outputs/v0_10m2m_hmax/snapshots/</code> "
                     "(written by <code>scripts/watch_checkpoints.py</code>).</div>")
        parts.append("<div class='grid'>")
        for p in evo:
            parts.append(
                f"<div class='card'><a href='{rel(p)}' target='_blank'>"
                f"<img loading='lazy' src='{rel(p)}' alt='{html.escape(p.name)}'></a>"
                f"<div class='cap'>{html.escape(rel(p))}</div></div>"
            )
        parts.append("</div>")

    csvs = sorted(VIZ.rglob("*.csv"))
    if csvs:
        parts.append("<h2>Data tables</h2><ul>")
        for p in csvs:
            parts.append(f"<li><a href='{rel(p)}'>{html.escape(rel(p))}</a></li>")
        parts.append("</ul>")

    parts.append("<h2>Refresh</h2><div class='sub'>Regenerate everything once:</div>")
    parts.append("<p><code>python scripts/visualize_all.py</code> &nbsp; "
                 "continuously: <code>python scripts/auto_visualize.py --interval 1500</code></p>")
    parts.append("</main>")

    out = Path(args.out)
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out} ({len(tiles)} tile figures, {len(evo)} evolution figures, "
          f"{len(summaries)} summaries)")


if __name__ == "__main__":
    main()
