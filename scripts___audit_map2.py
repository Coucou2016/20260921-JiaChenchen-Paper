"""Capture the authored->final numbering map by instrumenting renumber(), then list
every in-text reference with the final caption it resolves to.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_report as BR   # noqa: E402

CAPTURED = {}


def remap_capture(kind, html):
    cap_re = re.compile(rf"<figcaption><b>{kind}\s*(\d+)" if kind == "图"
                        else rf"<caption><b>{kind}\s*(\d+)")
    order = [int(m.group(1)) for m in cap_re.finditer(html)]
    CAPTURED[kind] = {old: i + 1 for i, old in enumerate(order)}
    return html


# Build the BODY the same way build() does, but intercept before renumber.
def build_body():
    from report_body import body as report_body
    BODY = report_body(BR.IMG, BR.TIMELINE_SVG)
    repl = {
        "@@PERTILE@@": BR.per_tile_table(), "@@HIGHORDER@@": BR.higher_order_table(),
        "@@PREM_ERROR@@": BR.pre_error_table(), "@@PREM_DECOMP@@": BR.pre_decomp_table(),
        "@@PREM_SOURCES@@": BR.pre_sources_table(), "@@PREM_TERRAIN@@": BR.pre_terrain_table(),
        "@@PREM_CORR@@": BR.pre_corr_table(),
        "@@PREM_VARDECOMP@@": BR.pre_vardecomp_table(),
        "@@PREM_DIALECT@@": BR.pre_dialect_table(),
        "@@PREM_CONS_CONT@@": BR.pre_cons_cont_table(),
        "@@PREM_CONS_CAT@@": BR.pre_cons_cat_table(),
        "@@PREM_CONS_DIST@@": BR.pre_cons_dist_table(),
        "@@PREM_CONS_VM@@": BR.pre_cons_vm_table(),
        "@@PREM_PAT_TABLE@@": BR.pat_table(), "@@PATX_TABLES@@": BR.patx_tables(),
        "@@PREM_MODEL@@": BR.pre_model_table(), "@@PREM_PAIRED@@": BR.pre_paired_table(),
    }
    for tok, txt in repl.items():
        BODY = BODY.replace(tok, txt)
    for tok, txt in BR.pre_cons_text().items():
        BODY = BODY.replace(tok, txt)
    for tok, txt in BR.pat_text().items():
        BODY = BODY.replace(tok, txt)
    for tok, txt in BR.patx_text().items():
        BODY = BODY.replace(tok, txt)
    return BODY


BODY = build_body()

# reproduce renumber's order capture, then run the real renumber
for kind in ("图", "表"):
    remap_capture(kind, BODY)
final = BR.renumber(BODY)

fig_cap, tab_cap = {}, {}
for m in re.finditer(r"<figcaption><b>图\s*(\d+)\s*([^<]{0,60})", final):
    fig_cap[int(m.group(1))] = m.group(2).strip()
for m in re.finditer(r"<caption><b>表\s*(\d+)\s*([^<]{0,60})", final):
    tab_cap[int(m.group(1))] = m.group(2).strip()

src = (ROOT / "scripts/report_body.py").read_text(encoding="utf-8")
lines = src.splitlines()

print("=" * 116)
print("REFERENCE -> FINAL CAPTION IT RESOLVES TO (from report_body.py)")
print("=" * 116)
for i, ln in enumerate(lines):
    if re.search(r"<figcaption><b>|<caption><b>", ln):
        continue
    for m in re.finditer(r"([图表])\s*(\d+)", ln):
        kind, n = m.group(1), int(m.group(2))
        pre = ln[max(0, m.start() - 1):m.start()]
        if pre in ("图", "表"):
            continue
        fnum = CAPTURED[kind].get(n)
        caps = fig_cap if kind == "图" else tab_cap
        title = caps.get(fnum, "<MISSING>")
        print(f"L{i+1:<5} {kind}{n:<4} -> {kind} {fnum}  {title[:56]}")

print()
print("=== authored->final maps ===")
for k in ("图", "表"):
    print(k, dict(sorted(CAPTURED[k].items(), key=lambda x: x[1])))
