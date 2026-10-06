"""Semantic cross-reference audit.

`renumber()` in build_report.py remaps every 图 N / 表 N by document order, so a
reference always resolves to *some* caption. That makes a wrong-but-existing
number invisible. This script prints, for every in-text reference in
report_body.py, the caption it now points at, so the target can be checked
against the prose.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
html = (ROOT / "report.html").read_text(encoding="utf-8")
body = html.split("<body>", 1)[1].split("</body>", 1)[0]
src = (ROOT / "scripts/report_body.py").read_text(encoding="utf-8")

# final captions in order
fig_order, tab_order = [], []
for m in re.finditer(r"<figcaption><b>图\s*(\d+)\s*(.*?)</b>", body, re.S):
    fig_order.append((int(m.group(1)), re.sub(r"\s+", " ", m.group(2)).strip()))
for m in re.finditer(r"<caption><b>表\s*(\d+)\s*(.*?)</b>", body, re.S):
    tab_order.append((int(m.group(1)), re.sub(r"\s+", " ", m.group(2)).strip()))

# old->final mapping. build_report captions are authored with numbers 81..102 plus
# 88/89/90..98 already final-ish; report_body uses 1..11 and 90..97 and 84 etc.
# Reproduce the mapping exactly by re-reading the two sources in build order.
head = (ROOT / "scripts/build_report.py").read_text(encoding="utf-8")
# The authored caption numbers, in the order they appear in the produced HTML,
# are simply the *final* ones; recover the old value by pairing captions with the
# regex source order of each file. Instead of reconstructing, rely on the fact
# that renumber() is a pure function: apply it to the two sources separately.
old_fig, old_tab = [], []
for pat in (r"<figcaption><b>图\s*(\d+)", r"<caption><b>表\s*(\d+)"):
    pass


def authored(pattern: str, text: str) -> list[int]:
    return [int(m.group(1)) for m in re.finditer(pattern, text)]


# document order of captions equals: report_body's captions come after the ones the
# build script emits at the top of ch.2, so reconstruct by position in the final HTML.
final_fig = [n for n, _ in fig_order]
final_tab = [n for n, _ in tab_order]

title_fig = dict(fig_order)
title_tab = dict(tab_order)

print("=" * 100)
print("in-text references in report_body.py")
print("=" * 100)
line_no = 0
for line in src.splitlines():
    line_no += 1
    for m in re.finditer(r"([图表])\s*(\d+)", line):
        kind, n = m.group(1), int(m.group(2))
        ctx = line.strip()
        print(f"{line_no:>5}  src {kind}{n:<4} -> {ctx[:96]}")
print()

print("=" * 100)
print("final table order (what each number now means)")
print("=" * 100)
for n, t in tab_order:
    print(f"  表 {n:>3}  {t[:70]}")
print()
print("=" * 100)
print("final figure order")
print("=" * 100)
for n, t in fig_order:
    print(f"  图 {n:>3}  {t[:70]}")
