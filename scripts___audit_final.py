"""Definitive source-level cross-reference check.

Authored captions live in two files:
  report_body.py  : 图 1,2,61,62,41..52(no 52?),63,64,71..85,20..28 ; 表 1..11,97
  build_report.py : 表 81..102 (with 88/89 generated programmatically)

`renumber()` maps each authored number to its document-order ordinal and rewrites
every reference with the same map, so a reference only resolves correctly when its
authored number equals the intended caption's authored number.

This script lists:
  (a) references whose authored number matches NO authored caption  -> definitely broken
  (b) each reference together with the nearest authored caption of the same kind
      (by source order within report_body.py) so a human can confirm intent.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

body_src = (ROOT / "scripts/report_body.py").read_text(encoding="utf-8")
build_src = (ROOT / "scripts/build_report.py").read_text(encoding="utf-8")

authored: dict[str, set[int]] = {"图": set(), "表": set()}
titles: dict[tuple[str, int], str] = {}

for src in (body_src, build_src):
    for m in re.finditer(r"<figcaption><b>图\s*(\d+)\s*([^<]{0,60})", src):
        authored["图"].add(int(m.group(1)))
        titles[("图", int(m.group(1)))] = m.group(2).strip()
    for m in re.finditer(r"<caption><b>表\s*(\d+)\s*([^<]{0,60})", src):
        authored["表"].add(int(m.group(1)))
        titles[("表", int(m.group(1)))] = m.group(2).strip()
# programmatic
for m in re.finditer(r'"([^"]+)",\s*"([^"]+)",\s*"(\d+)"', build_src):
    authored["表"].add(int(m.group(3)))
    titles[("表", int(m.group(3)))] = m.group(2) + " 机会校正一致性指标"

print("authored 图:", sorted(authored["图"]))
print("authored 表:", sorted(authored["表"]))
print()

lines = body_src.splitlines()
bad = []
for i, ln in enumerate(lines):
    if re.search(r"<figcaption><b>|<caption><b>", ln):
        continue
    for m in re.finditer(r"([图表])\s*(\d+)", ln):
        kind, n = m.group(1), int(m.group(2))
        # skip when the match is part of a preceding word fragment (e.g. 图表)
        pre = ln[max(0, m.start() - 1):m.start()]
        if pre == "图" or pre == "表":
            continue
        if n not in authored[kind]:
            bad.append((i + 1, kind, n, ln.strip()[:100]))

print("=" * 100)
print("(a) references whose number matches NO authored caption")
print("=" * 100)
for line, kind, n, txt in bad:
    print(f"  L{line:<5} {kind}{n}   {txt}")
if not bad:
    print("  none")

print()
print("=" * 100)
print("(b) full listing (authored numbers) with nearest same-kind captions")
print("=" * 100)
caps = []
for i, ln in enumerate(lines):
    m = re.search(r"<figcaption><b>图\s*(\d+)\s*([^<]{0,40})", ln)
    if m:
        caps.append((i, "图", int(m.group(1)), m.group(2).strip()))
    m = re.search(r"<caption><b>表\s*(\d+)\s*([^<]{0,40})", ln)
    if m:
        caps.append((i, "表", int(m.group(1)), m.group(2).strip()))
caps.sort()

for i, ln in enumerate(lines):
    if re.search(r"<figcaption><b>|<caption><b>", ln):
        continue
    for m in re.finditer(r"([图表])\s*(\d+)", ln):
        kind, n = m.group(1), int(m.group(2))
        pre = ln[max(0, m.start() - 1):m.start()]
        if pre in ("图", "表"):
            continue
        prev = [c for c in caps if c[1] == kind and c[0] <= i]
        nxt = [c for c in caps if c[1] == kind and c[0] >= i]
        pv = f"{prev[-1][2]}:{prev[-1][3][:18]}" if prev else "-"
        nx = f"{nxt[0][2]}:{nxt[0][3][:18]}" if nxt else "-"
        tag = "ok" if (prev and prev[-1][2] == n) or (nxt and nxt[0][2] == n) else "??"
        print(f"  L{i+1:<5} {tag} {kind}{n:<4} prev=[{pv:<26}] next=[{nx}]")
