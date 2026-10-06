"""For every in-text reference in report_body.py, show the nearest preceding and
nearest following caption of the same kind (by source line), with the authored
numbers. The reader can then judge which one the sentence means.

A reference whose number equals neither neighbour, or whose neighbours are the
intended target with a different number, is a bug.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "scripts/report_body.py").read_text(encoding="utf-8")
lines = src.splitlines()

# caption line index -> (kind, number, title)
caps = []   # (line_idx, kind, num, title)
for i, ln in enumerate(lines):
    m = re.search(r"<figcaption><b>图\s*(\d+)\s*([^<]{0,50})", ln)
    if m:
        caps.append((i, "图", int(m.group(1)), m.group(2).strip()))
    m = re.search(r"<caption><b>表\s*(\d+)\s*([^<]{0,50})", ln)
    if m:
        caps.append((i, "表", int(m.group(1)), m.group(2).strip()))
caps.sort()


def neighbours(kind: str, i: int):
    prev = [c for c in caps if c[1] == kind and c[0] <= i]
    nxt = [c for c in caps if c[1] == kind and c[0] >= i]
    return (prev[-1] if prev else None), (nxt[0] if nxt else None)


for i, ln in enumerate(lines):
    if re.search(r"<figcaption><b>|<caption><b>", ln):
        continue
    for m in re.finditer(r"([图表])\s*(\d+)", ln):
        kind, n = m.group(1), int(m.group(2))
        if m.start() and ln[max(0, m.start() - 30):m.start()].rstrip().endswith(("图", "表")):
            continue
        prev, nxt = neighbours(kind, i)
        pv = f"{kind}{prev[2]}('{prev[3][:22]}')" if prev else "-"
        nx = f"{kind}{nxt[2]}('{nxt[3][:22]}')" if nxt else "-"
        mark = "  "
        if prev and prev[2] == n and (nxt is None or nxt[2] != n):
            mark = "ok"
        elif nxt and nxt[2] == n:
            mark = "nx"
        else:
            mark = "??"
        print(f"L{i+1:<5} {mark} cited {kind}{n:<4} | prev {pv:<34} | next {nx}")
