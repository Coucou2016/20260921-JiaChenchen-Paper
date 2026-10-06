"""Reference-vs-neighbourhood audit on the built report.html.

For every in-text 图 N / 表 N, look at the captions within a window of W chars on
either side (images and svg stripped first so offsets are comparable). If the
referenced number is not among them, the number almost certainly belongs to a
neighbouring item and the pointer is wrong.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
body = (ROOT / "report.html").read_text(encoding="utf-8")
body = body.split("<body>", 1)[1].split("</body>", 1)[0]
body = re.sub(r"<img[^>]*>", "<img>", body)
body = re.sub(r"<svg.*?</svg>", "<svg/>", body, flags=re.S)

W = 3500

caps = []
for m in re.finditer(r"<figcaption><b>(图)\s*(\d+)", body):
    caps.append((m.start(), "图", int(m.group(2))))
for m in re.finditer(r"<caption><b>(表)\s*(\d+)", body):
    caps.append((m.start(), "表", int(m.group(2))))
caps.sort()
cap_nums = {(k, n) for _, k, n in caps}

# headings for reporting
heads = []
for m in re.finditer(r"<h2>(.*?)</h2>|<h3>(.*?)</h3>", body, re.S):
    heads.append((m.start(), re.sub(r"<[^>]+>", "", m.group(0)).strip()))


def section_of(pos: int) -> str:
    cur = "?"
    for hp, ht in heads:
        if hp <= pos:
            cur = ht
        else:
            break
    return cur


# references that are inside prose paragraphs only
in_prose = set()
for m in re.finditer(r"<p[^>]*>.*?</p>", body, re.S):
    for r in re.finditer(r"[图表]\s*\d+", m.group(0)):
        in_prose.add(m.start() + r.start())

flags = []
allrefs = []
for m in re.finditer(r"([图表])\s*(\d+)", body):
    if m.start() not in in_prose:
        continue
    kind, n = m.group(1), int(m.group(2))
    if (kind, n) not in cap_nums:
        flags.append((m.start(), f"{kind}{n} has no caption at all"))
        continue
    nearby = [c for c in caps if abs(c[0] - m.start()) <= W]
    near_nums = {(k, num) for _, k, num in nearby}
    allrefs.append((m.start(), kind, n, len(nearby)))
    if (kind, n) not in near_nums:
        # find nearest caption of same kind
        same = [c for c in caps if c[1] == kind]
        nearest = min(same, key=lambda c: abs(c[0] - m.start()))
        flags.append((m.start(),
                      f"{kind}{n} NOT within {W}ch; nearest {kind} is "
                      f"{nearest[2]} at {abs(nearest[0]-m.start())}ch"))

print(f"prose references checked: {len(allrefs)}")
print("=" * 100)
print("FLAGGED")
for pos, msg in flags:
    sec = section_of(pos)
    frag = re.sub(r"<[^>]+>", "", body[max(0, pos - 120):pos + 40])
    frag = re.sub(r"\s+", " ", frag)
    print(f"  [{sec[:24]}] {msg}")
    print(f"        ...{frag}")
print()
print(f"total flagged: {len(flags)}")
