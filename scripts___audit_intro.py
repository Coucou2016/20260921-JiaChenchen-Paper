"""For every caption in the built report, look at the prose paragraph that
introduces it (the nearest preceding/in-figure paragraph) and report the table or
figure number that prose cites. A caption introduced by "表 6 给出…" while the
caption itself reads "表 7" is the off-by-one pattern.

Only matches of the exact forms 图 N and 表 N (with optional space) inside prose are
counted; HTML attributes and code are stripped first.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
html = (ROOT / "report.html").read_text(encoding="utf-8")
body = html.split("<body>", 1)[1].split("</body>", 1)[0]

# strip images (base64 payloads) and svg to keep offsets meaningful
body = re.sub(r"<img[^>]*>", "<img>", body)
body = re.sub(r"<svg.*?</svg>", "<svg/>", body, flags=re.S)

caps = []
for m in re.finditer(r"<figcaption><b>(图)\s*(\d+)", body):
    caps.append((m.start(), m.group(1), int(m.group(2))))
for m in re.finditer(r"<caption><b>(表)\s*(\d+)", body):
    caps.append((m.start(), m.group(1), int(m.group(2))))
caps.sort()

paras = [(m.start(), m.end(), re.sub(r"<[^>]+>", "", m.group(0)))
         for m in re.finditer(r"<p[^>]*>.*?</p>", body, re.S)]

print("=" * 104)
print("cap №    leading paragraph's own citation(s)                 verdict")
print("=" * 104)
issues = []
for pos, kind, num in caps:
    # nearest paragraph that ends before the caption (the introducing prose)
    best = None
    for s, e, txt in paras:
        if e <= pos + 5:
            best = (s, e, txt)
        else:
            break
    if best is None:
        continue
    txt = re.sub(r"\s+", "", best[2])
    cites = sorted({int(m.group(2)) for m in
                    re.finditer(rf"({kind})\s*(\d+)", txt)})
    if not cites:
        continue
    tail = txt[-55:]
    ok = num in cites
    verdict = "ok" if ok else "*** MISMATCH ***"
    print(f"  {kind}{num:<4} cites {kind}{cites}   {verdict}   ...{tail}")
    if not ok:
        issues.append((kind, num, cites, tail))

print()
print("=" * 104)
print(f"{len(issues)} mismatching introductions")
for kind, num, cites, tail in issues:
    print(f"  caption {kind}{num} but intro cites {kind}{cites}:  ...{tail}")
