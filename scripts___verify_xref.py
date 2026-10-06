"""Post-fix verification: every prose reference must resolve, and every caption
that is introduced by prose must be the one the prose names.

Run after build_report.py. Exits non-zero when a mismatch is found.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
body = (ROOT / "report.html").read_text(encoding="utf-8")
body = body.split("<body>", 1)[1].split("</body>", 1)[0]
body = re.sub(r"<img[^>]*>", "<img>", body)
body = re.sub(r"<svg.*?</svg>", "<svg/>", body, flags=re.S)

caps = {}
for m in re.finditer(r"<figcaption><b>图\s*(\d+)\s*([^<]{0,60})", body):
    caps[("图", int(m.group(1)))] = m.group(2).strip()
for m in re.finditer(r"<caption><b>表\s*(\d+)\s*([^<]{0,60})", body):
    caps[("表", int(m.group(1)))] = m.group(2).strip()

fails = []

# 1. every reference must resolve
for m in re.finditer(r"([图表])\s*(\d+)", body):
    kind, n = m.group(1), int(m.group(2))
    if (kind, n) not in caps:
        fails.append(f"dangling {kind} {n}")

# 2. the paragraph introducing a caption must cite the caption's own number,
#    when it cites that kind of object at all
paras = [(m.start(), m.end(), re.sub(r"<[^>]+>", "", m.group(0)))
         for m in re.finditer(r"<p[^>]*>.*?</p>", body, re.S)]

for m in re.finditer(r"<(?:caption|figcaption)><b>(图|表)\s*(\d+)", body):
    kind, n = m.group(1), int(m.group(2))
    pos = m.start()
    best = None
    for s, e, txt in paras:
        if e <= pos + 5:
            best = txt
        else:
            break
    if best is None:
        continue
    txt = re.sub(r"\s+", "", best)
    hits = [int(x.group(2)) for x in re.finditer(rf"({kind})\s*(\d+)", txt)]
    if hits and n not in hits and abs(max(hits) - n) <= 2:
        # A forward reference to another section is legitimate, and a paragraph may
        # introduce several objects at once. Flag only when the paragraph never
        # names the caption it precedes but does cite one within two of it.
        fails.append(
            f"{kind}{n} introduced by prose citing only {kind}{hits}: ...{txt[-45:]}")

print(f"captions: {len(caps)}   checks failed: {len(fails)}")
for f in fails:
    print("  FAIL", f)
sys.exit(1 if fails else 0)
