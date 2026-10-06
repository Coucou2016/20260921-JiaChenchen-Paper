"""Compare references in the OLD report.md (pre-rebuild, consistent numbering)
with the same sentences in the NEW report.html.

The old file was produced by the same renumber() pass, so its numbering is the
intended one. Any sentence whose number changed between the two is a regression
introduced when new captions were inserted.
"""
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
old = (ROOT / "report.md").read_text(encoding="utf-8")

new = (ROOT / "report.html").read_text(encoding="utf-8")
new = new.split("<body>", 1)[1].split("</body>", 1)[0]
new = re.sub(r"<img[^>]*>", " ", new)
new = re.sub(r"<svg.*?</svg>", " ", new, flags=re.S)
new = re.sub(r"<[^>]+>", " ", new)
new = re.sub(r"\s+", " ", new)

old_flat = re.sub(r"\s+", " ", old)

# split into sentences on the CJK full stop and a few others
def sentences(t: str):
    parts = re.split(r"(?<=[。！？])", t)
    return [p.strip() for p in parts if len(p.strip()) > 12]


old_s = sentences(old_flat)
new_s = sentences(new)
new_index = {s[:40]: s for s in new_s}

CHANGED = []
for i, s in enumerate(old_s):
    if not re.search(r"[图表]\s*\d+", s):
        continue
    key = s[:40]
    cand = new_index.get(key)
    if cand is None:
        # fuzzy
        m = difflib.get_close_matches(s, new_s, n=1, cutoff=0.90)
        cand = m[0] if m else None
    if cand is None:
        continue
    a = re.findall(r"[图表]\s*\d+", s)
    b = re.findall(r"[图表]\s*\d+", cand)
    if a != b:
        CHANGED.append((s, cand))

print(f"sentences with table/figure refs compared: "
      f"{sum(1 for s in old_s if re.search(r'[图表]\\s*\\d+', s))}")
print(f"number set changed in: {len(CHANGED)}\n")
for oldc, newc in CHANGED:
    print("OLD:", oldc[:170])
    print("NEW:", newc[:170])
    print("-" * 100)
