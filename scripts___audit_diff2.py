"""Sentence-level comparison of table/figure references between the pre-existing
report.md and the freshly built report.html.

Both files were produced by the same renumber() pass, so a sentence that references
a different number in the two is either a pre-existing bug or a regression. Either
way it needs fixing.
"""
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
old = re.sub(r"\s+", " ", (ROOT / "report.md").read_text(encoding="utf-8"))

new = (ROOT / "report.html").read_text(encoding="utf-8")
new = new.split("<body>", 1)[1].split("</body>", 1)[0]
new = re.sub(r"<img[^>]*>", " ", new)
new = re.sub(r"<svg.*?</svg>", " ", new, flags=re.S)
new = re.sub(r"<[^>]+>", " ", new)
new = re.sub(r"\s+", " ", new)

ref_re = re.compile(r"[图表]\s*\d+")


def sentences(t):
    return [p.strip() for p in re.split(r"(?<=[。！？])", t) if len(p.strip()) > 10]


old_s = [s for s in sentences(old) if ref_re.search(s)]
new_s = sentences(new)
new_by_prefix = {}
for s in new_s:
    new_by_prefix.setdefault(s[:26], []).append(s)

print(f"old sentences with refs: {len(old_s)}\n")
mismatch = 0
for s in old_s:
    key = s[:26]
    cands = new_by_prefix.get(key, [])
    cand = cands[0] if cands else None
    if cand is None:
        m = difflib.get_close_matches(s, new_s, n=1, cutoff=0.92)
        cand = m[0] if m else None
    if cand is None:
        print(f"[NOT FOUND in new] {s[:150]}")
        mismatch += 1
        continue
    a = ref_re.findall(s)
    b = ref_re.findall(cand)
    if a != b:
        mismatch += 1
        print(f"OLD {a}  {s[:140]}")
        print(f"NEW {b}  {cand[:140]}")
        print("-" * 110)
print(f"\nmismatched sentences: {mismatch}")
