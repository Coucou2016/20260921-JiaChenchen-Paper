"""Print every 图/表 reference in report_body.py with surrounding prose, and mark
whether the quoted number is one of the authored captions in that file.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "scripts/report_body.py").read_text(encoding="utf-8")

authored_fig = {int(m.group(1)) for m in
                re.finditer(r"<figcaption><b>图\s*(\d+)", src)}
authored_tab = {int(m.group(1)) for m in
                re.finditer(r"<caption><b>表\s*(\d+)", src)}

print("authored figures:", sorted(authored_fig))
print("authored tables :", sorted(authored_tab))
print()

lines = src.splitlines()
# character offset per line start
offsets, tot = [], 0
for ln in lines:
    offsets.append(tot)
    tot += len(ln) + 1

for i, ln in enumerate(lines):
    if re.search(r"<figcaption><b>|<caption><b>", ln):
        continue
    for m in re.finditer(r"([图表])\s*(\d+)", ln):
        kind, n = m.group(1), int(m.group(2))
        known = authored_fig if kind == "图" else authored_tab
        flag = "OK " if n in known else "!! "
        # context: this line plus previous, trimmed
        prev = lines[i - 1] if i > 0 else ""
        ctx = (prev[-60:] + " ⟦" + ln[:200] + "⟧")
        ctx = re.sub(r"\s+", " ", ctx)
        print(f"{flag}L{i+1:<5} {kind}{n:<4} {ctx}")
