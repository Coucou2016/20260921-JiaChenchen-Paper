"""Print exact source lines around each suspect, UTF-8 clean."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "scripts/report_body.py").read_text(encoding="utf-8")
lines = src.splitlines()

WINDOWS = [(2715, 2800), (2520, 2560), (3070, 3090), (1705, 1715),
           (2330, 2340), (2880, 2900), (3185, 3200)]

out = []
for a, b in WINDOWS:
    out.append(f"################ lines {a}-{b} ################")
    for i in range(a - 1, min(b, len(lines))):
        out.append(f"{i+1:>5}| {lines[i]}")
    out.append("")

(ROOT / "outputs" / "_suspect_src.txt").write_text("\n".join(out), encoding="utf-8")
print("written")
