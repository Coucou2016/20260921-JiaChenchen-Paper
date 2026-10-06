"""List the authored caption numbers in both report sources, in document order."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

for name in ("report_body.py", "build_report.py"):
    src = (ROOT / "scripts" / name).read_text(encoding="utf-8")
    print(f"===== {name} =====")
    hits = []
    for m in re.finditer(r"<figcaption><b>图\s*(\d+)(.{0,44})", src):
        hits.append((m.start(), "图", m.group(1), m.group(2)))
    for m in re.finditer(r"<caption><b>表\s*(\d+)(.{0,44})", src):
        hits.append((m.start(), "表", m.group(1), m.group(2)))
    for _, kind, num, txt in sorted(hits):
        print(f"  {kind} {num:>4}  {txt.strip()}")
    print()

# the per-pair table ids generated programmatically
src = (ROOT / "scripts/build_report.py").read_text(encoding="utf-8")
print("===== programmatic captions =====")
for m in re.finditer(r'capid', src):
    line = src[:m.start()].count("\n") + 1
    ctx = src.splitlines()[line - 1].strip()
    print(f"  L{line}: {ctx}")
