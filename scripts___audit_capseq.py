"""Reconcile caption order between the pre-rebuild report.md and the new report.html.

If the only change is that new captions were inserted, the two lists share a common
subsequence. Printing the diff shows exactly which captions are new.
"""
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

old = (ROOT / "report.md").read_text(encoding="utf-8")
new_html = (ROOT / "report.html").read_text(encoding="utf-8")
new_html = new_html.split("<body>", 1)[1].split("</body>", 1)[0]


def caps_old(t):
    out = []
    for m in re.finditer(r"\*\*图\s*(\d+)\s*([^*]{0,60})\*\*", t):
        out.append(("图", int(m.group(1)), m.group(2).strip()))
    for m in re.finditer(r"\*\*表\s*(\d+)\s*([^*]{0,60})\*\*", t):
        out.append(("表", int(m.group(1)), m.group(2).strip()))
    return out


def caps_new(t):
    out = []
    for m in re.finditer(r"<figcaption><b>图\s*(\d+)\s*([^<]{0,60})", t):
        out.append(("图", int(m.group(1)), m.group(2).strip()))
    for m in re.finditer(r"<caption><b>表\s*(\d+)\s*([^<]{0,60})", t):
        out.append(("表", int(m.group(1)), m.group(2).strip()))
    return out


o, n = caps_old(old), caps_new(new_html)
print(f"old captions: {len(o)}   new captions: {len(n)}")

o_keys = [f"{k}{i} {t}" for k, i, t in o]
n_keys = [f"{k}{i} {t}" for k, i, t in n]

print("\n=== unified diff of caption sequence (old vs new) ===")
for line in difflib.unified_diff(o_keys, n_keys, lineterm="", n=1):
    print(line)
