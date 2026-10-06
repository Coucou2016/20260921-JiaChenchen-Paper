"""Audit figure/table cross-references in the rebuilt report.

Checks:
  1. every in-text reference 图 N / 表 N points at an existing caption;
  2. every caption is referenced at least once in the prose (orphan check);
  3. prints the caption order with the nearest preceding heading, so mis-ordered
     references can be spotted by eye.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
html = (ROOT / "report.html").read_text(encoding="utf-8")
body = html.split("<body>", 1)[1].split("</body>", 1)[0]

# ---- caption inventory, in document order -------------------------------
caps: list[tuple[int, str, str]] = []   # (number, kind, title)
for m in re.finditer(r"<figcaption><b>图\s*(\d+)\s*(.*?)</b>", body, re.S):
    caps.append((int(m.group(1)), "图", m.group(2)))
for m in re.finditer(r"<caption><b>表\s*(\d+)\s*(.*?)</b>", body, re.S):
    caps.append((int(m.group(1)), "表", m.group(2)))

figs = [c for c in caps if c[1] == "图"]
tabs = [c for c in caps if c[1] == "表"]

# heading positions, to report the containing section of each caption
heads = [(m.start(), re.sub(r"<[^>]+>", "", m.group(1)))
         for m in re.finditer(r"<h[234]>(.*?)</h[234]>", body, re.S)]
cap_pos = {}
for m in re.finditer(r"<figcaption><b>图\s*(\d+)|<caption><b>表\s*(\d+)", body):
    num = int(m.group(1) or m.group(2))
    kind = "图" if m.group(1) else "表"
    cap_pos[(kind, num)] = m.start()


def section_of(pos: int) -> str:
    cur = "?"
    for hp, ht in heads:
        if hp < pos:
            cur = ht
        else:
            break
    return cur


print(f"captions: {len(figs)} figures, {len(tabs)} tables")
print("\n--- figure captions ---")
for n, _, t in sorted(figs):
    print(f"  图 {n:>3}  [{section_of(cap_pos[('图', n)]):<28}] {t[:60]}")
print("\n--- table captions ---")
for n, _, t in sorted(tabs):
    print(f"  表 {n:>3}  [{section_of(cap_pos[('表', n)]):<28}] {t[:60]}")

# ---- reference inventory -------------------------------------------------
max_fig, max_tab = max(n for n, _, _ in figs), max(n for n, _, _ in tabs)
missing: list[str] = []
for m in re.finditer(r"图\s*(\d+)", body):
    n = int(m.group(1))
    if ("图", n) not in cap_pos:
        ctx = body[max(0, m.start() - 40):m.start() + 40].replace("\n", " ")
        missing.append(f"图 {n}  ...{ctx}...")
for m in re.finditer(r"表\s*(\d+)", body):
    n = int(m.group(1))
    if ("表", n) not in cap_pos:
        ctx = body[max(0, m.start() - 40):m.start() + 40].replace("\n", " ")
        missing.append(f"表 {n}  ...{ctx}...")

print("\n--- dangling references ---")
if missing:
    for s in missing:
        print("  " + s)
else:
    print("  none")

# ---- novelty-window heuristic -------------------------------------------
# A reference is suspicious when it sits far away from its caption, because
# tables were renumbered in document order while prose kept the old numbers.
sec_starts = {}
for hp, ht in heads:
    sec_starts.setdefault(ht, hp)
orphan = []
for (kind, n) in sorted(cap_pos):
    # find refs
    refs = [m.start() for m in re.finditer(rf"{kind}\s*{n}(?!\d)", body)]
    cap = cap_pos[(kind, n)]
    dist = min((abs(r - cap) for r in refs), default=None)
    if dist is None:
        orphan.append(f"{kind} {n} (never referenced)  [{section_of(cap)}]")
    elif dist > 60000:
        orphan.append(f"{kind} {n} (nearest ref {dist} chars away)  [{section_of(cap)}]")

print("\n--- unreferenced / far-away captions ---")
for s in orphan:
    print("  " + s)
