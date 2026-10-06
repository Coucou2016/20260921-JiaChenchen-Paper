"""Source-level semantic cross-reference audit.

`renumber()` in build_report.py rewrites every 图 N / 表 N (captions and in-text
references alike) to sequential document order. A reference therefore always
resolves, even when the author wrote the wrong number. The only way to catch that
is to check, in the *source*, whether the number a reference quotes actually
belongs to the figure/table the sentence is talking about.

Heuristic used here:
  * a reference is LOCAL when another caption sits within NEAR characters of it;
    a local reference must quote that caption's number;
  * a reference whose number matches no caption at all is definitely broken;
  * a reference whose number matches a caption in the same section is accepted;
  * anything else is printed for human review.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
NEAR = 500          # chars: within this radius a caption counts as "local"

files = ["report_body.py", "build_report.py"]
caps: list[tuple[int, str, int, str, str]] = []   # pos, kind, number, title, file
refs: list[tuple[int, str, int, str, str]] = []   # pos, kind, number, line, file

for fname in files:
    src = (ROOT / "scripts" / fname).read_text(encoding="utf-8")
    for m in re.finditer(r"<figcaption><b>图\s*(\d+)\s*([^<]{0,60})", src):
        caps.append((m.start(), "图", int(m.group(1)), m.group(2).strip(), fname))
    for m in re.finditer(r"<caption><b>表\s*(\d+)\s*([^<]{0,60})", src):
        caps.append((m.start(), "表", int(m.group(1)), m.group(2).strip(), fname))
    # programmatic caption ids
    for m in re.finditer(r'capid in \(\([^)]*\)\)', src):
        for mm in re.finditer(r'"([^"]+)",\s*"[^"]+",\s*"(\d+)"', m.group(0)):
            caps.append((m.start(), "表", int(mm.group(2)),
                         f"{mm.group(1)} 各分辨率机会校正一致性指标", fname))

# in-text references: skip captions themselves and skip the renumber() definition
for fname in ("report_body.py",):
    src = (ROOT / "scripts" / fname).read_text(encoding="utf-8")
    for m in re.finditer(r"([图表])\s*(\d+)", src):
        pre = src[max(0, m.start() - 60):m.start()]
        if re.search(r"<figcaption><b>$|<caption><b>$", pre):
            continue
        line = src[:m.start()].count("\n") + 1
        refs.append((m.start(), m.group(1), int(m.group(2)), line, fname))

# sections of report_body.py
rb = (ROOT / "scripts" / "report_body.py").read_text(encoding="utf-8")
heads = [(m.start(), re.sub(r"<[^>]+>", "", m.group(0)).strip())
         for m in re.finditer(r"<h2>(.*?)</h2>|<h3>(.*?)</h3>", rb, re.S)]


def section_of(pos: int) -> str:
    cur = "?"
    for hp, ht in heads:
        if hp <= pos:
            cur = ht
        else:
            break
    return cur


caps_by_num: dict[tuple[str, int], list] = {}
for c in caps:
    caps_by_num.setdefault((c[1], c[2]), []).append(c)

print(f"{len(refs)} in-text references, {len(caps)} authored captions\n")
print("=" * 110)
suspects = []
for pos, kind, num, line, fname in refs:
    sec = section_of(pos)
    key = (kind, num)
    if key not in caps_by_num:
        suspects.append(f"L{line:<5} {kind}{num:<4} NO SUCH CAPTION           [{sec}]")
        continue
    owner = caps_by_num[key][0]
    # nearest caption overall
    near = min(caps, key=lambda c: abs(c[0] - pos))
    near_dist = abs(near[0] - pos)
    # same-section match?
    same_sec = any(section_of(c[0]) == sec for c in caps_by_num[key])
    if near_dist <= NEAR and near[2] != num:
        suspects.append(
            f"L{line:<5} {kind}{num:<4} LOCAL MISMATCH: nearest is "
            f"{near[1]}{near[2]} '{near[3][:34]}' ({near_dist}ch)  [{sec}]")

print("--- suspects (need review) ---")
for s in suspects:
    print("  " + s)
if not suspects:
    print("  none")

print()
print("=" * 110)
print("--- every reference with its resolved caption ---")
for pos, kind, num, line, fname in refs:
    sec = section_of(pos)
    if (kind, num) in caps_by_num:
        owner = caps_by_num[(kind, num)][0]
        dist = abs(owner[0] - pos)
        flag = "  " if dist <= 3000 else "~ "
        print(f"{flag}L{line:<5} {kind}{num:<4} -> {kind}{num} '{owner[3][:40]}'  "
              f"dist={dist:<7} [{sec[:22]}]")
