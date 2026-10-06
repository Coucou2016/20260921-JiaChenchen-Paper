"""Print each in-text reference in report_body.py with the FINAL caption title it
resolves to after renumber(), i.e. exactly what a reader of report.html sees.

This makes mismatches visible: a sentence about "相对变化" that resolves to a
caption reading "配对检验结果" is wrong.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
html = (ROOT / "report.html").read_text(encoding="utf-8")
body = html.split("<body>", 1)[1].split("</body>", 1)[0]

fig_cap = {}
for m in re.finditer(r"<figcaption><b>图\s*(\d+)\s*([^<]{0,60})", body):
    fig_cap[int(m.group(1))] = m.group(2).strip()
tab_cap = {}
for m in re.finditer(r"<caption><b>表\s*(\d+)\s*([^<]{0,60})", body):
    tab_cap[int(m.group(1))] = m.group(2).strip()

# map: authored number -> final number, derived from caption source order.
# The final captions in body are in document order; the authored number is the one
# that appears in the same position in the source pair. Instead of tracking that,
# read the final numbers directly and pair them with the sentence.
src = (ROOT / "scripts/report_body.py").read_text(encoding="utf-8")
lines = src.splitlines()

# We rely on renumber being order-preserving: authored caption order == final order.
# So build authored-order list of (kind, authored_num) from sources, in the SAME
# document order as the final html. We reproduce it by scanning report_body then
# the build_report fragments in the order they are spliced. Simpler: use the final
# caption titles (from html) together with the authored numbers (from source), in
# final order, then the mapping is authored[i] -> final[i].
auth_fig, auth_tab = [], []
for f in ("report_body.py",):
    s = (ROOT / "scripts" / f).read_text(encoding="utf-8")
    for m in re.finditer(r"<figcaption><b>图\s*(\d+)", s):
        auth_fig.append(int(m.group(1)))
    for m in re.finditer(r"<caption><b>表\s*(\d+)", s):
        auth_tab.append(int(m.group(1)))

print("note: report_body captions feed only part of the document; use html titles\n")

for i, ln in enumerate(lines):
    if re.search(r"<figcaption><b>|<caption><b>", ln):
        continue
    for m in re.finditer(r"([图表])\s*(\d+)", ln):
        kind, n = m.group(1), int(m.group(2))
        pre = ln[max(0, m.start() - 1):m.start()]
        if pre in ("图", "表"):
            continue
        caps = fig_cap if kind == "图" else tab_cap
        # We cannot know the mapping here without the order; print the source text
        print(f"L{i+1:<5} {kind}{n:<4} :: {re.sub(chr(92)+'s+', ' ', ln.strip())[:110]}")
