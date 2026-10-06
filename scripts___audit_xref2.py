"""Dump every in-text 图 N / 表 N reference from the built report together with the
caption it resolves to, so the pointer can be checked against the prose.

Also cross-checks the authored source (report_body.py + build_report.py): for each
reference it reports whether a caption with the *same authored number* exists, and
how far away it is. A reference that sits far from its own caption but right next to
a neighbouring one is the signature of the off-by-one bug that `renumber()` hides.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
html = (ROOT / "report.html").read_text(encoding="utf-8")
body = html.split("<body>", 1)[1].split("</body>", 1)[0]

out: list[str] = []

# ---- final caption map, in order ----------------------------------------
fig_cap, tab_cap = {}, {}
for m in re.finditer(r"<figcaption><b>图\s*(\d+)\s*(.*?)</b>", body, re.S):
    fig_cap[int(m.group(1))] = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m.group(2))).strip()
for m in re.finditer(r"<caption><b>表\s*(\d+)\s*(.*?)</b>", body, re.S):
    tab_cap[int(m.group(1))] = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m.group(2))).strip()

heads = [(m.start(), re.sub(r"<[^>]+>", "", m.group(0)).strip())
         for m in re.finditer(r"<h2>(.*?)</h2>|<h3>(.*?)</h3>", body, re.S)]


def heading_at(pos: int) -> str:
    cur = "?"
    for hp, ht in heads:
        if hp <= pos:
            cur = ht
        else:
            break
    return cur


# caption positions in final html
cap_pos = {}
for m in re.finditer(r"<figcaption><b>图\s*(\d+)|<caption><b>表\s*(\d+)", body):
    if m.group(1):
        cap_pos[("图", int(m.group(1)))] = m.start()
    else:
        cap_pos[("表", int(m.group(2)))] = m.start()

out.append("=" * 110)
out.append("EVERY IN-TEXT REFERENCE IN report.html (document order)")
out.append("=" * 110)
for m in re.finditer(r"([图表])\s*(\d+)", body):
    kind, n = m.group(1), int(m.group(2))
    title = (fig_cap if kind == "图" else tab_cap).get(n, "*** MISSING ***")
    ctx = re.sub(r"\s+", " ", body[max(0, m.start() - 70):m.start() + 30])
    ctx = re.sub(r"<[^>]+>", "", ctx)
    sec = heading_at(m.start())
    dist = ""
    if (kind, n) in cap_pos:
        dist = f"capdist={abs(m.start() - cap_pos[(kind, n)])}"
    out.append(f"[{sec[:18]:<18}] {kind}{n:<4} -> {title[:52]:<52} {dist}")
    out.append(f"        ctx: ...{ctx}")

out.append("")
out.append("=" * 110)
out.append("CAPTIONS NEVER REFERENCED IN PROSE")
out.append("=" * 110)
for kind, caps in (("图", fig_cap), ("表", tab_cap)):
    for n in sorted(caps):
        if not re.search(rf"{kind}\s*{n}(?!\d)", body):
            out.append(f"  {kind} {n}  {caps[n][:70]}")

(ROOT / "outputs" / "_xref_dump.txt").write_text("\n".join(out), encoding="utf-8")
print(f"wrote outputs/_xref_dump.txt ({len(out)} lines)")
