"""Map every rendered figure number to its plate, its generator and the
report_body.py line that places it, and check figure and table numbering and
every in-text reference in the built HTML and Markdown twins."""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_report as BR                      # noqa: E402
from report_body import body as report_body    # noqa: E402

# --- the source of truth: the body with image names still in place ----------
body = report_body({k: k for k in BR.IMG}, "")

blocks = re.findall(r"<figure>(.*?)</figure>", body, re.S)
rows = []
for i, b in enumerate(blocks, 1):
    img = re.search(r'<img src="(fig[^"]+)"', b)
    old = re.search(r"<figcaption><b>图\s*(\d+)", b)
    rows.append((i, old.group(1) if old else "-", img.group(1) if img else "-"))

# --- where each plate is generated and where it is placed in the prose ------
src = (ROOT / "scripts/premodel_05_figs.py").read_text(encoding="utf-8").splitlines()
gen, cur = {}, None
for n, line in enumerate(src, 1):
    m = re.match(r"def\s+(\w+)\(", line)
    if m:
        cur = (m.group(1), n)
    s = re.search(r'save\(fig,\s*"([^"]+)"', line)
    if s and cur:
        gen.setdefault(s.group(1), cur)
rb = (ROOT / "scripts/report_body.py").read_text(encoding="utf-8").splitlines()
place = {}
for n, line in enumerate(rb, 1):
    for name in re.findall(r"IMG\['([^']+)'\]", line):
        place.setdefault(name, n)

print("rendered  internal  plate                            generator          line")
for i, old, img in rows:
    g, gl = gen.get(img, ("-", 0))
    print(f"图 {i:<7} {old:<9} {img:<32} {g:<18} "
          f"premodel_05:{gl}  report_body:{place.get(img, 0)}")

# --- numbering and references in the two finished artefacts -----------------
for art in ("report.html", "report.md"):
    txt = (ROOT / art).read_text(encoding="utf-8")
    figs = [int(x) for x in re.findall(r"图\s*(\d+)", txt)]
    tabs = [int(x) for x in re.findall(r"表\s*(\d+)", txt)]
    fcap = [int(x) for x in re.findall(r"<figcaption><b>图\s*(\d+)", txt)] or \
        [int(x) for x in re.findall(r"\*\*图\s*(\d+)", txt)]
    tcap = [int(x) for x in re.findall(r"<caption><b>表\s*(\d+)", txt)] or \
        [int(x) for x in re.findall(r"\*\*表\s*(\d+)", txt)]
    nf, nt = len(fcap), len(tcap)
    bad = sorted({n for n in figs if n > nf} | {n for n in tabs if n > nt})
    print(f"\n{art}: figures {nf} sequential "
          f"{fcap == list(range(1, nf + 1))} | tables {nt} sequential "
          f"{tcap == list(range(1, nt + 1))} | out-of-range refs {len(bad)} {bad}")
    print(f"  图 references {len(figs)}  表 references {len(tabs)}")

for tag in ("@@", "{IMG[", "待补"):
    print(f"leftover marker {tag}: {(ROOT / 'report.html').read_text(encoding='utf-8').count(tag)}")
