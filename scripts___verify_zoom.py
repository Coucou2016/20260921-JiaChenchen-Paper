"""Verify the rebuilt report after adding the zoom plate."""
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
h = open("report.html", encoding="utf-8").read()
m = open("report.md", encoding="utf-8").read()

figs = re.findall(r"<figcaption><b>图\s*(\d+)", h)
tabs = re.findall(r"<caption><b>表\s*(\d+)", h)
print("figs", len(figs), "sequential", figs == [str(i) for i in range(1, len(figs) + 1)])
print("tabs", len(tabs), "sequential", tabs == [str(i) for i in range(1, len(tabs) + 1)])
print("fig refs outside range:",
      sorted({int(x) for x in re.findall(r"图\s*(\d+)", h)} - {int(x) for x in figs}))
print("tab refs outside range:",
      sorted({int(x) for x in re.findall(r"表\s*(\d+)", h)} - {int(x) for x in tabs}))
print("duplicate fig numbers:", [x for x in set(figs) if figs.count(x) > 1] or "none")

print("\n=== where did the zoom plate land ===")
caps = re.findall(r"<figcaption><b>图\s*(\d+)(.*?)</figcaption>", h, re.S)
for n, c in caps:
    t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", c)).strip()
    if "放大" in t or "全域水深" in t or "细网格口径" in t:
        print(f"  图 {n}: {t[:60]}")

print("\n=== image embedding ===")
srcs = re.findall(r'<img[^>]*src="([^"]{0,30})', h)
print("  img tags:", len(srcs), "| non-inline:", [s for s in srcs if not s.startswith("data:")][:3])
figblocks = re.findall(r"<figure.*?</figure>", h, re.S)
print("  figure blocks:", len(figblocks),
      "| without inline img:", [i + 1 for i, f in enumerate(figblocks) if 'src="data:' not in f])

print("\n=== residual artefacts ===")
print("  原生:", h.count("原生"), "| @@:", h.count("@@"), "| {IMG[:", h.count("{IMG["))
print("  http:// or https://:", len(re.findall(r"https?://", h)))

print("\n=== the new caption text ===")
mm = re.search(r"<figcaption><b>图 10(.*?)</figcaption>", h, re.S)
if mm:
    txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", mm.group(1))).strip()
    print("  ", txt[:400])
    print("  ...")
    print("  ", txt[-320:])
    print("\n  full-width colon/semicolon/dash:",
          txt.count("："), txt.count("；"), txt.count("——"))
    sents = [s for s in re.split(r"[。]", txt) if s.strip()]
    over = [s for s in sents if len(s) > 95]
    print(f"  sentences {len(sents)}, over 95 chars: {len(over)}")
    for s in over:
        print("    OVER:", s[:120])
    print("  不是…而是:", len(re.findall(r"不是[^。]{0,40}而是", txt)))

print("\n=== markdown twin carries the plate ===")
print("  fig64 referenced in md:", "fig64_domain_zoom.png" in m)
print("  figures/fig64_domain_zoom.png exists:",
      os.path.exists("figures/fig64_domain_zoom.png"))
print("\n=== sizes ===")
for f in ("report.html", "report.md", "report.pdf",
          "outputs/report_figs/fig64_domain_zoom.png"):
    print(f"  {f:<46} {os.path.getsize(f)/1048576:>6.3f} MB")
