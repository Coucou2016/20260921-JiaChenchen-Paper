"""Independent verification of the rewritten chapter 2 and the numbering fix."""
import io
import json
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
h = open("report.html", encoding="utf-8").read()

print("=== numbering (the reported collision) ===")
figs = re.findall(r"<figcaption><b>图\s*(\d+)", h)
tabs = re.findall(r"<caption><b>表\s*(\d+)", h)
print("  figs", len(figs), "seq", figs == [str(i) for i in range(1, len(figs)+1)], figs[:26])
print("  tabs", len(tabs), "seq", tabs == [str(i) for i in range(1, len(tabs)+1)], tabs)
print("  dup fig numbers:", [x for x in set(figs) if figs.count(x) > 1] or "none")
print("  dup tab numbers:", [x for x in set(tabs) if tabs.count(x) > 1] or "none")
print("  fig refs out of range:",
      sorted({int(x) for x in re.findall(r"图\s*(\d+)", h)} - {int(x) for x in figs}))
print("  tab refs out of range:",
      sorted({int(x) for x in re.findall(r"表\s*(\d+)", h)} - {int(x) for x in tabs}))

print("\n=== chapter 2 section headings and order ===")
a, b = h.find("<h2>二"), h.find("<h2>三")
for m in re.finditer(r"<h3>(.*?)</h3>", h[a:b], re.S):
    print("  -", re.sub(r"<[^>]+>", "", m.group(1)).strip()[:40])

print("\n=== is 聚合 explained in 2.1, and how often later? ===")
ch = re.sub(r"<[^>]+>", "", h[a:b])
ch = " ".join(ch.split())
print("  first index of 聚合:", ch.find("聚合"), "of 粗网格模拟值:", ch.find("粗网格模拟值"))
print("  聚合 occurrences:", ch.count("聚合"), "| 原生:", ch.count("原生"),
      "| 原生减聚合:", ch.count("原生减聚合"))
print("  dangling '原生' contexts:", [s for s in re.findall(r".{18}原生.{18}", ch)][:6])

print("\n=== verify the worked example against its source json ===")
p = "outputs/premodel/cell_example.json"
if os.path.exists(p):
    e = json.load(open(p, encoding="utf-8"))
    print("  keys:", sorted(e.keys())[:20])
    for k in ("iy", "ix", "aggregate", "native", "simulated", "diff", "scenario"):
        if k in e:
            print(f"    {k} = {e[k]}")
else:
    print("  MISSING", p)

print("\n=== verify table 3 (error-source) numbers vs source ===")
for cand in ["outputs/premodel/error_sources.json",
             "outputs/premodel/factor_table.json",
             "outputs/premodel/tile_metrics.json"]:
    if os.path.exists(cand):
        d = json.load(open(cand, encoding="utf-8"))
        print(f"  {cand}: type={type(d).__name__}")
        if isinstance(d, dict):
            print("    keys:", list(d.keys())[:14])
        elif isinstance(d, list):
            print("    n =", len(d), "first:", str(d[0])[:200])

print("\n=== containment ===")
print("  non-inline src/href:",
      [x for x in re.findall(r'(?:src|href)="([^"]+)"', h) if not x.startswith("data:")] or "none")
print("  base64 imgs:", h.count("data:image/png;base64,"))
print("  http:", len(re.findall(r"https?://", h)), "| 待补充:", h.count("待补充"))
print("  leftovers @@/{IMG[:", h.count("@@"), h.count("{IMG["))

print("\n=== chapter 2 prose discipline ===")
print("  全角冒号:", ch.count("："), "| 分号:", ch.count("；"), "| 破折号:", ch.count("——"))
for pat, lbl in ((r"不是[^。\n]{0,40}而是", "不是…而是"),
                 (r"链路|回路|闭环|耦合|赋能|抓手", "生涩词"),
                 (r"维度上|值得注意的是", "抽象套语")):
    print(f"  {lbl}:", len(re.findall(pat, ch)))
