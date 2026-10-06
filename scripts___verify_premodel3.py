"""Verify the bias-sign-flip disclosure landed and nothing else regressed."""
import io
import json
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
h = open("report.html", encoding="utf-8").read()

print("=== disclosure present in built html ===")
for kw in ["符号全部翻转", "方向相反的体积失衡", "相互放大", "未变差瓦片占比",
           "没有一块瓦片变差"]:
    print(f"  {kw:<16} {h.count(kw)}")

print("\n=== quoted volume numbers vs source json ===")
r = json.load(open("outputs/premodel/premodel_results.json", encoding="utf-8"))
q = re.findall(r"正百分之[一二三四五六七八九十零点]+|负百分之[一二三四五六七八九十零点]+", h)
print("  prose mentions quantified percents:", len(q))

print("\n=== numbering and containment ===")
figs = re.findall(r"<figcaption><b>图\s*(\d+)", h)
tabs = re.findall(r"<caption><b>表\s*(\d+)", h)
print("  figs", len(figs), "seq", figs == [str(i) for i in range(1, len(figs)+1)])
print("  tabs", len(tabs), "seq", tabs == [str(i) for i in range(1, len(tabs)+1)])
print("  fig refs out of range:",
      sorted({int(x) for x in re.findall(r"图\s*(\d+)", h)} - {int(x) for x in figs}))
print("  tab refs out of range:",
      sorted({int(x) for x in re.findall(r"表\s*(\d+)", h)} - {int(x) for x in tabs}))
print("  non-inline src/href:",
      [x for x in re.findall(r'(?:src|href)="([^"]+)"', h) if not x.startswith("data:")] or "none")
print("  base64 imgs:", h.count("data:image/png;base64,"))
print("  http:", len(re.findall(r"https?://", h)), "| 待补充:", h.count("待补充"))
print("  leftovers @@/{IMG[:", h.count("@@"), h.count("{IMG["))

print("\n=== no-retrain check: premodel + SR model artefact mtimes ===")
import os
import time
for pat in ["outputs/premodel/ckpt_10m.pt", "outputs/v0_10m2m_hmax/last.pt",
            "outputs/finetune_deep/w01/snapshots/ep0187.pt"]:
    if os.path.exists(pat):
        print(f"  {pat:<52} {time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(pat)))}")
print("  report.html mtime:",
      time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime("report.html"))))
