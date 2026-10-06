"""Verify the second-pass cleanup: residual wording, headers, numbering, screenshots."""
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

h = open("report.html", encoding="utf-8").read()
m = open("report.md", encoding="utf-8").read()

print("=== residual 原生 ===")
print("  report.html:", h.count("原生"), "| report.md:", m.count("原生"))

print("\n=== numbering still intact after re-render ===")
figs = re.findall(r"<figcaption><b>图\s*(\d+)", h)
tabs = re.findall(r"<caption><b>表\s*(\d+)", h)
print("  figs", len(figs), figs == [str(i) for i in range(1, len(figs) + 1)])
print("  tabs", len(tabs), tabs == [str(i) for i in range(1, len(tabs) + 1)])
print("  refs out of range:",
      sorted({int(x) for x in re.findall(r"图\s*(\d+)", h)} - {int(x) for x in figs}),
      sorted({int(x) for x in re.findall(r"表\s*(\d+)", h)} - {int(x) for x in tabs}))

print("\n=== table 5 header in html ===")
c = re.search(r"<caption><b>表 5.*?</table>", h, re.S)
if c:
    ths = re.findall(r"<th[^>]*>(.*?)</th>", c.group(0), re.S)
    for t in ths:
        print("   |", re.sub(r"<br\s*/?>", "/", re.sub(r"<[^>]+>", "", t)).strip())
    print("  含 '原生':", "原生" in c.group(0), "| 含 '模拟均':", "模拟均" in c.group(0))

print("\n=== table captions 1/2/6 opening ===")
for n in (1, 2, 6):
    mm = re.search(r"<caption><b>表 %d(.*?)</caption>" % n, h, re.S)
    if mm:
        print(f"  表{n}:", re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", mm.group(1)))[:110])

print("\n=== short-name mapping present? ===")
print("  2.1 mentions 模拟均流速:", h.count("模拟均流速"))
print("  table5 note mentions 模拟均流速:", bool(re.search(r"表 5.*?模拟均流速", h, re.S)))
print("  聚合二米真值 leftover:", h.count("聚合二米真值") + m.count("聚合二米真值"))
print("  升采样聚合真值 leftover:", h.count("升采样聚合真值"))

print("\n=== screenshots exist and are fresh ===")
d = "outputs/report_figs/checks"
if os.path.isdir(d):
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        print(f"  {f:<34} {os.path.getsize(p)//1024:>6} KB")
else:
    print("  MISSING", d)

print("\n=== pdf text scan ===")
try:
    from pypdf import PdfReader
    r = PdfReader("report.pdf")
    txt = "".join((pg.extract_text() or "") for pg in r.pages)
    print("  pages:", len(r.pages), "| 原生 in pdf text:", txt.count("原生"))
except Exception as ex:
    print("  pdf scan skipped:", type(ex).__name__, ex)

print("\n=== build artifacts ===")
for f in ("report.html", "report.md", "report.pdf"):
    print(f"  {f:<12} {os.path.getsize(f)/1048576:>6.2f} MB")
