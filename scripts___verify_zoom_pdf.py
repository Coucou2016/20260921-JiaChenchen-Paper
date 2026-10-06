"""Final check: the plate is in the PDF, and nothing downstream was touched."""
import io
import os
import sys
import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pypdf import PdfReader

r = PdfReader("report.pdf")
print("pdf pages:", len(r.pages))
for pno in range(15, 20):
    p = r.pages[pno - 1]
    t = p.extract_text() or ""
    print(f"  page {pno}: images={len(p.images)}  "
          f"caption_here={'平均绝对误差' in t}  window_420={'420' in t}")

print("\nnew plate png sizes")
for f in ("outputs/report_figs/fig64_domain_zoom.png",
          "figures/fig64_domain_zoom.png"):
    print(f"  {f:<48} {os.path.getsize(f) // 1024:>6} KB")

print("\npremodel checkpoints and results (must be untouched by this task)")
for f in sorted(os.listdir("outputs/premodel")):
    if f.endswith(".pt") or f == "premodel_results.json":
        p = os.path.join("outputs/premodel", f)
        ts = datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%m-%d %H:%M")
        print(f"  {ts}  {f}")

print("\nfinal artefacts")
for f in ("report.html", "report.md", "report.pdf"):
    print(f"  {f:<12} {os.path.getsize(f) / 1048576:>6.2f} MB")
