"""Check figure embedding integrity and that the new figure 10 is present."""
import collections
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

h = open("report.html", encoding="utf-8").read()
print("figure tags:", len(re.findall(r"<figure", h)))

srcs = re.findall(r'<img[^>]*src="([^"]{0,40})', h)
kinds = collections.Counter(s.split(":")[0] if s.startswith("data") else s[:24] for s in srcs)
print("img src kinds:", dict(kinds), "| total img:", len(srcs))
print("non-data src:", [s for s in srcs if not s.startswith("data:")][:5])

figs = re.findall(r"<figure.*?</figure>", h, re.S)
print("figure blocks:", len(figs))
print("figures lacking an inline data: src:",
      [i + 1 for i, f in enumerate(figs) if 'src="data:' not in f])

m = re.search(r"<figcaption><b>图 10(.*?)</figcaption>", h, re.S)
print("\nfig 10 caption:", re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m.group(1)))[:230] if m else "MISSING")
print("fig 10 block has inline img:", 'src="data:' in figs[9] if len(figs) > 9 else None)

p = "outputs/report_figs/fig51_dialects.png"
print("\nfig51_dialects.png:", os.path.exists(p),
      f"{os.path.getsize(p) // 1024} KB" if os.path.exists(p) else "")

print("\n=== figures 1..36 first 40 chars of caption ===")
caps = re.findall(r"<figcaption><b>图\s*(\d+)(.*?)</figcaption>", h, re.S)
for n, c in caps:
    print(f"  {n:>3} {re.sub(chr(92)+'s+', ' ', re.sub(r'<[^>]+>', '', c)).strip()[:52]}")
