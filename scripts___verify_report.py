"""Verify the built report.html before regenerating md/pdf."""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
h = open("report.html", encoding="utf-8").read()

print(f"size: {len(h)/1e6:.2f} MB, chars {len(h)}")

# self-containment
bad = [x for x in re.findall(r'(?:src|href)="([^"]+)"', h) if not x.startswith("data:")]
print("non-inline src/href:", sorted(set(bad))[:10] or "none")
print("base64 images:", h.count("data:image/png;base64,"))
print("http refs:", len(re.findall(r"https?://", h)))

# leftovers
for p in ["@@", "{IMG[", "待补充"]:
    print(f"leftover {p!r}:", h.count(p))

# numbering
figs = re.findall(r"<figcaption><b>图\s*(\d+)", h)
tabs = re.findall(r"<caption><b>表\s*(\d+)", h)
print("figure tags:", figs)
print("table tags:", tabs)
print("fig sequential:", figs == [str(i) for i in range(1, len(figs) + 1)])
print("tab sequential:", tabs == [str(i) for i in range(1, len(tabs) + 1)])

refs_f = {int(x) for x in re.findall(r"图\s*(\d+)", h)}
refs_t = {int(x) for x in re.findall(r"表\s*(\d+)", h)}
print("fig refs outside range:", sorted(refs_f - {int(x) for x in figs}))
print("tab refs outside range:", sorted(refs_t - {int(x) for x in tabs}))
missing_f = sorted({int(x) for x in figs} - refs_f)
print("figures never cited in text:", missing_f)

# subplot labels present in captions
caps = re.findall(r"<figcaption>(.*?)</figcaption>", h, re.S)
print(f"\ncaptions: {len(caps)}, avg chars {sum(len(re.sub(chr(60)+'[^'+chr(62)+']*'+chr(62),'',c)) for c in caps)//max(1,len(caps))}")
short = [c[:40] for c in caps if len(c) < 260]
print("captions under 260 chars:", short)
