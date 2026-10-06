import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
s = open("report.html", encoding="utf-8").read()
print("size %.2f MB" % (len(s) / 1e6))
figs = re.findall(r"<figcaption><b>图 (\d+)", s)
tabs = re.findall(r"<caption><b>表 (\d+)", s)
print("figure captions:", figs)
print("table captions :", tabs)
print("sequential figs:", figs == [str(i) for i in range(1, len(figs) + 1)])
print("sequential tabs:", tabs == [str(i) for i in range(1, len(tabs) + 1)])
print("base64 images  :", s.count("data:image/png;base64,"))
ext = [u for u in re.findall(r'src="([^"]+)"', s) if not u.startswith("data:")]
print("non-base64 src :", ext)
print("external href  :", [u for u in re.findall(r'href="([^"]+)"', s) if not u.startswith("#")])
print("unresolved markers:", re.findall(r"@@\w+@@|\{IMG\[|\\x00", s))
# any figure reference pointing at a number that has no caption?
refs = sorted({int(x) for x in re.findall(r"图\s*(\d+)", s)})
print("referenced figure numbers:", refs)
missing = [r for r in refs if str(r) not in figs]
print("references without a caption:", missing)
trefs = sorted({int(x) for x in re.findall(r"表\s*(\d+)", s)})
print("referenced table numbers:", trefs)
print("table refs without a caption:", [r for r in trefs if str(r) not in tabs])
