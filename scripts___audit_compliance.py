"""Compliance counts for the finished artefacts: forbidden punctuation, banned
vocabulary, the 不是..而是 pairing, and the caption tail check for the new
figures.  Run after every rebuild."""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

BANNED = ["链路", "赋能", "抓手", "闭环", "维度上", "值得注意的是", "原生"]
PUNCT = [("全角冒号", "："), ("全角分号", "；"), ("破折号", "——")]

for art in ("report.html", "report.md"):
    raw = (ROOT / art).read_text(encoding="utf-8")
    flat = re.sub(r"\s+", "", raw)
    print(f"=== {art} ===")
    for name, ch in PUNCT:
        print(f"  {name}: {raw.count(ch)}")
    for w in BANNED:
        print(f"  {w}: {raw.count(w)}")
    m = re.findall(r"不是[^。；]{0,40}而是", flat)
    print(f"  不是..而是: {len(m)} {m[:3]}")
    print(f"  并非..而是: {len(re.findall(r'并非[^。；]{0,40}而是', flat))}")

# caption tails of the new section: every new caption must end with a number
h = (ROOT / "report.html").read_text(encoding="utf-8")
caps = re.findall(r"<figcaption>(.*?)</figcaption>", h, re.S)
print("\n=== last sentence of every figure caption, with a number present ===")
bad = 0
for c in caps:
    txt = re.sub(r"<[^>]+>", "", c)
    txt = re.sub(r"\s+", "", txt)
    last = [s for s in re.split(r"[。！？]", txt) if len(s) > 4]
    if not last:
        continue
    tail = last[-1]
    has = bool(re.search(r"\d", tail))
    if not has:
        bad += 1
        print(f"  NO NUMBER: {tail[:70]}")
print(f"  captions {len(caps)}, without a number in the final sentence: {bad}")
