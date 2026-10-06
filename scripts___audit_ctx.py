"""Print the exact prose around each suspect reference in the built report.html."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
body = (ROOT / "report.html").read_text(encoding="utf-8")
body = body.split("<body>", 1)[1].split("</body>", 1)[0]
body = re.sub(r"<img[^>]*>", "<img>", body)

PATTERNS = [
    "给出相对变化",
    "给出了完整结果",
    "把两条技术路线放在一起",
    "附录一的表",
    "不替代附录一",
    "最重要的信息是一致性",
    "同一组数据，形式不同",
    "本报告最有实际参考价值",
]

for kw in PATTERNS:
    for m in re.finditer(re.escape(kw), body):
        s = max(0, m.start() - 160)
        e = min(len(body), m.end() + 160)
        frag = re.sub(r"<[^>]+>", " ", body[s:e])
        frag = re.sub(r"\s+", " ", frag)
        print(f"### {kw}")
        print("   ", frag)
        # also show the next caption after this point
        nxt = re.search(r"<(?:caption|figcaption)><b>(表|图)\s*(\d+)\s*([^<]{0,50})",
                        body[m.start():m.start() + 3000])
        if nxt:
            print(f"    -> next caption: {nxt.group(1)} {nxt.group(2)} {nxt.group(3).strip()}")
        print()
