"""List every occurrence of the flagged patterns with enough context to rewrite."""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
s = open("scripts/report_body.py", encoding="utf-8").read()


def show(label, pat, pad=30):
    print(f"\n########## {label} ##########")
    for i, m in enumerate(re.finditer(pat, s), 1):
        a, b = max(0, m.start() - pad), min(len(s), m.end() + pad)
        frag = " ".join(s[a:b].split())
        print(f"{i:>2}. {frag}")


show("不是...而是", r"不是[^。；\n]{0,45}?而是")
show("并非...而是", r"并非[^。；\n]{0,45}?而是")
show("全角冒号", r"：", 26)
show("全角分号", r"；", 26)
show("维度上", r"维度上", 40)
show("值得注意的是", r"值得注意的是", 30)
show("引号短语", r"[“\"][^”\"]{2,14}[”\"]", 26)
show("破折号", r"——", 30)
