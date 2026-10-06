"""Audit the report prose for the patterns the user asked to remove."""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
s = open("scripts/report_body.py", encoding="utf-8").read()

# prose only: strip the html tags and the f-string interpolations
prose = re.sub(r"\{IMG\[[^\]]+\]\}", " ", s)
prose = re.sub(r"<[^>]+>", " ", prose)

checks = {
    "不是...而是...(A-B-C 式)": r"不是[^。；]{0,40}而是",
    "并非...而是...": r"并非[^。；]{0,40}而是",
    "与其说...不如说...": r"与其说[^。；]{0,40}不如说",
    "全角冒号": r"：",
    "全角分号": r"；",
    "破折号": r"——",
    "不等于号 != ": r"≠",
    "约等于号": r"≈",
    "大于号": r">",
    "小于号": r"<",
    "生涩词:链路": r"链路",
    "生涩词:回路": r"回路",
    "生涩词:闭环": r"闭环",
    "生涩词:耦合": r"耦合",
    "生涩词:赋能": r"赋能",
    "生涩词:抓手": r"抓手",
    "生涩词:维度(抽象用法)": r"维度上",
    "首先其次最后": r"首先.{0,80}其次",
    "综上所述": r"综上所述",
    "值得注意的是": r"值得注意的是",
    "总而言之": r"总而言之",
}
print(f"=== prose length: {len(prose)} chars ===\n")
for name, pat in checks.items():
    hits = [m.start() for m in re.finditer(pat, prose)]
    if hits:
        print(f"{name}: {len(hits)}")
        for h in hits[:4]:
            frag = prose[max(0, h - 34):h + 34].replace("\n", " ")
            print("    ...", " ".join(frag.split()), "...")

# sentence length distribution
sents = [x for x in re.split(r"[。！？]", re.sub(r"\s+", "", prose)) if len(x) > 8]
long_s = [x for x in sents if len(x) > 95]
print(f"\n=== sentences: {len(sents)}, over 95 chars: {len(long_s)} ===")
for x in sorted(long_s, key=len, reverse=True)[:6]:
    print(f"  [{len(x)}] {x[:110]}...")

# bracket-annotation density, a proxy for "term explained on first use"
print(f"\n=== parenthetical notes: {len(re.findall(r'（[^）]{4,}）', prose))} ===")
