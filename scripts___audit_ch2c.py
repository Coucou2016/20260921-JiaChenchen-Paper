import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
s = open("scripts/report_body.py", encoding="utf-8").read()
i = s.find("<h3>2.4　细网格口径与粗网格口径")
j = s.find("<h3>2.5　哪些条件让误差变大")
sec = s[i:j]
prose = re.sub(r"<table.*?</table>", "", sec, flags=re.S)
prose = re.sub(r"<figure.*?</figure>", "", prose, flags=re.S)
prose = re.sub(r"<code>.*?</code>", "", prose, flags=re.S)
prose = re.sub(r"<[^>]+>", "", prose)
print("new section prose chars:", len(prose))
print("full-width colon/semicolon/dash:", prose.count("："), prose.count("；"), prose.count("—"))
sent = [x for x in re.split(r"[。！？]", prose) if x.strip()]
over = [re.sub(r"\s", "", x) for x in sent if len(re.sub(r"\s", "", x)) > 95]
print("sentences:", len(sent), "over95:", len(over))
for x in over:
    print("  >", x)
bad = ["不是.{0,12}而是", "并非.{0,12}而是", "与其说", "链路", "回路", "闭环", "耦合",
       "赋能", "抓手", "维度上", "值得注意的是", "综上所述", "原生"]
for b in bad:
    n = len(re.findall(b, sec))
    if n:
        print("BAD", b, n)
print("--- whole chapter 2 ---")
a = s.find("<h2>二")
b = s.find("<h2>三")
ch = s[a:b]
p2 = re.sub(r"<table.*?</table>", "", ch, flags=re.S)
p2 = re.sub(r"<figure.*?</figure>", "", p2, flags=re.S)
p2 = re.sub(r"<code>.*?</code>", "", p2, flags=re.S)
p2 = re.sub(r"<[^>]+>", "", p2)
s2 = [x for x in re.split(r"[。！？]", p2) if x.strip()]
o2 = [re.sub(r"\s", "", x) for x in s2 if len(re.sub(r"\s", "", x)) > 95]
print("ch2 sentences", len(s2), "over95", len(o2))
for x in o2:
    print("  >", x[:150])
print("ch2 colon/semicolon/dash:", p2.count("："), p2.count("；"), p2.count("—"))
for b in bad:
    n = len(re.findall(b, ch))
    if n:
        print("CH2 BAD", b, n)
print("ch2 section headings:")
for m in re.finditer(r"<h3>(2\.\d+　[^<]+)</h3>", ch):
    print("  ", m.group(1))
