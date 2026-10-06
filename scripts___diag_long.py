import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
s = open("scripts/report_body.py", encoding="utf-8").read()
prose = re.sub(r"\{IMG\[[^\]]+\]\}", " ", s)
prose = re.sub(r"<[^>]+>", " ", prose)
sents = [x for x in re.split(r"[。！？]", re.sub(r"\s+", "", prose)) if len(x) > 8]
long_s = [x for x in sents if len(x) > 95]
print("total over 95:", len(long_s))
for x in long_s:
    print(f"[{len(x)}] {x[:200]}")
    print()
