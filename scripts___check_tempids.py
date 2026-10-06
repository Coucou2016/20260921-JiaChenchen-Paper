"""Check that no temporary figure/table number is used twice anywhere in scripts/."""
import collections
import glob
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
tabs = collections.Counter()
figs = collections.Counter()
where = {"表": {}, "图": {}}
for f in glob.glob("scripts/*.py"):
    if f.split("\\")[-1].split("/")[-1].startswith("_"):
        continue  # checker scripts embed caption patterns as regex, not as content
    s = open(f, encoding="utf-8").read()
    for m in re.finditer(r"caption><b>表\s*(\d+)", s):
        tabs[m.group(1)] += 1
        where["表"].setdefault(m.group(1), []).append(f)
    for m in re.finditer(r"figcaption><b>图\s*(\d+)", s):
        figs[m.group(1)] += 1
        where["图"].setdefault(m.group(1), []).append(f)
print("temp TABLE numbers:", sorted(int(k) for k in tabs))
print("  duplicates:", {k: v for k, v in tabs.items() if v > 1})
print("temp FIGURE numbers:", sorted(int(k) for k in figs))
print("  duplicates:", {k: v for k, v in figs.items() if v > 1})
seq_t = set(range(1, 23))
seq_f = set(range(1, 37))
bad_t = [int(k) for k in tabs if int(k) in seq_t and tabs[k] and k not in
         [str(i) for i in range(1, 11)]]
print("tables using a number that is also a final sequential value (1..22):",
      sorted(int(k) for k in tabs if int(k) <= 22))
print("figures using a number that is also a final sequential value (1..36):",
      sorted(int(k) for k in figs if int(k) <= 36))
