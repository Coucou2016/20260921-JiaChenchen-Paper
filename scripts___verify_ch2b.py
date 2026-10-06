"""Confirm the worked-example numbers and locate residual '原生' wording."""
import io
import json
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

e = json.load(open("outputs/premodel/cell_example.json", encoding="utf-8"))
print("=== worked example source ===")
for k in ("scenario", "resolution_m", "row", "col", "row_span_2m", "col_span_2m",
          "sub_cells_per_side", "sub_min_m", "sub_max_m", "sub_mean_m",
          "area_weighted_aggregate_m", "coarse_native_m", "native_minus_aggregate_m",
          "domain_median_abs_diff_wet_m", "n_wet_cells", "selection_rule"):
    if k in e:
        v = e[k]
        if isinstance(v, float):
            v = f"{v:.4f}"
        print(f"  {k:<32} {v}")

h = open("report.html", encoding="utf-8").read()
print("\n=== does the report quote those numbers? ===")
for tok in ["0.860", "0.750", "0.110", "2.873", "0.003"]:
    print(f"  {tok:<7} in html: {h.count(tok)}")

print("\n=== every residual 原生 occurrence in html ===")
for m in re.finditer(r"原生", h):
    a, b = max(0, m.start() - 70), min(len(h), m.end() + 70)
    seg = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", h[a:b]))
    print("  -", seg.strip()[:132])

print("\n=== is 原生 defined anywhere (terminology table / 速查)? ===")
a, b = h.find("<h2>二"), h.find("<h2>三")
ch = re.sub(r"<[^>]+>", "", h[a:b])
print("  2.9 速查表 contains 原生:", "原生" in ch[ch.find("本章术语速查"):])
print("  glossary appendix mentions 原生:", "原生" in h[h.find("附录一"):])
