"""Check whether the volume-relative-error regression is disclosed, and whether any
table quotes the frac_improved field, whose sign convention only fits lower-is-better metrics."""
import io
import json
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
r = json.load(open("outputs/premodel/premodel_results.json", encoding="utf-8"))

print("=== volume relative error and bias: identity vs premodel ===")
print(f"{'res':<5}{'identity vol':>14}{'premodel vol':>14}{'identity bias':>15}{'premodel bias':>15}")
for res in ["5m", "10m", "20m", "30m"]:
    g = r[res]["global"]
    print(f"{res:<5}{g['identity']['volume_rel']:>+14.3f}{g['premodel']['volume_rel']:>+14.3f}"
          f"{g['identity']['bias']:>+15.4f}{g['premodel']['bias']:>+15.4f}")

print("\n=== is 'frac_improved' used anywhere in the report or builders? ===")
for p in ["scripts/report_body.py", "scripts/build_report.py"]:
    s = open(p, encoding="utf-8").read()
    hits = ["frac_improved" in s, "改善比例" in s, "瓦片改善" in s]
    print(f"  {p}: frac_improved={hits[0]} 改善比例={hits[1]} 瓦片改善={hits[2]}")

print("\n=== how does the prose describe the volume error / bias direction? ===")
body = open("scripts/report_body.py", encoding="utf-8").read()
for kw in ["体积", "总体积", "水量"]:
    for m in re.finditer(rf"[^。\n]{{0,90}}{kw}[^。\n]{{0,90}}", body):
        seg = " ".join(m.group(0).split())
        if "前置" in seg or "粗网格" in seg or "校正" in seg:
            print("  -", seg[:150])

print("\n=== 10 m tile-level distribution of the premodel MAE delta ===")
p = r["10m"]["paired"]["premodel_vs_identity"]["mae"]
print(f"  mean {p['mean_delta']:+.4f}  median {p['median_delta']:+.4f}  "
      f"frac_improved {p['frac_improved']:.3f}")
print(f"  => {1 - p['frac_improved']:.1%} of tiles did not improve")
