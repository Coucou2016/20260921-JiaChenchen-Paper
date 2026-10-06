"""Independent verification of the inserted pre-model chapter and its numbers."""
import io
import json
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
h = open("report.html", encoding="utf-8").read()

print("=== chapter order ===")
for m in re.finditer(r"<h2>(.*?)</h2>", h, re.S):
    print("  -", re.sub(r"<[^>]+>", "", m.group(1)).strip()[:46])

figs = re.findall(r"<figcaption><b>图\s*(\d+)", h)
tabs = re.findall(r"<caption><b>表\s*(\d+)", h)
print(f"\nfigs {len(figs)} sequential={figs == [str(i) for i in range(1, len(figs)+1)]}")
print(f"tabs {len(tabs)} sequential={tabs == [str(i) for i in range(1, len(tabs)+1)]}")
rf = {int(x) for x in re.findall(r"图\s*(\d+)", h)} - {int(x) for x in figs}
rt = {int(x) for x in re.findall(r"表\s*(\d+)", h)} - {int(x) for x in tabs}
print("fig refs out of range:", sorted(rf), "| tab refs out of range:", sorted(rt))
print("uncited figures:", sorted({int(x) for x in figs} - {int(x) for x in re.findall(r'图\s*(\d+)', h)}))

print("\n=== self-containment ===")
ext = [x for x in re.findall(r'(?:src|href)="([^"]+)"', h) if not x.startswith("data:")]
print("non-inline src/href:", ext or "none")
print("base64 images:", h.count("data:image/png;base64,"))
print("http occurrences:", len(re.findall(r"https?://", h)), "(inline SVG xmlns only)")
print("leftovers @@ / {IMG[:", h.count("@@"), h.count("{IMG["))
print("待补充 count:", h.count("待补充"))

print("\n=== paired-test convention check (frac_improved) ===")
r = json.load(open("outputs/premodel/premodel_results.json", encoding="utf-8"))
for res in ["5m", "10m", "20m", "30m"]:
    if res not in r:
        continue
    for metric in ["mae", "csi005"]:
        d = r[res]["paired"]["premodel_vs_identity"][metric]
        print(f"  {res} {metric:7s} mean_delta={d['mean_delta']:+.4f} "
              f"frac_improved={d['frac_improved']:.3f}")

print("\n=== 10 m headline numbers ===")
g = r["10m"]["global"]
for arm in ["identity", "bilinear", "premodel"]:
    a = g[arm]
    print(f"  {arm:9s} MAE {a['mae']:.4f}  RMSE {a['rmse']:.4f}  "
          f"CSI@0.05 {a['csi@0.05']:.4f}  bias {a['bias']:+.4f}  vol {a['volume_rel']:+.3f}")
p = r["10m"]["paired"]["premodel_vs_identity"]["mae"]
print(f"  paired MAE: delta {p['mean_delta']:+.4f}  CI [{p['boot_ci_lo']:+.4f},"
      f" {p['boot_ci_hi']:+.4f}]  t {p['t']:+.2f}  dz {p['cohen_dz']:+.2f}")

print("\n=== does the report quote frac_improved for a CSI metric? ===")
body = open("scripts/report_body.py", encoding="utf-8").read()
tot = 0
for m in re.finditer(r"[^。\n]{0,80}%[^。\n]{0,60}(瓦片|像元)[^。\n]{0,30}", body):
    seg = " ".join(m.group(0).split())
    if "改善" in seg or "变差" in seg:
        print("  ", seg[:120])
        tot += 1
print("  (matches:", tot, ")")
