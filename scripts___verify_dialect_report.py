"""Check the rebuilt report: numbering, preservation, provenance, timestamps."""
import io
import json
import os
import re
import subprocess
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

h = open("report.html", encoding="utf-8").read()
m = open("report.md", encoding="utf-8").read()

print("=== numbering ===")
figs = re.findall(r"<figcaption><b>图\s*(\d+)", h)
tabs = re.findall(r"<caption><b>表\s*(\d+)", h)
print("  figs", len(figs), "seq", figs == [str(i) for i in range(1, len(figs) + 1)])
print("  tabs", len(tabs), "seq", tabs == [str(i) for i in range(1, len(tabs) + 1)])
print("  fig refs out of range:",
      sorted({int(x) for x in re.findall(r"图\s*(\d+)", h)} - {int(x) for x in figs}))
print("  tab refs out of range:",
      sorted({int(x) for x in re.findall(r"表\s*(\d+)", h)} - {int(x) for x in tabs}))
print("  residual 原生: html", h.count("原生"), "md", m.count("原生"))

print("\n=== new section 2.4 present and in the right slot ===")
a, b = h.find("<h2>二"), h.find("<h2>三")
subs = [re.sub(r"<[^>]+>", "", x).strip() for x in re.findall(r"<h3>(.*?)</h3>", h[a:b], re.S)]
for s in subs:
    print("  -", s[:44])
print("  count:", len(subs))

print("\n=== table 3 / figure 10 identity numbers vs source json ===")
d = json.load(open("outputs/premodel/scale_dialect.json", encoding="utf-8"))
print("  scenarios in json:", d.get("scenarios"), "| resolutions:", d.get("resolutions"))
for scen in d["results"]:
    for res, rec in d["results"][scen].items():
        fn = rec["fine_nn"]
        bw = rec["block_weighted"]
        print(f"  {scen:>5} {res:>5} fine={fn['mse_fine']:.4f} coarse={fn['mse_coarse']:.4f} "
              f"within={fn['mse_within']:.4f} cross={fn['cross']:+.2e} "
              f"| wsum={bw['sum']:.4f} excess_over_fine={bw['excess_over_fine']:+.2e}")
print("\n  roundtrip (nearest upsample -> aggregate):")
for k, v in d.get("roundtrip", {}).items():
    print(f"    {k}: max {v['max_abs']:.2e} m  mean {v['mean_abs']:.2e} m  n={v['n']}")

print("\n=== does the report quote those numbers? ===")
for tok in ("0.1150", "0.0999", "0.1564", "0.1932", "0.1610", "0.1389", "0.2123", "0.2768"):
    print(f"  {tok}: {h.count(tok)}")

print("\n=== scenario inventory (the flagged deviation) ===")
g = r"E:\Projects\20260921-JiaChenchen-Paper\dataset\grids"
for res in ("2m", "5m", "10m", "20m", "30m"):
    p = os.path.join(g, res)
    if os.path.isdir(p):
        fl = sorted(x for x in os.listdir(p) if x.endswith(".nc"))
        print(f"  {res}: {fl}")

print("\n=== downstream / model artefacts: anything touched today? ===")
cut = "2026-10-02 02:05"
hits = []
for root, dirs, files in os.walk("."):
    dirs[:] = [x for x in dirs if x not in (".git", "__pycache__", "node_modules", "checks")]
    if any(s in root for s in ("premodel", "grids", "outputs")):
        pass
    for f in files:
        if f.endswith((".pt", ".pth", ".ckpt", ".safetensors", ".nc")):
            p = os.path.join(root, f)
            try:
                t = os.path.getmtime(p)
            except OSError:
                continue
            ts = __import__("datetime").datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
            if ts > cut:
                hits.append((ts, os.path.getsize(p) // 1024, p))
print("  weights/nc files modified after", cut, ":", len(hits))
for ts, kb, p in sorted(hits, reverse=True)[:25]:
    print(f"    {ts}  {kb:>7} KB  {p}")

print("\n=== premodel dir mtimes ===")
for f in sorted(os.listdir("outputs/premodel")):
    p = os.path.join("outputs/premodel", f)
    ts = __import__("datetime").datetime.fromtimestamp(os.path.getmtime(p)).strftime("%m-%d %H:%M")
    print(f"  {ts}  {f}")
