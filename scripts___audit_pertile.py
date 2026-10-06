"""Inspect the per-tile test structure that a spatial block bootstrap would need."""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = r"E:\Projects\20260921-JiaChenchen-Paper"
os.chdir(ROOT)

d = json.load(open("outputs/v0_10m2m_hmax/visualizations/diagnostics_test.json",
                   encoding="utf-8"))
print("top keys:", list(d))
print("n_tiles:", d.get("n_tiles"))
pt = d.get("per_tile")
print("per_tile type:", type(pt).__name__)
if isinstance(pt, list):
    print("per_tile len:", len(pt))
    print("first entry:", json.dumps(pt[0], ensure_ascii=False)[:800])
    print("entry keys:", list(pt[0]) if isinstance(pt[0], dict) else "n/a")
if isinstance(pt, dict):
    k = list(pt)[0]
    print("per_tile first key:", k)
    print("first entry:", json.dumps(pt[k], ensure_ascii=False)[:800])

print("\n=== does it carry iy/ix/scenario? ===")
sample = pt[0] if isinstance(pt, list) else pt[list(pt)[0]]
for key in ("iy", "ix", "scenario", "tile", "id", "row", "col", "res"):
    print(f"  {key}: {sample.get(key) if isinstance(sample, dict) else 'n/a'}")

print("\n=== per-tile metrics json structure ===")
p = "outputs/v0_10m2m_hmax/visualizations/ep155/per_tile_metrics_test.json"
d2 = json.load(open(p, encoding="utf-8"))
print("type:", type(d2).__name__)
if isinstance(d2, list):
    print("len:", len(d2))
    print("[0]:", json.dumps(d2[0], ensure_ascii=False)[:600])
elif isinstance(d2, dict):
    print("keys:", list(d2)[:10])
    k = list(d2)[0]
    print(f"d['{k}']:", json.dumps(d2[k], ensure_ascii=False)[:600])

print("\n=== scenario/iy/ix availability across per-tile files ===")
for p in ["outputs/v0_10m2m_hmax/visualizations/ep155/per_tile_metrics_test.json",
          "outputs/finetune_deep/w01ext/visualizations/ep189/per_tile_metrics_test.json",
          "outputs/deep_sweep/D0/visualizations/ep018/per_tile_metrics_test.json"]:
    dd = json.load(open(p, encoding="utf-8"))
    s = dd[0] if isinstance(dd, list) else dd
    print(f"  {p}")
    print(f"    -> keys: {list(s)[:14] if isinstance(s, dict) else type(s).__name__}")
