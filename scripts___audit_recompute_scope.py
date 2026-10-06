"""Inventory what is recomputable WITHOUT retraining:
per-tile metric files, saved predictions, and epoch scans.
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = r"E:\Projects\20260921-JiaChenchen-Paper"
os.chdir(ROOT)


def show(path, n=1200):
    if not os.path.exists(path):
        print(f"  MISSING {path}")
        return
    sz = os.path.getsize(path)
    print(f"  {path}  ({sz:,} B)")
    if path.endswith(".json") and sz < 3_000_000:
        d = json.load(open(path, encoding="utf-8"))
        if isinstance(d, dict):
            print(f"    keys[:12]: {list(d)[:12]}")
            k0 = list(d)[0]
            print(f"    sample d['{k0}']: {json.dumps(d[k0])[:400]}")
        elif isinstance(d, list):
            print(f"    list len={len(d)}; [0]={json.dumps(d[0])[:400]}")


print("=== per-tile test metrics (baseline V0) ===")
for p in ["outputs/post_sweep/metrics/baseline_v0_test.json",
          "outputs/finetune_deep/metrics/winner_w01_ep187_test.json"]:
    show(p)

print("\n=== which run dirs hold per_tile_metrics_test.json ===")
hits = []
for dp, dn, fn in os.walk("outputs"):
    for f in fn:
        if "per_tile_metrics_test" in f and f.endswith(".json"):
            p = os.path.join(dp, f)
            hits.append((os.path.getmtime(p), p))
hits.sort(reverse=True)
for _, p in hits[:25]:
    print("  ", p)
print(f"  total {len(hits)}")

print("\n=== is there a baseline V0 per-tile TEST file? ===")
for _, p in hits:
    if "v0" in p.lower() or "baseline" in p.lower():
        print("  ", p)

print("\n=== saved prediction arrays ===")
preds = []
for dp, dn, fn in os.walk("outputs"):
    for f in fn:
        if f.endswith((".npy", ".npz")) and any(
                k in f.lower() for k in ("pred", "shot", "infer")):
            p = os.path.join(dp, f)
            preds.append((os.path.getsize(p), p))
preds.sort(reverse=True)
for sz, p in preds[:20]:
    print(f"  {sz/1024:>10.1f} KB  {p}")
print(f"  total {len(preds)}")

print("\n=== epoch_scan.json (drives the n=20 stats) ===")
show("outputs/finetune_deep/epoch_scan.json")

print("\n=== finetune_deep contents ===")
if os.path.isdir("outputs/finetune_deep"):
    for n in sorted(os.listdir("outputs/finetune_deep")):
        p = os.path.join("outputs/finetune_deep", n)
        tag = "DIR " if os.path.isdir(p) else "FILE"
        print(f"  {tag} {n}")

print("\n=== diagnostics_test.json (per-tile, 242?) ===")
show("outputs/v0_10m2m_hmax/visualizations/diagnostics_test.json", 0)
