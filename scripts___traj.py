"""Per-epoch trajectories for the deep-water fine-tune, for honest reading of the shape."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
j = json.loads((ROOT / "outputs" / "finetune_deep" / "epoch_scan.json").read_text(encoding="utf-8"))
s = j["scans"]
froz = j["refs"]["frozen_ep180"]

print("ep  |  w00 CSI005/100  |  w005 CSI005/100  |  w01 CSI005/100")
print(f"180 | frozen          | 0.5329/0.1840      | (frozen ep180)")
for e in range(181, 201):
    a = s["w00"][f"ep{e:04d}"]
    b = s["w005"][f"ep{e:04d}"]
    c = s["w01"][f"ep{e:04d}"]
    print(f"{e} | {a['CSI_005']:.4f}/{a['CSI_100']:.4f} | {b['CSI_005']:.4f}/{b['CSI_100']:.4f} "
          f"| {c['CSI_005']:.4f}/{c['CSI_100']:.4f}")

print("\nfrozen ep180:", f"CSI_005={froz['CSI_005']:.4f} CSI_100={froz['CSI_100']:.4f} "
      f"PeakDepth={froz['PeakDepthError']:.4f} VolErr={froz['VolumeRelativeError']:.4f}")
print("\nearly-vs-late mean CSI_100 (does the gain hold, or fade?):")
for arm, name in [("w00", "control"), ("w005", "w=0.05"), ("w01", "w=0.10")]:
    v = [s[arm][f"ep{e:04d}"]["CSI_100"] for e in range(181, 201)]
    print(f"  {name:8s} ep181-185 mean={sum(v[:5])/5:.4f}  ep196-200 mean={sum(v[-5:])/5:.4f}  "
          f"all mean={sum(v)/20:.4f}  best={max(v):.4f}")
