"""Exact numbers for the zoom-figure caption, on the final windows."""
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
PRE = r"E:\Projects\20260921-JiaChenchen-Paper\outputs\premodel"
D = {"5m": 2.0, "10m": 1.0, "20m": 0.5, "30m": 10.0 / 30.0}
REG = {"区域一": (1188, 1230, 468, 510), "区域二": (330, 372, 1068, 1110)}

for name, (r0, r1, c0, c1) in REG.items():
    print(f"\n{name}  10 m rows {r0} to {r1}, cols {c0} to {c1}, "
          f"window {(r1 - r0) * 10} m square")
    for res, d in D.items():
        a = np.load(f"{PRE}\\map_100a_{res}.npz")
        t = a["truth"].astype("float32")[int(r0 * d):int(r1 * d), int(c0 * d):int(c1 * d)]
        e = a["err"].astype("float32")[int(r0 * d):int(r1 * d), int(c0 * d):int(c1 * d)]
        fin = np.isfinite(t) & np.isfinite(e)
        wet = (t > 0.05) & fin
        print(f"  {res:>4} {t.shape[0]:>3}x{t.shape[1]:<3} "
              f"wet {wet.mean() * 100:5.1f}%  max {t[fin].max():5.2f}  "
              f"wetmean {t[wet].mean():5.2f}  MAE {np.abs(e[fin]).mean():.3f}  "
              f"RMSE {np.sqrt((e[fin] ** 2).mean()):.3f}")

# what the coarse grids lose, measured inside the window
print("\nwithin-window spread of the aggregated truth (std over wet cells)")
for name, (r0, r1, c0, c1) in REG.items():
    vals = []
    for res, d in D.items():
        a = np.load(f"{PRE}\\map_100a_{res}.npz")["truth"].astype("float32")
        t = a[int(r0 * d):int(r1 * d), int(c0 * d):int(c1 * d)]
        f = np.isfinite(t) & (t > 0.05)
        vals.append((res, float(t[f].std())))
    print(f"  {name}: " + "  ".join(f"{r} {v:.3f}" for r, v in vals))
