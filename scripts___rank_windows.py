"""Rank 420 m windows by how much coarsening flattens the field."""
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
PRE = r"E:\Projects\20260921-JiaChenchen-Paper\outputs\premodel"
A = {r: np.load(f"{PRE}\\map_100a_{r}.npz")["truth"].astype("float32")
     for r in ("5m", "10m", "20m", "30m")}
D = {"5m": 2.0, "10m": 1.0, "20m": 0.5, "30m": 10.0 / 30.0}
S = 42  # 420 m, divisible by 6 so every grid lands exactly
H, W = A["10m"].shape
rows = []
for i in range(0, H - S, 18):
    for j in range(0, W - S, 18):
        subs, ok = {}, True
        for r, d in D.items():
            t = A[r][int(i * d):int((i + S) * d), int(j * d):int((j + S) * d)]
            f = np.isfinite(t) & (t > 0.05)
            if f.mean() < 0.15:
                ok = False
                break
            subs[r] = t[f]
        if not ok:
            continue
        s5, s30 = subs["5m"].std(), subs["30m"].std()
        if s5 <= 0 or not np.isfinite(s30):
            continue
        rows.append((s30 / s5, s5, subs["5m"].mean(), subs["30m"].mean(), i, j,
                     float((A["10m"][i:i + S, j:j + S] > 0.05).mean())))

rows.sort()
print("windows where coarsening flattens the field most (low ratio = visible change)")
print(f"{'ratio':>6} {'std5m':>7} {'m5m':>6} {'m30m':>6} {'wet10':>6}  window")
for r, s5, m5, m30, i, j, wet in rows[:12]:
    print(f"{r:>6.3f} {s5:>7.3f} {m5:>6.2f} {m30:>6.2f} {wet:>6.2f}  "
          f"r{i}-{i+S} c{j}-{j+S}")

# also, spatially separated from region 1 (1188-1230, 468-510)
print("\nseparated from region one:")
n = 0
for r, s5, m5, m30, i, j, wet in rows:
    if abs(i - 1209) < 120 and abs(j - 489) < 120:
        continue
    print(f"{r:>6.3f} {s5:>7.3f} {m5:>6.2f} {m30:>6.2f} {wet:>6.2f}  r{i}-{i+S} c{j}-{j+S}")
    n += 1
    if n == 6:
        break
