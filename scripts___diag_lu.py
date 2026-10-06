import netCDF4 as nc
import numpy as np
import sys
sys.path.insert(0, "scripts")
import premodel_lib as L

land2 = L.load_static("2m", "Landuse")
print("land2 type", type(land2), land2.dtype)

f = nc.Dataset("dataset/grids/2m/flood_20a.nc")
dom = np.isfinite(L.clean(f.variables["h_max"][:]))
f.close()

n = (4608, 2592)
tot = None
for c in [1, 2, 3, 4, 5, 6, 7]:
    fr = L.aggregate(np.where(dom & (land2 == c), 1.0, np.nan).astype("float32"), 2, n)
    print("c", c, "nanmean", round(float(np.nanmean(fr)), 5), "max", float(np.nanmax(fr)))
    tot = fr.copy() if tot is None else tot + np.nan_to_num(fr)

print("sum nanmean", round(float(np.nanmean(tot)), 5))
print("sum max", float(np.nanmax(tot)), "sum min", float(np.nanmin(tot[dom.reshape(n[0], 2, n[1], 2).mean(axis=(1, 3)) > 0.5])))

mask = dom.reshape(n[0], 2, n[1], 2).mean(axis=(1, 3)) > 0.5
print("okd cells", int(mask.sum()), "frac>1.5", int((tot[mask] > 1.5).sum()))
