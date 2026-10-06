import numpy as np
import netCDF4 as nc
import sys
sys.path.insert(0, "scripts")
import premodel_lib as L

for res in ["2m", "5m", "10m", "20m", "30m"]:
    ds = nc.Dataset(L.GRIDS / res / "static.nc")
    print("===", res)
    for nm in L.STATIC_CHANNELS:
        a = np.asarray(ds[nm][:]).astype("float64")
        u = np.unique(a)
        neg = u[u < 0]
        print(f"  {nm:16s} min={a.min():.4g} max={a.max():.4g} "
              f"negvals={neg[:6].tolist()} n_neg={int((a < 0).sum())}")
    ds.close()
