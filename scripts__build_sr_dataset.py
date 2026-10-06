"""Build the aligned, patch-indexed Wellington flood SR dataset.

Reads (never writes) wellington-output-data/ and static_geo_data/.
Writes dataset/grids, dataset/rain, dataset/index, dataset/manifest.json.

Common domain, anchored at the shared NW corner, sized so a 480 m patch
lands on an integer number of cells at 2/5/10/20/30 m:
    west 1743040, north 5442720, width 12960 m, height 23040 m.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import netCDF4 as nc
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static_geo_data"
FLOOD = ROOT / "wellington-output-data"
OUT = ROOT / "dataset"

RES_LIST = [30, 20, 10, 5, 2]
FLOOD_RP = ["20a", "100a"]
WEST = 1_743_040.0
NORTH = 5_442_720.0
WIDTH_M = 12_960
HEIGHT_M = 23_040
PATCH_M = 480
NODATA = -9999.0

STATIC_FLOAT = [
    "DEM", "Slope", "Aspect_sin", "Aspect_cos", "Curv_plan", "Curv_profile",
    "Tpi", "Twi", "Dist_building", "Dist_road", "Dist_water", "Infiltration", "Manning",
]
STATIC_INT = {"Landuse": "i2", "Building_binary": "i1"}

LANDUSE = {
    1: {"name": "building", "manning": 0.5, "infiltration": 0.0},
    2: {"name": "road", "manning": 0.013, "infiltration": 0.0},
    3: {"name": "impervious", "manning": 0.013, "infiltration": 0.0},
    4: {"name": "pervious", "manning": 0.1, "infiltration": 9.17e-7},
    5: {"name": "missing_in_source", "manning": 0.035, "infiltration": 4.58e-7},
    6: {"name": "green", "manning": 0.1, "infiltration": 9.17e-7},
    7: {"name": "water", "manning": 0.035, "infiltration": 0.0},
}


def crop_shape(res: int) -> tuple[int, int]:
    return HEIGHT_M // res, WIDTH_M // res


def patch_shape(res: int) -> tuple[int, int]:
    return PATCH_M // res, PATCH_M // res


def read_ascii(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(6):
            f.readline()
        return np.loadtxt(f, dtype=np.float32)


def finite_mask(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & (a != NODATA)


def write_static(res: int) -> Path:
    ny, nx = crop_shape(res)
    py, px = patch_shape(res)
    out = OUT / "grids" / f"{res}m" / "static.nc"
    if out.exists():
        with nc.Dataset(out) as ds:
            if ds.dimensions["y"].size == ny and "DEM" in ds.variables and "Landuse" in ds.variables:
                print(f"  static {res}m exists, skip", flush=True)
                return out
    out.parent.mkdir(parents=True, exist_ok=True)
    src = STATIC / f"case_{res}m_100a"
    t0 = time.time()
    ds = nc.Dataset(str(out), "w", format="NETCDF4")
    ds.createDimension("y", ny)
    ds.createDimension("x", nx)
    y = (NORTH - (np.arange(ny) + 0.5) * res).astype(np.float64)
    x = (WEST + (np.arange(nx) + 0.5) * res).astype(np.float64)
    ds.createVariable("y", "f8", ("y",))[:] = y
    ds.createVariable("x", "f8", ("x",))[:] = x
    ds.setncatts({
        "resolution_m": res,
        "west_edge": WEST,
        "north_edge": NORTH,
        "height_m": HEIGHT_M,
        "width_m": WIDTH_M,
        "crs_inferred": "EPSG:2193 NZTM2000",
        "source_case": src.name,
        "note": "Cropped from the shared NW corner. Row 0 is north.",
    })
    for name in STATIC_FLOAT:
        print(f"  {res}m {name}", flush=True)
        arr = read_ascii(src / f"{name}.txt")[:ny, :nx]
        var = ds.createVariable(
            name, "f4", ("y", "x"), zlib=True, complevel=1, shuffle=True,
            chunksizes=(py, px), fill_value=np.float32(NODATA),
        )
        var[:] = arr
        del arr
    for name, dtype in STATIC_INT.items():
        print(f"  {res}m {name}", flush=True)
        arr = read_ascii(src / f"{name}.txt")[:ny, :nx]
        bad = ~finite_mask(arr)
        ivar = ds.createVariable(
            name, dtype, ("y", "x"), zlib=True, complevel=1, shuffle=True,
            chunksizes=(py, px), fill_value=np.int16(NODATA),
        )
        out_arr = np.full(arr.shape, NODATA, dtype=np.float32)
        out_arr[~bad] = arr[~bad]
        ivar[:] = out_arr
        del arr, out_arr
    ds.close()
    print(f"  static {res}m done in {time.time() - t0:.0f}s -> {out}", flush=True)
    return out


def write_flood(res: int, rp: str) -> Path:
    ny, nx = crop_shape(res)
    py, px = patch_shape(res)
    out = OUT / "grids" / f"{res}m" / f"flood_{rp}.nc"
    if out.exists():
        with nc.Dataset(out) as ds:
            if "h" in ds.variables and ds.variables["h"].shape[1:] == (ny, nx):
                print(f"  flood {rp} {res}m exists, skip", flush=True)
                return out
    out.parent.mkdir(parents=True, exist_ok=True)
    folder = FLOOD / f"{rp}_{res}m"
    t0 = time.time()
    ds = nc.Dataset(str(out), "w", format="NETCDF4")
    src_h = xr.open_dataset(folder / "flood_h.nc")
    times = src_h["time"].values.astype(np.float64)
    ds.createDimension("time", times.size)
    ds.createDimension("y", ny)
    ds.createDimension("x", nx)
    ds.createVariable("time", "f8", ("time",))[:] = times
    ds["time"].units = "seconds"
    y = (NORTH - (np.arange(ny) + 0.5) * res).astype(np.float64)
    x = (WEST + (np.arange(nx) + 0.5) * res).astype(np.float64)
    ds.createVariable("y", "f8", ("y",))[:] = y
    ds.createVariable("x", "f8", ("x",))[:] = x
    ds.setncatts({"return_period": rp, "resolution_m": res, "rainfall_duration_s": 21600})

    def dump(name: str, da: xr.DataArray, dims: tuple[str, ...], chunks: tuple[int, ...]) -> None:
        arr = np.asarray(da.values, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, :ny, :nx]
        else:
            arr = arr[:ny, :nx]
        var = ds.createVariable(
            name, "f4", dims, zlib=True, complevel=1, shuffle=True,
            chunksizes=chunks, fill_value=np.float32(NODATA),
        )
        var[:] = np.where(np.isfinite(arr), arr, NODATA)
        var.units = da.attrs.get("units", "")
        var.long_name = da.attrs.get("long_name", name)
        del arr

    print(f"  flood {rp} {res}m h", flush=True)
    dump("h", src_h["h"], ("time", "y", "x"), (1, py, px))
    src_h.close()
    for key, fname in (("hux", "flood_hux.nc"), ("hvy", "flood_hvy.nc")):
        print(f"  flood {rp} {res}m {key}", flush=True)
        s = xr.open_dataset(folder / fname)
        dump(key, s[key], ("time", "y", "x"), (1, py, px))
        s.close()
    print(f"  flood {rp} {res}m h_max", flush=True)
    s = xr.open_dataset(folder / "flood_h_max.nc")
    dump("h_max", s["h_max"], ("y", "x"), (py, px))
    s.close()
    ds.close()
    print(f"  flood {rp} {res}m done in {time.time() - t0:.0f}s", flush=True)
    return out


def write_rain() -> None:
    rain_dir = OUT / "rain"
    rain_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for rp in ["20a", "50a", "100a"]:
        src = STATIC / f"case_30m_{rp}" / "Precipitation_source_1.dat"
        a = np.loadtxt(src, dtype=np.float64)
        depth = a[:, 1].astype(np.float32)
        np.save(rain_dir / f"{rp}.npy", depth)
        summary[rp] = {
            "n_seconds": int(depth.size),
            "total_m": float(depth.sum()),
            "total_mm": float(depth.sum() * 1000),
            "peak_m_per_s": float(depth.max()),
            "peak_index_s": int(np.argmax(depth) + 1),
            "has_flood_output": rp in FLOOD_RP,
            "units": "m/s depth increment (1 s steps, 6 h). Inferred from the 6 h total.",
        }
    (rain_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("rain", summary, flush=True)


def split_of(iy: int) -> str:
    if iy <= 29:
        return "train"
    if iy == 30 or iy == 37:
        return "buffer"
    if 31 <= iy <= 36:
        return "val"
    return "test"


def build_index() -> None:
    """Validity from 2 m DEM; wet fraction from 2 m h_max. Geography decides the split."""
    ny, nx = crop_shape(2)
    py, px = patch_shape(2)
    npy, npx = HEIGHT_M // PATCH_M, WIDTH_M // PATCH_M
    with nc.Dataset(OUT / "grids" / "2m" / "static.nc") as ds:
        dem = np.array(ds["DEM"][:], dtype=np.float32)
    valid = finite_mask(dem)
    hmax = {}
    for rp in FLOOD_RP:
        with nc.Dataset(OUT / "grids" / "2m" / f"flood_{rp}.nc") as ds:
            hmax[rp] = np.array(ds["h_max"][:], dtype=np.float32)
    patches = []
    counts = {"train": 0, "val": 0, "test": 0, "buffer": 0}
    usable = {"train": 0, "val": 0, "test": 0}
    for iy in range(npy):
        for ix in range(npx):
            sl = (slice(iy * py, (iy + 1) * py), slice(ix * px, (ix + 1) * px))
            v = valid[sl]
            frac = float(v.mean())
            wet = {}
            for rp in FLOOD_RP:
                h = hmax[rp][sl]
                both = v & finite_mask(h)
                wet[rp] = float(((h > 0.05) & both).sum() / max(both.sum(), 1))
            sp = split_of(iy)
            keep = frac >= 0.70 and sp != "buffer"
            rec = {
                "iy": iy, "ix": ix, "split": sp,
                "valid_frac": round(frac, 4),
                "wet_frac": {k: round(v_, 4) for k, v_ in wet.items()},
                "usable": keep,
            }
            patches.append(rec)
            counts[sp] += 1
            if keep:
                usable[sp] += 1
    index = {
        "patch_m": PATCH_M,
        "origin_west": WEST,
        "origin_north": NORTH,
        "width_m": WIDTH_M,
        "height_m": HEIGHT_M,
        "n_patch_y": npy,
        "n_patch_x": npx,
        "split_rule": {
            "train_iy": "0-29",
            "buffer_iy": "30 and 37",
            "val_iy": "31-36",
            "test_iy": "38-47",
            "usable_if": "valid_frac>=0.70 and split!=buffer",
        },
        "counts_all": counts,
        "counts_usable": usable,
        "patches": patches,
    }
    index_dir = OUT / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "patches_480m.json").write_text(json.dumps(index), encoding="utf-8")
    print("index usable", usable, "all", counts, flush=True)
    del dem, hmax


def _summ(a: np.ndarray) -> dict:
    v = a[finite_mask(a)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "std": float(v.std()),
    }


def build_norm_stats() -> None:
    """Mean/std on usable training patches only, at 2 m (HR) and 10 m (primary LR)."""
    index = json.loads((OUT / "index" / "patches_480m.json").read_text(encoding="utf-8"))
    train = [p for p in index["patches"] if p["usable"] and p["split"] == "train"]
    stats = {}
    for res, names in (
        (2, STATIC_FLOAT + list(STATIC_INT)),
        (10, STATIC_FLOAT + list(STATIC_INT)),
    ):
        py, px = patch_shape(res)
        with nc.Dataset(OUT / "grids" / f"{res}m" / "static.nc") as ds:
            for name in names:
                acc = []
                arr = np.array(ds[name][:])
                for p in train:
                    block = arr[p["iy"] * py:(p["iy"] + 1) * py, p["ix"] * px:(p["ix"] + 1) * px]
                    acc.append(block[finite_mask(block)].astype(np.float64))
                cat = np.concatenate(acc) if acc else np.array([])
                stats[f"static_{res}m_{name}"] = _summ(cat) if cat.size else {"n": 0}
                del arr
    for res in (2, 10):
        py, px = patch_shape(res)
        for rp in FLOOD_RP:
            with nc.Dataset(OUT / "grids" / f"{res}m" / f"flood_{rp}.nc") as ds:
                for name in ("h_max", "h", "hux", "hvy"):
                    acc = []
                    arr = np.array(ds[name][:])
                    for p in train:
                        block = arr[..., p["iy"] * py:(p["iy"] + 1) * py, p["ix"] * px:(p["ix"] + 1) * px]
                        if name == "h":
                            block = block[1:]  # drop the dry t=0 frame
                        acc.append(block[finite_mask(block)].ravel().astype(np.float64))
                    cat = np.concatenate(acc) if acc else np.array([])
                    stats[f"flood_{rp}_{res}m_{name}"] = _summ(cat)
                    del arr
    (OUT / "index" / "norm_stats_train.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("norm stats", len(stats), flush=True)


def check_alignment() -> dict:
    """2 m DEM block-mean vs 10 m DEM, on the common crop."""
    with nc.Dataset(OUT / "grids" / "2m" / "static.nc") as ds:
        d2 = np.array(ds["DEM"][:1000, :1000], dtype=np.float32)
    with nc.Dataset(OUT / "grids" / "10m" / "static.nc") as ds:
        d10 = np.array(ds["DEM"][:200, :200], dtype=np.float32)
    b = d2.reshape(200, 5, 200, 5)
    with np.errstate(all="ignore"):
        agg = np.nanmean(np.where(finite_mask(b), b, np.nan), axis=(1, 3))
    both = finite_mask(d10) & np.isfinite(agg)
    c, g = d10[both], agg[both]
    report = {
        "n": int(both.sum()),
        "corr": float(np.corrcoef(c, g)[0, 1]),
        "rmse_m": float(np.sqrt(np.mean((c - g) ** 2))),
        "bias_10m_minus_2m_m": float(np.mean(c - g)),
    }
    print("DEM nest check", report, flush=True)
    return report


def write_manifest(dem_check: dict) -> None:
    manifest = {
        "task": "guided super-resolution of Wellington urban flood depth",
        "primary_pair": {"lr_m": 10, "hr_m": 2, "scale": 5, "target": "h_max"},
        "cascade": ["30m->10m", "10m->2m"],
        "patch_m": PATCH_M,
        "domain": {
            "west": WEST, "north": NORTH,
            "width_m": WIDTH_M, "height_m": HEIGHT_M,
            "crs_inferred": "EPSG:2193",
        },
        "return_periods_with_flood": FLOOD_RP,
        "return_periods_rain_only": ["50a"],
        "dropped_frames": "t=0 s is dry and is excluded by the reader",
        "landuse": LANDUSE,
        "static_channels": STATIC_FLOAT + list(STATIC_INT),
        "dem_nest_check_10m_vs_2m": dem_check,
        "split": "north-to-south bands with a one-patch buffer; see index/patches_480m.json",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_rain()
    for res in RES_LIST:
        write_static(res)
        for rp in FLOOD_RP:
            write_flood(res, rp)
    build_index()
    dem_check = check_alignment()
    build_norm_stats()
    write_manifest(dem_check)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
