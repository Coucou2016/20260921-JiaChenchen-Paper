"""Patch reader for the Wellington flood super-resolution dataset.

Each item is one 480 m tile. Low-resolution flood depth is paired with the
high-resolution static stack (terrain, land use, distances) and the
high-resolution depth target. The geographic split is fixed in
index/patches_480m.json; do not reshuffle patches.
"""

from __future__ import annotations

import json
from pathlib import Path

import netCDF4 as nc
import numpy as np

NODATA = -9999.0
HERE = Path(__file__).resolve().parent

STATIC_CHANNELS = [
    "DEM", "Slope", "Aspect_sin", "Aspect_cos", "Curv_plan", "Curv_profile",
    "Tpi", "Twi", "Dist_building", "Dist_road", "Dist_water",
    "Infiltration", "Manning", "Landuse", "Building_binary",
]


def _clean(a: np.ndarray) -> np.ndarray:
    out = np.array(a, dtype=np.float32, copy=True)
    out[~np.isfinite(out) | (out == NODATA)] = np.nan
    return out


class WellingtonSRDataset:
    def __init__(
        self,
        root: str | Path | None = None,
        lr_res: int = 10,
        hr_res: int = 2,
        split: str = "train",
        target: str = "h_max",
        scenarios: tuple[str, ...] = ("20a", "100a"),
        include_flow: bool = False,
        static_res: str = "hr",
        allow_fractional_scale: bool = False,
    ) -> None:
        if target not in ("h_max", "h"):
            raise ValueError("target must be 'h_max' or 'h'")
        integer_ok = (lr_res % hr_res == 0) or (hr_res % lr_res == 0)
        if not allow_fractional_scale and not integer_ok:
            raise ValueError(
                f"resolutions {lr_res} and {hr_res} are not integer multiples; "
                "set allow_fractional_scale=True for pairs like 5→2"
            )
        if allow_fractional_scale:
            # Same physical 480 m patch at every resolution; verify extent via cell counts.
            if (480 % lr_res) != 0 or (480 % hr_res) != 0:
                raise ValueError(f"480 m patch must divide evenly by lr={lr_res} and hr={hr_res}")
        self.root = Path(root) if root is not None else HERE
        self.lr_res = lr_res
        self.hr_res = hr_res
        self.allow_fractional_scale = allow_fractional_scale
        self.scale = float(lr_res) / float(hr_res)
        self.split = split
        self.target = target
        self.scenarios = scenarios
        self.include_flow = include_flow
        self.static_res = hr_res if static_res == "hr" else lr_res
        index = json.loads((self.root / "index" / "patches_480m.json").read_text(encoding="utf-8"))
        patches = [p for p in index["patches"] if p["usable"] and p["split"] == split]
        # Flood frames are hourly from 0 to 21600 s. Frame 0 is dry.
        times = [0] if target == "h_max" else list(range(1, 7))
        self.samples = [
            {"patch": p, "scenario": rp, "time": t}
            for p in patches
            for rp in scenarios
            for t in times
        ]
        self._files: dict[tuple, nc.Dataset] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _ds(self, kind: str, res: int, scenario: str | None = None) -> nc.Dataset:
        key = (kind, res, scenario)
        if key not in self._files:
            if kind == "static":
                path = self.root / "grids" / f"{res}m" / "static.nc"
            else:
                path = self.root / "grids" / f"{res}m" / f"flood_{scenario}.nc"
            self._files[key] = nc.Dataset(path)
        return self._files[key]

    def _window(self, res: int, iy: int, ix: int) -> tuple[slice, slice]:
        py = 480 // res
        return slice(iy * py, (iy + 1) * py), slice(ix * py, (ix + 1) * py)

    def _static(self, res: int, iy: int, ix: int) -> np.ndarray:
        ys, xs = self._window(res, iy, ix)
        ds = self._ds("static", res)
        planes = [_clean(ds[name][ys, xs]) for name in STATIC_CHANNELS]
        return np.stack(planes, axis=0)

    def _flood(self, res: int, scenario: str, name: str, iy: int, ix: int, time: int | None) -> np.ndarray:
        ys, xs = self._window(res, iy, ix)
        ds = self._ds("flood", res, scenario)
        if name == "h_max":
            return _clean(ds[name][ys, xs])
        return _clean(ds[name][time, ys, xs])

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        p = sample["patch"]
        iy, ix = p["iy"], p["ix"]
        rp, t = sample["scenario"], sample["time"]
        names = ["h", "hux", "hvy"] if self.include_flow and self.target == "h" else [self.target if self.target == "h_max" else "h"]
        lr = np.stack([self._flood(self.lr_res, rp, n, iy, ix, None if n == "h_max" else t) for n in names], axis=0)
        hr_name = "h_max" if self.target == "h_max" else "h"
        hr = self._flood(self.hr_res, rp, hr_name, iy, ix, None if hr_name == "h_max" else t)[None]
        static = self._static(self.static_res, iy, ix)
        mask = np.isfinite(hr[0]) & np.isfinite(static[0])
        return {
            "lr": lr,
            "hr": hr,
            "static": static,
            "mask": mask.astype(np.float32),
            "scenario": rp,
            "time_index": t,
            "iy": iy,
            "ix": ix,
            "scale": self.scale,
            "static_names": STATIC_CHANNELS,
            "lr_names": names,
        }

    def close(self) -> None:
        for ds in self._files.values():
            ds.close()
        self._files.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
