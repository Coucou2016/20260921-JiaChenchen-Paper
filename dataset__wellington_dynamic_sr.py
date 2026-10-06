"""Dynamic single-frame flood SR: LR [h, hux, hvy, valid] → HR depth (plan §二十一)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from dataset.normalization import StaticNormalizer, select_static_channels
from dataset.wellington_sr import WellingtonSRDataset


class WellingtonDynamicSRDataset(Dataset):
    """Hourly frames with unit-width discharge channels (t=0 already dropped by reader)."""

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        lr_res: int = 10,
        hr_res: int = 2,
        depth_ref: float = 0.10,
        geo_mode: str = "all",
        scenarios: tuple[str, ...] = ("20a", "100a"),
    ) -> None:
        self.base = WellingtonSRDataset(
            root=root,
            lr_res=lr_res,
            hr_res=hr_res,
            split=split,
            target="h",
            scenarios=scenarios,
            include_flow=True,
            static_res="hr",
        )
        self.depth_ref = depth_ref
        self.geo_mode = geo_mode
        self.lr_res = lr_res
        self.hr_res = hr_res
        self.normalizer = StaticNormalizer(hr_res=hr_res, root=self.base.root)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        x = self.base[idx]
        # lr: (3, Hl, Wl) = h, hux, hvy
        lr = torch.as_tensor(x["lr"], dtype=torch.float32)
        hr = torch.as_tensor(x["hr"], dtype=torch.float32)
        static = torch.as_tensor(x["static"], dtype=torch.float32)
        mask = torch.as_tensor(x["mask"], dtype=torch.bool)

        lr_valid = torch.isfinite(lr[:1]).float()  # validity from depth channel
        lr = torch.nan_to_num(lr, nan=0.0)
        hr = torch.nan_to_num(hr, nan=0.0)

        static_cont, landuse = self.normalizer.split_static(static)
        static_cont, landuse_sel = select_static_channels(static_cont, landuse, self.geo_mode)
        landuse_out = landuse if landuse_sel is None else landuse_sel
        use_landuse = landuse_sel is not None

        return {
            "lr": lr,  # 3 channels
            "lr_valid": lr_valid,
            "hr": hr,
            "static_cont": static_cont,
            "landuse": landuse_out,
            "use_landuse": use_landuse,
            "mask": mask,
            "scenario": x["scenario"],
            "time_index": int(x["time_index"]),
            "iy": int(x["iy"]),
            "ix": int(x["ix"]),
            "scale": float(x["scale"]),
            "lr_res": float(self.lr_res),
            "hr_res": float(self.hr_res),
            "depth_ref": float(self.depth_ref),
        }
