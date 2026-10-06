"""Task-layer wrapper for fixed-scale 10 m → 2 m (and other integer pairs).

Does not reshuffle geographic splits. Adds lr_valid channel, depth nan filling,
and static split (continuous + landuse) with train-only normalization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from dataset.normalization import StaticNormalizer, depth_encode, select_static_channels
from dataset.wellington_sr import WellingtonSRDataset


class WellingtonFixedSRDataset(Dataset):
    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        lr_res: int = 10,
        hr_res: int = 2,
        target: str = "h_max",
        depth_ref: float = 0.10,
        geo_mode: str = "all",
        scenarios: tuple[str, ...] = ("20a", "100a"),
        include_flow: bool = False,
        allow_fractional_scale: bool = False,
    ) -> None:
        self.base = WellingtonSRDataset(
            root=root,
            lr_res=lr_res,
            hr_res=hr_res,
            split=split,
            target=target,
            scenarios=scenarios,
            include_flow=include_flow,
            static_res="hr",
            allow_fractional_scale=allow_fractional_scale,
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
        lr = torch.as_tensor(x["lr"], dtype=torch.float32)
        hr = torch.as_tensor(x["hr"], dtype=torch.float32)
        static = torch.as_tensor(x["static"], dtype=torch.float32)
        mask = torch.as_tensor(x["mask"], dtype=torch.bool)

        lr_valid = torch.isfinite(lr).float()
        lr = torch.nan_to_num(lr, nan=0.0)
        hr = torch.nan_to_num(hr, nan=0.0)

        static_cont, landuse = self.normalizer.split_static(static)
        static_cont, landuse_sel = select_static_channels(static_cont, landuse, self.geo_mode)
        if landuse_sel is None:
            # Dummy zero landuse (embedding padding) when geo ablation drops it.
            landuse_out = torch.zeros_like(landuse)
            use_landuse = False
        else:
            landuse_out = landuse_sel
            use_landuse = True

        return {
            "lr": lr,
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


def collate_fixed(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate tensors; keep scalar metadata as tensors for broadcasting."""
    out: dict[str, Any] = {}
    tensor_keys = (
        "lr", "lr_valid", "hr", "static_cont", "landuse", "mask",
    )
    for k in tensor_keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["use_landuse"] = bool(batch[0]["use_landuse"])
    for k in ("lr_res", "hr_res", "scale", "depth_ref"):
        out[k] = torch.tensor([b[k] for b in batch], dtype=torch.float32)
    out["scenario"] = [b["scenario"] for b in batch]
    out["iy"] = torch.tensor([b["iy"] for b in batch], dtype=torch.long)
    out["ix"] = torch.tensor([b["ix"] for b in batch], dtype=torch.long)
    out["time_index"] = torch.tensor([b["time_index"] for b in batch], dtype=torch.long)
    return out
