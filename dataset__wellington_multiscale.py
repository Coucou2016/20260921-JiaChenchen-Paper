"""Multi-resolution / arbitrary-scale Wellington SR dataset (plan §十七–二十).

Physical patch is always 480 m × 480 m. Scale is float(lr_res)/float(hr_res).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from torch.utils.data import ConcatDataset, Dataset

from dataset.wellington_fixed_sr import WellingtonFixedSRDataset

# Seen pairs for stage M1 / M2 (integer). Unseen stress: 5→2 (×2.5), 20→2, 30→2.
DEFAULT_SEEN_PAIRS: tuple[tuple[int, int], ...] = (
    (20, 10),
    (10, 5),
    (30, 10),
    (20, 5),
    (10, 2),
    (30, 5),
)

FRACTIONAL_PAIRS: tuple[tuple[int, int], ...] = (
    (5, 2),  # ×2.5 — core arbitrary-scale test
)

STRESS_PAIRS: tuple[tuple[int, int], ...] = (
    (20, 2),  # ×10
    (30, 2),  # ×15
)


def _needs_fractional(lr: int, hr: int) -> bool:
    return (lr % hr != 0) and (hr % lr != 0)


class WellingtonMultiscaleDataset(Dataset):
    """Concatenation of fixed-SR datasets over resolution pairs."""

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        pairs: Sequence[tuple[int, int]] = DEFAULT_SEEN_PAIRS,
        target: str = "h_max",
        depth_ref: float = 0.10,
        geo_mode: str = "all",
        scenarios: tuple[str, ...] = ("20a", "100a"),
    ) -> None:
        self.datasets = []
        for lr, hr in pairs:
            self.datasets.append(
                WellingtonFixedSRDataset(
                    root=root,
                    split=split,
                    lr_res=lr,
                    hr_res=hr,
                    target=target,
                    depth_ref=depth_ref,
                    geo_mode=geo_mode,
                    scenarios=scenarios,
                    allow_fractional_scale=_needs_fractional(lr, hr),
                )
            )
        self.concat = ConcatDataset(self.datasets)
        self.pairs = list(pairs)
        # Map each global index to its (lr_res, hr_res), so a batch sampler can
        # group samples of equal spatial size without inspecting tensors.
        self.pair_of_index: list[tuple[int, int]] = []
        for (lr, hr), sub in zip(self.pairs, self.datasets):
            self.pair_of_index.extend([(int(lr), int(hr))] * len(sub))

    def groups(self) -> dict[tuple[int, int], list[int]]:
        """Global indices grouped by resolution pair (equal LR/HR shape)."""
        out: dict[tuple[int, int], list[int]] = {p: [] for p in self.pairs}
        for i, p in enumerate(self.pair_of_index):
            out.setdefault(p, []).append(i)
        return out

    def __len__(self) -> int:
        return len(self.concat)

    def __getitem__(self, idx: int):
        return self.concat[idx]
