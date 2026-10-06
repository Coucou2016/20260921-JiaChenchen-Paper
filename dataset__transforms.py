"""Augmentation stubs. V0: no flip / rotation (aspect and flow are directional)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AugmentConfig:
    flip: bool = False
    rotation: bool = False


def physics_consistent_augment(sample: dict, cfg: AugmentConfig | None = None) -> dict:
    """Identity for V0. Future: sync Aspect_sin/cos and hux/hvy with geometric transforms."""
    cfg = cfg or AugmentConfig()
    if cfg.flip or cfg.rotation:
        raise NotImplementedError(
            "Physics-consistent flip/rotation is deferred until after V0 is stable "
            "(must transform Aspect and discharge vectors)."
        )
    return sample
