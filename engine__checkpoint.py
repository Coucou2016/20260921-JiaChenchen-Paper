"""Checkpoint helpers. Best model keyed by val CSI@0.05, not PSNR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, Any],
    config: dict | None = None,
    scaler=None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
        "config": config,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state = ckpt["model"]
    # Backward compatibility: checkpoints written before the direct-head ablation fix
    # (e.g. the frozen V0 ep180 baseline) have no ``direct_head.*`` keys, because that
    # head was added after they were trained and `predict_residual=True` means it was
    # never used anyway. Loading such a checkpoint with strict=True would now fail, so
    # missing keys are tolerated when they belong to an UNUSED branch. Extra keys in the
    # checkpoint that the model does not have are still an error (strict for those).
    model_keys = set(model.state_dict().keys())
    missing = [k for k in model_keys if k not in state]
    unexpected = [k for k in state if k not in model_keys]
    benign = [k for k in missing if k.startswith("direct_head.")]
    if benign and len(benign) == len(missing):
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state)
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt
