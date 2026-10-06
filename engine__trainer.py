"""Training loop for fixed-scale HydroGeo-SRNO (plan §二十六)."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.amp import GradScaler, autocast

from engine.checkpoint import save_checkpoint
from metrics.aggregation import average_metrics
from metrics.flood_metrics import compute_flood_metrics


def move_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def train_step(
    batch: dict,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    max_grad_norm: float = 1.0,
    use_amp: bool = False,
) -> dict[str, float]:
    model.train()
    batch = move_to_device(batch, device)
    optimizer.zero_grad(set_to_none=True)

    amp_ctx = autocast(device_type=device.type, enabled=use_amp and device.type == "cuda")
    with amp_ctx:
        output = model(
            lr=batch["lr"],
            lr_valid=batch["lr_valid"],
            static_cont=batch["static_cont"],
            landuse=batch["landuse"],
            lr_res=batch["lr_res"],
            hr_res=batch["hr_res"],
        )
        loss_dict = criterion(output=output, target=batch["hr"], mask=batch["mask"])
        loss = loss_dict["total"]

    if scaler is not None and use_amp and device.type == "cuda":
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

    return {k: float(v.detach().item()) if torch.is_tensor(v) else float(v) for k, v in loss_dict.items()}


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    metric_list = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = move_to_device(batch, device)
        output = model(
            lr=batch["lr"],
            lr_valid=batch["lr_valid"],
            static_cont=batch["static_cont"],
            landuse=batch["landuse"],
            lr_res=batch["lr_res"],
            hr_res=batch["hr_res"],
        )
        metric_list.append(compute_flood_metrics(output["depth"], batch["hr"], batch["mask"]))
    return average_metrics(metric_list)


class EarlyStopper:
    def __init__(self, patience: int = 40, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best = None
        self.bad = 0

    def step(self, value: float) -> bool:
        """Return True if training should stop."""
        if self.best is None:
            self.best = value
            return False
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience
