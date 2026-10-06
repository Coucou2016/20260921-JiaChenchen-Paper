"""Verify the winner snapshot is a resumable checkpoint before extending from it."""
from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "outputs" / "finetune_deep" / "w01" / "snapshots" / "ep0187.pt"

ck = torch.load(CK, map_location="cpu", weights_only=False)
print(f"file: {CK}")
print(f"top-level keys: {sorted(ck.keys())}")
print(f"epoch: {ck.get('epoch')}")
print(f"has model state: {isinstance(ck.get('model'), dict)}")
print(f"has optimizer state: {ck.get('optimizer') is not None}")
if ck.get("optimizer") is not None:
    lrs = [pg.get("lr") for pg in ck["optimizer"].get("param_groups", [])]
    print(f"optimizer param_group lrs (restored, then overridden by --sched_epochs/--lr): {lrs}")
if "scheduler" in ck:
    print(f"scheduler state present: {type(ck['scheduler']).__name__ if ck['scheduler'] else None}")
cfg = ck.get("config") or {}
loss = (cfg.get("loss") or {}) if isinstance(cfg, dict) else {}
print(f"loss cfg in ckpt: { {k: loss.get(k) for k in ('name','w_deep','deep_threshold','deep_delta')} }")
