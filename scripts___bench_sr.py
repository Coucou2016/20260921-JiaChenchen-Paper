"""Scratch benchmark: isolate dataset-load vs GPU step time for the SR task."""
from __future__ import annotations
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from dataset.wellington_fixed_sr import WellingtonFixedSRDataset, collate_fixed
from engine.evaluator import build_model
from engine.trainer import train_step
from losses.flood_loss import FloodLoss
import yaml

cfg = yaml.safe_load((ROOT / "configs/v0_10m2m_hmax.yaml").read_text(encoding="utf-8"))
ds = WellingtonFixedSRDataset(root=ROOT / "dataset", split="val", lr_res=10, hr_res=2,
                              target="h_max", geo_mode="all", scenarios=("20a", "100a"))
print("n val", len(ds))
# time raw item loads
t0 = time.time()
for i in range(10):
    b = ds[i]
t_load = (time.time() - t0) / 10
print(f"raw __getitem__: {t_load*1000:.0f} ms/item -> est {t_load*550:.0f}s for 550 tiles")

items = [ds[i] for i in range(8)]
batch = collate_fixed(items[:1])
dev = torch.device("cuda")
model = build_model("hydrogeo_srno", cfg).to(dev)
crit = FloodLoss(w_depth=1.0, w_wet=0.3, w_log=0.2, w_boundary=0.1, w_extreme=0.1)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
# warm
for _ in range(2):
    train_step(batch, model, crit, opt, None, dev, max_grad_norm=1.0, use_amp=False)
torch.cuda.synchronize()
t0 = time.time()
N = 12
for _ in range(N):
    train_step(batch, model, crit, opt, None, dev, max_grad_norm=1.0, use_amp=False)
torch.cuda.synchronize()
dt = (time.time() - t0) / N
print(f"train_step (48x48 base, b=1): {dt*1000:.0f} ms/step -> est {dt*550:.0f}s/epoch (550 tiles)")
# forward only on 240 base
torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    with torch.no_grad():
        model(lr=batch["lr"], lr_valid=batch["lr_valid"], static_cont=batch["static_cont"],
              landuse=batch["landuse"], lr_res=batch["lr_res"], hr_res=batch["hr_res"])
    torch.cuda.synchronize()
print(f"forward only: {(time.time()-t0)/5*1000:.0f} ms")
print("vram used GB", torch.cuda.max_memory_allocated()/1024**3)
