#!/usr/bin/env python
"""Experiment A driver: bilinear-residual cross-resolution learning (10 m -> 2 m).

Trains a residual learner on top of a plain interpolation of the coarse input and
evaluates it with exactly the same split, seed and metric protocol as the direct
V0 model, so the numbers are directly comparable.

Anti-drift notes (this box)
---------------------------
* A COMPLETE training run still exits 120 on this Windows box because the
  interpreter emits "Exception ignored on flushing sys.stdout" during teardown.
  Completion is therefore proved from artefacts (history rows + last.pt +
  done sentinel), never from the exit code.  ``--smoke-steps`` is the exception:
  it is meant to stop early.
* Training writes ~17 MB per checkpoint per epoch; with ``--save-every`` the
  snapshot cost is bounded.
* Single 4 GB GPU: use ``--wait-pid`` to serialise behind another job.

Usage
-----
    python -u scripts/train_bilinear_residual.py --smoke-steps 30 --epochs 1 \
        --out_dir outputs/v1_bilinear_residual/_smoke
    python -u scripts/train_bilinear_residual.py --epochs 15 \
        --out_dir outputs/v1_bilinear_residual/depth
    python -u scripts/train_bilinear_residual.py --base_mode nearest --epochs 15 \
        --out_dir outputs/v1_bilinear_residual/nearest
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.wellington_fixed_sr import WellingtonFixedSRDataset, collate_fixed
from engine.checkpoint import save_checkpoint
from engine.evaluator import build_model
from engine.reproducibility import make_generator, seed_everything, seed_worker, write_provenance
from engine.trainer import EarlyStopper, evaluate_loader, train_step
from losses.flood_loss import FloodLoss, MaskedL1Loss

FROZEN = ROOT / "outputs" / "v0_10m2m_hmax" / "last.pt"


def log(msg: str, d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (d / "driver.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                         capture_output=True, text=True, errors="ignore").stdout
    return str(pid) in out


def build_loss(cfg: dict) -> torch.nn.Module:
    lc = cfg.get("loss", {})
    if lc.get("name", "flood") == "masked_l1":
        return MaskedL1Loss()
    return FloodLoss(
        w_depth=lc.get("w_depth", 1.0), w_wet=lc.get("w_wet", 0.30),
        w_log=lc.get("w_log", 0.20), w_boundary=lc.get("w_boundary", 0.10),
        w_extreme=lc.get("w_extreme", 0.10),
        wet_threshold=lc.get("wet_threshold", 0.05),
        depth_ref=cfg.get("model", {}).get("depth_ref", 0.10),
        extreme_quantile=lc.get("extreme_quantile", 0.95),
        w_deep=lc.get("w_deep", 0.0),
        deep_threshold=lc.get("deep_threshold", 1.0),
        deep_quantile=lc.get("deep_quantile", None),
        deep_delta=lc.get("deep_delta", 5.0),
    )


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v1_bilinear_residual.yaml")
    ap.add_argument("--out_dir", default="outputs/v1_bilinear_residual/depth")
    ap.add_argument("--base_mode", default=None, choices=[None, "bilinear", "nearest"])
    ap.add_argument("--residual_space", default=None, choices=[None, "log", "depth"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=None,
                    help="cap optimizer steps PER epoch (smoke tests)")
    ap.add_argument("--smoke-steps", type=int, default=None,
                    help="global step cap: stop the whole run after this many steps")
    ap.add_argument("--save-every", type=int, default=1, help="checkpoint every N epochs")
    ap.add_argument("--snapshots", action="store_true",
                    help="also write snapshots/epNNNN.pt each epoch (for a full-val scan)")
    ap.add_argument("--eval_max_batches", type=int, default=None, help="default: full val")
    ap.add_argument("--resume", default=None, help="optional checkpoint to initialise from")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--wait-pid", type=int, default=None)
    ap.add_argument("--timeout-h", type=float, default=10.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if args.base_mode:
        cfg.setdefault("model", {})["base_mode"] = args.base_mode
    if args.residual_space:
        cfg.setdefault("model", {})["residual_space"] = args.residual_space
    cfg.setdefault("optimizer", {})["lr"] = args.lr
    cfg.setdefault("train", {})["epochs"] = args.epochs
    cfg.setdefault("train", {})["seed"] = args.seed
    if args.patience is not None:
        cfg.setdefault("early_stopping", {})["patience"] = args.patience

    out_dir = ROOT / args.out_dir
    base_mode = cfg["model"].get("base_mode", "bilinear")
    residual_space = cfg["model"].get("residual_space", "depth")

    if args.dry_run:
        print(f"DRY-RUN -> {out_dir} base={base_mode} space={residual_space} "
              f"epochs={args.epochs} lr={args.lr}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    log_f = open(log_path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)

    if args.wait_pid:
        log(f"waiting for pid={args.wait_pid} to exit (single GPU)", out_dir)
        t0 = time.time()
        while alive(args.wait_pid):
            if (time.time() - t0) / 3600 > args.timeout_h:
                log(f"timeout waiting for pid={args.wait_pid}; aborting", out_dir)
                return
            time.sleep(30)
        log(f"pid={args.wait_pid} gone after {(time.time()-t0)/60:.1f} min", out_dir)

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        raise SystemExit("CUDA_VISIBLE_DEVICES is empty: training would silently run on CPU.")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = cfg["train"]
    bs = int(train_cfg.get("batch_size", 1))
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb < 6 and bs > 1:
            bs = 1
    use_amp = False  # Maxwell sm_50, FP16 unstable

    print(f"device={device} batch_size={bs} base_mode={base_mode} "
          f"residual_space={residual_space} amp={use_amp}")

    ds_cfg = cfg["dataset"]
    root = ROOT / ds_cfg.get("root", "dataset")
    scenarios = tuple(ds_cfg.get("scenarios", ["20a", "100a"]))
    kw = dict(lr_res=int(ds_cfg.get("lr_res", 10)), hr_res=int(ds_cfg.get("hr_res", 2)),
              target=ds_cfg.get("target", "h_max"), geo_mode=ds_cfg.get("geo_mode", "all"),
              scenarios=scenarios)
    train_ds = WellingtonFixedSRDataset(root=root, split="train", **kw)
    val_ds = WellingtonFixedSRDataset(root=root, split="val", **kw)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    dl_workers = int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=dl_workers,
                              collate_fn=collate_fixed, pin_memory=(device.type == "cuda"),
                              worker_init_fn=seed_worker,
                              generator=make_generator(args.seed))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=dl_workers,
                            collate_fn=collate_fixed, pin_memory=(device.type == "cuda"))

    model = build_model(cfg["model"]["name"], cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters={n_params:,}")
    criterion = build_loss(cfg)
    opt_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(opt_cfg.get("lr", 1e-4)),
                                  weight_decay=float(opt_cfg.get("weight_decay", 1e-5)))

    start_epoch = 1
    parent_epoch = 0
    if args.resume and Path(args.resume).exists():
        from engine.checkpoint import load_checkpoint
        ck = load_checkpoint(args.resume, model, map_location=device)
        parent_epoch = int(ck.get("epoch", 0))
        start_epoch = parent_epoch + 1
        # A fine-tune must not restore the parent's stale lr / fast-forward the cosine:
        # the scheduler is built FRESH over exactly the epochs this run will execute.
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr
        print(f"warm-start from {args.resume} at parent epoch {parent_epoch}; lr={args.lr:.2e}")

    # ``--epochs`` is the number of epochs THIS run executes, so a warm-start arm is
    # directly comparable with the existing w00/w01 fine-tunes (20 epochs @ lr 1e-4
    # from the frozen ep180 checkpoint) regardless of the absolute epoch index.
    n_epochs = max(args.epochs, 1)
    sched = CosineAnnealingLR(optimizer, T_max=n_epochs)

    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    write_provenance(out_dir, cfg, args.seed, extra={
        "base_mode": base_mode, "residual_space": residual_space,
        "batch_size": bs, "amp": use_amp, "parameters": n_params,
        "experiment": "A_bilinear_residual",
        "parent": str(Path(args.resume).name) if args.resume else None,
        "parent_epoch": parent_epoch, "horizon_epochs": n_epochs, "lr": args.lr,
        "checkpoint_rule": "best validation CSI_005 (tile-macro, full val)",
    }, root=ROOT)

    es = cfg.get("early_stopping", {})
    stopper = EarlyStopper(patience=int(es.get("patience", 40)), mode="max")
    best_csi = -1.0
    best_rmse = float("inf")
    global_step = 0
    history_path = out_dir / "history.jsonl"
    log_every = int(train_cfg.get("log_every_steps", 50))
    t_run = time.time()
    done_sentinel = False

    for epoch in range(start_epoch, start_epoch + n_epochs):
        t_ep = time.time()
        model.train()
        running = []
        for step_i, batch in enumerate(train_loader, start=1):
            stats = train_step(batch, model, criterion, optimizer, None, device,
                               max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
                               use_amp=use_amp)
            running.append(stats["total"])
            global_step += 1
            if log_every and step_i % log_every == 0:
                print(f"  epoch={epoch} step={step_i}/{len(train_loader)} "
                      f"loss={stats['total']:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")
            if args.max_steps is not None and step_i >= args.max_steps:
                break
            if args.smoke_steps is not None and global_step >= args.smoke_steps:
                break
        sched.step()

        val_metrics = evaluate_loader(model, val_loader, device,
                                      max_batches=args.eval_max_batches)
        mean_loss = sum(running) / max(len(running), 1)
        csi = val_metrics.get("CSI_005", float("nan"))
        rmse = val_metrics.get("RMSE_wet", float("nan"))
        ep_sec = time.time() - t_ep
        print(f"epoch={epoch} (run {epoch-start_epoch+1}/{n_epochs}) step={global_step} "
              f"loss={mean_loss:.4f} "
              f"CSI_005={csi:.4f} RMSE_wet={rmse:.4f} CSI_100={val_metrics.get('CSI_100'):.4f} "
              f"sec={ep_sec:.1f}")

        row = {"epoch": epoch, "global_step": global_step, "loss": mean_loss,
               "sec": ep_sec, "base_mode": base_mode, "residual_space": residual_space,
               **val_metrics}
        with history_path.open("a", encoding="utf-8") as hf:
            hf.write(json.dumps(row) + "\n")

        if epoch % args.save_every == 0:
            save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, val_metrics, cfg)
        if args.snapshots:
            snap_dir = out_dir / "snapshots"
            snap_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(snap_dir / f"ep{epoch:04d}.pt", model, None, epoch,
                            val_metrics, cfg)
        if csi == csi and csi >= best_csi:
            best_csi = csi
            save_checkpoint(out_dir / "best_csi.pt", model, optimizer, epoch, val_metrics, cfg)
        if rmse == rmse and rmse <= best_rmse:
            best_rmse = rmse
            save_checkpoint(out_dir / "best_rmse_wet.pt", model, optimizer, epoch, val_metrics, cfg)
        (out_dir / "best_summary.json").write_text(json.dumps({
            "best_csi": best_csi, "best_rmse_wet": best_rmse, "last_epoch": epoch,
            "elapsed_sec": time.time() - t_run, "parameters": n_params,
            "base_mode": base_mode, "residual_space": residual_space}, indent=2),
            encoding="utf-8")

        stop = False
        if args.smoke_steps is not None and global_step >= args.smoke_steps:
            print("reached smoke step cap; stopping")
            stop = True
        if not stop and stopper.step(csi if csi == csi else -1.0):
            print(f"early stop at epoch {epoch}")
            stop = True
        if stop:
            done_sentinel = True
            break
    else:
        done_sentinel = True

    elapsed = (time.time() - t_run) / 3600
    # The sentinel must print even for a smoke run so the orchestrator can tell an
    # intentional early stop from a crash.
    print(f"done. checkpoints in {out_dir} elapsed_h={elapsed:.2f} "
          f"best_csi={best_csi:.4f} sentinel={'explicit' if done_sentinel else 'loop-exit'}")
    log_f.close()


if __name__ == "__main__":
    main()
