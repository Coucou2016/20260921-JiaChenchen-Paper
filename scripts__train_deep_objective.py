#!/usr/bin/env python
"""Experiment B (the E7 plan): deep-water objective rework, fine-tuned on the frozen V0.

Three changes relative to the existing deep fine-tune (``run_deep_finetune.py``):

1. **Cosine-decaying shallow weights.**  The deep term keeps a CONSTANT weight
   (default 0.10) while the shallow-side weights ``w_depth`` and ``w_log``
   cosine-decay to a floor over the horizon, so the deep term's *share* of the
   objective rises instead of being pinned at a fixed ratio.

2. **Continuous per-pixel deep weight.**  The binary ``t >= 1 m`` mask (and the
   per-tile quantile that collapses onto shallow water on mostly-dry tiles) is
   replaced by a linear ramp in true depth, so deep pixels always receive weight
   (see ``FloodLoss(continuous_deep=True, deep_scale=...)``).

3. **Domain-pooled selection.**  The reported/selected statistic is the
   domain-pooled wet RMSE on the FULL 182-tile validation split (``RMSE_wet_domain``),
   not the tile-macro average and not a 60-batch subset.

Everything else (split, seed, evaluator, frozen parent) is unchanged so the arm is
comparable with the existing ``w00``/``w01`` fine-tunes.

Usage
-----
    python -u scripts/train_deep_objective.py --smoke-steps 30 --epochs 1 \
        --out_dir outputs/deep_objective/_smoke
    python -u scripts/train_deep_objective.py --epochs 12 \
        --out_dir outputs/deep_objective/E7 --w_deep 0.10 --decay_floor 0.25
    python -u scripts/train_deep_objective.py --epochs 12 \
        --out_dir outputs/deep_objective/E7ctrl --w_deep 0.0 --decay_floor 0.25
"""
from __future__ import annotations

import argparse
import json
import math
import os
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
from engine.checkpoint import load_checkpoint, save_checkpoint
from engine.evaluator import build_model
from engine.reproducibility import make_generator, seed_everything, seed_worker, write_provenance
from engine.trainer import EarlyStopper, move_to_device, train_step
from losses.flood_loss import FloodLoss
from metrics.aggregation import FloodMetricAccumulator, average_metrics
from metrics.flood_metrics import compute_flood_metrics

FROZEN = ROOT / "outputs" / "v0_10m2m_hmax" / "last.pt"
CONFIG = "configs/v0_10m2m_hmax.yaml"


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


@torch.no_grad()
def evaluate_both(model, loader, device, max_batches=None):
    """Tile-macro metrics AND domain-pooled metrics in one pass."""
    model.eval()
    tiles = []
    acc = FloodMetricAccumulator()
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = move_to_device(batch, device)
        out = model(lr=batch["lr"], lr_valid=batch["lr_valid"],
                    static_cont=batch["static_cont"], landuse=batch["landuse"],
                    lr_res=batch["lr_res"], hr_res=batch["hr_res"])
        pred, tgt, mask = out["depth"], batch["hr"], batch["mask"]
        tiles.append(compute_flood_metrics(pred, tgt, mask))
        acc.update(pred, tgt, mask)
    macro = average_metrics(tiles)
    pooled = acc.compute()
    merged = dict(macro)
    merged.update(pooled)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--resume", default=str(FROZEN), help="parent checkpoint (frozen V0)")
    ap.add_argument("--out_dir", default="outputs/deep_objective/E7")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--w_deep", type=float, default=0.10)
    ap.add_argument("--deep_threshold", type=float, default=1.0)
    ap.add_argument("--deep_scale", type=float, default=4.0,
                    help="continuous ramp scale: weight=1 at threshold, +1 per deep_scale metres")
    ap.add_argument("--continuous_deep", action="store_true", default=True)
    ap.add_argument("--binary_deep", dest="continuous_deep", action="store_false")
    ap.add_argument("--decay_floor", type=float, default=0.25,
                    help="shallow weights decay to floor*base over the horizon (1.0 disables)")
    ap.add_argument("--smoke-steps", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--snapshots", action="store_true", default=True)
    ap.add_argument("--eval_max_batches", type=int, default=None, help="default: full val")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--wait-pid", type=int, default=None)
    ap.add_argument("--timeout-h", type=float, default=10.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    out_dir = ROOT / args.out_dir
    if args.dry_run:
        print(f"DRY-RUN {out_dir} w_deep={args.w_deep} cont={args.continuous_deep} "
              f"floor={args.decay_floor} epochs={args.epochs} lr={args.lr}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train.log", "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)

    def log(msg):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

    if args.wait_pid:
        import subprocess
        log(f"waiting for pid={args.wait_pid} (single GPU)")
        t0 = time.time()
        while str(args.wait_pid) in subprocess.run(
                ["tasklist", "/FI", f"PID eq {args.wait_pid}", "/NH"],
                capture_output=True, text=True, errors="ignore").stdout:
            if (time.time() - t0) / 3600 > args.timeout_h:
                log("timeout; aborting")
                return
            time.sleep(30)
        log(f"pid gone after {(time.time()-t0)/60:.1f} min")

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        raise SystemExit("CUDA_VISIBLE_DEVICES is empty; refusing to run on CPU.")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = cfg["train"]
    ds_cfg = cfg["dataset"]
    root = ROOT / ds_cfg.get("root", "dataset")
    scenarios = tuple(ds_cfg.get("scenarios", ["20a", "100a"]))
    kw = dict(lr_res=int(ds_cfg.get("lr_res", 10)), hr_res=int(ds_cfg.get("hr_res", 2)),
              target=ds_cfg.get("target", "h_max"), geo_mode=ds_cfg.get("geo_mode", "all"),
              scenarios=scenarios)
    train_ds = WellingtonFixedSRDataset(root=root, split="train", **kw)
    val_ds = WellingtonFixedSRDataset(root=root, split="val", **kw)
    bs = 1
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0,
                              collate_fn=collate_fixed, pin_memory=(device.type == "cuda"),
                              worker_init_fn=seed_worker, generator=make_generator(args.seed))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0,
                            collate_fn=collate_fixed, pin_memory=(device.type == "cuda"))

    lc = cfg.get("loss", {})
    base_w_depth = float(lc.get("w_depth", 1.0))
    base_w_log = float(lc.get("w_log", 0.20))
    criterion = FloodLoss(
        w_depth=base_w_depth, w_wet=lc.get("w_wet", 0.30), w_log=base_w_log,
        w_boundary=lc.get("w_boundary", 0.10), w_extreme=lc.get("w_extreme", 0.10),
        wet_threshold=lc.get("wet_threshold", 0.05),
        depth_ref=cfg["model"].get("depth_ref", 0.10),
        w_deep=args.w_deep, deep_threshold=args.deep_threshold,
        deep_quantile=None, deep_delta=float(lc.get("deep_delta", 5.0)),
        continuous_deep=bool(args.continuous_deep) and args.w_deep > 0,
        deep_scale=args.deep_scale,
    )
    model = build_model(cfg["model"]["name"], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=float(cfg.get("optimizer", {}).get("weight_decay", 1e-5)))
    ck = load_checkpoint(args.resume, model, map_location=device)
    start_epoch = int(ck.get("epoch", 0)) + 1
    for pg in optimizer.param_groups:
        pg["lr"] = args.lr
    log(f"resumed {args.resume} at epoch {start_epoch}; lr={args.lr:.2e}")

    sched = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    stopper = EarlyStopper(patience=int(args.patience or args.epochs), mode="max")

    write_provenance(out_dir, cfg, args.seed, extra={
        "experiment": "B_deep_objective_E7",
        "parent": str(Path(args.resume).name),
        "w_deep": args.w_deep, "deep_threshold": args.deep_threshold,
        "deep_scale": args.deep_scale, "continuous_deep": bool(args.continuous_deep),
        "decay_floor": args.decay_floor, "lr": args.lr, "horizon": args.epochs,
        "selection": "min domain-pooled RMSE_wet on FULL val (guarded by CSI_005)",
    }, root=ROOT)
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    history_path = out_dir / "history.jsonl"
    t_run = time.time()
    best_pooled_rmse = float("inf")
    best_parent_csi = float(ck.get("metrics", {}).get("CSI_005", float("nan")))
    global_step = 0
    log_every = int(train_cfg.get("log_every_steps", 50))
    snap_dir = out_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        # ---- cosine decay of the shallow-side weights, deep weight held constant
        prog = (epoch - start_epoch) / max(args.epochs - 1, 1)
        factor = args.decay_floor + (1.0 - args.decay_floor) * 0.5 * (1 + math.cos(math.pi * prog))
        criterion.w_depth = base_w_depth * factor
        criterion.w_log = base_w_log * factor
        t_ep = time.time()
        model.train()
        running = []
        comp = {}
        for step_i, batch in enumerate(train_loader, start=1):
            stats = train_step(batch, model, criterion, optimizer, None, device,
                               max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
                               use_amp=False)
            running.append(stats["total"])
            for k, v in stats.items():
                if k != "total":
                    comp.setdefault(k, []).append(v)
            global_step += 1
            if log_every and step_i % log_every == 0:
                log(f"  ep{epoch} step={step_i}/{len(train_loader)} loss={stats['total']:.4f} "
                    f"l_deep={stats.get('l_deep', 0):.4f} l_depth={stats.get('l_depth', 0):.4f} "
                    f"w_depth={criterion.w_depth:.3f} w_log={criterion.w_log:.3f}")
            if args.max_steps is not None and step_i >= args.max_steps:
                break
            if args.smoke_steps is not None and global_step >= args.smoke_steps:
                break
        sched.step()

        val = evaluate_both(model, val_loader, device, max_batches=args.eval_max_batches)
        mean_loss = sum(running) / max(len(running), 1)
        pooled_rmse = val.get("RMSE_wet_domain", float("nan"))
        macro_csi = val.get("CSI_005", float("nan"))
        ep_sec = time.time() - t_ep
        log(f"epoch={epoch} step={global_step} loss={mean_loss:.4f} "
            f"w_depth={criterion.w_depth:.3f} w_log={criterion.w_log:.3f} "
            f"w_deep_share={criterion.w_deep/(criterion.w_deep+criterion.w_depth+criterion.w_log+criterion.w_wet+criterion.w_boundary+criterion.w_extreme):.3f} "
            f"| pooled RMSE_wet={pooled_rmse:.4f} MAE_wet={val.get('MAE_wet_domain'):.4f} "
            f"bias>=1m={val.get('Bias_1m_domain'):.4f} | macro CSI_005={macro_csi:.4f} "
            f"CSI_100={val.get('CSI_100'):.4f} | sec={ep_sec:.0f}")

        row = {"epoch": epoch, "global_step": global_step, "loss": mean_loss,
               "sec": ep_sec, "w_depth": criterion.w_depth, "w_log": criterion.w_log,
               "w_deep": criterion.w_deep, "decay_factor": factor,
               "tr_l_deep": float(sum(comp.get("l_deep", [0])) / max(len(comp.get("l_deep", [1])), 1)),
               **val}
        with history_path.open("a", encoding="utf-8") as hf:
            hf.write(json.dumps(row) + "\n")

        save_checkpoint(snap_dir / f"ep{epoch:04d}.pt", model, None, epoch, val, cfg)
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, val, cfg)
        # selection: lowest domain-pooled wet RMSE
        if pooled_rmse == pooled_rmse and pooled_rmse < best_pooled_rmse:
            best_pooled_rmse = pooled_rmse
            save_checkpoint(out_dir / "best_pooled_rmse.pt", model, optimizer, epoch, val, cfg)
            (out_dir / "best_pooled_summary.json").write_text(json.dumps({
                "epoch": epoch, "RMSE_wet_domain": pooled_rmse,
                "MAE_wet_domain": val.get("MAE_wet_domain"),
                "Bias_1m_domain": val.get("Bias_1m_domain"),
                "Bias_3m_domain": val.get("Bias_3m_domain"),
                "CSI_005": macro_csi, "CSI_100": val.get("CSI_100"),
                "VolumeRelativeError": val.get("VolumeRelativeError"),
                "parent_CSI_005": best_parent_csi}, indent=2), encoding="utf-8")
            log(f"  ↑ new best pooled RMSE_wet={best_pooled_rmse:.4f} @ep{epoch}")
        (out_dir / "best_summary.json").write_text(json.dumps({
            "best_pooled_rmse": best_pooled_rmse, "last_epoch": epoch,
            "elapsed_sec": time.time() - t_run,
            "w_depth": criterion.w_depth, "w_log": criterion.w_log,
            "w_deep": criterion.w_deep}, indent=2), encoding="utf-8")

        stop = False
        if args.smoke_steps is not None and global_step >= args.smoke_steps:
            stop = True
        if not stop and stopper.step(macro_csi if macro_csi == macro_csi else -1.0):
            log(f"early stop at epoch {epoch}")
            stop = True
        if stop:
            break

    log(f"done. checkpoints in {out_dir} elapsed_h={(time.time()-t_run)/3600:.2f} "
        f"best_pooled_rmse={best_pooled_rmse:.4f}")
    log_f.close()


if __name__ == "__main__":
    main()
