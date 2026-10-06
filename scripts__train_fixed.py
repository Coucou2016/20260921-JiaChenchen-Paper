#!/usr/bin/env python
"""Train fixed-scale HydroGeo-SRNO (default 10 m → 2 m, h_max).

Supports --resume from outputs/.../last.pt. Logs to train.log + history.jsonl.
Checkpoints by val CSI@0.05 and wet RMSE (not PSNR).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.wellington_fixed_sr import WellingtonFixedSRDataset, collate_fixed
from engine.checkpoint import load_checkpoint, save_checkpoint
from engine.evaluator import build_model
from engine.trainer import EarlyStopper, evaluate_loader, train_step
from losses.flood_loss import FloodLoss, MaskedL1Loss


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loss(cfg: dict):
    lc = cfg.get("loss", {})
    if lc.get("name", "flood") == "masked_l1":
        return MaskedL1Loss()
    return FloodLoss(
        w_depth=lc.get("w_depth", 1.0),
        w_wet=lc.get("w_wet", 0.30),
        w_log=lc.get("w_log", 0.20),
        w_boundary=lc.get("w_boundary", 0.10),
        w_extreme=lc.get("w_extreme", 0.10),
        wet_threshold=lc.get("wet_threshold", 0.05),
        depth_ref=cfg.get("model", {}).get("depth_ref", 0.10),
        extreme_quantile=lc.get("extreme_quantile", 0.95),
        w_deep=lc.get("w_deep", 0.0),
        deep_threshold=lc.get("deep_threshold", 1.0),
        deep_quantile=lc.get("deep_quantile", None),
        deep_delta=lc.get("deep_delta", 5.0),
    )


def apply_ablation(cfg: dict, ablation_path: Path | None, name: str | None) -> dict:
    if not name:
        return cfg
    assert ablation_path is not None
    abl = load_config(ablation_path)
    block = abl["ablations"][name]
    cfg = json.loads(json.dumps(cfg))
    if "geo_mode" in block:
        cfg.setdefault("dataset", {})["geo_mode"] = block["geo_mode"]
    if "model" in block:
        for k, v in block["model"].items():
            if isinstance(v, dict):
                cfg.setdefault("model", {}).setdefault(k, {}).update(v)
            else:
                cfg.setdefault("model", {})[k] = v
    if "loss" in block:
        cfg.setdefault("loss", {}).update(block["loss"])
    gm = cfg["dataset"].get("geo_mode", "all")
    geo = cfg.setdefault("model", {}).setdefault("geo_encoder", {})
    if gm == "lr_only":
        geo["in_cont"] = 0
        geo["use_landuse"] = False
    elif gm == "dem":
        geo["in_cont"] = 1
        geo["use_landuse"] = False
    elif gm == "dem_building":
        geo["in_cont"] = 2
        geo["use_landuse"] = False
    elif gm == "topography":
        geo["in_cont"] = 8
        geo["use_landuse"] = False
    return cfg


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/v0_10m2m_hmax.yaml")
    parser.add_argument("--ablation", type=str, default=None, help="A0..A7")
    parser.add_argument("--ablation_config", type=str, default="configs/v0_10m2m_hmax_ablation.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="path to last.pt (default: out_dir/last.pt if exists)")
    parser.add_argument("--batch_size", type=int, default=None)
    # Loss overrides for deep-water sweeps (config stays the single source of truth otherwise)
    parser.add_argument("--w_deep", type=float, default=None)
    parser.add_argument("--deep_delta", type=float, default=None)
    parser.add_argument("--deep_threshold", type=float, default=None)
    parser.add_argument("--w_extreme", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--sched_epochs", type=int, default=None,
                        help="fine-tune: build a fresh cosine horizon over N epochs and do NOT "
                             "fast-forward the scheduler on resume (use together with --resume). "
                             "Without this, resuming at ep180 with --epochs 175 fast-forwards 180 "
                             "steps and leaves lr ~2e-7, so the model barely moves.")
    parser.add_argument("--lr", type=float, default=None,
                        help="override optimizer.lr (also the fine-tune cosine start lr)")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--eval_max_batches", type=int, default=None,
                        help="limit val batches per epoch (faster sweep); None = full val")
    parser.add_argument("--tag", type=str, default=None, help="suffix for the output dir, e.g. D1")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    cfg = apply_ablation(cfg, ROOT / args.ablation_config if args.ablation else None, args.ablation)

    # Apply sweep overrides
    if args.w_deep is not None:
        cfg.setdefault("loss", {})["w_deep"] = args.w_deep
    if args.deep_delta is not None:
        cfg.setdefault("loss", {})["deep_delta"] = args.deep_delta
    if args.deep_threshold is not None:
        cfg.setdefault("loss", {})["deep_threshold"] = args.deep_threshold
    if args.w_extreme is not None:
        cfg.setdefault("loss", {})["w_extreme"] = args.w_extreme
    if args.lr is not None:
        cfg.setdefault("optimizer", {})["lr"] = args.lr
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.num_workers is not None:
        cfg.setdefault("train", {})["num_workers"] = args.num_workers
    if args.tag is not None:
        base = cfg.get("output", {}).get("dir", "outputs/run")
        cfg.setdefault("output", {})["dir"] = f"{base}_{args.tag}"
    if args.out_dir is not None:
        cfg.setdefault("output", {})["dir"] = args.out_dir
    if args.patience is not None:
        cfg.setdefault("early_stopping", {})["patience"] = args.patience
    if args.eval_max_batches is not None:
        cfg.setdefault("train", {})["eval_max_batches"] = args.eval_max_batches

    train_cfg = cfg.get("train", {})
    set_seed(int(train_cfg.get("seed", 42)))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # A stray CUDA_VISIBLE_DEVICES="" would silently make training run on CPU (~50x slower).
    # Fail loudly instead, unless the user explicitly asked for CPU.
    if not args.device and os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        raise SystemExit(
            "CUDA_VISIBLE_DEVICES is empty: training would silently run on CPU. "
            "Unset it, or pass --device cpu if that is really intended.")

    # Maxwell / Pascal without reliable FP16: force amp off if sm < 7
    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(0)
        if major < 7 and use_amp:
            print(f"disabling AMP on compute capability {major}.{minor} (unstable FP16)")
            use_amp = False
            train_cfg["amp"] = False

    bs = int(args.batch_size or train_cfg.get("batch_size", 2))
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb < 6 and bs > 1:
            print(f"VRAM {vram_gb:.1f} GB < 6: forcing batch_size=1 (was {bs})")
            bs = 1
    train_cfg["batch_size"] = bs

    out_dir = ROOT / cfg.get("output", {}).get("dir", "outputs/run")
    if args.ablation:
        out_dir = out_dir.parent / f"{out_dir.name}_{args.ablation}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    log_f = open(log_path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)

    print(f"device={device} batch_size={bs} amp={use_amp}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")

    ds_cfg = cfg["dataset"]
    root = ROOT / ds_cfg.get("root", "dataset")
    geo_mode = ds_cfg.get("geo_mode", "all")

    train_ds = WellingtonFixedSRDataset(
        root=root,
        split="train",
        lr_res=int(ds_cfg.get("lr_res", 10)),
        hr_res=int(ds_cfg.get("hr_res", 2)),
        target=ds_cfg.get("target", "h_max"),
        geo_mode=geo_mode,
        scenarios=tuple(ds_cfg.get("scenarios", ["20a", "100a"])),
    )
    val_ds = WellingtonFixedSRDataset(
        root=root,
        split="val",
        lr_res=int(ds_cfg.get("lr_res", 10)),
        hr_res=int(ds_cfg.get("hr_res", 2)),
        target=ds_cfg.get("target", "h_max"),
        geo_mode=geo_mode,
        scenarios=tuple(ds_cfg.get("scenarios", ["20a", "100a"])),
    )
    print(f"train={len(train_ds)} val={len(val_ds)} geo_mode={geo_mode}")

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_fixed,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_fixed,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg.get("model", {}).get("name", "hydrogeo_srno"), cfg).to(device)
    criterion = build_loss(cfg)
    opt_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 1e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-5)),
    )

    epochs = int(train_cfg.get("epochs", 300))
    warmup = int(cfg.get("scheduler", {}).get("warmup_epochs", 0))
    if warmup > 0 and epochs > warmup:
        sched = SequentialLR(
            optimizer,
            [
                LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
                CosineAnnealingLR(optimizer, T_max=max(epochs - warmup, 1)),
            ],
            milestones=[warmup],
        )
    else:
        sched = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    start_epoch = 1
    best_csi = -1.0
    best_rmse_wet = float("inf")
    best_deep_csi = float("-inf")
    global_step = 0
    resume_path = args.resume
    if resume_path is None and (out_dir / "last.pt").exists():
        resume_path = str(out_dir / "last.pt")

    # Fine-tune mode: a caller that resumes AND asks for a scheduler reset. The reset must be
    # applied *after* load_checkpoint(), because optimizer.load_state_dict() restores the
    # checkpoint's own lr (3.7e-5 at ep180) and would otherwise clobber the configured base lr.
    sched_reset = args.sched_epochs is not None
    horizon = int(args.sched_epochs) if sched_reset else 0

    if resume_path:
        print(f"resuming from {resume_path}")
        ckpt = load_checkpoint(resume_path, model, optimizer, map_location=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if sched_reset:
            # Derive the stopping epoch instead of trusting the caller to reconcile --epochs
            # (an absolute epoch index) with the horizon length.
            epochs = start_epoch + horizon - 1
            base_lr = float(opt_cfg.get("lr", 1e-4))
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr
                pg["initial_lr"] = base_lr
            sched = CosineAnnealingLR(optimizer, T_max=max(horizon, 1))
            print(f"fine-tune: lr reset to {base_lr:.3e}, cosine horizon over {horizon} epochs "
                  f"(epochs {start_epoch}..{epochs}, no scheduler fast-forward)")
        metrics = ckpt.get("metrics") or {}
        best_csi = float(metrics.get("CSI_005", best_csi)) if metrics.get("CSI_005") == metrics.get("CSI_005") else best_csi
        # Prefer explicit best trackers if present
        if (out_dir / "best_csi.pt").exists():
            try:
                b = torch.load(out_dir / "best_csi.pt", map_location="cpu", weights_only=False)
                best_csi = float((b.get("metrics") or {}).get("CSI_005", best_csi))
            except Exception:
                pass
        if (out_dir / "best_rmse_wet.pt").exists():
            try:
                b = torch.load(out_dir / "best_rmse_wet.pt", map_location="cpu", weights_only=False)
                best_rmse_wet = float((b.get("metrics") or {}).get("RMSE_wet", best_rmse_wet))
            except Exception:
                pass
        # Fast-forward scheduler
        if not sched_reset:
            for _ in range(start_epoch - 1):
                sched.step()
        print(f"resume at epoch={start_epoch} best_csi={best_csi:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.3e}")

    es_cfg = cfg.get("early_stopping", {})
    stopper = EarlyStopper(patience=int(es_cfg.get("patience", 40)), mode=es_cfg.get("mode", "max"))
    # Warm stopper with current best so resume does not reset patience unfairly
    if best_csi > 0:
        stopper.best = best_csi

    max_steps = train_cfg.get("max_steps")
    log_every = int(train_cfg.get("log_every_steps", 50))
    history_path = out_dir / "history.jsonl"

    print(f"training epochs {start_epoch}..{epochs} → {out_dir}")
    print(f"early_stopping: metric={es_cfg.get('metric', 'CSI_005')} "
          f"mode={es_cfg.get('mode', 'max')} patience={stopper.patience}")
    t_run = time.time()

    for epoch in range(start_epoch, epochs + 1):
        t_ep = time.time()
        model.train()
        running = []
        comp_acc: dict[str, list[float]] = {}
        for step_i, batch in enumerate(train_loader, start=1):
            stats = train_step(
                batch,
                model,
                criterion,
                optimizer,
                scaler=None,
                device=device,
                max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
                use_amp=use_amp,
            )
            running.append(stats["total"])
            for key, val in stats.items():
                if key != "total":
                    comp_acc.setdefault(key, []).append(val)
            global_step += 1
            if log_every and step_i % log_every == 0:
                print(
                    f"  epoch={epoch} step={step_i}/{len(train_loader)} "
                    f"loss={stats['total']:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )
            if max_steps is not None and global_step >= int(max_steps):
                break
        sched.step()

        eval_batches = train_cfg.get("eval_max_batches")
        val_metrics = evaluate_loader(model, val_loader, device, max_batches=eval_batches)
        mean_loss = sum(running) / max(len(running), 1)
        csi = val_metrics.get("CSI_005", float("nan"))
        rmse_wet = val_metrics.get("RMSE_wet", float("nan"))
        ep_sec = time.time() - t_ep
        print(
            f"epoch={epoch}/{epochs} step={global_step} loss={mean_loss:.4f} "
            f"CSI_005={csi:.4f} RMSE_wet={rmse_wet:.4f} "
            f"MAE_wet={val_metrics.get('MAE_wet', float('nan')):.4f} "
            f"sec={ep_sec:.1f}"
        )

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "loss": mean_loss,
            "sec": ep_sec,
            **{f"tr_{k}": float(sum(v) / max(len(v), 1)) for k, v in comp_acc.items()},
            **val_metrics,
        }
        with history_path.open("a", encoding="utf-8") as hf:
            hf.write(json.dumps(row) + "\n")

        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, val_metrics, cfg)
        if csi == csi and csi >= best_csi:
            best_csi = csi
            save_checkpoint(out_dir / "best_csi.pt", model, optimizer, epoch, val_metrics, cfg)
            print(f"  ↑ new best CSI_005={best_csi:.4f}")
        if rmse_wet == rmse_wet and rmse_wet <= best_rmse_wet:
            best_rmse_wet = rmse_wet
            save_checkpoint(out_dir / "best_rmse_wet.pt", model, optimizer, epoch, val_metrics, cfg)
            print(f"  ↑ new best RMSE_wet={best_rmse_wet:.4f}")

        # Deep-water tracker. A fine-tune aims at the deep tail, which is NOT what CSI_005
        # selects for: best_csi.pt can easily land on an epoch that is overall-nice but
        # deep-water-neutral. Track CSI_100 separately so the deep-tail checkpoint is always
        # available (also updates best_deep_summary.json with its provenance).
        deep_csi = val_metrics.get("CSI_100", float("nan"))
        if deep_csi == deep_csi and deep_csi > best_deep_csi:
            best_deep_csi = deep_csi
            save_checkpoint(out_dir / "best_deep_csi.pt", model, optimizer, epoch, val_metrics, cfg)
            (out_dir / "best_deep_summary.json").write_text(
                json.dumps({"best_deep_csi": best_deep_csi, "epoch": epoch,
                            "CSI_005": csi, "CSI_030": val_metrics.get("CSI_030"),
                            "CSI_100": deep_csi,
                            "PeakDepthError": val_metrics.get("PeakDepthError"),
                            "RMSE_wet": rmse_wet}, indent=2), encoding="utf-8")
            print(f"  ↑ new best CSI_100={best_deep_csi:.4f}")

        (out_dir / "metrics_last.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
        (out_dir / "best_summary.json").write_text(
            json.dumps(
                {
                    "best_csi": best_csi,
                    "best_rmse_wet": best_rmse_wet,
                    "last_epoch": epoch,
                    "elapsed_sec": time.time() - t_run,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if max_steps is not None and global_step >= int(max_steps):
            print("reached max_steps; stopping (smoke mode)")
            break
        if stopper.step(csi if csi == csi else -1.0):
            print(f"early stop at epoch {epoch} (patience={stopper.patience})")
            break

    print(f"done. checkpoints in {out_dir} elapsed_h={(time.time()-t_run)/3600:.2f}")
    log_f.close()


if __name__ == "__main__":
    main()
