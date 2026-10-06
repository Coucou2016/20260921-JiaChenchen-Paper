#!/usr/bin/env python
"""Train multiscale HydroGeo-SRNO (stage M1/M2) — arbitrary-scale / scale-transfer.

This replaces a stub that did NOT implement its own configuration. The previous
version hard-coded ``batch_size=1``, ``AdamW(lr=1e-4)``, ``FloodLoss()`` and
``use_amp=False``, ran a single pass over the loader with no epoch loop, no
validation and no checkpointing, while the YAML claimed 200 epochs, batch size 2
and AMP. It also trained on the pair [5, 2] that the config called the core
arbitrary-scale test.

What is implemented here:

* the config is the single source of truth (epochs, batch size, AMP, optimizer,
  scheduler, loss weights, seed);
* a full epoch loop with per-epoch validation, best/worst checkpointing by
  seen-pair score, and early stopping;
* a resolution-pair-homogeneous batch sampler, so batch_size > 1 is possible even
  though different pairs have different LR/HR spatial shapes;
* strict train/eval pair separation: every pair in ``test_seen_pairs`` or
  ``test_unseen_pairs`` is asserted disjoint from ``train_pairs`` before any
  training starts, and unseen pairs are never optimised on.

Usage:
    python scripts/train_multiscale.py --config configs/v1_arbitrary_scale.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.wellington_fixed_sr import collate_fixed
from dataset.wellington_multiscale import WellingtonMultiscaleDataset
from engine.checkpoint import save_checkpoint
from engine.evaluator import build_model
from engine.reproducibility import make_generator, seed_everything, seed_worker, write_provenance
from engine.trainer import EarlyStopper, evaluate_loader, train_step
from losses.flood_loss import FloodLoss, MaskedL1Loss


class ResolutionPairBatchSampler:
    """Yield batches whose members all share one (lr_res, hr_res) shape.

    Different resolution pairs produce different spatial sizes, so a plain
    DataLoader cannot stack them. Grouping by pair keeps the batch size the YAML
    asks for without padding or dropping samples.
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool = True, seed: int = 42) -> None:
        self.groups = dataset.groups()
        self.batch_size = max(1, int(batch_size))
        self.shuffle = shuffle
        self.rng = torch.Generator().manual_seed(seed)

    def __iter__(self):
        batches: list[list[int]] = []
        for _, ids in self.groups.items():
            ids = list(ids)
            if self.shuffle and ids:
                perm = torch.randperm(len(ids), generator=self.rng).tolist()
                ids = [ids[i] for i in perm]
            for j in range(0, len(ids), self.batch_size):
                batches.append(ids[j:j + self.batch_size])
        if self.shuffle and batches:
            perm = torch.randperm(len(batches), generator=self.rng).tolist()
            batches = [batches[i] for i in perm]
        yield from batches

    def __len__(self) -> int:
        return sum((len(v) + self.batch_size - 1) // self.batch_size
                   for v in self.groups.values())


def build_loss(cfg: dict):
    lc = cfg.get("loss", {})
    if lc.get("name", "flood") == "masked_l1":
        return MaskedL1Loss(
            wet_threshold=lc.get("wet_threshold", 0.05),
            w_wet=lc.get("w_wet", 0.0),
        )
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


def assert_pairs_disjoint(train_pairs, eval_pair_groups) -> None:
    """Fail loudly if any evaluation pair was also trained on."""
    train = {tuple(int(x) for x in p) for p in train_pairs}
    for label, pairs in eval_pair_groups.items():
        overlap = train & {tuple(int(x) for x in p) for p in pairs}
        if overlap:
            raise SystemExit(
                f"train/eval pair leakage: {sorted(overlap)} appears in both "
                f"train_pairs and {label}. Remove it from one of them."
            )


def evaluate_pairs(model, loader_for_pair, device) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for pair, loader in loader_for_pair.items():
        out[str(pair)] = evaluate_loader(model, loader, device)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/v1_arbitrary_scale.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    train_cfg = cfg.get("train", {})
    seed = int(train_cfg.get("seed", 42))
    seed_everything(seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(0)
        if major < 7 and use_amp:
            print(f"disabling AMP on compute capability {major}.x (unstable FP16)")
            use_amp = False

    ds_cfg = cfg["dataset"]
    train_pairs = [tuple(int(v) for v in p) for p in ds_cfg["train_pairs"]]
    seen_pairs = [tuple(int(v) for v in p) for p in ds_cfg.get("test_seen_pairs", [])]
    unseen_pairs = [tuple(int(v) for v in p) for p in ds_cfg.get("test_unseen_pairs", [])]

    assert_pairs_disjoint(
        train_pairs,
        {"test_seen_pairs": seen_pairs, "test_unseen_pairs": unseen_pairs},
    )

    root = ROOT / ds_cfg.get("root", "dataset")
    common = dict(root=root, target=ds_cfg.get("target", "h_max"),
                  geo_mode=ds_cfg.get("geo_mode", "all"),
                  scenarios=tuple(ds_cfg.get("scenarios", ["20a", "100a"])))

    train_ds = WellingtonMultiscaleDataset(split="train", pairs=train_pairs, **common)

    bs = int(train_cfg.get("batch_size", 1))
    epochs = int(args.epochs or train_cfg.get("epochs", 200))
    out_dir = ROOT / cfg.get("output", {}).get("dir", "outputs/v1_arbitrary_scale")
    out_dir.mkdir(parents=True, exist_ok=True)

    # one validation loader per pair so we can score seen and unseen separately
    val_loaders: dict[tuple[int, int], DataLoader] = {}
    for pair in train_pairs + unseen_pairs:
        vds = WellingtonMultiscaleDataset(split="val", pairs=[pair], **common)
        val_loaders[pair] = DataLoader(
            vds, batch_size=bs, shuffle=False, collate_fn=collate_fixed,
            num_workers=int(train_cfg.get("num_workers", 0)),
        )

    sampler = ResolutionPairBatchSampler(train_ds, batch_size=bs, shuffle=True, seed=seed)
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler, collate_fn=collate_fixed,
        num_workers=int(train_cfg.get("num_workers", 0)),
        worker_init_fn=seed_worker, generator=make_generator(seed),
    )

    model = build_model(cfg.get("model", {}).get("name", "hydrogeo_srno"), cfg).to(device)
    crit = build_loss(cfg)
    opt_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 1e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-5)),
    )
    warmup = int(cfg.get("scheduler", {}).get("warmup_epochs", 0))
    if warmup > 0 and epochs > warmup:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        sched = SequentialLR(
            optimizer,
            [LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
             CosineAnnealingLR(optimizer, T_max=max(epochs - warmup, 1))],
            milestones=[warmup],
        )
    else:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        sched = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    write_provenance(out_dir, cfg, seed, extra={
        "train_pairs": train_pairs,
        "test_seen_pairs": seen_pairs,
        "test_unseen_pairs": unseen_pairs,
        "batch_size": bs,
        "epochs": epochs,
        "amp": use_amp,
        "checkpoint_rule": "best mean CSI_005 over seen pairs; worst seen-pair CSI_005",
    }, root=ROOT)

    es = cfg.get("early_stopping", {})
    stopper = EarlyStopper(patience=int(es.get("patience", 40)), mode=es.get("mode", "max"))
    best_mean, best_worst = -1.0, -1.0
    max_steps = args.max_steps or train_cfg.get("max_steps")
    global_step = 0
    history = out_dir / "history.jsonl"
    print(f"device={device} amp={use_amp} bs={bs} epochs={epochs} "
          f"train_pairs={train_pairs} unseen={unseen_pairs}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        run_loss = []
        for batch in train_loader:
            stats = train_step(batch, model, crit, optimizer, None, device,
                               max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
                               use_amp=use_amp)
            run_loss.append(stats["total"])
            global_step += 1
            if max_steps and global_step >= int(max_steps):
                break
        sched.step()

        seen_metrics = {p: evaluate_loader(model, val_loaders[p], device) for p in train_pairs}
        unseen_metrics = {p: evaluate_loader(model, val_loaders[p], device) for p in unseen_pairs}

        def _csi(m: dict) -> float:
            v = m.get("CSI_005", float("nan"))
            return -1.0 if v != v else float(v)

        seen_csi = [_csi(seen_metrics[p]) for p in train_pairs]
        mean_csi = sum(seen_csi) / max(len(seen_csi), 1)
        worst_csi = min(seen_csi)

        row = {
            "epoch": epoch, "global_step": global_step,
            "loss": sum(run_loss) / max(len(run_loss), 1),
            "sec": time.time() - t0,
            "seen_mean_csi005": mean_csi, "seen_worst_csi005": worst_csi,
            "seen": {f"lr{l}_hr{h}": seen_metrics[(l, h)] for l, h in train_pairs},
            "unseen": {f"lr{l}_hr{h}": unseen_metrics[(l, h)] for l, h in unseen_pairs},
        }
        with history.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        print(f"epoch={epoch}/{epochs} loss={row['loss']:.4f} "
              f"seen_mean_CSI005={mean_csi:.4f} seen_worst={worst_csi:.4f} "
              f"sec={row['sec']:.0f}")

        save_checkpoint(out_dir / "last_multiscale.pt", model, optimizer, epoch,
                        {"seen_mean_csi005": mean_csi}, cfg)
        if mean_csi > best_mean:
            best_mean = mean_csi
            save_checkpoint(out_dir / "best_mean.pt", model, optimizer, epoch,
                            {"seen_mean_csi005": mean_csi}, cfg)
        if worst_csi > best_worst:
            best_worst = worst_csi
            save_checkpoint(out_dir / "best_worst.pt", model, optimizer, epoch,
                            {"seen_worst_csi005": worst_csi}, cfg)

        if max_steps and global_step >= int(max_steps):
            print("reached max_steps; stopping (smoke mode)")
            break
        if stopper.step(mean_csi):
            print(f"early stop at epoch {epoch}")
            break

    (out_dir / "best_summary.json").write_text(json.dumps(
        {"best_mean_csi005": best_mean, "best_worst_csi005": best_worst,
         "epochs_run": epoch}, indent=2), encoding="utf-8")
    print(f"done. checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
