"""Reproducibility helpers: full seeding and a per-run provenance record.

``set_seed`` previously set only ``random`` and ``torch``. NumPy was untouched
and DataLoader workers were not reseeded, so two nominally identical runs could
diverge. ``seed_everything`` closes both gaps and optionally forces deterministic
algorithms.

``write_provenance`` records the seed, the resolved config digest, the git commit
and the environment into every run directory, so any number in the report can be
traced back to the exact run that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG that can influence a run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn so workers are reproducible."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root), capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_provenance(
    out_dir: str | Path,
    cfg: dict,
    seed: int,
    extra: dict | None = None,
    root: str | Path | None = None,
) -> Path:
    """Write provenance.json into a run directory and return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(root) if root is not None else out_dir
    cfg_text = json.dumps(cfg, sort_keys=True, default=str)
    record = {
        "seed": int(seed),
        "config_sha256": _sha256_text(cfg_text),
        "git_commit": _git_commit(root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if extra:
        record.update(extra)
    path = out_dir / "provenance.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
