"""Run directories, metadata and seeding."""

from __future__ import annotations

import json
import os
import platform
import re
import secrets
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import lightning as L
import torch
from omegaconf import OmegaConf

from src.config import Config, validate_config
from src.logging_utils import get_logger
from src.schemas import ConfigError, RunPaths

logger = get_logger(__name__)

_RUN_NAME_RE = re.compile(r"^\d{8}_\d{6}_[A-Za-z0-9_-]+_[0-9a-f]{4}$")
_MAX_COLLISION_ATTEMPTS = 1000


def _sanitize(name: str) -> str:
    """Sanitize an experiment name to ``[A-Za-z0-9_-]``."""
    return re.sub(r"[^A-Za-z0-9_-]", "-", name)


def _build_run_paths(run_name: str, output_root: Path) -> RunPaths:
    """Derive all RunPaths locations for a run name without creating anything."""
    runs_dir = output_root / "runs"
    run_dir = runs_dir / run_name
    return RunPaths(
        run_name=run_name,
        run_dir=run_dir,
        checkpoints_dir=run_dir / "checkpoints",
        reports_dir=run_dir / "reports",
        figures_dir=output_root / "figures" / run_name,
        log_file=output_root / "logs" / f"{run_name}.log",
        config_path=run_dir / "config.yaml",
        metadata_path=run_dir / "metadata.json",
        metrics_path=run_dir / "metrics.json",
        predictions_path=run_dir / "predictions.csv",
        split_path=run_dir / "split.json",
    )


def create_run(cfg: Config, now: datetime | None = None) -> RunPaths:
    """Create a new run directory tree and return its paths.

    Never overwrites an existing run: the 4-hex suffix is regenerated on
    collision.  Writes the resolved ``config.yaml`` into the run directory.
    """
    now = now or datetime.now()
    output_root = Path(cfg.paths.output_root)
    runs_dir = output_root / "runs"
    exp_name = _sanitize(cfg.experiment.name)
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    paths: RunPaths | None = None
    for _ in range(_MAX_COLLISION_ATTEMPTS):
        suffix = secrets.token_hex(2)  # 4 hex characters
        run_name = f"{timestamp}_{exp_name}_{suffix}"
        candidate = _build_run_paths(run_name, output_root)
        try:
            candidate.run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        paths = candidate
        break

    if paths is None:
        raise ConfigError(
            f"Could not create a unique run directory under {runs_dir} "
            f"after {_MAX_COLLISION_ATTEMPTS} attempts"
        )

    # Create required subdirectories.
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=False)
    paths.reports_dir.mkdir(parents=True, exist_ok=False)
    paths.figures_dir.mkdir(parents=True, exist_ok=False)
    paths.log_file.parent.mkdir(parents=True, exist_ok=True)

    # Write the resolved config for reproducibility.
    OmegaConf.save(OmegaConf.structured(cfg), paths.config_path)

    logger.info("event=create_run run_name=%s run_dir=%s", paths.run_name, paths.run_dir)
    return paths


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed torch/numpy/python RNGs. ``deterministic`` enables torch determinism."""
    L.seed_everything(seed=seed, workers=True)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    logger.info(
        "event=set_seed seed=%d deterministic=%s determinism_guaranteed=false",
        seed,
        deterministic,
    )


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip() or None
    except Exception:  # noqa: BLE001 - git may be unavailable
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
        return bool(out.decode().strip())
    except Exception:  # noqa: BLE001 - git may be unavailable
        return None


def collect_environment() -> dict:
    """Collect runtime environment facts (versions, GPU, git, argv)."""
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "lightning_version": L.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "argv": list(sys.argv),
    }


def write_metadata(paths: RunPaths, meta: dict) -> None:
    """Write metadata JSON to ``paths.metadata_path``."""
    paths.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with paths.metadata_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def update_metadata(paths: RunPaths, **fields) -> None:
    """Read-modify-write metadata JSON, merging ``fields``."""
    meta = {}
    if paths.metadata_path.exists():
        with paths.metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    meta.update(fields)
    write_metadata(paths, meta)


def load_run(run_name: str, output_root: Path) -> tuple[RunPaths, Config]:
    """Load a previously created run's paths and validated config."""
    output_root = Path(output_root)
    paths = _build_run_paths(run_name, output_root)
    if not paths.run_dir.exists():
        raise ConfigError(f"Run {run_name!r} not found under {output_root}")
    if not paths.config_path.exists():
        raise ConfigError(f"Run {run_name!r} is missing config.yaml")
    dict_cfg = OmegaConf.load(paths.config_path)
    cfg = validate_config(dict_cfg)
    return paths, cfg
