"""Tests for src.run_manager."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import src.run_manager as rm
from src.run_manager import (
    collect_environment,
    create_run,
    load_run,
    set_seed,
    update_metadata,
    write_metadata,
)


def test_run_name_format(tiny_config):
    paths = create_run(tiny_config, now=datetime(2026, 1, 2, 3, 4, 5))
    assert re.match(r"^20260102_030405_smoke_[0-9a-f]{4}$", paths.run_name)


def test_run_dirs_exist(tiny_config):
    paths = create_run(tiny_config, now=datetime(2026, 1, 1))
    assert paths.run_dir.exists()
    assert paths.checkpoints_dir.exists()
    assert paths.reports_dir.exists()
    assert paths.figures_dir.exists()
    assert paths.log_file.parent.exists()


def test_run_name_uniqueness_50_calls(tiny_config):
    now = datetime(2026, 1, 1)
    names = {create_run(tiny_config, now=now).run_name for _ in range(50)}
    assert len(names) == 50


def test_config_yaml_roundtrip(tiny_config):
    paths = create_run(tiny_config, now=datetime(2026, 1, 1))
    loaded = OmegaConf.load(paths.config_path)
    assert loaded.experiment.seed == tiny_config.experiment.seed
    assert loaded.training.epochs == tiny_config.training.epochs
    assert loaded.paths.output_root == tiny_config.paths.output_root


def test_seed_is_reproducible():
    set_seed(42)
    t1 = torch.rand(4).clone()
    n1 = np.random.rand(4).copy()
    set_seed(42)
    assert torch.allclose(t1, torch.rand(4))
    assert np.allclose(n1, np.random.rand(4))


def test_collect_environment_keys():
    env = collect_environment()
    for key in ("python_version", "torch_version", "lightning_version", "platform", "cpu_count"):
        assert key in env
    assert "git_commit" in env
    assert "git_dirty" in env
    assert "argv" in env


def test_metadata_write_and_update(tiny_config):
    paths = create_run(tiny_config, now=datetime(2026, 1, 1))
    write_metadata(paths, {"status": "running", "seed": 42})
    update_metadata(paths, status="completed", duration_s=10)
    meta = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    assert meta["status"] == "completed"
    assert meta["seed"] == 42
    assert meta["duration_s"] == 10


def test_load_run_roundtrip(tiny_config):
    paths = create_run(tiny_config, now=datetime(2026, 1, 1))
    loaded_paths, loaded_cfg = load_run(paths.run_name, Path(tiny_config.paths.output_root))
    assert loaded_paths.run_name == paths.run_name
    assert loaded_cfg.experiment.seed == tiny_config.experiment.seed
    assert loaded_cfg.paths.output_root == tiny_config.paths.output_root


def test_load_run_missing_raises(tiny_config):
    import pytest

    from src.schemas import ConfigError

    with pytest.raises(ConfigError):
        load_run("does_not_exist", Path(tiny_config.paths.output_root))


def test_git_unavailable_returns_none(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise OSError("git not found")

    monkeypatch.setattr(rm.subprocess, "check_output", _boom)
    assert rm._git_commit() is None
    assert rm._git_dirty() is None
