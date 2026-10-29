"""Shared pytest fixtures for the speed-estimation pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.core.config import Config, compose_config, validate_config


@pytest.fixture
def tmp_output_root(tmp_path: Path) -> Path:
    """A per-test output root (nothing touches the repo's real output/)."""
    return tmp_path / "output"


@pytest.fixture
def tiny_config(tmp_path: Path) -> Config:
    """A smoke-sized Config whose outputs and data live under ``tmp_path``."""
    cfg = validate_config(compose_config(["experiment=smoke"]))
    cfg.paths.output_root = str(tmp_path / "output")
    cfg.paths.data_root = str(tmp_path / "data")
    cfg.data.split_manifest = str(tmp_path / "data" / "splits" / "synth.json")
    return cfg


@pytest.fixture
def rng() -> np.random.Generator:
    """A deterministic NumPy RNG for tests."""
    return np.random.default_rng(12345)
