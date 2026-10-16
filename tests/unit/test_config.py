"""Tests for src.core.config compose/validate."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from src.core.config import Config, compose_config, validate_config
from src.core.schemas import ConfigError


def test_compose_default():
    cfg = compose_config()
    assert cfg.experiment.seed == 42
    assert cfg.experiment.name == "baseline"


def test_compose_smoke_overrides():
    cfg = compose_config(["experiment=smoke"])
    assert cfg.experiment.name == "smoke"
    assert cfg.training.epochs == 2
    assert cfg.data.synthetic.num_sequences == 6
    assert cfg.tracking.enabled is False


def test_compose_override_learning_rate():
    cfg = compose_config(["training.learning_rate=0.001"])
    assert cfg.training.learning_rate == 0.001


def test_compose_model_defaults():
    cfg = compose_config()
    assert cfg.model.encoder.name == "backbone"
    assert cfg.model.encoder.backbone == "resnet18"
    assert cfg.model.temporal.name == "convlstm"
    assert cfg.model.head.activation == "none"
    assert cfg.model.head.hidden_dims == []


def test_compose_encoder_temporal_head():
    cfg = compose_config(["encoder=resnet50", "temporal=gru", "head=mlp"])
    assert cfg.model.encoder.backbone == "resnet50"
    assert cfg.model.temporal.name == "gru"
    assert cfg.model.head.activation == "relu"
    assert cfg.model.head.hidden_dims == [128, 64]


def test_validate_returns_typed_config():
    cfg = validate_config(compose_config())
    assert isinstance(cfg, Config)
    assert cfg.experiment.seed == 42
    assert cfg.data.split_manifest.endswith(f"seed{cfg.experiment.seed}.json")


def test_validate_unknown_key_raises():
    with pytest.raises(ConfigError):
        validate_config(OmegaConf.create({"training": {"batch_sze": 5}}))


def test_validate_wrong_type_raises():
    with pytest.raises(ConfigError):
        validate_config(OmegaConf.create({"training": {"batch_size": "not_an_int"}}))


def test_validate_enter_gt_exit_raises():
    cfg = OmegaConf.create(
        {"postprocessing": {"stationary": {"enter_threshold_mph": 3.0, "exit_threshold_mph": 2.5}}}
    )
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_validate_persistence_lt_one_raises():
    cfg = OmegaConf.create({"postprocessing": {"stationary": {"enter_persistence_frames": 0}}})
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_validate_unknown_smoothing_raises():
    cfg = OmegaConf.create({"postprocessing": {"smoothing": {"method": "spline"}}})
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_validate_partial_config_uses_defaults():
    cfg = validate_config(OmegaConf.create({"experiment": {"seed": 7}}))
    assert cfg.experiment.seed == 7
    assert cfg.training.epochs == 5  # default preserved


def test_compose_comma2k19():
    cfg = validate_config(compose_config(["dataset=comma2k19"]))
    assert cfg.data.source == "comma2k19"
    assert cfg.data.frame_rate_hz == 20
    assert cfg.data.window_stride >= 1
    assert cfg.data.comma2k19.cache_image_size == [128, 128]
    assert cfg.data.comma2k19.max_segments is None


def test_evaluation_device_defaults_auto():
    cfg = validate_config(compose_config())
    assert cfg.evaluation.device == "auto"
