"""Tests for src.models.training (WP4): LightningModule + trainer builder + checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from src.core.config import ModelCfg, TrainingCfg
from src.core.run_manager import create_run
from src.models.architectures import build_model
from src.models.training import (
    SpeedLightningModule,
    build_trainer,
    load_model_from_checkpoint,
)


def make_model_cfg() -> ModelCfg:
    return ModelCfg(
        name="tiny_cnn_gru",
        encoder={"name": "tiny_cnn", "pretrained": False, "feature_dim": 64},
        temporal={"name": "gru", "hidden_size": 32, "num_layers": 1},
        checkpoint=None,
    )


def make_training_cfg() -> TrainingCfg:
    return TrainingCfg()


def _batch(batch_size: int = 4) -> dict:
    return {
        "frames": torch.randn(batch_size, 5, 3, 32, 32),
        "target_mph": torch.randn(batch_size) * 20.0 + 30.0,
        "frame_index": torch.arange(batch_size),
        "sequence_id": [f"s{i}" for i in range(batch_size)],
        "timestamp": torch.rand(batch_size),
    }


def test_lightning_module_training_step():
    lm = SpeedLightningModule(build_model(make_model_cfg()), make_training_cfg())
    loss = lm.training_step(_batch(), 0)
    assert torch.isfinite(loss)


def test_lightning_module_validation_step():
    lm = SpeedLightningModule(build_model(make_model_cfg()), make_training_cfg())
    loss = lm.validation_step(_batch(), 0)
    assert torch.isfinite(loss)


def test_rmse_loss_supported():
    cfg = make_training_cfg()
    cfg.loss = "rmse"
    lm = SpeedLightningModule(build_model(make_model_cfg()), cfg)
    loss = lm.training_step(_batch(), 0)
    assert torch.isfinite(loss)


def test_predict_step_returns_metadata():
    lm = SpeedLightningModule(build_model(make_model_cfg()), make_training_cfg())
    out = lm.predict_step(_batch(4), 0)
    assert set(out) == {
        "raw_speed_mph",
        "target_mph",
        "frame_index",
        "sequence_id",
        "timestamp",
    }
    assert out["raw_speed_mph"].shape == (4,)


def test_build_trainer_paths_inside_run_dir(tiny_config):
    paths = create_run(tiny_config)
    trainer = build_trainer(tiny_config, paths)
    assert Path(trainer.default_root_dir).resolve() == paths.run_dir.resolve()


def test_load_model_from_checkpoint_roundtrip(tmp_path: Path):
    model = build_model(make_model_cfg())
    model.eval()
    # Lightning-style checkpoint with "model." prefix.
    ckpt = {"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}
    path = tmp_path / "best.ckpt"
    torch.save(ckpt, path)

    loaded = load_model_from_checkpoint(path, make_model_cfg())
    x = torch.randn(2, 5, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))


def test_model_export_roundtrip(tmp_path: Path):
    from src.models.training import load_model_export, save_model_export

    model = build_model(make_model_cfg())
    model.eval()
    path = tmp_path / "model.pt"
    save_model_export(model, path, make_model_cfg())
    loaded = load_model_export(path)
    x = torch.randn(2, 5, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))


def test_tracker_metrics_callback_logs_epoch_metrics():
    from src.models.training import TrackerMetricsCallback

    class SpyTracker:
        def __init__(self) -> None:
            self.calls: list = []

        def log_metrics(self, metrics: dict, step: int | None = None) -> None:
            self.calls.append((metrics, step))

    class FakeTrainer:
        current_epoch = 3
        callback_metrics = {
            "train/loss_epoch": torch.tensor(1.5),
            "val/loss": torch.tensor(0.5),
            "val/mae_mph": torch.tensor(2.5),
        }

    spy = SpyTracker()
    callback = TrackerMetricsCallback(spy)
    callback.on_train_epoch_end(FakeTrainer(), None)
    callback.on_validation_epoch_end(FakeTrainer(), None)

    assert spy.calls == [
        ({"train/loss_epoch": 1.5}, 3),
        ({"val/loss": 0.5, "val/mae_mph": 2.5}, 3),
    ]
