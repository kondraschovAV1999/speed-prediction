"""Lightning module and Trainer builder (WP4)."""

from __future__ import annotations

import sys
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch import Tensor

from src.core.config import Config, ModelCfg, TrainingCfg
from src.core.schemas import ConfigError, RunPaths
from src.models.architectures import BaseSpeedModel, build_model
from src.tracking.tracker import ExperimentTracker


def _rmse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """RMSE (mph): sqrt(MSE). Same optimum as MSE; epsilon avoids inf gradient at 0."""
    return torch.sqrt(F.mse_loss(pred, target) + 1e-8)


def _make_loss_fn(cfg: TrainingCfg):
    """Return the loss function selected by ``cfg.loss`` (mph units)."""
    if cfg.loss == "mse":
        return F.mse_loss
    if cfg.loss in ("l1", "mae"):  # L1 loss == Mean Absolute Error
        return F.l1_loss
    if cfg.loss == "rmse":
        return _rmse_loss
    if cfg.loss == "huber":
        delta = cfg.huber_delta
        return lambda pred, target: F.huber_loss(pred, target, delta=delta)
    raise ConfigError(f"Unknown loss: {cfg.loss!r}")


class SpeedLightningModule(L.LightningModule):
    """Wraps a :class:`BaseSpeedModel` for training/eval with Lightning."""

    def __init__(self, model: BaseSpeedModel, training_cfg: TrainingCfg):
        super().__init__()
        self.model = model
        self.cfg = training_cfg
        self.loss_fn = _make_loss_fn(training_cfg)

    def forward(self, frames: Tensor) -> Tensor:
        return self.model(frames)

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        pred = self.model(batch["frames"])
        loss = self.loss_fn(pred, batch["target_mph"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        pred = self.model(batch["frames"])
        loss = self.loss_fn(pred, batch["target_mph"])
        self.log("val/loss", loss, on_epoch=True)
        mae = torch.abs(pred - batch["target_mph"]).mean()
        self.log("val/mae_mph", mae, on_epoch=True)
        rmse = torch.sqrt(torch.mean((pred - batch["target_mph"]) ** 2) + 1e-8)
        self.log("val/rmse_mph", rmse, on_epoch=True)
        return loss

    def predict_step(self, batch: dict, batch_idx: int) -> dict:
        """Return raw predictions plus batch metadata (for offline evaluation)."""
        pred = self.model(batch["frames"])
        return {
            "raw_speed_mph": pred,
            "target_mph": batch["target_mph"],
            "frame_index": batch["frame_index"],
            "sequence_id": batch["sequence_id"],
            "timestamp": batch["timestamp"],
        }

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )


def _metric_value(value) -> float:
    """Convert a Lightning metric value (Tensor or float) to a plain float."""
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


class TrackerMetricsCallback(L.Callback):
    """Forward epoch-level train/val metrics to the :class:`ExperimentTracker`.

    Lightning records ``train/loss``, ``val/loss`` and ``val/mae_mph`` via
    ``self.log``; this callback copies the epoch-level aggregates to the tracker
    (e.g. MLflow) so progress is visible per epoch.
    """

    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._log(trainer, "train/loss_epoch", "train/loss")

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._log(trainer, "val/loss", "val/mae_mph")

    def _log(self, trainer: L.Trainer, *keys: str) -> None:
        metrics: dict[str, float] = {}
        for key in keys:
            value = trainer.callback_metrics.get(key)
            if value is not None:
                metrics[key] = _metric_value(value)
        if metrics:
            self.tracker.log_metrics(metrics, step=trainer.current_epoch)


def build_trainer(
    cfg: Config, paths: RunPaths, tracker: ExperimentTracker | None = None
) -> L.Trainer:
    """Build a Lightning Trainer writing everything under ``paths.run_dir``.

    If ``tracker`` is given, per-epoch train/val metrics are also forwarded to it
    (e.g. MLflow) via :class:`TrackerMetricsCallback`.
    """
    logger = CSVLogger(save_dir=paths.run_dir, name="training_logs")
    checkpoint = ModelCheckpoint(
        dirpath=paths.checkpoints_dir,
        monitor=cfg.training.monitor,
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best",
    )
    early_stopping = EarlyStopping(
        monitor=cfg.training.monitor,
        mode="min",
        patience=cfg.training.early_stopping_patience,
    )
    callbacks = [checkpoint, early_stopping]
    if tracker is not None:
        callbacks.append(TrackerMetricsCallback(tracker))
    return L.Trainer(
        default_root_dir=paths.run_dir,
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        enable_progress_bar=bool(sys.stdout.isatty()),
        deterministic=cfg.experiment.deterministic,
        gradient_clip_val=cfg.training.gradient_clip_val,
        num_sanity_val_steps=0,
    )


def load_model_from_checkpoint(path: Path | str, cfg: ModelCfg) -> BaseSpeedModel:
    """Load a model from a Lightning checkpoint (strips the ``model.`` prefix)."""
    model = build_model(cfg)
    checkpoint = torch.load(str(path), map_location="cpu")
    state = checkpoint["state_dict"]
    stripped = {
        (key[6:] if key.startswith("model.") else key): value for key, value in state.items()
    }
    model.load_state_dict(stripped)
    model.eval()
    return model


def save_model_export(model: BaseSpeedModel, path: Path | str, cfg: ModelCfg) -> None:
    """Save a standalone, self-describing model export (no Lightning wrapper)."""
    payload = {
        "state_dict": model.state_dict(),
        "model_name": cfg.name,
        "encoder": cfg.encoder,
        "temporal": cfg.temporal,
        "head": cfg.head,
    }
    torch.save(payload, str(path))


def load_model_export(path: Path | str) -> BaseSpeedModel:
    """Load a standalone model export produced by :func:`save_model_export`."""
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ModelCfg(
        name=payload.get("model_name", "encoder_recurrent"),
        encoder=payload["encoder"],
        temporal=payload["temporal"],
        head=payload.get("head", {}),
    )
    model = build_model(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
