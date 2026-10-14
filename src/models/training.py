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
from src.core.schemas import RunPaths
from src.models.architectures import BaseSpeedModel, build_model


class SpeedLightningModule(L.LightningModule):
    """Wraps a :class:`BaseSpeedModel` for training/eval with Lightning."""

    def __init__(self, model: BaseSpeedModel, training_cfg: TrainingCfg):
        super().__init__()
        self.model = model
        self.cfg = training_cfg
        self.loss_fn = {"mse": F.mse_loss, "l1": F.l1_loss}[training_cfg.loss]

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


def build_trainer(cfg: Config, paths: RunPaths) -> L.Trainer:
    """Build a Lightning Trainer writing everything under ``paths.run_dir``."""
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
    return L.Trainer(
        default_root_dir=paths.run_dir,
        logger=logger,
        callbacks=[checkpoint, early_stopping],
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
    }
    torch.save(payload, str(path))


def load_model_export(path: Path | str) -> BaseSpeedModel:
    """Load a standalone model export produced by :func:`save_model_export`."""
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ModelCfg(
        name=payload["model_name"], encoder=payload["encoder"], temporal=payload["temporal"]
    )
    model = build_model(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
