"""Models (WP4): base class, registry, and the tiny CNN+GRU skeleton."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.core.config import ModelCfg
from src.core.schemas import ConfigError


class BaseSpeedModel(nn.Module):
    """Regresses speed (mph) from a window of frames ``[B, T, C, H, W]``."""

    def forward(self, frames: Tensor) -> Tensor:  # -> [B]
        raise NotImplementedError


class TinyCNNEncoder(nn.Module):
    """Per-frame conv encoder: 3 conv layers -> adaptive pool -> feature vector.

    Uses no BatchNorm (safe for batch size 1).  Adaptive average pooling to a
    small 4x4 grid preserves just enough spatial layout for the temporal module
    to recover motion cues (a 1x1 global pool would be translation-invariant).
    """

    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, feature_dim)

    def forward(self, x: Tensor) -> Tensor:  # x: [B, C, H, W]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return self.fc(x.flatten(1))  # [B, feature_dim]


class GRUTemporal(nn.Module):
    """GRU over the per-frame feature sequence; returns the last timestep."""

    def __init__(self, input_dim: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.hidden_size = hidden_size

    def forward(self, x: Tensor) -> Tensor:  # x: [B, T, input_dim]
        out, _ = self.gru(x)
        return out[:, -1, :]  # [B, hidden_size]


class TinyCNNGRU(BaseSpeedModel):
    """Skeleton model: per-frame CNN encoder + GRU temporal + linear head."""

    def __init__(self, encoder: dict, temporal: dict):
        super().__init__()
        feature_dim = int(encoder.get("feature_dim", 64))
        hidden_size = int(temporal.get("hidden_size", 32))
        num_layers = int(temporal.get("num_layers", 1))

        encoder_name = encoder.get("name", "tiny_cnn")
        if encoder_name not in ENCODER_REGISTRY:
            raise ConfigError(f"Unknown encoder {encoder_name!r}")
        temporal_name = temporal.get("name", "gru")
        if temporal_name not in TEMPORAL_REGISTRY:
            raise ConfigError(f"Unknown temporal module {temporal_name!r}")

        self.encoder = ENCODER_REGISTRY[encoder_name](feature_dim=feature_dim)
        self.temporal = TEMPORAL_REGISTRY[temporal_name](
            input_dim=feature_dim, hidden_size=hidden_size, num_layers=num_layers
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, frames: Tensor) -> Tensor:  # frames: [B, T, C, H, W]
        b, t, c, h, w = frames.shape
        feats = self.encoder(frames.reshape(b * t, c, h, w))  # [B*T, feature_dim]
        feats = feats.reshape(b, t, -1)
        last = self.temporal(feats)  # [B, hidden_size]
        return self.head(last).squeeze(-1)  # [B]


ENCODER_REGISTRY: dict[str, type[nn.Module]] = {"tiny_cnn": TinyCNNEncoder}
TEMPORAL_REGISTRY: dict[str, type[nn.Module]] = {"gru": GRUTemporal}

MODEL_REGISTRY: dict[str, type[BaseSpeedModel]] = {"tiny_cnn_gru": TinyCNNGRU}


def build_model(cfg: ModelCfg) -> BaseSpeedModel:
    """Build a model from typed config; unknown name raises :class:`ConfigError`."""
    if cfg.name not in MODEL_REGISTRY:
        raise ConfigError(f"Unknown model {cfg.name!r}. Valid models: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[cfg.name](encoder=cfg.encoder, temporal=cfg.temporal)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
