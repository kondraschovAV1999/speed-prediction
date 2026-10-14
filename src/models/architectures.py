"""Models (WP4): base class, registry, and the tiny CNN+GRU skeleton."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torchvision.models import ResNet18_Weights

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

    def __init__(self, feature_dim: int = 64, **kwargs):
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


class ResNet18Encoder(nn.Module):
    """Frozen (or trainable) ResNet18 backbone with the classification head removed.

    ``spatial=False`` -> global-average pool to a vector ``[B, feature_dim]``.
    ``spatial=True``  -> conv stem only, 1x1-projected to ``feature_dim`` channels
                         ``[B, feature_dim, H/32, W/32]``.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        pretrained: bool = True,
        frozen: bool = True,
        spatial: bool = False,
        weights_path: str | None = None,
    ):
        super().__init__()
        weights = (
            None
            if weights_path is not None
            else (ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        )
        backbone = torchvision.models.resnet18(weights=weights)
        if weights_path is not None:
            backbone.load_state_dict(torch.load(weights_path, map_location="cpu"))

        if spatial:
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # -> [B,512,H/32,W/32]
            self.proj = nn.Conv2d(512, feature_dim, 1)
        else:
            backbone.fc = nn.Identity()  # avgpool + flatten -> [B, 512]
            self.backbone = backbone
            self.proj = nn.Linear(512, feature_dim) if feature_dim != 512 else nn.Identity()

        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()  # BN running stats; avoids B=1 train-mode BN errors

    def forward(self, x: Tensor) -> Tensor:  # x: [B, C, H, W]
        return self.proj(self.backbone(x))


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


class LSTMTemporal(nn.Module):
    """LSTM over the per-frame feature sequence; returns the last timestep."""

    def __init__(self, input_dim: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True
        )

    def forward(self, x: Tensor) -> Tensor:  # x: [B, T, input_dim]
        out, _ = self.lstm(x)
        return out[:, -1, :]  # [B, hidden_size]


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell (Shi et al. 2015): one gate convolution over [x, h]."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=pad)

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        h, c = state
        i, f, g, o = self.conv(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c_next = f * c + i * g
        return o * torch.tanh(c_next), c_next


class ConvLSTM(nn.Module):
    """Spatially recurrent ConvLSTM (single layer), batch-first [B, T, C, H, W]."""

    def __init__(self, input_dim: int, hidden_size: int, kernel_size: int = 3, num_layers: int = 1):
        super().__init__()
        if num_layers != 1:
            raise ConfigError("ConvLSTM num_layers must be 1 (stacking is future work)")
        self.cell = ConvLSTMCell(input_dim, hidden_size, kernel_size)
        self.hidden_size = hidden_size

    def forward(self, x: Tensor) -> Tensor:  # [B, T, C, H, W] -> [B, hidden_size, H, W]
        b, t, _, hh, ww = x.shape
        h = torch.zeros(b, self.hidden_size, hh, ww, device=x.device)
        c = torch.zeros_like(h)
        for step in range(t):
            h, c = self.cell(x[:, step], (h, c))
        return h


class EncoderRecurrent(BaseSpeedModel):
    """Generic encoder -> per-frame features -> recurrent temporal -> head.

    The encoder may return a per-frame vector ``[B*T, feature_dim]`` (fed to an
    LSTM/GRU) or a per-frame spatial map ``[B*T, C, H, W]`` (fed to a ConvLSTM).
    ``forward`` dispatches on the number of dimensions of the encoder output.
    """

    def __init__(self, encoder: dict, temporal: dict):
        super().__init__()
        feature_dim = int(encoder.get("feature_dim", 64))
        hidden_size = int(temporal.get("hidden_size", 32))

        encoder_name = encoder.get("name", "tiny_cnn")
        if encoder_name not in ENCODER_REGISTRY:
            raise ConfigError(f"Unknown encoder {encoder_name!r}")
        temporal_name = temporal.get("name", "gru")
        if temporal_name not in TEMPORAL_REGISTRY:
            raise ConfigError(f"Unknown temporal module {temporal_name!r}")

        self.encoder = ENCODER_REGISTRY[encoder_name](
            **{k: v for k, v in encoder.items() if k != "name"}
        )
        tmp_kwargs = {k: v for k, v in temporal.items() if k != "name"}
        tmp_kwargs["input_dim"] = feature_dim
        self.temporal = TEMPORAL_REGISTRY[temporal_name](**tmp_kwargs)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, frames: Tensor) -> Tensor:  # frames: [B, T, C, H, W]
        b, t, c, h, w = frames.shape
        feats = self.encoder(frames.reshape(b * t, c, h, w))
        if feats.ndim == 2:  # vector path (LSTM/GRU)
            last = self.temporal(feats.reshape(b, t, -1))  # [B, hidden]
        else:  # spatial path (ConvLSTM)
            _, ch, fh, fw = feats.shape
            last = self.temporal(feats.reshape(b, t, ch, fh, fw))  # [B, hidden, fh, fw]
            last = last.mean(dim=(2, 3))  # [B, hidden]
        return self.head(last).squeeze(-1)  # [B]


ENCODER_REGISTRY: dict[str, type[nn.Module]] = {
    "tiny_cnn": TinyCNNEncoder,
    "resnet18": ResNet18Encoder,
}
TEMPORAL_REGISTRY: dict[str, type[nn.Module]] = {
    "gru": GRUTemporal,
    "lstm": LSTMTemporal,
    "convlstm": ConvLSTM,
}

MODEL_REGISTRY: dict[str, type[BaseSpeedModel]] = {
    "tiny_cnn_gru": EncoderRecurrent,
    "cnn_lstm": EncoderRecurrent,
    "resnet_lstm": EncoderRecurrent,
    "resnet_convlstm": EncoderRecurrent,
}


def build_model(cfg: ModelCfg) -> BaseSpeedModel:
    """Build a model from typed config; unknown name raises :class:`ConfigError`."""
    if cfg.name not in MODEL_REGISTRY:
        raise ConfigError(f"Unknown model {cfg.name!r}. Valid models: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[cfg.name](encoder=cfg.encoder, temporal=cfg.temporal)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
