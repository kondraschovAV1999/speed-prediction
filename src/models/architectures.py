"""Models (WP4): base class, registry, and the tiny CNN+GRU skeleton."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torch import Tensor

from src.core.config import ModelCfg
from src.core.schemas import ConfigError
from src.models.convlstm import ConvLSTM as _ConvLSTMImpl


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


# Pre-trained torchvision backbones usable as (frozen) feature extractors.
# Keys: build = constructor, weights = Weights enum, dim = native feature dim,
# head = classification-head attribute, features = spatial-stem attr (None = strip
# avgpool+fc ResNet-style).
_BACKBONES: dict[str, dict] = {
    "resnet18": {
        "build": tvm.resnet18,
        "weights": tvm.ResNet18_Weights,
        "dim": 512,
        "head": "fc",
        "features": None,
    },
    "resnet50": {
        "build": tvm.resnet50,
        "weights": tvm.ResNet50_Weights,
        "dim": 2048,
        "head": "fc",
        "features": None,
    },
    "mobilenet_v3_small": {
        "build": tvm.mobilenet_v3_small,
        "weights": tvm.MobileNet_V3_Small_Weights,
        "dim": 576,
        "head": "classifier",
        "features": "features",
    },
    "mobilenet_v3_large": {
        "build": tvm.mobilenet_v3_large,
        "weights": tvm.MobileNet_V3_Large_Weights,
        "dim": 960,
        "head": "classifier",
        "features": "features",
    },
    "efficientnet_b0": {
        "build": tvm.efficientnet_b0,
        "weights": tvm.EfficientNet_B0_Weights,
        "dim": 1280,
        "head": "classifier",
        "features": "features",
    },
}

# Pre-trained torchvision Vision Transformers usable as (frozen) CLS-token
# feature extractors. Keys: build = constructor, weights = Weights enum.
# ``vit_h_14`` is deliberately excluded (SWAG-only weights, no IMAGENET1K_V1).
_VIT_BUILDERS: dict[str, tuple] = {
    "vit_b_16": (tvm.vit_b_16, tvm.ViT_B_16_Weights),
    "vit_b_32": (tvm.vit_b_32, tvm.ViT_B_32_Weights),
    "vit_l_16": (tvm.vit_l_16, tvm.ViT_L_16_Weights),
    "vit_l_32": (tvm.vit_l_32, tvm.ViT_L_32_Weights),
}


class BackboneEncoder(nn.Module):
    """Frozen (or trainable) pre-trained backbone with the classification head removed.

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
        backbone: str = "resnet18",
    ):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ConfigError(f"Unknown backbone {backbone!r}. Valid: {sorted(_BACKBONES)}")
        spec = _BACKBONES[backbone]
        weights = (
            None
            if weights_path is not None
            else (spec["weights"].IMAGENET1K_V1 if pretrained else None)
        )
        model = spec["build"](weights=weights)
        if weights_path is not None:
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))

        native_dim = int(spec["dim"])
        if spatial:
            if spec["features"] is None:  # ResNet-style: strip avgpool + fc
                self.backbone = nn.Sequential(*list(model.children())[:-2])
            else:
                self.backbone = model.features
            self.proj = (
                nn.Conv2d(native_dim, feature_dim, 1)
                if feature_dim != native_dim
                else nn.Identity()
            )
        else:
            setattr(model, spec["head"], nn.Identity())  # remove classification head
            self.backbone = model
            self.proj = (
                nn.Linear(native_dim, feature_dim) if feature_dim != native_dim else nn.Identity()
            )

        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()  # BN running stats; avoids B=1 train-mode BN errors

    def forward(self, x: Tensor) -> Tensor:  # x: [B, C, H, W]
        return self.proj(self.backbone(x))


class ViTEncoder(nn.Module):
    """Frozen (or trainable) ImageNet ViT feature extractor via the CLS token.

    torchvision's ViT extracts the CLS token internally before applying ``heads``,
    so replacing ``heads`` with ``Identity`` yields ``[B, hidden]`` per frame
    directly.  Input is fixed at ``image_size`` (default 224) so the pretrained
    positional embeddings match; there is no interpolation.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        pretrained: bool = True,
        frozen: bool = True,
        weights_path: str | None = None,
        variant: str = "vit_b_16",
        image_size: int = 224,
    ):
        super().__init__()
        if variant not in _VIT_BUILDERS:
            raise ConfigError(f"Unknown ViT variant {variant!r}. Valid: {sorted(_VIT_BUILDERS)}")
        build, weights_enum = _VIT_BUILDERS[variant]
        weights = (
            None
            if weights_path is not None
            else (weights_enum.IMAGENET1K_V1 if pretrained else None)
        )
        model = build(weights=weights, image_size=image_size)
        if weights_path is not None:
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        native_dim = int(model.hidden_dim)
        model.heads = nn.Identity()  # forward now returns the CLS token [B, hidden]
        self.backbone = model
        self.proj = (
            nn.Linear(native_dim, feature_dim) if feature_dim != native_dim else nn.Identity()
        )
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

    def forward(self, x: Tensor) -> Tensor:  # x: [B, C, H, W]
        return self.proj(self.backbone(x))  # [B, feature_dim]


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


class ConvLSTM(nn.Module):
    """ConvLSTM temporal module (Shi et al. 2015, with peephole connections).

    Thin adapter over the vendored KL4805 reference implementation
    (``src/models/convlstm.py``), presenting the pipeline's batch-first
    ``[B, T, C, H, W] -> [B, hidden_size, H, W]`` interface. Converts the
    config's scalar ``kernel_size`` to the tuple the reference expects.
    """

    def __init__(
        self, input_dim: int, hidden_size: int = 32, kernel_size: int = 3, num_layers: int = 1
    ):
        super().__init__()
        if num_layers != 1:
            raise ConfigError("ConvLSTM num_layers must be 1 (stacking is future work)")
        self._impl = _ConvLSTMImpl(
            input_dim=input_dim,
            hidden_dim=int(hidden_size),
            kernel_size=(int(kernel_size), int(kernel_size)),
            num_layers=1,
            batch_first=True,
            bias=True,
            return_all_layers=False,
        )
        self.hidden_size = int(hidden_size)

    def forward(self, x: Tensor) -> Tensor:  # [B, T, C, H, W] -> [B, hidden_size, H, W]
        _, last_states = self._impl(x)
        h, _ = last_states[0]  # last layer's (h, c)
        return h


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "none": nn.Identity,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}

# Encoders that support a spatial stem (feed a ConvLSTM); the rest are vector-only.
_SPATIAL_ENCODERS = {"backbone", "resnet18"}


def _build_head(input_dim: int, head: dict) -> nn.Module:
    """Build the regression head from a config dict.

    ``head.hidden_dims`` is the list of intermediate layer widths (empty => a
    single ``Linear(input_dim, 1)``). ``head.activation`` and ``head.dropout``
    are applied after each hidden layer, never after the final scalar output.
    """
    hidden_dims = [int(d) for d in head.get("hidden_dims", []) or []]
    activation = head.get("activation", "none")
    dropout = float(head.get("dropout", 0.0))
    if activation not in _ACTIVATIONS:
        raise ConfigError(f"Unknown head activation {activation!r}. Valid: {sorted(_ACTIVATIONS)}")
    dims = [input_dim, *hidden_dims, 1]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:  # hidden layer, not the final scalar output
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            if activation != "none":
                layers.append(_ACTIVATIONS[activation]())
    return nn.Sequential(*layers)


class EncoderRecurrent(BaseSpeedModel):
    """Generic encoder -> per-frame features -> recurrent temporal -> head.

    The encoder may return a per-frame vector ``[B*T, feature_dim]`` (fed to an
    LSTM/GRU) or a per-frame spatial map ``[B*T, C, H, W]`` (fed to a ConvLSTM).
    ``forward`` dispatches on the number of dimensions of the encoder output.
    """

    def __init__(self, encoder: dict, temporal: dict, head: dict | None = None):
        super().__init__()
        head = head or {}
        encoder_name = encoder.get("name", "tiny_cnn")
        if encoder_name not in ENCODER_REGISTRY:
            raise ConfigError(f"Unknown encoder {encoder_name!r}")
        temporal_name = temporal.get("name", "gru")
        if temporal_name not in TEMPORAL_REGISTRY:
            raise ConfigError(f"Unknown temporal module {temporal_name!r}")

        # Spatial mode belongs to the temporal module: ConvLSTM consumes a spatial
        # map, LSTM/GRU consume a per-frame vector.
        spatial = temporal_name == "convlstm"
        encoder_kwargs = {k: v for k, v in encoder.items() if k != "name"}
        if encoder_name in _SPATIAL_ENCODERS:
            encoder_kwargs["spatial"] = spatial
        elif spatial:
            raise ConfigError(f"encoder {encoder_name!r} is vector-only and cannot feed a ConvLSTM")

        feature_dim = int(encoder.get("feature_dim", 64))
        hidden_size = int(temporal.get("hidden_size", 32))

        self.encoder = ENCODER_REGISTRY[encoder_name](**encoder_kwargs)
        tmp_kwargs = {k: v for k, v in temporal.items() if k != "name"}
        tmp_kwargs["input_dim"] = feature_dim
        self.temporal = TEMPORAL_REGISTRY[temporal_name](**tmp_kwargs)
        self.head = _build_head(hidden_size, head)

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
    "resnet18": BackboneEncoder,  # legacy alias (backbone defaults to "resnet18")
    "backbone": BackboneEncoder,
    "vit": ViTEncoder,
}
TEMPORAL_REGISTRY: dict[str, type[nn.Module]] = {
    "gru": GRUTemporal,
    "lstm": LSTMTemporal,
    "convlstm": ConvLSTM,
}


def build_model(cfg: ModelCfg) -> BaseSpeedModel:
    """Compose an :class:`EncoderRecurrent` from ``encoder``/``temporal``/``head``."""
    return EncoderRecurrent(encoder=cfg.encoder, temporal=cfg.temporal, head=cfg.head)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
