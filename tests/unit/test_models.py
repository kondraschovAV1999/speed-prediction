"""Tests for src.models.architectures (WP4)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.core.config import ModelCfg
from src.core.schemas import ConfigError
from src.models.architectures import build_model, count_parameters


def make_model_cfg() -> ModelCfg:
    return ModelCfg(
        name="tiny_cnn_gru",
        encoder={"name": "tiny_cnn", "pretrained": False, "feature_dim": 64},
        temporal={"name": "gru", "hidden_size": 32, "num_layers": 1},
        checkpoint=None,
    )


def test_cnn_lstm_shape():
    cfg = ModelCfg(
        name="cnn_lstm",
        encoder={"name": "tiny_cnn", "pretrained": False, "feature_dim": 64},
        temporal={"name": "lstm", "hidden_size": 32, "num_layers": 1},
        checkpoint=None,
    )
    model = build_model(cfg)
    out = model(torch.randn(2, 5, 3, 32, 32))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_output_shape_b1():
    model = build_model(make_model_cfg())
    out = model(torch.randn(1, 5, 3, 32, 32))
    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_output_shape_b4():
    model = build_model(make_model_cfg())
    out = model(torch.randn(4, 5, 3, 32, 32))
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_mlp_head_shape():
    cfg = make_model_cfg()
    cfg.head = {"hidden_dims": [32, 16], "activation": "relu", "dropout": 0.1}
    model = build_model(cfg)
    out = model(torch.randn(2, 5, 3, 32, 32))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()
    # MLP head: more than the single Linear of the default `linear` head
    assert len(list(model.head)) > 1


def test_convlstm_with_vector_encoder_raises():
    cfg = ModelCfg(
        encoder={"name": "vit", "variant": "vit_b_16", "pretrained": False, "feature_dim": 768},
        temporal={"name": "convlstm", "hidden_size": 32, "kernel_size": 3, "num_layers": 1},
    )
    with pytest.raises(ConfigError):
        build_model(cfg)


def test_unknown_head_activation_raises():
    cfg = make_model_cfg()
    cfg.head = {"hidden_dims": [16], "activation": "swish", "dropout": 0.0}
    with pytest.raises(ConfigError):
        build_model(cfg)


def test_unknown_encoder_raises():
    cfg = make_model_cfg()
    cfg.encoder = {"name": "definitely_not_an_encoder", "pretrained": True, "feature_dim": 64}
    with pytest.raises(ConfigError):
        build_model(cfg)


def test_resnet_lstm_frozen_shape_and_trainable():
    cfg = ModelCfg(
        name="resnet_lstm",
        encoder={
            "name": "resnet18",
            "pretrained": False,
            "feature_dim": 512,
            "frozen": True,
            "spatial": False,
        },
        temporal={"name": "lstm", "hidden_size": 64, "num_layers": 1},
        checkpoint=None,
    )
    model = build_model(cfg)
    out = model(torch.randn(2, 3, 3, 128, 128))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()
    # frozen backbone: zero trainable params inside the backbone
    backbone_trainable = [p for p in model.encoder.backbone.parameters() if p.requires_grad]
    assert len(backbone_trainable) == 0
    # temporal + head still trainable, but fewer trainable than total params
    assert count_parameters(model) > 0
    assert count_parameters(model) < sum(p.numel() for p in model.parameters())


def test_resnet50_lstm_shape_and_frozen():
    cfg = ModelCfg(
        name="resnet50_lstm",
        encoder={
            "name": "backbone",
            "backbone": "resnet50",
            "pretrained": False,
            "feature_dim": 2048,
            "frozen": True,
            "spatial": False,
        },
        temporal={"name": "lstm", "hidden_size": 64, "num_layers": 1},
        checkpoint=None,
    )
    model = build_model(cfg)
    out = model(torch.randn(2, 3, 3, 128, 128))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()
    # frozen backbone: zero trainable params inside the backbone
    assert len([p for p in model.encoder.backbone.parameters() if p.requires_grad]) == 0
    assert count_parameters(model) > 0
    assert count_parameters(model) < sum(p.numel() for p in model.parameters())


def test_param_count_positive():
    assert count_parameters(build_model(make_model_cfg())) > 0


def test_overfit_one_batch():
    torch.manual_seed(0)
    model = build_model(make_model_cfg())
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    frames = torch.randn(8, 5, 3, 32, 32)
    target = torch.randn(8) * 20.0 + 30.0
    with torch.no_grad():
        initial = F.mse_loss(model(frames), target).item()
    for _ in range(60):
        opt.zero_grad()
        loss = F.mse_loss(model(frames), target)
        loss.backward()
        opt.step()
    final = F.mse_loss(model(frames), target).item()
    assert final <= 0.7 * initial


def test_convlstm_output_shape():
    from src.models.architectures import ConvLSTM

    m = ConvLSTM(input_dim=8, hidden_size=16, kernel_size=3)
    out = m(torch.randn(2, 5, 8, 4, 4))
    assert out.shape == (2, 16, 4, 4)


def test_resnet_convlstm_spatial_shape_and_frozen():
    cfg = ModelCfg(
        name="resnet_convlstm",
        encoder={
            "name": "resnet18",
            "pretrained": False,
            "feature_dim": 32,
            "frozen": True,
        },
        temporal={"name": "convlstm", "hidden_size": 32, "kernel_size": 3, "num_layers": 1},
        checkpoint=None,
    )
    model = build_model(cfg)
    out = model(torch.randn(2, 3, 3, 128, 128))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()
    # encoder returns a 4-D spatial map with the projected channel count
    feats = model.encoder(torch.randn(4, 3, 128, 128))
    assert feats.ndim == 4
    assert feats.shape[1] == 32
    assert len([p for p in model.encoder.backbone.parameters() if p.requires_grad]) == 0


def test_mobilenet_lstm_shape_and_frozen():
    cfg = ModelCfg(
        name="mobilenet_lstm",
        encoder={
            "name": "backbone",
            "backbone": "mobilenet_v3_small",
            "pretrained": False,
            "feature_dim": 576,
            "frozen": True,
            "spatial": False,
        },
        temporal={"name": "lstm", "hidden_size": 64, "num_layers": 1},
        checkpoint=None,
    )
    model = build_model(cfg)
    out = model(torch.randn(2, 3, 3, 128, 128))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()
    assert len([p for p in model.encoder.backbone.parameters() if p.requires_grad]) == 0


def test_efficientnet_lstm_shape_and_frozen():
    cfg = ModelCfg(
        name="efficientnet_lstm",
        encoder={
            "name": "backbone",
            "backbone": "efficientnet_b0",
            "pretrained": False,
            "feature_dim": 1280,
            "frozen": True,
            "spatial": False,
        },
        temporal={"name": "lstm", "hidden_size": 64, "num_layers": 1},
        checkpoint=None,
    )
    model = build_model(cfg)
    out = model(torch.randn(2, 3, 3, 128, 128))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()
    assert len([p for p in model.encoder.backbone.parameters() if p.requires_grad]) == 0


def test_unknown_backbone_raises():
    cfg = ModelCfg(
        name="mobilenet_lstm",
        encoder={
            "name": "backbone",
            "backbone": "not_a_backbone",
            "pretrained": False,
            "feature_dim": 128,
            "frozen": True,
            "spatial": False,
        },
        temporal={"name": "lstm", "hidden_size": 64, "num_layers": 1},
        checkpoint=None,
    )
    with pytest.raises(ConfigError):
        build_model(cfg)


def test_vit_lstm_shape_and_frozen():
    cfg = ModelCfg(
        name="vit_lstm",
        encoder={
            "name": "vit",
            "variant": "vit_b_16",
            "pretrained": False,
            "feature_dim": 768,
            "frozen": True,
            "image_size": 224,
        },
        temporal={"name": "lstm", "hidden_size": 64, "num_layers": 1},
        checkpoint=None,
    )
    model = build_model(cfg)
    out = model(torch.randn(2, 3, 3, 224, 224))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()
    # frozen backbone: zero trainable params inside the ViT backbone
    assert len([p for p in model.encoder.backbone.parameters() if p.requires_grad]) == 0
    # temporal + head still trainable, but fewer trainable than total params
    assert count_parameters(model) > 0
    assert count_parameters(model) < sum(p.numel() for p in model.parameters())


def test_vit_encoder_projects_feature_dim():
    from src.models.architectures import ViTEncoder

    enc = ViTEncoder(feature_dim=256, pretrained=False, frozen=True, variant="vit_b_16")
    out = enc(torch.randn(3, 3, 224, 224))
    assert out.shape == (3, 256)
    assert torch.isfinite(out).all()


def test_unknown_vit_variant_raises():
    cfg = ModelCfg(
        name="vit_lstm",
        encoder={
            "name": "vit",
            "variant": "not_a_vit",
            "pretrained": False,
            "feature_dim": 768,
            "frozen": True,
        },
        temporal={"name": "lstm", "hidden_size": 64, "num_layers": 1},
        checkpoint=None,
    )
    with pytest.raises(ConfigError):
        build_model(cfg)


def test_deeplabv3_lstm_shape_and_frozen():
    cfg = ModelCfg(
        encoder={
            "name": "deeplabv3",
            "variant": "resnet50",
            "pretrained": False,
            "feature_dim": 512,
            "frozen": True,
        },
        temporal={"name": "lstm", "hidden_size": 64, "num_layers": 1},
    )
    model = build_model(cfg)
    out = model(torch.randn(2, 3, 3, 128, 128))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()
    assert len([p for p in model.encoder.backbone.parameters() if p.requires_grad]) == 0
    assert count_parameters(model) > 0


def test_deeplabv3_convlstm_spatial_shape_and_frozen():
    cfg = ModelCfg(
        encoder={
            "name": "deeplabv3",
            "variant": "resnet50",
            "pretrained": False,
            "feature_dim": 32,
            "frozen": True,
        },
        temporal={"name": "convlstm", "hidden_size": 32, "kernel_size": 3, "num_layers": 1},
    )
    model = build_model(cfg)
    out = model(torch.randn(2, 3, 3, 128, 128))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()
    # encoder returns a 4-D spatial map with the projected channel count (stride 8)
    feats = model.encoder(torch.randn(4, 3, 128, 128))
    assert feats.ndim == 4
    assert feats.shape[1] == 32
    assert feats.shape[2] == 16  # 128 / 8
    assert len([p for p in model.encoder.backbone.parameters() if p.requires_grad]) == 0


def test_unknown_deeplab_variant_raises():
    cfg = ModelCfg(
        encoder={
            "name": "deeplabv3",
            "variant": "not_a_variant",
            "pretrained": False,
            "feature_dim": 512,
        },
        temporal={"name": "lstm", "hidden_size": 64, "num_layers": 1},
    )
    with pytest.raises(ConfigError):
        build_model(cfg)
