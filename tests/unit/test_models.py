"""Tests for src.models (WP4)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.config import ModelCfg
from src.models import build_model, count_parameters
from src.schemas import ConfigError


def make_model_cfg() -> ModelCfg:
    return ModelCfg(
        name="tiny_cnn_gru",
        encoder={"name": "tiny_cnn", "pretrained": False, "feature_dim": 64},
        temporal={"name": "gru", "hidden_size": 32, "num_layers": 1},
        checkpoint=None,
    )


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


def test_unknown_model_name_raises():
    with pytest.raises(ConfigError):
        build_model(ModelCfg(name="nope", encoder={}, temporal={}))


def test_unknown_encoder_raises():
    cfg = make_model_cfg()
    cfg.encoder = {"name": "resnet", "pretrained": True, "feature_dim": 64}
    with pytest.raises(ConfigError):
        build_model(cfg)


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
