"""Tests for src.benchmark (WP9)."""

from __future__ import annotations

import numpy as np
import torch

from src.benchmark import benchmark_end_to_end, benchmark_model
from src.models import build_model


def make_model(tiny_config):
    return build_model(tiny_config.model)


def test_benchmark_model_samples(tiny_config):
    model = make_model(tiny_config)
    example = torch.randn(1, 5, 3, 32, 32)
    samples, summary = benchmark_model(model, example, "cpu", num_warmup=2, num_iters=10)
    assert len(samples) == 10
    assert (samples > 0).all()
    assert summary["n"] == 10
    assert summary["p95_ms"] >= summary["median_ms"]
    assert summary["device"] == "cpu"


def test_benchmark_model_eval_and_no_grad(tiny_config):
    model = make_model(tiny_config)
    example = torch.randn(1, 5, 3, 32, 32)
    benchmark_model(model, example, "cpu", num_warmup=2, num_iters=5)
    assert not model.training
    for p in model.parameters():
        assert p.grad is None


def test_benchmark_end_to_end(tiny_config):
    from src.pipeline import StreamingSpeedEstimator
    from src.postprocessing import SpeedPostProcessor
    from src.preprocessing import build_transform

    model = make_model(tiny_config)
    transform = build_transform(tiny_config.preprocessing)
    proc = SpeedPostProcessor(tiny_config.postprocessing)
    t = tiny_config.data.sequence_length
    k = tiny_config.data.frame_step
    max_gap = tiny_config.data.max_gap_factor / tiny_config.data.frame_rate_hz
    est = StreamingSpeedEstimator(model, transform, proc, t, k, max_gap, "cpu")

    h, w = tiny_config.preprocessing.image_size
    n = 20
    frames = np.random.randint(0, 255, (n, h, w, 3), dtype=np.uint8)
    timestamps = np.arange(n) * 0.1
    samples, summary = benchmark_end_to_end(est, frames, timestamps, num_warmup=2, num_iters=5)
    assert len(samples) == 5
    assert (samples > 0).all()
    assert summary["n"] == 5
    assert summary["p95_ms"] >= summary["median_ms"]
