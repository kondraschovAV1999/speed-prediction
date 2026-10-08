"""Runtime benchmark (WP9): model-only and end-to-end latency samples."""

from __future__ import annotations

import time

import numpy as np
import torch

from src.evaluation import latency_summary
from src.pipeline import resolve_device


def benchmark_model(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    device: str,
    num_warmup: int,
    num_iters: int,
) -> tuple[np.ndarray, dict]:
    """Time a single forward pass on a prepared input, batch size 1."""
    dev = resolve_device(device)
    model = model.to(dev).eval()
    example_input = example_input.to(dev)

    with torch.inference_mode():
        for _ in range(num_warmup):
            model(example_input)
        samples = np.empty(num_iters, dtype=float)
        for i in range(num_iters):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(example_input)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            samples[i] = (time.perf_counter() - start) * 1000.0

    summary = latency_summary(samples)
    summary["device"] = str(dev)
    return samples, summary


def benchmark_end_to_end(
    estimator,
    frames: np.ndarray,
    timestamps: np.ndarray,
    num_warmup: int,
    num_iters: int,
) -> tuple[np.ndarray, dict]:
    """Time a full streaming pass (transform + forward + post-process) per frame."""
    n = len(frames)

    for _ in range(num_warmup):
        estimator.reset()
        for j in range(n):
            estimator.update(frames[j], float(timestamps[j]))

    samples = np.empty(num_iters, dtype=float)
    for i in range(num_iters):
        estimator.reset()
        start = time.perf_counter()
        for j in range(n):
            estimator.update(frames[j], float(timestamps[j]))
        samples[i] = (time.perf_counter() - start) * 1000.0 / n

    summary = latency_summary(samples)
    summary["device"] = str(getattr(estimator, "device", "unknown"))
    return samples, summary
