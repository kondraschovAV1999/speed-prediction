"""Tests for src.data.preprocessing (transforms + window indexing)."""

from __future__ import annotations

import numpy as np
import torch

from src.core.config import PreprocessingCfg
from src.data.preprocessing import (
    Compose,
    Normalize,
    Resize,
    ToTensor,
    build_transform,
    build_window_index,
    window_frame_indices,
)


def test_build_window_index_count_and_endpoints():
    idx = build_window_index([5, 6], sequence_length=3, stride=1, frame_step=1)
    # seq0 length 5 -> ends 2,3,4 ; seq1 length 6 -> ends 2,3,4,5
    assert len(idx) == 7
    assert idx[0] == (0, 2)
    assert idx[-1] == (1, 5)


def test_window_frame_indices_ascending():
    assert window_frame_indices(5, 3, 1) == [3, 4, 5]
    assert window_frame_indices(6, 3, 2) == [2, 4, 6]


def test_build_window_index_frame_step_2():
    # min_end = (3-1)*2 = 4
    assert build_window_index([10], 3, 1, 2) == [
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (0, 9),
    ]


def test_short_sequence_zero_windows():
    assert build_window_index([3], 5, 1, 1) == []


def test_to_tensor_range_and_shape():
    frame = np.full((4, 4, 3), 255, dtype=np.uint8)
    out = ToTensor()(frame)
    assert out.shape == (3, 4, 4)
    assert out.dtype == torch.float32
    assert float(out.min()) == 1.0
    assert float(out.max()) == 1.0


def test_resize_shape_and_dtype():
    frame = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
    out = Resize([16, 16])(frame)
    assert out.shape == (16, 16, 3)
    assert out.dtype == np.uint8


def test_normalize_applies_mean_std():
    norm = Normalize([0.5, 0.5, 0.5], [2.0, 2.0, 2.0])
    frame = torch.ones(3, 4, 4)
    out = norm(frame)
    assert torch.allclose(out, torch.full((3, 4, 4), 0.25))


def test_build_transform_output():
    cfg = PreprocessingCfg(image_size=[16, 16])
    transform = build_transform(cfg)
    frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    out = transform(frame)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 16, 16)
    assert out.dtype == torch.float32


def test_compose_order():
    # Resize then ToTensor: final is a float32 tensor in [0,1]
    compose = Compose([Resize([8, 8]), ToTensor()])
    frame = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    out = compose(frame)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 8, 8)
    assert float(out.max()) <= 1.0 and float(out.min()) >= 0.0
