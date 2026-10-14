"""Tests for src.data.comma2k19 (offline: no OpenCV, no network)."""

from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest

from src.core.schemas import DataValidationError
from src.data.comma2k19 import CachedFrameSource, discover_segments


@pytest.fixture
def cached_npz(tmp_path: Path) -> Path:
    frames = np.arange(5 * 8 * 8 * 3, dtype=np.uint8).reshape(5, 8, 8, 3)
    path = tmp_path / "seg.npz"
    np.savez(path, frames=frames, speeds_mph=np.ones(5), timestamps=np.arange(5))
    return path


def test_cached_frame_source_len_and_get(cached_npz):
    src = CachedFrameSource(cached_npz)
    assert len(src) == 5
    got = src.get([0, 2])
    assert got.shape == (2, 8, 8, 3)
    assert got.dtype == np.uint8


def test_discover_segments_route_seg(tmp_path: Path):
    seg = tmp_path / "routeA" / "0"
    seg.mkdir(parents=True)
    (seg / "global_pose").mkdir()
    (seg / "video.hevc").write_bytes(b"")
    (tmp_path / "routeA" / "1").mkdir(parents=True)  # no global_pose -> skipped
    assert discover_segments(tmp_path) == [("routeA", "0", seg)]


def test_video_frame_source_missing_cv2_raises(tmp_path: Path, monkeypatch):
    from src.data.comma2k19 import VideoFrameSource

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("No module named 'cv2'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    src = VideoFrameSource(tmp_path, n_frames=1, image_size=(8, 8))
    with pytest.raises(DataValidationError, match="OpenCV"):
        src.get([0])
