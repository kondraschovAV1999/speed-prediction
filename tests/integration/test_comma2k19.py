"""Real-data comma2k19 integration (skipped unless data/raw/Chunk_1 present)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.comma2k19 import MPH_PER_MPS, decode_segment_frames, discover_segments

pytestmark = pytest.mark.slow

RAW_ROOT = Path("data/raw/Chunk_1")


@pytest.mark.skipif(not RAW_ROOT.exists(), reason="comma2k19 data not present")
def test_decode_one_real_segment():
    segments = discover_segments(RAW_ROOT)
    assert segments, "no segments found"
    seg_dir = segments[0][2]
    gp = seg_dir / "global_pose"
    frame_times = np.load(gp / "frame_times.npy")
    velocities = np.load(gp / "frame_velocities.npy")

    frames = decode_segment_frames(seg_dir, (64, 64))
    assert frames.shape[0] == frame_times.shape[0]
    assert frames.shape[1:] == (64, 64, 3)
    assert frames.dtype == np.uint8
    # 20 Hz, dt = 0.05 s
    assert np.allclose(np.diff(frame_times), 0.05, atol=1e-9)
    # speeds finite, positive after m/s -> mph
    speeds_mph = np.linalg.norm(velocities, axis=1) * MPH_PER_MPS
    assert np.isfinite(speeds_mph).all()
    assert speeds_mph.min() > 0
