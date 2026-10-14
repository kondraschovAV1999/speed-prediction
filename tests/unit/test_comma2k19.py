"""Tests for src.data.comma2k19 (offline: no OpenCV, no network)."""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pytest

import src.data.comma2k19 as c2k
from src.core.config import Comma2k19Cfg, DataCfg
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


def test_cache_segment_mph_conversion(tmp_path: Path, monkeypatch):
    seg = tmp_path / "seg"
    gp = seg / "global_pose"
    gp.mkdir(parents=True)
    velocities = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])  # 3 and 4 m/s
    np.save(gp / "frame_times.npy", np.array([0.0, 0.05]))
    np.save(gp / "frame_velocities.npy", velocities)
    monkeypatch.setattr(
        c2k, "decode_segment_frames", lambda d, s: np.zeros((2, 4, 4, 3), dtype=np.uint8)
    )
    info = c2k.cache_segment(seg, tmp_path / "out.npz", (4, 4))
    assert info["n_frames"] == 2
    data = np.load(tmp_path / "out.npz")
    assert np.allclose(data["speeds_mph"], [3.0 * c2k.MPH_PER_MPS, 4.0 * c2k.MPH_PER_MPS])
    assert data["frames"].shape == (2, 4, 4, 3)


def test_cache_segment_frame_count_mismatch_raises(tmp_path: Path, monkeypatch):
    seg = tmp_path / "seg"
    gp = seg / "global_pose"
    gp.mkdir(parents=True)
    np.save(gp / "frame_times.npy", np.array([0.0, 0.05, 0.10]))  # 3 frames
    np.save(gp / "frame_velocities.npy", np.zeros((3, 3)))
    monkeypatch.setattr(
        c2k, "decode_segment_frames", lambda d, s: np.zeros((2, 4, 4, 3), dtype=np.uint8)
    )
    with pytest.raises(DataValidationError):
        c2k.cache_segment(seg, tmp_path / "out.npz", (4, 4))


def test_load_sequences_ids_and_max_segments(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for i in range(3):
        fname = f"route{i}__0.npz"
        np.savez(
            cache_dir / fname,
            frames=np.zeros((5, 4, 4, 3), dtype=np.uint8),
            speeds_mph=np.ones(5),
            timestamps=np.arange(5),
        )
    manifest = {
        "segments": {
            f"route{i}/0": {"file": f"route{i}__0.npz", "n_frames": 5, "group_id": f"route{i}"}
            for i in range(3)
        }
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))

    cfg = DataCfg(comma2k19=Comma2k19Cfg(cache_dir=str(cache_dir), max_segments=2))
    seqs = c2k.load_comma2k19_sequences(cfg, seed=0)
    assert len(seqs) == 2
    assert seqs[0].sequence_id == "route0/0"
    assert seqs[0].group_id == "route0"
    assert seqs[0].speeds_mph.shape == (5,)
    assert seqs[0].timestamps.shape == (5,)


def test_load_sequences_missing_manifest_raises(tmp_path: Path):
    cfg = DataCfg(comma2k19=Comma2k19Cfg(cache_dir=str(tmp_path / "nope")))
    with pytest.raises(DataValidationError):
        c2k.load_comma2k19_sequences(cfg, seed=0)


def test_preprocess_flow_glue(tmp_path: Path, monkeypatch):
    """discover -> cache -> manifest -> load, end to end (decode mocked)."""
    seg = tmp_path / "raw" / "routeA" / "0"
    gp = seg / "global_pose"
    gp.mkdir(parents=True)
    (seg / "video.hevc").write_bytes(b"")
    np.save(gp / "frame_times.npy", np.array([0.0, 0.05]))
    np.save(gp / "frame_velocities.npy", np.array([[5.0, 0, 0], [0, 6.0, 0]]))
    monkeypatch.setattr(
        c2k, "decode_segment_frames", lambda d, s: np.zeros((2, 4, 4, 3), dtype=np.uint8)
    )

    cache_dir = tmp_path / "cache"
    manifest: dict = {}
    for route, sid, seg_dir in c2k.discover_segments(tmp_path / "raw"):
        fname = f"{route}__{sid}.npz"
        info = c2k.cache_segment(seg_dir, cache_dir / fname, (4, 4))
        manifest[f"{route}/{sid}"] = {
            "file": fname,
            "n_frames": info["n_frames"],
            "group_id": route,
        }
    c2k.build_manifest(cache_dir, manifest)

    cfg = DataCfg(comma2k19=Comma2k19Cfg(cache_dir=str(cache_dir)))
    seqs = c2k.load_comma2k19_sequences(cfg, seed=0)
    assert len(seqs) == 1
    assert seqs[0].sequence_id == "routeA/0"
    assert seqs[0].speeds_mph.shape == (2,)


def test_build_datasets_dispatches_comma2k19(tmp_path: Path):
    from src.core.config import compose_config, validate_config
    from src.data.dataset import build_datasets

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    manifest_segments = {}
    for g in range(6):
        fname = f"route{g}__0.npz"
        np.savez(
            cache_dir / fname,
            frames=np.zeros((20, 8, 8, 3), dtype=np.uint8),
            speeds_mph=np.ones(20) * 20.0,
            timestamps=np.arange(20) * 0.05,
        )
        manifest_segments[f"route{g}/0"] = {"file": fname, "n_frames": 20, "group_id": f"route{g}"}
    (cache_dir / "manifest.json").write_text(json.dumps({"segments": manifest_segments}))

    cfg = validate_config(compose_config(["dataset=comma2k19", "preprocessing.image_size=[8,8]"]))
    cfg.data.comma2k19.cache_dir = str(cache_dir)
    cfg.data.split_manifest = str(tmp_path / "splits" / "m.json")

    datasets = build_datasets(cfg, seed=42)
    assert set(datasets) == {"train", "val", "test"}
    assert all(len(ds) > 0 for ds in datasets.values())
