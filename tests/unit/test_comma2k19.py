"""Tests for src.data.comma2k19 (offline: no OpenCV, no network)."""

from __future__ import annotations

import builtins
import io
import json
from pathlib import Path

import numpy as np
import pytest

import src.data.comma2k19 as c2k
from src.core.config import Comma2k19Cfg, DataCfg
from src.core.schemas import DataValidationError
from src.data.comma2k19 import CachedFrameSource, discover_segments


def _write_npy(path: Path, arr: np.ndarray) -> None:
    """Write an npy-format file WITHOUT a .npy extension (mimics comma2k19 data)."""
    buf = io.BytesIO()
    np.save(buf, arr)
    path.write_bytes(buf.getvalue())


def _write_cache(cache_dir: Path, base: str, frames, speeds, timestamps) -> tuple[str, str]:
    """Write a segment's frames (``.npy``) + meta (``.npz``); return the two filenames."""
    np.save(cache_dir / f"{base}.frames.npy", np.asarray(frames))
    np.savez(
        cache_dir / f"{base}.meta.npz",
        speeds_mph=np.asarray(speeds),
        timestamps=np.asarray(timestamps),
    )
    return f"{base}.frames.npy", f"{base}.meta.npz"


@pytest.fixture
def cached_frames(tmp_path: Path) -> Path:
    frames = np.arange(5 * 8 * 8 * 3, dtype=np.uint8).reshape(5, 8, 8, 3)
    path = tmp_path / "seg.frames.npy"
    np.save(path, frames)
    return path


def test_cached_frame_source_len_and_get(cached_frames):
    src = CachedFrameSource(cached_frames)
    assert len(src) == 5
    got = src.get([0, 2])
    assert got.shape == (2, 8, 8, 3)
    assert got.dtype == np.uint8


def test_cached_frame_source_picklable(cached_frames):
    import pickle

    src = CachedFrameSource(cached_frames)
    restored = pickle.loads(pickle.dumps(src))
    assert len(restored) == 5
    got = restored.get([0, 2])
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
    _write_npy(gp / "frame_times", np.array([0.0, 0.05]))
    _write_npy(gp / "frame_velocities", velocities)
    monkeypatch.setattr(
        c2k, "decode_segment_frames", lambda d, s: np.zeros((2, 4, 4, 3), dtype=np.uint8)
    )
    info = c2k.cache_segment(seg, tmp_path / "out.frames.npy", tmp_path / "out.meta.npz", (4, 4))
    assert info["n_frames"] == 2
    frames = np.load(tmp_path / "out.frames.npy")
    meta = np.load(tmp_path / "out.meta.npz")
    assert np.allclose(meta["speeds_mph"], [3.0 * c2k.MPH_PER_MPS, 4.0 * c2k.MPH_PER_MPS])
    assert frames.shape == (2, 4, 4, 3)


def test_cache_segment_frame_count_mismatch_raises(tmp_path: Path, monkeypatch):
    seg = tmp_path / "seg"
    gp = seg / "global_pose"
    gp.mkdir(parents=True)
    _write_npy(gp / "frame_times", np.array([0.0, 0.05, 0.10]))  # 3 frames
    _write_npy(gp / "frame_velocities", np.zeros((3, 3)))
    monkeypatch.setattr(
        c2k, "decode_segment_frames", lambda d, s: np.zeros((2, 4, 4, 3), dtype=np.uint8)
    )
    with pytest.raises(DataValidationError):
        c2k.cache_segment(seg, tmp_path / "out.frames.npy", tmp_path / "out.meta.npz", (4, 4))


def test_load_sequences_ids_and_max_segments(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for i in range(3):
        _write_cache(
            cache_dir,
            f"route{i}__0",
            np.zeros((5, 4, 4, 3), dtype=np.uint8),
            np.ones(5),
            np.arange(5),
        )
    manifest = {
        "segments": {
            f"route{i}/0": {
                "frames_file": f"route{i}__0.frames.npy",
                "meta_file": f"route{i}__0.meta.npz",
                "n_frames": 5,
                "group_id": f"route{i}",
            }
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
    _write_npy(gp / "frame_times", np.array([0.0, 0.05]))
    _write_npy(gp / "frame_velocities", np.array([[5.0, 0, 0], [0, 6.0, 0]]))
    monkeypatch.setattr(
        c2k, "decode_segment_frames", lambda d, s: np.zeros((2, 4, 4, 3), dtype=np.uint8)
    )

    cache_dir = tmp_path / "cache"
    manifest: dict = {}
    for route, sid, seg_dir in c2k.discover_segments(tmp_path / "raw"):
        base = f"{route}__{sid}"
        info = c2k.cache_segment(
            seg_dir, cache_dir / f"{base}.frames.npy", cache_dir / f"{base}.meta.npz", (4, 4)
        )
        manifest[f"{route}/{sid}"] = {
            "frames_file": f"{base}.frames.npy",
            "meta_file": f"{base}.meta.npz",
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
        _write_cache(
            cache_dir,
            f"route{g}__0",
            np.zeros((20, 8, 8, 3), dtype=np.uint8),
            np.ones(20) * 20.0,
            np.arange(20) * 0.05,
        )
        manifest_segments[f"route{g}/0"] = {
            "frames_file": f"route{g}__0.frames.npy",
            "meta_file": f"route{g}__0.meta.npz",
            "n_frames": 20,
            "group_id": f"route{g}",
        }
    (cache_dir / "manifest.json").write_text(json.dumps({"segments": manifest_segments}))

    cfg = validate_config(
        compose_config(
            [
                "dataset=comma2k19",
                "preprocessing.image_size=[8,8]",
                "data.sequence_length=5",
                "data.frame_step=1",
                "data.window_stride=1",
            ]
        )
    )
    cfg.data.comma2k19.cache_dir = str(cache_dir)
    cfg.data.split_manifest = str(tmp_path / "splits" / "m.json")

    datasets = build_datasets(cfg, seed=42)
    assert set(datasets) == {"train", "val", "test"}
    assert all(len(ds) > 0 for ds in datasets.values())
