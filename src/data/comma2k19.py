"""comma2k19 ingestion (M2-a): frame sources, discovery, decode, cache, load.

OpenCV is imported lazily inside the functions that need it, so this module can
be imported in offline tests (it only imports numpy/pathlib/stdlib at top level).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.core.config import DataCfg
from src.core.logging_utils import get_logger
from src.core.schemas import DataValidationError, SpeedSequence

logger = get_logger(__name__)

# 1 m/s = 3600 / 1609.344 mph (exact).
MPH_PER_MPS = 2.2369362921


def _require_cv2():
    """Import cv2 or raise a clear, actionable error."""
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:
        raise DataValidationError(
            "OpenCV is required for comma2k19 video decoding. Install with "
            "`pip install -e '.[video]'`."
        ) from exc
    return cv2


class CachedFrameSource:
    """Memory-mapped frame source over an (uncompressed) cached .npz.

    Picklable: only the path is serialized; the mmap'd handle is reopened in the
    worker process (required for ``training.num_workers > 0``).
    """

    def __init__(self, path: str | Path):
        self._path = str(path)
        self._f = np.load(self._path, mmap_mode="r")
        self._n = int(self._f["frames"].shape[0])

    def __len__(self) -> int:
        return self._n

    def get(self, indices: Sequence[int]) -> np.ndarray:  # uint8 [k, H, W, C]
        return np.asarray(self._f["frames"][list(indices)])

    def __getstate__(self) -> dict:
        # The mmap'd npz holds a non-picklable file handle; serialize only the
        # path and reopen it on the other side.
        return {"_path": self._path, "_n": self._n}

    def __setstate__(self, state: dict) -> None:
        self._path = state["_path"]
        self._n = state["_n"]
        self._f = np.load(self._path, mmap_mode="r")


class VideoFrameSource:
    """Lazy cv2 decode of a segment's video.hevc (fallback when use_cache=false)."""

    def __init__(self, segment_dir: Path, n_frames: int, image_size: tuple[int, int]):
        self.segment_dir = segment_dir
        self.n_frames = n_frames
        self.image_size = image_size

    def __len__(self) -> int:
        return self.n_frames

    def get(self, indices: Sequence[int]) -> np.ndarray:
        cv2 = _require_cv2()
        h, w = self.image_size
        cap = cv2.VideoCapture(str(self.segment_dir / "video.hevc"))
        if not cap.isOpened():
            raise DataValidationError(f"Could not open video: {self.segment_dir / 'video.hevc'}")
        out = []
        for idx in sorted(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raise DataValidationError(f"Failed to decode frame {idx} in {self.segment_dir}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.append(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA))
        cap.release()
        return np.stack(out)


def discover_segments(root: Path) -> list[tuple[str, str, Path]]:
    """Return sorted ``(route, seg, segment_dir)`` tuples under ``root``."""
    segments: list[tuple[str, str, Path]] = []
    for route_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for seg_dir in sorted(p for p in route_dir.iterdir() if p.is_dir()):
            if (seg_dir / "global_pose").is_dir() and (seg_dir / "video.hevc").exists():
                segments.append((route_dir.name, seg_dir.name, seg_dir))
    return segments


def decode_segment_frames(segment_dir: Path, image_size: tuple[int, int]) -> np.ndarray:
    """Decode video.hevc to ``uint8 [N, H, W, 3]`` RGB, downscaled to ``image_size``."""
    cv2 = _require_cv2()
    h, w = image_size
    cap = cv2.VideoCapture(str(segment_dir / "video.hevc"))
    if not cap.isOpened():
        raise DataValidationError(f"Could not open video: {segment_dir / 'video.hevc'}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        return np.empty((0, h, w, 3), dtype=np.uint8)
    return np.stack(frames).astype(np.uint8)


def cache_segment(segment_dir: Path, cache_path: Path, image_size: tuple[int, int]) -> dict:
    """Decode + cache one segment to an *uncompressed* .npz; return ``{"n_frames": int}``."""
    global_pose = segment_dir / "global_pose"
    # comma2k19 global_pose arrays are npy-format but stored WITHOUT a .npy extension.
    frame_times = np.load(global_pose / "frame_times")
    velocities = np.load(global_pose / "frame_velocities")
    speeds_mph = np.linalg.norm(velocities, axis=1) * MPH_PER_MPS

    frames = decode_segment_frames(segment_dir, image_size)
    if frames.shape[0] != frame_times.shape[0]:
        raise DataValidationError(
            f"{segment_dir}: decoded {frames.shape[0]} frames but "
            f"frame_times has {frame_times.shape[0]}"
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez (NOT savez_compressed) so the cache is memory-mappable.
    np.savez(cache_path, frames=frames, speeds_mph=speeds_mph, timestamps=frame_times)
    return {"n_frames": int(frame_times.shape[0])}


def build_manifest(cache_dir: Path, segments: dict[str, dict]) -> None:
    """Write manifest.json mapping ``<route>/<seg>`` -> {file, n_frames, group_id}."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "manifest.json").write_text(
        json.dumps({"segments": segments}, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_comma2k19_sequences(cfg: DataCfg, seed: int) -> list[SpeedSequence]:
    """Load cached comma2k19 segments into SpeedSequence objects (sorted, optional cap)."""
    cache_dir = Path(cfg.comma2k19.cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise DataValidationError(
            f"Manifest not found at {manifest_path}; run scripts/preprocess.py first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = sorted(manifest["segments"].items())
    if cfg.comma2k19.max_segments is not None:
        entries = entries[: cfg.comma2k19.max_segments]

    sequences: list[SpeedSequence] = []
    for seg_id, info in entries:
        npz_path = cache_dir / info["file"]
        data = np.load(npz_path, mmap_mode="r")
        sequences.append(
            SpeedSequence(
                sequence_id=seg_id,
                group_id=info["group_id"],
                frames=CachedFrameSource(npz_path),
                speeds_mph=np.asarray(data["speeds_mph"], dtype=float),
                timestamps=np.asarray(data["timestamps"], dtype=float),
            )
        )
    return sequences
