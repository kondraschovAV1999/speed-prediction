"""comma2k19 ingestion (M2-a): frame sources, discovery, decode, cache, load.

OpenCV is imported lazily inside the functions that need it, so this module can
be imported in offline tests (it only imports numpy/pathlib/stdlib at top level).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.core.logging_utils import get_logger
from src.core.schemas import DataValidationError

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
    """Memory-mapped frame source over an (uncompressed) cached .npz."""

    def __init__(self, path: str | Path):
        self._f = np.load(str(path), mmap_mode="r")
        self._n = int(self._f["frames"].shape[0])

    def __len__(self) -> int:
        return self._n

    def get(self, indices: Sequence[int]) -> np.ndarray:  # uint8 [k, H, W, C]
        return np.asarray(self._f["frames"][list(indices)])


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
