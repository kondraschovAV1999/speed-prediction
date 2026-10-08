"""Shared contracts: exceptions, dataclasses and protocols.

This module is the single source of truth for the shapes and types passed
between modules.  Do not change these definitions without orchestrator
approval (see docs/PLAN.md section 5.1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class PipelineError(Exception):
    """Base class for all project-specific errors."""


class ConfigError(PipelineError):
    """Raised on invalid or inconsistent configuration."""


class DataValidationError(PipelineError):
    """Raised when input data violates an invariant (gaps, duplicates, etc.)."""


class EmptyEvaluationError(PipelineError):
    """Raised when evaluation is requested on an empty set of predictions."""


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #
class FrameSource(Protocol):
    """Lazy-friendly frame access. M1: InMemoryFrames. M2: video-backed."""

    def __len__(self) -> int: ...

    def get(self, indices: Sequence[int]) -> np.ndarray:  # uint8 [k, H, W, C]
        ...


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpeedSequence:
    """A time-ordered run of frames with per-frame speed labels."""

    sequence_id: str  # unique; gap-split children are "<orig>#<k>"
    group_id: str  # source drive/video; SPLITS ARE MADE ON THIS
    frames: FrameSource
    speeds_mph: np.ndarray  # float64 [N]; NaN = missing label
    timestamps: np.ndarray  # float64 [N] seconds, strictly increasing after validation


@dataclass(frozen=True)
class SpeedPrediction:
    """Streaming-inference output (one per new frame once the buffer is full)."""

    sequence_id: str
    frame_index: int  # index of the window's LAST frame within the sequence
    timestamp: float
    raw_speed_mph: float  # model output, before any post-processing
    final_speed_mph: float  # exactly 0.0 when stationary
    stationary: bool
    inference_ms: float  # NaN if not measured


@dataclass(frozen=True)
class PostProcessResult:
    final_speed_mph: float
    stationary: bool


# --------------------------------------------------------------------------- #
# Evaluation / run containers
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationResult:
    split: str
    n_samples: int
    overall: dict  # final speed vs truth: mae, rmse, r2, median_ae (None if undefined)
    overall_raw: dict  # same metrics on raw_speed_mph
    buckets: dict  # name -> {n, mae, rmse, r2, median_ae}
    stationary: dict  # precision, recall, f1, tp, fp, fn, tn, n_true_stationary
    latency: dict | None = field(default=None)  # filled by benchmark, not evaluation

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict (NaN/inf mapped to ``None``)."""
        return _json_safe(
            {
                "split": self.split,
                "n_samples": self.n_samples,
                "overall": self.overall,
                "overall_raw": self.overall_raw,
                "buckets": self.buckets,
                "stationary": self.stationary,
                "latency": self.latency,
            }
        )


@dataclass(frozen=True)
class RunPaths:
    """Filesystem locations for a single run (all under ``output/``)."""

    run_name: str
    run_dir: Path
    checkpoints_dir: Path
    reports_dir: Path
    figures_dir: Path
    log_file: Path
    config_path: Path
    metadata_path: Path
    metrics_path: Path
    predictions_path: Path
    split_path: Path


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _json_safe(obj: Any) -> Any:
    """Recursively convert NaN/inf floats to ``None`` and numpy scalars to python."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    return obj
