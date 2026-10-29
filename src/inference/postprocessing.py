"""Causal post-processing and stationary detection (WP5).

Pipeline per frame: ``raw -> clip -> smoothing -> stationary -> final``.

* ``raw_speed_mph`` is stored before clipping (clipping is explicit config).
* Smoothing is causal (EMA or trailing median; no future values).
* The stationary detector is a hysteresis state machine evaluated after
  including the current frame.
* ``final_speed_mph`` is the clipped/smoothed value; ``stationary`` is an
  independent boolean output and never zeroes the speed.
"""

from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np
import pandas as pd

from src.core.config import PostprocessingCfg, StationaryCfg
from src.core.schemas import PostProcessResult

logger = logging.getLogger(__name__)

# Canonical predictions-table column order (docs/PLAN.md section 5.4).
_CANONICAL_COLUMNS = [
    "split",
    "sequence_id",
    "frame_index",
    "timestamp",
    "true_speed_mph",
    "raw_speed_mph",
    "final_speed_mph",
    "stationary_pred",
    "inference_ms",
]


class StationaryDetector:
    """Hysteresis detector: MOVING <-> STOPPED with separate enter/exit persistence."""

    MOVING = "moving"
    STOPPED = "stopped"

    def __init__(self, cfg: StationaryCfg):
        if cfg.enter_threshold_mph > cfg.exit_threshold_mph:
            raise ValueError(
                "enter_threshold_mph must be <= exit_threshold_mph "
                f"(got {cfg.enter_threshold_mph} > {cfg.exit_threshold_mph})"
            )
        if cfg.enter_persistence_frames < 1 or cfg.exit_persistence_frames < 1:
            raise ValueError("persistence frames must be >= 1")
        self._cfg = cfg
        self.reset()

    def reset(self) -> None:
        """Reset to the configured initial state with cleared counters."""
        self._state = self._cfg.initial_state
        self._enter_count = 0
        self._exit_count = 0

    @property
    def stationary(self) -> bool:
        return self._state == self.STOPPED

    def update(self, speed_mph: float) -> bool:
        """Update the state machine with one speed and return ``stationary``."""
        if self._state == self.MOVING:
            if speed_mph <= self._cfg.enter_threshold_mph:
                self._enter_count += 1
            else:
                self._enter_count = 0
            if self._enter_count >= self._cfg.enter_persistence_frames:
                self._state = self.STOPPED
                self._exit_count = 0
        else:  # STOPPED
            if speed_mph > self._cfg.exit_threshold_mph:
                self._exit_count += 1
            else:
                self._exit_count = 0
            if self._exit_count >= self._cfg.exit_persistence_frames:
                self._state = self.MOVING
                self._enter_count = 0
        return self.stationary


class SpeedPostProcessor:
    """Applies clip -> smoothing -> stationary to a stream of raw predictions."""

    def __init__(self, cfg: PostprocessingCfg):
        self._cfg = cfg
        self._detector: StationaryDetector | None = None
        if cfg.stationary.enabled:
            self._detector = StationaryDetector(cfg.stationary)
        else:
            logger.warning("event=stationary_disabled note=stationary_metrics_not_meaningful")
        self._ema_prev: float | None = None
        self._median_buffer: deque[float] = deque()

    def reset(self) -> None:
        """Reset smoothing state and the detector (if enabled)."""
        if self._detector is not None:
            self._detector.reset()
        self._ema_prev = None
        self._median_buffer.clear()

    def step(self, raw_mph: float) -> PostProcessResult:
        """Process one raw prediction and return ``(final_speed_mph, stationary)``."""
        raw = float(raw_mph)
        if math.isnan(raw) or math.isinf(raw):
            raise ValueError(f"raw_speed_mph must be finite, got {raw!r}")

        value = self._clip(raw)
        value = self._smooth(value)

        stationary = False
        if self._detector is not None:
            stationary = self._detector.update(value)

        # Speed and stationary are independent outputs: the detector's state no
        # longer zeroes the speed value.
        final = value
        if final == 0.0:  # normalize -0.0 -> 0.0 (harmless safety net)
            final = 0.0
        return PostProcessResult(final_speed_mph=final, stationary=stationary)

    def process_sequence(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Process a full sequence via ``reset()`` + a loop of ``step()``.

        Returns ``(final_speeds_mph, stationary_flags)``.
        """
        raw = np.asarray(raw, dtype=float)
        self.reset()
        finals = np.empty(raw.shape[0], dtype=float)
        flags = np.empty(raw.shape[0], dtype=bool)
        for i, r in enumerate(raw):
            result = self.step(r)
            finals[i] = result.final_speed_mph
            flags[i] = result.stationary
        return finals, flags

    # -- internal stages ---------------------------------------------------- #
    def _clip(self, value: float) -> float:
        clip = self._cfg.clip
        if not clip.enabled:
            return value
        if clip.min_mph is not None and value < clip.min_mph:
            value = clip.min_mph
        if clip.max_mph is not None and value > clip.max_mph:
            value = clip.max_mph
        return value

    def _smooth(self, value: float) -> float:
        method = self._cfg.smoothing.method
        if method == "none":
            return value
        if method == "ema":
            alpha = self._cfg.smoothing.ema_alpha
            if self._ema_prev is None:
                self._ema_prev = value
                return value
            smoothed = alpha * value + (1.0 - alpha) * self._ema_prev
            self._ema_prev = smoothed
            return smoothed
        if method == "median":
            self._median_buffer.append(value)
            window = self._cfg.smoothing.median_window
            if len(self._median_buffer) > window:
                self._median_buffer.popleft()
            return float(np.median(np.asarray(self._median_buffer, dtype=float)))
        raise ValueError(f"Unknown smoothing method: {method!r}")


def postprocess_dataframe(df: pd.DataFrame, cfg: PostprocessingCfg) -> pd.DataFrame:
    """Add ``final_speed_mph`` and ``stationary_pred`` to a raw predictions table.

    Groups by ``sequence_id``, orders by ``frame_index``, resets the processor
    between sequences, and requires stride-1 consecutive ``frame_index`` values
    within each sequence (raises :class:`ValueError` otherwise).
    """
    required = {"sequence_id", "frame_index", "raw_speed_mph"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"predictions table missing required columns: {sorted(missing)}")

    df = df.sort_values(["sequence_id", "frame_index"]).reset_index(drop=True)
    processor = SpeedPostProcessor(cfg)

    finals: list[np.ndarray] = []
    flags: list[np.ndarray] = []
    for _, group in df.groupby("sequence_id", sort=False):
        indices = group["frame_index"].to_numpy(dtype=np.int64)
        if indices.shape[0] > 1 and np.any(np.diff(indices) != 1):
            raise ValueError(
                f"sequence {group['sequence_id'].iloc[0]!r} has non-consecutive frame_index"
            )
        f, s = processor.process_sequence(group["raw_speed_mph"].to_numpy(dtype=float))
        finals.append(f)
        flags.append(s)

    df["final_speed_mph"] = np.concatenate(finals)
    df["stationary_pred"] = np.concatenate(flags)

    present = [c for c in _CANONICAL_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in _CANONICAL_COLUMNS]
    return df[present + extra].reset_index(drop=True)
