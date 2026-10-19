"""Tests for src.inference.postprocessing (detector vectors, clip, smoothing, dataframe)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.config import ClipCfg, PostprocessingCfg, SmoothingCfg, StationaryCfg
from src.inference.postprocessing import (
    SpeedPostProcessor,
    StationaryDetector,
    postprocess_dataframe,
)


def make_cfg(
    enter: float = 2.0,
    exit_: float = 2.5,
    n: int = 3,
    m: int = 3,
    smoothing: str = "none",
    clip_enabled: bool = False,
    clip_min: float = 0.0,
    clip_max: float | None = None,
    stationary_enabled: bool = True,
    ema_alpha: float = 0.5,
    median_window: int = 5,
) -> PostprocessingCfg:
    return PostprocessingCfg(
        clip=ClipCfg(enabled=clip_enabled, min_mph=clip_min, max_mph=clip_max),
        smoothing=SmoothingCfg(method=smoothing, ema_alpha=ema_alpha, median_window=median_window),
        stationary=StationaryCfg(
            enabled=stationary_enabled,
            enter_threshold_mph=enter,
            exit_threshold_mph=exit_,
            enter_persistence_frames=n,
            exit_persistence_frames=m,
            initial_state="moving",
        ),
    )


def run(speeds, cfg) -> tuple[np.ndarray, np.ndarray]:
    proc = SpeedPostProcessor(cfg)
    return proc.process_sequence(np.array(speeds, dtype=float))


# --------------------------------------------------------------------------- #
# Reference vectors from docs/PLAN.md section 6.1
# --------------------------------------------------------------------------- #
def test_vector_basic_enter():
    finals, flags = run([5, 1.9, 1.9, 1.9, 1.9], make_cfg())
    assert list(flags) == [False, False, False, True, True]
    assert np.allclose(finals, [5.0, 1.9, 1.9, 1.9, 1.9])


def test_vector_enter_boundary_counts_equal_as_low():
    finals, flags = run([5, 1.9, 1.9, 2.0, 1.9], make_cfg())
    assert list(flags) == [False, False, False, True, True]


def test_vector_counter_reset():
    finals, flags = run([5, 1.9, 5, 1.9, 1.9, 1.9], make_cfg())
    assert list(flags) == [False, False, False, False, False, True]


def test_vector_exit_boundary_strict():
    det = StationaryDetector(
        StationaryCfg(
            enter_threshold_mph=2.0,
            exit_threshold_mph=2.5,
            enter_persistence_frames=3,
            exit_persistence_frames=3,
            initial_state="moving",
        )
    )
    assert [det.update(s) for s in [5.0, 1.9, 1.9, 1.9]] == [False, False, False, True]
    # 2.5 is not > 2.5: stays stopped
    assert [det.update(s) for s in [2.5, 2.5, 2.5]] == [True, True, True]
    # 2.6 > 2.5: exits after M=3 frames
    assert [det.update(s) for s in [2.6, 2.6, 2.6]] == [True, True, False]


# --------------------------------------------------------------------------- #
# Stationary flag is independent of the final speed value
# --------------------------------------------------------------------------- #
def test_stationary_flag_independent_of_final():
    finals, flags = run([5, 1.9, 1.9, 1.9, 1.9], make_cfg())
    # flags still follow the hysteresis state machine
    assert list(flags) == [False, False, False, True, True]
    # but final_speed_mph is the (unclipped, unsmoothed) prediction, never zeroed
    assert np.allclose(finals, [5.0, 1.9, 1.9, 1.9, 1.9])
    # a stationary frame may carry a non-zero final speed
    assert finals[3] == 1.9
    assert bool(flags[3]) is True


def test_negative_zero_normalized():
    cfg = make_cfg(clip_enabled=False, stationary_enabled=True)
    proc = SpeedPostProcessor(cfg)
    # -0.0 input, clip off, smoothing none; detector will mark stationary soon
    finals, flags = proc.process_sequence(np.array([-0.0, -0.0, -0.0]))
    assert list(flags) == [False, False, True]
    assert finals[2] == 0.0
    assert not np.signbit(finals[2])


# --------------------------------------------------------------------------- #
# Clipping
# --------------------------------------------------------------------------- #
def test_clip_enabled():
    cfg = make_cfg(clip_enabled=True, clip_min=0.0, clip_max=50.0, stationary_enabled=False)
    finals, _ = run([-5.0, 100.0], cfg)
    assert np.allclose(finals, [0.0, 50.0])


def test_clip_disabled_preserves_values():
    cfg = make_cfg(clip_enabled=False, stationary_enabled=False)
    finals, _ = run([-5.0, 100.0], cfg)
    assert np.allclose(finals, [-5.0, 100.0])


# --------------------------------------------------------------------------- #
# Causal smoothing (hand-computed)
# --------------------------------------------------------------------------- #
def test_ema_hand_computed():
    cfg = make_cfg(smoothing="ema", ema_alpha=0.5, stationary_enabled=False)
    finals, _ = run([1.0, 3.0], cfg)
    # s0 = 1.0 ; s1 = 0.5*3 + 0.5*1 = 2.0
    assert np.allclose(finals, [1.0, 2.0])


def test_median_hand_computed():
    cfg = make_cfg(smoothing="median", median_window=3, stationary_enabled=False)
    finals, _ = run([1.0, 5.0, 3.0, 4.0], cfg)
    # [1] -> 1 ; [1,5] -> 3 ; [1,5,3] -> 3 ; [5,3,4] -> 4
    assert np.allclose(finals, [1.0, 3.0, 3.0, 4.0])


# --------------------------------------------------------------------------- #
# Error handling and state management
# --------------------------------------------------------------------------- #
def test_nan_raises():
    proc = SpeedPostProcessor(make_cfg())
    with pytest.raises(ValueError):
        proc.step(float("nan"))
    with pytest.raises(ValueError):
        proc.step(float("inf"))


def test_reset_between_sequences():
    proc = SpeedPostProcessor(make_cfg())
    _f, flags_a = proc.process_sequence(np.array([5.0, 1.9, 1.9, 1.9, 1.9]))
    assert bool(flags_a[-1]) is True
    # Second sequence must start from initial_state (moving), not carry state.
    _f, flags_b = proc.process_sequence(np.array([5.0, 5.0, 5.0]))
    assert list(flags_b) == [False, False, False]


def test_process_sequence_equals_step_loop():
    cfg = make_cfg(smoothing="ema", ema_alpha=0.5, clip_enabled=True, clip_min=0.0)
    raw = np.array([3.0, 1.5, 2.0, 4.0, -1.0])
    proc = SpeedPostProcessor(cfg)
    finals, flags = proc.process_sequence(raw)

    proc2 = SpeedPostProcessor(cfg)
    proc2.reset()
    f2 = np.empty(5)
    s2 = np.empty(5, dtype=bool)
    for i, r in enumerate(raw):
        res = proc2.step(r)
        f2[i] = res.final_speed_mph
        s2[i] = res.stationary
    assert np.allclose(finals, f2)
    assert np.array_equal(flags, s2)


def test_disabled_detector_all_false():
    cfg = make_cfg(stationary_enabled=False)
    finals, flags = run([1.0, 1.0, 1.0], cfg)
    assert list(flags) == [False, False, False]
    assert np.allclose(finals, [1.0, 1.0, 1.0])


def test_enter_gt_exit_raises():
    with pytest.raises(ValueError):
        StationaryDetector(StationaryCfg(enter_threshold_mph=3.0, exit_threshold_mph=2.5))


def test_persistence_lt_one_raises():
    with pytest.raises(ValueError):
        StationaryDetector(StationaryCfg(enter_persistence_frames=0))


# --------------------------------------------------------------------------- #
# postprocess_dataframe
# --------------------------------------------------------------------------- #
def _raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["val", "val", "val", "val"],
            "sequence_id": ["s1", "s1", "s1", "s1"],
            "frame_index": [0, 1, 2, 3],
            "timestamp": [0.0, 0.1, 0.2, 0.3],
            "true_speed_mph": [1.9, 1.9, 1.9, 1.9],
            "raw_speed_mph": [1.9, 1.9, 1.9, 1.9],
            "inference_ms": [float("nan")] * 4,
        }
    )


def test_postprocess_dataframe_adds_columns_and_order():
    out = postprocess_dataframe(_raw_df(), make_cfg())
    expected = [
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
    assert list(out.columns) == expected
    # 1.9 for 4 consecutive frames -> stops at index 2 (N=3)
    assert list(out["stationary_pred"]) == [False, False, True, True]
    # final speed keeps the predicted value, even while stationary
    assert out["final_speed_mph"].iloc[2] == 1.9
    assert out["final_speed_mph"].iloc[3] == 1.9


def test_postprocess_dataframe_nonconsecutive_raises():
    df = pd.DataFrame(
        {
            "sequence_id": ["s1", "s1", "s1"],
            "frame_index": [0, 2, 3],
            "raw_speed_mph": [1.0, 1.0, 1.0],
        }
    )
    with pytest.raises(ValueError):
        postprocess_dataframe(df, make_cfg())


def test_postprocess_dataframe_resets_between_sequences():
    df = pd.DataFrame(
        {
            "sequence_id": ["a", "a", "a", "b", "b", "b"],
            "frame_index": [0, 1, 2, 0, 1, 2],
            "raw_speed_mph": [1.9, 1.9, 1.9, 5.0, 5.0, 5.0],
        }
    )
    out = postprocess_dataframe(df, make_cfg())
    # sequence a ends stationary; sequence b restarts in MOVING state
    assert list(out["stationary_pred"]) == [False, False, True, False, False, False]
