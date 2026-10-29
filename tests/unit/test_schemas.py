"""Tests for src.core.schemas (dataclass construction and JSON serialization)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.core.schemas import (
    EvaluationResult,
    PostProcessResult,
    RunPaths,
    SpeedPrediction,
)


def _result() -> EvaluationResult:
    return EvaluationResult(
        split="test",
        n_samples=3,
        overall={"mae": 1.5, "rmse": 2.0, "r2": None, "median_ae": 1.0},
        overall_raw={"mae": 1.6, "rmse": 2.1, "r2": 0.5, "median_ae": 1.1},
        buckets={
            "low": {"n": 2, "mae": 1.0, "rmse": 1.2, "r2": None, "median_ae": 0.8},
            "high": {"n": 0, "mae": None, "rmse": None, "r2": None, "median_ae": None},
        },
        stationary={
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
            "tp": 1,
            "fp": 1,
            "fn": 1,
            "tn": 1,
            "n_true_stationary": 2,
        },
        latency=None,
    )


def test_evaluation_result_to_dict_is_json_serializable():
    d = _result().to_dict()
    # json.dumps must not raise (no NaN/inf)
    json.dumps(d)
    assert d["split"] == "test"
    assert d["n_samples"] == 3
    assert d["overall"]["r2"] is None
    assert d["buckets"]["high"]["n"] == 0


def test_to_dict_maps_nan_to_none():
    r = EvaluationResult("test", 1, {"mae": float("nan")}, {}, {}, {}, None)
    d = r.to_dict()
    assert d["overall"]["mae"] is None


def test_to_dict_maps_inf_to_none():
    r = EvaluationResult("test", 1, {"rmse": float("inf")}, {}, {}, {}, None)
    d = r.to_dict()
    assert d["overall"]["rmse"] is None


def test_speed_prediction_construction():
    p = SpeedPrediction(
        sequence_id="s1",
        frame_index=7,
        timestamp=0.7,
        raw_speed_mph=3.2,
        final_speed_mph=0.0,
        stationary=True,
        inference_ms=float("nan"),
    )
    assert p.sequence_id == "s1"
    assert p.frame_index == 7
    assert p.stationary is True
    assert math.isnan(p.inference_ms)


def test_postprocess_result_construction():
    r = PostProcessResult(final_speed_mph=0.0, stationary=True)
    assert r.final_speed_mph == 0.0
    assert r.stationary is True


def test_runpaths_construction():
    rp = RunPaths(
        run_name="r",
        run_dir=Path("x/r"),
        checkpoints_dir=Path("x/r/checkpoints"),
        reports_dir=Path("x/r/reports"),
        figures_dir=Path("x/figures/r"),
        log_file=Path("x/logs/r.log"),
        config_path=Path("x/r/config.yaml"),
        metadata_path=Path("x/r/metadata.json"),
        metrics_path=Path("x/r/metrics.json"),
        predictions_path=Path("x/r/predictions.csv"),
        split_path=Path("x/r/split.json"),
    )
    assert rp.run_name == "r"
    assert rp.metrics_path.name == "metrics.json"
