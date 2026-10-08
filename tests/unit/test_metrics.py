"""Tests for src.evaluation metrics."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import BucketCfg, EvaluationCfg
from src.evaluation import (
    evaluate_predictions,
    latency_summary,
    regression_metrics,
    stationary_metrics,
)
from src.schemas import EmptyEvaluationError


def _eval_cfg() -> EvaluationCfg:
    return EvaluationCfg(
        stationary_label_threshold_mph=2.0,
        speed_buckets=[
            BucketCfg("low", 0, 15),
            BucketCfg("medium", 15, 45),
            BucketCfg("high", 45, None),
        ],
    )


# --------------------------------------------------------------------------- #
# Regression metrics (hand-computed)
# --------------------------------------------------------------------------- #
def test_regression_metrics_hand_computed():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = y_true + 0.5
    m = regression_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(0.5)
    assert m["rmse"] == pytest.approx(0.5)
    assert m["median_ae"] == pytest.approx(0.5)
    # R2 = 1 - 1.25/10 = 0.875
    assert m["r2"] == pytest.approx(0.875)


def test_regression_metrics_perfect_prediction():
    y = np.array([3.0, 1.0, 4.0, 2.0])
    m = regression_metrics(y, y)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["median_ae"] == 0.0
    assert m["r2"] == pytest.approx(1.0)


def test_regression_metrics_constant_truth_r2_none():
    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([4.0, 6.0, 5.0])
    m = regression_metrics(y_true, y_pred)
    assert m["r2"] is None
    assert m["mae"] is not None


def test_regression_metrics_n1_r2_none():
    m = regression_metrics(np.array([2.0]), np.array([2.5]))
    assert m["r2"] is None
    assert m["mae"] == pytest.approx(0.5)


def test_regression_metrics_empty_returns_all_none():
    m = regression_metrics(np.array([]), np.array([]))
    assert m == {"mae": None, "rmse": None, "r2": None, "median_ae": None}


# --------------------------------------------------------------------------- #
# Stationary metrics
# --------------------------------------------------------------------------- #
def test_stationary_confusion_hand_computed():
    true_speed = np.array([1.0, 3.0, 0.0, 5.0])
    pred = np.array([True, False, True, True])
    m = stationary_metrics(true_speed, pred, label_threshold=2.0)
    assert m["tp"] == 2
    assert m["fp"] == 1
    assert m["fn"] == 0
    assert m["tn"] == 1
    assert m["n_true_stationary"] == 2
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(0.8)


def test_stationary_truth_exactly_threshold_is_stationary():
    m = stationary_metrics(np.array([2.0]), np.array([True]), label_threshold=2.0)
    assert m["n_true_stationary"] == 1
    assert m["tp"] == 1


def test_stationary_zero_predicted_positives():
    m = stationary_metrics(np.array([1.0, 3.0]), np.array([False, False]), label_threshold=2.0)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["tp"] == 0 and m["fp"] == 0


# --------------------------------------------------------------------------- #
# Latency summary
# --------------------------------------------------------------------------- #
def test_latency_summary():
    s = latency_summary(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))
    assert s["mean_ms"] == pytest.approx(22.0)
    assert s["median_ms"] == pytest.approx(3.0)
    # np.percentile linear interpolation: 0.95*(5-1)=3.8 -> 4 + 0.8*(100-4) = 80.8
    assert s["p95_ms"] == pytest.approx(80.8)
    assert s["fps"] == pytest.approx(1000.0 / 22.0)
    assert s["n"] == 5


# --------------------------------------------------------------------------- #
# evaluate_predictions
# --------------------------------------------------------------------------- #
def _df() -> pd.DataFrame:
    # speeds spanning all three buckets + stationary frames
    true = np.array([0.0, 1.0, 2.0, 10.0, 20.0, 50.0, 80.0])
    pred = true.copy()
    return pd.DataFrame(
        {
            "split": ["test"] * 7,
            "sequence_id": ["s1"] * 7,
            "frame_index": np.arange(7),
            "timestamp": np.arange(7) * 0.1,
            "true_speed_mph": true,
            "raw_speed_mph": pred,
            "final_speed_mph": pred,
            "stationary_pred": [True, True, True, False, False, False, False],
            "inference_ms": [float("nan")] * 7,
        }
    )


def test_evaluate_predictions_buckets():
    result = evaluate_predictions(_df(), _eval_cfg(), "test")
    assert result.split == "test"
    assert result.n_samples == 7
    assert result.buckets["low"]["n"] == 4  # 0,1,2,10
    assert result.buckets["medium"]["n"] == 1  # 20
    assert result.buckets["high"]["n"] == 2  # 50,80
    assert result.stationary["n_true_stationary"] == 3  # 0,1,2
    assert result.overall["mae"] == 0.0  # perfect prediction


def test_evaluate_predictions_empty_bucket():
    cfg = _eval_cfg()
    # no high-speed frames -> high bucket empty
    true = np.array([5.0, 10.0])
    df = pd.DataFrame(
        {
            "sequence_id": ["s1"] * 2,
            "frame_index": [0, 1],
            "true_speed_mph": true,
            "raw_speed_mph": true,
            "final_speed_mph": true,
            "stationary_pred": [False, False],
        }
    )
    result = evaluate_predictions(df, cfg, "test")
    assert result.buckets["high"] == {
        "n": 0,
        "mae": None,
        "rmse": None,
        "r2": None,
        "median_ae": None,
    }
    assert result.buckets["low"]["n"] == 2


def test_evaluate_predictions_empty_raises():
    with pytest.raises(EmptyEvaluationError):
        evaluate_predictions(pd.DataFrame(), _eval_cfg(), "test")


def test_evaluate_predictions_unsorted_raises():
    df = pd.DataFrame(
        {
            "sequence_id": ["s1", "s1", "s1"],
            "frame_index": [0, 2, 1],  # out of order
            "true_speed_mph": [1.0, 2.0, 3.0],
            "raw_speed_mph": [1.0, 2.0, 3.0],
            "final_speed_mph": [1.0, 2.0, 3.0],
            "stationary_pred": [False, False, False],
        }
    )
    with pytest.raises(ValueError):
        evaluate_predictions(df, _eval_cfg(), "test")


def test_to_dict_json_serializable():
    result = evaluate_predictions(_df(), _eval_cfg(), "test")
    # add a None R2 scenario: constant truth in a bucket is not possible here, so
    # just verify to_dict produces JSON-safe output.
    d = result.to_dict()
    json.dumps(d)
    assert d["split"] == "test"
    # no NaN/inf anywhere
    text = json.dumps(d)
    assert "NaN" not in text and "Infinity" not in text
