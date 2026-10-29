"""Tests for src.evaluation.reporting (WP7)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.core.config import BucketCfg, EvaluationCfg
from src.core.schemas import RunPaths
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.reporting import read_predictions_csv, write_metrics_json, write_predictions_csv


def make_paths(tmp_path) -> RunPaths:
    run_dir = tmp_path / "r"
    return RunPaths(
        run_name="r",
        run_dir=run_dir,
        checkpoints_dir=run_dir / "checkpoints",
        reports_dir=run_dir / "reports",
        figures_dir=tmp_path / "figures" / "r",
        log_file=tmp_path / "r.log",
        config_path=run_dir / "config.yaml",
        metadata_path=run_dir / "metadata.json",
        metrics_path=run_dir / "metrics.json",
        predictions_path=run_dir / "predictions.csv",
        split_path=run_dir / "split.json",
    )


def eval_cfg() -> EvaluationCfg:
    return EvaluationCfg(
        stationary_label_threshold_mph=2.0,
        speed_buckets=[
            BucketCfg("low", 0, 15),
            BucketCfg("medium", 15, 45),
            BucketCfg("high", 45, None),
        ],
    )


def make_df() -> pd.DataFrame:
    true = np.array([0.0, 1.0, 2.0, 10.0, 20.0, 50.0, 80.0])
    return pd.DataFrame(
        {
            "split": ["test"] * 7,
            "sequence_id": ["s1"] * 7,
            "frame_index": np.arange(7),
            "timestamp": np.arange(7) * 0.1,
            "true_speed_mph": true,
            "raw_speed_mph": true.copy(),
            "final_speed_mph": true.copy(),
            "stationary_pred": [True, True, True, False, False, False, False],
            "inference_ms": [float("nan")] * 7,
        }
    )


def test_metrics_json_roundtrip_and_schema(tmp_path):
    paths = make_paths(tmp_path)
    result = evaluate_predictions(make_df(), eval_cfg(), "test")
    latency = {"model_only": {"mean_ms": 1.0}, "end_to_end": {"mean_ms": 2.0}}
    write_metrics_json(paths, {"test": result}, latency, {"best_checkpoint": "best.ckpt"})
    data = json.loads(paths.metrics_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["run_name"] == "r"
    assert set(data["splits"]) == {"test"}
    assert "overall" in data["splits"]["test"]
    assert "buckets" in data["splits"]["test"]
    assert data["latency"]["model_only"]["mean_ms"] == 1.0
    assert data["extra"]["best_checkpoint"] == "best.ckpt"


def test_metrics_json_no_nan(tmp_path):
    paths = make_paths(tmp_path)
    result = evaluate_predictions(make_df(), eval_cfg(), "test")
    write_metrics_json(paths, {"test": result}, None, None)
    text = paths.metrics_path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text


def test_predictions_csv_roundtrip(tmp_path):
    paths = make_paths(tmp_path)
    df = make_df()
    write_predictions_csv(paths, df)
    back = read_predictions_csv(paths.predictions_path)
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
    assert list(back.columns) == expected
    assert np.allclose(back["true_speed_mph"], df["true_speed_mph"])
    assert list(back["sequence_id"]) == list(df["sequence_id"])


def test_no_negative_zero_in_csv(tmp_path):
    paths = make_paths(tmp_path)
    df = make_df()
    df.loc[0, "final_speed_mph"] = -0.0
    write_predictions_csv(paths, df)
    text = paths.predictions_path.read_text(encoding="utf-8")
    assert "-0.0" not in text
