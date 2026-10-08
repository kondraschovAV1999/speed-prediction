"""Tests for src.visualization (WP7)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BucketCfg, EvaluationCfg
from src.evaluation import evaluate_predictions
from src.schemas import RunPaths
from src.visualization import (
    generate_all_figures,
    plot_error_distribution,
    plot_predicted_vs_actual,
    plot_speed_timeseries,
)


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


def make_df(speeds=None, stationary=None) -> pd.DataFrame:
    true = np.array([0.0, 1.0, 2.0, 10.0, 20.0, 50.0, 80.0] if speeds is None else speeds)
    n = len(true)
    if stationary is None:
        flags = [bool(s <= 2.0) for s in true]
    else:
        flags = list(stationary)
    return pd.DataFrame(
        {
            "split": ["test"] * n,
            "sequence_id": ["s1"] * n,
            "frame_index": np.arange(n),
            "timestamp": np.arange(n) * 0.1,
            "true_speed_mph": true,
            "raw_speed_mph": true.copy(),
            "final_speed_mph": true.copy(),
            "stationary_pred": flags,
            "inference_ms": [float("nan")] * n,
        }
    )


def test_generate_all_figures_returns_six_pngs(tmp_path):
    paths = make_paths(tmp_path)
    df = make_df()
    result = evaluate_predictions(df, eval_cfg(), "test")
    fig_paths = generate_all_figures(df, result, np.array([1.0, 2.0, 3.0]), paths)
    assert len(fig_paths) == 6
    for p in fig_paths:
        assert p.exists()
        assert p.stat().st_size > 0


def test_empty_bucket_does_not_raise(tmp_path):
    paths = make_paths(tmp_path)
    df = make_df(speeds=[5.0, 10.0, 12.0])  # no high bucket
    result = evaluate_predictions(df, eval_cfg(), "test")
    assert result.buckets["high"]["mae"] is None
    fig_paths = generate_all_figures(df, result, np.array([1.0]), paths)
    assert len(fig_paths) == 6


def test_single_class_stationary_does_not_raise(tmp_path):
    paths = make_paths(tmp_path)
    df = make_df(speeds=[10.0, 20.0, 30.0, 40.0], stationary=[False, False, False, False])
    result = evaluate_predictions(df, eval_cfg(), "test")
    assert result.stationary["n_true_stationary"] == 0
    fig_paths = generate_all_figures(df, result, np.array([1.0]), paths)
    assert len(fig_paths) == 6


def test_individual_plots_create_files(tmp_path):
    paths = make_paths(tmp_path)
    df = make_df()
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_predicted_vs_actual(df, paths.figures_dir / "a.png")
    plot_speed_timeseries(df, paths.figures_dir / "b.png")
    plot_error_distribution(df, paths.figures_dir / "c.png")
    for name in ("a.png", "b.png", "c.png"):
        assert (paths.figures_dir / name).stat().st_size > 0
