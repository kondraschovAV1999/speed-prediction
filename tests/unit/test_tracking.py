"""Tests for src.tracking (WP8)."""

from __future__ import annotations

from pathlib import Path

import pytest

import src.tracking as trk
from src.config import Config, ExperimentCfg, MLflowCfg, PathsCfg, TrackingCfg
from src.run_manager import create_run
from src.tracking import (
    MLflowTracker,
    NoOpTracker,
    _flatten_params,
    _local_sqlite_uri,
    create_tracker,
)


def test_noop_accepts_all_calls():
    t = NoOpTracker()
    t.start_run("r")
    t.log_params({"a": 1})
    t.log_metrics({"m": 0.5})
    t.log_artifact("x")
    t.finish()


def test_create_tracker_disabled_returns_noop(tiny_config):
    tiny_config.tracking.enabled = False
    paths = create_run(tiny_config)
    assert isinstance(create_tracker(tiny_config, paths), NoOpTracker)


def test_create_tracker_enabled_returns_mlflow(tiny_config):
    tiny_config.tracking.enabled = True
    tiny_config.tracking.backend = "mlflow"
    paths = create_run(tiny_config)
    assert isinstance(create_tracker(tiny_config, paths), MLflowTracker)


def test_flatten_params_dotted_keys():
    flat = _flatten_params({"a": 1, "nested": {"b": 2, "c": {"d": 3}}})
    assert flat == {"a": "1", "nested.b": "2", "nested.c.d": "3"}


def test_flatten_params_truncates_long_values():
    from src.tracking import MAX_PARAM_VAL_LENGTH

    flat = _flatten_params({"k": "x" * (MAX_PARAM_VAL_LENGTH + 1000)})
    assert len(flat["k"]) <= MAX_PARAM_VAL_LENGTH


def test_local_sqlite_uri_encodes_spaces():
    uri = _local_sqlite_uri(Path("C:/out dir"))
    assert uri.startswith("sqlite:///")
    assert "out%20dir" in uri


def test_mlflow_failure_does_not_raise(tiny_config, monkeypatch):
    tiny_config.tracking.enabled = True
    paths = create_run(tiny_config)
    tracker = create_tracker(tiny_config, paths)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(trk.mlflow, "start_run", _boom)
    monkeypatch.setattr(trk.mlflow, "log_params", _boom)
    monkeypatch.setattr(trk.mlflow, "log_metrics", _boom)
    monkeypatch.setattr(trk.mlflow, "log_artifact", _boom)
    monkeypatch.setattr(trk.mlflow, "end_run", _boom)

    # None of these may raise.
    tracker.start_run("x")
    tracker.log_params({"a": 1})
    tracker.log_metrics({"m": 0.5})
    tracker.log_artifact("file.txt")
    tracker.finish()


@pytest.mark.mlflow
def test_mlflow_run_tagged(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    out_root = tmp_path / "out dir"  # include a space to exercise URI encoding
    cfg = Config(
        experiment=ExperimentCfg(name="tt", seed=0),
        paths=PathsCfg(output_root=str(out_root)),
        tracking=TrackingCfg(
            enabled=True,
            backend="mlflow",
            mlflow=MLflowCfg(tracking_uri=None, experiment_name="test-exp"),
        ),
    )
    paths = create_run(cfg)
    tracker = create_tracker(cfg, paths)
    tracker.start_run("my_run")
    tracker.log_params({"a": 1})
    tracker.log_metrics({"m": 0.5})
    tracker.finish()

    client = mlflow.MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    exp = client.get_experiment_by_name("test-exp")
    assert exp is not None
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert any(r.data.tags.get("local_run_name") == "my_run" for r in runs)
