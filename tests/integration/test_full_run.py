"""Integration tests (WP10): full run, artifacts, failure metadata, CLI smoke."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.config import compose_config
from src.experiment import run_experiment


def _smoke_cfg(tmp_path: Path):
    cfg = compose_config(["experiment=smoke"])
    out = tmp_path / "output"
    OmegaConf.update(cfg, "paths.output_root", str(out))
    OmegaConf.update(cfg, "data.split_manifest", str(tmp_path / "data" / "splits" / "synth.json"))
    return cfg


@pytest.mark.integration
def test_full_run_creates_all_artifacts(tmp_path):
    paths = run_experiment(_smoke_cfg(tmp_path))
    run_dir = paths.run_dir
    for name in (
        "config.yaml",
        "metadata.json",
        "metrics.json",
        "predictions.csv",
        "split.json",
    ):
        assert (run_dir / name).exists(), f"missing {name}"
    assert (run_dir / "checkpoints" / "best.ckpt").exists()
    assert (run_dir / "checkpoints" / "last.ckpt").exists()
    assert (run_dir / "reports" / "latency_samples.csv").exists()
    assert (run_dir / "training_logs").exists()
    assert paths.log_file.exists()
    for fig in (
        "speed_timeseries.png",
        "predicted_vs_actual.png",
        "error_distribution.png",
        "error_by_bucket.png",
        "stationary_confusion_matrix.png",
        "latency.png",
    ):
        assert (paths.figures_dir / fig).exists(), f"missing figure {fig}"


@pytest.mark.integration
def test_metrics_json_content(tmp_path):
    paths = run_experiment(_smoke_cfg(tmp_path))
    data = json.loads((paths.run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert "test" in data["splits"]
    test = data["splits"]["test"]
    for bucket in ("low", "medium", "high"):
        assert test["buckets"][bucket]["n"] > 0
    assert test["stationary"]["n_true_stationary"] > 0
    assert "model_only" in data["latency"]
    assert "end_to_end" in data["latency"]
    for key in ("mae", "rmse", "r2", "median_ae"):
        assert key in test["overall"]
        assert key in test["overall_raw"]


@pytest.mark.integration
def test_stationary_rows_have_zero_final(tmp_path):
    paths = run_experiment(_smoke_cfg(tmp_path))
    import pandas as pd

    df = pd.read_csv(paths.run_dir / "predictions.csv")
    stationary_rows = df[df["stationary_pred"] == True]  # noqa: E712
    assert len(stationary_rows) > 0
    assert (stationary_rows["final_speed_mph"] == 0.0).all()


@pytest.mark.integration
def test_two_runs_distinct_dirs(tmp_path):
    p1 = run_experiment(_smoke_cfg(tmp_path))
    p2 = run_experiment(_smoke_cfg(tmp_path))
    assert p1.run_dir != p2.run_dir
    assert p1.run_name != p2.run_name


@pytest.mark.integration
def test_forced_failure_marks_metadata(tmp_path, monkeypatch):
    import src.experiment as exp

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(exp, "build_datasets", _boom)
    with pytest.raises(RuntimeError):
        run_experiment(_smoke_cfg(tmp_path))
    runs = list((tmp_path / "output" / "runs").iterdir())
    assert len(runs) == 1
    meta = json.loads((runs[0] / "metadata.json").read_text(encoding="utf-8"))
    assert meta["status"] == "failed"


@pytest.mark.integration
def test_train_test_groups_disjoint(tmp_path):
    paths = run_experiment(_smoke_cfg(tmp_path))
    split = json.loads((paths.run_dir / "split.json").read_text(encoding="utf-8"))
    train = set(split["splits"]["train"])
    test = set(split["splits"]["test"])
    assert not (train & test)


@pytest.mark.integration
def test_cli_smoke_subprocess(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    out = tmp_path / "output"
    hydra_dir = tmp_path / "hydra"
    split_manifest = str(tmp_path / "data" / "splits" / "synth.json").replace(chr(92), "/")
    cmd = [
        sys.executable,
        "scripts/train.py",
        "experiment=smoke",
        f"paths.output_root={str(out).replace(chr(92), '/')}",
        f"data.split_manifest={split_manifest}",
        f"hydra.run.dir={str(hydra_dir).replace(chr(92), '/')}",
        "tracking.enabled=false",
    ]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert not (repo_root / "outputs").exists()
    assert not (repo_root / "lightning_logs").exists()
    assert (out / "runs").exists()
