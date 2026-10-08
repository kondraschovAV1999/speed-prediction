"""Integration tests for the secondary scripts (WP11)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.config import compose_config
from src.experiment import run_experiment

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("scripts")
    cfg = compose_config(["experiment=smoke"])
    OmegaConf.update(cfg, "paths.output_root", str(tmp / "output"))
    OmegaConf.update(cfg, "data.split_manifest", str(tmp / "data" / "splits" / "synth.json"))
    paths = run_experiment(cfg)
    return tmp, paths


def _run(args, cwd=None):
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.integration
def test_evaluate_script(trained):
    tmp, paths = trained
    proc = _run(["scripts/evaluate.py", paths.run_name, "--output-root", str(tmp / "output")])
    assert proc.returncode == 0, proc.stderr
    eval_dirs = list(paths.reports_dir.glob("eval_*"))
    assert len(eval_dirs) == 1
    assert (eval_dirs[0] / "metrics.json").exists()
    # metrics.json (canonical) untouched
    assert (paths.run_dir / "metrics.json").exists()


@pytest.mark.integration
def test_visualize_script(trained):
    tmp, paths = trained
    proc = _run(["scripts/visualize.py", paths.run_name, "--output-root", str(tmp / "output")])
    assert proc.returncode == 0, proc.stderr
    sub_dirs = [p for p in paths.figures_dir.iterdir() if p.is_dir()]
    assert len(sub_dirs) == 1
    assert len(list(sub_dirs[0].glob("*.png"))) == 6


@pytest.mark.integration
def test_predict_script(trained, tmp_path):
    tmp, paths = trained
    # Build a frame folder from synthetic frames.
    from src.config import validate_config
    from src.data import generate_synthetic_sequences

    cfg = validate_config(compose_config(["experiment=smoke"]))
    cfg.data.synthetic.num_sequences = 1
    cfg.data.synthetic.frames_per_sequence = 20
    seq = generate_synthetic_sequences(cfg.data, seed=7)[0]
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for i in range(len(seq.frames)):
        np.save(frame_dir / f"frame_{i:04d}.npy", seq.frames.get([i])[0])

    checkpoint = paths.checkpoints_dir / "best.ckpt"
    proc = _run(
        [
            "scripts/predict.py",
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(frame_dir),
            "--output-root",
            str(tmp_path / "predict_out"),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    out_runs = list((tmp_path / "predict_out" / "runs").iterdir())
    assert len(out_runs) == 1
    assert (out_runs[0] / "predictions.csv").exists()
