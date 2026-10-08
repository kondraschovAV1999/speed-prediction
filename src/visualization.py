"""Figure generation (WP7). All plots use the Agg backend and never ``show()``."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.schemas import EvaluationResult, RunPaths  # noqa: E402


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_predicted_vs_actual(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots()
    truth = df["true_speed_mph"].to_numpy(dtype=float)
    pred = df["final_speed_mph"].to_numpy(dtype=float)
    ax.scatter(truth, pred, s=4, alpha=0.6)
    lo, hi = float(truth.min()), float(truth.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("true speed (mph)")
    ax.set_ylabel("predicted speed (mph)")
    _save(fig, out)


def plot_speed_timeseries(df: pd.DataFrame, out: Path, sequence_id: str | None = None) -> None:
    if sequence_id is None:
        sequence_id = str(df["sequence_id"].iloc[0])
    sub = df[df["sequence_id"] == sequence_id].sort_values("frame_index")
    fig, ax = plt.subplots()
    ax.plot(sub["frame_index"], sub["true_speed_mph"], label="true")
    ax.plot(sub["frame_index"], sub["final_speed_mph"], label="predicted")
    ax.set_xlabel("frame index")
    ax.set_ylabel("speed (mph)")
    ax.legend()
    _save(fig, out)


def plot_error_distribution(df: pd.DataFrame, out: Path) -> None:
    errors = (df["final_speed_mph"] - df["true_speed_mph"]).to_numpy(dtype=float)
    fig, ax = plt.subplots()
    ax.hist(errors, bins=30)
    ax.set_xlabel("error (mph)")
    ax.set_ylabel("count")
    _save(fig, out)


def plot_error_by_bucket(result: EvaluationResult, out: Path) -> None:
    names = list(result.buckets.keys())
    maes = [result.buckets[n].get("mae") for n in names]
    values = [0.0 if m is None else float(m) for m in maes]
    fig, ax = plt.subplots()
    ax.bar(names, values)
    ax.set_xlabel("bucket")
    ax.set_ylabel("MAE (mph)")
    _save(fig, out)


def plot_stationary_confusion(result: EvaluationResult, out: Path) -> None:
    s = result.stationary
    cm = np.array([[s["tn"], s["fp"]], [s["fn"], s["tp"]]], dtype=float)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["moving", "stationary"])
    ax.set_yticklabels(["moving", "stationary"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    _save(fig, out)


def plot_latency(samples_ms: np.ndarray, summary: dict | None, out: Path) -> None:
    samples = np.asarray(samples_ms, dtype=float)
    fig, ax = plt.subplots()
    if samples.size:
        ax.hist(samples, bins=30)
    ax.set_xlabel("latency (ms)")
    ax.set_ylabel("count")
    _save(fig, out)


def generate_all_figures(
    df: pd.DataFrame,
    result: EvaluationResult,
    latency_samples: np.ndarray,
    paths: RunPaths,
) -> list[Path]:
    """Generate all six figures under ``paths.figures_dir`` and return their paths."""
    figures_dir = paths.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_speed_timeseries(df, figures_dir / "speed_timeseries.png")
    plot_predicted_vs_actual(df, figures_dir / "predicted_vs_actual.png")
    plot_error_distribution(df, figures_dir / "error_distribution.png")
    plot_error_by_bucket(result, figures_dir / "error_by_bucket.png")
    plot_stationary_confusion(result, figures_dir / "stationary_confusion_matrix.png")
    plot_latency(latency_samples, None, figures_dir / "latency.png")
    return [
        figures_dir / name
        for name in (
            "speed_timeseries.png",
            "predicted_vs_actual.png",
            "error_distribution.png",
            "error_by_bucket.png",
            "stationary_confusion_matrix.png",
            "latency.png",
        )
    ]
