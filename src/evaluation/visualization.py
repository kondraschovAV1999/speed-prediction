"""Figure generation (WP7). All plots use the Agg backend and never ``show()``."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.core.schemas import EvaluationResult, RunPaths  # noqa: E402


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


def plot_speed_timeseries(df: pd.DataFrame, out: Path) -> None:
    """Plot three representative sequences (low/med/high mean speed), full length."""
    means = df.groupby("sequence_id")["true_speed_mph"].mean()
    if means.empty:
        return
    order = means.sort_values().index.tolist()
    picks = list(dict.fromkeys([order[0], order[len(order) // 2], order[-1]]))
    fig, axes = plt.subplots(len(picks), 1, figsize=(8, 3 * len(picks)), squeeze=False)
    for ax, seq_id in zip(axes[:, 0], picks, strict=True):
        sub = df[df["sequence_id"] == seq_id].sort_values("frame_index")
        ax.plot(sub["frame_index"], sub["true_speed_mph"], label="true")
        ax.plot(sub["frame_index"], sub["final_speed_mph"], label="predicted")
        ax.set_title(f"{seq_id}  (mean {float(means[seq_id]):.1f} mph)")
        ax.set_xlabel("frame index")
        ax.set_ylabel("speed (mph)")
        ax.legend(fontsize="small")
    fig.tight_layout()
    _save(fig, out)


def plot_speed_profile(df: pd.DataFrame, out: Path) -> None:
    """Mean speed (true + predicted) vs frame index, with ±1 std bands."""
    agg = (
        df.groupby("frame_index")
        .agg(
            true_mean=("true_speed_mph", "mean"),
            true_std=("true_speed_mph", "std"),
            pred_mean=("final_speed_mph", "mean"),
            pred_std=("final_speed_mph", "std"),
        )
        .sort_index()
    )
    x = agg.index.to_numpy(dtype=float)
    true_std = agg["true_std"].fillna(0.0).to_numpy(dtype=float)
    pred_std = agg["pred_std"].fillna(0.0).to_numpy(dtype=float)
    fig, ax = plt.subplots()
    ax.plot(x, agg["true_mean"], color="tab:blue", label="true mean")
    ax.fill_between(
        x, agg["true_mean"] - true_std, agg["true_mean"] + true_std, color="tab:blue", alpha=0.2
    )
    ax.plot(x, agg["pred_mean"], color="tab:orange", label="predicted mean")
    ax.fill_between(
        x, agg["pred_mean"] - pred_std, agg["pred_mean"] + pred_std, color="tab:orange", alpha=0.2
    )
    ax.set_xlabel("frame index")
    ax.set_ylabel("speed (mph)")
    ax.legend()
    _save(fig, out)


def plot_calibration(df: pd.DataFrame, out: Path) -> None:
    """Mean predicted speed ± 1 std vs true speed (binned), with y=x diagonal."""
    truth = df["true_speed_mph"].to_numpy(dtype=float)
    pred = df["final_speed_mph"].to_numpy(dtype=float)
    if truth.size == 0:
        return
    bins = np.linspace(float(truth.min()), float(truth.max()), 15)
    bin_idx = np.digitize(truth, bins)
    centers: list[float] = []
    means: list[float] = []
    stds: list[float] = []
    for i in range(1, len(bins)):
        mask = bin_idx == i
        if not mask.any():
            continue
        centers.append((bins[i - 1] + bins[i]) / 2.0)
        means.append(float(pred[mask].mean()))
        stds.append(float(pred[mask].std()))
    fig, ax = plt.subplots()
    ax.errorbar(centers, means, yerr=stds, fmt="o", capsize=3, color="tab:orange")
    lo, hi = float(truth.min()), float(truth.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("true speed (mph)")
    ax.set_ylabel("predicted speed (mph)")
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
    predictions: dict[str, pd.DataFrame],
    results: dict[str, EvaluationResult],
    latency_samples: np.ndarray,
    paths: RunPaths,
) -> list[Path]:
    """Generate all figures under ``paths.figures_dir`` and return their paths.

    Per-split: predicted-vs-actual + error-distribution. Test-only: representative
    speed timeseries, average speed profile, calibration, bucket errors, stationary
    confusion. Plus latency.
    """
    figures_dir = paths.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    for split_name, df in predictions.items():
        plot_predicted_vs_actual(df, figures_dir / f"predicted_vs_actual_{split_name}.png")
        plot_error_distribution(df, figures_dir / f"error_distribution_{split_name}.png")
        names.append(f"predicted_vs_actual_{split_name}.png")
        names.append(f"error_distribution_{split_name}.png")

    if "test" in predictions:
        test_df = predictions["test"]
        plot_speed_timeseries(test_df, figures_dir / "speed_timeseries.png")
        plot_speed_profile(test_df, figures_dir / "speed_profile.png")
        plot_calibration(test_df, figures_dir / "calibration.png")
        names.extend(["speed_timeseries.png", "speed_profile.png", "calibration.png"])

    if "test" in results:
        plot_error_by_bucket(results["test"], figures_dir / "error_by_bucket.png")
        plot_stationary_confusion(results["test"], figures_dir / "stationary_confusion_matrix.png")
        names.extend(["error_by_bucket.png", "stationary_confusion_matrix.png"])

    plot_latency(latency_samples, None, figures_dir / "latency.png")
    names.append("latency.png")

    return [figures_dir / name for name in names]
