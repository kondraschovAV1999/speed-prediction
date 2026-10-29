"""Figure generation (WP7). All plots use the Agg backend and never ``show()``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import is_dataclass, fields
from pathlib import Path
from typing import Any

import matplotlib

from src.core.config import TimeseriesCfg

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.core.schemas import EvaluationResult, RunPaths  # noqa: E402

VisualizationConfig = Mapping[str, Any]

_DEFAULTS: dict[str, Any] = {
    "fig_size": [8, 6],
    "timeseries": {
        "width": 10,
        "height_per_sequence": 3,
    },
    "font_size": 12,
    "title_font_size": 14,
    "label_font_size": 12,
    "tick_font_size": 10,
    "legend_font_size": 10,
    "dpi": 300,
    "bbox_inches": "tight",
}


def _get(cfg, key: str):
    if cfg is not None:
        if isinstance(cfg, Mapping) and key in cfg:
            return cfg[key]
        if is_dataclass(cfg) and not isinstance(cfg, type):
            if any(f.name == key for f in fields(cfg)):
                return getattr(cfg, key)

    defaults = {
        "fig_size": (4.3, 2.7),
        "timeseries": TimeseriesCfg(),
        "font_size": 12.0,
        "title_font_size": 14.0,
        "label_font_size": 16.0,
        "tick_font_size": 10.0,
        "legend_font_size": 10.0,
        "dpi": 300,
        "bbox_inches": "tight",
    }
    return defaults[key]


def _subplots(
        cfg: VisualizationConfig | None,
        *,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[float, float] | None = None,
        squeeze: bool = True,
):
    """Create a figure and explicitly configure its typography."""
    if figsize is None:
        figsize = tuple(_get(cfg, "fig_size"))

    with plt.rc_context(
            {
                "font.size": _get(cfg, "font_size"),
                "axes.titlesize": _get(cfg, "title_font_size"),
                "axes.labelsize": _get(cfg, "label_font_size"),
                "xtick.labelsize": _get(cfg, "tick_font_size"),
                "ytick.labelsize": _get(cfg, "tick_font_size"),
                "legend.fontsize": _get(cfg, "legend_font_size"),
            }
    ):
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            squeeze=squeeze,
        )

    return fig, axes


def _apply_typography(
        ax,
        cfg: VisualizationConfig | None,
) -> None:
    """Apply configured font sizes to existing axes and text."""
    ax.title.set_fontsize(_get(cfg, "title_font_size"))
    ax.xaxis.label.set_fontsize(_get(cfg, "label_font_size"))
    ax.yaxis.label.set_fontsize(_get(cfg, "label_font_size"))
    ax.tick_params(
        axis="both",
        labelsize=_get(cfg, "tick_font_size"),
    )

    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(_get(cfg, "legend_font_size"))

    for text in ax.texts:
        text.set_fontsize(_get(cfg, "font_size"))


def _save(
        fig,
        out: Path,
        cfg: VisualizationConfig | None = None,
) -> None:
    """Apply typography, save the figure, and close it."""
    for ax in fig.axes:
        _apply_typography(ax, cfg)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        dpi=_get(cfg, "dpi"),
        bbox_inches=_get(cfg, "bbox_inches"),
    )
    plt.close(fig)


def plot_predicted_vs_actual(
        df: pd.DataFrame,
        out: Path,
        cfg: VisualizationConfig | None = None,
) -> None:
    """Plot predicted speed against actual speed."""
    if df.empty:
        return

    fig, ax = _subplots(cfg)

    truth = df["true_speed_mph"].to_numpy(dtype=float)
    pred = df["final_speed_mph"].to_numpy(dtype=float)

    ax.scatter(truth, pred, s=4, alpha=0.6)

    lo = float(min(truth.min(), pred.min()))
    hi = float(max(truth.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)

    ax.set_xlabel("true (mph)")
    ax.set_ylabel("predicted (mph)")

    _save(fig, out, cfg)


def plot_speed_timeseries(
        df: pd.DataFrame,
        out: Path,
        cfg: VisualizationConfig | None = None,
) -> list[Path]:
    """Plot low-, medium-, and high-mean-speed sequences as separate figures."""
    means = df.groupby("sequence_id")["true_speed_mph"].mean()

    if means.empty:
        return []

    order = means.sort_values().index.tolist()
    picks = list(
        dict.fromkeys(
            [order[0], order[len(order) // 2], order[-1]]
        )
    )

    labels = ["low", "medium", "high"]
    ts_cfg = _get(cfg, "timeseries")
    figsize = (
        float(ts_cfg.width),
        float(ts_cfg.height_per_sequence),
    )

    output_paths: list[Path] = []

    for label, seq_id in zip(labels, picks, strict=False):
        sub = df[df["sequence_id"] == seq_id].sort_values("frame_index")

        fig, ax = _subplots(cfg, figsize=figsize)

        ax.plot(
            sub["frame_index"],
            sub["true_speed_mph"],
            label="true",
        )
        ax.plot(
            sub["frame_index"],
            sub["final_speed_mph"],
            label="predicted",
        )

        # ax.set_title(
        #     f"{seq_id} (mean {float(means[seq_id]):.1f} mph)"
        # )
        ax.set_xlabel("frame")
        ax.set_ylabel("speed (mph)")
        ax.legend(fontsize=_get(cfg, "legend_font_size"))

        fig.tight_layout()

        # Use the original output name with the sequence category appended.
        out_path = out.with_name(f"{out.stem}_{label}{out.suffix}")
        _save(fig, out_path, cfg)
        output_paths.append(out_path)

    return output_paths


def plot_speed_profile(
        df: pd.DataFrame,
        out: Path,
        cfg: VisualizationConfig | None = None,
) -> None:
    """Plot mean true and predicted speeds with plus/minus one std bands."""
    if df.empty:
        return

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
    true_mean = agg["true_mean"].to_numpy(dtype=float)
    pred_mean = agg["pred_mean"].to_numpy(dtype=float)
    true_std = agg["true_std"].fillna(0.0).to_numpy(dtype=float)
    pred_std = agg["pred_std"].fillna(0.0).to_numpy(dtype=float)

    fig, ax = _subplots(cfg)

    ax.plot(x, true_mean, color="tab:blue", label="true mean")
    ax.fill_between(
        x,
        true_mean - true_std,
        true_mean + true_std,
        color="tab:blue",
        alpha=0.2,
    )
    ax.plot(x, pred_mean, color="tab:orange", label="predicted mean")
    ax.fill_between(
        x,
        pred_mean - pred_std,
        pred_mean + pred_std,
        color="tab:orange",
        alpha=0.2,
    )

    ax.set_xlabel("frame index")
    ax.set_ylabel("speed (mph)")
    ax.legend(fontsize=_get(cfg, "legend_font_size"))

    _save(fig, out, cfg)


def plot_calibration(
        df: pd.DataFrame,
        out: Path,
        cfg: VisualizationConfig | None = None,
) -> None:
    """Plot binned mean predicted speed against true speed."""
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

    fig, ax = _subplots(cfg)

    ax.errorbar(
        centers,
        means,
        yerr=stds,
        fmt="o",
        capsize=3,
        color="tab:orange",
    )

    lo, hi = float(truth.min()), float(truth.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)

    ax.set_xlabel("true (mph)")
    ax.set_ylabel("predicted (mph)")

    _save(fig, out, cfg)


def plot_error_distribution(
        df: pd.DataFrame,
        out: Path,
        cfg: VisualizationConfig | None = None,
) -> None:
    """Plot the distribution of prediction errors."""
    if df.empty:
        return

    errors = (
            df["final_speed_mph"] - df["true_speed_mph"]
    ).to_numpy(dtype=float)

    fig, ax = _subplots(cfg)
    ax.hist(errors, bins=30)
    # ax.set_xlabel("error (mph)")
    # ax.set_ylabel("count")

    _save(fig, out, cfg)


def plot_error_by_bucket(
        result: EvaluationResult,
        out: Path,
        cfg: VisualizationConfig | None = None,
) -> None:
    """Plot mean absolute error for each evaluation bucket."""
    names = list(result.buckets.keys())
    maes = [result.buckets[name].get("mae") for name in names]
    values = [0.0 if value is None else float(value) for value in maes]

    fig, ax = _subplots(cfg)
    ax.bar(names, values)
    # ax.set_xlabel("bucket")
    ax.set_ylabel("MAE (mph)")

    fig.tight_layout()
    _save(fig, out, cfg)


def plot_stationary_confusion(
        result: EvaluationResult,
        out: Path,
        cfg: VisualizationConfig | None = None,
) -> None:
    """Plot the moving/stationary confusion matrix."""
    s = result.stationary
    cm = np.array(
        [[s["tn"], s["fp"]], [s["fn"], s["tp"]]],
        dtype=float,
    )

    fig, ax = _subplots(cfg)
    ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["moving", "stationary"])
    ax.set_yticklabels(["moving", "stationary"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
            )

    _save(fig, out, cfg)


def plot_latency(
        samples_ms: np.ndarray,
        summary: dict | None,
        out: Path,
        cfg: VisualizationConfig | None = None,
) -> None:
    """Plot the distribution of inference latency samples."""
    samples = np.asarray(samples_ms, dtype=float)

    fig, ax = _subplots(cfg)

    if samples.size:
        ax.hist(samples, bins=30)

    ax.set_xlabel("latency (ms)")
    ax.set_ylabel("count")

    _save(fig, out, cfg)


def generate_all_figures(
        predictions: dict[str, pd.DataFrame],
        results: dict[str, EvaluationResult],
        latency_samples: np.ndarray,
        paths: RunPaths,
        cfg: VisualizationConfig | None = None,
) -> list[Path]:
    """Generate all figures under paths.figures_dir and return their paths."""
    figures_dir = paths.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    names: list[str] = []

    for split_name, df in predictions.items():
        plot_predicted_vs_actual(
            df,
            figures_dir / f"predicted_vs_actual_{split_name}.pdf",
            cfg,
        )
        plot_error_distribution(
            df,
            figures_dir / f"error_distribution_{split_name}.pdf",
            cfg,
        )

        names.extend(
            [
                f"predicted_vs_actual_{split_name}.pdf",
                f"error_distribution_{split_name}.pdf",
            ]
        )

    if "test" in predictions:
        test_df = predictions["test"]

        timeseries_paths = plot_speed_timeseries(
            test_df,
            figures_dir / "speed_timeseries.pdf",
            cfg,
        )
        names.extend(path.name for path in timeseries_paths)

        plot_speed_profile(
            test_df,
            figures_dir / "speed_profile.pdf",
            cfg,
        )
        plot_calibration(
            test_df,
            figures_dir / "calibration.pdf",
            cfg,
        )

        names.extend(
            [
                "speed_timeseries.pdf",
                "speed_profile.pdf",
                "calibration.pdf",
            ]
        )

    if "test" in results:
        plot_error_by_bucket(
            results["test"],
            figures_dir / "error_by_bucket.pdf",
            cfg,
        )
        plot_stationary_confusion(
            results["test"],
            figures_dir / "stationary_confusion_matrix.pdf",
            cfg,
        )

        names.extend(
            [
                "error_by_bucket.pdf",
                "stationary_confusion_matrix.pdf",
            ]
        )

    plot_latency(
        latency_samples,
        None,
        figures_dir / "latency.pdf",
        cfg,
    )
    names.append("latency.pdf")

    return [figures_dir / name for name in names]
