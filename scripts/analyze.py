#!/usr/bin/env python3
"""
Experiment Results Analyzer and Visualizer.

This script scans a parent directory containing experiment run folders
(formatted with a leading timestamp `YYYYMMDD_HHMMSS_...`), filters them by a
user-defined date/time interval, extracts model architectures (`backbone` and `temporal`)
from `config.yaml` and evaluation metrics (`val` and `test` MAE/RMSE) from `metrics.json`,
and generates high-quality PDF bar charts comparing:
1. Model backbones grouped by each temporal architecture.
2. The best performing temporal combination for each backbone.
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
# Use non-interactive PDF/Agg backend for reliable headless execution
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Regular expression to extract timestamp from folder names like '20261019_232409_baseline_36df'
FOLDER_TIMESTAMP_REGEX = re.compile(r"^(\d{8}_\d{6})(?:_|$)")


def parse_datetime_arg(dt_str: Optional[str], is_end: bool = False) -> Optional[datetime]:
    """
    Parses a datetime string from CLI arguments into a datetime object.
    Supports flexible formats:
      - YYYYMMDD_HHMMSS (e.g., 20261019_232409)
      - YYYY-MM-DD HH:MM:SS or YYYY-MM-DDTHH:MM:SS
      - YYYYMMDD (e.g., 20261019) -> defaults to 00:00:00 (or 23:59:59 if is_end=True)
      - YYYY-MM-DD (e.g., 2026-10-19) -> defaults to 00:00:00 (or 23:59:59 if is_end=True)
    """
    if not dt_str:
        return None

    dt_str = dt_str.strip()
    formats_with_time = [
        "%Y%m%d_%H%M%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y%m%d_%H%M",
        "%Y-%m-%d %H:%M",
    ]
    for fmt in formats_with_time:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue

    formats_date_only = [
        "%Y%m%d",
        "%Y-%m-%d",
    ]
    for fmt in formats_date_only:
        try:
            dt = datetime.strptime(dt_str, fmt)
            if is_end:
                return dt.replace(hour=23, minute=59, second=59)
            return dt
        except ValueError:
            continue

    raise ValueError(
        f"Unrecognized date/time format: '{dt_str}'. "
        "Expected formats like 'YYYYMMDD_HHMMSS', 'YYYY-MM-DD HH:MM:SS', or 'YYYYMMDD'."
    )


def extract_folder_datetime(folder_name: str) -> Optional[datetime]:
    """
    Extracts the datetime object from an experiment folder name prefix.
    Returns None if the folder name does not start with YYYYMMDD_HHMMSS.
    """
    match = FOLDER_TIMESTAMP_REGEX.match(folder_name)
    if not match:
        return None
    timestamp_str = match.group(1)
    try:
        return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def extract_config_fields(config_data: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extracts backbone and temporal model names from config.yaml structure.
    Handles both nested `model.encoder.backbone` and direct `model.backbone`,
    as well as dict or string representations of `model.temporal`.
    """
    model_cfg = config_data.get("model", {})
    if not isinstance(model_cfg, dict):
        return "unknown_backbone", "unknown_temporal"

    # 1. Extract backbone (check model.backbone first, then model.encoder.backbone)
    backbone = model_cfg.get("backbone")
    if not backbone:
        encoder_cfg = model_cfg.get("encoder", {})
        if isinstance(encoder_cfg, dict):
            backbone = encoder_cfg.get("backbone") or encoder_cfg.get("name")
        elif isinstance(encoder_cfg, str):
            backbone = encoder_cfg

    if not backbone:
        backbone = model_cfg.get("name", "unknown_backbone")

    # 2. Extract temporal (can be a dict with 'name' or a direct string)
    temporal_cfg = model_cfg.get("temporal")
    if isinstance(temporal_cfg, dict):
        temporal = temporal_cfg.get("name", "unknown_temporal")
    elif isinstance(temporal_cfg, str):
        temporal = temporal_cfg
    else:
        temporal = "none"

    return str(backbone), str(temporal)


def extract_metrics_fields(metrics_data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Extracts val and test overall MAE and RMSE from metrics.json structure.
    Also supports fallback if 'split' is used instead of 'splits'.
    """
    splits = metrics_data.get("splits") or metrics_data.get("split") or {}

    def get_metric(split_name: str, metric_name: str) -> Optional[float]:
        split_obj = splits.get(split_name, {})
        if not isinstance(split_obj, dict):
            return None
        overall_obj = split_obj.get("overall", {})
        if not isinstance(overall_obj, dict):
            return None
        val = overall_obj.get(metric_name)
        return float(val) if val is not None else None

    return {
        "val_mae": get_metric("val", "mae"),
        "val_rmse": get_metric("val", "rmse"),
        "test_mae": get_metric("test", "mae"),
        "test_rmse": get_metric("test", "rmse"),
    }


def load_experiments(
    base_dir: Path,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Iterates through all subdirectories in `base_dir`, filters by timestamp,
    and loads config.yaml and metrics.json into a structured Pandas DataFrame.
    """
    if not base_dir.exists() or not base_dir.is_dir():
        raise FileNotFoundError(f"Base directory not found or not a directory: {base_dir}")

    records: List[Dict[str, Any]] = []
    subdirs = sorted([p for p in base_dir.iterdir() if p.is_dir()])

    logger.info(f"Scanning {len(subdirs)} folders in '{base_dir}'...")

    for folder in subdirs:
        folder_dt = extract_folder_datetime(folder.name)
        if folder_dt is None:
            logger.debug(f"Skipping '{folder.name}': does not match YYYYMMDD_HHMMSS prefix.")
            continue

        # Apply date/time interval filtering
        if start_dt and folder_dt < start_dt:
            continue
        if end_dt and folder_dt > end_dt:
            continue

        config_path = folder / "config.yaml"
        metrics_path = folder / "metrics.json"

        if not config_path.is_file() or not metrics_path.is_file():
            logger.warning(
                f"Skipping '{folder.name}': missing config.yaml or metrics.json."
            )
            continue

        try:
            with open(config_path, "r", encoding="utf-8") as f_cfg:
                config_data = yaml.safe_load(f_cfg) or {}
            with open(metrics_path, "r", encoding="utf-8") as f_met:
                metrics_data = json.load(f_met) or {}
        except Exception as exc:
            logger.error(f"Error reading files in '{folder.name}': {exc}")
            continue

        backbone, temporal = extract_config_fields(config_data)
        metrics = extract_metrics_fields(metrics_data)

        # Verify at least one metric is present
        if all(v is None for v in metrics.values()):
            logger.warning(f"Skipping '{folder.name}': no valid val/test MAE or RMSE found.")
            continue

        exp_name = config_data.get("experiment", {}).get("name", folder.name)
        run_name = metrics_data.get("run_name", folder.name)

        records.append({
            "folder": folder.name,
            "run_name": run_name,
            "experiment_name": exp_name,
            "timestamp": folder_dt,
            "backbone": backbone,
            "temporal": temporal,
            "val_mae": metrics["val_mae"],
            "val_rmse": metrics["val_rmse"],
            "test_mae": metrics["test_mae"],
            "test_rmse": metrics["test_rmse"],
        })

    df = pd.DataFrame(records)
    return df


def annotate_bars(ax: plt.Axes, rects: Any, font_size: float) -> None:
    """
    Attaches a text label above each bar displaying its numeric height.
    """
    for rect in rects:
        height = rect.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2.0, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=max(font_size * 0.78, 6.0),
            rotation=0,
        )


# Predefined metric views so every comparison produces:
# 1) All 4 metrics combined
# 2) Separate Validation vs Testing figures (both MAE-first and RMSE-first)
# 3) Separate MAE vs RMSE figures (both combinedVal/Test and per-split single metric)
METRIC_VIEWS: List[Tuple[str, str, List[Tuple[str, str, str]]]] = [
    (
        "all_metrics",
        "Val & Test (MAE & RMSE)",
        [
            ("val_mae", "Val MAE", "#2b5c8f"),
            ("test_mae", "Test MAE", "#d95f02"),
            ("val_rmse", "Val RMSE", "#6baed6"),
            ("test_rmse", "Test RMSE", "#fe9929"),
        ],
    ),
    (
        "val_only",
        "Validation Split (MAE & RMSE)",
        [
            ("val_rmse", "Val RMSE", "#6baed6"),
            ("val_mae", "Val MAE", "#2b5c8f"),
        ],
    ),
    (
        "test_only",
        "Test Split (MAE & RMSE)",
        [
            ("test_rmse", "Test RMSE", "#fe9929"),
            ("test_mae", "Test MAE", "#d95f02"),
        ],
    ),
    (
        "rmse_mae_val",
        "Validation Split (RMSE & MAE)",
        [
            ("val_mae", "Val MAE", "#08306b"),
            ("val_rmse", "Val RMSE", "#9ecae1"),
        ],
    ),
    (
        "rmse_mase_val",
        "Validation Split (RMSE & MAE)",
        [
            ("val_mae", "Val MAE", "#08306b"),
            ("val_rmse", "Val RMSE", "#9ecae1")
        ],
    ),
    (
        "rmse_mae_test",
        "Test Split (RMSE & MAE)",
        [
            ("test_mae", "Test MAE", "#7f0000"),
            ("test_rmse", "Test RMSE", "#fdae6b"),
        ],
    ),
    (
        "mae_only",
        "MAE Comparison (Val & Test)",
        [
            ("val_mae", "Val MAE", "#2b5c8f"),
            ("test_mae", "Test MAE", "#d95f02"),
        ],
    ),
    (
        "rmse_only",
        "RMSE Comparison (Val & Test)",
        [
            ("val_rmse", "Val RMSE", "#6baed6"),
            ("test_rmse", "Test RMSE", "#fe9929"),
        ],
    ),
    (
        "val_mae",
        "Validation MAE",
        [
            ("val_mae", "Val MAE", "#2b5c8f"),
        ],
    ),
    (
        "val_rmse",
        "Validation RMSE",
        [
            ("val_rmse", "Val RMSE", "#6baed6"),
        ],
    ),
    (
        "test_mae",
        "Test MAE",
        [
            ("test_mae", "Test MAE", "#d95f02"),
        ],
    ),
    (
        "test_rmse",
        "Test RMSE",
        [
            ("test_rmse", "Test RMSE", "#fe9929"),
        ],
    ),
]


def plot_grouped_bars(
    df_group: pd.DataFrame,
    x_col: str,
    title: str,
    xlabel: str,
    output_path: Path,
    figsize: Tuple[float, float],
    font_size: float,
    xtick_rotation: float,
    metric_specs: List[Tuple[str, str, str]],
    ylabel: str = "Error (mph)",
    label_map: Optional[Dict[str, str]] = None,
    headroom_factor: float = 1.38,
) -> None:
    """
    Draws a grouped (or single) bar chart for the specified `metric_specs`
    across categories in `x_col`, and saves the figure as a PDF without a title
    or x-axis label, with configurable x-tick rotation and a non-overlapping
    two-line internal legend.
    """
    # Configure font sizes dynamically
    plt.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size * 1.05,
        "xtick.labelsize": font_size * 0.95,
        "ytick.labelsize": font_size * 0.95,
        "legend.fontsize": font_size * 0.7,
    })

    categories = df_group[x_col].tolist()
    n_cats = len(categories)
    n_metrics = len(metric_specs)

    x = np.arange(n_cats)
    total_group_width = 0.55 if n_metrics == 1 else (0.68 if n_metrics == 2 else 0.78)
    bar_width = total_group_width / n_metrics

    fig, ax = plt.subplots(figsize=figsize)

    for idx, (col_name, label, color) in enumerate(metric_specs):
        values = df_group[col_name].astype(float).values
        offset = (idx - (n_metrics - 1) / 2.0) * bar_width
        rects = ax.bar(
            x + offset,
            values,
            width=bar_width * 0.92,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.92,
        )
        annotate_bars(ax, rects, font_size)

    # Set x-axis tick labels
    if label_map:
        xtick_labels = [label_map.get(cat, str(cat)) for cat in categories]
    else:
        xtick_labels = [str(cat) for cat in categories]

    ax.set_xticks(x)
    # Align rotated labels cleanly toward the tick mark when angled
    ha_align = "right" if xtick_rotation > 0 else ("left" if xtick_rotation < 0 else "center")
    ax.set_xticklabels(
        xtick_labels,
        rotation=xtick_rotation,
        ha=ha_align,
        rotation_mode="anchor" if xtick_rotation != 0 else None,
    )
    ax.set_ylabel(ylabel, labelpad=8, fontweight="bold")

    # Add subtle horizontal gridlines for readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Expand y-limit generously so bar annotations never collide with the internal two-line legend,
    # while keeping y-axis tick numbers strictly within the natural data range.
    y_vals = df_group[[m[0] for m in metric_specs]].astype(float).values
    if y_vals.size > 0 and not np.all(np.isnan(y_vals)):
        y_max = np.nanmax(y_vals)
        if not np.isnan(y_max) and y_max > 0:
            natural_top = y_max * 1.12
            ax.set_ylim(0, natural_top)
            # Lock in the tick locations that make sense for the actual data range
            valid_ticks = [t for t in ax.get_yticks() if 0 <= t <= natural_top]
            actual_headroom = headroom_factor if n_metrics > 1 else 1.15
            ax.set_ylim(0, y_max * actual_headroom)
            ax.set_yticks(valid_ticks)

    if n_metrics > 1:
        # Force a 2-row legend inside the plot:
        # - For 4 metrics -> ncol=2 produces 2 rows of 2 items
        # - For 2 metrics -> ncol=1 produces 2 rows of 1 item
        legend_ncol = max(1, (n_metrics + 1) // 2)
        ax.legend(
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            edgecolor="0.8",
            ncol=legend_ncol,
            loc="best",
        )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved PDF figure -> {output_path}")


def render_all_metric_views(
    df_group: pd.DataFrame,
    x_col: str,
    base_title: str,
    xlabel: str,
    subdir: Path,
    file_prefix: str,
    figsize: Tuple[float, float],
    font_size: float,
    xtick_rotation: float,
    label_map: Optional[Dict[str, str]] = None,
    headroom_factor: float = 1.38,
) -> None:
    """
    Generates all PDF figures for a given comparison group:
      - all_metrics (Val & Test MAE/RMSE)
      - val_only (Val MAE & RMSE)
      - test_only (Test MAE & RMSE)
      - rmse_mae_val / rmse_mase_val (Val RMSE & MAE)
      - rmse_mae_test (Test RMSE & MAE)
      - mae_only (Val & Test MAE)
      - rmse_only (Val & Test RMSE)
      - val_mae, val_rmse, test_mae, test_rmse (individual split+metric figures)
    """
    for view_key, view_subtitle, metric_specs in METRIC_VIEWS:
        if "mae" in view_key and "rmse" not in view_key:
            ylabel = "MAE (mph)"
        elif "rmse" in view_key and "mae" not in view_key and "mase" not in view_key:
            ylabel = "RMSE (mph)"
        else:
            ylabel = "Error (mph)"

        pdf_path = subdir / view_key / f"{file_prefix}_{view_key}.pdf"
        full_title = f"{base_title}\n[{view_subtitle}]"

        plot_grouped_bars(
            df_group=df_group,
            x_col=x_col,
            title=full_title,
            xlabel=xlabel,
            output_path=pdf_path,
            figsize=figsize,
            font_size=font_size,
            xtick_rotation=xtick_rotation,
            metric_specs=metric_specs,
            ylabel=ylabel,
            label_map=label_map,
            headroom_factor=headroom_factor,
        )


def generate_all_comparisons(
    df: pd.DataFrame,
    output_dir: Path,
    selection_metric: str,
    aggregate_duplicates: str,
    figsize: Tuple[float, float],
    font_size: float,
    xtick_rotation: float,
    headroom_factor: float,
) -> None:
    """
    Generates PDF bar charts across all splits (val/test) and metrics (MAE/RMSE) for:
    1. Each `temporal` architecture comparing `backbone` models (`by_temporal/`).
    2. Each `backbone` architecture comparing `temporal` models (`by_backbone/`).
    3. Best performing `temporal` combination for each `backbone` (`best_combinations/best_temporal_per_backbone/`).
    4. Best performing `backbone` combination for each `temporal` (`best_combinations/best_backbone_per_temporal/`).
    """
    metric_cols = ["val_mae", "val_rmse", "test_mae", "test_rmse"]

    # Step 1: Deduplicate or average runs with the exact same (temporal, backbone) pair
    if aggregate_duplicates == "best":
        df_sorted = df.sort_values(by=selection_metric, ascending=True, na_position="last")
        df_dedup = df_sorted.drop_duplicates(subset=["temporal", "backbone"], keep="first").copy()
    else:
        df_dedup = (
            df.groupby(["temporal", "backbone"], as_index=False)[metric_cols]
            .mean()
        )

    # -------------------------------------------------------------------------
    # A. Group by Temporal Architecture -> compare Backbones
    # -------------------------------------------------------------------------
    for temporal_name, group_df in df_dedup.groupby("temporal"):
        group_df = group_df.sort_values(by=selection_metric, ascending=True)
        safe_temporal = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(temporal_name))

        render_all_metric_views(
            df_group=group_df,
            x_col="backbone",
            base_title=f"Backbone Comparison — Temporal: {temporal_name}",
            xlabel="Backbone Model",
            subdir=output_dir / "by_temporal",
            file_prefix=f"temporal_{safe_temporal}",
            figsize=figsize,
            font_size=font_size,
            xtick_rotation=xtick_rotation,
            headroom_factor=headroom_factor,
        )

    # -------------------------------------------------------------------------
    # B. Group by Backbone Model -> compare Temporal Architectures
    # -------------------------------------------------------------------------
    for backbone_name, group_df in df_dedup.groupby("backbone"):
        group_df = group_df.sort_values(by=selection_metric, ascending=True)
        safe_backbone = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(backbone_name))

        render_all_metric_views(
            df_group=group_df,
            x_col="temporal",
            base_title=f"Temporal Comparison — Backbone: {backbone_name}",
            xlabel="Temporal Architecture",
            subdir=output_dir / "by_backbone",
            file_prefix=f"backbone_{safe_backbone}",
            figsize=figsize,
            font_size=font_size,
            xtick_rotation=xtick_rotation,
            headroom_factor=headroom_factor,
        )

    # -------------------------------------------------------------------------
    # C. Best Temporal Combination for each Backbone
    # -------------------------------------------------------------------------
    df_best_sorted = df_dedup.sort_values(by=selection_metric, ascending=True, na_position="last")
    best_per_backbone = df_best_sorted.drop_duplicates(subset=["backbone"], keep="first").copy()
    best_per_backbone = best_per_backbone.sort_values(by=selection_metric, ascending=True)

    backbone_label_map = {
        row["backbone"]: f"{row['backbone']}\n({row['temporal']})"
        for _, row in best_per_backbone.iterrows()
    }

    render_all_metric_views(
        df_group=best_per_backbone,
        x_col="backbone",
        base_title=f"Best Temporal Combination per Backbone (Selected by {selection_metric})",
        xlabel="Backbone (Best Temporal Architecture)",
        subdir=output_dir / "best_combinations" / "per_backbone",
        file_prefix="best_per_backbone",
        figsize=figsize,
        font_size=font_size,
        xtick_rotation=xtick_rotation,
        label_map=backbone_label_map,
        headroom_factor=headroom_factor,
    )

    # -------------------------------------------------------------------------
    # D. Best Backbone Combination for each Temporal Architecture
    # -------------------------------------------------------------------------
    best_per_temporal = df_best_sorted.drop_duplicates(subset=["temporal"], keep="first").copy()
    best_per_temporal = best_per_temporal.sort_values(by=selection_metric, ascending=True)

    temporal_label_map = {
        row["temporal"]: f"{row['temporal']}\n({row['backbone']})"
        for _, row in best_per_temporal.iterrows()
    }

    render_all_metric_views(
        df_group=best_per_temporal,
        x_col="temporal",
        base_title=f"Best Backbone Combination per Temporal Group (Selected by {selection_metric})",
        xlabel="Temporal Architecture (Best Backbone)",
        subdir=output_dir / "best_combinations" / "per_temporal",
        file_prefix="best_per_temporal",
        figsize=figsize,
        font_size=font_size,
        xtick_rotation=xtick_rotation,
        label_map=temporal_label_map,
        headroom_factor=headroom_factor,
    )

    # -------------------------------------------------------------------------
    # E. Save CSV summary tables
    # -------------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = output_dir / "experiments_summary.csv"
    df.sort_values(by=["temporal", "backbone", "timestamp"]).to_csv(summary_csv_path, index=False)
    logger.info(f"Saved full extracted table -> {summary_csv_path}")

    best_backbone_csv = output_dir / "best_per_backbone_summary.csv"
    best_per_backbone.to_csv(best_backbone_csv, index=False)
    logger.info(f"Saved best-per-backbone table -> {best_backbone_csv}")

    best_temporal_csv = output_dir / "best_per_temporal_summary.csv"
    best_per_temporal.to_csv(best_temporal_csv, index=False)
    logger.info(f"Saved best-per-temporal table -> {best_temporal_csv}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse experiment folders (config.yaml & metrics.json), filter by date/time "
            "interval, and plot PDF bar comparisons per temporal group and best backbone combinations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--base-dir",
        type=Path,
        required=True,
        help="Path to the parent directory containing experiment folders (e.g., output/runs).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory where output PDF figures and summary CSVs will be saved.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help=(
            "Inclusive start date/time filter. Supported formats: "
            "'YYYYMMDD_HHMMSS' (e.g. 20261019_000000), 'YYYY-MM-DD HH:MM:SS', or 'YYYY-MM-DD'."
        ),
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help=(
            "Inclusive end date/time filter. Supported formats: "
            "'YYYYMMDD_HHMMSS' (e.g. 20261020_235959), 'YYYY-MM-DD HH:MM:SS', or 'YYYY-MM-DD'."
        ),
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        choices=["val_mae", "val_rmse", "test_mae", "test_rmse", "rmse_mae_val", "rmse_mase_val", "rmse_mae_test"],
        default="val_mae",
        help="Metric used (lowest value wins) to pick the best temporal combination for each backbone.",
    )
    parser.add_argument(
        "--duplicate-strategy",
        type=str,
        choices=["best", "mean"],
        default="best",
        help=(
            "How to handle multiple runs with the exact same (backbone, temporal) pair: "
            "'best' picks the run with the lowest selection-metric; 'mean' averages the metrics."
        ),
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[10.0, 6.0],
        metavar=("WIDTH", "HEIGHT"),
        help="Width and height of the generated PDF figures in inches (e.g. --figsize 12 6).",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=12.0,
        help="Base font size for axis labels, ticks, and bar annotations.",
    )
    parser.add_argument(
        "--xtick-rotation",
        type=float,
        default=0.0,
        help="Rotation angle in degrees for x-axis tick labels (e.g., 15, 30, 45).",
    )
    parser.add_argument(
        "--headroom-factor",
        type=float,
        default=1.38,
        help="Multiplier for the top y-axis limit to create headroom for the internal legend.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        start_dt = parse_datetime_arg(args.start_date, is_end=False)
        end_dt = parse_datetime_arg(args.end_date, is_end=True)
    except ValueError as err:
        logger.error(str(err))
        return 1

    if start_dt and end_dt and start_dt > end_dt:
        logger.error(f"Start date ({start_dt}) cannot be after end date ({end_dt}).")
        return 1

    try:
        df = load_experiments(
            base_dir=args.base_dir,
            start_dt=start_dt,
            end_dt=end_dt,
        )
    except Exception as err:
        logger.error(f"Failed to load experiments: {err}")
        return 1

    if df.empty:
        logger.warning(
            "No matching experiment folders with valid config.yaml and metrics.json "
            "were found for the given directory and date/time range."
        )
        return 0

    logger.info(
        f"Loaded {len(df)} valid experiment run(s) across "
        f"{df['temporal'].nunique()} temporal group(s) and "
        f"{df['backbone'].nunique()} backbone(s)."
    )

    generate_all_comparisons(
        df=df,
        output_dir=args.output_dir,
        selection_metric=args.selection_metric,
        aggregate_duplicates=args.duplicate_strategy,
        figsize=(float(args.figsize[0]), float(args.figsize[1])),
        font_size=args.font_size,
        xtick_rotation=args.xtick_rotation,
        headroom_factor=args.headroom_factor,
    )

    logger.info("All plots and summaries generated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())