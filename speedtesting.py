#!/usr/bin/env python3
"""Evaluate dashcam speed predictions against reference speeds.

Input CSV required column names:
  ground_truth_mph,predicted_mph
Optional column:
  latency_ms

The script evaluates existing predictions; it does not train a model or read video.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STATIONARY_THRESHOLD_MPH = 2.0


def safe_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def classification_metrics(y_true_moving: np.ndarray,
                           y_pred_moving: np.ndarray) -> dict[str, Any]:
    # Positive class = moving.
    tp = int(np.sum(y_true_moving & y_pred_moving))
    tn = int(np.sum(~y_true_moving & ~y_pred_moving))
    fp = int(np.sum(~y_true_moving & y_pred_moving))
    fn = int(np.sum(y_true_moving & ~y_pred_moving))
    total = tp + tn + fp + fn

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)

    return {
        "positive_class": "moving",
        "stationary_threshold_mph": STATIONARY_THRESHOLD_MPH,
        "accuracy": (tp + tn) / total if total else None,
        "precision_moving": precision,
        "recall_moving": recall,
        "f1_moving": f1,
        "confusion_matrix": {
            "true_stationary_pred_stationary_tn": tn,
            "true_stationary_pred_moving_fp": fp,
            "true_moving_pred_stationary_fn": fn,
            "true_moving_pred_moving_tp": tp,
        },
    }


def evaluate(csv_path: Path) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    required = {"ground_truth_mph", "predicted_mph"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {', '.join(sorted(missing))}"
        )
    if df.empty:
        raise ValueError("CSV contains no data rows.")

    for col in ("ground_truth_mph", "predicted_mph"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    valid = (
        df["ground_truth_mph"].notna()
        & df["predicted_mph"].notna()
        & np.isfinite(df["ground_truth_mph"])
        & np.isfinite(df["predicted_mph"])
    )
    dropped_rows = int((~valid).sum())
    df = df.loc[valid].copy()
    if df.empty:
        raise ValueError("No valid numeric ground-truth/prediction pairs found.")

    # Speeds should be non-negative. Reject invalid values rather than silently
    # clipping them, because clipping would distort reported performance.
    if (df[["ground_truth_mph", "predicted_mph"]] < 0).any().any():
        raise ValueError("Speed values must be non-negative (mph).")

    truth = df["ground_truth_mph"].to_numpy(dtype=float)
    pred = df["predicted_mph"].to_numpy(dtype=float)
    errors = pred - truth
    abs_errors = np.abs(errors)
    mse = float(np.mean(errors ** 2))
    truth_var = float(np.sum((truth - np.mean(truth)) ** 2))
    r2 = (
        1.0 - float(np.sum(errors ** 2)) / truth_var
        if truth_var > 0 else None
