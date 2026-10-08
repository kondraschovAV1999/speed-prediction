"""Evaluation metrics (WP6): regression, bucket, stationary and latency.

Pure NumPy/pandas/sklearn; no torch.  Metrics that are undefined (empty set,
constant truth, R² with n<2) are reported as ``None`` so they serialize to
JSON ``null`` rather than NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.core.config import EvaluationCfg
from src.core.schemas import EmptyEvaluationError, EvaluationResult
from src.evaluation.buckets import assign_buckets, buckets_from_cfg


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE / RMSE / R² / median AE. R² is ``None`` when n < 2 or truth is constant."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = y_true.shape[0]
    if n == 0:
        return {"mae": None, "rmse": None, "r2": None, "median_ae": None}
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    median_ae = float(np.median(np.abs(errors)))
    r2 = None
    if n >= 2 and np.var(y_true) > 0.0:
        r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2, "median_ae": median_ae}


def stationary_metrics(
    true_speed: np.ndarray, pred_stationary: np.ndarray, label_threshold: float
) -> dict:
    """Confusion metrics with positive class = stationary (truth ``<= threshold``)."""
    true_speed = np.asarray(true_speed, dtype=float)
    pred_stationary = np.asarray(pred_stationary, dtype=bool)
    true_stationary = true_speed <= label_threshold

    tp = int(np.sum(true_stationary & pred_stationary))
    fp = int(np.sum(~true_stationary & pred_stationary))
    fn = int(np.sum(true_stationary & ~pred_stationary))
    tn = int(np.sum(~true_stationary & ~pred_stationary))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_true_stationary": int(np.sum(true_stationary)),
    }


def latency_summary(samples_ms: np.ndarray) -> dict:
    """Summary statistics over latency samples (mean, median, p95, fps, n)."""
    samples = np.asarray(samples_ms, dtype=float)
    n = samples.shape[0]
    if n == 0:
        return {"mean_ms": None, "median_ms": None, "p95_ms": None, "fps": None, "n": 0}
    mean_ms = float(np.mean(samples))
    median_ms = float(np.median(samples))
    p95_ms = float(np.percentile(samples, 95))
    fps = (1000.0 / mean_ms) if mean_ms > 0 else float("inf")
    return {"mean_ms": mean_ms, "median_ms": median_ms, "p95_ms": p95_ms, "fps": fps, "n": n}


def _check_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Verify the table is ordered by ``(sequence_id, frame_index)``."""
    if len(df) < 2:
        return df
    seq = df["sequence_id"].to_numpy()
    idx = df["frame_index"].to_numpy()
    seq_decreased = seq[1:] < seq[:-1]
    same_seq = seq[1:] == seq[:-1]
    idx_not_increasing = same_seq & (idx[1:] <= idx[:-1])
    if np.any(seq_decreased) or np.any(idx_not_increasing):
        raise ValueError("predictions table must be sorted by (sequence_id, frame_index)")
    return df


def evaluate_predictions(df: pd.DataFrame, cfg: EvaluationCfg, split: str) -> EvaluationResult:
    """Evaluate a single split's predictions table into an :class:`EvaluationResult`."""
    if df.empty:
        raise EmptyEvaluationError(f"No predictions available for split {split!r}")

    required = {"true_speed_mph", "final_speed_mph", "raw_speed_mph", "stationary_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"predictions table missing required columns: {sorted(missing)}")

    df = _check_sorted(df)
    y_true = df["true_speed_mph"].to_numpy(dtype=float)
    y_final = df["final_speed_mph"].to_numpy(dtype=float)
    y_raw = df["raw_speed_mph"].to_numpy(dtype=float)
    pred_stationary = df["stationary_pred"].to_numpy(dtype=bool)

    overall = regression_metrics(y_true, y_final)
    overall_raw = regression_metrics(y_true, y_raw)

    buckets = buckets_from_cfg(cfg)
    assigned = assign_buckets(y_true, buckets)
    bucket_results: dict[str, dict] = {}
    for b in buckets:
        mask = assigned == b.name
        n = int(mask.sum())
        if n == 0:
            bucket_results[b.name] = {
                "n": 0,
                "mae": None,
                "rmse": None,
                "r2": None,
                "median_ae": None,
            }
        else:
            metrics = regression_metrics(y_true[mask], y_final[mask])
            bucket_results[b.name] = {"n": n, **metrics}

    stationary = stationary_metrics(y_true, pred_stationary, cfg.stationary_label_threshold_mph)

    return EvaluationResult(
        split=split,
        n_samples=int(len(df)),
        overall=overall,
        overall_raw=overall_raw,
        buckets=bucket_results,
        stationary=stationary,
        latency=None,
    )
