"""Metrics/predictions writing and reading (WP7)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.schemas import EvaluationResult, RunPaths, _json_safe

# Canonical predictions-table column order (docs/PLAN.md section 5.4).
_CANONICAL_COLUMNS = [
    "split",
    "sequence_id",
    "frame_index",
    "timestamp",
    "true_speed_mph",
    "raw_speed_mph",
    "final_speed_mph",
    "stationary_pred",
    "inference_ms",
]


def write_metrics_json(
    paths: RunPaths,
    results: dict[str, EvaluationResult],
    latency: dict | None,
    extra: dict | None,
) -> None:
    """Write the canonical ``metrics.json`` (schema_version 1; NaN -> null)."""
    payload = _json_safe(
        {
            "schema_version": 1,
            "run_name": paths.run_name,
            "splits": {split: result.to_dict() for split, result in results.items()},
            "latency": latency or {},
            "extra": extra or {},
        }
    )
    paths.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with paths.metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)


def write_predictions_csv(paths: RunPaths, df: pd.DataFrame) -> None:
    """Write the predictions table in the canonical §5.4 column order.

    Normalizes ``-0.0`` in ``final_speed_mph`` to ``0.0`` so the CSV never
    contains ``-0.0``.
    """
    present = [c for c in _CANONICAL_COLUMNS if c in df.columns]
    out = df[present].copy()
    if "final_speed_mph" in out.columns:
        values = out["final_speed_mph"].to_numpy(dtype=float)
        values[np.signbit(values) & (values == 0.0)] = 0.0
        out["final_speed_mph"] = values
    paths.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(paths.predictions_path, index=False)


def read_predictions_csv(path: Path | str) -> pd.DataFrame:
    """Read a predictions CSV back into a DataFrame."""
    return pd.read_csv(path)
