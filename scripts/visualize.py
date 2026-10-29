"""Regenerate figures from a run's predictions.csv (WP11).

Usage: ``python scripts/visualize.py <run_name> [--output-root output]``
Writes figures to ``figures/<run_name>/<timestamp>/``.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

from src.core.run_manager import load_run
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.reporting import read_predictions_csv
from src.evaluation.visualization import generate_all_figures

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate figures from predictions.csv")
    parser.add_argument("run_name", help="Name of the run to visualize")
    parser.add_argument("--output-root", default="output", help="Output root directory")
    args = parser.parse_args()

    paths, cfg = load_run(args.run_name, Path(args.output_root))
    df = read_predictions_csv(paths.predictions_path)
    test_df = df[df["split"] == "test"]
    result = evaluate_predictions(test_df, cfg.evaluation, "test")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_dir = paths.figures_dir / timestamp
    sub_paths = replace(paths, figures_dir=sub_dir)
    figures = generate_all_figures(
        predictions={"test": test_df},
        results={"test": result},
        latency_samples=np.array([], dtype=float),
        paths=sub_paths,
        cfg=cfg.visualization,
    )

    logger.info(
        "event=figures_written count=%d path=%s",
        len(figures),
        sub_dir,
    )


if __name__ == "__main__":
    main()
