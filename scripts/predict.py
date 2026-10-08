"""Stream predictions over a frame folder with a saved checkpoint (WP11).

Usage: ``python scripts/predict.py --checkpoint <path> --input <dir> [--output-root output]``

The input directory must contain ``*.npy`` files (one ``uint8 [H, W, C]`` frame
each), sorted by filename; timestamps are ``i / frame_rate_hz``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import compose_config, validate_config
from src.pipeline import StreamingSpeedEstimator
from src.postprocessing import SpeedPostProcessor
from src.preprocessing import build_transform
from src.reporting import write_predictions_csv
from src.run_manager import create_run, update_metadata
from src.training import load_model_from_checkpoint

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream speed predictions over a frame folder")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint")
    parser.add_argument("--input", required=True, help="Directory of .npy frame files")
    parser.add_argument("--output-root", default="output", help="Output root directory")
    args = parser.parse_args()

    cfg = validate_config(compose_config())
    cfg.model.checkpoint = args.checkpoint
    cfg.paths.output_root = args.output_root

    model = load_model_from_checkpoint(args.checkpoint, cfg.model)
    transform = build_transform(cfg.preprocessing)
    postprocessor = SpeedPostProcessor(cfg.postprocessing)
    max_gap = cfg.data.max_gap_factor / cfg.data.frame_rate_hz
    estimator = StreamingSpeedEstimator(
        model,
        transform,
        postprocessor,
        cfg.data.sequence_length,
        cfg.data.frame_step,
        max_gap,
        cfg.benchmark.device,
        sequence_id="predict",
    )

    input_dir = Path(args.input)
    frame_files = sorted(input_dir.glob("*.npy"))
    if not frame_files:
        raise SystemExit(f"no .npy frames found in {input_dir}")

    predictions = []
    for i, path in enumerate(frame_files):
        frame = np.load(path).astype(np.uint8)
        timestamp = i / cfg.data.frame_rate_hz
        pred = estimator.update(frame, timestamp)
        if pred is not None:
            predictions.append(pred)

    paths = create_run(cfg)
    df = pd.DataFrame(
        {
            "split": ["predict"] * len(predictions),
            "sequence_id": [p.sequence_id for p in predictions],
            "frame_index": [p.frame_index for p in predictions],
            "timestamp": [p.timestamp for p in predictions],
            "true_speed_mph": [float("nan")] * len(predictions),
            "raw_speed_mph": [p.raw_speed_mph for p in predictions],
            "final_speed_mph": [p.final_speed_mph for p in predictions],
            "stationary_pred": [p.stationary for p in predictions],
            "inference_ms": [p.inference_ms for p in predictions],
        }
    )
    write_predictions_csv(paths, df)
    update_metadata(paths, status="completed", n_predictions=len(predictions))
    logger.info(
        "event=predictions_written count=%d path=%s", len(predictions), paths.predictions_path
    )


if __name__ == "__main__":
    main()
