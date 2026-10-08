"""Re-run evaluation from a saved run's checkpoint (WP11).

Usage: ``python scripts/evaluate.py <run_name> [--output-root output]``
Writes results to ``reports/eval_<timestamp>/`` (never overwrites metrics.json).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from src.experiment import evaluate_run
from src.run_manager import load_run
from src.schemas import _json_safe

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run evaluation from a saved checkpoint")
    parser.add_argument("run_name", help="Name of the run to evaluate")
    parser.add_argument("--output-root", default="output", help="Output root directory")
    args = parser.parse_args()

    paths, cfg = load_run(args.run_name, Path(args.output_root))
    results = evaluate_run(paths, cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = paths.reports_dir / f"eval_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=False)

    payload = _json_safe(
        {
            "schema_version": 1,
            "run_name": paths.run_name,
            "splits": {split: result.to_dict() for split, result in results.items()},
            "latency": {},
            "extra": {"source": "evaluate.py"},
        }
    )
    with (eval_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    logger.info("event=eval_written path=%s", eval_dir / "metrics.json")


if __name__ == "__main__":
    main()
