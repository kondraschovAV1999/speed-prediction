"""Experiment orchestration: the train flow (WP10)."""

from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.benchmark import benchmark_end_to_end, benchmark_model
from src.config import Config, validate_config
from src.data import build_dataloaders, build_datasets
from src.evaluation import evaluate_predictions
from src.logging_utils import setup_logging
from src.models import build_model, count_parameters
from src.pipeline import StreamingSpeedEstimator, collect_raw_predictions
from src.postprocessing import SpeedPostProcessor, postprocess_dataframe
from src.preprocessing import build_transform
from src.reporting import write_metrics_json, write_predictions_csv
from src.run_manager import (
    collect_environment,
    create_run,
    set_seed,
    update_metadata,
    write_metadata,
)
from src.schemas import RunPaths
from src.tracking import create_tracker
from src.training import (
    SpeedLightningModule,
    build_trainer,
    load_model_from_checkpoint,
)
from src.visualization import generate_all_figures

logger = logging.getLogger(__name__)


def _config_params(cfg: Config) -> dict:
    return OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)


def run_experiment(cfg: DictConfig) -> RunPaths:
    """Run the full M1 train -> evaluate -> benchmark flow and write all artifacts."""
    cfg = validate_config(cfg)
    started = time.time()
    paths = create_run(cfg)
    setup_logging(paths.log_file, level=cfg.logging.level)
    set_seed(cfg.experiment.seed, deterministic=cfg.experiment.deterministic)

    write_metadata(
        paths,
        {
            "status": "running",
            "seed": cfg.experiment.seed,
            "environment": collect_environment(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    tracker = create_tracker(cfg, paths)
    tracker.start_run(paths.run_name)
    tracker.log_params(_config_params(cfg))

    try:
        # 2. Data + model
        datasets = build_datasets(cfg, cfg.experiment.seed, run_split_path=paths.split_path)
        loaders = build_dataloaders(datasets, cfg)
        model = build_model(cfg.model)
        logger.info("event=model_parameter_count count=%d", count_parameters(model))

        # 3. Train (train/val only; test untouched).
        trainer = build_trainer(cfg, paths)
        lightning = SpeedLightningModule(model, cfg.training)
        trainer.fit(
            lightning,
            train_dataloaders=loaders["train"],
            val_dataloaders=loaders["val"],
        )
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        best_val_metric = float(trainer.callback_metrics.get(cfg.training.monitor, float("nan")))

        # 4. Evaluate each requested split.
        loaded_model = load_model_from_checkpoint(best_checkpoint, cfg.model)
        results: dict = {}
        prediction_frames: list[pd.DataFrame] = []
        for split_name in cfg.evaluation.splits_to_evaluate:
            raw_df = collect_raw_predictions(loaded_model, loaders[split_name], "cpu", split_name)
            final_df = postprocess_dataframe(raw_df, cfg.postprocessing)
            results[split_name] = evaluate_predictions(final_df, cfg.evaluation, split_name)
            prediction_frames.append(final_df)

        predictions_df = pd.concat(prediction_frames, ignore_index=True).sort_values(
            ["split", "sequence_id", "frame_index"]
        )

        # 5. Benchmark + artifacts.
        latency: dict | None = None
        latency_samples: np.ndarray = np.array([], dtype=float)
        if cfg.benchmark.enabled:
            example = next(iter(loaders["val"]))["frames"][:1]
            model_samples, model_summary = benchmark_model(
                loaded_model,
                example,
                cfg.benchmark.device,
                cfg.benchmark.num_warmup,
                cfg.benchmark.num_iters,
            )

            seq = datasets["val"].sequences[0]
            frames = seq.frames.get(range(len(seq.frames)))
            timestamps = seq.timestamps
            transform = build_transform(cfg.preprocessing)
            postprocessor = SpeedPostProcessor(cfg.postprocessing)
            max_gap = cfg.data.max_gap_factor / cfg.data.frame_rate_hz
            estimator = StreamingSpeedEstimator(
                loaded_model,
                transform,
                postprocessor,
                cfg.data.sequence_length,
                cfg.data.frame_step,
                max_gap,
                cfg.benchmark.device,
                sequence_id=seq.sequence_id,
            )
            e2e_samples, e2e_summary = benchmark_end_to_end(
                estimator, frames, timestamps, cfg.benchmark.num_warmup, cfg.benchmark.num_iters
            )
            latency = {"model_only": model_summary, "end_to_end": e2e_summary}
            latency_samples = e2e_samples

            latency_csv = paths.reports_dir / "latency_samples.csv"
            pd.DataFrame({"model_only_ms": model_samples, "end_to_end_ms": e2e_samples}).to_csv(
                latency_csv, index=False
            )

        write_metrics_json(
            paths,
            results,
            latency,
            {"best_checkpoint": best_checkpoint, "best_val_metric": best_val_metric},
        )
        write_predictions_csv(paths, predictions_df)

        test_df = predictions_df[predictions_df["split"] == "test"]
        test_result = results.get("test")
        if test_result is not None:
            generate_all_figures(test_df, test_result, latency_samples, paths)

        # 6. Log metrics/artifacts and mark completed.
        if "test" in results:
            t = results["test"]
            tracker.log_metrics(
                {
                    k: v
                    for k, v in {
                        "test/mae": t.overall.get("mae"),
                        "test/rmse": t.overall.get("rmse"),
                        "test/r2": t.overall.get("r2"),
                        "test/median_ae": t.overall.get("median_ae"),
                        "test/stationary_f1": t.stationary.get("f1"),
                    }.items()
                    if v is not None
                }
            )
        tracker.log_artifact(paths.metrics_path)
        tracker.log_artifact(paths.predictions_path)
        tracker.log_artifact(paths.config_path)

        update_metadata(
            paths,
            status="completed",
            duration_s=time.time() - started,
            best_checkpoint=best_checkpoint,
            best_val_metric=best_val_metric,
        )
        tracker.finish("FINISHED")
        logger.info("event=run_completed run_name=%s", paths.run_name)
        return paths
    except Exception:
        logger.exception("event=run_failed")
        update_metadata(paths, status="failed", error=traceback.format_exc())
        tracker.finish("FAILED")
        raise
