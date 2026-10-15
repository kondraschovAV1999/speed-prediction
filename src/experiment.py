"""Experiment orchestration: the train flow (WP10)."""

from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.core.config import Config, validate_config
from src.core.logging_utils import setup_logging
from src.core.run_manager import (
    collect_environment,
    create_run,
    set_seed,
    update_metadata,
    write_metadata,
)
from src.core.schemas import RunPaths
from src.data.dataset import build_dataloaders, build_datasets
from src.data.preprocessing import build_transform
from src.evaluation.benchmark import benchmark_end_to_end, benchmark_model
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.reporting import write_metrics_json, write_predictions_csv
from src.evaluation.visualization import generate_all_figures
from src.inference.pipeline import StreamingSpeedEstimator, collect_raw_predictions
from src.inference.postprocessing import SpeedPostProcessor, postprocess_dataframe
from src.models.architectures import build_model, count_parameters
from src.models.training import (
    SpeedLightningModule,
    build_trainer,
    load_model_from_checkpoint,
    save_model_export,
)
from src.tracking.tracker import create_tracker

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
        trainer = build_trainer(cfg, paths, tracker=tracker)
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
        save_model_export(loaded_model, paths.run_dir / "model.pt", cfg.model)
        results: dict = {}
        predictions: dict[str, pd.DataFrame] = {}
        eval_splits = list(cfg.evaluation.splits_to_evaluate)
        for split_name in ("train", *eval_splits):
            logger.info("event=evaluate_split_start split=%s", split_name)
            raw_df = collect_raw_predictions(
                loaded_model, loaders[split_name], cfg.evaluation.device, split_name
            )
            final_df = postprocess_dataframe(raw_df, cfg.postprocessing)
            predictions[split_name] = final_df
            if split_name in eval_splits:
                results[split_name] = evaluate_predictions(final_df, cfg.evaluation, split_name)
            logger.info("event=evaluate_split_complete split=%s n=%d", split_name, len(final_df))

        predictions_df = pd.concat(
            [predictions[s] for s in eval_splits], ignore_index=True
        ).sort_values(["split", "sequence_id", "frame_index"])

        # 5. Benchmark + artifacts.
        latency: dict | None = None
        latency_samples: np.ndarray = np.array([], dtype=float)
        if cfg.benchmark.enabled:
            logger.info("event=benchmark_start iters=%d", cfg.benchmark.num_iters)
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
            logger.info("event=benchmark_complete")

        write_metrics_json(
            paths,
            results,
            latency,
            {"best_checkpoint": best_checkpoint, "best_val_metric": best_val_metric},
        )
        write_predictions_csv(paths, predictions_df)

        generate_all_figures(predictions, results, latency_samples, paths)

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


def evaluate_run(paths: RunPaths, cfg: Config) -> dict:
    """Re-run evaluation from the best checkpoint without re-training (WP11).

    Returns ``{split_name: EvaluationResult}``.
    """
    datasets = build_datasets(cfg, cfg.experiment.seed, run_split_path=paths.split_path)
    loaders = build_dataloaders(datasets, cfg)
    best = paths.checkpoints_dir / "best.ckpt"
    model = load_model_from_checkpoint(best, cfg.model)
    results: dict = {}
    for split_name in cfg.evaluation.splits_to_evaluate:
        raw = collect_raw_predictions(model, loaders[split_name], cfg.evaluation.device, split_name)
        final = postprocess_dataframe(raw, cfg.postprocessing)
        results[split_name] = evaluate_predictions(final, cfg.evaluation, split_name)
    return results
