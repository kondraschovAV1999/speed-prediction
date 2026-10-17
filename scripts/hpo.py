"""Directed hyperparameter search with Optuna (TPE).

The search space is defined in ``configs/hpo.yaml`` (not hard-coded here);
``--trials`` / ``--epochs`` / ``--seed`` / ``--study-name`` / ``--config``
override the budget. Objective: minimize ``val/rmse``.

Usage:
    python scripts/hpo.py                            # search space from configs/hpo.yaml
    python scripts/hpo.py --trials 20 --epochs 25
    python scripts/hpo.py --trials 2 dataset=synthetic experiment=smoke   # smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna
from omegaconf import DictConfig, OmegaConf

from src.core.config import compose_config
from src.core.logging_utils import get_logger, setup_logging
from src.experiment import run_experiment

logger = get_logger(__name__)

DEFAULT_HPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "hpo.yaml"


def _suggest_head(trial: optuna.Trial, head_cfg: DictConfig) -> list[dict]:
    """Suggest a configurable-depth MLP head with per-layer units/activation/dropout."""
    depth = trial.suggest_int("head_depth", int(head_cfg.depth.low), int(head_cfg.depth.high))
    layers: list[dict] = []
    for i in range(depth):
        units = trial.suggest_int(
            f"head_{i}_units",
            int(head_cfg.units.low),
            int(head_cfg.units.high),
            log=bool(head_cfg.units.log),
        )
        activation = trial.suggest_categorical(f"head_{i}_activation", list(head_cfg.activations))
        dropout = trial.suggest_float(
            f"head_{i}_dropout", float(head_cfg.dropout.low), float(head_cfg.dropout.high)
        )
        layers.append({"units": int(units), "activation": activation, "dropout": float(dropout)})
    return layers


def make_objective(base_overrides: list[str], search_space: DictConfig, epochs: int):
    """Return an Optuna objective that runs one full experiment per trial."""
    encoders = list(search_space.encoders)
    temporals = list(search_space.temporals)
    feature_dims = list(search_space.feature_dims)
    seq = search_space.sequence_length
    wd = search_space.weight_decay
    head_cfg = search_space.head

    def objective(trial: optuna.Trial) -> float:
        encoder = trial.suggest_categorical("encoder", encoders)
        temporal = trial.suggest_categorical("temporal", temporals)
        sequence_length = trial.suggest_int(
            "sequence_length", int(seq.low), int(seq.high), step=int(seq.step)
        )
        weight_decay = trial.suggest_float(
            "weight_decay", float(wd.low), float(wd.high), log=bool(wd.log)
        )
        feature_dim = trial.suggest_categorical("feature_dim", feature_dims)
        head_layers = _suggest_head(trial, head_cfg)

        overrides = [
            *base_overrides,
            f"encoder={encoder}",
            f"temporal={temporal}",
            f"data.sequence_length={sequence_length}",
            f"training.weight_decay={weight_decay}",
            f"model.encoder.feature_dim={feature_dim}",
            f"training.epochs={epochs}",
            "training.monitor=val/rmse_mph",  # align checkpoint/early-stop with the objective
        ]
        cfg = compose_config(overrides)
        OmegaConf.update(cfg, "model.head", {"layers": head_layers}, merge=False)
        paths = run_experiment(cfg)  # validates internally
        metrics = json.loads(paths.metrics_path.read_text(encoding="utf-8"))
        val_rmse = float(metrics["splits"]["val"]["overall"]["rmse"])
        trial.set_user_attr("run_name", paths.run_name)
        logger.info("event=hpo_trial val_rmse=%.4f params=%s", val_rmse, trial.params)
        return val_rmse

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description="Directed hyperparameter search (Optuna TPE)")
    parser.add_argument("--config", default=None, help="Path to the HPO search-space YAML")
    parser.add_argument("--trials", type=int, default=None, help="Override budget.n_trials")
    parser.add_argument("--epochs", type=int, default=None, help="Override budget.epochs")
    parser.add_argument("--study-name", default=None, help="Override budget.study_name")
    parser.add_argument("--seed", type=int, default=None, help="Override budget.seed")
    parser.add_argument(
        "overrides", nargs="*", default=[], help="Extra Hydra overrides (e.g. dataset=synthetic)"
    )
    args = parser.parse_args()

    hpo_path = Path(args.config) if args.config else DEFAULT_HPO_CONFIG
    hpo_cfg = OmegaConf.load(hpo_path)
    n_trials = args.trials if args.trials is not None else int(hpo_cfg.budget.n_trials)
    epochs = args.epochs if args.epochs is not None else int(hpo_cfg.budget.epochs)
    seed = args.seed if args.seed is not None else int(hpo_cfg.budget.seed)
    study_name = args.study_name if args.study_name is not None else hpo_cfg.budget.study_name

    setup_logging(None, level="INFO")
    objective = make_objective(list(args.overrides), hpo_cfg.search_space, epochs)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))

    completed = any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if completed:
        logger.info(
            "event=hpo_complete best_val_rmse=%.4f best_params=%s",
            study.best_value,
            study.best_params,
        )
    else:
        logger.warning("event=hpo_complete no_completed_trials n_trials=%d", len(study.trials))
    out = Path("output") / "hpo_trials.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out, index=False)
    logger.info("event=hpo_trials_written path=%s n_trials=%d", out, len(study.trials))


if __name__ == "__main__":
    main()
