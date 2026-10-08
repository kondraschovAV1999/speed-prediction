"""Train entry point. Thin: compose config and delegate to ``run_experiment``."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.experiment import run_experiment


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_experiment(cfg)


if __name__ == "__main__":
    main()
