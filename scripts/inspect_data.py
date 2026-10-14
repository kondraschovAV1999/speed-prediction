"""Inspect the comma2k19 cache: counts, speed stats, 20 Hz, split sizes."""

from __future__ import annotations

import hydra
import numpy as np
from omegaconf import DictConfig

from src.core.config import validate_config
from src.core.logging_utils import get_logger, setup_logging
from src.data.comma2k19 import load_comma2k19_sequences
from src.data.dataset import make_group_split

logger = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = validate_config(cfg)
    setup_logging(None, level=cfg.logging.level)
    sequences = load_comma2k19_sequences(cfg.data, cfg.experiment.seed)
    logger.info("event=inspect_sequences count=%d", len(sequences))
    if not sequences:
        logger.warning("event=inspect_no_sequences")
        return
    for seq in sequences:
        logger.info(
            "event=inspect_sequence id=%s group=%s frames=%d",
            seq.sequence_id,
            seq.group_id,
            len(seq.frames),
        )
    seq = sequences[0]
    dt = float(np.median(np.diff(seq.timestamps)))
    speeds = seq.speeds_mph
    logger.info("event=inspect_median_dt dt=%.4f expected=0.05", dt)
    logger.info(
        "event=inspect_speed min=%.2f max=%.2f mean=%.2f",
        float(speeds.min()),
        float(speeds.max()),
        float(speeds.mean()),
    )
    group_ids = sorted({s.group_id for s in sequences})
    if len(group_ids) >= 3:
        split = make_group_split(group_ids, cfg.data.split_ratios, cfg.experiment.seed)
        for name, groups in split.items():
            logger.info("event=inspect_split split=%s groups=%d", name, len(groups))
    else:
        logger.warning(
            "event=inspect_split_skipped groups=%d reason=need_at_least_3_routes_for_a_split",
            len(group_ids),
        )


if __name__ == "__main__":
    main()
