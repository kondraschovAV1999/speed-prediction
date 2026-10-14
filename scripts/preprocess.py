"""Stage-1 CLI: decode comma2k19 segments into the interim cache + manifest."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.core.config import validate_config
from src.core.logging_utils import get_logger, setup_logging
from src.core.schemas import DataValidationError
from src.data.comma2k19 import build_manifest, cache_segment, discover_segments

logger = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = validate_config(cfg)
    setup_logging(None, level=cfg.logging.level)
    data = cfg.data
    cache_dir = Path(data.comma2k19.cache_dir)
    image_size = tuple(data.comma2k19.cache_image_size)

    segments = discover_segments(Path(data.root))
    if not segments:
        raise DataValidationError(f"No segments found under {Path(data.root)!r}")

    manifest: dict[str, dict] = {}
    max_segments = data.comma2k19.max_segments
    for route, seg, seg_dir in segments:
        if max_segments is not None and len(manifest) >= max_segments:
            break
        seg_id = f"{route}/{seg}"
        fname = f"{route}__{seg}.npz"
        info = cache_segment(seg_dir, cache_dir / fname, image_size)
        manifest[seg_id] = {"file": fname, "n_frames": info["n_frames"], "group_id": route}
        logger.info("event=cache_segment segment=%s frames=%d", seg_id, info["n_frames"])

    build_manifest(cache_dir, manifest)
    logger.info("event=preprocess_complete segments=%d cache_dir=%s", len(manifest), cache_dir)


if __name__ == "__main__":
    main()
