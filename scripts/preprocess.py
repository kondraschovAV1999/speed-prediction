"""Stage-1 CLI: decode comma2k19 segments into the interim cache + manifest."""

from __future__ import annotations

import time
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
    total = len(segments) if max_segments is None else min(len(segments), max_segments)
    logger.info("event=preprocess_start segments=%d cache_dir=%s", total, cache_dir)
    for i, (route, seg, seg_dir) in enumerate(segments, start=1):
        if max_segments is not None and len(manifest) >= max_segments:
            break
        seg_id = f"{route}/{seg}"
        base = f"{route}__{seg}"
        frames_fname = f"{base}.frames.npy"
        meta_fname = f"{base}.meta.npz"
        logger.info("event=cache_segment_start index=%d/%d segment=%s", i, total, seg_id)
        started = time.perf_counter()
        info = cache_segment(seg_dir, cache_dir / frames_fname, cache_dir / meta_fname, image_size)
        elapsed = time.perf_counter() - started
        manifest[seg_id] = {
            "frames_file": frames_fname,
            "meta_file": meta_fname,
            "n_frames": info["n_frames"],
            "group_id": route,
        }
        logger.info(
            "event=cache_segment segment=%s frames=%d elapsed_s=%.1f",
            seg_id,
            info["n_frames"],
            elapsed,
        )

    build_manifest(cache_dir, manifest)
    logger.info("event=preprocess_complete segments=%d cache_dir=%s", len(manifest), cache_dir)


if __name__ == "__main__":
    main()
