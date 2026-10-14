"""Sequences, validation, splits and synthetic data (WP2).

WP3 appends ``SpeedDataset`` / ``build_datasets`` / ``build_dataloaders`` to this
module.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.core.config import Config, DataCfg
from src.core.logging_utils import get_logger
from src.core.schemas import ConfigError, DataValidationError, SpeedSequence
from src.data.comma2k19 import load_comma2k19_sequences
from src.data.preprocessing import build_transform, window_frame_indices

logger = get_logger(__name__)

# Pixels of horizontal shift per mph per frame for the synthetic stripe pattern.
_PIXELS_PER_MPH = 1.0


class InMemoryFrames:
    """A :class:`FrameSource` backed by a ``uint8 [N, H, W, C]`` array."""

    def __init__(self, frames: np.ndarray):
        arr = np.asarray(frames, dtype=np.uint8)
        if arr.ndim != 4:
            raise ValueError(f"frames must be a 4-D array [N, H, W, C], got shape {arr.shape}")
        self._frames = arr

    def __len__(self) -> int:
        return self._frames.shape[0]

    def get(self, indices: Sequence[int]) -> np.ndarray:
        return self._frames[list(indices)]  # uint8 [k, H, W, C]


# --------------------------------------------------------------------------- #
# Synthetic data generation
# --------------------------------------------------------------------------- #
def _synthetic_speed_profile(n: int, rng: np.random.Generator) -> np.ndarray:
    """Stopped -> ramp up -> hold -> ramp down -> stopped (fractions of length)."""
    peak = float(rng.uniform(50.0, 70.0))
    n_stop1 = int(round(n * 0.15))
    n_ramp_up = int(round(n * 0.30))
    n_hold = int(round(n * 0.15))
    n_ramp_down = int(round(n * 0.25))
    n_stop2 = n - n_stop1 - n_ramp_up - n_hold - n_ramp_down

    speeds = np.empty(n, dtype=np.float64)
    pos = 0
    speeds[pos : pos + n_stop1] = rng.uniform(0.0, 1.5, size=n_stop1)
    pos += n_stop1
    speeds[pos : pos + n_ramp_up] = np.linspace(1.5, peak, num=n_ramp_up)
    pos += n_ramp_up
    speeds[pos : pos + n_hold] = peak
    pos += n_hold
    speeds[pos : pos + n_ramp_down] = np.linspace(peak, 1.5, num=n_ramp_down)
    pos += n_ramp_down
    speeds[pos:] = rng.uniform(0.0, 1.5, size=n_stop2)
    return speeds


def _make_stripe_frame(h: int, w: int, offset: float, stripe_width: int = 4) -> np.ndarray:
    """A vertical-stripe frame (uint8 [H, W, 3]) shifted by ``offset`` pixels."""
    x = np.arange(w, dtype=np.float64)
    shifted = np.mod(x - offset, w)
    band = (np.floor(shifted / stripe_width).astype(np.int64)) % 2
    intensity = (band * 255).astype(np.uint8)  # [w]
    frame = np.empty((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = intensity
    frame[:, :, 1] = intensity
    frame[:, :, 2] = intensity
    return frame


def generate_synthetic_sequences(cfg: DataCfg, seed: int) -> list[SpeedSequence]:
    """Generate deterministic synthetic sequences (motion proportional to speed)."""
    rng = np.random.default_rng(seed)
    syn = cfg.synthetic
    h, w = int(syn.image_size[0]), int(syn.image_size[1])
    sequences: list[SpeedSequence] = []
    for i in range(syn.num_sequences):
        n = int(syn.frames_per_sequence)
        speeds = _synthetic_speed_profile(n, rng)
        frames = np.empty((n, h, w, 3), dtype=np.uint8)
        offset = 0.0
        for t in range(n):
            offset = (offset + speeds[t] * _PIXELS_PER_MPH) % w
            stripe = _make_stripe_frame(h, w, offset)
            noise = rng.normal(0.0, syn.noise_std, size=(h, w, 3))
            frame = np.clip(stripe.astype(np.float64) / 255.0 + noise, 0.0, 1.0)
            frames[t] = (frame * 255.0).astype(np.uint8)
        timestamps = np.arange(n, dtype=np.float64) / cfg.frame_rate_hz
        sequences.append(
            SpeedSequence(
                sequence_id=f"synth_{i:02d}",
                group_id=f"synth_{i:02d}",
                frames=InMemoryFrames(frames),
                speeds_mph=speeds,
                timestamps=timestamps,
            )
        )
    return sequences


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _rebuild_child(seq: SpeedSequence, indices: np.ndarray, sequence_id: str) -> SpeedSequence:
    """Materialize a child sequence over ``indices`` (into the original frames)."""
    frames = InMemoryFrames(seq.frames.get(list(indices)))
    return SpeedSequence(
        sequence_id=sequence_id,
        group_id=seq.group_id,
        frames=frames,
        speeds_mph=seq.speeds_mph[indices].copy(),
        timestamps=seq.timestamps[indices].copy(),
    )


def validate_sequence(seq: SpeedSequence, cfg: DataCfg) -> list[SpeedSequence]:
    """Validate a sequence, applying the §6.7 policies.

    Returns one or more sequences: duplicates are dropped (if configured) and
    time gaps split the sequence into ``<id>#<k>`` children.
    """
    timestamps = np.asarray(seq.timestamps, dtype=float)
    speeds = np.asarray(seq.speeds_mph, dtype=float)
    n = timestamps.shape[0]
    if n == 0:
        return []

    keep = np.ones(n, dtype=bool)

    if n >= 2:
        diffs = np.diff(timestamps)
        if np.any(diffs < 0):
            raise DataValidationError(
                f"sequence {seq.sequence_id}: timestamps are not monotonically increasing"
            )
        duplicates = diffs == 0
        if np.any(duplicates):
            n_dup = int(duplicates.sum())
            if cfg.on_duplicate_timestamp == "raise":
                raise DataValidationError(
                    f"sequence {seq.sequence_id}: {n_dup} duplicate timestamp(s)"
                )
            logger.warning(
                "event=duplicate_timestamps sequence_id=%s dropped=%d",
                seq.sequence_id,
                n_dup,
            )
            keep[1:] &= ~duplicates

    # Apply drop_later filtering before further checks.
    if not keep.all():
        timestamps = timestamps[keep]
        speeds = speeds[keep]
        orig_indices = np.nonzero(keep)[0]
        n = timestamps.shape[0]
    else:
        orig_indices = np.arange(n, dtype=np.int64)

    valid = ~np.isnan(speeds)
    if not cfg.allow_negative_speed and np.any(speeds[valid] < 0):
        raise DataValidationError(
            f"sequence {seq.sequence_id}: negative speed labels present "
            "(set allow_negative_speed to permit)"
        )
    if np.any(speeds[valid] > cfg.max_speed_mph):
        raise DataValidationError(
            f"sequence {seq.sequence_id}: speed exceeds max_speed_mph={cfg.max_speed_mph}"
        )

    # Irregular frame rate warning (no resampling).
    if n >= 2:
        dt = np.diff(timestamps)
        nominal = 1.0 / cfg.frame_rate_hz
        median_dt = float(np.median(dt))
        if median_dt > 0 and abs(median_dt - nominal) / nominal > 0.10:
            logger.warning(
                "event=irregular_frame_rate sequence_id=%s median_dt=%.4f nominal=%.4f",
                seq.sequence_id,
                median_dt,
                nominal,
            )

    # Gap splitting.
    threshold = cfg.max_gap_factor / cfg.frame_rate_hz
    if n >= 2:
        gap_starts = np.nonzero(np.diff(timestamps) > threshold)[0] + 1
    else:
        gap_starts = np.array([], dtype=int)

    if gap_starts.size == 0:
        if keep.all():
            return [seq]
        return [_rebuild_child(seq, orig_indices, seq.sequence_id)]

    children: list[SpeedSequence] = []
    starts = [0, *gap_starts.tolist()]
    ends = [*gap_starts.tolist(), n]
    for k, (s, e) in enumerate(zip(starts, ends, strict=True)):
        children.append(_rebuild_child(seq, orig_indices[s:e], f"{seq.sequence_id}#{k}"))
    return children


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def make_group_split(
    group_ids: list[str], ratios: dict[str, float], seed: int
) -> dict[str, list[str]]:
    """Deterministic shuffle of sorted unique group ids, partitioned by ratios."""
    unique = sorted(set(group_ids))
    order = list(unique)
    np.random.default_rng(seed).shuffle(order)
    n = len(order)

    train_n = int(round(ratios["train"] * n))
    val_n = int(round(ratios["val"] * n))
    test_n = n - train_n - val_n
    if train_n < 1 or val_n < 1 or test_n < 1:
        raise DataValidationError(
            f"too few groups ({n}) to form non-empty train/val/test splits "
            f"(got sizes train={train_n}, val={val_n}, test={test_n})"
        )
    return {
        "train": order[:train_n],
        "val": order[train_n : train_n + val_n],
        "test": order[train_n + val_n :],
    }


def assert_no_leakage(split: dict[str, list[str]]) -> None:
    """Raise if any group appears in more than one split."""
    seen: set[str] = set()
    for name, groups in split.items():
        for group in groups:
            if group in seen:
                raise DataValidationError(
                    f"group {group!r} appears in multiple splits (first in {name})"
                )
            seen.add(group)


def load_or_create_split(
    sequences: list[SpeedSequence],
    cfg: DataCfg,
    seed: int,
    run_split_path: Path,
) -> dict[str, list[str]]:
    """Load the split manifest if present, else create and persist it.

    Reuses an existing manifest for reproducibility, verifying it covers exactly
    the available groups. A copy is always written to ``run_split_path``.
    """
    group_ids = sorted({s.group_id for s in sequences})
    manifest_path = Path(cfg.split_manifest)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split = manifest["splits"]
        covered = {g for groups in split.values() for g in groups}
        if covered != set(group_ids):
            raise DataValidationError(
                f"split manifest {manifest_path} does not cover exactly the available "
                f"groups (manifest={sorted(covered)}, actual={sorted(group_ids)})"
            )
        assert_no_leakage(split)
    else:
        split = make_group_split(group_ids, cfg.split_ratios, seed)
        manifest = {"seed": seed, "ratios": cfg.split_ratios, "splits": split}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    run_split_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": seed, "ratios": cfg.split_ratios, "splits": split}
    run_split_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return split


# --------------------------------------------------------------------------- #
# Dataset and dataloaders (WP3)
# --------------------------------------------------------------------------- #
class SpeedDataset(Dataset):
    """Maps a window index over a list of sequences to ``(frames, target, meta)``."""

    def __init__(
        self,
        sequences: list[SpeedSequence],
        window_index: list[tuple[int, int]],
        transform,
        sequence_length: int,
        frame_step: int,
    ):
        self.sequences = sequences
        self.window_index = window_index
        self.transform = transform
        self.sequence_length = sequence_length
        self.frame_step = frame_step

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, index: int) -> dict:
        seq_idx, end_idx = self.window_index[index]
        seq = self.sequences[seq_idx]
        indices = window_frame_indices(end_idx, self.sequence_length, self.frame_step)
        frames = seq.frames.get(indices)  # uint8 [T, H, W, C]
        frames_tensor = torch.stack([self.transform(frame) for frame in frames])
        return {
            "frames": frames_tensor,  # [T, C, H, W]
            "target_mph": torch.tensor(seq.speeds_mph[end_idx], dtype=torch.float32),
            "timestamp": torch.tensor(seq.timestamps[end_idx], dtype=torch.float64),
            "frame_index": torch.tensor(end_idx, dtype=torch.int64),
            "sequence_id": seq.sequence_id,
        }


def _build_split_windows(
    sequences: list[SpeedSequence], data_cfg: DataCfg, stride: int
) -> list[tuple[int, int]]:
    """Enumerate windows for a split, dropping NaN-target windows."""
    from src.data.preprocessing import build_window_index

    seq_lengths = [len(s.frames) for s in sequences]
    windows: list[tuple[int, int]] = []
    dropped_nan = 0
    for seq_idx, end_idx in build_window_index(
        seq_lengths, data_cfg.sequence_length, stride, data_cfg.frame_step
    ):
        if np.isnan(sequences[seq_idx].speeds_mph[end_idx]):
            dropped_nan += 1
            continue
        windows.append((seq_idx, end_idx))
    if dropped_nan:
        logger.info("event=nan_windows_dropped count=%d", dropped_nan)
    return windows


def build_datasets(
    cfg: Config, seed: int, run_split_path: Path | None = None
) -> dict[str, SpeedDataset]:
    """Generate/validate synthetic sequences, split, and build per-split datasets."""
    data_cfg = cfg.data

    if data_cfg.source == "synthetic":
        sequences = generate_synthetic_sequences(data_cfg, seed)
    elif data_cfg.source == "comma2k19":
        sequences = load_comma2k19_sequences(data_cfg, seed)
    else:
        raise ConfigError(f"Unknown data source: {data_cfg.source!r}")
    validated: list[SpeedSequence] = []
    for seq in sequences:
        validated.extend(validate_sequence(seq, data_cfg))

    if run_split_path is None:
        run_split_path = Path(data_cfg.split_manifest).with_name("split_run.json")
    split = load_or_create_split(validated, data_cfg, seed, run_split_path)

    transform = build_transform(cfg.preprocessing)
    datasets: dict[str, SpeedDataset] = {}
    for split_name in ("train", "val", "test"):
        split_sequences = [s for s in validated if s.group_id in split[split_name]]
        stride = data_cfg.window_stride if split_name == "train" else 1
        windows = _build_split_windows(split_sequences, data_cfg, stride=stride)
        if not windows:
            raise DataValidationError(
                f"split {split_name!r} produced zero windows "
                "(sequences too short or all labels NaN)"
            )
        datasets[split_name] = SpeedDataset(
            split_sequences,
            windows,
            transform,
            data_cfg.sequence_length,
            data_cfg.frame_step,
        )

    logger.info(
        "event=dataset_sizes train=%d val=%d test=%d",
        len(datasets["train"]),
        len(datasets["val"]),
        len(datasets["test"]),
    )
    return datasets


def build_dataloaders(datasets: dict[str, SpeedDataset], cfg: Config) -> dict[str, DataLoader]:
    """Build DataLoaders: train shuffled with ``drop_last``, val/test unshuffled."""
    loaders: dict[str, DataLoader] = {}
    for split_name, dataset in datasets.items():
        is_train = split_name == "train"
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=is_train,
            drop_last=is_train,
            num_workers=cfg.training.num_workers,
            persistent_workers=cfg.training.num_workers > 0,
        )
    return loaders
