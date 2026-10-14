"""Tests for src.data.dataset (WP2): synthetic data, validation, splits."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from src.core.config import DataCfg, SyntheticCfg
from src.core.schemas import DataValidationError, SpeedSequence
from src.data.dataset import (
    InMemoryFrames,
    assert_no_leakage,
    generate_synthetic_sequences,
    load_or_create_split,
    make_group_split,
    validate_sequence,
)


def make_seq(
    timestamps: list[float],
    speeds: list[float],
    seq_id: str = "s0",
    group_id: str = "g0",
) -> SpeedSequence:
    n = len(timestamps)
    frames = np.zeros((n, 4, 4, 3), dtype=np.uint8)
    return SpeedSequence(
        sequence_id=seq_id,
        group_id=group_id,
        frames=InMemoryFrames(frames),
        speeds_mph=np.asarray(speeds, dtype=float),
        timestamps=np.asarray(timestamps, dtype=float),
    )


def small_cfg(frames: int = 60) -> DataCfg:
    return DataCfg(
        synthetic=SyntheticCfg(
            num_sequences=3, frames_per_sequence=frames, image_size=[16, 16], noise_std=0.02
        )
    )


def _ends_by_seq(ds):
    ends: dict[int, list[int]] = {}
    for seq_idx, end in ds.window_index:
        ends.setdefault(seq_idx, []).append(end)
    return ends


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
def test_synthetic_determinism():
    cfg = small_cfg()
    a = generate_synthetic_sequences(cfg, seed=42)
    b = generate_synthetic_sequences(cfg, seed=42)
    assert len(a) == len(b)
    for sa, sb in zip(a, b, strict=True):
        assert np.array_equal(sa.speeds_mph, sb.speeds_mph)
        assert np.array_equal(
            sa.frames.get(range(len(sa.frames))), sb.frames.get(range(len(sb.frames)))
        )


def test_synthetic_different_seed_differs():
    cfg = small_cfg()
    a = generate_synthetic_sequences(cfg, seed=42)
    b = generate_synthetic_sequences(cfg, seed=43)
    assert not np.array_equal(a[0].speeds_mph, b[0].speeds_mph)


def test_synthetic_all_buckets_and_stationary():
    cfg = small_cfg(frames=60)
    for s in generate_synthetic_sequences(cfg, seed=0):
        speeds = s.speeds_mph
        assert (speeds < 15).any(), "low bucket missing"
        assert ((speeds >= 15) & (speeds < 45)).any(), "medium bucket missing"
        assert (speeds >= 45).any(), "high bucket missing"
        assert (speeds <= 2.0).any(), "stationary frames missing"


def test_synthetic_frames_shape_dtype():
    cfg = small_cfg()
    s = generate_synthetic_sequences(cfg, seed=0)[0]
    frames = s.frames.get(range(len(s.frames)))
    assert frames.dtype == np.uint8
    assert frames.ndim == 4
    h, w = cfg.synthetic.image_size
    assert frames.shape[1:] == (h, w, 3)
    assert len(s.frames) == cfg.synthetic.frames_per_sequence
    assert s.timestamps.shape == (cfg.synthetic.frames_per_sequence,)
    assert np.allclose(np.diff(s.timestamps), 1.0 / cfg.frame_rate_hz)


def test_synthetic_group_ids_unique():
    cfg = small_cfg()
    seqs = generate_synthetic_sequences(cfg, seed=0)
    ids = [s.group_id for s in seqs]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------- #
# Validation policies
# --------------------------------------------------------------------------- #
def test_gap_split_child_lengths_and_ids():
    cfg = DataCfg()  # threshold = 1.5/10 = 0.15s
    ts = [0.0, 0.1, 0.2, 0.3, 0.4, 1.0, 1.1, 1.2, 1.3]
    seq = make_seq(ts, [float(i) for i in range(9)], seq_id="s")
    children = validate_sequence(seq, cfg)
    assert len(children) == 2
    assert children[0].sequence_id == "s#0"
    assert children[1].sequence_id == "s#1"
    assert len(children[0].frames) == 5
    assert len(children[1].frames) == 4
    assert children[0].group_id == children[1].group_id == "g0"


def test_no_gap_returns_original():
    cfg = DataCfg()
    seq = make_seq([0.0, 0.1, 0.2], [1.0, 2.0, 3.0])
    children = validate_sequence(seq, cfg)
    assert len(children) == 1
    assert children[0] is seq  # unchanged instance


def test_nonmonotonic_raises():
    with pytest.raises(DataValidationError):
        validate_sequence(make_seq([0.0, 0.1, 0.05], [1.0, 2.0, 3.0]), DataCfg())


def test_duplicate_raise():
    with pytest.raises(DataValidationError):
        validate_sequence(make_seq([0.0, 0.1, 0.1], [1.0, 2.0, 3.0]), DataCfg())


def test_duplicate_drop_later():
    cfg = DataCfg(on_duplicate_timestamp="drop_later")
    seq = make_seq([0.0, 0.1, 0.1, 0.2], [1.0, 2.0, 3.0, 4.0])
    children = validate_sequence(seq, cfg)
    assert len(children) == 1
    child = children[0]
    assert np.array_equal(child.timestamps, [0.0, 0.1, 0.2])
    assert np.array_equal(child.speeds_mph, [1.0, 2.0, 4.0])


def test_negative_speed_raises():
    with pytest.raises(DataValidationError):
        validate_sequence(make_seq([0.0, 0.1], [-1.0, 2.0]), DataCfg())


def test_negative_speed_allowed():
    cfg = DataCfg(allow_negative_speed=True)
    assert len(validate_sequence(make_seq([0.0, 0.1], [-1.0, 2.0]), cfg)) == 1


def test_max_speed_raises():
    cfg = DataCfg(max_speed_mph=100)
    with pytest.raises(DataValidationError):
        validate_sequence(make_seq([0.0, 0.1], [50.0, 150.0]), cfg)


def test_irregular_rate_warns(caplog):
    cfg = DataCfg()  # nominal 0.1s; median 0.12 -> 20% deviation, no gap
    seq = make_seq([0.0, 0.12, 0.24, 0.36], [1.0, 2.0, 3.0, 4.0])
    with caplog.at_level(logging.WARNING):
        validate_sequence(seq, cfg)
    assert "irregular_frame_rate" in caplog.text


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def test_make_group_split_deterministic_disjoint_nonempty():
    groups = [f"g{i}" for i in range(12)]
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    s1 = make_group_split(groups, ratios, seed=42)
    s2 = make_group_split(groups, ratios, seed=42)
    assert s1 == s2
    combined = s1["train"] + s1["val"] + s1["test"]
    assert sorted(combined) == sorted(groups)
    assert all(len(s1[k]) > 0 for k in s1)
    # disjoint
    assert len(set(combined)) == len(combined)


def test_make_group_split_too_few_raises():
    with pytest.raises(DataValidationError):
        make_group_split(["g0", "g1"], {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)


def test_assert_no_leakage_raises():
    with pytest.raises(DataValidationError):
        assert_no_leakage({"train": ["a", "b"], "val": ["c", "a"]})


def test_assert_no_leakage_ok():
    assert_no_leakage({"train": ["a", "b"], "val": ["c"], "test": ["d"]})


def test_load_or_create_split_manifest_reuse(tmp_path: Path):
    cfg = DataCfg()
    cfg.split_manifest = str(tmp_path / "splits" / "manifest.json")
    seqs = [make_seq([0.0, 0.1], [1.0, 2.0], seq_id=f"s{i}", group_id=f"g{i}") for i in range(6)]
    run_path = tmp_path / "split.json"
    s1 = load_or_create_split(seqs, cfg, seed=42, run_split_path=run_path)
    assert Path(cfg.split_manifest).exists()
    # reuse ignores the new seed
    s2 = load_or_create_split(seqs, cfg, seed=999, run_split_path=run_path)
    assert s1 == s2
    # copy written to run split path
    assert run_path.exists()
    payload = json.loads(run_path.read_text())
    assert payload["splits"] == s1


def test_load_or_create_split_mismatched_groups_raises(tmp_path: Path):
    cfg = DataCfg()
    cfg.split_manifest = str(tmp_path / "m.json")
    seqs = [make_seq([0.0, 0.1], [1.0, 2.0], group_id=f"g{i}") for i in range(6)]
    load_or_create_split(seqs, cfg, seed=42, run_split_path=tmp_path / "s1.json")
    seqs2 = [make_seq([0.0, 0.1], [1.0, 2.0], group_id=f"g{i + 10}") for i in range(6)]
    with pytest.raises(DataValidationError):
        load_or_create_split(seqs2, cfg, seed=42, run_split_path=tmp_path / "s2.json")


class TestDataset:
    """WP3 additions: SpeedDataset + build_datasets + build_dataloaders."""

    def test_build_datasets_item_shapes_and_dtypes(self, tiny_config):
        from src.data.dataset import build_datasets

        datasets = build_datasets(tiny_config, seed=42)
        t = tiny_config.data.sequence_length
        h, w = tiny_config.preprocessing.image_size
        for split_name in ("train", "val", "test"):
            ds = datasets[split_name]
            assert len(ds) > 0
            item = ds[0]
            assert set(item) == {
                "frames",
                "target_mph",
                "timestamp",
                "frame_index",
                "sequence_id",
            }
            assert item["frames"].shape == (t, 3, h, w)
            assert item["frames"].dtype == torch.float32
            assert item["target_mph"].dtype == torch.float32
            assert item["timestamp"].dtype == torch.float64
            assert item["frame_index"].dtype == torch.int64
            assert isinstance(item["sequence_id"], str)

    def test_label_equals_speed_at_end(self, tiny_config):
        from src.data.dataset import build_datasets

        datasets = build_datasets(tiny_config, seed=42)
        ds = datasets["val"]
        for i in range(len(ds)):
            item = ds[i]
            seq_idx, end_idx = ds.window_index[i]
            assert item["target_mph"].item() == pytest.approx(
                ds.sequences[seq_idx].speeds_mph[end_idx]
            )

    def test_no_group_overlap_between_splits(self, tiny_config):
        from src.data.dataset import build_datasets

        datasets = build_datasets(tiny_config, seed=42)
        groups = {name: {s.group_id for s in ds.sequences} for name, ds in datasets.items()}
        assert not (groups["train"] & groups["val"])
        assert not (groups["train"] & groups["test"])
        assert not (groups["val"] & groups["test"])

    def test_train_stride_two_val_test_stride_one(self, tiny_config):
        from src.data.dataset import build_datasets

        tiny_config.data.window_stride = 2
        datasets = build_datasets(tiny_config, seed=42)
        for _seq_idx, ends in _ends_by_seq(datasets["train"]).items():
            assert all(b - a == 2 for a, b in zip(ends, ends[1:], strict=False))
        for split_name in ("val", "test"):
            for _seq_idx, ends in _ends_by_seq(datasets[split_name]).items():
                assert all(b - a == 1 for a, b in zip(ends, ends[1:], strict=False))

    def test_build_dataloaders(self, tiny_config):
        from src.data.dataset import build_dataloaders, build_datasets

        datasets = build_datasets(tiny_config, seed=42)
        loaders = build_dataloaders(datasets, tiny_config)
        assert set(loaders) == {"train", "val", "test"}
        # batch dict contract
        batch = next(iter(loaders["train"]))
        t = tiny_config.data.sequence_length
        h, w = tiny_config.preprocessing.image_size
        assert batch["frames"].shape == (tiny_config.training.batch_size, t, 3, h, w)
        assert batch["target_mph"].shape == (tiny_config.training.batch_size,)
        assert isinstance(batch["sequence_id"], list)
