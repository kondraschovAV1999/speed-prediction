"""Tests for src.inference.pipeline (WP9): raw collection, streaming, parity."""

from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import SpeedDataset, build_dataloaders, build_datasets
from src.data.preprocessing import build_transform, build_window_index
from src.inference.pipeline import StreamingSpeedEstimator, collect_raw_predictions
from src.inference.postprocessing import SpeedPostProcessor, postprocess_dataframe
from src.models.architectures import build_model


def _make_estimator(tiny_config, model=None, sequence_id: str = ""):
    model = model if model is not None else build_model(tiny_config.model)
    transform = build_transform(tiny_config.preprocessing)
    proc = SpeedPostProcessor(tiny_config.postprocessing)
    t = tiny_config.data.sequence_length
    k = tiny_config.data.frame_step
    max_gap = tiny_config.data.max_gap_factor / tiny_config.data.frame_rate_hz
    est = StreamingSpeedEstimator(model, transform, proc, t, k, max_gap, "cpu", sequence_id)
    return est, transform, model


def test_streaming_returns_none_until_buffer_full(tiny_config):
    est, _, _ = _make_estimator(tiny_config)
    t = tiny_config.data.sequence_length
    h, w = tiny_config.preprocessing.image_size
    results = []
    for i in range(t + 3):
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        results.append(est.update(frame, i * 0.1))
    assert all(r is None for r in results[: t - 1])
    assert all(r is not None for r in results[t - 1 :])


def test_streaming_gap_resets_buffer(tiny_config):
    est, _, _ = _make_estimator(tiny_config)
    t = tiny_config.data.sequence_length
    h, w = tiny_config.preprocessing.image_size
    frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    for i in range(t):
        est.update(frame, i * 0.1)
    # A large timestamp jump clears the buffer; the next frame is not a full window.
    assert est.update(frame, 100.0) is None


def test_collect_raw_predictions_columns_and_eval(tiny_config):
    datasets = build_datasets(tiny_config, seed=42)
    loaders = build_dataloaders(datasets, tiny_config)
    model = build_model(tiny_config.model)
    df = collect_raw_predictions(model, loaders["val"], "cpu", "val")
    assert list(df.columns) == [
        "split",
        "sequence_id",
        "frame_index",
        "timestamp",
        "true_speed_mph",
        "raw_speed_mph",
        "inference_ms",
    ]
    assert len(df) > 0
    assert not model.training
    assert df["inference_ms"].isna().all()


def test_streaming_offline_parity(tiny_config):
    datasets = build_datasets(tiny_config, seed=42)
    val_ds = datasets["val"]
    seq = val_ds.sequences[0]

    transform = build_transform(tiny_config.preprocessing)
    t = tiny_config.data.sequence_length
    k = tiny_config.data.frame_step

    # Offline path.
    windows = build_window_index([len(seq.frames)], t, 1, k)
    single_ds = SpeedDataset([seq], windows, transform, t, k)
    loader = DataLoader(single_ds, batch_size=1, shuffle=False)
    model = build_model(tiny_config.model)
    raw_df = collect_raw_predictions(model, loader, "cpu", "test")
    final_df = postprocess_dataframe(raw_df, tiny_config.postprocessing)

    # Streaming path (same model weights).
    est, _, _ = _make_estimator(tiny_config, model=model, sequence_id=seq.sequence_id)
    predictions = []
    for i in range(len(seq.frames)):
        frame = seq.frames.get([i])[0]
        p = est.update(frame, float(seq.timestamps[i]))
        if p is not None:
            predictions.append(p)

    assert len(predictions) == len(final_df)

    offline_raw = final_df["raw_speed_mph"].to_numpy(dtype=float)
    offline_final = final_df["final_speed_mph"].to_numpy(dtype=float)
    offline_stationary = final_df["stationary_pred"].to_numpy(dtype=bool)

    stream_raw = np.array([p.raw_speed_mph for p in predictions])
    stream_final = np.array([p.final_speed_mph for p in predictions])
    stream_stationary = np.array([p.stationary for p in predictions])

    assert np.allclose(offline_raw, stream_raw, atol=1e-5)
    assert np.allclose(offline_final, stream_final, atol=1e-6)
    assert np.array_equal(offline_stationary, stream_stationary)
