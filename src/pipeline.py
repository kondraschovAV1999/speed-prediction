"""Raw prediction collection and the streaming estimator (WP9)."""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import torch

from src.schemas import SpeedPrediction

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> torch.device:
    """Resolve a device string ("auto" -> cuda if available, else cpu)."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def collect_raw_predictions(
    model: torch.nn.Module, dataloader, device: str, split: str
) -> pd.DataFrame:
    """Run the model over a dataloader and return the raw predictions table.

    Columns (canonical raw order, §5.4 without final/stationary):
    ``split, sequence_id, frame_index, timestamp, true_speed_mph, raw_speed_mph,
    inference_ms``. ``inference_ms`` is NaN in batch evaluation.
    """
    model.eval()
    dev = resolve_device(device)
    model = model.to(dev)

    rows: dict[str, list] = {
        "split": [],
        "sequence_id": [],
        "frame_index": [],
        "timestamp": [],
        "true_speed_mph": [],
        "raw_speed_mph": [],
        "inference_ms": [],
    }
    with torch.inference_mode():
        for batch in dataloader:
            frames = batch["frames"].to(dev)
            pred = model(frames).cpu().numpy()
            n = pred.shape[0]
            rows["split"].extend([split] * n)
            rows["sequence_id"].extend(batch["sequence_id"])
            rows["frame_index"].extend(batch["frame_index"].cpu().numpy().tolist())
            rows["timestamp"].extend(batch["timestamp"].cpu().numpy().tolist())
            rows["true_speed_mph"].extend(batch["target_mph"].cpu().numpy().tolist())
            rows["raw_speed_mph"].extend(pred.tolist())
            rows["inference_ms"].extend([float("nan")] * n)

    df = pd.DataFrame(rows)
    return df.sort_values(["sequence_id", "frame_index"]).reset_index(drop=True)


class StreamingSpeedEstimator:
    """Causal streaming inference: transform -> model -> post-process per frame.

    Returns ``None`` until the ring buffer holds a full window; clears the
    buffer and resets post-processing on a timestamp gap or non-increasing
    timestamp (mirroring the offline gap split).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        transform,
        postprocessor,
        sequence_length: int,
        frame_step: int,
        max_gap_s: float,
        device: str,
        sequence_id: str = "",
    ):
        self.model = model
        self.transform = transform
        self.postprocessor = postprocessor
        self.sequence_length = sequence_length
        self.frame_step = frame_step
        self.max_gap_s = max_gap_s
        self.device = resolve_device(device)
        self.sequence_id = sequence_id
        self.model = self.model.to(self.device).eval()
        self.reset()

    def reset(self) -> None:
        """Reset buffer, timestamps and post-processing state."""
        self._buffer: list[torch.Tensor] = []
        self._timestamps: list[float] = []
        self._last_ts: float | None = None
        self._frame_counter = 0
        self.postprocessor.reset()

    def _clear_for_gap(self) -> None:
        self._buffer = []
        self._timestamps = []
        self._frame_counter = 0
        self.postprocessor.reset()

    def update(self, frame_hwc_uint8: np.ndarray, timestamp: float) -> SpeedPrediction | None:
        """Process one frame; returns a prediction once the buffer is full."""
        if self._last_ts is not None:
            dt = timestamp - self._last_ts
            if dt <= 0 or dt > self.max_gap_s:
                logger.warning(
                    "event=stream_gap sequence_id=%s last=%.4f curr=%.4f",
                    self.sequence_id,
                    self._last_ts,
                    timestamp,
                )
                self._clear_for_gap()

        transformed = self.transform(frame_hwc_uint8)  # [C, H, W]
        self._buffer.append(transformed)
        self._timestamps.append(timestamp)
        self._last_ts = timestamp
        frame_index = self._frame_counter
        self._frame_counter += 1

        required = (self.sequence_length - 1) * self.frame_step + 1
        if len(self._buffer) < required:
            return None

        # Window frames: positions -((T-1)*k), ..., -k, 0 relative to the end.
        t = self.sequence_length
        k = self.frame_step
        positions = [len(self._buffer) - 1 - (t - 1 - i) * k for i in range(t)]
        window = torch.stack([self._buffer[p] for p in positions]).unsqueeze(0)  # [1,T,C,H,W]

        with torch.inference_mode():
            start = time.perf_counter()
            raw = self.model(window.to(self.device)).cpu().item()
            end = time.perf_counter()

        result = self.postprocessor.step(raw)
        return SpeedPrediction(
            sequence_id=self.sequence_id,
            frame_index=frame_index,
            timestamp=timestamp,
            raw_speed_mph=float(raw),
            final_speed_mph=result.final_speed_mph,
            stationary=result.stationary,
            inference_ms=(end - start) * 1000.0,
        )
