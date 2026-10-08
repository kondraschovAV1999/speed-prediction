"""Frame transforms and window indexing (WP3).

Frames are ``uint8 [H, W, C]`` at the I/O boundary and ``float32 [C, H, W]``
inside tensors.  No OpenCV: resizing uses ``torch.nn.functional.interpolate``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.config import PreprocessingCfg


class Resize:
    """Resize a ``uint8 [H, W, C]`` frame to ``size=(H, W)`` via bilinear interpolate."""

    def __init__(self, size: Sequence[int]):
        self.size = (int(size[0]), int(size[1]))

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).unsqueeze(0).float()
        t = F.interpolate(t, size=self.size, mode="bilinear", align_corners=False)
        out = t.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()
        return out


class ToTensor:
    """Convert ``uint8 [H, W, C]`` -> ``float32 [C, H, W]`` in [0, 1]."""

    def __call__(self, frame: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0


class Normalize:
    """Normalize a ``float32 [C, H, W]`` tensor with fixed mean/std constants."""

    def __init__(self, mean: list[float], std: list[float]):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        return (frame - self.mean) / self.std


class Compose:
    """Apply a sequence of per-frame transforms in order."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, frame: np.ndarray) -> torch.Tensor:
        result: np.ndarray | torch.Tensor = frame
        for transform in self.transforms:
            result = transform(result)
        assert isinstance(result, torch.Tensor)
        return result


def build_transform(cfg: PreprocessingCfg) -> Compose:
    """Build the per-frame transform pipeline (Resize -> ToTensor -> [Normalize])."""
    transforms: list = [Resize(cfg.image_size), ToTensor()]
    if cfg.normalize:
        transforms.append(Normalize(cfg.mean, cfg.std))
    return Compose(transforms)


def build_window_index(
    seq_lengths: list[int], sequence_length: int, stride: int, frame_step: int
) -> list[tuple[int, int]]:
    """Enumerate valid window end positions ``(seq_idx, end_idx)``.

    A window of ``T`` frames with ``frame_step=k`` needs ``end >= (T-1)*k``.
    No padding: sequences shorter than a window contribute zero windows.
    """
    min_end = (sequence_length - 1) * frame_step
    windows: list[tuple[int, int]] = []
    for seq_idx, length in enumerate(seq_lengths):
        for end in range(min_end, length, stride):
            windows.append((seq_idx, end))
    return windows


def window_frame_indices(end_idx: int, sequence_length: int, frame_step: int) -> list[int]:
    """Frame indices for a window ending at ``end_idx``, in ascending order."""
    return [end_idx - (sequence_length - 1 - i) * frame_step for i in range(sequence_length)]
