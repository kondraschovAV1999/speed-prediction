"""Speed buckets: config-driven binning of speeds for evaluation reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.schemas import ConfigError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.config import EvaluationCfg


@dataclass(frozen=True)
class BucketSpec:
    """A half-open speed interval ``[min_mph, max_mph)``; ``max_mph=None`` = +inf."""

    name: str
    min_mph: float
    max_mph: float | None


def buckets_from_cfg(cfg: EvaluationCfg) -> list[BucketSpec]:
    """Build :class:`BucketSpec` objects from the typed evaluation config."""
    buckets: list[BucketSpec] = []
    seen: set[str] = set()
    for b in cfg.speed_buckets:
        name = b.name
        if name in seen:
            raise ConfigError(f"Duplicate bucket name in config: {name!r}")
        seen.add(name)
        buckets.append(BucketSpec(name=name, min_mph=float(b.min_mph), max_mph=b.max_mph))
    return buckets


def assign_buckets(speeds: np.ndarray, buckets: list[BucketSpec]) -> np.ndarray:
    """Assign each speed to exactly one bucket name (``min <= v < max``).

    Returns an object array of names.  A value that falls outside every bucket
    raises :class:`ConfigError`; overlapping buckets also raise.
    """
    speeds = np.asarray(speeds, dtype=float)
    if speeds.ndim == 0:
        speeds = speeds.reshape(1)
    names = np.empty(speeds.shape[0], dtype=object)
    assigned = np.zeros(speeds.shape[0], dtype=bool)
    for b in buckets:
        mask = speeds >= b.min_mph
        if b.max_mph is not None:
            mask &= speeds < b.max_mph
        if np.any(assigned & mask):
            raise ConfigError(f"Overlapping bucket definitions involving {b.name!r}")
        names[mask] = b.name
        assigned |= mask
    if not assigned.all():
        idx = int(np.argmin(~assigned))
        raise ConfigError(f"Speed value {speeds[idx]} falls outside all configured buckets")
    return names
