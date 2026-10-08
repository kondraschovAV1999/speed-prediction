"""Tests for src.buckets (bucket assignment and config parsing)."""

from __future__ import annotations

import numpy as np
import pytest

from src.buckets import BucketSpec, assign_buckets, buckets_from_cfg
from src.config import BucketCfg, EvaluationCfg
from src.schemas import ConfigError

BUCKETS = [
    BucketSpec("low", 0.0, 15.0),
    BucketSpec("medium", 15.0, 45.0),
    BucketSpec("high", 45.0, None),
]


def test_boundary_assignment():
    speeds = np.array([0.0, 14.99, 15.0, 44.99, 45.0, 200.0])
    names = assign_buckets(speeds, BUCKETS)
    assert list(names) == ["low", "low", "medium", "medium", "high", "high"]


def test_max_none_treated_as_inf():
    names = assign_buckets(np.array([45.0, 1e9]), BUCKETS)
    assert list(names) == ["high", "high"]


def test_value_below_first_bucket_raises():
    with pytest.raises(ConfigError):
        assign_buckets(np.array([-1.0]), BUCKETS)


def test_assign_returns_object_array_of_names():
    names = assign_buckets(np.array([10.0, 30.0, 60.0]), BUCKETS)
    assert names.dtype == object
    assert list(names) == ["low", "medium", "high"]


def test_buckets_from_cfg_ordering_and_fields():
    cfg = EvaluationCfg(
        speed_buckets=[
            BucketCfg("low", 0, 15),
            BucketCfg("medium", 15, 45),
            BucketCfg("high", 45, None),
        ]
    )
    specs = buckets_from_cfg(cfg)
    assert [s.name for s in specs] == ["low", "medium", "high"]
    assert specs[0].min_mph == 0.0 and specs[0].max_mph == 15.0
    assert specs[2].max_mph is None


def test_buckets_from_cfg_duplicate_names_raise():
    cfg = EvaluationCfg(speed_buckets=[BucketCfg("low", 0, 15), BucketCfg("low", 15, 45)])
    with pytest.raises(ConfigError):
        buckets_from_cfg(cfg)
