"""Typed configuration: dataclasses mirroring ``configs/default.yaml`` 1:1,
plus ``compose_config`` / ``validate_config``.

Downstream code receives typed objects (``cfg.postprocessing.stationary.
enter_threshold_mph``), never raw dicts.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.core.schemas import ConfigError

logger = logging.getLogger(__name__)

# Repository root: three levels up (src/core/config.py -> <root>/).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"


# --------------------------------------------------------------------------- #
# Config dataclasses (mirror configs/default.yaml 1:1)
# --------------------------------------------------------------------------- #
@dataclass
class PathsCfg:
    output_root: str = "output"
    data_root: str = "data"


@dataclass
class ExperimentCfg:
    name: str = "baseline"
    seed: int = 42
    deterministic: bool = False


@dataclass
class SyntheticCfg:
    num_sequences: int = 12
    frames_per_sequence: int = 120
    image_size: list[int] = field(default_factory=lambda: [32, 32])
    noise_std: float = 0.02


@dataclass
class Comma2k19Cfg:
    speed_source: str = "global_pose"
    use_cache: bool = True
    cache_dir: str = "data/interim/comma2k19"
    cache_image_size: list[int] = field(default_factory=lambda: [128, 128])
    max_segments: int | None = None  # null = all; int = limit for smoke/pilot


@dataclass
class DataCfg:
    source: str = "synthetic"
    root: str = "data/processed"
    split_manifest: str = "data/splits/synthetic_seed42.json"
    split_ratios: dict[str, float] = field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )
    sequence_length: int = 10  # T frames per window
    window_stride: int = 1  # stride between window end indices (eval requires 1)
    frame_step: int = 1  # take every k-th frame inside a window
    frame_rate_hz: int = 10
    max_gap_factor: float = 1.5  # dt > factor/frame_rate_hz splits the sequence
    on_duplicate_timestamp: str = "raise"  # raise | drop_later
    allow_negative_speed: bool = False
    max_speed_mph: float = 200
    synthetic: SyntheticCfg = field(default_factory=SyntheticCfg)
    comma2k19: Comma2k19Cfg = field(default_factory=Comma2k19Cfg)


@dataclass
class PreprocessingCfg:
    image_size: list[int] = field(default_factory=lambda: [32, 32])  # [H, W]
    normalize: bool = True
    mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class ModelCfg:
    name: str = "tiny_cnn_gru"
    encoder: dict[str, Any] = field(default_factory=dict)
    temporal: dict[str, Any] = field(default_factory=dict)
    checkpoint: str | None = None


@dataclass
class TrainingCfg:
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 0.0001
    weight_decay: float = 0.00001
    optimizer: str = "adamw"
    loss: str = "mse"  # mse | l1
    monitor: str = "val/mae_mph"
    early_stopping_patience: int = 5
    gradient_clip_val: float | None = None
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32-true"
    num_workers: int = 0


@dataclass
class ClipCfg:
    enabled: bool = True
    min_mph: float = 0.0
    max_mph: float | None = None


@dataclass
class SmoothingCfg:
    method: str = "none"  # none | ema | median
    ema_alpha: float = 0.5
    median_window: int = 5


@dataclass
class StationaryCfg:
    enabled: bool = True
    enter_threshold_mph: float = 2.0  # <= enter for N frames -> STOPPED
    exit_threshold_mph: float = 2.5  # > exit for M frames -> MOVING
    enter_persistence_frames: int = 3
    exit_persistence_frames: int = 3
    initial_state: str = "moving"


@dataclass
class PostprocessingCfg:
    clip: ClipCfg = field(default_factory=ClipCfg)
    smoothing: SmoothingCfg = field(default_factory=SmoothingCfg)
    stationary: StationaryCfg = field(default_factory=StationaryCfg)


@dataclass
class BucketCfg:
    name: str = ""
    min_mph: float = 0.0
    max_mph: float | None = None


@dataclass
class EvaluationCfg:
    stationary_label_threshold_mph: float = 2.0
    speed_buckets: list[BucketCfg] = field(default_factory=list)
    splits_to_evaluate: list[str] = field(default_factory=lambda: ["val", "test"])
    device: str = "auto"  # "auto" -> cuda if available, else cpu (see resolve_device)


@dataclass
class BenchmarkCfg:
    enabled: bool = True
    device: str = "auto"
    num_warmup: int = 10
    num_iters: int = 100


@dataclass
class MLflowCfg:
    tracking_uri: str | None = None  # null => local store under output/mlflow
    experiment_name: str = "speed-estimation"


@dataclass
class TrackingCfg:
    enabled: bool = True
    backend: str = "mlflow"
    mlflow: MLflowCfg = field(default_factory=MLflowCfg)


@dataclass
class LoggingCfg:
    level: str = "INFO"


@dataclass
class Config:
    experiment: ExperimentCfg = field(default_factory=ExperimentCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    data: DataCfg = field(default_factory=DataCfg)
    preprocessing: PreprocessingCfg = field(default_factory=PreprocessingCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    training: TrainingCfg = field(default_factory=TrainingCfg)
    postprocessing: PostprocessingCfg = field(default_factory=PostprocessingCfg)
    evaluation: EvaluationCfg = field(default_factory=EvaluationCfg)
    benchmark: BenchmarkCfg = field(default_factory=BenchmarkCfg)
    tracking: TrackingCfg = field(default_factory=TrackingCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)


# --------------------------------------------------------------------------- #
# Compose / validate
# --------------------------------------------------------------------------- #
def compose_config(overrides: Sequence[str] = (), config_dir: Path | None = None) -> DictConfig:
    """Compose the Hydra config. ``config_dir`` defaults to the repo ``configs/``."""
    cfg_dir = str(config_dir or DEFAULT_CONFIG_DIR)
    try:
        with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
            return compose(config_name="default", overrides=list(overrides))
    except Exception as exc:  # noqa: BLE001 - surface as project error
        raise ConfigError(f"Failed to compose config from {cfg_dir!r}: {exc}") from exc


def validate_config(cfg: DictConfig) -> Config:
    """Merge a composed DictConfig into the structured schema and validate.

    Raises :class:`ConfigError` on unknown keys, wrong types, or semantic
    violations (e.g. ``enter > exit`` thresholds).
    """
    try:
        clean = _strip_hydra(cfg)
        schema = OmegaConf.structured(Config)
        # OmegaConf.structured(Config) rejects unknown keys (ConfigKeyError) and
        # wrong types (ValidationError) on merge for dataclass-backed fields.
        merged = OmegaConf.merge(schema, clean)
        obj = OmegaConf.to_object(merged)
        if not isinstance(obj, Config):
            raise ConfigError("Merged config did not produce a Config object")
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid config: {exc}") from exc

    _validate_semantics(obj)
    return obj


def _strip_hydra(cfg: DictConfig) -> DictConfig:
    """Return a copy of ``cfg`` without the Hydra-internal ``hydra`` node."""
    clean = OmegaConf.create({})
    for key in cfg.keys():
        if key == "hydra":
            continue
        clean[key] = cfg[key]
    return clean


def _validate_semantics(obj: Config) -> None:
    """Cross-field validation that OmegaConf's per-field types cannot express."""
    s = obj.postprocessing.stationary
    if s.enter_threshold_mph > s.exit_threshold_mph:
        raise ConfigError(
            "enter_threshold_mph must be <= exit_threshold_mph "
            f"(got {s.enter_threshold_mph} > {s.exit_threshold_mph})"
        )
    if s.enter_persistence_frames < 1 or s.exit_persistence_frames < 1:
        raise ConfigError("stationary persistence frames must be >= 1")
    if obj.postprocessing.smoothing.method not in ("none", "ema", "median"):
        raise ConfigError(f"Unknown smoothing method: {obj.postprocessing.smoothing.method!r}")
    if obj.training.loss not in ("mse", "l1"):
        raise ConfigError(f"Unknown loss: {obj.training.loss!r}")
    if obj.data.on_duplicate_timestamp not in ("raise", "drop_later"):
        raise ConfigError(f"Unknown on_duplicate_timestamp: {obj.data.on_duplicate_timestamp!r}")
    if obj.data.sequence_length < 1:
        raise ConfigError("sequence_length must be >= 1")
    if obj.data.frame_step < 1:
        raise ConfigError("frame_step must be >= 1")
    if obj.data.window_stride < 1:
        raise ConfigError("window_stride must be >= 1")
    ratios = obj.data.split_ratios
    if set(ratios.keys()) != {"train", "val", "test"}:
        raise ConfigError("split_ratios must define exactly train, val and test")
    if any(v <= 0 for v in ratios.values()):
        raise ConfigError("split_ratios values must be positive")
