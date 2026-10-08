"""Experiment tracking behind an abstraction (WP8).

MLflow is optional: every MLflow call is wrapped so a failure logs a WARNING
and never raises.  The default local backend is a SQLite store (file stores are
phased out in MLflow 3.x — see docs/DECISIONS.md).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import quote

from src.config import Config
from src.logging_utils import get_logger
from src.schemas import RunPaths

logger = get_logger(__name__)

try:  # pragma: no cover - import availability
    import mlflow
    from mlflow.utils.validation import MAX_PARAM_VAL_LENGTH
except ImportError:  # pragma: no cover
    mlflow = None
    MAX_PARAM_VAL_LENGTH = 250


class ExperimentTracker(ABC):
    """Minimal tracking interface (implementations: MLflow, NoOp)."""

    @abstractmethod
    def start_run(self, run_name: str, tags: dict | None = None) -> None: ...

    @abstractmethod
    def log_params(self, params: dict) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: dict, step: int | None = None) -> None: ...

    @abstractmethod
    def log_artifact(self, path: Path | str) -> None: ...

    @abstractmethod
    def finish(self, status: str = "FINISHED") -> None: ...


class NoOpTracker(ExperimentTracker):
    """Accepts every call and does nothing."""

    def start_run(self, run_name: str, tags: dict | None = None) -> None:
        return None

    def log_params(self, params: dict) -> None:
        return None

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        return None

    def log_artifact(self, path: Path | str) -> None:
        return None

    def finish(self, status: str = "FINISHED") -> None:
        return None


class MLflowTracker(ExperimentTracker):
    """MLflow-backed tracker. All MLflow calls are failure-isolated."""

    def __init__(self, cfg: Config, paths: RunPaths):
        self._cfg = cfg
        self._paths = paths
        self._uri = cfg.tracking.mlflow.tracking_uri or _local_sqlite_uri(
            Path(cfg.paths.output_root)
        )

    # -- internal ---------------------------------------------------------- #
    def _safe(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - MLflow must never break the run
            logger.warning("event=mlflow_error error=%s", exc)
            return None

    # -- interface --------------------------------------------------------- #
    def start_run(self, run_name: str, tags: dict | None = None) -> None:
        def _start() -> None:
            if mlflow is None:
                raise RuntimeError("mlflow is not installed")
            mlflow.set_tracking_uri(self._uri)
            mlflow.set_experiment(self._cfg.tracking.mlflow.experiment_name)
            mlflow.start_run(run_name=run_name, tags={"local_run_name": run_name, **(tags or {})})

        self._safe(_start)

    def log_params(self, params: dict) -> None:
        self._safe(lambda: mlflow.log_params(_flatten_params(params)))

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        self._safe(lambda: mlflow.log_metrics(metrics, step=step))

    def log_artifact(self, path: Path | str) -> None:
        self._safe(lambda: mlflow.log_artifact(str(path)))

    def finish(self, status: str = "FINISHED") -> None:
        self._safe(lambda: mlflow.end_run(status=status))


def _local_sqlite_uri(output_root: Path) -> str:
    """Build a ``sqlite:///`` URI under ``<output_root>/mlflow``."""
    db_path = str((Path(output_root) / "mlflow" / "mlflow.db").resolve()).replace("\\", "/")
    return "sqlite:///" + quote(db_path, safe="/:")


def _flatten_params(params: dict, prefix: str = "") -> dict[str, str]:
    """Flatten nested dicts to dotted keys; truncate values to MLflow's limit."""
    flat: dict[str, str] = {}
    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_params(value, full_key))
        else:
            flat[full_key] = str(value)[:MAX_PARAM_VAL_LENGTH]
    return flat


def create_tracker(cfg: Config, paths: RunPaths) -> ExperimentTracker:
    """Return a NoOp tracker when disabled, else an MLflow tracker."""
    if not cfg.tracking.enabled:
        return NoOpTracker()
    if cfg.tracking.backend == "mlflow":
        return MLflowTracker(cfg, paths)
    logger.warning("event=unknown_tracking_backend backend=%s", cfg.tracking.backend)
    return NoOpTracker()
