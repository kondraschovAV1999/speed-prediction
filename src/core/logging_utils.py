"""Logging setup and helpers.

Structured logging: stdlib logging with a fixed format and ``key=value`` event
messages. Handlers attach to the **root** logger so Lightning/MLflow logs also
reach the log file.
"""

from __future__ import annotations

import logging
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

# Handlers installed by setup_logging; tracked so repeated calls are idempotent.
_OWN_HANDLERS: list[logging.Handler] = []


def setup_logging(log_file: Path | None, level: str = "INFO") -> None:
    """Configure root logging with a console handler and optional file handler.

    Idempotent: any handler previously installed by this function is removed
    before re-adding, so calling twice never duplicates output.
    """
    root = logging.getLogger()
    for handler in _OWN_HANDLERS:
        root.removeHandler(handler)
        handler.close()
    _OWN_HANDLERS.clear()

    root.setLevel(level.upper())
    formatter = logging.Formatter(_LOG_FORMAT)

    # Hydra (during @hydra.main init) calls logging.config.dictConfig, which
    # disables pre-existing loggers (disable_existing_loggers default). That
    # silently drops messages from any module-level `get_logger(__name__)` that
    # was created at import time, before main() ran. Re-enable them here.
    for lg in logging.Logger.manager.loggerDict.values():
        if isinstance(lg, logging.Logger) and lg.disabled:
            lg.disabled = False

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)
    _OWN_HANDLERS.append(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        _OWN_HANDLERS.append(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for ``name`` (propagates to the root handlers)."""
    return logging.getLogger(name)
