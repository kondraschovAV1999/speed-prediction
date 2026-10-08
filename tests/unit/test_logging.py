"""Tests for src.logging_utils."""

from __future__ import annotations

import logging
from pathlib import Path

from src.logging_utils import get_logger, setup_logging


def _flush() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()


def test_logging_writes_to_file(tmp_path: Path):
    log_file = tmp_path / "sub" / "test.log"
    setup_logging(log_file, level="INFO")
    get_logger("test.logging").info("event=hello value=42")
    _flush()
    content = log_file.read_text(encoding="utf-8")
    assert "event=hello value=42" in content
    assert "test.logging" in content


def test_logging_called_twice_keeps_handler_count(tmp_path: Path):
    log_file = tmp_path / "test.log"
    setup_logging(log_file)
    before = len(logging.getLogger().handlers)
    setup_logging(log_file)
    after = len(logging.getLogger().handlers)
    assert before == after


def test_logging_called_twice_does_not_duplicate_lines(tmp_path: Path):
    log_file = tmp_path / "test.log"
    setup_logging(log_file)
    get_logger("t").info("event=first")
    setup_logging(log_file)
    get_logger("t").info("event=second")
    _flush()
    content = log_file.read_text(encoding="utf-8")
    assert content.count("event=first") == 1
    assert content.count("event=second") == 1


def test_logging_creates_parent_dir(tmp_path: Path):
    log_file = tmp_path / "a" / "b" / "c.log"
    setup_logging(log_file)
    assert log_file.parent.exists()
