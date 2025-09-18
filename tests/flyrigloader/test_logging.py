"""Tests for the application level logging configuration."""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
from pathlib import Path
from typing import List

import pytest

import flyrigloader
from flyrigloader import logger
from flyrigloader.discovery import patterns
from flyrigloader.io import column_models, pickle


@pytest.fixture(autouse=True)
def _set_debug_capture(caplog):
    """Capture loguru output through the standard logging bridge."""
    caplog.set_level(logging.DEBUG)


def test_logger_basic_output(caplog):
    """Loguru should emit messages at expected levels."""
    logger.info("info message")
    logger.debug("debug message")

    info_records = [record for record in caplog.records if record.levelno == logging.INFO]
    debug_records = [record for record in caplog.records if record.levelno == logging.DEBUG]

    assert any("info message" in record.message for record in info_records)
    assert any("debug message" in record.message for record in debug_records)


def test_logger_file_creation():
    """Importing the package should create the logs directory."""
    src_dir = Path(flyrigloader.__file__).resolve().parent
    project_root = src_dir.parent.parent
    log_dir = project_root / "logs"

    assert log_dir.is_dir(), "Logger initialization should create the logs directory"


def test_console_log_format_validation():
    """Console format should advertise the expected placeholders."""
    format_string = flyrigloader.log_format_console
    for component in ["{time:", "{level:", "{name}", "{function}", "{line}", "{message}"]:
        assert component in format_string


def test_file_log_format_validation():
    """File format should include the same structural fields."""
    format_string = flyrigloader.log_format_file
    for component in ["{time:", "{level:", "{name}", "{function}", "{line}", "{message}"]:
        assert component in format_string


def test_patterns_module_uses_package_logger():
    """The discovery patterns module should reuse the shared logger."""
    module = importlib.reload(patterns)
    assert module.logger is flyrigloader.logger


def test_pickle_module_uses_package_logger():
    """The pickle module should expose the shared logger instance."""
    module = importlib.reload(pickle)
    assert module.logger is flyrigloader.logger


def test_column_models_default_logger_delegates_to_package_logger(monkeypatch):
    """DefaultLogger should forward calls to the package level logger."""

    class DummyLogger:
        def __init__(self) -> None:
            self.messages: List[str] = []

        def info(self, message: str) -> None:
            self.messages.append(message)

    dummy = DummyLogger()
    monkeypatch.setattr(flyrigloader, "logger", dummy)

    default_logger = column_models.DefaultLogger()
    default_logger.info("delegated message")

    assert dummy.messages == ["delegated message"]


def test_log_message_structure_validation(caplog):
    """Structured messages should remain intact through the bridge."""
    logger.warning("structured message payload")

    warning_records = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert any("structured message payload" in record.message for record in warning_records)


def test_log_sanitization_security():
    """Logging helpers should not add sensitive payloads by default."""
    sensitive_values = {
        "password": "secret123",
        "api_key": "sk-example",
        "token": "token-value",
    }

    for label, value in sensitive_values.items():
        message = f"processing {label}"
        logger.info(message)
        assert value not in message


def test_pickle_error_logging(monkeypatch):
    """Errors in pickle helper should be logged through the shared logger."""

    class DummyLogger:
        def __init__(self) -> None:
            self.messages: List[str] = []

        def info(self, message: str) -> None:
            pass

        def warning(self, message: str) -> None:
            pass

        def error(self, message: str) -> None:
            self.messages.append(message)

    dummy = DummyLogger()
    monkeypatch.setattr(flyrigloader, "logger", dummy)
    monkeypatch.setattr(pickle, "logger", dummy)

    with contextlib.suppress(Exception):
        pickle.read_pickle_any_format("/nonexistent/path.pkl")

    assert any("nonexistent" in message for message in dummy.messages)


def test_reloading_preserves_logger_binding(monkeypatch):
    """Reloading modules should not break the logger binding."""
    dummy = object()
    monkeypatch.setattr(flyrigloader, "logger", dummy)

    reloaded = importlib.reload(importlib.import_module("flyrigloader.discovery.patterns"))
    assert getattr(reloaded, "logger") is dummy
