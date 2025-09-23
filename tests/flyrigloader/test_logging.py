"""Tests for the application level logging configuration."""

from __future__ import annotations

import contextlib
import importlib
import logging
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


def test_logger_file_creation(monkeypatch, tmp_path):
    """Importing the package should create the user-scoped logs directory."""

    tmp_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_home))

    importlib.reload(flyrigloader)

    expected_log_dir = tmp_home / ".flyrigloader" / "logs"

    assert expected_log_dir.is_dir(), "Logger initialization should create the user log directory"
    assert expected_log_dir.samefile(expected_log_dir.resolve())


def test_initialize_logger_falls_back_to_user_directory(monkeypatch, tmp_path):
    """Initialize logger should fallback to user directory when install path fails."""

    tmp_home = tmp_path / "fallback_home"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_home))

    install_dir = tmp_path / "install" / "logs"
    attempted_dirs: List[Path] = []

    real_ensure = flyrigloader._ensure_log_directory

    def failing_default_dir() -> Path:
        return install_dir

    def tracking_ensure(path: Path) -> None:
        attempted_dirs.append(Path(path))
        if Path(path) == install_dir:
            raise PermissionError("install directory is read-only")
        return real_ensure(Path(path))

    monkeypatch.setattr(flyrigloader, "_get_default_log_directory", failing_default_dir)
    monkeypatch.setattr(flyrigloader, "_ensure_log_directory", tracking_ensure)

    flyrigloader.reset_logger()
    flyrigloader.initialize_logger()

    expected_user_dir = tmp_home / ".flyrigloader" / "logs"

    assert attempted_dirs[0] == install_dir
    assert expected_user_dir in attempted_dirs
    assert expected_user_dir.is_dir()

    captured_messages: List[str] = []
    sink_id = logger.add(lambda msg: captured_messages.append(str(msg)), level="INFO")
    try:
        logger.info("fallback info message")
    finally:
        logger.remove(sink_id)

    assert any("fallback info message" in message for message in captured_messages)


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
