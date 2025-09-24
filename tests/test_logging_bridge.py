"""Tests for logging bridge ensuring Loguru captures standard logging output."""

import io
from pathlib import Path

import logging

import pytest

from flyrigloader import configure_test_logger, logger, reset_logger
from flyrigloader.io.loaders import load_data_file
from flyrigloader.registries import LoaderRegistry


class _TemporaryLoader:
    """Simple loader implementation for testing purposes."""

    priority = 0

    def load(self, path: Path):
        return {"path": str(path)}

    def supports_extension(self, extension: str) -> bool:  # pragma: no cover - trivial
        return extension.endswith(".dummy")


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Ensure Loguru logger starts clean for each test."""

    reset_logger()
    configure_test_logger(console_level="DEBUG", disable_console_colors=True)
    yield
    reset_logger()


def test_standard_logging_is_routed_to_loguru():
    """Standard logging should be captured by Loguru sinks via the intercept handler."""

    log_stream = io.StringIO()
    sink_id = logger.add(log_stream, format="{level}:{message}")

    std_logger = logging.getLogger("flyrigloader.test")
    std_logger.setLevel(logging.DEBUG)
    std_logger.debug("standard logging to capture")

    logger.remove(sink_id)

    log_output = log_stream.getvalue()
    assert "standard logging to capture" in log_output


def test_load_data_file_logs_are_captured_by_loguru(tmp_path):
    """Calling load_data_file should log through Loguru sinks."""

    registry = LoaderRegistry()
    extension = ".dummy"

    if registry.get_loader_for_extension(extension):
        pytest.skip("Dummy extension already registered; cannot run test in isolation.")

    registry.register_loader(extension, _TemporaryLoader)

    data_file = tmp_path / f"sample{extension}"
    data_file.write_text("payload")

    log_stream = io.StringIO()
    sink_id = logger.add(log_stream, format="{message}")

    try:
        load_data_file(data_file)
    finally:
        logger.remove(sink_id)
        registry.unregister_loader(extension)

    log_output = log_stream.getvalue()
    assert f"Loading data from file: {data_file}" in log_output
