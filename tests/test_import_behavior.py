"""Tests guarding against side effects during module import."""

from __future__ import annotations

import importlib
import sys

import pytest

import flyrigloader


@pytest.mark.usefixtures("_configure_logging_silence")
def test_import_does_not_add_additional_loguru_sinks():
    """Reloading the package should not implicitly add Loguru sinks."""

    from loguru import logger

    logger.remove()
    baseline_id = logger.add(sys.stderr)
    baseline_handlers = set(logger._core.handlers.keys())

    importlib.reload(flyrigloader)

    assert set(logger._core.handlers.keys()) == baseline_handlers

    logger.remove(baseline_id)


@pytest.fixture
def _configure_logging_silence():
    """Ensure no persistent logger configuration bleeds between tests."""

    from loguru import logger

    existing = list(logger._core.handlers.keys())
    for handler_id in existing:
        logger.remove(handler_id)

    yield

    existing = list(logger._core.handlers.keys())
    for handler_id in existing:
        logger.remove(handler_id)
