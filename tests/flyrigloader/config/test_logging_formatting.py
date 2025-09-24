import logging

import pytest

from flyrigloader.config.validators import (
    validate_config_version,
    validate_version_compatibility,
)


def test_validate_config_version_logs_detected_version(caplog):
    """Detected versions should be interpolated into log messages."""

    caplog.set_level(logging.INFO)

    config = {"schema_version": "1.2.3"}

    validate_config_version(config)

    messages = [record.message for record in caplog.records if record.levelno == logging.INFO]
    assert any("Detected configuration version: 1.2.3" in message for message in messages)


def test_validate_version_compatibility_logs_supported_version(caplog):
    """Compatibility results should include the evaluated version."""

    caplog.set_level(logging.INFO)

    validate_version_compatibility("1.2.3", "1.2.3")

    messages = [record.message for record in caplog.records if record.levelno == logging.INFO]
    assert any("Configuration version 1.2.3 is supported" in message for message in messages)
