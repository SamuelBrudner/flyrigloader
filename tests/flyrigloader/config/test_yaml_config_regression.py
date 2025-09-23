"""Regression tests for YAML configuration error handling."""
import json
import sys
import types

import pytest
from pydantic import ValidationError

# Provide a lightweight YAML stub if PyYAML is unavailable.
if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")

    class YAMLError(Exception):
        """Basic YAML error placeholder used for testing."""

    def _ensure_text(stream):
        if hasattr(stream, "read"):
            return stream.read()
        return stream

    def safe_load(stream):
        text = _ensure_text(stream)
        if not text:
            return {}
        return json.loads(text)

    def dump(data, stream=None, **kwargs):
        text = json.dumps(data)
        if stream is None:
            return text
        stream.write(text)
        return None

    yaml_stub.safe_load = safe_load
    yaml_stub.dump = dump
    yaml_stub.safe_dump = dump
    yaml_stub.YAMLError = YAMLError
    sys.modules["yaml"] = yaml_stub

from flyrigloader.config.yaml_config import load_config
from flyrigloader.exceptions import ConfigError


def test_load_config_invalid_config_raises_config_error(monkeypatch):
    """Invalid dictionary input should raise a ConfigError with detailed context."""
    invalid_config = {
        "schema_version": "1.0.0",
        "project": {
            "directories": {"major_data_directory": "/data"}
        },
        "datasets": {
            "invalid_dataset": {
                "rig": "rig-1",
                "dates_vials": [1, 2, 3],
            }
        },
    }

    validation_error = ValidationError.from_exception_data(
        "LegacyConfigAdapter",
        [
            {
                "type": "dict_type",
                "loc": ("datasets", "invalid_dataset", "dates_vials"),
                "msg": "Input should be a valid dictionary",
                "input": [1, 2, 3],
            }
        ],
    )

    class RaisingAdapter:
        def __init__(self, *args, **kwargs):
            raise validation_error

    monkeypatch.setattr(
        "flyrigloader.config.yaml_config.LegacyConfigAdapter",
        RaisingAdapter,
    )

    with pytest.raises(ConfigError) as excinfo:
        load_config(invalid_config)

    message = str(excinfo.value)
    assert "Configuration validation failed" in message
    assert "datasets -> invalid_dataset -> dates_vials" in message
    assert isinstance(excinfo.value.__cause__, ValidationError)
