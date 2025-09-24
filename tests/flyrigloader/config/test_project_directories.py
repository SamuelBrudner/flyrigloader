"""Targeted tests for ProjectConfig directory validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from flyrigloader.config.models import ProjectConfig


def test_project_config_directory_validation_rejects_missing_paths(tmp_path, monkeypatch):
    """Directory validation should fail fast when paths do not exist."""

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    nonexistent = tmp_path / "missing"

    with pytest.raises(ValidationError) as exc_info:
        ProjectConfig(directories={"major_data_directory": str(nonexistent)})

    assert "Path not found" in str(exc_info.value)
