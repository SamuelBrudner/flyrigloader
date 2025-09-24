"""Targeted tests for configurable path security policies."""

import pytest

from flyrigloader.config.models import PathSecurityConfig, ProjectConfig
from flyrigloader.config.validators import PathSecurityPolicy


def test_project_config_allows_sensitive_root_with_policy() -> None:
    """Explicitly allowing /var should permit those directories."""
    config = ProjectConfig(
        directories={"major_data_directory": "/var/lib/service"},
        path_security=PathSecurityConfig(allowed_roots=["/var"]),
    )

    assert config.directories == {"major_data_directory": "/var/lib/service"}


def test_project_config_rejects_invalid_path_security_entries() -> None:
    """Non-absolute allow lists should fail fast with actionable errors."""
    with pytest.raises(ValueError, match="Invalid path_security configuration"):
        ProjectConfig(
            directories={"major_data_directory": "/var/lib/service"},
            path_security=PathSecurityConfig(allowed_roots=["relative/path"]),
        )


def test_path_security_config_merges_with_base_policy() -> None:
    """Custom allow lists extend base policy roots without duplication."""
    base_policy = PathSecurityPolicy(allowed_roots=("/tmp",))
    config = PathSecurityConfig(allowed_roots=["/var"])

    merged_policy = config.to_policy(base_policy)

    assert set(merged_policy.allowed_roots) == {"/tmp", "/var"}
