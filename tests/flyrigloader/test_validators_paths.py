"""Tests for secure path validation behavior."""

import pytest

from flyrigloader.config.validators import (
    PathSecurityPolicy,
    path_existence_validator,
)


def test_path_existence_validator_rejects_sensitive_root() -> None:
    """Sensitive system roots should be rejected with a clear message."""
    with pytest.raises(PermissionError) as exc_info:
        path_existence_validator("/etc/passwd")

    assert "sensitive system root '/etc'" in str(exc_info.value)


def test_path_existence_validator_allows_usr_local_subdirectory() -> None:
    """System subdirectories like /usr/local should pass validation."""
    assert path_existence_validator("/usr/local/testdata") is True


def test_path_existence_validator_allows_var_directories_by_default() -> None:
    """Legitimate usage under /var should not be blocked by default validation."""
    assert path_existence_validator("/var/lib/flyrigloader") is True


def test_path_existence_validator_respects_allow_roots(caplog: pytest.LogCaptureFixture) -> None:
    """Allow roots should explicitly permit otherwise blocked directories."""
    policy = PathSecurityPolicy(allow_roots=["/custom/data"])

    with caplog.at_level("DEBUG"):
        assert path_existence_validator("/custom/data/session", security_policy=policy) is True

    assert any(
        "allowed by configured allow root '/custom/data'" in message
        for message in caplog.messages
    )


def test_path_existence_validator_respects_deny_roots() -> None:
    """Deny roots should block access even when the default validator would allow it."""
    policy = PathSecurityPolicy(deny_roots=["/srv/secure"])

    with pytest.raises(PermissionError) as exc_info:
        path_existence_validator("/srv/secure/archive", security_policy=policy)

    assert "Access to sensitive system root '/srv/secure'" in str(exc_info.value)
