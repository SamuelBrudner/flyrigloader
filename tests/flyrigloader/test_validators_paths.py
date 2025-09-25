"""Tests for secure path validation behavior."""

import pytest

from flyrigloader.config.validators import (
    ConfigValidationError,
    PathSecurityPolicy,
    path_existence_validator,
    pattern_validation,
)


def test_path_existence_validator_rejects_sensitive_root(caplog: pytest.LogCaptureFixture) -> None:
    """Sensitive system roots should be rejected loudly and identify the failing validator."""
    with caplog.at_level("ERROR"):
        with pytest.raises(ConfigValidationError) as exc_info:
            path_existence_validator("/etc/passwd")

    message = str(exc_info.value)
    assert "storage.path_existence_validator" in message
    assert "sensitive system root '/etc'" in message
    assert any(
        "validator=storage.path_existence_validator" in record
        for record in caplog.messages
    ), caplog.messages


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

    with pytest.raises(ConfigValidationError) as exc_info:
        path_existence_validator("/srv/secure/archive", security_policy=policy)

    assert "Access to sensitive system root '/srv/secure'" in str(exc_info.value)


def test_pattern_validation_raises_centralized_error(caplog: pytest.LogCaptureFixture) -> None:
    """Pattern validation failures should raise a centralized validation error with logging."""
    with caplog.at_level("ERROR"):
        with pytest.raises(ConfigValidationError) as exc_info:
            pattern_validation(42)

    assert "discovery.pattern_validation" in str(exc_info.value)
    assert any(
        "validator=discovery.pattern_validation" in message
        for message in caplog.messages
    ), caplog.messages
