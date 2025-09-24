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


def test_path_existence_validator_respects_allowed_roots_policy() -> None:
    """Legitimate usage under /var succeeds when explicitly allowed."""
    policy = PathSecurityPolicy(allowed_roots=("/var",))

    assert path_existence_validator("/var/lib/service", policy=policy) is True


def test_path_existence_validator_still_blocks_traversal_attempts() -> None:
    """Traversal patterns remain blocked even when root is allowed."""
    policy = PathSecurityPolicy(allowed_roots=("/var",))

    with pytest.raises(ValueError, match="Path traversal is not allowed"):
        path_existence_validator("/var/lib/../../etc/passwd", policy=policy)
