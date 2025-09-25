"""Tests for dependency provider isolation mechanisms."""

from __future__ import annotations

import pytest

from flyrigloader.api import (
    DefaultDependencyProvider,
    get_dependency_provider,
    set_dependency_provider,
    use_dependency_provider,
)


class _MarkerDependencyProvider(DefaultDependencyProvider):
    """Custom provider used to verify isolation semantics in tests."""


@pytest.mark.usefixtures("dependency_provider_state_guard")
def test_use_dependency_provider_scopes_override() -> None:
    """Overrides within the context manager should not leak globally."""

    original = get_dependency_provider()
    override = _MarkerDependencyProvider()

    with use_dependency_provider(override):
        assert get_dependency_provider() is override

    assert get_dependency_provider() is original


def test_autouse_reset_fixture_prevents_leakage() -> None:
    """An override in one test should not leak into another test."""

    set_dependency_provider(_MarkerDependencyProvider())
    assert isinstance(get_dependency_provider(), _MarkerDependencyProvider)


def test_autouse_reset_fixture_runs_before_each_test() -> None:
    """Subsequent tests should see the default dependency provider."""

    provider = get_dependency_provider()
    assert isinstance(provider, DefaultDependencyProvider)
    assert not isinstance(provider, _MarkerDependencyProvider)
