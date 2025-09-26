"""Tests for dependency provider isolation mechanisms."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from flyrigloader.api import (
    AbstractDependencyProvider,
    DefaultDependencyProvider,
    get_dependency_provider,
    reset_dependency_provider,
    set_dependency_provider,
    use_dependency_provider,
)


class _MarkerDependencyProvider(DefaultDependencyProvider):
    """Custom provider used to verify isolation semantics in tests."""


class _ProtocolOnlyDependencyProvider(AbstractDependencyProvider):
    """Minimal provider implementing the abstract interface only."""

    def __init__(self) -> None:
        sentinel = SimpleNamespace()
        self._config = sentinel
        self._discovery = sentinel
        self._io = sentinel
        self._utils = sentinel

    @property
    def config(self) -> SimpleNamespace:
        return self._config

    @property
    def discovery(self) -> SimpleNamespace:
        return self._discovery

    @property
    def io(self) -> SimpleNamespace:
        return self._io

    @property
    def utils(self) -> SimpleNamespace:
        return self._utils


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


def test_reset_outside_override_wins_over_context_exit() -> None:
    """Explicit resets should take precedence over context restoration."""

    original = get_dependency_provider()
    override = _MarkerDependencyProvider()

    with use_dependency_provider(override):
        reset_dependency_provider()
        reset_provider = get_dependency_provider()
        assert isinstance(reset_provider, DefaultDependencyProvider)
        assert reset_provider is not override
        assert reset_provider is not original

    final_provider = get_dependency_provider()
    assert isinstance(final_provider, DefaultDependencyProvider)
    assert final_provider is not override
    assert final_provider is not original


def test_set_dependency_provider_accepts_protocol_implementations() -> None:
    """Any concrete :class:`AbstractDependencyProvider` should be accepted."""

    protocol_only_provider = _ProtocolOnlyDependencyProvider()

    set_dependency_provider(protocol_only_provider)
    assert get_dependency_provider() is protocol_only_provider


def test_use_dependency_provider_accepts_protocol_implementations() -> None:
    """Context manager should allow lightweight protocol implementations."""

    protocol_only_provider = _ProtocolOnlyDependencyProvider()
    original = get_dependency_provider()

    with use_dependency_provider(protocol_only_provider):
        assert get_dependency_provider() is protocol_only_provider

    assert get_dependency_provider() is original
