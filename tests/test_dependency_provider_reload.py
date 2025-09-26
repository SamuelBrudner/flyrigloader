"""Tests validating dependency provider behavior across module reloads."""

from __future__ import annotations

import importlib
from typing import Any


def _make_stub_provider(dependencies_module: Any) -> Any:
    """Create a minimal provider implementation for reload testing."""

    class _StubProvider(dependencies_module.AbstractDependencyProvider):
        def __init__(self) -> None:
            self._config = object()
            self._discovery = object()
            self._io = object()
            self._utils = object()

        @property
        def config(self) -> Any:  # pragma: no cover - behavior validated via identity
            return self._config

        @property
        def discovery(self) -> Any:  # pragma: no cover - behavior validated via identity
            return self._discovery

        @property
        def io(self) -> Any:  # pragma: no cover - behavior validated via identity
            return self._io

        @property
        def utils(self) -> Any:  # pragma: no cover - behavior validated via identity
            return self._utils

    return _StubProvider()


def test_custom_provider_remains_valid_after_module_reload() -> None:
    import flyrigloader.api.dependencies as dependencies

    provider = _make_stub_provider(dependencies)
    dependencies.set_dependency_provider(provider)

    try:
        reloaded = importlib.reload(dependencies)

        assert reloaded.get_dependency_provider() is provider

        # Validation should still accept the provider defined prior to reload.
        reloaded.set_dependency_provider(provider)
    finally:
        importlib.reload(dependencies).reset_dependency_provider()


def test_default_provider_class_refreshes_on_reload() -> None:
    import flyrigloader.api.dependencies as dependencies

    original_class = dependencies.DefaultDependencyProvider

    try:
        reloaded = importlib.reload(dependencies)
        assert reloaded.DefaultDependencyProvider is not original_class
    finally:
        importlib.reload(dependencies).reset_dependency_provider()


def test_reload_does_not_pollute_builtins_namespace() -> None:
    import builtins
    import flyrigloader.api.dependencies as dependencies

    provider_bases_attr = "_flyrigloader_dependency_provider_bases"
    default_classes_attr = "_flyrigloader_default_provider_classes"

    assert not hasattr(builtins, provider_bases_attr)
    assert not hasattr(builtins, default_classes_attr)

    try:
        reloaded = importlib.reload(dependencies)
        assert not hasattr(builtins, provider_bases_attr)
        assert not hasattr(builtins, default_classes_attr)
    finally:
        importlib.reload(dependencies).reset_dependency_provider()


def test_default_provider_identity_consistent_without_reload() -> None:
    import flyrigloader.api as api
    import flyrigloader.api.dependencies as dependencies

    assert api.DefaultDependencyProvider is dependencies.DefaultDependencyProvider
