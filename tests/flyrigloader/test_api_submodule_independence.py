"""Tests ensuring API submodules are decoupled from :mod:`flyrigloader.api._core`."""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Iterable

import pytest


class _ExplosiveCore(types.ModuleType):
    """Module stub that fails loudly when any attribute access occurs."""

    def __getattr__(self, item: str) -> object:  # pragma: no cover - defensive guard
        raise AssertionError(f"Unexpected access to api._core attribute '{item}' during import")


def _clear_api_modules(exceptions: Iterable[str] = ()) -> None:
    """Remove cached flyrigloader.api modules except those explicitly exempted."""

    preserved = tuple(exceptions)
    for name in list(sys.modules):
        if not name.startswith("flyrigloader.api"):
            continue
        if name in preserved:
            continue
        sys.modules.pop(name)


@pytest.mark.parametrize(
    ("module_name", "expected_symbols"),
    [
        ("config", ["_load_and_validate_config", "ensure_dir_exists", "get_path_absolute"]),
        ("manifest", ["discover_experiment_manifest", "validate_manifest"]),
        ("registry", ["get_loader_capabilities", "get_registered_loaders"]),
    ],
)
def test_api_submodule_import_does_not_require_core(monkeypatch: pytest.MonkeyPatch, module_name: str, expected_symbols: list[str]) -> None:
    """Verify targeted submodules import successfully without depending on _core."""

    _clear_api_modules()

    monkeypatch.setitem(sys.modules, "flyrigloader.api._core", _ExplosiveCore("flyrigloader.api._core"))

    submodule = importlib.import_module(f"flyrigloader.api.{module_name}")

    for symbol in expected_symbols:
        assert hasattr(submodule, symbol), f"Expected {symbol} on flyrigloader.api.{module_name}"

