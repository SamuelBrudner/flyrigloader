"""Behavioral tests for the public ``flyrigloader.api`` Kedro surface."""

from __future__ import annotations

import importlib
import sys
from contextlib import suppress
from types import ModuleType
from typing import Callable

import pytest
from loguru import logger


def _reload_api(monkeypatch: pytest.MonkeyPatch, prepare: Callable[[], None] | None = None):
    """Reload ``flyrigloader.api`` with optional preparation logic."""

    if prepare is not None:
        prepare()

    if "flyrigloader.api" in sys.modules:
        api_module = importlib.reload(sys.modules["flyrigloader.api"])
        submodules = [
            name
            for name in list(sys.modules)
            if name.startswith("flyrigloader.api.") and sys.modules[name] is not None
        ]
        for name in submodules:
            importlib.reload(sys.modules[name])
        return api_module

    return importlib.import_module("flyrigloader.api")


def test_import_without_kedro_does_not_log_warning(monkeypatch: pytest.MonkeyPatch):
    """Importing the API without Kedro should not emit warnings."""

    original_find_spec = importlib.util.find_spec

    def find_spec(name: str, package: ModuleType | None = None):  # type: ignore[override]
        if name == "kedro":
            return None
        return original_find_spec(name, package)

    def prepare() -> None:
        monkeypatch.delitem(sys.modules, "kedro", raising=False)
        monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING")

    try:
        api = _reload_api(monkeypatch, prepare=prepare)
    finally:
        with suppress(ValueError):
            logger.remove(sink_id)
        monkeypatch.setattr(importlib.util, "find_spec", original_find_spec)

    assert not any("Kedro integration" in message for message in messages)
    assert api.check_kedro_available  # attribute exists


def test_check_kedro_available_raises_when_missing(monkeypatch: pytest.MonkeyPatch):
    """``check_kedro_available`` should raise a ``FlyRigLoaderError`` when Kedro is missing."""

    original_find_spec = importlib.util.find_spec

    def find_spec(name: str, package: ModuleType | None = None):  # type: ignore[override]
        if name == "kedro":
            return None
        return original_find_spec(name, package)

    def prepare() -> None:
        monkeypatch.delitem(sys.modules, "kedro", raising=False)
        monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    api = _reload_api(monkeypatch, prepare=prepare)

    with pytest.raises(api.FlyRigLoaderError):
        api.check_kedro_available()

    monkeypatch.setattr(importlib.util, "find_spec", original_find_spec)
