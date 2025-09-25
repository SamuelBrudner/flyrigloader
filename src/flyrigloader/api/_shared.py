"""Shared helpers for :mod:`flyrigloader.api` submodules."""

from __future__ import annotations

import importlib
from typing import Any, Callable


def resolve_api_override(
    target_globals: dict[str, Any],
    name: str,
    fallback: Callable[..., Any] | Any,
) -> Callable[..., Any] | Any:
    """Return patched attribute from :mod:`flyrigloader.api` when available."""

    try:
        api_module = importlib.import_module("flyrigloader.api")
    except Exception:
        target_globals[name] = fallback
        return fallback

    override = getattr(api_module, name, fallback)
    target_globals[name] = override
    return override


__all__ = ["resolve_api_override"]

