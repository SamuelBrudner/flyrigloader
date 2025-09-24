"""Registry for tracking API facade modules across reloads.

This module intentionally lives outside the ``flyrigloader.api`` package so that
state survives the test suite's module reload cycles. Tests occasionally hold
references to an older ``flyrigloader.api`` module instance and patch private
hooks on that object. When the facade package is reloaded, those patches would
otherwise be lost. The registry keeps weak references to every facade module
that has been imported so the implementation can honor whichever instance was
patched.
"""

from __future__ import annotations

import threading
import weakref
from types import ModuleType
from typing import Tuple

__all__ = ["register_facade_module", "get_facade_modules"]

_lock = threading.RLock()
_modules: "weakref.WeakSet[ModuleType]" = weakref.WeakSet()


def register_facade_module(module: ModuleType) -> None:
    """Record a facade module so patches on it can be discovered later."""
    if not isinstance(module, ModuleType):
        raise TypeError(
            "register_facade_module expects a module; "
            f"got {type(module)!r}"
        )

    with _lock:
        _modules.add(module)


def get_facade_modules() -> Tuple[ModuleType, ...]:
    """Return a snapshot of the currently known facade modules."""
    with _lock:
        # ``WeakSet`` iteration happens under the lock so we can provide a
        # consistent snapshot to callers without exposing its internal state.
        return tuple(_modules)
