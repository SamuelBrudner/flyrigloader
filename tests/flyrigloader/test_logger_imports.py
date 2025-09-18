"""Tests ensuring modules use the package level logger."""

from __future__ import annotations

import importlib
from typing import Iterable

import flyrigloader


def _reload_modules(module_names: Iterable[str]) -> Iterable[object]:
    for module_name in module_names:
        module = importlib.import_module(module_name)
        importlib.reload(module)
        yield module


def test_modules_reference_package_logger():
    modules = _reload_modules(
        [
            "flyrigloader.api",
            "flyrigloader.utils",
            "flyrigloader.utils.dataframe",
            "flyrigloader.utils.paths",
            "flyrigloader.discovery",
            "flyrigloader.discovery.files",
            "flyrigloader.discovery.patterns",
            "flyrigloader.io.pickle",
        ]
    )

    for module in modules:
        module_logger = getattr(module, "logger", None)
        if module_logger is not None:
            assert module_logger is flyrigloader.logger
