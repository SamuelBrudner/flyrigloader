"""Tests for `_load_and_validate_config` that avoid silent dependency stubs."""

from __future__ import annotations

import contextlib
import importlib
import logging
import sys
import types
from pathlib import Path

import pytest


class _FailFastModule(types.ModuleType):
    """Minimal module replacement that fails loudly on unexpected access."""

    def __init__(self, name: str, allowed_attributes: dict[str, object]) -> None:
        super().__init__(name)
        for attr_name, value in allowed_attributes.items():
            setattr(self, attr_name, value)

    def __getattr__(self, item: str) -> object:  # pragma: no cover - defensive guard
        raise RuntimeError(
            f"Unexpected attribute access '{item}' on fail-fast stub module '{self.__name__}'."
        )


def _install_fail_fast_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a fail-fast YAML stub when PyYAML is unavailable."""

    try:
        importlib.import_module("yaml")
    except ModuleNotFoundError:
        def _unavailable_safe_load(*_: object, **__: object) -> None:
            raise ModuleNotFoundError(
                "PyYAML is required for configuration loading tests."
            )

        yaml_stub = _FailFastModule(
            "yaml",
            {
                "safe_load": _unavailable_safe_load,
                "YAMLError": RuntimeError,
            },
        )
        monkeypatch.setitem(sys.modules, "yaml", yaml_stub)


def _install_logging_backed_loguru(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a lightweight Loguru substitute if the real logger is missing."""

    try:
        importlib.import_module("loguru")
    except ModuleNotFoundError:
        test_logger = logging.getLogger("flyrigloader.tests.loguru_stub")

        class _LoguruLikeLogger:
            def add(self, handler, *args, **kwargs):  # pragma: no cover - logging helper
                handler_id = id(handler)
                handler.setLevel(kwargs.get("level", logging.INFO))
                test_logger.addHandler(handler)
                return handler_id

            def remove(self, handler_id):  # pragma: no cover - logging helper
                for handler in list(test_logger.handlers):
                    if id(handler) == handler_id:
                        test_logger.removeHandler(handler)

            def info(self, message, *args, **kwargs):
                test_logger.info(message, *args, **kwargs)

            def debug(self, message, *args, **kwargs):
                test_logger.debug(message, *args, **kwargs)

            def warning(self, message, *args, **kwargs):
                test_logger.warning(message, *args, **kwargs)

            def error(self, message, *args, **kwargs):
                test_logger.error(message, *args, **kwargs)

        monkeypatch.setitem(
            sys.modules,
            "loguru",
            types.SimpleNamespace(logger=_LoguruLikeLogger()),
        )


@pytest.fixture
def api_module(monkeypatch: pytest.MonkeyPatch):
    """Import ``flyrigloader.api`` with fail-fast fallbacks for optional deps."""

    src_root = Path(__file__).resolve().parents[2] / "src"
    monkeypatch.syspath_prepend(str(src_root))

    _install_fail_fast_yaml(monkeypatch)
    _install_logging_backed_loguru(monkeypatch)

    module = importlib.import_module("flyrigloader.api")
    yield module
    with contextlib.suppress(KeyError):
        del sys.modules["flyrigloader.api"]


class DummyConfigProvider:
    """Minimal configuration provider used to exercise dependency injection."""

    def load_config(self, path):
        return {"loaded": str(path)}

    def get_ignore_patterns(self, config, experiment=None):
        return []

    def get_mandatory_substrings(self, config, experiment=None):
        return []

    def get_dataset_info(self, config, dataset_name: str):
        return {}

    def get_experiment_info(self, config, experiment_name: str):
        return {}


def test_load_and_validate_config_with_custom_provider(tmp_path, api_module):
    deps = api_module._create_test_dependency_provider(
        config_provider=DummyConfigProvider()
    )
    cfg_file = tmp_path / "cfg.yaml"
    result = api_module._load_and_validate_config(cfg_file, None, "test_op", deps)
    assert result == {"loaded": str(cfg_file)}

