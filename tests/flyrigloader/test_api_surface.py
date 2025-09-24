import importlib
import sys

import pytest


def _clear_api_modules():
    """Remove cached flyrigloader.api modules to force a fresh import."""
    for name in [mod for mod in sys.modules if mod == "flyrigloader.api" or mod.startswith("flyrigloader.api.")]:
        sys.modules.pop(name)


@pytest.fixture
def reload_api(monkeypatch):
    def _reload():
        _clear_api_modules()
        return importlib.import_module("flyrigloader.api")

    return _reload


def test_import_without_kedro_does_not_log_warning(monkeypatch, caplog, reload_api):
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "kedro":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    for name in [mod for mod in sys.modules if mod == "kedro" or mod.startswith("kedro.")]:
        sys.modules.pop(name)

    caplog.set_level("WARNING")
    caplog.clear()

    module = reload_api()

    assert not any(
        "Kedro integration" in record.getMessage() for record in caplog.records
    ), "Import should not emit Kedro warnings."

    assert hasattr(module, "create_kedro_dataset"), "API import should still expose Kedro helpers"


def test_check_kedro_available_raises_when_missing(monkeypatch, reload_api):
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "kedro":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    for name in [mod for mod in sys.modules if mod == "kedro" or mod.startswith("kedro.")]:
        sys.modules.pop(name)

    module = reload_api()

    with pytest.raises(module.FlyRigLoaderError):
        module.check_kedro_available()
