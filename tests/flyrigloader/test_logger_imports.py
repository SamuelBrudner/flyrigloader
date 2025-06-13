"""Tests ensuring modules use the package level logger."""
import importlib
import flyrigloader

class DummyLogger:
    def debug(self, *a, **k):
        pass
    def info(self, *a, **k):
        pass
    def warning(self, *a, **k):
        pass
    def error(self, *a, **k):
        pass
    def configure(self, *a, **k):
        pass

def test_modules_reference_package_logger(monkeypatch):
    dummy = DummyLogger()
    monkeypatch.setattr(flyrigloader, "logger", dummy)

    modules = [
        flyrigloader.api,
        flyrigloader.utils,
        flyrigloader.utils.dataframe,
        flyrigloader.utils.paths,
        flyrigloader.discovery,
        flyrigloader.discovery.files,
        flyrigloader.discovery.patterns,
        flyrigloader.io.pickle,
        flyrigloader.io.column_models,
    ]

    for module in modules:
        importlib.reload(module)
        assert getattr(module, "logger") is dummy
