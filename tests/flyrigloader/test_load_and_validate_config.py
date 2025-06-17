import importlib
import sys
import types
from pathlib import Path

# Create minimal stubs for external dependencies to allow import
class DummyLogger:
    def add(self, *a, **k):
        pass
    def remove(self, *a, **k):
        pass
    def info(self, *a, **k):
        pass
    def debug(self, *a, **k):
        pass
    def warning(self, *a, **k):
        pass
    def error(self, *a, **k):
        pass

sys.modules['loguru'] = types.SimpleNamespace(logger=DummyLogger())

yaml_module = types.ModuleType('yaml')
yaml_module.safe_load = lambda s: {}
sys.modules['yaml'] = yaml_module

column_models = types.ModuleType('flyrigloader.io.column_models')
column_models.ColumnConfig = None
column_models.ColumnConfigDict = dict
column_models.ColumnDimension = None
column_models.get_default_config_path = lambda: None
column_models.load_column_config = lambda *a, **k: None

io_pkg = types.ModuleType('flyrigloader.io')
io_pkg.column_models = column_models
sys.modules['flyrigloader.io'] = io_pkg
sys.modules['flyrigloader.io.column_models'] = column_models

files_module = types.ModuleType('flyrigloader.discovery.files')
files_module.discover_files = lambda *a, **k: []
sys.modules['flyrigloader.discovery.files'] = files_module

# Import the module under test after stubbing dependencies
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
api = importlib.import_module('flyrigloader.api')

class DummyConfigProvider:
    def load_config(self, path):
        return {'loaded': str(path)}
    def get_ignore_patterns(self, config, experiment=None):
        return []
    def get_mandatory_substrings(self, config, experiment=None):
        return []
    def get_dataset_info(self, config, dataset_name: str):
        return {}
    def get_experiment_info(self, config, experiment_name: str):
        return {}


def test_load_and_validate_config_with_custom_provider(tmp_path):
    deps = api._create_test_dependency_provider(config_provider=DummyConfigProvider())
    cfg_file = tmp_path / 'cfg.yaml'
    result = api._load_and_validate_config(cfg_file, None, 'test_op', deps)
    assert result == {'loaded': str(cfg_file)}
