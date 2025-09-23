import importlib
import sys
import types
from typing import Generic, TypeVar

import pytest


if "kedro" not in sys.modules:
    _In = TypeVar("_In")
    _Out = TypeVar("_Out")

    class _AbstractDataset(Generic[_In, _Out]):  # pragma: no cover - simple stub for tests
        def __init__(self, *args, **kwargs):
            pass

    kedro_module = types.ModuleType("kedro")
    io_module = types.ModuleType("kedro.io")

    io_module.AbstractDataset = _AbstractDataset
    kedro_module.io = io_module

    sys.modules["kedro"] = kedro_module
    sys.modules["kedro.io"] = io_module

from flyrigloader.discovery.files import FileManifest, FileInfo
from flyrigloader.kedro.catalog import create_flyrigloader_catalog_entry


@pytest.fixture
def sample_file_info(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("sample,data\n")
    return FileInfo(
        path=str(file_path),
        kedro_dataset_name="dataset",
        kedro_namespace="namespace",
    )


def test_catalog_entry_type_is_importable_from_manifest(sample_file_info):
    manifest = FileManifest(files=[sample_file_info])
    entries = manifest.generate_kedro_catalog_entries()
    dataset_type = entries[sample_file_info.get_kedro_dataset_path()]["type"]

    module_path, class_name = dataset_type.rsplit(".", 1)
    module = importlib.import_module(module_path)

    assert hasattr(module, class_name)


def test_create_catalog_entry_type_is_importable(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text("---\n")

    entry = create_flyrigloader_catalog_entry(
        dataset_name="dataset",
        config_path=config_path,
        experiment_name="experiment",
        validate_entry=False,
    )

    dataset_type = entry["config"]["type"]
    module_path, class_name = dataset_type.rsplit(".", 1)
    module = importlib.import_module(module_path)

    assert hasattr(module, class_name)
