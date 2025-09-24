"""Tests ensuring the public API facade delegates to feature-specific modules."""

import importlib
import sys

import pytest


@pytest.mark.parametrize(
    ("module_name", "attributes"),
    [
        (
            "flyrigloader.api.discovery",
            (
                "discover_experiment_manifest",
                "validate_manifest",
                "get_registered_loaders",
                "get_loader_capabilities",
            ),
        ),
        (
            "flyrigloader.api.loading",
            (
                "load_data_file",
                "transform_to_dataframe",
                "load_experiment_files",
                "load_dataset_files",
                "process_experiment_data",
            ),
        ),
        (
            "flyrigloader.api.paths",
            (
                "_resolve_base_directory",
                "get_file_statistics",
                "ensure_dir_exists",
                "check_if_file_exists",
                "get_path_relative_to",
                "get_path_absolute",
                "get_common_base_dir",
            ),
        ),
        (
            "flyrigloader.api.columns",
            ("get_default_column_config",),
        ),
        (
            "flyrigloader.api.parameters",
            (
                "get_experiment_parameters",
                "get_dataset_parameters",
            ),
        ),
    ],
)
def test_api_submodule_alignment(module_name, attributes):
    """Each submodule should expose the same callables as the facade."""

    importlib.invalidate_caches()
    for name in list(sys.modules):
        if name == "flyrigloader.api" or name.startswith("flyrigloader.api."):
            sys.modules.pop(name)

    facade = importlib.import_module("flyrigloader.api")
    submodule = importlib.import_module(module_name)

    for attr in attributes:
        assert getattr(submodule, attr) is getattr(facade, attr)
