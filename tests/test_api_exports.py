"""Tests for the public API surface of :mod:`flyrigloader.api`."""

from __future__ import annotations

import importlib


CORE_EXPORTS = {
    "_create_test_dependency_provider",
    "_discover_experiment_manifest",
    "_load_data_file",
    "_transform_to_dataframe",
    "deprecated",
    "get_dataset_parameters",
    "get_experiment_parameters",
    "load_data_file",
    "load_dataset_files",
    "load_experiment_files",
    "process_experiment_data",
    "transform_to_dataframe",
}


def test_core_exports_are_eagerly_available():
    """Eagerly exported names should exist immediately after import."""

    module = importlib.import_module("flyrigloader.api")

    module_globals = set(module.__dict__)

    missing = sorted(name for name in CORE_EXPORTS if name not in module_globals)

    assert missing == []
