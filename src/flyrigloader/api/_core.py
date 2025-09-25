"""Compatibility façade for legacy imports.

The original :mod:`flyrigloader.api._core` module grew into a monolithic file that
reimplemented logic owned by lower layers.  The implementation now lives in
specialised modules (``manifest``, ``loading``, ``transformation`` and
``registry``) and this module simply re-exports those entry points so that the
public API remains stable.
"""

from flyrigloader.discovery.files import (
    discover_experiment_manifest as _discover_experiment_manifest,
)
from flyrigloader.io.loaders import load_data_file as _load_data_file
from flyrigloader.io.transformers import transform_to_dataframe as _transform_to_dataframe

from .loading import (
    _create_test_dependency_provider,
    deprecated,
    get_dataset_parameters,
    get_experiment_parameters,
    load_data_file,
    load_dataset_files,
    load_experiment_files,
    process_experiment_data,
)
from .manifest import discover_experiment_manifest, validate_manifest
from .registry import get_loader_capabilities, get_registered_loaders
from .transformation import transform_to_dataframe

__all__ = [
    "_create_test_dependency_provider",
    "_discover_experiment_manifest",
    "_load_data_file",
    "_transform_to_dataframe",
    "deprecated",
    "discover_experiment_manifest",
    "get_dataset_parameters",
    "get_experiment_parameters",
    "get_loader_capabilities",
    "get_registered_loaders",
    "load_data_file",
    "load_dataset_files",
    "load_experiment_files",
    "process_experiment_data",
    "transform_to_dataframe",
    "validate_manifest",
]
