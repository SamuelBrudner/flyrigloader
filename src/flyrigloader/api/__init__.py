"""Public entry points for :mod:`flyrigloader.api`."""

from __future__ import annotations

from flyrigloader.exceptions import FlyRigLoaderError

from .config import (
    CONFIG_SOURCE_ERROR_MESSAGE,
    MISSING_DATA_DIR_ERROR,
    _attach_metadata_bucket,
    _coerce_config_for_version_validation,
    _load_and_validate_config,
    _raise_path_validation_error,
    _resolve_base_directory,
    _resolve_config_source,
    _validate_config_parameters,
    check_if_file_exists,
    ensure_dir_exists,
    get_common_base_dir,
    get_default_column_config,
    get_file_statistics,
    get_path_absolute,
    get_path_relative_to,
)
from .dependencies import (
    DefaultDependencyProvider,
    get_dependency_provider,
    reset_dependency_provider,
    set_dependency_provider,
)
from .kedro import FlyRigLoaderDataSet, check_kedro_available, create_kedro_dataset
from .manifest import discover_experiment_manifest, validate_manifest
from .registry import get_loader_capabilities, get_registered_loaders
from ._core import (
    _create_test_dependency_provider,
    _discover_experiment_manifest,
    _load_data_file,
    _transform_to_dataframe,
    deprecated,
    get_dataset_parameters,
    get_experiment_parameters,
    load_data_file,
    load_dataset_files,
    load_experiment_files,
    process_experiment_data,
    transform_to_dataframe,
)
from flyrigloader.config.models import create_config
from flyrigloader.io.column_models import get_config_from_source
from flyrigloader.io.pickle import read_pickle_any_format
from flyrigloader.io.transformers import make_dataframe_from_config

__all__ = [
    "CONFIG_SOURCE_ERROR_MESSAGE",
    "MISSING_DATA_DIR_ERROR",
    "DefaultDependencyProvider",
    "FlyRigLoaderDataSet",
    "FlyRigLoaderError",
    "_attach_metadata_bucket",
    "_coerce_config_for_version_validation",
    "_create_test_dependency_provider",
    "_discover_experiment_manifest",
    "_load_and_validate_config",
    "_load_data_file",
    "_raise_path_validation_error",
    "_resolve_base_directory",
    "_resolve_config_source",
    "_validate_config_parameters",
    "_transform_to_dataframe",
    "check_if_file_exists",
    "check_kedro_available",
    "create_config",
    "create_kedro_dataset",
    "deprecated",
    "discover_experiment_manifest",
    "ensure_dir_exists",
    "get_common_base_dir",
    "get_config_from_source",
    "get_dataset_parameters",
    "get_default_column_config",
    "get_dependency_provider",
    "get_experiment_parameters",
    "get_file_statistics",
    "get_loader_capabilities",
    "get_path_absolute",
    "get_path_relative_to",
    "get_registered_loaders",
    "load_data_file",
    "load_dataset_files",
    "load_experiment_files",
    "make_dataframe_from_config",
    "process_experiment_data",
    "read_pickle_any_format",
    "reset_dependency_provider",
    "set_dependency_provider",
    "transform_to_dataframe",
    "validate_manifest",
]
