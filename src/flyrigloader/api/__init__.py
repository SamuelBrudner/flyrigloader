"""High-level API facade for flyrigloader.

This module intentionally re-exports functionality from dedicated feature
modules so downstream consumers can continue importing from
``flyrigloader.api`` while the implementation lives elsewhere.
"""

from __future__ import annotations

import sys

from flyrigloader.config.models import LegacyConfigAdapter, create_config
from flyrigloader.discovery.files import (
    discover_experiment_manifest as _discover_experiment_manifest,
    discover_files,
)
from flyrigloader.exceptions import FlyRigLoaderError
from flyrigloader.io.column_models import (
    ColumnConfig,
    ColumnConfigDict,
    ColumnDimension,
    get_config_from_source,
    get_default_config_path,
    load_column_config,
)
from flyrigloader.io.pickle import read_pickle_any_format
from flyrigloader.io.transformers import make_dataframe_from_config

from .columns import get_default_column_config
from .configuration import (
    coerce_config_for_version_validation as _coerce_config_for_version_validation,
    load_and_validate_config as _load_and_validate_config,
    resolve_config_source as _resolve_config_source,
)
from .dependencies import (
    ConfigProvider,
    DefaultDependencyProvider,
    DiscoveryProvider,
    IOProvider,
    UtilsProvider,
    get_dependency_provider,
    reset_dependency_provider,
    set_dependency_provider,
)
from flyrigloader._api_registry import register_facade_module as _register_facade_module

from .discovery import (
    discover_experiment_manifest,
    get_loader_capabilities,
    get_registered_loaders,
    validate_manifest,
)
from .helpers import _attach_metadata_bucket, _create_test_dependency_provider
from .kedro import (
    FlyRigLoaderDataSet,
    check_kedro_available,
    create_kedro_dataset,
)
from .loading import (
    load_data_file,
    load_dataset_files,
    load_experiment_files,
    process_experiment_data,
    transform_to_dataframe,
)
from .parameters import (
    get_dataset_parameters,
    get_experiment_parameters,
)
from .paths import (
    MISSING_DATA_DIR_ERROR,
    _resolve_base_directory,
    check_if_file_exists,
    ensure_dir_exists,
    get_common_base_dir,
    get_file_statistics,
    get_path_absolute,
    get_path_relative_to,
)

__all__ = [
    "discover_experiment_manifest",
    "load_data_file",
    "transform_to_dataframe",
    "create_config",
    "validate_manifest",
    "create_kedro_dataset",
    "check_kedro_available",
    "get_registered_loaders",
    "get_loader_capabilities",
    "load_experiment_files",
    "load_dataset_files",
    "process_experiment_data",
    "get_experiment_parameters",
    "get_dataset_parameters",
    "_resolve_base_directory",
    "get_dependency_provider",
    "set_dependency_provider",
    "reset_dependency_provider",
    "get_file_statistics",
    "ensure_dir_exists",
    "check_if_file_exists",
    "get_path_relative_to",
    "get_path_absolute",
    "get_common_base_dir",
    "get_default_column_config",
    "read_pickle_any_format",
    "make_dataframe_from_config",
    "get_config_from_source",
    "MISSING_DATA_DIR_ERROR",
    "FlyRigLoaderError",
]

__all__.sort()

_register_facade_module(sys.modules[__name__])
