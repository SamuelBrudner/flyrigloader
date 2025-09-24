"""Configuration utilities exposed through :mod:`flyrigloader.api`."""

from __future__ import annotations

from ._core import (
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

__all__ = [
    "CONFIG_SOURCE_ERROR_MESSAGE",
    "MISSING_DATA_DIR_ERROR",
    "_attach_metadata_bucket",
    "_coerce_config_for_version_validation",
    "_load_and_validate_config",
    "_raise_path_validation_error",
    "_resolve_base_directory",
    "_resolve_config_source",
    "_validate_config_parameters",
    "check_if_file_exists",
    "ensure_dir_exists",
    "get_common_base_dir",
    "get_default_column_config",
    "get_file_statistics",
    "get_path_absolute",
    "get_path_relative_to",
]
