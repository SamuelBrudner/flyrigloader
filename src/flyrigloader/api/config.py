"""Configuration utilities exposed through :mod:`flyrigloader.api`."""

from __future__ import annotations

import copy
import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from semantic_version import Version

from flyrigloader import logger
from flyrigloader.config.models import LegacyConfigAdapter
from flyrigloader.config.validators import validate_config_version
from flyrigloader.config.versioning import CURRENT_SCHEMA_VERSION
from flyrigloader.exceptions import FlyRigLoaderError
from flyrigloader.io.column_models import ColumnConfigDict

from .dependencies import DefaultDependencyProvider, get_dependency_provider


MISSING_DATA_DIR_ERROR = (
    "No data directory specified. Either provide base_directory parameter "
    "or ensure 'major_data_directory' is set in config."
)

CONFIG_SOURCE_ERROR_MESSAGE = "Exactly one of 'config_path' or 'config' must be provided"


def _validate_config_parameters(
    config_path: Optional[Union[str, Path]],
    config: Optional[Dict[str, Any]],
    operation_name: str,
) -> None:
    """Ensure callers provide exactly one configuration source."""

    logger.debug(f"Validating config parameters for {operation_name}")

    both_missing = config_path is None and config is None
    both_provided = config_path is not None and config is not None

    if both_missing or both_provided:
        if both_missing:
            logger.error(
                f"Configuration source validation failed for {operation_name}: neither config nor config_path was provided"
            )
        else:
            logger.error(
                f"Configuration source validation failed for {operation_name}: both config and config_path were provided"
            )
        raise ValueError(CONFIG_SOURCE_ERROR_MESSAGE)

    logger.debug(f"Config parameter validation successful for {operation_name}")


def _load_and_validate_config(
    config_path: Optional[Union[str, Path]],
    config: Optional[Union[Dict[str, Any], Any]],
    operation_name: str,
    deps: Optional[DefaultDependencyProvider] = None,
) -> Dict[str, Any]:
    """Load and validate configuration with enhanced error handling."""

    if deps is None:
        deps = get_dependency_provider()

    logger.debug(f"Loading and validating config for {operation_name}")

    if config_path is not None:
        try:
            logger.info(f"Loading configuration from file: {config_path}")
            config_dict = deps.config.load_config(config_path)
            logger.debug(f"Successfully loaded config from {config_path}")
        except FileNotFoundError as exc:
            error_msg = (
                f"Configuration file not found for {operation_name}: {config_path}. "
                "Please ensure the file exists and the path is correct."
            )
            logger.error(error_msg)
            raise FlyRigLoaderError(error_msg) from exc
        except Exception as exc:
            error_msg = (
                f"Failed to load configuration for {operation_name} from {config_path}: {exc}. "
                "Please check the file format and syntax."
            )
            logger.error(error_msg)
            raise FlyRigLoaderError(error_msg) from exc
    else:
        logger.debug(f"Using pre-loaded configuration for {operation_name}")
        config_dict = copy.deepcopy(config)

    if hasattr(config_dict, "keys") and hasattr(config_dict, "__getitem__") and hasattr(config_dict, "get"):
        logger.debug(f"Configuration is dict-like for {operation_name}")

        if not isinstance(config_dict, dict):
            try:
                dict_config: Dict[str, Any] = {}
                for key in config_dict.keys():
                    dict_config[key] = config_dict[key]
                config_dict = dict_config
                logger.debug(f"Converted LegacyConfigAdapter to dictionary for {operation_name}")
            except Exception as exc:
                error_msg = (
                    f"Failed to convert configuration to dictionary for {operation_name}: {exc}. "
                    "Configuration must be convertible to dictionary structure."
                )
                logger.error(error_msg)
                raise FlyRigLoaderError(error_msg) from exc
    elif not isinstance(config_dict, dict):
        error_msg = (
            f"Invalid configuration format for {operation_name}: "
            f"Expected dictionary or dict-like object, got {type(config_dict).__name__}. "
            "Configuration must be a valid dictionary structure or LegacyConfigAdapter."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)

    logger.debug(f"Configuration validation successful for {operation_name}")
    return config_dict


def _resolve_config_source(
    config: Optional[Union[Dict[str, Any], Any]],
    config_path: Optional[Union[str, Path]],
    operation_name: str,
    deps: Optional[DefaultDependencyProvider],
) -> Dict[str, Any]:
    """Return the concrete configuration dictionary for the requested operation."""

    _validate_config_parameters(config_path, config, operation_name)

    if config is not None:
        logger.debug(f"Using provided configuration object for {operation_name}")
        return config  # type: ignore[return-value]

    assert config_path is not None
    logger.debug(f"Loading configuration from file source {config_path} for {operation_name}")
    return _load_and_validate_config(config_path, None, operation_name, deps)


def _coerce_config_for_version_validation(config_obj: Any) -> Union[Dict[str, Any], str]:
    """Normalize configuration objects before schema version validation."""

    if isinstance(config_obj, (dict, str)):
        return config_obj

    if isinstance(config_obj, MutableMapping):
        logger.debug(
            f"Converted MutableMapping configuration of type {type(config_obj).__name__} for version validation"
        )
        return dict(config_obj)

    model_dump = getattr(config_obj, "model_dump", None)
    if callable(model_dump):
        try:
            dumped_config = model_dump()
            logger.debug(
                f"Converted Pydantic model {type(config_obj).__name__} to dictionary via model_dump for version validation"
            )
            return dumped_config
        except Exception as exc:
            logger.debug(
                f"Failed to convert configuration {type(config_obj).__name__} using model_dump(): {exc}"
            )

    to_dict = getattr(config_obj, "to_dict", None)
    if callable(to_dict):
        try:
            dict_config = to_dict()
            logger.debug(
                f"Converted configuration {type(config_obj).__name__} using to_dict() for version validation"
            )
            return dict_config
        except Exception as exc:
            logger.debug(
                f"Failed to convert configuration {type(config_obj).__name__} using to_dict(): {exc}"
            )

    raise TypeError(
        f"Configuration data must be dict-like or convertible, got {type(config_obj)}"
    )


def _attach_metadata_bucket(
    discovery_result: Union[List[str], Dict[str, Any]],
) -> Union[List[str], Dict[str, Dict[str, Any]]]:
    """Ensure discovery results provide a nested ``metadata`` dictionary."""

    if not isinstance(discovery_result, dict):
        return discovery_result

    normalised: Dict[str, Dict[str, Any]] = {}

    for path, payload in discovery_result.items():
        path_str = str(path)

        if not isinstance(payload, dict):
            normalised[path_str] = {"path": path_str, "metadata": {}}
            continue

        flattened = dict(payload)
        flattened.setdefault("path", path_str)

        existing_metadata = flattened.get("metadata")
        metadata_bucket: Dict[str, Any] = {}
        if isinstance(existing_metadata, dict):
            metadata_bucket.update(existing_metadata)

        flattened["metadata"] = metadata_bucket
        normalised[path_str] = flattened  # type: ignore[assignment]

    return normalised


def _resolve_base_directory(
    config: Union[Dict[str, Any], LegacyConfigAdapter],
    base_directory: Optional[Union[str, Path]],
    operation_name: str,
) -> Union[str, Path]:
    """Resolve base directory following documented precedence order."""

    logger.debug(f"Starting base directory resolution for {operation_name}")
    logger.debug(
        "Resolution precedence: 1) explicit parameter, 2) config major_data_directory, 3) FLYRIGLOADER_DATA_DIR env var"
    )

    resolved_directory: Optional[Union[str, Path]] = None
    resolution_source: Optional[str] = None

    if base_directory is not None:
        resolved_directory = base_directory
        resolution_source = "explicit function parameter"
        logger.info(f"Using explicit base_directory parameter: {base_directory}")
        logger.debug(f"Resolution source: {resolution_source} (precedence 1)")

    if resolved_directory is None:
        logger.debug("No explicit base_directory provided, checking config for major_data_directory")

        config_directory = None
        if hasattr(config, "get"):
            config_directory = (
                config.get("project", {})  # type: ignore[arg-type]
                .get("directories", {})
                .get("major_data_directory")
            )
        else:
            logger.warning(f"Unexpected config type for {operation_name}: {type(config)}")

        if config_directory:
            resolved_directory = config_directory
            resolution_source = "configuration major_data_directory"
            logger.info(f"Using major_data_directory from config: {config_directory}")
            logger.debug(f"Resolution source: {resolution_source} (precedence 2)")
        else:
            logger.debug("No major_data_directory found in config")

    if resolved_directory is None:
        logger.debug("No config major_data_directory found, checking FLYRIGLOADER_DATA_DIR environment variable")
        env_directory = os.environ.get("FLYRIGLOADER_DATA_DIR")

        if env_directory:
            resolved_directory = env_directory
            resolution_source = "FLYRIGLOADER_DATA_DIR environment variable"
            logger.info(f"Using FLYRIGLOADER_DATA_DIR environment variable: {env_directory}")
            logger.debug(f"Resolution source: {resolution_source} (precedence 3)")
        else:
            logger.debug("No FLYRIGLOADER_DATA_DIR environment variable found")

    if not resolved_directory:
        error_msg = (
            f"No data directory specified for {operation_name}. "
            "Tried all resolution methods in precedence order:\n"
            "1. Explicit 'base_directory' parameter (not provided)\n"
            "2. Configuration 'project.directories.major_data_directory' (not found)\n"
            "3. Environment variable 'FLYRIGLOADER_DATA_DIR' (not set)\n\n"
            "Please provide a data directory using one of these methods:\n"
            "- Pass base_directory parameter to the function\n"
            "- Set 'major_data_directory' in your config:\n"
            "  project:\n    directories:\n      major_data_directory: /path/to/data\n"
            "- Set environment variable: export FLYRIGLOADER_DATA_DIR=/path/to/data"
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)

    resolved_path = Path(resolved_directory)

    if resolved_path.exists():
        logger.debug(f"Resolved directory exists: {resolved_path}")
    else:
        logger.warning(f"Resolved directory does not exist: {resolved_path}")
        logger.warning(f"This may cause file discovery failures in {operation_name}")

    logger.info(f"✓ Base directory resolution complete for {operation_name}")
    logger.info(f"  Resolved path: {resolved_directory}")
    logger.info(f"  Resolution source: {resolution_source}")
    logger.debug(f"  Absolute path: {resolved_path.resolve()}")

    return resolved_directory


def get_default_column_config(
    _deps: Optional[DefaultDependencyProvider] = None,
) -> ColumnConfigDict:
    """Get the default column configuration with enhanced testability."""

    operation_name = "get_default_column_config"

    if _deps is None:
        _deps = get_dependency_provider()

    logger.info("Loading default column configuration")

    try:
        result = _deps.io.get_config_from_source(None)

        if hasattr(result, "columns"):
            column_count = len(result.columns) if hasattr(result.columns, "__len__") else 0
            logger.info(f"Successfully loaded default configuration with {column_count} columns")
            logger.debug(
                f"Column names: {list(result.columns.keys()) if hasattr(result.columns, 'keys') else 'N/A'}"
            )
        else:
            logger.info(f"Successfully loaded default configuration, type: {type(result).__name__}")

        return result
    except Exception as exc:
        error_msg = (
            f"Failed to load default column configuration for {operation_name}: {exc}. "
            "Please ensure the default configuration file exists and is valid."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from exc


def get_file_statistics(
    path: Union[str, Path],
    _deps: Optional[DefaultDependencyProvider] = None,
) -> Dict[str, Any]:
    """Get comprehensive statistics for a file with enhanced testability."""

    operation_name = "get_file_statistics"

    if _deps is None:
        _deps = get_dependency_provider()

    logger.debug(f"Getting file statistics for: {path}")

    if not path:
        error_msg = (
            f"Invalid path for {operation_name}: '{path}'. "
            "Path must be a non-empty string or Path object."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)

    try:
        result = _deps.utils.get_file_stats(path)
        logger.debug(f"File statistics result: {result}")
        return result
    except FileNotFoundError as exc:
        error_msg = (
            f"File not found for {operation_name}: {path}. "
            "Please ensure the file exists and the path is correct."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from exc
    except Exception as exc:
        error_msg = (
            f"Failed to get file statistics for {operation_name}: {exc}. "
            "Please check the path format."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from exc


def ensure_dir_exists(
    path: Union[str, Path],
    _deps: Optional[DefaultDependencyProvider] = None,
) -> Path:
    """Ensure a directory exists, creating it if necessary."""

    operation_name = "ensure_dir_exists"

    if _deps is None:
        _deps = get_dependency_provider()

    logger.debug(f"Ensuring directory exists: {path}")

    if not path:
        error_msg = (
            f"Invalid path for {operation_name}: '{path}'. "
            "Path must be a non-empty string or Path object."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)

    try:
        result = _deps.utils.ensure_directory_exists(path)
        logger.debug(f"Directory ensured: {result}")
        return result
    except Exception as exc:
        error_msg = (
            f"Failed to ensure directory exists for {operation_name}: {exc}. "
            "Please check the path format and permissions."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from exc


def check_if_file_exists(
    path: Union[str, Path],
    _deps: Optional[DefaultDependencyProvider] = None,
) -> bool:
    """Check whether a file exists."""

    operation_name = "check_if_file_exists"

    if _deps is None:
        _deps = get_dependency_provider()

    logger.debug(f"Checking if file exists: {path}")

    if not path:
        error_msg = (
            f"Invalid path for {operation_name}: '{path}'. "
            "Path must be a non-empty string or Path object."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)

    try:
        result = _deps.utils.check_file_exists(path)
        logger.debug(f"File exists: {result}")
        return result
    except Exception as exc:
        error_msg = (
            f"Failed to check file existence for {operation_name} at {path}: {exc}. "
            "Please check the path format."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from exc


def _raise_path_validation_error(prefix: str, operation_name: str, path_value: Any, suffix: str) -> None:
    """Raise a path validation error with standardized formatting."""

    error_msg = f"{prefix}{operation_name}: '{path_value}{suffix}"
    logger.error(error_msg)
    raise FlyRigLoaderError(error_msg)


def get_path_relative_to(
    path: Union[str, Path],
    base_dir: Union[str, Path],
    _deps: Optional[DefaultDependencyProvider] = None,
) -> Path:
    """Get a path relative to a base directory with enhanced testability."""

    operation_name = "get_path_relative_to"

    if _deps is None:
        _deps = get_dependency_provider()

    logger.debug(f"Getting relative path: {path} relative to {base_dir}")

    if not path:
        _raise_path_validation_error(
            "Invalid path for ",
            operation_name,
            path,
            "'. path must be a non-empty string or Path object.",
        )
    if not base_dir:
        _raise_path_validation_error(
            "Invalid base_dir for ",
            operation_name,
            base_dir,
            "'. base_dir must be a non-empty string or Path object.",
        )
    try:
        result = _deps.utils.get_relative_path(path, base_dir)
        logger.debug(f"Relative path result: {result}")
        return result
    except ValueError as exc:
        error_msg = (
            f"Path {path} is not within base directory {base_dir} for {operation_name}. "
            "Please ensure the path is within the specified base directory."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from exc
    except Exception as exc:
        error_msg = (
            f"Failed to get relative path for {operation_name}: {exc}. "
            "Please check the path formats."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from exc


def get_path_absolute(
    path: Union[str, Path],
    base_dir: Union[str, Path],
    _deps: Optional[DefaultDependencyProvider] = None,
) -> Path:
    """Convert a relative path to an absolute path with enhanced testability."""

    operation_name = "get_path_absolute"

    if _deps is None:
        _deps = get_dependency_provider()

    logger.debug(f"Getting absolute path: {path} with base {base_dir}")

    if not path:
        _raise_path_validation_error(
            "Invalid path for ",
            operation_name,
            path,
            "'. path must be a non-empty string or Path object.",
        )
    if not base_dir:
        _raise_path_validation_error(
            "Invalid base_dir for ",
            operation_name,
            base_dir,
            "'. base_dir must be a non-empty string or Path object.",
        )
    try:
        result = _deps.utils.get_absolute_path(path, base_dir)
        logger.debug(f"Absolute path result: {result}")
        return result
    except Exception as exc:
        error_msg = (
            f"Failed to get absolute path for {operation_name}: {exc}. "
            "Please check the path formats."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from exc


def get_common_base_dir(
    paths: List[Union[str, Path]],
    _deps: Optional[DefaultDependencyProvider] = None,
) -> Optional[Path]:
    """Find the common base directory for a list of paths with enhanced testability."""

    operation_name = "get_common_base_dir"

    if _deps is None:
        _deps = get_dependency_provider()

    logger.debug(f"Finding common base directory for {len(paths) if paths else 0} paths")

    if not isinstance(paths, list):
        error_msg = (
            f"Invalid paths parameter for {operation_name}: {type(paths).__name__}. "
            "paths must be a list of path strings or Path objects."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)

    if not paths:
        logger.debug("Empty paths list provided, returning None")
        return None

    for idx, path in enumerate(paths):
        if not path:
            error_msg = (
                f"Invalid path at index {idx} for {operation_name}: '{path}'. "
                "All paths must be non-empty strings or Path objects."
            )
            logger.error(error_msg)
            raise FlyRigLoaderError(error_msg)

    try:
        result = _deps.utils.find_common_base_directory(paths)
        if result:
            logger.debug(f"Found common base directory: {result}")
        else:
            logger.debug("No common base directory found")
        return result
    except Exception as exc:
        error_msg = (
            f"Failed to find common base directory for {operation_name}: {exc}. "
            "Please check the path formats."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from exc


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

