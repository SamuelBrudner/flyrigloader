"""High-level loading helpers for :mod:`flyrigloader.api`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import functools
import warnings

import pandas as pd

from flyrigloader import logger
from flyrigloader.exceptions import FlyRigLoaderError
from flyrigloader.io.loaders import load_data_file as _load_data_file

from ._shared import resolve_api_override
from .config import (
    _attach_metadata_bucket,
    _resolve_base_directory,
    _resolve_config_source,
)
from .dependencies import (
    ConfigProvider,
    DefaultDependencyProvider,
    DiscoveryProvider,
    IOProvider,
    UtilsProvider,
    get_dependency_provider,
)


def _get_api_override(name: str, fallback: Callable[..., Any]) -> Callable[..., Any]:
    """Return patched attribute from :mod:`flyrigloader.api` when available."""

    return resolve_api_override(globals(), name, fallback)


def load_data_file(
    file_path: Union[str, Path],
    validate_format: bool = True,
    loader: Optional[str] = None,
    _deps: Optional[DefaultDependencyProvider] = None,
) -> Dict[str, Any]:
    """Load raw data from a single file without DataFrame transformation."""

    operation_name = "load_data_file"

    if _deps is None:
        _deps = get_dependency_provider()

    logger.debug("📁 Loading data file: %s", file_path)

    if not file_path:
        error_msg = (
            f"Invalid file_path for {operation_name}: '{file_path}'. "
            "file_path must be a non-empty string or Path object pointing to the data file."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)

    try:
        loader_impl = _get_api_override("_load_data_file", _load_data_file)
        raw_data = loader_impl(file_path, loader)

        if validate_format and not isinstance(raw_data, dict):
            logger.error(
                "Invalid data format while loading %s: expected dict, got %s",
                file_path,
                type(raw_data).__name__,
            )
            raise FlyRigLoaderError(
                f"Invalid data format: expected dict, got {type(raw_data).__name__}"
            )

        if isinstance(raw_data, dict):
            logger.debug("Loaded data columns: %s", list(raw_data.keys()))

        return raw_data
    except FlyRigLoaderError:
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        error_msg = (
            f"Failed to load data from {file_path} for {operation_name}: {exc}. "
            "Please check the file format and ensure it's a valid data file."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from exc


def load_experiment_files(
    config: Optional[Union[Dict[str, Any], Any]] = None,
    experiment_name: str = "",
    config_path: Optional[Union[str, Path]] = None,
    base_directory: Optional[Union[str, Path]] = None,
    pattern: str = "*.*",
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
    extract_metadata: bool = False,
    parse_dates: bool = False,
    _deps: Optional[DefaultDependencyProvider] = None,
) -> Union[List[str], Dict[str, Dict[str, Any]]]:
    """High-level function to load files for a specific experiment."""

    operation_name = "load_experiment_files"

    if _deps is None:
        _deps = get_dependency_provider()

    logger.info("Loading experiment files for experiment '%s'", experiment_name)
    logger.debug(
        "Parameters: pattern=%s, recursive=%s, extensions=%s, extract_metadata=%s, parse_dates=%s",
        pattern,
        recursive,
        extensions,
        extract_metadata,
        parse_dates,
    )

    if not experiment_name or not isinstance(experiment_name, str):
        error_msg = (
            f"Invalid experiment_name for {operation_name}: '{experiment_name}'. "
            "experiment_name must be a non-empty string representing the experiment identifier."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)

    config_dict = _resolve_config_source(config, config_path, operation_name, _deps)
    base_directory = _resolve_base_directory(config_dict, base_directory, operation_name)

    try:
        _deps.config.get_experiment_info(config_dict, experiment_name)
    except KeyError as exc:
        available_experiments = list(config_dict.get("experiments", {}).keys())
        error_msg = (
            f"Experiment '{experiment_name}' not found in configuration. "
            f"Available experiments: {available_experiments}. "
            "Please check the experiment name and ensure it's defined in your configuration."
        )
        logger.error(error_msg)
        raise KeyError(error_msg) from exc

    try:
        result = _deps.discovery.discover_experiment_files(
            config=config_dict,
            experiment_name=experiment_name,
            base_directory=base_directory,
            pattern=pattern,
            recursive=recursive,
            extensions=extensions,
            extract_metadata=extract_metadata,
            parse_dates=parse_dates,
        )

        if extract_metadata or parse_dates:
            result = _attach_metadata_bucket(result)

        file_count = len(result) if isinstance(result, (list, dict)) else 0
        logger.info(
            "Successfully discovered %s files for experiment '%s'",
            file_count,
            experiment_name,
        )
        return result
    except Exception as exc:  # pragma: no cover - defensive logging
        error_msg = (
            f"Failed to discover files for experiment '{experiment_name}': {exc}. "
            "Please check the experiment configuration and data directory structure."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from exc


def load_dataset_files(
    config: Optional[Union[Dict[str, Any], Any]] = None,
    dataset_name: str = "",
    config_path: Optional[Union[str, Path]] = None,
    base_directory: Optional[Union[str, Path]] = None,
    pattern: str = "*.*",
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
    extract_metadata: bool = False,
    parse_dates: bool = False,
    _deps: Optional[DefaultDependencyProvider] = None,
) -> Union[List[str], Dict[str, Dict[str, Any]]]:
    """High-level function to load files for a specific dataset."""

    operation_name = "load_dataset_files"

    if _deps is None:
        _deps = get_dependency_provider()

    logger.info("Loading dataset files for dataset '%s'", dataset_name)
    logger.debug(
        "Parameters: pattern=%s, recursive=%s, extensions=%s, extract_metadata=%s, parse_dates=%s",
        pattern,
        recursive,
        extensions,
        extract_metadata,
        parse_dates,
    )

    if not dataset_name or not isinstance(dataset_name, str):
        error_msg = (
            f"Invalid dataset_name for {operation_name}: '{dataset_name}'. "
            "dataset_name must be a non-empty string representing the dataset identifier."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)

    config_dict = _resolve_config_source(config, config_path, operation_name, _deps)
    base_directory = _resolve_base_directory(config_dict, base_directory, operation_name)

    try:
        _deps.config.get_dataset_info(config_dict, dataset_name)
    except KeyError as exc:
        available_datasets = list(config_dict.get("datasets", {}).keys())
        error_msg = (
            f"Dataset '{dataset_name}' not found in configuration. "
            f"Available datasets: {available_datasets}. "
            "Please check the dataset name and ensure it's defined in your configuration."
        )
        logger.error(error_msg)
        raise KeyError(error_msg) from exc

    try:
        result = _deps.discovery.discover_dataset_files(
            config=config_dict,
            dataset_name=dataset_name,
            base_directory=base_directory,
            pattern=pattern,
            recursive=recursive,
            extensions=extensions,
            extract_metadata=extract_metadata,
            parse_dates=parse_dates,
        )

        if extract_metadata or parse_dates:
            result = _attach_metadata_bucket(result)

        file_count = len(result) if isinstance(result, (list, dict)) else 0
        logger.info(
            "Successfully discovered %s files for dataset '%s'",
            file_count,
            dataset_name,
        )
        return result
    except Exception as exc:  # pragma: no cover - defensive logging
        error_msg = (
            f"Failed to discover files for dataset '{dataset_name}': {exc}. "
            "Please check the dataset configuration and data directory structure."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from exc


def get_experiment_parameters(
    config: Optional[Union[Dict[str, Any], Any]] = None,
    experiment_name: str = "",
    config_path: Optional[Union[str, Path]] = None,
    _deps: Optional[DefaultDependencyProvider] = None,
) -> Dict[str, Any]:
    """Get parameters for a specific experiment."""

    operation_name = "get_experiment_parameters"

    if _deps is None:
        _deps = get_dependency_provider()

    logger.info("Getting parameters for experiment '%s'", experiment_name)

    if not experiment_name or not isinstance(experiment_name, str):
        error_msg = (
            f"Invalid experiment_name for {operation_name}: '{experiment_name}'. "
            "experiment_name must be a non-empty string representing the experiment identifier."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)

    config_dict = _resolve_config_source(config, config_path, operation_name, _deps)

    try:
        experiment_info = _deps.config.get_experiment_info(config_dict, experiment_name)
    except KeyError as exc:
        available_experiments = list(config_dict.get("experiments", {}).keys())
        error_msg = (
            f"Experiment '{experiment_name}' not found in configuration. "
            f"Available experiments: {available_experiments}. "
            "Please check the experiment name and ensure it's defined in your configuration."
        )
        logger.error(error_msg)
        raise KeyError(error_msg) from exc

    parameters = experiment_info.get("parameters", {})
    if isinstance(parameters, dict):
        logger.debug("Experiment parameters: %s", list(parameters.keys()))
    return parameters


def get_dataset_parameters(
    config: Optional[Union[Dict[str, Any], Any]] = None,
    dataset_name: str = "",
    config_path: Optional[Union[str, Path]] = None,
    _deps: Optional[DefaultDependencyProvider] = None,
) -> Dict[str, Any]:
    """Get parameters for a specific dataset."""

    operation_name = "get_dataset_parameters"

    if _deps is None:
        _deps = get_dependency_provider()

    logger.info("Getting parameters for dataset '%s'", dataset_name)

    if not dataset_name or not isinstance(dataset_name, str):
        error_msg = (
            f"Invalid dataset_name for {operation_name}: '{dataset_name}'. "
            "dataset_name must be a non-empty string representing the dataset identifier."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)

    config_dict = _resolve_config_source(config, config_path, operation_name, _deps)

    try:
        dataset_info = _deps.config.get_dataset_info(config_dict, dataset_name)
    except KeyError as exc:
        available_datasets = list(config_dict.get("datasets", {}).keys())
        error_msg = (
            f"Dataset '{dataset_name}' not found in configuration. "
            f"Available datasets: {available_datasets}. "
            "Please check the dataset name and ensure it's defined in your configuration."
        )
        logger.error(error_msg)
        raise KeyError(error_msg) from exc

    parameters = dataset_info.get("parameters", {})
    if isinstance(parameters, dict):
        logger.debug("Dataset parameters: %s", list(parameters.keys()))
    return parameters


def deprecated(reason: str, alternative: str):
    """Decorator for marking deprecated functions."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                f"{func.__name__} is deprecated. {reason}. Use {alternative} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


@deprecated(
    reason="Monolithic approach is less flexible. Use the new decoupled architecture",
    alternative="load_data_file() + transform_to_dataframe()",
)
def process_experiment_data(
    data_path: Union[str, Path],
    *,
    column_config_path: Optional[Union[str, Path, Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    strict_schema: bool = False,
    _deps: Optional[DefaultDependencyProvider] = None,
) -> pd.DataFrame:
    """Process experimental data and return a pandas DataFrame."""

    operation_name = "process_experiment_data"

    if _deps is None:
        _deps = get_dependency_provider()

    logger.info("📊 Processing experimental data from: %s", data_path)

    try:
        raw_data = load_data_file(
            file_path=data_path,
            validate_format=True,
            _deps=_deps,
        )

        from . import transformation as _transformation_module

        df = _transformation_module.transform_to_dataframe(
            raw_data=raw_data,
            column_config_path=column_config_path,
            metadata=metadata,
            add_file_path=True,
            file_path=data_path,
            strict_schema=strict_schema,
            _deps=_deps,
        )

        logger.info("✓ Successfully processed experimental data")
        logger.debug("DataFrame columns: %s", list(df.columns))
        return df
    except Exception as exc:  # pragma: no cover - defensive logging
        error_msg = (
            f"Failed to process experimental data from {data_path}: {exc}. "
            "Please check the file format and column configuration compatibility."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from exc


def _create_test_dependency_provider(
    config_provider: Optional[ConfigProvider] = None,
    discovery_provider: Optional[DiscoveryProvider] = None,
    io_provider: Optional[IOProvider] = None,
    utils_provider: Optional[UtilsProvider] = None,
) -> DefaultDependencyProvider:
    """Create a dependency provider with optional overrides for testing."""

    logger.debug("Creating test dependency provider")

    test_provider = DefaultDependencyProvider()

    if config_provider is not None:
        test_provider._config_module = config_provider
    if discovery_provider is not None:
        test_provider._discovery_module = discovery_provider
    if io_provider is not None:
        test_provider._io_module = io_provider
    if utils_provider is not None:
        test_provider._utils_module = utils_provider

    return test_provider


__all__ = [
    "_create_test_dependency_provider",
    "deprecated",
    "get_dataset_parameters",
    "get_experiment_parameters",
    "load_data_file",
    "load_dataset_files",
    "load_experiment_files",
    "process_experiment_data",
]
