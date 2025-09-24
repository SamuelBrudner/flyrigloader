"""Configuration helpers shared across the API facade."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Union

from flyrigloader import logger
from flyrigloader.exceptions import FlyRigLoaderError

from .dependencies import DefaultDependencyProvider, get_dependency_provider

CONFIG_SOURCE_ERROR_MESSAGE = "Exactly one of 'config_path' or 'config' must be provided"


def validate_config_parameters(
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


def resolve_config_source(
    config: Optional[Union[Dict[str, Any], Any]],
    config_path: Optional[Union[str, Path]],
    operation_name: str,
    deps: Optional[DefaultDependencyProvider],
) -> Dict[str, Any]:
    """Return the concrete configuration dictionary for the requested operation."""

    validate_config_parameters(config_path, config, operation_name)

    if config is not None:
        logger.debug(f"Using provided configuration object for {operation_name}")
        return config  # type: ignore[return-value]

    assert config_path is not None
    logger.debug(f"Loading configuration from file source {config_path} for {operation_name}")
    return load_and_validate_config(config_path, None, operation_name, deps)


def load_and_validate_config(
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


def coerce_config_for_version_validation(config_obj: Any) -> Union[Dict[str, Any], str]:
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
