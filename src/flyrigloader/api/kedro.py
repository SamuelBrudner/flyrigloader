"""Optional Kedro integration helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, Optional, Union

from flyrigloader import logger
from flyrigloader.exceptions import FlyRigLoaderError

from .configuration import load_and_validate_config
from .dependencies import DefaultDependencyProvider, get_dependency_provider

_KEDRO_IMPORT_ERROR: Optional[ModuleNotFoundError] = None
try:
    _KEDRO_SPEC = importlib.util.find_spec("kedro")
except (ValueError, ModuleNotFoundError):
    _KEDRO_SPEC = None

if _KEDRO_SPEC is not None:
    try:  # pragma: no branch - executed once at import
        from flyrigloader.kedro.datasets import FlyRigLoaderDataSet
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
        FlyRigLoaderDataSet = None  # type: ignore[assignment]
        _KEDRO_IMPORT_ERROR = exc
else:  # pragma: no cover - environment specific
    FlyRigLoaderDataSet = None  # type: ignore[assignment]
    _KEDRO_IMPORT_ERROR = ModuleNotFoundError(
        "kedro is not installed; FlyRigLoader Kedro integration is unavailable."
    )

__all__ = [
    "FlyRigLoaderDataSet",
    "check_kedro_available",
    "create_kedro_dataset",
]


def check_kedro_available() -> None:
    """Raise a clear error if Kedro support is unavailable."""

    if FlyRigLoaderDataSet is not None:
        return

    message = (
        "Kedro integration requires the 'kedro' package. "
        "Install flyrigloader with the 'kedro' extra or add kedro to your environment."
    )
    if _KEDRO_IMPORT_ERROR is not None:
        logger.error(f"Kedro integration unavailable: {_KEDRO_IMPORT_ERROR}")
    raise FlyRigLoaderError(message) from _KEDRO_IMPORT_ERROR


def create_kedro_dataset(
    config_path: Union[str, Path],
    experiment_name: str,
    *,
    recursive: bool = True,
    extract_metadata: bool = True,
    parse_dates: bool = True,
    dataset_options: Optional[Dict[str, Any]] = None,
    _deps: Optional[DefaultDependencyProvider] = None,
) -> "FlyRigLoaderDataSet":
    """Factory function for creating Kedro dataset instances."""

    check_kedro_available()

    operation_name = "create_kedro_dataset"

    if _deps is None:
        _deps = get_dependency_provider()

    logger.info(f"🏗️ Creating Kedro dataset for experiment '{experiment_name}'")
    logger.debug(
        "Dataset parameters: config_path=%s, recursive=%s, extract_metadata=%s, parse_dates=%s",
        config_path,
        recursive,
        extract_metadata,
        parse_dates,
    )

    if not config_path:
        error_msg = (
            f"Invalid config_path for {operation_name}: '{config_path}'. "
            "config_path must be a non-empty string or Path object pointing to the configuration file."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)

    if not experiment_name or not isinstance(experiment_name, str):
        error_msg = (
            f"Invalid experiment_name for {operation_name}: '{experiment_name}'. "
            "experiment_name must be a non-empty string representing the experiment identifier."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)

    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        error_msg = (
            f"Configuration file not found for {operation_name}: {config_path}. "
            "Please ensure the configuration file exists and the path is correct."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        logger.debug("Pre-validating configuration for dataset creation")
        config_dict = load_and_validate_config(str(config_path), None, operation_name, _deps)

        experiments = config_dict.get("experiments")
        if not experiments:
            error_msg = (
                f"No experiments found in configuration for {operation_name}. "
                "Configuration must contain an 'experiments' section."
            )
            logger.error(error_msg)
            raise FlyRigLoaderError(error_msg)

        if experiment_name not in experiments:
            available_experiments = list(experiments.keys())
            error_msg = (
                f"Experiment '{experiment_name}' not found in configuration. "
                f"Available experiments: {available_experiments}. "
                "Please check the experiment name and ensure it's defined in your configuration."
            )
            logger.error(error_msg)
            raise FlyRigLoaderError(error_msg)

        dataset_kwargs = {
            "config_path": str(config_path_obj.resolve()),
            "experiment_name": experiment_name,
            "recursive": recursive,
            "extract_metadata": extract_metadata,
            "parse_dates": parse_dates,
        }

        if dataset_options:
            logger.debug("Adding custom dataset options: %s", list(dataset_options.keys()))
            dataset_kwargs.update(dataset_options)

        logger.debug("Creating FlyRigLoaderDataSet instance")
        dataset = FlyRigLoaderDataSet(
            filepath=dataset_kwargs["config_path"],
            experiment_name=dataset_kwargs["experiment_name"],
            recursive=dataset_kwargs["recursive"],
            extract_metadata=dataset_kwargs["extract_metadata"],
            parse_dates=dataset_kwargs["parse_dates"],
        )

        logger.info(f"✓ Successfully created Kedro dataset for experiment '{experiment_name}'")
        logger.debug("  Dataset configuration: %s", config_path)
        logger.debug("  Dataset options: %s", list(dataset_kwargs.keys()))

        return dataset

    except Exception as exc:
        if isinstance(exc, (ValueError, FileNotFoundError, FlyRigLoaderError)):
            raise

        error_msg = (
            f"Failed to create Kedro dataset for {operation_name}: {exc}. "
            "Please check the configuration file and experiment parameters."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from exc
