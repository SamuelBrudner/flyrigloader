"""Parameter access public API entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from flyrigloader import logger
from flyrigloader.config.models import LegacyConfigAdapter
from flyrigloader.exceptions import FlyRigLoaderError

from .configuration import resolve_config_source as _resolve_config_source
from .dependencies import DefaultDependencyProvider, get_dependency_provider

def get_experiment_parameters(
    config: Optional[Union[Dict[str, Any], Any]] = None,
    experiment_name: str = "",
    config_path: Optional[Union[str, Path]] = None,
    _deps: Optional[DefaultDependencyProvider] = None
) -> Dict[str, Any]:
    """
    Get parameters for a specific experiment with enhanced testability.
    
    This function supports comprehensive dependency injection for testing scenarios
    through the _deps parameter, enabling pytest.monkeypatch patterns. It now accepts
    Pydantic models directly for improved type safety.
    
    Args:
        config: Pre-loaded configuration dictionary, Pydantic model, or LegacyConfigAdapter
        experiment_name: Name of the experiment to get parameters for
        config_path: Path to the YAML configuration file (alternative to config)
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Dictionary containing experiment parameters
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        KeyError: If the experiment doesn't exist in the config
        ValueError: If neither config_path nor config is provided, or if both are provided
        
    Example:
        >>> # Using Pydantic model directly
        >>> config = create_config(
        ...     project_name="fly_behavior",
        ...     base_directory="/data/experiments"
        ... )
        >>> params = get_experiment_parameters(
        ...     config=config,
        ...     experiment_name="plume_navigation"
        ... )
        >>> print(f"Found {len(params)} parameters")
    """
    operation_name = "get_experiment_parameters"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.info(f"Getting parameters for experiment '{experiment_name}'")
    
    # Validate experiment_name parameter
    if not experiment_name or not isinstance(experiment_name, str):
        error_msg = (
            f"Invalid experiment_name for {operation_name}: '{experiment_name}'. "
            "experiment_name must be a non-empty string representing the experiment identifier."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    config_dict = _resolve_config_source(config, config_path, operation_name, _deps)
    
    # Get experiment info with enhanced error handling
    try:
        experiment_info = _deps.config.get_experiment_info(config_dict, experiment_name)
        logger.debug(f"Retrieved experiment info for '{experiment_name}'")
    except KeyError as e:
        available_experiments = list(config_dict.get("experiments", {}).keys())
        error_msg = (
            f"Experiment '{experiment_name}' not found in configuration. "
            f"Available experiments: {available_experiments}. "
            "Please check the experiment name and ensure it's defined in your configuration."
        )
        logger.error(error_msg)
        raise KeyError(error_msg) from e
    
    # Extract parameters with logging
    parameters = experiment_info.get("parameters", {})
    param_count = len(parameters) if isinstance(parameters, dict) else 0
    logger.info(f"Retrieved {param_count} parameters for experiment '{experiment_name}'")
    logger.debug(f"Parameter keys: {list(parameters.keys()) if isinstance(parameters, dict) else 'N/A'}")
    
    return parameters

def get_dataset_parameters(
    config: Optional[Union[Dict[str, Any], Any]] = None,
    dataset_name: str = "",
    config_path: Optional[Union[str, Path]] = None,
    _deps: Optional[DefaultDependencyProvider] = None
) -> Dict[str, Any]:
    """
    Get parameters for a specific dataset with enhanced testability.
    
    This function supports comprehensive dependency injection for testing scenarios
    through the _deps parameter, enabling pytest.monkeypatch patterns. It now accepts
    Pydantic models directly for improved type safety.
    
    Args:
        config: Pre-loaded configuration dictionary, Pydantic model, or LegacyConfigAdapter
        dataset_name: Name of the dataset to get parameters for
        config_path: Path to the YAML configuration file (alternative to config)
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Dictionary containing dataset parameters
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        KeyError: If the dataset doesn't exist in the config
        ValueError: If neither config_path nor config is provided, or if both are provided
        
    Example:
        >>> # Using Pydantic model directly
        >>> config = create_config(
        ...     project_name="fly_behavior",
        ...     base_directory="/data/experiments"
        ... )
        >>> params = get_dataset_parameters(
        ...     config=config,
        ...     dataset_name="plume_tracking"
        ... )
        >>> print(f"Found {len(params)} parameters")
    """
    operation_name = "get_dataset_parameters"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.info(f"Getting parameters for dataset '{dataset_name}'")
    
    # Validate dataset_name parameter
    if not dataset_name or not isinstance(dataset_name, str):
        error_msg = (
            f"Invalid dataset_name for {operation_name}: '{dataset_name}'. "
            "dataset_name must be a non-empty string representing the dataset identifier."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    config_dict = _resolve_config_source(config, config_path, operation_name, _deps)
    
    # Get dataset info with enhanced error handling
    try:
        dataset_info = _deps.config.get_dataset_info(config_dict, dataset_name)
        logger.debug(f"Retrieved dataset info for '{dataset_name}'")
    except KeyError as e:
        available_datasets = list(config_dict.get("datasets", {}).keys())
        error_msg = (
            f"Dataset '{dataset_name}' not found in configuration. "
            f"Available datasets: {available_datasets}. "
            "Please check the dataset name and ensure it's defined in your configuration."
        )
        logger.error(error_msg)
        raise KeyError(error_msg) from e
    
    # Extract parameters with logging
    parameters = dataset_info.get("parameters", {})
    param_count = len(parameters) if isinstance(parameters, dict) else 0
    logger.info(f"Retrieved {param_count} parameters for dataset '{dataset_name}'")
    logger.debug(f"Parameter keys: {list(parameters.keys()) if isinstance(parameters, dict) else 'N/A'}")
    
    return parameters
