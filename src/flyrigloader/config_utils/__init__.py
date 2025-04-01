"""
Configuration management for flyrigloader.

This module provides a type-safe, flexible configuration system based on Pydantic,
offering environment variable support, validation, and immutability.
"""

# Import and export the Pydantic-based API
from .models import (
    AppConfig,
    HardwareConfig,
    ExperimentConfig,
    DiscoveryConfig,
    ConfigSettings,
)

from .manager import (
    ConfigManager,
    default_config_manager,
    get_config,
    get_hardware_config,
    get_experiment_config,
)

# Import filter functionality
from .filter import (
    filter_config_by_experiment,
    get_experiment_config as get_filtered_experiment_config,
    extract_experiment_names,
    extract_dataset_names,
)

__all__ = [
    # Model classes
    "AppConfig",
    "HardwareConfig",
    "ExperimentConfig",
    "DiscoveryConfig",
    "ConfigSettings",
    
    # Manager classes and functions
    "ConfigManager",
    "default_config_manager",
    "get_config",
    "get_hardware_config",
    "get_experiment_config",
    
    # Filter functionality
    "filter_config_by_experiment",
    "get_filtered_experiment_config",  # Renamed to avoid conflict
    "extract_experiment_names",
    "extract_dataset_names",
]

# Initialize default configuration manager
# This makes configuration instantly available to importing modules
default_config = get_config()
