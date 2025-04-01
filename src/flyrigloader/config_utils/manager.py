"""
manager.py - Configuration management with immutability and type safety.

This module provides the ConfigManager class, which is responsible for loading,
validating, and providing access to configuration values using Pydantic models
for type safety and validation.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, TypeVar, Type, cast, get_type_hints

import yaml
from loguru import logger

from pydantic import BaseModel, ValidationError

from ..core.utils import (
    PathLike, ensure_path, 
    deep_update, deep_merge,
    get_env_var, get_env_path, safe_load_yaml
)
from .models import AppConfig, ConfigSettings, HardwareConfig, ExperimentConfig

T = TypeVar("T", bound=BaseModel)


class ConfigManager:
    """
    Configuration manager with type safety and immutability.
    
    This class manages loading, merging, and validating configuration files,
    providing type-safe access to configuration values using Pydantic models.
    It ensures configurations are immutable and properly validated.
    """
    
    def __init__(
        self,
        base_dir: Optional[PathLike] = None,
        config_settings: Optional[ConfigSettings] = None,
        run_specific_config_path: Optional[PathLike] = None,
    ):
        """
        Initialize the configuration manager.
        
        Args:
            base_dir: Optional base directory for configuration files
            config_settings: Optional custom configuration settings
            run_specific_config_path: Optional path to run-specific config
        """
        self.base_dir = ensure_path(base_dir) if base_dir else Path(os.getcwd())
        
        # Use provided settings or create defaults
        if config_settings:
            self.settings = config_settings
        else:
            # Initialize with base_dir
            self.settings = ConfigSettings(config_dir=self.base_dir / "config")
            
        # Store the run-specific config path
        self.run_specific_config_path = (
            ensure_path(run_specific_config_path) if run_specific_config_path else None
        )
        
        # Initialize configuration
        self._raw_config: Dict[str, Any] = {}
        self._app_config: Optional[AppConfig] = None
        self.reload()
    
    def _initialize_default_config(self, error_message: str) -> None:
        """
        Initialize configuration with defaults when errors occur.
        
        Args:
            error_message: Error message to log
        """
        logger.error(error_message)
        self._app_config = AppConfig()
        logger.warning("Using default configuration due to errors")
    
    def reload(self) -> None:
        """Reload configuration from all sources."""
        try:
            # Load and merge configuration files
            self._raw_config = self._load_merged_config()
            
            # Create validated Pydantic model
            self._app_config = AppConfig.model_validate(self._raw_config)
            
            logger.info("Configuration loaded and validated successfully")
        except ValidationError as e:
            self._initialize_default_config(f"Configuration validation failed: {e}")
        except Exception as e:
            self._initialize_default_config(f"Error loading configuration: {e}")
    
    def _load_config_file(self, config_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load a single configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary or None if error
        """
        if config_path.exists():
            result, error_info = safe_load_yaml(config_path)
            if error_info["success"]:
                return result
            logger.error(f"Error loading config file {config_path}: {error_info.get('error', 'Unknown error')}")
        return None
    
    def _load_merged_config(self) -> Dict[str, Any]:
        """
        Load and merge all configuration files.
        
        Returns:
            Merged configuration dictionary
        """
        # Create empty dictionary to start with
        config: Dict[str, Any] = {}
        
        # Load base config first
        if base_config := self._load_config_file(self.settings.get_base_config_path()):
            config.update(base_config)
        
        # Load hardware config
        if hardware_config := self._load_config_file(self.settings.get_hardware_config_path()):
            deep_update(config, hardware_config)
        
        # Load local config (overrides previous)
        if local_config := self._load_config_file(self.settings.get_local_config_path()):
            deep_update(config, local_config)
        
        # Load run-specific config if provided (highest priority)
        if self.run_specific_config_path and (run_config := self._load_config_file(self.run_specific_config_path)):
            deep_update(config, run_config)
        
        return config
    
    @property
    def config(self) -> AppConfig:
        """
        Get the validated application configuration.
        
        Returns:
            AppConfig instance with validated configuration
        """
        if self._app_config is None:
            # This shouldn't happen as we initialize in __init__, but just in case
            self._app_config = AppConfig()
        return self._app_config
    
    def get_model(self, model_cls: Type[T]) -> T:
        """
        Get a specific configuration model.
        
        Args:
            model_cls: Pydantic model class to retrieve
            
        Returns:
            Instance of the requested model
        """
        if self._app_config is None:
            self.reload()
            
        if model_cls == AppConfig:
            return cast(T, self.config)
        elif model_cls == HardwareConfig:
            return cast(T, self.config.hardware)
        elif model_cls == ExperimentConfig:
            return cast(T, self.config.experiment)
        else:
            # For other models, validate raw config against the model
            return model_cls.model_validate(self._raw_config)
    
    def get_value(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation path.
        
        Args:
            path: Dot-separated path to the value (e.g., 'hardware.camera.exposure')
            default: Default value to return if not found
            
        Returns:
            The configuration value or default
        """
        if not self._app_config:
            return default
            
        current = self._app_config.model_dump()
        for part in path.split('.'):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return self._app_config.model_dump() if self._app_config else {}
    
    def save_to_file(self, filepath: PathLike) -> None:
        """
        Save the current configuration to a YAML file.
        
        Args:
            filepath: Path to save the configuration to
        """
        filepath = ensure_path(filepath)
        config_dict = self.to_dict()
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to file
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {filepath}")


# Create a global instance for convenience
default_config_manager = ConfigManager()


def get_config() -> AppConfig:
    """
    Get the application configuration from the default manager.
    
    This is a convenience function for code that doesn't need
    the full ConfigManager functionality.
    
    Returns:
        AppConfig instance
    """
    return default_config_manager.config


def get_hardware_config() -> HardwareConfig:
    """
    Get hardware configuration from the default manager.
    
    Returns:
        HardwareConfig instance
    """
    return default_config_manager.config.hardware


def get_experiment_config() -> ExperimentConfig:
    """
    Get experiment configuration from the default manager.
    
    Returns:
        ExperimentConfig instance
    """
    return default_config_manager.config.experiment
