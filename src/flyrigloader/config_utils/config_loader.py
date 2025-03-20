"""
config_loader.py - Module for loading and merging YAML configuration files.

This module provides functionality for loading YAML files, merging multiple
config files with proper overriding behavior, and standardizing config values.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import copy

import yaml
from loguru import logger

from ..utils.files import ensure_file_exists, safe_load_yaml


def coerce_config_values(config: Dict[str, Any]) -> None:
    """
    Standardize various config values throughout the config dictionary.
    
    This function normalizes values for specific keys to ensure consistent types:
    - 'wind': Standardizes boolean/string values to numeric (0/1)
    - 'co2_primed': Standardizes to boolean (True/False)
    - 'tank_air': Standardizes to boolean (True/False)
    - 'humidity': Standardizes to float (percentage)
    - 'temperature': Standardizes to float (degrees)
    - 'intensity': Standardizes to integer
    - 'odor': Standardizes boolean/string values 
    
    Args:
        config: Configuration dictionary to update (in-place)
    """
    # Common value patterns for reuse
    common_values = {
        'boolean_true': [True, 'true', 'True', 'TRUE', '1', 1],
        'boolean_false': [False, 'false', 'False', 'FALSE', '0', 0, 'none', 'None', 'NONE', None],
        'temperature_na': ['na', 'NA', 'n/a', 'N/A', 'unknown'],
        'humidity_na': ['na', 'NA', 'n/a', 'N/A', 'unknown'],
    }
    
    # Define handlers for different field types
    field_handlers = {
        # Fields to normalize as numeric (0/1/etc.)
        'numeric_fields': {
            'wind': {
                # Values that should become 0
                'zero_values': common_values['boolean_false'],
                # Values that should become 1
                'one_values': common_values['boolean_true']
            },
            'intensity': {
                # No specific mapping, but ensure it's treated as integer
                'default': 0,
                'force_type': int
            }
        },
        # Fields to normalize as boolean (True/False)
        'boolean_fields': {
            'co2_primed': {
                # Values that should become False
                'false_values': common_values['boolean_false'],
                # Values that should become True
                'true_values': common_values['boolean_true']
            },
            'tank_air': {
                # Values that should become False
                'false_values': common_values['boolean_false'],
                # Values that should become True
                'true_values': common_values['boolean_true']
            },
            'odor': {
                # Values that should become False
                'false_values': common_values['boolean_false'],
                # Values that should become True
                'true_values': common_values['boolean_true']
            }
        },
        # Fields to normalize as float
        'float_fields': {
            'humidity': {
                # Values that should become NaN
                'nan_values': common_values['humidity_na'],
                # Ensure value is between 0-100
                'min_value': 0.0,
                'max_value': 100.0,
                # Default value if outside range or invalid
                'default': float('nan')
            },
            'temperature': {
                # Values that should become NaN
                'nan_values': common_values['temperature_na'],
                # No specific range limits for temperature (could be Celsius or Fahrenheit)
                'default': float('nan')
            }
        },
        # Fields to normalize as string, with specific mapping
        'string_fields': {
            'genotype': {
                # Standardize common genotype nomenclature
                'mapping': {
                    'wt': 'wild_type',
                    'WT': 'wild_type',
                    'wildtype': 'wild_type',
                    'wild-type': 'wild_type'
                },
                'default': 'unknown'
            }
        }
    }
    
    # Recursively walk the dictionary
    for key, val in list(config.items()):
        # Handle nested dictionaries recursively first
        if isinstance(val, dict):
            coerce_config_values(val)
            continue
            
        # Handle numeric fields (convert to 0, 1, or the numeric value)
        if key in field_handlers['numeric_fields']:
            _handle_numeric_field(config, key, val, field_handlers['numeric_fields'][key])
            continue
                
        # Handle boolean fields (convert to True/False)
        if key in field_handlers['boolean_fields']:
            _handle_boolean_field(config, key, val, field_handlers['boolean_fields'][key])
            continue
                
        # Handle float fields
        if key in field_handlers['float_fields']:
            _handle_float_field(config, key, val, field_handlers['float_fields'][key])
            continue
                    
        # Handle string fields with mapping
        if key in field_handlers['string_fields']:
            _handle_string_field(config, key, val, field_handlers['string_fields'][key])


def _handle_numeric_field(config: Dict[str, Any], key: str, val: Any, field_config: Dict[str, Any]) -> None:
    """Handle numeric field standardization."""
    if 'zero_values' in field_config and val in field_config['zero_values']:
        config[key] = 0
        return
        
    if 'one_values' in field_config and val in field_config['one_values']:
        config[key] = 1
        return
        
    # Convert numeric strings to actual numbers
    if isinstance(val, str) and val.replace('.', '', 1).isdigit():
        if 'force_type' in field_config and field_config['force_type'] == int:
            config[key] = int(float(val))
        else:
            config[key] = float(val) if '.' in val else int(val)
        return
        
    if 'force_type' in field_config:
        try:
            config[key] = field_config['force_type'](val)
        except (ValueError, TypeError):
            config[key] = field_config.get('default', 0)


def _handle_boolean_field(config: Dict[str, Any], key: str, val: Any, field_config: Dict[str, Any]) -> None:
    """Handle boolean field standardization."""
    if val in field_config['false_values']:
        config[key] = False
    elif val in field_config['true_values']:
        config[key] = True
    else:
        # Handle unexpected values
        default_value = field_config.get('default', False)
        logger.warning(
            f"Unexpected value '{val}' for boolean field '{key}'. "
            f"Expected one of: {field_config['true_values'] + field_config['false_values']}. "
            f"Setting to default: {default_value}"
        )
        config[key] = default_value


def _handle_float_field(config: Dict[str, Any], key: str, val: Any, field_config: Dict[str, Any]) -> None:
    """Handle float field standardization."""
    # Handle nan values
    if 'nan_values' in field_config and val in field_config['nan_values']:
        config[key] = float('nan')
        return
        
    # Try to convert to float
    try:
        float_val = float(val)
        
        # Apply min/max constraints if specified
        if 'min_value' in field_config and float_val < field_config['min_value']:
            float_val = field_config['min_value']
        if 'max_value' in field_config and float_val > field_config['max_value']:
            float_val = field_config['max_value']
            
        config[key] = float_val
    except (ValueError, TypeError):
        config[key] = field_config.get('default', float('nan'))


def _handle_string_field(config: Dict[str, Any], key: str, val: Any, field_config: Dict[str, Any]) -> None:
    """Handle string field standardization."""
    if 'mapping' in field_config and val in field_config['mapping']:
        config[key] = field_config['mapping'][val]
    elif val is None or val == '':
        config[key] = field_config.get('default', 'unknown')


def coerce_config_values_copy(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardize config values throughout the config dictionary, returning a new copy.
    
    This function creates a deep copy of the input dictionary and then applies
    the standard coercion, ensuring the original dictionary remains unchanged.
    
    Args:
        config: Configuration dictionary to standardize
        
    Returns:
        A new dictionary with standardized values
    """
    # Create a deep copy to avoid modifying the original
    config_copy = copy.deepcopy(config)
    
    # Apply the standard coercion to the copy
    coerce_config_values(config_copy)
    
    return config_copy


def deep_update(original: Dict[str, Any], update: Dict[str, Any]) -> None:
    """
    Update original dictionary with values from update dictionary, recursively.
    
    Args:
        original: Dictionary to update (modified in-place)
        update: Dictionary with values to update original with
    """
    for key, val in update.items():
        if key in original and isinstance(original[key], dict) and isinstance(val, dict):
            deep_update(original[key], val)
        else:
            original[key] = val


def load_merged_config(
    base_config_path: Union[str, Path],
    hardware_config_path: Optional[Union[str, Path]] = None,
    local_config_path: Optional[Union[str, Path]] = None,
    run_specific_config_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Load and merge multiple configuration files in a specific order:
      1) base_config (loaded first as foundation)
      2) hardware_config (overrides relevant parts of base_config)
      3) local_config (overrides previous)
      4) run_specific_config (overrides previous)
    Then coerce config values in the final config.
    
    Args:
        base_config_path: Path to the base configuration file (required)
        hardware_config_path: Path to the hardware configuration file
        local_config_path: Optional path to a local configuration file
        run_specific_config_path: Optional path to a run-specific configuration file
        
    Returns:
        Merged configuration dictionary
    """
    # Convert string paths to Path objects
    base_config_path = Path(base_config_path)
    
    # 1) Load base config (required)
    config = _load_base_config(base_config_path)
    
    # 2) Load and merge the hardware configuration
    if hardware_config_path:
        hardware_config_path = Path(hardware_config_path)
        config = _merge_hardware_config(config, hardware_config_path)
    
    # 3) Load and merge the local configuration if provided
    if local_config_path:
        local_config_path = Path(local_config_path)
        config = _merge_optional_config(config, local_config_path, "local")
    
    # 4) Load and merge the run-specific configuration if provided
    if run_specific_config_path:
        run_specific_config_path = Path(run_specific_config_path)
        config = _merge_optional_config(config, run_specific_config_path, "run-specific")
    
    # 5) Standardize values in the final merged config
    coerce_config_values(config)
    
    return config


def _load_base_config(base_config_path: Path) -> Dict[str, Any]:
    """Load the base configuration, which is required."""
    config = safe_load_yaml(base_config_path)
    if not config:
        error_msg = f"Base config not found or empty: {base_config_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    logger.info(f"Loaded base configuration from {base_config_path}")
    return config


def _merge_hardware_config(config: Dict[str, Any], hardware_config_path: Path) -> Dict[str, Any]:
    """Load and merge hardware configuration with base config."""
    if hardware_config := safe_load_yaml(hardware_config_path):
        deep_update(config, hardware_config)
        logger.info(f"Merged hardware configuration from {hardware_config_path}")
    else:
        logger.warning(f"Hardware configuration not found or empty: {hardware_config_path}")
    return config


def _merge_optional_config(config: Dict[str, Any], config_path: Path, config_type: str) -> Dict[str, Any]:
    """Load and merge an optional configuration file."""
    if optional_config := safe_load_yaml(config_path):
        deep_update(config, optional_config)
        logger.info(f"Merged {config_type} configuration from {config_path}")
    else:
        logger.warning(f"{config_type.capitalize()} configuration not found at {config_path}")
    return config


def get_default_config_paths(base_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Get the default configuration paths based on typical fly rig project structure.
    
    Args:
        base_dir: Optional base directory to use. If None, uses current working directory.
        
    Returns:
        Dictionary with default paths for various config files
    """
    if base_dir is None:
        base_dir = Path.cwd()
        
    conf_dir = base_dir / "conf"
    
    return {
        "base": conf_dir / "config.yaml",
        "hardware": conf_dir / "hardware.yaml",
        "local": conf_dir / "local.yaml",
        "run_specific": conf_dir / "run_specific_configuration.yaml"
    }