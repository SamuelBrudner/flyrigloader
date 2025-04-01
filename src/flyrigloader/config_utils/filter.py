"""
filter.py - Module for filtering configuration data by experiment.

This module provides functions to filter a full configuration to focus on 
a specific experiment, with options for how the resulting structure should look.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path
from loguru import logger
from datetime import datetime

from ..core.utils import create_metadata, create_error_metadata


def filter_config_by_experiment(
    config: Dict[str, Any], 
    experiment_name: str,
    transform_structure: bool = False,
    preserve_all_keys: bool = True,
    preserved_keys: Optional[List[str]] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Filter configuration to focus on a specific experiment.
    
    This function encapsulates the common logic for filtering configurations
    by experiment name.
    
    Args:
        config: Full configuration dictionary
        experiment_name: Name of the experiment to filter for
        transform_structure: If True, transforms the experiment entry into a top-level 'experiment' key.
                            If False, keeps experiment under the 'experiments' dictionary.
        preserve_all_keys: If True, preserves all keys from the original config.
                          If False, only preserves keys specified in preserved_keys.
        preserved_keys: List of specific keys to preserve when preserve_all_keys is False.
                       Defaults to ['paths', 'rigs', 'datasets'] if None.
        
    Returns:
        Tuple of (filtered_config, metadata) where filtered_config is the filtered 
        configuration dictionary and metadata contains status information and any error details
    """
    try:
        metadata = create_metadata()
        
        # Set default preserved keys if none provided
        if preserved_keys is None:
            preserved_keys = ['paths', 'rigs', 'datasets']
        
        # Validate experiment exists
        exists, exp_metadata = _validate_experiment_exists(config, experiment_name)
        if not exists:
            metadata["error"] = exp_metadata["error"]
            logger.error(metadata["error"])
            return {}, metadata
        
        # Create the initial filtered configuration
        filtered_config = _create_filtered_config(config, experiment_name, transform_structure)
        
        # Add other configuration keys based on preserve settings
        _add_preserved_keys(filtered_config, config, preserve_all_keys, preserved_keys)
        
        metadata["success"] = True
        return filtered_config, metadata
        
    except Exception as e:
        logger.error(f"Error filtering config by experiment: {str(e)}")
        return {}, create_error_metadata(e)


def _validate_experiment_exists(config: Dict[str, Any], experiment_name: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that the specified experiment exists in the config.
    
    Args:
        config: Configuration dictionary
        experiment_name: Name of experiment to check
        
    Returns:
        Tuple of (exists, metadata) where exists is a boolean indicating if the 
        experiment exists and metadata contains status information and any error details
    """
    try:
        metadata = create_metadata()
        
        if experiment_name not in config.get('experiments', {}):
            metadata["error"] = f"Experiment '{experiment_name}' not found in config['experiments']"
            logger.error(metadata["error"])
            return False, metadata
            
        metadata["success"] = True
        return True, metadata
        
    except Exception as e:
        logger.error(f"Error validating experiment existence: {str(e)}")
        return False, create_error_metadata(e)


def _create_filtered_config(
    config: Dict[str, Any], 
    experiment_name: str, 
    transform_structure: bool
) -> Dict[str, Any]:
    """Create the initial filtered configuration with experiment data."""
    exp_entry = config['experiments'][experiment_name]
    
    # Return either transformed or original structure based on transform_structure flag
    return {
        'experiment': exp_entry,
        'datasets': _extract_relevant_datasets(config, exp_entry)
    } if transform_structure else {
        'experiments': {experiment_name: exp_entry}
    }


def _extract_relevant_datasets(
    config: Dict[str, Any], 
    experiment_entry: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract only the datasets that are used by the experiment."""
    relevant_datasets = {}
    for ds_name in experiment_entry.get('datasets', []):
        ds_entry = config.get('datasets', {}).get(ds_name)
        if ds_entry is None:
            logger.warning(f"Warning: dataset '{ds_name}' not found in config['datasets']")
        else:
            relevant_datasets[ds_name] = ds_entry
    return relevant_datasets


def _add_preserved_keys(
    filtered_config: Dict[str, Any],
    original_config: Dict[str, Any],
    preserve_all_keys: bool,
    preserved_keys: List[str]
) -> None:
    """Add preserved keys from the original config to the filtered config."""
    keys_to_copy = list(original_config.keys()) if preserve_all_keys else preserved_keys
    
    # Skip any keys that would overwrite the filtered config keys
    keys_to_skip = set(filtered_config.keys())
    
    for key in keys_to_copy:
        if key not in keys_to_skip and key in original_config:
            filtered_config[key] = original_config[key]


def get_experiment_config(
    config: Dict[str, Any],
    experiment_name: str,
    *,
    structure: str = "transformed",
    include_all_keys: bool = False,
    filter_criteria: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Get configuration for a specific experiment with standardized options.
    
    This function provides a standardized approach to filtering configuration
    for a specific experiment, with explicit options to control the structure
    and included keys.
    
    Args:
        config: Full configuration dictionary
        experiment_name: Name of the experiment to filter for
        structure: Structure of the resulting configuration:
            - "transformed": Places experiment data under a top-level 'experiment' key (default)
            - "preserved": Keeps experiment under the original 'experiments' dictionary
        include_all_keys: Whether to include all keys from the original config:
            - True: Include all keys from the original config (may be verbose)
            - False: Only include essential keys like 'paths', 'rigs', 'datasets' (default)
        filter_criteria: Optional dictionary of additional filtering criteria to apply.
                       Keys can be 'vials', 'rigs', etc. with values specifying what to include.
                       Example: {'vials': ['A1', 'B2'], 'rigs': ['rig1', 'rig2']}
            
    Returns:
        Tuple of (filtered_config, metadata) where filtered_config is the filtered 
        configuration dictionary for the specified experiment and metadata contains 
        status information and any error details
    """
    try:
        metadata = create_metadata()
        
        # Validate structure parameter
        structure_valid, struct_metadata = _validate_structure_param(structure)
        if not structure_valid:
            return {}, struct_metadata
            
        # Map user-friendly parameters to implementation parameters
        transform_structure, preserve_all_keys = _map_config_params(structure, include_all_keys)
        
        # Filter config by experiment
        filtered_config, exp_metadata = filter_config_by_experiment(
            config, 
            experiment_name, 
            transform_structure=transform_structure,
            preserve_all_keys=preserve_all_keys
        )
        
        if not exp_metadata["success"]:
            return {}, exp_metadata
            
        # Apply additional filtering criteria if provided
        if filter_criteria:
            try:
                filtered_config = _apply_filter_criteria(filtered_config, filter_criteria)
            except Exception as e:
                metadata["warning"] = f"Error applying filter criteria: {str(e)}"
                logger.warning(metadata["warning"])
        
        metadata["success"] = True
        return filtered_config, metadata
        
    except Exception as e:
        logger.error(f"Error getting experiment config: {str(e)}")
        return {}, create_error_metadata(e)


def _validate_structure_param(structure: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate the structure parameter for get_experiment_config.
    
    Args:
        structure: Structure parameter value to validate
        
    Returns:
        Tuple of (is_valid, metadata) where is_valid is a boolean indicating if the
        parameter is valid and metadata contains status information and any error details
    """
    try:
        metadata = create_metadata()
        valid_structures = ["transformed", "preserved"]
        
        if structure not in valid_structures:
            metadata["error"] = f"Invalid structure '{structure}'. Must be one of: {valid_structures}"
            logger.error(metadata["error"])
            return False, metadata
            
        metadata["success"] = True
        return True, metadata
        
    except Exception as e:
        logger.error(f"Error validating structure parameter: {str(e)}")
        return False, create_error_metadata(e)


def _map_config_params(structure: str, include_all_keys: bool) -> tuple:
    """Map user-friendly parameters to the underlying implementation parameters."""
    transform_structure = (structure == "transformed")
    preserve_all_keys = include_all_keys
    return transform_structure, preserve_all_keys


def _apply_filter_criteria(
    config: Dict[str, Any],
    filter_criteria: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply additional filtering criteria to a configuration.
    
    Filters elements in the configuration based on the specified criteria.
    For example, limit which vials or rigs are included.
    
    Args:
        config: Configuration dictionary to filter
        filter_criteria: Criteria to apply, with keys like 'vials', 'rigs'
        
    Returns:
        Filtered configuration dictionary
    """
    # Create a shallow copy to avoid modifying the original
    filtered = dict(config)
    
    # Apply experiment-level filtering if we have a 'transformed' structure
    if 'experiment' in filtered:
        for key, values in filter_criteria.items():
            if key in filtered['experiment']:
                # Handle list values (e.g., vials)
                if isinstance(filtered['experiment'][key], list):
                    filtered['experiment'][key] = [
                        item for item in filtered['experiment'][key] 
                        if item in values
                    ]
                # Handle dict values (e.g., vial_configs)
                elif isinstance(filtered['experiment'][key], dict):
                    filtered['experiment'][key] = {
                        k: v for k, v in filtered['experiment'][key].items()
                        if k in values
                    }
    
    # Apply experiment-level filtering if we have a 'preserved' structure
    elif 'experiments' in filtered:
        for exp_name, exp_data in filtered['experiments'].items():
            for key, values in filter_criteria.items():
                if key in exp_data:
                    # Handle list values
                    if isinstance(exp_data[key], list):
                        filtered['experiments'][exp_name][key] = [
                            item for item in exp_data[key] 
                            if item in values
                        ]
                    # Handle dict values
                    elif isinstance(exp_data[key], dict):
                        filtered['experiments'][exp_name][key] = {
                            k: v for k, v in exp_data[key].items()
                            if k in values
                        }
    
    # Apply rig-level filtering
    if 'rigs' in filtered and 'rigs' in filter_criteria:
        filtered['rigs'] = {
            rig_name: rig_config for rig_name, rig_config in filtered['rigs'].items()
            if rig_name in filter_criteria['rigs']
        }
    
    # Apply dataset-level filtering
    if 'datasets' in filtered and 'datasets' in filter_criteria:
        filtered['datasets'] = {
            ds_name: ds_config for ds_name, ds_config in filtered['datasets'].items()
            if ds_name in filter_criteria['datasets']
        }
    
    return filtered


def extract_experiment_names(config: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    """
    Extract a list of all experiment names from a configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Tuple of (experiment_names, metadata) where experiment_names is a list of 
        experiment names and metadata contains status information and any error details
    """
    try:
        metadata = create_metadata()
        
        if not config or 'experiments' not in config:
            metadata["error"] = "No experiments found in configuration"
            return [], metadata
            
        experiment_names = list(config.get('experiments', {}).keys())
        
        metadata["success"] = True
        metadata["count"] = len(experiment_names)
        return experiment_names, metadata
        
    except Exception as e:
        logger.error(f"Error extracting experiment names: {str(e)}")
        return [], create_error_metadata(e)


def extract_dataset_names(
    config: Dict[str, Any], 
    experiment_name: Optional[str] = None
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Extract dataset names from a configuration.
    
    If experiment_name is provided, only returns datasets associated with that experiment.
    Otherwise, returns all datasets in the configuration.
    
    Args:
        config: Full configuration dictionary
        experiment_name: Optional name of experiment to filter by
        
    Returns:
        Tuple of (dataset_names, metadata) where dataset_names is a list of 
        dataset names and metadata contains status information and any error details
    """
    try:
        metadata = create_metadata()
        
        if not config:
            metadata["error"] = "Empty configuration provided"
            return [], metadata
            
        # If experiment name is provided, get datasets for that experiment
        if experiment_name:
            # Check if experiment exists
            if experiment_name not in config.get('experiments', {}):
                metadata["error"] = f"Experiment '{experiment_name}' not found"
                return [], metadata
                
            # Get datasets from experiment configuration
            exp_config = config['experiments'][experiment_name]
            dataset_names = exp_config.get('datasets', [])
            
        # Otherwise, get all datasets in the configuration
        else:
            dataset_names = list(config.get('datasets', {}).keys())
        
        metadata["success"] = True
        metadata["count"] = len(dataset_names)
        return dataset_names, metadata
        
    except Exception as e:
        logger.error(f"Error extracting dataset names: {str(e)}")
        return [], create_error_metadata(e)