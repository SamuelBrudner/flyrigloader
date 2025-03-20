"""
filter.py - Module for filtering configuration data by experiment.

This module provides functions to filter a full configuration to focus on 
a specific experiment, with options for how the resulting structure should look.
"""

from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path
from loguru import logger


def filter_config_by_experiment(
    config: Dict[str, Any], 
    experiment_name: str,
    transform_structure: bool = False,
    preserve_all_keys: bool = True,
    preserved_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
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
        Filtered configuration dictionary
        
    Raises:
        ValueError: If the specified experiment is not found in the config
    """
    # Set default preserved keys if none provided
    if preserved_keys is None:
        preserved_keys = ['paths', 'rigs', 'datasets']
    
    # Validate experiment exists
    _validate_experiment_exists(config, experiment_name)
    
    # Create the initial filtered configuration
    filtered_config = _create_filtered_config(config, experiment_name, transform_structure)
    
    # Add other configuration keys based on preserve settings
    _add_preserved_keys(filtered_config, config, preserve_all_keys, preserved_keys)
    
    return filtered_config


def _validate_experiment_exists(config: Dict[str, Any], experiment_name: str) -> None:
    """Validate that the specified experiment exists in the config."""
    if experiment_name not in config.get('experiments', {}):
        error_msg = f"Experiment '{experiment_name}' not found in config['experiments']"
        logger.error(error_msg)
        raise ValueError(error_msg)


def _create_filtered_config(
    config: Dict[str, Any], 
    experiment_name: str, 
    transform_structure: bool
) -> Dict[str, Any]:
    """Create the initial filtered configuration with experiment data."""
    exp_entry = config['experiments'][experiment_name]
    
    if transform_structure:
        # Create a transformed structure with experiment data at top level
        filtered_config = {
            'experiment': exp_entry,
            'datasets': _extract_relevant_datasets(config, exp_entry)
        }
    else:
        # Preserve original structure with experiments dictionary
        filtered_config = {
            'experiments': {experiment_name: exp_entry}
        }
    
    return filtered_config


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
    """Add keys from the original config to the filtered config based on preservation settings."""
    if preserve_all_keys:
        # Copy all other keys except those we've already processed
        for key in original_config:
            if key != 'experiments' and key not in filtered_config:
                filtered_config[key] = original_config[key]
    else:
        # Copy only specified keys
        for key in preserved_keys:
            if key in original_config and key not in filtered_config:
                filtered_config[key] = original_config[key]


def get_experiment_config(
    config: Dict[str, Any],
    experiment_name: str,
    *,
    structure: str = "transformed",
    include_all_keys: bool = False,
    filter_criteria: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
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
        Filtered configuration dictionary for the specified experiment
        
    Raises:
        ValueError: If experiment_name is not found in config
        ValueError: If structure is not one of the valid options
    """
    # Validate and map parameters
    _validate_structure_param(structure)
    transform_params = _map_config_params(structure, include_all_keys)
    
    # Get initial filtered config
    filtered_config = filter_config_by_experiment(
        config,
        experiment_name,
        **transform_params
    )
    
    # Apply additional filtering if specified
    if filter_criteria:
        filtered_config = _apply_filter_criteria(filtered_config, filter_criteria)
    
    return filtered_config


def _validate_structure_param(structure: str) -> None:
    """Validate the structure parameter for get_experiment_config."""
    valid_structures = ["transformed", "preserved"]
    if structure not in valid_structures:
        raise ValueError(f"structure must be one of {valid_structures}")


def _map_config_params(structure: str, include_all_keys: bool) -> Dict[str, Any]:
    """Map user-friendly parameters to the underlying implementation parameters."""
    return {
        "transform_structure": structure == "transformed",
        "preserve_all_keys": include_all_keys
    }


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
    filtered_config = config.copy()
    
    # Process each filter criterion
    for key, allowed_values in filter_criteria.items():
        # Skip if the key doesn't exist in config
        if key not in filtered_config:
            continue
            
        # If the config item is a dictionary, filter its keys
        if isinstance(filtered_config[key], dict):
            filtered_config[key] = {
                k: v for k, v in filtered_config[key].items()
                if k in allowed_values
            }
    
    # Special handling for datasets when filtering by vials or rigs
    if 'datasets' in filtered_config and ('vials' in filter_criteria or 'rigs' in filter_criteria):
        # Collect datasets to remove
        datasets_to_remove = []
        
        for dataset_name, dataset_config in filtered_config['datasets'].items():
            # Filter dataset by vials if specified
            if 'vials' in filter_criteria and 'vials' in dataset_config:
                dataset_config['vials'] = [
                    vial for vial in dataset_config['vials'] 
                    if vial in filter_criteria['vials']
                ]
                
            # Filter dataset by rigs if specified
            if 'rigs' in filter_criteria and 'rigs' in dataset_config:
                dataset_config['rigs'] = [
                    rig for rig in dataset_config['rigs'] 
                    if rig in filter_criteria['rigs']
                ]
                
            # Mark dataset for removal if it has no vials or rigs left after filtering
            if ('vials' in dataset_config and not dataset_config['vials']) or \
               ('rigs' in dataset_config and not dataset_config['rigs']):
                datasets_to_remove.append(dataset_name)
        
        # Remove datasets that were marked for removal
        for dataset_name in datasets_to_remove:
            del filtered_config['datasets'][dataset_name]
    
    # Handle the special case of 'dates_vials' in dataset configurations
    if 'datasets' in filtered_config and 'vials' in filter_criteria:
        # Collect datasets to remove
        datasets_to_remove = []
        
        for dataset_name, dataset_config in filtered_config['datasets'].items():
            if 'dates_vials' in dataset_config:
                # For each date, filter the vials
                # Collect dates to remove
                dates_to_remove = []
                
                for date, vials in dataset_config['dates_vials'].items():
                    filtered_vials = [v for v in vials if v in filter_criteria['vials']]
                    if filtered_vials:
                        dataset_config['dates_vials'][date] = filtered_vials
                    else:
                        # Mark date for removal if no vials left
                        dates_to_remove.append(date)
                
                # Remove dates that were marked for removal
                for date in dates_to_remove:
                    del dataset_config['dates_vials'][date]
                
                # Mark dataset for removal if no dates left
                if not dataset_config['dates_vials']:
                    datasets_to_remove.append(dataset_name)
        
        # Remove datasets that were marked for removal
        for dataset_name in datasets_to_remove:
            del filtered_config['datasets'][dataset_name]
    
    return filtered_config


def extract_experiment_names(config: Dict[str, Any]) -> List[str]:
    """
    Extract a list of all experiment names from a configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        List of experiment names
    """
    experiments = config.get('experiments', {})
    return list(experiments.keys())


def extract_dataset_names(
    config: Dict[str, Any], 
    experiment_name: Optional[str] = None
) -> List[str]:
    """
    Extract dataset names from a configuration.
    
    If experiment_name is provided, only returns datasets associated with that experiment.
    Otherwise, returns all datasets in the configuration.
    
    Args:
        config: Full configuration dictionary
        experiment_name: Optional name of experiment to filter by
        
    Returns:
        List of dataset names
    """
    if experiment_name:
        # Get datasets for specific experiment
        experiment = config.get('experiments', {}).get(experiment_name, {})
        return experiment.get('datasets', [])
    else:
        # Get all datasets
        datasets = config.get('datasets', {})
        return list(datasets.keys())