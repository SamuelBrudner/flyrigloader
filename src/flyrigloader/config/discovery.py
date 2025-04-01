"""
Config-aware file discovery utilities.

Combines YAML configuration with file discovery functionality.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from flyrigloader.discovery.files import discover_files
from flyrigloader.config.yaml_config import (
    load_config,
    get_ignore_patterns,
    get_mandatory_substrings,
    get_dataset_info,
    get_experiment_info
)


def discover_files_with_config(
    config: Dict[str, Any],
    directory: Union[str, List[str]],
    pattern: str,
    recursive: bool = False,
    extensions: Optional[List[str]] = None,
    experiment: Optional[str] = None
) -> List[str]:
    """
    Discover files using configuration-aware filtering.
    
    This uses the configuration to determine ignore patterns and mandatory substrings
    based on project-wide settings and experiment-specific overrides.
    
    Args:
        config: The loaded configuration dictionary
        directory: The directory or list of directories to search in
        pattern: File pattern to match (glob format)
        recursive: If True, search recursively through subdirectories
        extensions: Optional list of file extensions to filter by (without the dot)
        experiment: Optional experiment name to use experiment-specific filters
        
    Returns:
        List of file paths matching the criteria
    """
    # Get ignore patterns from config (project + experiment level)
    ignore_patterns = get_ignore_patterns(config, experiment)
    
    # Get mandatory substrings from config (project + experiment level)
    mandatory_substrings = get_mandatory_substrings(config, experiment)
    
    # Use the core discovery function with config-derived filters
    return discover_files(
        directory=directory,
        pattern=pattern,
        recursive=recursive,
        extensions=extensions,
        ignore_patterns=ignore_patterns,
        mandatory_substrings=mandatory_substrings
    )


def discover_experiment_files(
    config: Dict[str, Any],
    experiment_name: str,
    base_directory: Union[str, Path],
    pattern: str = "*.*",
    recursive: bool = True,
    extensions: Optional[List[str]] = None
) -> List[str]:
    """
    Discover files related to a specific experiment.
    
    This uses the experiment's dataset definitions and filter settings
    to find relevant files.
    
    Args:
        config: The loaded configuration dictionary
        experiment_name: Name of the experiment to use for discovery
        base_directory: Base directory to search in (often the major_data_directory)
        pattern: File pattern to match (glob format), defaults to all files
        recursive: If True, search recursively through subdirectories
        extensions: Optional list of file extensions to filter by
        
    Returns:
        List of file paths relevant to the experiment
        
    Raises:
        KeyError: If the experiment does not exist in the configuration
    """
    # Get experiment information
    experiment_info = get_experiment_info(config, experiment_name)
    
    # Get the list of datasets for this experiment
    dataset_names = experiment_info.get("datasets", [])
    
    # Collect all date-specific directories to search in
    search_dirs = []
    for dataset_name in dataset_names:
        try:
            dataset_info = get_dataset_info(config, dataset_name)
            # Get all date directories for this dataset
            dates = dataset_info.get("dates_vials", {}).keys()
            for date in dates:
                date_dir = Path(base_directory) / str(date)
                if date_dir.exists():
                    search_dirs.append(str(date_dir))
        except KeyError:
            # Skip datasets that don't exist in the config
            continue
    
    # If no search directories were found, use the base directory
    if not search_dirs:
        search_dirs = [base_directory]
    
    # Discover files using config-aware filtering
    return discover_files_with_config(
        config=config,
        directory=search_dirs,
        pattern=pattern,
        recursive=recursive,
        extensions=extensions,
        experiment=experiment_name
    )


def discover_dataset_files(
    config: Dict[str, Any],
    dataset_name: str,
    base_directory: Union[str, Path],
    pattern: str = "*.*",
    recursive: bool = True,
    extensions: Optional[List[str]] = None
) -> List[str]:
    """
    Discover files related to a specific dataset.
    
    This uses the dataset's date-vial definitions to find relevant files.
    
    Args:
        config: The loaded configuration dictionary
        dataset_name: Name of the dataset to use for discovery
        base_directory: Base directory to search in (often the major_data_directory)
        pattern: File pattern to match (glob format), defaults to all files
        recursive: If True, search recursively through subdirectories
        extensions: Optional list of file extensions to filter by
        
    Returns:
        List of file paths relevant to the dataset
        
    Raises:
        KeyError: If the dataset does not exist in the configuration
    """
    # Get dataset information
    dataset_info = get_dataset_info(config, dataset_name)
    
    # Get the dates for this dataset
    dates = dataset_info.get("dates_vials", {}).keys()
    
    # Collect all date-specific directories to search in
    search_dirs = []
    for date in dates:
        date_dir = Path(base_directory) / str(date)
        if date_dir.exists():
            search_dirs.append(str(date_dir))
    
    # If no search directories were found, use the base directory
    if not search_dirs:
        search_dirs = [base_directory]
    
    # Get project-level ignore patterns (no experiment-specific ones)
    ignore_patterns = get_ignore_patterns(config)
    
    # Get project-level mandatory substrings (no experiment-specific ones)
    mandatory_substrings = get_mandatory_substrings(config)
    
    # Use the core discovery function with dataset-specific directories
    return discover_files(
        directory=search_dirs,
        pattern=pattern,
        recursive=recursive,
        extensions=extensions,
        ignore_patterns=ignore_patterns,
        mandatory_substrings=mandatory_substrings
    )
