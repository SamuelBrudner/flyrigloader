"""
High-level API for flyrigloader.

This module provides simple entry points for external projects to use flyrigloader
functionality without having to directly import from multiple submodules.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from flyrigloader.config.yaml_config import (
    load_config,
    get_ignore_patterns,
    get_mandatory_substrings,
    get_dataset_info,
    get_experiment_info
)
from flyrigloader.config.discovery import (
    discover_files_with_config,
    discover_experiment_files,
    discover_dataset_files
)
from flyrigloader.discovery.files import discover_files
from flyrigloader.io.pickle import (
    read_pickle_any_format,
    make_dataframe_from_matrix,
    make_dataframe_from_config
)
from flyrigloader.io.column_models import (
    ColumnConfig,
    ColumnConfigDict,
    ColumnDimension,
    get_config_from_source,
    get_default_config_path,
    load_column_config
)


def load_experiment_files(
    config_path: Union[str, Path],
    experiment_name: str,
    data_directory: Optional[Union[str, Path]] = None,
    file_pattern: str = "*.*",
    recursive: bool = True,
    extensions: Optional[List[str]] = None
) -> List[str]:
    """
    High-level function to load files for a specific experiment.
    
    Args:
        config_path: Path to the YAML configuration file
        experiment_name: Name of the experiment to load files for
        data_directory: Optional override for the data directory (if not specified, uses config)
        file_pattern: File pattern to search for (defaults to all files)
        recursive: Whether to search recursively (defaults to True)
        extensions: Optional list of file extensions to filter by
        
    Returns:
        List of file paths for the experiment
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        KeyError: If the experiment doesn't exist in the config
    """
    # Load the configuration
    config = load_config(config_path)
    
    # Determine the data directory
    if data_directory is None:
        if "project" in config and "directories" in config["project"]:
            data_directory = config["project"]["directories"].get("major_data_directory")
            
        if not data_directory:
            raise ValueError(
                "No data directory specified. Either provide data_directory parameter "
                "or ensure 'major_data_directory' is set in config."
            )
    
    # Discover experiment files
    return discover_experiment_files(
        config=config,
        experiment_name=experiment_name,
        base_directory=data_directory,
        pattern=file_pattern,
        recursive=recursive,
        extensions=extensions
    )


def load_dataset_files(
    config_path: Union[str, Path],
    dataset_name: str,
    data_directory: Optional[Union[str, Path]] = None,
    file_pattern: str = "*.*",
    recursive: bool = True,
    extensions: Optional[List[str]] = None
) -> List[str]:
    """
    High-level function to load files for a specific dataset.
    
    Args:
        config_path: Path to the YAML configuration file
        dataset_name: Name of the dataset to load files for
        data_directory: Optional override for the data directory (if not specified, uses config)
        file_pattern: File pattern to search for (defaults to all files)
        recursive: Whether to search recursively (defaults to True)
        extensions: Optional list of file extensions to filter by
        
    Returns:
        List of file paths for the dataset
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        KeyError: If the dataset doesn't exist in the config
    """
    # Load the configuration
    config = load_config(config_path)
    
    # Determine the data directory
    if data_directory is None:
        if "project" in config and "directories" in config["project"]:
            data_directory = config["project"]["directories"].get("major_data_directory")
            
        if not data_directory:
            raise ValueError(
                "No data directory specified. Either provide data_directory parameter "
                "or ensure 'major_data_directory' is set in config."
            )
    
    # Discover dataset files
    return discover_dataset_files(
        config=config,
        dataset_name=dataset_name,
        base_directory=data_directory,
        pattern=file_pattern,
        recursive=recursive,
        extensions=extensions
    )


def get_experiment_parameters(
    config_path: Union[str, Path],
    experiment_name: str
) -> Dict[str, Any]:
    """
    Get parameters for a specific experiment.
    
    Args:
        config_path: Path to the YAML configuration file
        experiment_name: Name of the experiment to get parameters for
        
    Returns:
        Dictionary containing experiment parameters
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        KeyError: If the experiment doesn't exist in the config
    """
    # Load the configuration
    config = load_config(config_path)
    
    # Get experiment info
    experiment_info = get_experiment_info(config, experiment_name)
    
    # Extract parameters
    return experiment_info.get("parameters", {})


def process_experiment_data(
    data_path: Union[str, Path],
    column_config_path: Optional[Union[str, Path, Dict[str, Any], ColumnConfigDict]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Process experimental data using column configuration.
    
    Args:
        data_path: Path to the pickle file containing experimental data
        column_config_path: Path to column configuration file, configuration dictionary,
                            or ColumnConfigDict instance. If None, uses default configuration.
        metadata: Optional dictionary of metadata to add to the DataFrame
        
    Returns:
        DataFrame with processed experimental data
        
    Raises:
        FileNotFoundError: If the data or config file doesn't exist
        ValueError: If required columns are missing from the data
    """
    # Read the experimental data
    exp_matrix = read_pickle_any_format(data_path)
    
    # Create DataFrame using column configuration
    return make_dataframe_from_config(
        exp_matrix=exp_matrix,
        config_source=column_config_path,
        metadata=metadata
    )


def get_default_column_config() -> ColumnConfigDict:
    """
    Get the default column configuration.
    
    Returns:
        ColumnConfigDict with the default configuration
    """
    # Load the default configuration
    return get_config_from_source(None)
