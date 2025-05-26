"""
High-level API for flyrigloader.

This module provides simple entry points for external projects to use flyrigloader
functionality without having to directly import from multiple submodules.
"""
from pathlib import Path
import copy
from typing import Dict, List, Any, Optional, Union


MISSING_DATA_DIR_ERROR = (
    "No data directory specified. Either provide base_directory parameter "
    "or ensure 'major_data_directory' is set in config."
)


def _resolve_base_directory(
    config: Dict[str, Any], base_directory: Optional[Union[str, Path]]
) -> Union[str, Path]:
    """Return the resolved base directory or raise ValueError."""
    if base_directory is None:
        base_directory = (
            config.get("project", {})
            .get("directories", {})
            .get("major_data_directory")
        )

    if not base_directory:
        raise ValueError(MISSING_DATA_DIR_ERROR)
    return base_directory

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
from flyrigloader.discovery.stats import get_file_stats
from flyrigloader.io.pickle import (
    read_pickle_any_format,
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
from flyrigloader.utils.paths import (
    get_relative_path,
    get_absolute_path,
    check_file_exists,
    ensure_directory_exists,
    find_common_base_directory
)


def load_experiment_files(
    config_path: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    experiment_name: str = "",
    base_directory: Optional[Union[str, Path]] = None,
    pattern: str = "*.*",
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
    extract_metadata: bool = False,
    parse_dates: bool = False
) -> Union[List[str], Dict[str, Dict[str, Any]]]:
    """
    High-level function to load files for a specific experiment.
    
    Args:
        config_path: Path to the YAML configuration file
        config: Pre-loaded configuration dictionary (can be a Kedro-style parameters dictionary)
        experiment_name: Name of the experiment to load files for
        base_directory: Optional override for the data directory (if not specified, uses config)
        pattern: File pattern to search for in glob format (e.g., "*.csv", "data_*.pkl")
        recursive: Whether to search recursively (defaults to True)
        extensions: Optional list of file extensions to filter by
        extract_metadata: If True, extract metadata from filenames using config patterns
        parse_dates: If True, attempt to parse dates from filenames
        
    Returns:
        If extract_metadata or parse_dates is True: Dictionary mapping file paths to metadata
        Otherwise: List of file paths for the experiment
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        KeyError: If the experiment doesn't exist in the config
        ValueError: If neither config_path nor config is provided, or if both are provided
    """
    # Validate config parameters
    if (config_path is None and config is None) or (config_path is not None and config is not None):
        raise ValueError("Exactly one of 'config_path' or 'config' must be provided")
    
    # Load the configuration if path is provided, otherwise make a deep copy of the config
    if config_path is not None:
        config_dict = load_config(config_path)
    else:
        # Make a deep copy to avoid modifying the original dictionary
        config_dict = copy.deepcopy(config)
    
    # Determine the data directory
    base_directory = _resolve_base_directory(config_dict, base_directory)
    
    # Discover experiment files
    return discover_experiment_files(
        config=config_dict,
        experiment_name=experiment_name,
        base_directory=base_directory,
        pattern=pattern,
        recursive=recursive,
        extensions=extensions,
        extract_metadata=extract_metadata,
        parse_dates=parse_dates
    )


def load_dataset_files(
    config_path: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    dataset_name: str = "",
    base_directory: Optional[Union[str, Path]] = None,
    pattern: str = "*.*",
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
    extract_metadata: bool = False,
    parse_dates: bool = False
) -> Union[List[str], Dict[str, Dict[str, Any]]]:
    """
    High-level function to load files for a specific dataset.
    
    Args:
        config_path: Path to the YAML configuration file
        config: Pre-loaded configuration dictionary (can be a Kedro-style parameters dictionary)
        dataset_name: Name of the dataset to load files for
        base_directory: Optional override for the data directory (if not specified, uses config)
        pattern: File pattern to search for in glob format (e.g., "*.csv", "data_*.pkl")
        recursive: Whether to search recursively (defaults to True)
        extensions: Optional list of file extensions to filter by
        extract_metadata: If True, extract metadata from filenames using config patterns
        parse_dates: If True, attempt to parse dates from filenames
        
    Returns:
        If extract_metadata or parse_dates is True: Dictionary mapping file paths to metadata
        Otherwise: List of file paths for the dataset
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        KeyError: If the dataset doesn't exist in the config
        ValueError: If neither config_path nor config is provided, or if both are provided
    """
    # Validate config parameters
    if (config_path is None and config is None) or (config_path is not None and config is not None):
        raise ValueError("Exactly one of 'config_path' or 'config' must be provided")
    
    # Load the configuration if path is provided, otherwise make a deep copy of the config
    if config_path is not None:
        config_dict = load_config(config_path)
    else:
        # Make a deep copy to avoid modifying the original dictionary
        config_dict = copy.deepcopy(config)
    
    # Determine the data directory
    base_directory = _resolve_base_directory(config_dict, base_directory)
    
    # Discover dataset files
    return discover_dataset_files(
        config=config_dict,
        dataset_name=dataset_name,
        base_directory=base_directory,
        pattern=pattern,
        recursive=recursive,
        extensions=extensions,
        extract_metadata=extract_metadata,
        parse_dates=parse_dates
    )


def get_experiment_parameters(
    config_path: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    experiment_name: str = ""
) -> Dict[str, Any]:
    """
    Get parameters for a specific experiment.
    
    Args:
        config_path: Path to the YAML configuration file
        config: Pre-loaded configuration dictionary (can be a Kedro-style parameters dictionary)
        experiment_name: Name of the experiment to get parameters for
        
    Returns:
        Dictionary containing experiment parameters
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        KeyError: If the experiment doesn't exist in the config
        ValueError: If neither config_path nor config is provided, or if both are provided
    """
    # Validate config parameters
    if (config_path is None and config is None) or (config_path is not None and config is not None):
        raise ValueError("Exactly one of 'config_path' or 'config' must be provided")
    
    # Load the configuration if path is provided, otherwise make a deep copy of the config
    if config_path is not None:
        config_dict = load_config(config_path)
    else:
        # Make a deep copy to avoid modifying the original dictionary
        config_dict = copy.deepcopy(config)
    
    # Get experiment info
    experiment_info = get_experiment_info(config_dict, experiment_name)
    
    # Extract parameters
    return experiment_info.get("parameters", {})


def get_dataset_parameters(
    config_path: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    dataset_name: str = "",
) -> Dict[str, Any]:
    """Get parameters for a specific dataset."""

    if (config_path is None and config is None) or (config_path is not None and config is not None):
        raise ValueError("Exactly one of 'config_path' or 'config' must be provided")

    if config_path is not None:
        config_dict = load_config(config_path)
    else:
        config_dict = copy.deepcopy(config)

    dataset_info = get_dataset_info(config_dict, dataset_name)
    return dataset_info.get("parameters", {})


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


#
# File and path utilities
#
# Note: For standard path operations, consider using Python's pathlib directly:
#  - Path.name - Get filename without directory (instead of get_file_name)
#  - Path.suffix - Get file extension with dot (instead of get_extension)
#  - Path.parent - Get parent directory (instead of get_parent_dir)
#  - Path.resolve() - Normalize path (instead of normalize_file_path)
#

def get_file_statistics(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive statistics for a file.
    
    Args:
        path: Path to the file
        
    Returns:
        Dictionary containing file statistics (size, modification time, etc.)
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    return get_file_stats(path)


def ensure_dir_exists(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object pointing to the directory
    """
    return ensure_directory_exists(path)


def check_if_file_exists(path: Union[str, Path]) -> bool:
    """
    Check if a file exists.
    
    Args:
        path: Path to check
        
    Returns:
        True if the file exists, False otherwise
    """
    return check_file_exists(path)


def get_path_relative_to(path: Union[str, Path], base_dir: Union[str, Path]) -> Path:
    """
    Get a path relative to a base directory.
    
    Args:
        path: Path to convert
        base_dir: Base directory
        
    Returns:
        Relative path
        
    Raises:
        ValueError: If the path is not within the base directory
    """
    return get_relative_path(path, base_dir)


def get_path_absolute(path: Union[str, Path], base_dir: Union[str, Path]) -> Path:
    """
    Convert a relative path to an absolute path.
    
    Args:
        path: Relative path
        base_dir: Base directory
        
    Returns:
        Absolute path
    """
    return get_absolute_path(path, base_dir)


def get_common_base_dir(paths: List[Union[str, Path]]) -> Optional[Path]:
    """
    Find the common base directory for a list of paths.
    
    Args:
        paths: List of paths to analyze
        
    Returns:
        Common base directory or None if no common base can be found
    """
    return find_common_base_directory(paths)
