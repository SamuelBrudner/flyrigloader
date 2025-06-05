"""
YAML configuration handling utilities.

Functions for loading, parsing, and accessing configuration data from YAML files.
"""
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import yaml


def validate_config_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a configuration dictionary to ensure it has the expected structure.
    
    This function is used to validate Kedro-style parameters dictionaries directly
    passed to flyrigloader functions.
    
    Args:
        config: The configuration dictionary to validate
        
    Returns:
        The validated configuration dictionary
        
    Raises:
        ValueError: If the configuration dictionary is invalid
    """
    # Perform basic structure validation
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")
    
    # Check for required top-level keys (minimal validation)
    required_sections = []
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section in configuration: {section}")

    # Validate dates_vials structure within datasets if present
    if "datasets" in config:
        datasets = config["datasets"]
        if not isinstance(datasets, dict):
            raise ValueError("'datasets' must be a dictionary")
        for name, ds in datasets.items():
            if not isinstance(ds, dict):
                continue
            if "dates_vials" in ds:
                dates_vials = ds["dates_vials"]
                if not isinstance(dates_vials, dict):
                    raise ValueError(
                        f"Dataset '{name}' dates_vials must be a dictionary"
                    )
                for key, value in dates_vials.items():
                    if not isinstance(key, str):
                        raise ValueError(
                            f"Dataset '{name}' dates_vials key '{key}' must be a string"
                        )
                    if not isinstance(value, list):
                        raise ValueError(
                            f"Dataset '{name}' dates_vials value for '{key}' must be a list"
                        )

    return config


def load_config(config_path_or_dict: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Load a YAML configuration file or validate a pre-loaded configuration dictionary.
    
    Args:
        config_path_or_dict: Path to the YAML configuration file or a pre-loaded dictionary
        
    Returns:
        Dictionary containing the configuration data
        
    Raises:
        FileNotFoundError: If the configuration file does not exist
        yaml.YAMLError: If the YAML file is invalid
        ValueError: If the input is neither a valid path nor a dictionary
    """
    # If input is already a dictionary (Kedro-style parameters)
    if isinstance(config_path_or_dict, dict):
        return validate_config_dict(config_path_or_dict)
    
    # Check for invalid input types
    if not isinstance(config_path_or_dict, (str, Path)):
        raise ValueError(f"Invalid input type: {type(config_path_or_dict)}. Expected a string, Path, or dictionary.")
    
    # Otherwise treat as a path
    config_path = Path(config_path_or_dict)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML configuration: {e}") from e


def get_ignore_patterns(
    config: Dict[str, Any],
    experiment: Optional[str] = None
) -> List[str]:
    """
    Get ignore patterns from the configuration.
    
    This combines project-level ignore patterns with any experiment-specific patterns.
    The patterns from the config are converted to glob patterns if they don't already
    contain wildcard characters.
    
    Args:
        config: The loaded configuration dictionary
        experiment: Optional experiment name to get experiment-specific patterns
        
    Returns:
        List of glob-formatted ignore patterns
    """
    # Start with project-level ignore patterns
    patterns = []
    if "project" in config and "ignore_substrings" in config["project"]:
        # Convert simple substrings to glob patterns
        patterns.extend(
            _convert_to_glob_pattern(pattern) 
            for pattern in config["project"]["ignore_substrings"]
        )
    
    # Add experiment-specific patterns if specified
    if experiment and "experiments" in config and experiment in config["experiments"]:
        experiment_config = config["experiments"][experiment]
        if "filters" in experiment_config and "ignore_substrings" in experiment_config["filters"]:
            # Convert experiment-specific substrings to glob patterns
            patterns.extend(
                _convert_to_glob_pattern(pattern) 
                for pattern in experiment_config["filters"]["ignore_substrings"]
            )
    
    return patterns


def _convert_to_glob_pattern(pattern: str) -> str:
    """
    Convert a simple substring pattern to a glob pattern if needed.
    
    If the pattern already has glob wildcards (* or ?), leave it as is.
    Otherwise, wrap it with * on both sides for substring matching.
    
    Args:
        pattern: The original pattern string
        
    Returns:
        A glob pattern that will match the original substring
    """
    # Return pattern as-is if it already contains wildcards, otherwise wrap with asterisks
    return pattern if ('*' in pattern or '?' in pattern) else f"*{pattern}*"


def get_mandatory_substrings(
    config: Dict[str, Any],
    experiment: Optional[str] = None
) -> List[str]:
    """
    Get mandatory substrings from the configuration.
    
    This combines project-level mandatory substrings with any experiment-specific substrings.
    
    Args:
        config: The loaded configuration dictionary
        experiment: Optional experiment name to get experiment-specific substrings
        
    Returns:
        List of mandatory substrings
    """
    # Start with project-level mandatory substrings (if any)
    substrings = []
    if "project" in config and "mandatory_experiment_strings" in config["project"]:
        substrings.extend(config["project"]["mandatory_experiment_strings"])
    
    # Add experiment-specific mandatory substrings if specified
    if experiment and "experiments" in config and experiment in config["experiments"]:
        experiment_config = config["experiments"][experiment]
        if "filters" in experiment_config and "mandatory_experiment_strings" in experiment_config["filters"]:
            substrings.extend(experiment_config["filters"]["mandatory_experiment_strings"])
    
    return substrings


def get_dataset_info(
    config: Dict[str, Any],
    dataset_name: str
) -> Dict[str, Any]:
    """
    Get information about a specific dataset from the configuration.
    
    Args:
        config: The loaded configuration dictionary
        dataset_name: Name of the dataset to retrieve
        
    Returns:
        Dictionary with dataset information
        
    Raises:
        KeyError: If the dataset does not exist in the configuration
    """
    if "datasets" not in config or dataset_name not in config["datasets"]:
        raise KeyError(f"Dataset '{dataset_name}' not found in configuration")
    
    return config["datasets"][dataset_name]


def get_experiment_info(
    config: Dict[str, Any],
    experiment_name: str
) -> Dict[str, Any]:
    """
    Get information about a specific experiment from the configuration.
    
    Args:
        config: The loaded configuration dictionary
        experiment_name: Name of the experiment to retrieve
        
    Returns:
        Dictionary with experiment information
        
    Raises:
        KeyError: If the experiment does not exist in the configuration
    """
    if "experiments" not in config or experiment_name not in config["experiments"]:
        raise KeyError(f"Experiment '{experiment_name}' not found in configuration")
    
    return config["experiments"][experiment_name]


def get_all_experiment_names(config: Dict[str, Any]) -> List[str]:
    """Return a list of experiment names defined in the configuration."""
    return list(config.get("experiments", {}).keys())


def get_extraction_patterns(
    config: Dict[str, Any],
    experiment: Optional[str] = None,
    dataset_name: Optional[str] = None
) -> Optional[List[str]]:
    """
    Get patterns for extracting metadata from filenames.
    
    This function combines:
    1. Project-level extraction patterns
    2. Experiment-specific extraction patterns (if experiment is provided)
    3. Dataset-specific extraction patterns (if dataset_name is provided)
    
    Only one of experiment or dataset_name should be provided.
    
    Args:
        config: The loaded configuration dictionary
        experiment: Optional experiment name to get extraction patterns for
        dataset_name: Optional dataset name to get extraction patterns for
        
    Returns:
        List of regex patterns for extracting metadata, or None if no patterns are defined
    """
    patterns = []
    
    # Get project-level extraction patterns
    if "project" in config and "extraction_patterns" in config["project"]:
        patterns.extend(config["project"]["extraction_patterns"])
    
    # Get experiment-specific extraction patterns
    if experiment and "experiments" in config and experiment in config["experiments"]:
        experiment_config = config["experiments"][experiment]
        if "metadata" in experiment_config and "extraction_patterns" in experiment_config["metadata"]:
            patterns.extend(experiment_config["metadata"]["extraction_patterns"])
    
    # Get dataset-specific extraction patterns
    if dataset_name and "datasets" in config and dataset_name in config["datasets"]:
        dataset_config = config["datasets"][dataset_name]
        if "metadata" in dataset_config and "extraction_patterns" in dataset_config["metadata"]:
            patterns.extend(dataset_config["metadata"]["extraction_patterns"])
    
    # Return patterns if not empty, or None
    return patterns or None


def get_all_dataset_names(config: Dict[str, Any]) -> List[str]:
    """
    Return a list of all dataset names defined in the configuration.

    Enhanced with validation and logging for test observability.

    Args:
        config: The loaded configuration dictionary.

    Returns:
        List of dataset names. Returns an empty list if no datasets are defined.
        
    Raises:
        ValueError: If configuration structure is invalid
    """
    logger.debug("Retrieving all dataset names from configuration")
    
    if not isinstance(config, dict):
        error_msg = "Configuration must be a dictionary"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    datasets_section = config.get("datasets", {})
    if not isinstance(datasets_section, dict):
        error_msg = "'datasets' section must be a dictionary"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    dataset_names = list(datasets_section.keys())
    logger.info(f"Found {len(dataset_names)} datasets: {dataset_names}")
    return dataset_names


# === Test-Specific Entry Points for Enhanced Testing ===

def create_test_yaml_loader(mock_safe_load: Callable) -> YAMLLoaderProtocol:
    """
    Create a test-specific YAML loader with custom behavior.
    
    This function provides a test entry point for comprehensive mocking scenarios,
    enabling controlled YAML loading behavior during test execution per TST-REF-003.
    
    Args:
        mock_safe_load: Callable that mimics yaml.safe_load behavior
        
    Returns:
        YAMLLoaderProtocol implementation for testing
    """
    class TestYAMLLoader:
        def safe_load(self, file_handle) -> Any:
            return mock_safe_load(file_handle)
    
    return TestYAMLLoader()


def create_test_file_system(
    mock_exists: Callable[[Path], bool],
    mock_open: Callable[[Path, str], Any]
) -> FileSystemProtocol:
    """
    Create a test-specific file system with mocked operations.
    
    This function provides a test entry point for comprehensive file system mocking,
    enabling controlled file existence and content scenarios during test execution.
    
    Args:
        mock_exists: Callable that mimics Path.exists() behavior
        mock_open: Callable that mimics open() behavior
        
    Returns:
        FileSystemProtocol implementation for testing
    """
    class TestFileSystem:
        def exists(self, path: Path) -> bool:
            return mock_exists(path)
        
        def open(self, path: Path, mode: str = 'r'):
            return mock_open(path, mode)
    
    return TestFileSystem()


def create_test_validator(
    mock_validate_structure: Optional[Callable] = None,
    mock_validate_dates_vials: Optional[Callable] = None
) -> ConfigValidatorProtocol:
    """
    Create a test-specific configuration validator with custom validation behavior.
    
    This function provides a test entry point for comprehensive validation mocking,
    enabling controlled validation behavior and error scenarios during test execution.
    
    Args:
        mock_validate_structure: Optional callable for structure validation
        mock_validate_dates_vials: Optional callable for dates_vials validation
        
    Returns:
        ConfigValidatorProtocol implementation for testing
    """
    class TestValidator:
        def validate_structure(self, config: Dict[str, Any]) -> Dict[str, Any]:
            if mock_validate_structure:
                return mock_validate_structure(config)
            return config
        
        def validate_dates_vials(self, config: Dict[str, Any]) -> None:
            if mock_validate_dates_vials:
                mock_validate_dates_vials(config)
    
    return TestValidator()
