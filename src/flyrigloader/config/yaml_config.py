"""
YAML configuration handling utilities.

Functions for loading, parsing, and accessing configuration data from YAML files.
"""
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file into a dictionary.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration data
        
    Raises:
        FileNotFoundError: If the configuration file does not exist
        yaml.YAMLError: If the YAML file is invalid
    """
    config_path = Path(config_path)
    
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
