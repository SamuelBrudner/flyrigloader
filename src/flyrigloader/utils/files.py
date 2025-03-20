"""
File handling utilities for safe operations on files and directories.

This module provides standardized functions for common file operations 
like loading YAML files and handling error conditions consistently.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Callable, Union

import yaml
from loguru import logger

T = TypeVar('T')


def safe_load_yaml(file_path: Union[str, Path], default: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Safely load a YAML file with error handling.
    
    Args:
        file_path: Path to the YAML file to load
        default: Default value to return if file is not found or is empty (defaults to empty dict)
        
    Returns:
        Dictionary containing the loaded YAML data, or default if file not found or empty
    """
    if default is None:
        default = {}
        
    try:
        with open(file_path, 'r') as f:
            result = yaml.safe_load(f)
            return result if result is not None else default
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
        return default
    except yaml.YAMLError as e:
        logger.warning(f"Invalid YAML in {file_path}: {e}")
        return default
    except Exception as e:
        logger.warning(f"Error loading {file_path}: {e}")
        return default


def ensure_file_exists(file_path: Union[str, Path]) -> bool:
    """
    Check if a file exists.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if the file exists, False otherwise
    """
    return os.path.isfile(Path(file_path))


def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory to ensure exists
        
    Returns:
        Path object pointing to the directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def with_logging(
    function: Callable,
    args: List[Any] = None,
    kwargs: Dict[str, Any] = None,
    error_message: str = "Operation failed",
    default_value: Any = None
) -> Any:
    """
    Execute a function with standardized error logging.
    
    Args:
        function: The function to execute
        args: Positional arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        error_message: Message to log if function raises an exception
        default_value: Value to return if function raises an exception
        
    Returns:
        The result of the function, or default_value if an exception is raised
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
        
    try:
        return function(*args, **kwargs)
    except Exception as e:
        logger.warning(f"{error_message}: {e}")
        return default_value


def check_and_log(
    condition: bool,
    warning_message: str,
    default_value: T,
    log_level: str = "warning"
) -> Optional[T]:
    """
    Check a condition and log a message if it fails.
    
    This is a utility function for the common pattern:
    if not condition:
        logger.warning(message)
        return default_value
    
    Args:
        condition: The condition to check
        warning_message: Message to log if condition is False
        default_value: Value to return if condition is False
        log_level: Log level to use (default: "warning")
        
    Returns:
        None if condition is False, otherwise doesn't return (allows for use in if statements)
    """
    if not condition:
        if log_level == "warning":
            logger.warning(warning_message)
        elif log_level == "error":
            logger.error(warning_message)
        elif log_level == "info":
            logger.info(warning_message)
        elif log_level == "debug":
            logger.debug(warning_message)
        
        return default_value
    return None


def list_directory_with_pattern(
    directory: Union[str, Path], 
    pattern: str = "*", 
    recursive: bool = False
) -> List[Path]:
    """
    List files in a directory that match a pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        recursive: Whether to search recursively
        
    Returns:
        List of Path objects that match the pattern
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []
        
    if recursive:
        return list(directory.glob(f"**/{pattern}"))
    else:
        return list(directory.glob(pattern))