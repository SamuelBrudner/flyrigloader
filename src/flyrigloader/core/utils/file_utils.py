"""
File handling utilities for safe operations on files and directories.

This module provides standardized functions for common file operations 
like loading YAML files and handling error conditions consistently.

Note on separation of concerns:
- This module focuses on basic file system operations and simple file format handling
- For data-specific file operations (CSV, Parquet, etc.), use the readers/formats module
- For DataFrame-related operations with lineage tracking, use pipeline/data_assembly
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import glob
from loguru import logger

import yaml
from .path_utils import PathLike, ensure_path

T = TypeVar('T')


def safe_load_yaml(file_path: PathLike, default: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Safely load a YAML file with explicit error handling.
    
    Args:
        file_path: Path to the YAML file to load
        default: Default value to return if file is not found or is empty (defaults to empty dict)
        
    Returns:
        Dictionary containing the loaded YAML data, or default if file not found or empty
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file cannot be parsed as YAML
        
    Example:
        >>> try:
        >>>     data = safe_load_yaml("config.yaml")
        >>>     # Use data
        >>> except Exception as e:
        >>>     logger.error(f"Failed to load YAML: {str(e)}")
    """
    try:
        if default is None:
            default = {}
        
        path = ensure_path(file_path)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"YAML file not found: {path}")
            
        with open(path, 'r') as f:
            result = yaml.safe_load(f)
            if result is None:
                logger.warning(f"YAML file is empty or contains only comments: {path}")
                return default
            return result
    except FileNotFoundError:
        # Re-raise the FileNotFoundError
        raise
    except Exception as e:
        logger.error(f"Failed to load YAML file {file_path}: {str(e)}")
        raise ValueError(f"Failed to load YAML file: {str(e)}") from e


def ensure_file_exists(file_path: PathLike) -> bool:
    """
    Check if a file exists.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if the file exists, False otherwise
    """
    try:
        path = ensure_path(file_path)
        return path.exists() and path.is_file()
    except Exception as e:
        logger.error(f"Failed to check if file exists: {str(e)}")
        return False


def ensure_directory(directory: PathLike) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory to ensure exists
        
    Returns:
        Path object pointing to the directory
        
    Raises:
        ValueError: If the path exists but is a file, or if the directory cannot be created
    """
    try:
        dir_path = ensure_path(directory)
        
        # Check if path exists and is a file
        if dir_path.exists() and not dir_path.is_dir():
            raise ValueError(f"Path exists but is not a directory: {dir_path}")
            
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    except Exception as e:
        logger.error(f"Failed to ensure directory exists: {str(e)}")
        raise ValueError(f"Failed to ensure directory exists: {str(e)}") from e


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
    directory: PathLike, 
    pattern: str = "*", 
    recursive: bool = False
) -> List[Path]:
    """
    List files in a directory that match a pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match (e.g., "*.pkl")
        recursive: Whether to search recursively
        
    Returns:
        List of Path objects that match the pattern
        
    Raises:
        ValueError: If the directory does not exist or is not a directory
    """
    try:
        # Convert to Path object
        dir_path = ensure_path(directory)
        
        # Check if directory exists
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {dir_path}")
            return []
            
        if not dir_path.is_dir():
            logger.warning(f"Path is not a directory: {dir_path}")
            return []
        
        # Determine the search pattern based on recursive flag
        if recursive:
            # Use Path.glob with recursive ** pattern
            search_pattern = f"**/{pattern}" if pattern != "*" else "**/*"
            matching_files = list(dir_path.glob(search_pattern))
            # Filter out directories
            matching_files = [f for f in matching_files if f.is_file()]
        else:
            # Use Path.glob with non-recursive pattern
            matching_files = [f for f in dir_path.glob(pattern) if f.is_file()]
        
        logger.debug(f"Found {len(matching_files)} files matching '{pattern}' in {dir_path}")
        return matching_files
    except Exception as e:
        logger.error(f"Failed to list directory with pattern: {str(e)}")
        raise ValueError(f"Failed to list directory with pattern: {str(e)}") from e


def read_file_content(file_path: PathLike) -> str:
    """
    Read content from a file as text.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        Content of the file as a string
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file cannot be read
    """
    try:
        path = ensure_path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        return path.read_text()
    except FileNotFoundError:
        # Re-raise the FileNotFoundError
        raise
    except Exception as e:
        logger.error(f"Failed to read file content: {str(e)}")
        raise ValueError(f"Failed to read file content: {str(e)}") from e


def write_file_content(file_path: PathLike, content: str, create_parents: bool = True) -> None:
    """
    Write content to a file as text.
    
    Args:
        file_path: Path to the file to write
        content: Content to write
        create_parents: Whether to create parent directories if they don't exist
        
    Raises:
        ValueError: If the file cannot be written
    """
    try:
        path = ensure_path(file_path)
        
        if create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
            
        path.write_text(content)
    except Exception as e:
        logger.error(f"Failed to write to file: {str(e)}")
        raise ValueError(f"Failed to write to file: {str(e)}") from e


def save_yaml(file_path: PathLike, data: Dict[str, Any], create_parents: bool = True) -> None:
    """
    Save data to a YAML file.
    
    Args:
        file_path: Path to the file to write
        data: Data to write as YAML
        create_parents: Whether to create parent directories if they don't exist
        
    Raises:
        ValueError: If the file cannot be written
    """
    try:
        path = ensure_path(file_path)
        
        if create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
            
        with open(path, 'w') as f:
            yaml.dump(data, f)
    except Exception as e:
        logger.error(f"Failed to save YAML to file: {str(e)}")
        raise ValueError(f"Failed to save YAML to file: {str(e)}") from e
