"""Filesystem and path related public API entry points."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from flyrigloader import logger
from flyrigloader.config.models import LegacyConfigAdapter
from flyrigloader.exceptions import FlyRigLoaderError

from .dependencies import DefaultDependencyProvider, get_dependency_provider

MISSING_DATA_DIR_ERROR = (
    "No data directory specified. Either provide base_directory parameter "
    "or ensure 'major_data_directory' is set in config."
)

def _resolve_base_directory(
    config: Union[Dict[str, Any], LegacyConfigAdapter],
    base_directory: Optional[Union[str, Path]],
    operation_name: str
) -> Union[str, Path]:
    """
    Resolve base directory following documented precedence order with comprehensive audit logging.
    
    Precedence order (first match wins):
    1. Explicit base_directory parameter (function argument)
    2. Configuration-defined major_data_directory 
    3. Environment variable FLYRIGLOADER_DATA_DIR (for CI/CD scenarios)
    
    Each resolution step is logged at appropriate levels for scientific reproducibility
    and debugging workflows. This ensures transparent data path resolution with
    clear audit trails.
    
    Args:
        config: Configuration dictionary or LegacyConfigAdapter
        base_directory: Optional explicit override for base directory (highest precedence)
        operation_name: Name of operation for error context and logging
        
    Returns:
        Resolved base directory path
        
    Raises:
        FlyRigLoaderError: If no valid base directory can be resolved
    """
    logger.debug(f"Starting base directory resolution for {operation_name}")
    logger.debug(f"Resolution precedence: 1) explicit parameter, 2) config major_data_directory, 3) FLYRIGLOADER_DATA_DIR env var")
    
    resolved_directory = None
    resolution_source = None
    
    # Precedence 1: Explicit base_directory parameter (highest priority)
    if base_directory is not None:
        resolved_directory = base_directory
        resolution_source = "explicit function parameter"
        logger.info(f"Using explicit base_directory parameter: {base_directory}")
        logger.debug(f"Resolution source: {resolution_source} (precedence 1)")
    
    # Precedence 2: Configuration-defined major_data_directory
    if resolved_directory is None:
        logger.debug("No explicit base_directory provided, checking config for major_data_directory")
        
        # Support both dict and LegacyConfigAdapter
        if hasattr(config, 'get'):
            # Dict-like access (works for both dict and LegacyConfigAdapter)
            config_directory = (
                config.get("project", {})
                .get("directories", {})
                .get("major_data_directory")
            )
        else:
            logger.warning(f"Unexpected config type for {operation_name}: {type(config)}")
            config_directory = None
        
        if config_directory:
            resolved_directory = config_directory
            resolution_source = "configuration major_data_directory"
            logger.info(f"Using major_data_directory from config: {config_directory}")
            logger.debug(f"Resolution source: {resolution_source} (precedence 2)")
        else:
            logger.debug("No major_data_directory found in config")
    
    # Precedence 3: Environment variable FLYRIGLOADER_DATA_DIR (lowest priority)
    if resolved_directory is None:
        logger.debug("No config major_data_directory found, checking FLYRIGLOADER_DATA_DIR environment variable")
        env_directory = os.environ.get("FLYRIGLOADER_DATA_DIR")
        
        if env_directory:
            resolved_directory = env_directory
            resolution_source = "FLYRIGLOADER_DATA_DIR environment variable"
            logger.info(f"Using FLYRIGLOADER_DATA_DIR environment variable: {env_directory}")
            logger.debug(f"Resolution source: {resolution_source} (precedence 3)")
        else:
            logger.debug("No FLYRIGLOADER_DATA_DIR environment variable found")

    # Validation: Ensure we have a resolved directory
    if not resolved_directory:
        error_msg = (
            f"No data directory specified for {operation_name}. "
            "Tried all resolution methods in precedence order:\n"
            "1. Explicit 'base_directory' parameter (not provided)\n"
            "2. Configuration 'project.directories.major_data_directory' (not found)\n"
            "3. Environment variable 'FLYRIGLOADER_DATA_DIR' (not set)\n\n"
            "Please provide a data directory using one of these methods:\n"
            "- Pass base_directory parameter to the function\n"
            "- Set 'major_data_directory' in your config:\n"
            "  project:\n    directories:\n      major_data_directory: /path/to/data\n"
            "- Set environment variable: export FLYRIGLOADER_DATA_DIR=/path/to/data"
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    # Convert to Path for validation and final logging
    resolved_path = Path(resolved_directory)
    
    # Existence check with appropriate logging level
    if resolved_path.exists():
        logger.debug(f"Resolved directory exists: {resolved_path}")
    else:
        logger.warning(f"Resolved directory does not exist: {resolved_path}")
        logger.warning(f"This may cause file discovery failures in {operation_name}")
    
    # Final audit log entry
    logger.info(f"✓ Base directory resolution complete for {operation_name}")
    logger.info(f"  Resolved path: {resolved_directory}")
    logger.info(f"  Resolution source: {resolution_source}")
    logger.debug(f"  Absolute path: {resolved_path.resolve()}")
    
    return resolved_directory

def get_file_statistics(
    path: Union[str, Path],
    _deps: Optional[DefaultDependencyProvider] = None
) -> Dict[str, Any]:
    """
    Get comprehensive statistics for a file with enhanced testability.
    
    This function supports comprehensive dependency injection for testing scenarios
    through the _deps parameter, enabling pytest.monkeypatch patterns.
    
    Args:
        path: Path to the file
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Dictionary containing file statistics (size, modification time, etc.)
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the path is invalid
    """
    operation_name = "get_file_statistics"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.debug(f"Getting file statistics for: {path}")
    
    # Validate path parameter
    if not path:
        error_msg = (
            f"Invalid path for {operation_name}: '{path}'. "
            "path must be a non-empty string or Path object."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    try:
        result = _deps.utils.get_file_stats(path)
        logger.debug(f"Successfully retrieved file statistics for: {path}")
        return result
    except FileNotFoundError as e:
        error_msg = (
            f"File not found for {operation_name}: {path}. "
            "Please ensure the file exists and the path is correct."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e
    except Exception as e:
        error_msg = (
            f"Failed to get file statistics for {operation_name} from {path}: {e}. "
            "Please check the file path and permissions."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e

def ensure_dir_exists(
    path: Union[str, Path],
    _deps: Optional[DefaultDependencyProvider] = None
) -> Path:
    """
    Ensure a directory exists, creating it if necessary, with enhanced testability.
    
    Args:
        path: Directory path
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Path object pointing to the directory
        
    Raises:
        ValueError: If the path is invalid
    """
    operation_name = "ensure_dir_exists"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.debug(f"Ensuring directory exists: {path}")
    
    # Validate path parameter
    if not path:
        error_msg = (
            f"Invalid path for {operation_name}: '{path}'. "
            "path must be a non-empty string or Path object."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    try:
        result = _deps.utils.ensure_directory_exists(path)
        logger.debug(f"Successfully ensured directory exists: {result}")
        return result
    except Exception as e:
        error_msg = (
            f"Failed to ensure directory exists for {operation_name} at {path}: {e}. "
            "Please check the path and permissions."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e

def check_if_file_exists(
    path: Union[str, Path],
    _deps: Optional[DefaultDependencyProvider] = None
) -> bool:
    """
    Check if a file exists with enhanced testability.
    
    Args:
        path: Path to check
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        True if the file exists, False otherwise
        
    Raises:
        ValueError: If the path is invalid
    """
    operation_name = "check_if_file_exists"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.debug(f"Checking if file exists: {path}")
    
    # Validate path parameter
    if not path:
        error_msg = (
            f"Invalid path for {operation_name}: '{path}'. "
            "path must be a non-empty string or Path object."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    try:
        result = _deps.utils.check_file_exists(path)
        logger.debug(f"File existence check for {path}: {result}")
        return result
    except Exception as e:
        error_msg = (
            f"Failed to check file existence for {operation_name} at {path}: {e}. "
            "Please check the path format."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e

def get_path_relative_to(
    path: Union[str, Path], 
    base_dir: Union[str, Path],
    _deps: Optional[DefaultDependencyProvider] = None
) -> Path:
    """
    Get a path relative to a base directory with enhanced testability.
    
    Args:
        path: Path to convert
        base_dir: Base directory
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Relative path
        
    Raises:
        ValueError: If the path is not within the base directory or parameters are invalid
    """
    operation_name = "get_path_relative_to"

    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()

    logger.debug(f"Getting relative path: {path} relative to {base_dir}")

    # Validate parameters
    if not path:
        _raise_path_validation_error(
            'Invalid path for ',
            operation_name,
            path,
            "'. path must be a non-empty string or Path object.",
        )
    if not base_dir:
        _raise_path_validation_error(
            'Invalid base_dir for ',
            operation_name,
            base_dir,
            "'. base_dir must be a non-empty string or Path object.",
        )
    try:
        result = _deps.utils.get_relative_path(path, base_dir)
        logger.debug(f"Relative path result: {result}")
        return result
    except ValueError as e:
        error_msg = (
            f"Path {path} is not within base directory {base_dir} for {operation_name}. "
            "Please ensure the path is within the specified base directory."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e
    except Exception as e:
        error_msg = (
            f"Failed to get relative path for {operation_name}: {e}. "
            "Please check the path formats."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e

def _raise_path_validation_error(prefix: str, operation_name: str, path_value: Any, suffix: str) -> None:
    """
    Raise a path validation error with standardized formatting.
    
    This function centralizes error formatting for path validation failures
    and ensures consistent error messaging across path utility functions.
    
    Args:
        prefix: Error message prefix
        operation_name: Name of the operation that failed validation
        path_value: The invalid path value
        suffix: Error message suffix
        
    Raises:
        FlyRigLoaderError: Always raises with formatted error message
    """
    error_msg = f"{prefix}{operation_name}: '{path_value}{suffix}"
    logger.error(error_msg)
    raise FlyRigLoaderError(error_msg)

def get_path_absolute(
    path: Union[str, Path], 
    base_dir: Union[str, Path],
    _deps: Optional[DefaultDependencyProvider] = None
) -> Path:
    """
    Convert a relative path to an absolute path with enhanced testability.
    
    Args:
        path: Relative path
        base_dir: Base directory
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Absolute path
        
    Raises:
        ValueError: If parameters are invalid
    """
    operation_name = "get_path_absolute"

    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()

    logger.debug(f"Getting absolute path: {path} with base {base_dir}")

    # Validate parameters
    if not path:
        _raise_path_validation_error(
            'Invalid path for ',
            operation_name,
            path,
            "'. path must be a non-empty string or Path object.",
        )
    if not base_dir:
        _raise_path_validation_error(
            'Invalid base_dir for ',
            operation_name,
            base_dir,
            "'. base_dir must be a non-empty string or Path object.",
        )
    try:
        result = _deps.utils.get_absolute_path(path, base_dir)
        logger.debug(f"Absolute path result: {result}")
        return result
    except Exception as e:
        error_msg = (
            f"Failed to get absolute path for {operation_name}: {e}. "
            "Please check the path formats."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e

def get_common_base_dir(
    paths: List[Union[str, Path]],
    _deps: Optional[DefaultDependencyProvider] = None
) -> Optional[Path]:
    """
    Find the common base directory for a list of paths with enhanced testability.
    
    Args:
        paths: List of paths to analyze
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Common base directory or None if no common base can be found
        
    Raises:
        ValueError: If the paths parameter is invalid
    """
    operation_name = "get_common_base_dir"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.debug(f"Finding common base directory for {len(paths) if paths else 0} paths")
    
    # Validate paths parameter
    if not isinstance(paths, list):
        error_msg = (
            f"Invalid paths parameter for {operation_name}: {type(paths).__name__}. "
            "paths must be a list of path strings or Path objects."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    if not paths:
        logger.debug("Empty paths list provided, returning None")
        return None
    
    # Validate individual paths
    for i, path in enumerate(paths):
        if not path:
            error_msg = (
                f"Invalid path at index {i} for {operation_name}: '{path}'. "
                "All paths must be non-empty strings or Path objects."
            )
            logger.error(error_msg)
            raise FlyRigLoaderError(error_msg)
    
    try:
        result = _deps.utils.find_common_base_directory(paths)
        if result:
            logger.debug(f"Found common base directory: {result}")
        else:
            logger.debug("No common base directory found")
        return result
    except Exception as e:
        error_msg = (
            f"Failed to find common base directory for {operation_name}: {e}. "
            "Please check the path formats."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e
