"""
Centralized data loading module providing registry-based file format handling.

This module implements the unified data loading interface that separates loading
from transformation concerns, following the architectural principles outlined in
the technical specification. It provides the load_data_file() function that
automatically selects appropriate loaders based on file extensions through
the LoaderRegistry.

Key Features:
- Registry-based format selection with O(1) lookup performance
- Automatic format detection based on file extensions
- Pluggable loader registration for extensibility
- Separation of loading from transformation logic
- Comprehensive error handling with domain-specific exceptions
- Thread-safe operations through registry singleton pattern

Architecture Integration:
The loader module serves as a centralized entry point for all data loading
operations, replacing the previous monolithic approach where discovery,
loading, and transformation were combined. This separation enables users
to intercept and modify data at any pipeline stage.

Usage Example:
    >>> from pathlib import Path
    >>> from flyrigloader.io.loaders import load_data_file
    >>> 
    >>> # Automatic format detection
    >>> data = load_data_file(Path('experiment.pkl'))
    >>> 
    >>> # Explicit loader specification
    >>> data = load_data_file(Path('data.custom'), loader='custom')
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

from flyrigloader.registries import LoaderRegistry


# Configure module-level logging
logger = logging.getLogger(__name__)


class LoaderError(Exception):
    """Base exception for loader-related errors.
    
    This exception is raised when file loading operations fail,
    providing context about the error condition to enable
    appropriate error handling in client code.
    """
    
    def __init__(self, message: str, file_path: Optional[Path] = None, 
                 loader_type: Optional[str] = None):
        """Initialize LoaderError with context information.
        
        Args:
            message: Human-readable error description
            file_path: Path to the file that caused the error
            loader_type: Type of loader that failed
        """
        super().__init__(message)
        self.file_path = file_path
        self.loader_type = loader_type
        self.message = message
    
    def __str__(self) -> str:
        """Return formatted error message with context."""
        parts = [self.message]
        if self.file_path:
            parts.append(f"File: {self.file_path}")
        if self.loader_type:
            parts.append(f"Loader: {self.loader_type}")
        return " | ".join(parts)


class UnsupportedFormatError(LoaderError):
    """Exception raised when no loader is found for file format.
    
    This exception is raised when attempting to load a file with
    an extension that has no registered loader in the LoaderRegistry.
    """
    pass


def load_data_file(
    file_path: Union[str, Path], 
    loader: Optional[str] = None
) -> Any:
    """Load raw data from file using registered loader with automatic format detection.
    
    This function provides the centralized entry point for all data loading
    operations in FlyRigLoader. It automatically selects the appropriate
    loader based on file extension through the LoaderRegistry, enabling
    pluggable format support without modifying core code.
    
    The function focuses solely on loading raw data without any transformation,
    adhering to the separation of concerns principle where loading, discovery,
    and transformation are distinct pipeline stages.
    
    Args:
        file_path: Path to the file to load. Can be string or Path object.
        loader: Optional loader identifier. If None, loader is automatically
               selected based on file extension through LoaderRegistry.
               
    Returns:
        Raw data object loaded from file. The specific type depends on the
        loader implementation (typically dict, list, or numpy array).
        
    Raises:
        UnsupportedFormatError: If no loader is registered for the file extension
                               and no explicit loader is provided.
        LoaderError: If the loader fails to load the file due to corruption,
                    permission issues, or other I/O problems.
        FileNotFoundError: If the specified file does not exist.
        PermissionError: If the file cannot be accessed due to permission issues.
        
    Example:
        >>> from pathlib import Path
        >>> from flyrigloader.io.loaders import load_data_file
        >>> 
        >>> # Automatic format detection based on extension
        >>> data = load_data_file('experiment_data.pkl')
        >>> 
        >>> # Using Path object
        >>> data = load_data_file(Path('results/analysis.pkl'))
        >>> 
        >>> # Explicit loader specification (for custom formats)
        >>> data = load_data_file('data.custom', loader='custom_loader')
        
    Technical Notes:
        - The function uses LoaderRegistry singleton for O(1) loader lookup
        - Loader instances are created fresh for each load operation
        - All errors are logged before raising to aid debugging
        - The function is thread-safe through registry implementation
    """
    # Convert string paths to Path objects for consistent handling
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    # Log the loading attempt for debugging and monitoring
    logger.debug(f"Loading data from file: {file_path}")
    
    # Validate file exists before attempting to load
    if not file_path.exists():
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Validate file is readable
    if not file_path.is_file():
        error_msg = f"Path is not a file: {file_path}"
        logger.error(error_msg)
        raise LoaderError(error_msg, file_path=file_path)
    
    try:
        # Get loader class from registry
        loader_class = None
        loader_name = None
        
        if loader is not None:
            # Explicit loader provided - this would require extending LoaderRegistry
            # For now, treat it as extension-based lookup
            logger.debug(f"Explicit loader requested: {loader}")
            # This could be enhanced to support named loaders in the registry
            loader_class = LoaderRegistry().get_loader_for_extension(f'.{loader}')
            loader_name = loader
        else:
            # Automatic format detection based on file extension
            file_extension = file_path.suffix.lower()
            logger.debug(f"Detecting loader for extension: {file_extension}")
            
            if not file_extension:
                error_msg = f"Cannot determine file format: no extension found for {file_path}"
                logger.error(error_msg)
                raise UnsupportedFormatError(error_msg, file_path=file_path)
            
            loader_class = LoaderRegistry().get_loader_for_extension(file_extension)
            loader_name = file_extension
        
        # Check if loader was found
        if loader_class is None:
            error_msg = f"No loader registered for format: {loader_name}"
            logger.error(error_msg)
            raise UnsupportedFormatError(error_msg, file_path=file_path, loader_type=loader_name)
        
        # Create loader instance and load data
        logger.debug(f"Using loader: {loader_class.__name__}")
        loader_instance = loader_class()
        
        # Load raw data using the loader
        raw_data = loader_instance.load(file_path)
        
        # Log successful load for monitoring
        logger.info(f"Successfully loaded data from {file_path} using {loader_class.__name__}")
        
        return raw_data
        
    except FileNotFoundError:
        # Re-raise file not found errors as-is
        raise
    except PermissionError as e:
        # Handle permission errors with context
        error_msg = f"Permission denied accessing file: {file_path}"
        logger.error(error_msg)
        raise PermissionError(error_msg) from e
    except UnsupportedFormatError:
        # Re-raise format errors as-is
        raise
    except Exception as e:
        # Handle all other loader-specific errors
        error_msg = f"Failed to load data from {file_path}: {str(e)}"
        logger.error(error_msg)
        raise LoaderError(error_msg, file_path=file_path, loader_type=loader_name) from e


def get_supported_formats() -> dict:
    """Get all supported file formats and their registered loaders.
    
    This function provides introspection capabilities for the loading system,
    allowing users and other components to discover what file formats are
    currently supported through the LoaderRegistry.
    
    Returns:
        Dictionary mapping file extensions to loader class names.
        Extensions include the leading dot (e.g., '.pkl').
        
    Example:
        >>> formats = get_supported_formats()
        >>> print(formats)
        {'.pkl': 'PickleLoader', '.json': 'JsonLoader'}
        
    Technical Notes:
        - Results are ordered by loader priority (highest first)
        - The function is thread-safe through registry implementation
        - Changes to registry are immediately reflected in results
    """
    registry = LoaderRegistry()
    all_loaders = registry.get_all_loaders()
    
    # Convert loader classes to their names for readability
    format_info = {
        extension: loader_class.__name__ 
        for extension, loader_class in all_loaders.items()
    }
    
    logger.debug(f"Supported formats: {list(format_info.keys())}")
    return format_info


def is_format_supported(file_path: Union[str, Path]) -> bool:
    """Check if a file format is supported by registered loaders.
    
    This function provides a quick way to test if a file can be loaded
    without actually attempting the load operation.
    
    Args:
        file_path: Path to check for format support
        
    Returns:
        True if the file format is supported, False otherwise
        
    Example:
        >>> is_format_supported('data.pkl')
        True
        >>> is_format_supported('data.unknown')
        False
        
    Technical Notes:
        - Only checks file extension, not file content validity
        - Uses LoaderRegistry for O(1) lookup performance
        - Thread-safe through registry implementation
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    file_extension = file_path.suffix.lower()
    if not file_extension:
        return False
    
    registry = LoaderRegistry()
    loader_class = registry.get_loader_for_extension(file_extension)
    
    is_supported = loader_class is not None
    logger.debug(f"Format support check for {file_extension}: {is_supported}")
    
    return is_supported


# Export public interface
__all__ = [
    'load_data_file',
    'get_supported_formats', 
    'is_format_supported',
    'LoaderError',
    'UnsupportedFormatError'
]