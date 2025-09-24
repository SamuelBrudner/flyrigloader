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

from pathlib import Path
from typing import Any, Dict, Optional, Union

from flyrigloader import logger
from flyrigloader.registries import LoaderRegistry, get_loader_capabilities
from flyrigloader.exceptions import RegistryError

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
        # Get registry instance for centralized loader resolution
        registry = LoaderRegistry()
        loader_class = None
        loader_name = None
        
        if loader is not None:
            # Explicit loader provided - treat as extension-based lookup with enhanced error handling
            logger.debug(f"Explicit loader requested: {loader}")
            extension_key = f'.{loader}' if not loader.startswith('.') else loader
            loader_class = registry.get_loader_for_extension(extension_key)
            loader_name = loader
            
            if loader_class is None:
                # Use RegistryError for registry-specific failures
                error_msg = f"No loader registered for explicitly requested format: {loader}"
                logger.error(error_msg)
                raise RegistryError(
                    error_msg, 
                    error_code="REGISTRY_002",
                    context={
                        'requested_loader': loader,
                        'extension_key': extension_key,
                        'file_path': str(file_path),
                        'available_loaders': list(registry.get_all_loaders().keys())
                    }
                )
        else:
            # Automatic format detection based on file extension with priority resolution
            file_extension = file_path.suffix.lower()
            logger.debug(f"Detecting loader for extension: {file_extension}")
            
            if not file_extension:
                error_msg = f"Cannot determine file format: no extension found for {file_path}"
                logger.error(error_msg)
                raise UnsupportedFormatError(error_msg, file_path=file_path)
            
            # Use registry with priority-based resolution (automatically handled by LoaderRegistry)
            loader_class = registry.get_loader_for_extension(file_extension)
            loader_name = file_extension
        
        # Enhanced loader resolution with RegistryError for registry failures
        if loader_class is None:
            error_msg = f"No loader registered for format: {loader_name}"
            logger.error(error_msg)
            
            # Get capability information for better error context
            all_loaders = registry.get_all_loaders()
            supported_formats = list(all_loaders.keys())
            
            raise RegistryError(
                error_msg,
                error_code="REGISTRY_002", 
                context={
                    'requested_format': loader_name,
                    'file_path': str(file_path),
                    'supported_formats': supported_formats,
                    'registry_size': len(all_loaders),
                    'priority_resolution': 'automatic'
                }
            )
        
        # Registry-based loader instantiation with enhanced logging
        logger.debug(f"Using registry-resolved loader: {loader_class.__name__}")
        
        # Get loader capabilities for enhanced monitoring and validation
        loader_capabilities = registry.get_loader_capabilities(loader_name or file_path.suffix.lower())
        if loader_capabilities:
            logger.debug(f"Loader capabilities: priority={loader_capabilities.get('priority', 'unknown')}, "
                        f"priority_enum={loader_capabilities.get('priority_name', 'unknown')}")
        
        try:
            # Create loader instance through registry-validated class
            loader_instance = loader_class()
            
            # Load raw data using the registry-provided loader
            raw_data = loader_instance.load(file_path)
            
            # Log successful load with enhanced context
            logger.info(f"Successfully loaded data from {file_path} using registry-resolved {loader_class.__name__}")
            
            return raw_data
            
        except Exception as loader_error:
            # Enhanced error handling for loader instantiation and execution failures
            error_msg = f"Registry-resolved loader {loader_class.__name__} failed to load {file_path}: {str(loader_error)}"
            logger.error(error_msg)
            raise LoaderError(
                error_msg, 
                file_path=file_path, 
                loader_type=loader_class.__name__
            ) from loader_error
        
    except FileNotFoundError:
        # Re-raise file not found errors as-is
        raise
    except PermissionError as e:
        # Handle permission errors with context
        error_msg = f"Permission denied accessing file: {file_path}"
        logger.error(error_msg)
        raise PermissionError(error_msg) from e
    except UnsupportedFormatError:
        # Re-raise format errors as-is - these are user-facing format issues
        raise
    except RegistryError:
        # Re-raise registry errors as-is - these are registry-specific issues
        raise
    except Exception as e:
        # Handle all other loader-specific errors with enhanced context
        error_msg = f"Failed to load data from {file_path}: {str(e)}"
        logger.error(error_msg)
        raise LoaderError(error_msg, file_path=file_path, loader_type=loader_name) from e


def get_supported_formats() -> Dict[str, str]:
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


def get_loader_registry_capabilities() -> Dict[str, Dict[str, Any]]:
    """Get comprehensive capability information for all registered loaders.
    
    This function provides detailed registry introspection capabilities per
    Section 5.2.4 enhancement requirements, exposing loader metadata including
    priority information, registration source, and loader-specific capabilities.
    
    Returns:
        Dictionary mapping file extensions to comprehensive loader information:
        - loader_class: Name of the loader class
        - loader_module: Module where the loader is defined
        - priority: Numeric priority value (higher = higher priority)
        - priority_enum: Priority enumeration (BUILTIN, USER, PLUGIN, OVERRIDE)
        - priority_name: Human-readable priority level name
        - registration_metadata: Context about when and how loader was registered
        - capabilities: Loader-specific capability information
        - supports_extension: Whether loader explicitly supports the extension
        
    Example:
        >>> capabilities = get_loader_registry_capabilities()
        >>> pkl_info = capabilities.get('.pkl', {})
        >>> print(f"Pickle loader priority: {pkl_info.get('priority', 'unknown')}")
        >>> print(f"Registration source: {pkl_info.get('registration_metadata', {}).get('source', 'unknown')}")
        
    Technical Notes:
        - Implements O(1) lookup performance per Section 5.2.4 scaling considerations
        - Results include priority resolution following BUILTIN < USER < PLUGIN < OVERRIDE hierarchy
        - Thread-safe through registry implementation
        - Capabilities introspection has <1ms overhead per Section 5.2.4 requirements
    """
    registry = LoaderRegistry()
    
    try:
        # Get comprehensive capabilities for all registered loaders
        all_capabilities = registry.get_all_loader_capabilities()
        
        # Enhanced logging for monitoring registry introspection
        logger.info(
            f"Retrieved capabilities for {len(all_capabilities)} registered loaders",
            extra={
                'capability_count': len(all_capabilities),
                'extensions': list(all_capabilities.keys()),
                'introspection_function': 'get_loader_registry_capabilities'
            }
        )
        
        return all_capabilities
        
    except Exception as e:
        # Handle registry introspection failures with RegistryError
        error_msg = f"Failed to retrieve loader registry capabilities: {str(e)}"
        logger.error(error_msg)
        raise RegistryError(
            error_msg,
            error_code="REGISTRY_010",
            context={
                'operation': 'capabilities_introspection',
                'error_type': type(e).__name__,
                'original_error': str(e)
            }
        ) from e


def get_loader_capability_for_extension(extension: str) -> Optional[Dict[str, Any]]:
    """Get detailed capability information for a specific file extension loader.
    
    This function provides targeted loader introspection for specific file formats,
    enabling users to understand loader capabilities, priorities, and metadata
    for individual extensions.
    
    Args:
        extension: File extension to get capabilities for (e.g., '.pkl', 'pkl')
                  Leading dot will be added if not present.
        
    Returns:
        Dictionary containing detailed loader capability information, or None if
        no loader is registered for the extension. Contains same fields as
        get_loader_registry_capabilities() but for a single extension.
        
    Example:
        >>> pkl_capabilities = get_loader_capability_for_extension('.pkl')
        >>> if pkl_capabilities:
        ...     print(f"Priority: {pkl_capabilities['priority']}")
        ...     print(f"Source: {pkl_capabilities['registration_metadata']['source']}")
        >>> 
        >>> # Works with or without leading dot
        >>> json_capabilities = get_loader_capability_for_extension('json')
        
    Technical Notes:
        - O(1) lookup performance through registry implementation
        - Handles extension normalization (adds leading dot if missing)
        - Returns None for unsupported extensions rather than raising exceptions
        - Thread-safe through registry singleton pattern
    """
    # Normalize extension format
    if not extension.startswith('.'):
        extension = f'.{extension}'
    
    try:
        # Use registry capability introspection directly
        capabilities = get_loader_capabilities(extension)
        
        if capabilities:
            logger.debug(
                f"Retrieved capabilities for extension {extension}",
                extra={
                    'extension': extension,
                    'loader_class': capabilities.get('loader_class', 'unknown'),
                    'priority': capabilities.get('priority', 'unknown')
                }
            )
        else:
            logger.debug(f"No capabilities found for extension {extension}")
        
        return capabilities
        
    except Exception as e:
        # Log error but don't raise - return None for graceful handling
        logger.warning(
            f"Failed to retrieve capabilities for extension {extension}: {str(e)}",
            extra={
                'extension': extension,
                'error_type': type(e).__name__,
                'operation': 'single_extension_capabilities'
            }
        )
        return None


def get_loader_priority_info() -> Dict[str, Dict[str, Any]]:
    """Get priority resolution information for all registered loaders.
    
    This function exposes the loader priority resolution system following the
    BUILTIN < USER < PLUGIN < OVERRIDE hierarchy per Section 0.3.1 registry
    foundation requirements.
    
    Returns:
        Dictionary mapping file extensions to priority resolution information:
        - numeric_priority: Raw numeric priority value
        - priority_enum: RegistryPriority enumeration value
        - priority_name: Human-readable priority level (BUILTIN, USER, PLUGIN, OVERRIDE)
        - loader_class: Name of the loader class
        - registration_source: How the loader was registered (api, plugin, entry_point, etc.)
        
    Example:
        >>> priority_info = get_loader_priority_info()
        >>> for ext, info in priority_info.items():
        ...     print(f"{ext}: {info['priority_name']} priority ({info['numeric_priority']})")
        
    Technical Notes:
        - Results are automatically sorted by priority (highest first)
        - Priority resolution follows BUILTIN=10, USER=20, PLUGIN=30, OVERRIDE=40 hierarchy
        - Enables understanding of which loader will be selected for each extension
        - Thread-safe through registry implementation
    """
    registry = LoaderRegistry()
    
    try:
        priority_info = {}
        
        # Get all loaders and their priority information
        with registry._registry_lock:
            for extension, loader_class in registry._registry.items():
                priority_data = registry.get_priority_info(extension)
                metadata = registry.get_registration_metadata(extension)
                
                if priority_data:
                    priority_info[extension] = {
                        'extension': extension,
                        'loader_class': loader_class.__name__,
                        'numeric_priority': priority_data['numeric_priority'],
                        'priority_enum': priority_data['priority_enum'],
                        'priority_name': priority_data['priority_name'],
                        'registration_source': metadata.get('source', 'unknown') if metadata else 'unknown',
                        'registration_time': metadata.get('registration_time', 'unknown') if metadata else 'unknown'
                    }
        
        # Sort by priority for clear presentation
        sorted_priority_info = dict(
            sorted(priority_info.items(), 
                  key=lambda x: x[1]['numeric_priority'], 
                  reverse=True)
        )
        
        logger.info(
            f"Retrieved priority information for {len(sorted_priority_info)} loaders",
            extra={
                'loader_count': len(sorted_priority_info),
                'priority_levels': list(set(info['priority_name'] for info in sorted_priority_info.values()))
            }
        )
        
        return sorted_priority_info
        
    except Exception as e:
        # Handle priority introspection failures
        error_msg = f"Failed to retrieve loader priority information: {str(e)}"
        logger.error(error_msg)
        raise RegistryError(
            error_msg,
            error_code="REGISTRY_005",
            context={
                'operation': 'priority_introspection',
                'error_type': type(e).__name__,
                'original_error': str(e)
            }
        ) from e


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
    # Core loading functionality
    'load_data_file',
    
    # Format support utilities
    'get_supported_formats', 
    'is_format_supported',
    
    # Registry capability introspection functions (new exports per Section 5.2.4)
    'get_loader_registry_capabilities',
    'get_loader_capability_for_extension', 
    'get_loader_priority_info',
    
    # Exception classes
    'LoaderError',
    'UnsupportedFormatError'
]