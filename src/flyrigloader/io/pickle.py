"""
Module for pure data loading from various pickle file formats with enhanced registry integration.

This module focuses exclusively on loading data from different types of pickle files 
from fly behavior rigs, including compressed files (.pkl, .pklz, .pkl.gz), with robust 
error handling and format detection. 

Data transformation and DataFrame creation logic has been moved to the transformers 
module for improved separation of concerns. This module maintains the dependency 
injection patterns for enhanced testability and modular component architecture.

Enhanced Registry Integration (v1.0):
- Supports LoaderRegistry-based automatic format detection with priority-based resolution
- Enables plugin-style extensibility for new file formats using RegistryPriority enum
- Provides thread-safe registration operations using threading.RLock for concurrency control
- Implements comprehensive audit logging with INFO-level registration event tracking
- Integrates with automatic entry-point discovery mechanism per Section 0.2.1 requirements
- Uses RegistryError exceptions for enhanced error handling during registration operations
- Provides backward compatibility with existing code while enabling future extensibility
"""

import gzip
import pickle
import pandas as pd
from pathlib import Path
from threading import RLock
from typing import Dict, Any, Union, Optional, List, Tuple, Protocol, runtime_checkable, Callable, Type
from flyrigloader import logger
from flyrigloader.registries import LoaderRegistry, BaseLoader, RegistryPriority, auto_register
from flyrigloader.exceptions import RegistryError


# Dependency Injection Protocols and Interfaces

@runtime_checkable
class FileSystemProvider(Protocol):
    """Protocol for file system operations to enable dependency injection."""
    
    def path_exists(self, path: Union[str, Path]) -> bool:
        """Check if a path exists."""
        ...
    
    def open_file(self, path: Union[str, Path], mode: str) -> Any:
        """Open a file with the specified mode."""
        ...


@runtime_checkable
class CompressionProvider(Protocol):
    """Protocol for compression operations to enable dependency injection."""
    
    def open_gzip(self, path: Union[str, Path], mode: str) -> Any:
        """Open a gzip file with the specified mode."""
        ...


@runtime_checkable
class PickleProvider(Protocol):
    """Protocol for pickle operations to enable dependency injection."""
    
    def load(self, file_obj) -> Any:
        """Load data from a pickle file object."""
        ...


@runtime_checkable
class DataFrameProvider(Protocol):
    """Protocol for pandas operations to enable dependency injection - focused on pickle reading."""
    
    def read_pickle(self, path: Union[str, Path]) -> Any:
        """Read a pickle file using pandas."""
        ...
    
    def create_dataframe(self, data: Dict[str, Any], **kwargs) -> Any:
        """Create a DataFrame from data - not used in pure data loading."""
        ...
    
    def create_series(self, data: Any, **kwargs) -> Any:
        """Create a Series from data - not used in pure data loading."""
        ...


class DefaultFileSystemProvider:
    """Default implementation of FileSystemProvider using standard library."""
    
    def path_exists(self, path: Union[str, Path]) -> bool:
        """Check if a path exists using pathlib."""
        return Path(path).exists()
    
    def open_file(self, path: Union[str, Path], mode: str):
        """Open a file using built-in open function."""
        return open(path, mode)


class DefaultCompressionProvider:
    """Default implementation of CompressionProvider using gzip."""
    
    def open_gzip(self, path: Union[str, Path], mode: str):
        """Open a gzip file using gzip.open."""
        return gzip.open(path, mode)


class DefaultPickleProvider:
    """Default implementation of PickleProvider using pickle module."""
    
    def load(self, file_obj) -> Any:
        """Load data using pickle.load."""
        return pickle.load(file_obj)


class DefaultDataFrameProvider:
    """Default implementation of DataFrameProvider using pandas for pickle reading only."""
    
    def read_pickle(self, path: Union[str, Path]) -> Any:
        """Read a pickle file using pandas."""
        return pd.read_pickle(path)
    
    def create_dataframe(self, data: Dict[str, Any], **kwargs) -> Any:
        """Create a DataFrame using pandas - not used in pure data loading."""
        raise NotImplementedError("DataFrame creation moved to transformers module")
    
    def create_series(self, data: Any, **kwargs) -> Any:
        """Create a Series using pandas - not used in pure data loading."""
        raise NotImplementedError("Series creation moved to transformers module")


class DependencyContainer:
    """
    Container for managing configurable dependencies to support dependency injection.
    
    This enables comprehensive unit testing through dependency substitution and
    controlled I/O behavior during test execution per TST-REF-003 requirements.
    """
    
    def __init__(
        self,
        filesystem_provider: Optional[FileSystemProvider] = None,
        compression_provider: Optional[CompressionProvider] = None,
        pickle_provider: Optional[PickleProvider] = None,
        dataframe_provider: Optional[DataFrameProvider] = None
    ):
        """
        Initialize dependency container with configurable providers.
        
        Args:
            filesystem_provider: Provider for file system operations
            compression_provider: Provider for compression operations  
            pickle_provider: Provider for pickle operations
            dataframe_provider: Provider for DataFrame operations
        """
        self.filesystem = filesystem_provider or DefaultFileSystemProvider()
        self.compression = compression_provider or DefaultCompressionProvider()
        self.pickle = pickle_provider or DefaultPickleProvider()
        self.dataframe = dataframe_provider or DefaultDataFrameProvider()
        
        logger.debug("Initialized DependencyContainer with providers")


# Global default dependency container - can be overridden for testing
_default_dependencies = DependencyContainer()

# Thread-safe registration lock for concurrent registration operations
_registration_lock = RLock()


@auto_register(registry_type="loader", key=".pkl", priority=RegistryPriority.BUILTIN.value, priority_enum=RegistryPriority.BUILTIN)
@auto_register(registry_type="loader", key=".pklz", priority=RegistryPriority.BUILTIN.value, priority_enum=RegistryPriority.BUILTIN)
@auto_register(registry_type="loader", key=".pkl.gz", priority=RegistryPriority.BUILTIN.value, priority_enum=RegistryPriority.BUILTIN)
class PickleLoader:
    """
    Enhanced pickle file loader with dependency injection support and registry integration.
    
    This class implements modular component decomposition per TST-REF-002 requirements,
    separating pickle loading operations from DataFrame transformation components to
    reduce coupling and enable controlled I/O behavior during test execution.
    
    Registry Integration:
    - Automatically registers itself for .pkl, .pklz, and .pkl.gz extensions using @auto_register decorator
    - Supports LoaderRegistry-based format detection with priority-based resolution
    - Provides backward compatibility with existing code
    - Uses thread-safe registration operations with comprehensive audit logging
    """
    
    def __init__(self, dependencies: Optional[DependencyContainer] = None):
        """
        Initialize PickleLoader with configurable dependencies.
        
        Args:
            dependencies: Dependency container for injectable providers.
                         If None, uses global default dependencies.
        """
        self.deps = dependencies or _default_dependencies
        logger.debug("Initialized PickleLoader with dependency injection support")
    
    def load(self, path: Union[str, Path]) -> Any:
        """
        Load data from a file path (BaseLoader protocol method).
        
        This method provides the registry-compatible interface for loading data
        from pickle files. It delegates to load_pickle_any_format for actual
        loading logic.
        
        Args:
            path: Path to the pickle file to load
            
        Returns:
            Loaded data object
        """
        # Delegate to load_pickle_any_format which handles validation and conversion
        return self.load_pickle_any_format(path)
    
    def supports_extension(self, extension: str) -> bool:
        """
        Check if this loader supports the given file extension.
        
        Args:
            extension: File extension to check
            
        Returns:
            True if extension is supported, False otherwise
        """
        supported_extensions = {'.pkl', '.pklz', '.pkl.gz'}
        if not extension.startswith('.'):
            extension = f'.{extension}'
        return extension in supported_extensions
    
    @property
    def priority(self) -> int:
        """Priority for this loader (BaseLoader protocol requirement).
        
        Uses RegistryPriority.BUILTIN as this is a core system loader.
        
        Returns:
            Integer priority (higher values = higher priority)
        """
        return RegistryPriority.BUILTIN.value  # Built-in priority for core pickle loader
    
    def _validate_path_input(self, path: Union[str, Path]) -> Path:
        """
        Enhanced parameter validation with detailed error messages.
        
        Args:
            path: Input path to validate
            
        Returns:
            Normalized Path object
            
        Raises:
            ValueError: If path format is invalid with detailed explanation
            TypeError: If path type is unexpected
        """
        if path is None:
            raise ValueError("Path cannot be None. Expected a valid file path as string or Path object.")
        
        if not isinstance(path, (str, Path)):
            raise TypeError(
                f"Invalid path type: expected string or pathlib.Path, got {type(path).__name__}. "
                f"Path value: {repr(path)}"
            )
        
        if isinstance(path, str) and not path.strip():
            raise ValueError("Path cannot be empty string. Expected a valid file path.")
        
        try:
            normalized_path = Path(path)
        except Exception as e:
            raise ValueError(f"Failed to convert path to Path object: {e}. Input path: {repr(path)}") from e
            
        logger.debug(f"Validated and normalized path: {normalized_path}")
        return normalized_path
    
    def _check_file_existence(self, path: Path) -> None:
        """
        Check file existence with enhanced error handling and logging.
        
        Args:
            path: Path to check
            
        Raises:
            FileNotFoundError: If file does not exist with detailed path information
        """
        try:
            if not self.deps.filesystem.path_exists(path):
                error_msg = (
                    f"File not found: '{path}'. "
                    f"Please verify the file exists and the path is correct. "
                    f"Current working directory: {Path.cwd()}"
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                raise
            error_msg = f"Error checking file existence for '{path}': {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        logger.debug(f"Confirmed file exists: {path}")
    
    def _attempt_gzipped_pickle_load(self, path: Path) -> Tuple[Any, Optional[str]]:
        """
        Attempt to load file as gzipped pickle with detailed error tracking.
        
        Args:
            path: Path to the file
            
        Returns:
            Tuple of (loaded_object, error_message)
            If successful, error_message is None
            If failed, loaded_object is None and error_message contains details
        """
        try:
            logger.debug(f"Attempting gzipped pickle load: {path}")
            with self.deps.compression.open_gzip(path, 'rb') as f:
                obj = self.deps.pickle.load(f)
            logger.info(f"Successfully loaded gzipped pickle: {path}")
            return obj, None
            
        except gzip.BadGzipFile as e:
            error_msg = f"Not a valid gzipped file: {e}"
            logger.debug(error_msg)
            return None, error_msg
            
        except OSError as e:
            error_msg = f"OS error reading gzipped file '{path}': {e}"
            logger.debug(error_msg)
            return None, error_msg
            
        except pickle.UnpicklingError as e:
            error_msg = f"Invalid pickle format in gzipped file '{path}': {e}"
            logger.debug(error_msg)
            return None, error_msg
            
        except EOFError as e:
            error_msg = f"Unexpected end of file in gzipped format '{path}': {e}"
            logger.debug(error_msg)
            return None, error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error loading gzipped pickle '{path}': {type(e).__name__}: {e}"
            logger.debug(error_msg)
            return None, error_msg
    
    def _attempt_regular_pickle_load(self, path: Path) -> Tuple[Any, Optional[str]]:
        """
        Attempt to load file as regular pickle with detailed error tracking.
        
        Args:
            path: Path to the file
            
        Returns:
            Tuple of (loaded_object, error_message)
            If successful, error_message is None
            If failed, loaded_object is None and error_message contains details
        """
        try:
            logger.debug(f"Attempting regular pickle load: {path}")
            with self.deps.filesystem.open_file(path, 'rb') as f:
                obj = self.deps.pickle.load(f)
            logger.info(f"Successfully loaded regular pickle: {path}")
            return obj, None
            
        except PermissionError as e:
            error_msg = f"Permission denied accessing file '{path}': {e}"
            logger.error(error_msg)
            # Re-raise permission errors immediately as they are not retry-able
            raise PermissionError(f"Cannot access file due to permissions: {e}") from e
            
        except pickle.UnpicklingError as e:
            error_msg = f"Invalid pickle format in regular file '{path}': {e}"
            logger.debug(error_msg)
            return None, error_msg
            
        except EOFError as e:
            error_msg = f"Unexpected end of file in regular format '{path}': {e}"
            logger.debug(error_msg)
            return None, error_msg
            
        except OSError as e:
            error_msg = f"OS error reading regular file '{path}': {e}"
            logger.debug(error_msg)
            return None, error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error loading regular pickle '{path}': {type(e).__name__}: {e}"
            logger.debug(error_msg)
            return None, error_msg
    
    def _attempt_pandas_pickle_load(self, path: Path) -> Tuple[Any, Optional[str]]:
        """
        Attempt to load file using pandas read_pickle with detailed error tracking.
        
        Args:
            path: Path to the file
            
        Returns:
            Tuple of (loaded_object, error_message)
            If successful, error_message is None
            If failed, loaded_object is None and error_message contains details
        """
        try:
            logger.debug(f"Attempting pandas pickle load: {path}")
            obj = self.deps.dataframe.read_pickle(path)
            logger.info(f"Successfully loaded pickle using pandas: {path}")
            return obj, None
            
        except pickle.UnpicklingError as e:
            error_msg = f"Invalid pickle format for pandas reader '{path}': {e}"
            logger.debug(error_msg)
            return None, error_msg
            
        except Exception as e:
            # Pandas can raise various errors that are difficult to enumerate
            error_msg = f"Error with pandas read_pickle '{path}': {type(e).__name__}: {e}"
            logger.debug(error_msg)
            return None, error_msg
    
    def load_pickle_any_format(self, path: Union[str, Path]) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Read a pickle file in any common format with automatic detection and enhanced error handling.
        
        This method implements a systematic approach with detailed error tracking:
        1. Validate input parameters with comprehensive error messages
        2. Check file existence with informative error reporting
        3. Attempt gzipped pickle loading with gzip.open + pickle.load
        4. Attempt regular pickle loading with open + pickle.load  
        5. Attempt pandas pickle loading with pd.read_pickle
        6. Provide comprehensive error reporting if all approaches fail
        
        Args:
            path: Path to the pickle file to read
            
        Returns:
            Dictionary data (exp_matrix style) or DataFrame
            
        Raises:
            FileNotFoundError: If the file does not exist with detailed path info
            ValueError: If the path format is invalid with helpful guidance
            TypeError: If the path type is unexpected
            PermissionError: If the file cannot be accessed due to permissions
            RuntimeError: If all loading approaches fail with comprehensive error details
        """
        # Enhanced parameter validation per Section 2.2.8 requirements
        validated_path = self._validate_path_input(path)
        
        # File existence check with detailed error reporting
        self._check_file_existence(validated_path)
        
        # Track all attempted approaches for comprehensive error reporting
        error_details = []
        
        # Approach 1: Try gzipped pickle with dependency injection
        obj, error = self._attempt_gzipped_pickle_load(validated_path)
        if obj is not None:
            return obj
        if error:
            error_details.append(f"Gzipped pickle: {error}")
        
        # Approach 2: Try regular pickle with dependency injection
        obj, error = self._attempt_regular_pickle_load(validated_path)
        if obj is not None:
            return obj
        if error:
            error_details.append(f"Regular pickle: {error}")
        
        # Approach 3: Try pandas pickle with dependency injection
        obj, error = self._attempt_pandas_pickle_load(validated_path)
        if obj is not None:
            return obj
        if error:
            error_details.append(f"Pandas pickle: {error}")
        
        # All approaches failed - provide comprehensive error report
        error_summary = "; ".join(error_details) if error_details else "Unknown errors in all approaches"
        final_error_msg = (
            f"Failed to load pickle file '{validated_path}' using all available methods. "
            f"Attempted approaches: {error_summary}. "
            f"Please verify the file is a valid pickle file and not corrupted."
        )
        logger.error(final_error_msg)
        raise RuntimeError(final_error_msg)
    
    def read_pickle_any_format(self, path: Union[str, Path]) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Read a pickle file in any common format with automatic detection (alias for load_pickle_any_format).
        
        This method provides an alternative interface name for the same functionality,
        supporting both naming conventions for backward compatibility and user preference.
        
        Args:
            path: Path to the pickle file to read
            
        Returns:
            Dictionary data (exp_matrix style) or DataFrame
            
        Raises:
            FileNotFoundError: If the file does not exist with detailed path info
            ValueError: If the path format is invalid with helpful guidance
            TypeError: If the path type is unexpected
            PermissionError: If the file cannot be accessed due to permissions
            RuntimeError: If all loading approaches fail with comprehensive error details
        """
        return self.load_pickle_any_format(path)


# Registry Integration and Auto-Registration

def _initialize_pickle_loader_registry():
    """
    Initialize the LoaderRegistry with default PickleLoader for supported extensions using priority-based resolution.
    
    This function automatically registers the PickleLoader class for all pickle
    file extensions it supports, enabling registry-based automatic format detection
    with thread-safe operations and comprehensive audit logging per Section 5.2.5 requirements.
    """
    with _registration_lock:
        try:
            registry = LoaderRegistry()
            supported_extensions = ['.pkl', '.pklz', '.pkl.gz']
            
            # Use RegistryPriority.BUILTIN for core pickle loader registration
            priority = RegistryPriority.BUILTIN
            
            for extension in supported_extensions:
                logger.info(f"Registering PickleLoader for extension '{extension}' with priority {priority.name}")
                registry.register_loader(extension, PickleLoader, priority=priority.value)
                logger.info(f"Successfully registered PickleLoader for extension '{extension}'")
            
            logger.info(f"PickleLoader registration completed for {len(supported_extensions)} extensions: {supported_extensions}")
            
        except Exception as e:
            error_msg = f"Failed to register PickleLoader: {e}"
            logger.error(error_msg)
            raise RegistryError(error_msg, error_code="REGISTRY_001", context={
                "extensions": supported_extensions,
                "loader_class": "PickleLoader",
                "priority": priority.name if 'priority' in locals() else "Unknown"
            }) from e


# Initialize registry on module import
# Note: @auto_register decorator provides automatic registration, but we also
# perform manual registration for redundancy and explicit audit logging
_initialize_pickle_loader_registry()


def load_data_file(path: Union[str, Path], loader: Optional[str] = None) -> Any:
    """
    Load data from a file using registry-based format detection.
    
    This function provides the registry-compatible interface for loading files
    with automatic format detection based on file extension. If no specific
    loader is provided, the LoaderRegistry is consulted for format selection.
    
    Args:
        path: Path to the file to load
        loader: Optional specific loader name to use (defaults to registry lookup)
        
    Returns:
        Loaded data object
        
    Raises:
        RuntimeError: If no suitable loader is found for the file extension
        FileNotFoundError: If the file does not exist
        ValueError: If the path format is invalid
    """
    if loader is None:
        registry = LoaderRegistry()
        path_obj = Path(path)
        
        # Handle compressed extensions like .pkl.gz
        if path_obj.suffix == '.gz' and path_obj.stem.endswith('.pkl'):
            extension = '.pkl.gz'
        else:
            extension = path_obj.suffix
            
        loader_class = registry.get_loader_for_extension(extension)
        if loader_class is None:
            all_loaders = registry.get_all_loaders()
            supported_extensions = list(all_loaders.keys())
            
            raise RuntimeError(
                f"No registered loader found for file extension '{extension}'. "
                f"Supported extensions: {supported_extensions}"
            )
        
        # Create loader instance and load the file
        loader_instance = loader_class()
        return loader_instance.load(path_obj)
    else:
        # Handle specific loader requests (future extensibility)
        raise NotImplementedError("Specific loader selection not yet implemented")





# Test-specific entry points for controlled I/O behavior per TST-REF-003 requirements

def set_global_dependencies(dependencies: DependencyContainer) -> DependencyContainer:
    """
    Set global dependencies for testing scenarios.
    
    This function provides a test-specific entry point for dependency injection,
    allowing comprehensive mocking and controlled I/O behavior during test execution.
    
    Args:
        dependencies: New dependency container to use globally
        
    Returns:
        Previous dependency container for restoration
    """
    global _default_dependencies
    previous = _default_dependencies
    _default_dependencies = dependencies
    logger.debug("Updated global dependencies for testing")
    return previous


def reset_global_dependencies() -> None:
    """
    Reset global dependencies to defaults.
    
    This function provides a clean state for test isolation,
    ensuring no test dependencies leak between test runs.
    """
    global _default_dependencies
    _default_dependencies = DependencyContainer()
    logger.debug("Reset global dependencies to defaults")


# Convenience functions for backward compatibility

def load_experimental_data(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load experimental data from a pickle file with automatic format detection.
    
    This is a convenience wrapper around read_pickle_any_format for backward compatibility.
    Uses enhanced registry system with priority-based loader resolution and comprehensive
    audit logging for traceability.
    
    Args:
        path: Path to the pickle file to read
        
    Returns:
        Dictionary containing the loaded experimental data
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file format is invalid
        PermissionError: If the file cannot be accessed
        RuntimeError: If the file cannot be loaded
        RegistryError: If registry lookup fails
    """
    logger.info(f"Loading experimental data from: {path}")
    try:
        result = read_pickle_any_format(path)
        logger.info(f"Successfully loaded experimental data from: {path}")
        return result
    except Exception as e:
        logger.error(f"Failed to load experimental data from {path}: {e}")
        raise


def read_pickle_any_format(path: Union[str, Path]) -> Union[Dict[str, Any], pd.DataFrame]:
    """
    Convenience function for reading pickle files using enhanced registry-based format detection.
    
    This function provides backward compatibility while leveraging the new registry
    system with priority-based loader resolution and comprehensive audit logging.
    It first attempts to use the registry for format selection, then falls back to 
    the default PickleLoader if needed.
    
    Args:
        path: Path to the pickle file to read
        
    Returns:
        Dictionary data (exp_matrix style) or DataFrame
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the path format is invalid
        PermissionError: If the file cannot be accessed due to permissions
        RuntimeError: If all loading approaches fail
        RegistryError: If registry operations fail
    """
    logger.info(f"Reading pickle file with registry-based format detection: {path}")
    try:
        # First try registry-based loading with priority resolution
        result = load_data_file(path)
        logger.info(f"Successfully loaded pickle file via registry: {path}")
        return result
    except RuntimeError as e:
        logger.warning(f"Registry-based loading failed for {path}, falling back to direct PickleLoader: {e}")
        # Fall back to direct PickleLoader if registry fails
        loader = PickleLoader()
        result = loader.load_pickle_any_format(path)
        logger.info(f"Successfully loaded pickle file via fallback loader: {path}")
        return result
    except Exception as e:
        logger.error(f"Failed to read pickle file {path}: {e}")
        raise


# Test helper factory functions

def create_test_pickle_loader(
    filesystem_provider: Optional[FileSystemProvider] = None,
    compression_provider: Optional[CompressionProvider] = None,
    pickle_provider: Optional[PickleProvider] = None,
    dataframe_provider: Optional[DataFrameProvider] = None
) -> PickleLoader:
    """
    Create a PickleLoader instance with test-specific dependencies.
    
    This factory function enables easy creation of PickleLoader instances with
    mocked dependencies for comprehensive unit testing scenarios.
    
    Args:
        filesystem_provider: Mock filesystem operations
        compression_provider: Mock compression operations
        pickle_provider: Mock pickle operations
        dataframe_provider: Mock DataFrame operations
        
    Returns:
        PickleLoader instance configured with test dependencies
    """
    test_deps = DependencyContainer(
        filesystem_provider=filesystem_provider,
        compression_provider=compression_provider,
        pickle_provider=pickle_provider,
        dataframe_provider=dataframe_provider
    )
    return PickleLoader(test_deps)


# Registry Test Helper Functions

def register_test_loader(extension: str, loader_class: Type[BaseLoader], priority: int = None) -> None:
    """
    Register a test loader for a specific file extension with thread-safe operations.
    
    This function is intended for testing scenarios where custom loaders
    need to be registered for specific file extensions during test execution.
    Uses RegistryPriority.USER by default for test loaders.
    
    Args:
        extension: File extension (e.g., '.test', '.mock')
        loader_class: Test loader class implementing BaseLoader protocol
        priority: Priority for the loader (defaults to RegistryPriority.USER)
    """
    with _registration_lock:
        try:
            if priority is None:
                priority = RegistryPriority.USER.value
                
            registry = LoaderRegistry()
            logger.info(f"Registering test loader '{loader_class.__name__}' for extension '{extension}' with priority {priority}")
            registry.register_loader(extension, loader_class, priority)
            logger.info(f"Successfully registered test loader for extension: {extension}")
            
        except Exception as e:
            error_msg = f"Failed to register test loader for extension '{extension}': {e}"
            logger.error(error_msg)
            raise RegistryError(error_msg, error_code="REGISTRY_002", context={
                "extension": extension,
                "loader_class": loader_class.__name__,
                "priority": priority
            }) from e


def clear_loader_registry() -> None:
    """
    Clear all registered loaders for testing isolation with thread-safe operations.
    
    This function provides a clean slate for each test by clearing all
    registered loaders and re-initializing with the default PickleLoader.
    Uses thread-safe operations and comprehensive audit logging.
    """
    with _registration_lock:
        try:
            logger.info("Clearing loader registry for testing isolation")
            registry = LoaderRegistry()
            registry.clear()
            logger.info("Loader registry cleared successfully")
            
            logger.info("Re-initializing loader registry with default PickleLoader")
            _initialize_pickle_loader_registry()
            logger.info("Loader registry reinitialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to clear and reinitialize loader registry: {e}"
            logger.error(error_msg)
            raise RegistryError(error_msg, error_code="REGISTRY_003", context={
                "operation": "clear_and_reinitialize"
            }) from e


def get_registered_loaders() -> Dict[str, Type[BaseLoader]]:
    """
    Get all registered loaders for testing verification.
    
    Returns:
        Dictionary mapping file extensions to their registered loader classes
    """
    registry = LoaderRegistry()
    return registry.get_all_loaders()
