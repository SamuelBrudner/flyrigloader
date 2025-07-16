"""
Module for pure data loading from various pickle file formats.

This module focuses exclusively on loading data from different types of pickle files 
from fly behavior rigs, including compressed files (.pkl, .pklz, .pkl.gz), with robust 
error handling and format detection. 

Data transformation and DataFrame creation logic has been moved to the transformers 
module for improved separation of concerns. This module maintains the dependency 
injection patterns for enhanced testability and modular component architecture.

Registry Integration:
- Supports LoaderRegistry-based automatic format detection
- Enables plugin-style extensibility for new file formats
- Provides backward compatibility with existing code
"""

import gzip
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, Optional, List, Tuple, Protocol, runtime_checkable, Callable
from flyrigloader import logger


# Registry Integration Protocols and Interfaces

@runtime_checkable
class BaseLoader(Protocol):
    """Protocol for registry-compatible file loaders."""
    
    def load(self, path: Union[str, Path]) -> Any:
        """Load data from a file path."""
        ...
    
    def supports_extension(self, extension: str) -> bool:
        """Check if this loader supports the given file extension."""
        ...


class LoaderRegistry:
    """
    Registry for file format loaders supporting automatic format detection.
    
    This class implements the registry pattern for extensible file loading,
    allowing plugins to register new file format handlers without modifying
    core code. Thread-safe singleton implementation with O(1) lookup performance.
    """
    
    _instance = None
    _loaders: Dict[str, BaseLoader] = {}
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, extension: str, loader: BaseLoader) -> None:
        """
        Register a loader for a specific file extension.
        
        Args:
            extension: File extension (e.g., '.pkl', '.pklz')
            loader: Loader instance implementing BaseLoader protocol
        """
        if not extension.startswith('.'):
            extension = f'.{extension}'
        
        cls._loaders[extension] = loader
        logger.debug(f"Registered loader for extension: {extension}")
    
    @classmethod
    def get_loader(cls, extension: str) -> Optional[BaseLoader]:
        """
        Get registered loader for file extension.
        
        Args:
            extension: File extension to look up
            
        Returns:
            Loader instance or None if not found
        """
        if not extension.startswith('.'):
            extension = f'.{extension}'
            
        return cls._loaders.get(extension)
    
    @classmethod
    def get_loader_for_file(cls, path: Union[str, Path]) -> Optional[BaseLoader]:
        """
        Get registered loader for a file path.
        
        Args:
            path: File path to determine extension from
            
        Returns:
            Loader instance or None if not found
        """
        path_obj = Path(path)
        
        # Handle compressed extensions like .pkl.gz
        if path_obj.suffix == '.gz' and path_obj.stem.endswith('.pkl'):
            extension = '.pkl.gz'
        else:
            extension = path_obj.suffix
            
        return cls.get_loader(extension)
    
    @classmethod
    def get_registered_extensions(cls) -> List[str]:
        """Get list of all registered file extensions."""
        return list(cls._loaders.keys())
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered loaders (primarily for testing)."""
        cls._loaders.clear()
        logger.debug("Cleared loader registry")


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


class PickleLoader:
    """
    Enhanced pickle file loader with dependency injection support and registry integration.
    
    This class implements modular component decomposition per TST-REF-002 requirements,
    separating pickle loading operations from DataFrame transformation components to
    reduce coupling and enable controlled I/O behavior during test execution.
    
    Registry Integration:
    - Automatically registers itself for .pkl, .pklz, and .pkl.gz extensions
    - Supports LoaderRegistry-based format detection
    - Provides backward compatibility with existing code
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
    Initialize the LoaderRegistry with default PickleLoader for supported extensions.
    
    This function automatically registers the PickleLoader class for all pickle
    file extensions it supports, enabling registry-based automatic format detection.
    """
    pickle_loader = PickleLoader()
    supported_extensions = ['.pkl', '.pklz', '.pkl.gz']
    
    for extension in supported_extensions:
        LoaderRegistry.register(extension, pickle_loader)
    
    logger.debug(f"Registered PickleLoader for extensions: {supported_extensions}")


# Initialize registry on module import
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
        registered_loader = LoaderRegistry.get_loader_for_file(path)
        if registered_loader is None:
            path_obj = Path(path)
            extension = path_obj.suffix
            if path_obj.suffix == '.gz' and path_obj.stem.endswith('.pkl'):
                extension = '.pkl.gz'
            
            raise RuntimeError(
                f"No registered loader found for file extension '{extension}'. "
                f"Supported extensions: {LoaderRegistry.get_registered_extensions()}"
            )
        
        return registered_loader.load(path)
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
    
    Args:
        path: Path to the pickle file to read
        
    Returns:
        Dictionary containing the loaded experimental data
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file format is invalid
        PermissionError: If the file cannot be accessed
        RuntimeError: If the file cannot be loaded
    """
    return read_pickle_any_format(path)


def read_pickle_any_format(path: Union[str, Path]) -> Union[Dict[str, Any], pd.DataFrame]:
    """
    Convenience function for reading pickle files using registry-based format detection.
    
    This function provides backward compatibility while leveraging the new registry
    system for automatic format detection. It first attempts to use the registry
    for format selection, then falls back to the default PickleLoader if needed.
    
    Args:
        path: Path to the pickle file to read
        
    Returns:
        Dictionary data (exp_matrix style) or DataFrame
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the path format is invalid
        PermissionError: If the file cannot be accessed due to permissions
        RuntimeError: If all loading approaches fail
    """
    try:
        # First try registry-based loading
        return load_data_file(path)
    except RuntimeError:
        # Fall back to direct PickleLoader if registry fails
        loader = PickleLoader()
        return loader.load_pickle_any_format(path)


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

def register_test_loader(extension: str, loader: BaseLoader) -> None:
    """
    Register a test loader for a specific file extension.
    
    This function is intended for testing scenarios where custom loaders
    need to be registered for specific file extensions during test execution.
    
    Args:
        extension: File extension (e.g., '.test', '.mock')
        loader: Test loader instance implementing BaseLoader protocol
    """
    LoaderRegistry.register(extension, loader)
    logger.debug(f"Registered test loader for extension: {extension}")


def clear_loader_registry() -> None:
    """
    Clear all registered loaders for testing isolation.
    
    This function provides a clean slate for each test by clearing all
    registered loaders and re-initializing with the default PickleLoader.
    """
    LoaderRegistry.clear_registry()
    _initialize_pickle_loader_registry()
    logger.debug("Cleared and reinitialized loader registry")


def get_registered_loaders() -> Dict[str, BaseLoader]:
    """
    Get all registered loaders for testing verification.
    
    Returns:
        Dictionary mapping file extensions to their registered loaders
    """
    return LoaderRegistry._loaders.copy()
