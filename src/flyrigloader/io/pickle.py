"""
Module for loading data from various pickle file formats.

This module specializes in handling different types of pickle files from fly behavior rigs, 
including compressed files, with robust error handling and standardized output.

Enhanced with dependency injection patterns for improved testability and modular
component decomposition for better separation of concerns.
"""

import gzip
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, Optional, List, Tuple, Protocol, runtime_checkable
from abc import ABC, abstractmethod
from flyrigloader import logger
from flyrigloader.io.column_models import (
    load_column_config, 
    ColumnConfigDict, 
    ColumnConfig, 
    SpecialHandlerType,
    get_config_from_source
)


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
    """Protocol for pandas DataFrame operations to enable dependency injection."""
    
    def read_pickle(self, path: Union[str, Path]) -> Any:
        """Read a pickle file using pandas."""
        ...
    
    def create_dataframe(self, data: Dict[str, Any], **kwargs) -> Any:
        """Create a DataFrame from data."""
        ...
    
    def create_series(self, data: Any, **kwargs) -> Any:
        """Create a Series from data."""
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
    """Default implementation of DataFrameProvider using pandas."""
    
    def read_pickle(self, path: Union[str, Path]) -> Any:
        """Read a pickle file using pandas."""
        return pd.read_pickle(path)
    
    def create_dataframe(self, data: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """Create a DataFrame using pandas."""
        return pd.DataFrame(data, **kwargs)
    
    def create_series(self, data: Any, **kwargs) -> pd.Series:
        """Create a Series using pandas."""
        return pd.Series(data, **kwargs)


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
    Enhanced pickle file loader with dependency injection support for comprehensive testing.
    
    This class implements modular component decomposition per TST-REF-002 requirements,
    separating pickle loading operations from DataFrame transformation components to
    reduce coupling and enable controlled I/O behavior during test execution.
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


class DataFrameTransformer:
    """
    Modular DataFrame transformation component with dependency injection support.
    
    This class implements component decomposition per TST-REF-002 requirements,
    separating DataFrame creation and transformation operations from pickle loading
    to reduce coupling and enable comprehensive unit testing.
    """
    
    def __init__(self, dependencies: Optional[DependencyContainer] = None):
        """
        Initialize DataFrameTransformer with configurable dependencies.
        
        Args:
            dependencies: Dependency container for injectable providers.
                         If None, uses global default dependencies.
        """
        self.deps = dependencies or _default_dependencies
        logger.debug("Initialized DataFrameTransformer with dependency injection support")
    
    def _validate_exp_matrix_input(self, exp_matrix: Any) -> Dict[str, Any]:
        """
        Enhanced validation of exp_matrix input with detailed error messages.
        
        Args:
            exp_matrix: Input data to validate
            
        Returns:
            Validated dictionary
            
        Raises:
            TypeError: If input is not a dictionary with detailed explanation
            ValueError: If dictionary is empty or malformed
        """
        if exp_matrix is None:
            raise ValueError(
                "exp_matrix cannot be None. Expected a dictionary containing experimental data "
                "with keys representing column names and values containing the data arrays."
            )
        
        if not isinstance(exp_matrix, dict):
            raise TypeError(
                f"exp_matrix must be a dictionary, got {type(exp_matrix).__name__}. "
                f"Expected format: Dict[str, Any] where keys are column names and values are data arrays. "
                f"Received value: {repr(exp_matrix)}"
            )
        
        if not exp_matrix:
            raise ValueError(
                "exp_matrix cannot be empty. Expected a dictionary with at least one key-value pair "
                "representing experimental data columns."
            )
        
        logger.debug(f"Validated exp_matrix with {len(exp_matrix)} columns: {list(exp_matrix.keys())}")
        return exp_matrix
    
    def _validate_time_dimension_with_details(self, exp_matrix: Dict[str, Any]) -> int:
        """
        Validate time dimension presence with enhanced error reporting.
        
        Args:
            exp_matrix: Dictionary containing experimental data
            
        Returns:
            Length of time dimension
            
        Raises:
            ValueError: If time dimension is missing or invalid with detailed guidance
        """
        if 't' not in exp_matrix:
            available_keys = list(exp_matrix.keys())
            raise ValueError(
                f"exp_matrix must contain a 't' key for time values. "
                f"Available keys: {available_keys}. "
                f"Please ensure your experimental data includes time information."
            )
        
        time_data = exp_matrix['t']
        if not hasattr(time_data, '__len__'):
            raise ValueError(
                f"Time data 't' must be array-like with length, got {type(time_data).__name__}. "
                f"Expected numpy array, list, or pandas Series with time values."
            )
        
        time_length = len(time_data)
        if time_length == 0:
            raise ValueError("Time dimension 't' cannot be empty. Expected array of time values.")
        
        logger.debug(f"Validated time dimension with length: {time_length}")
        return time_length


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

def read_pickle_any_format(path: Union[str, Path]) -> Union[Dict[str, Any], pd.DataFrame]:
    """
    Convenience function for reading pickle files using the default PickleLoader.
    
    This function maintains backward compatibility while providing access to the
    enhanced dependency injection capabilities for testing scenarios.
    
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
    loader = PickleLoader()
    return loader.load_pickle_any_format(path)


def ensure_1d_array(array, name):
    """
    Enhanced utility function for ensuring 1D array format with detailed error reporting.
    
    This function provides improved error handling and logging for better test
    observability and assertion capabilities per Section 2.2.8 requirements.
    
    Args:
        array: Input array to process
        name: Name of the array for error reporting
        
    Returns:
        1D array version of input
        
    Raises:
        ValueError: If array cannot be converted to 1D with detailed explanation
    """
    if array is None:
        raise ValueError(f"Array '{name}' cannot be None. Expected a valid array-like input.")
    
    try:
        arr = np.asarray(array)
    except Exception as e:
        raise ValueError(f"Failed to convert '{name}' to numpy array: {e}") from e
    
    if arr.size == 0:
        logger.warning(f"Array '{name}' is empty")
        return arr
    
    # If shape is (N, 1) or (1, N), ravel to (N,)
    if arr.ndim == 2 and 1 in arr.shape:
        result = arr.ravel()
        logger.debug(f"Converted '{name}' from shape {arr.shape} to 1D with length {len(result)}")
        return result
    
    # If it's already 1D, return as-is
    elif arr.ndim == 1:
        logger.debug(f"Array '{name}' already 1D with length {len(arr)}")
        return arr
    
    # Multi-dimensional arrays that can't be converted
    else:
        raise ValueError(
            f"Column '{name}' has shape {arr.shape}, which cannot be converted to 1D. "
            f"Only arrays with shape (N,), (N, 1), or (1, N) can be converted to 1D. "
            f"If you need multi-dimensional data, consider storing each dimension separately "
            f"or flattening the data explicitly before processing."
        )


# Legacy functions maintained for backward compatibility - enhanced implementations
# are in the class-based components above

def extract_columns_from_matrix(
    exp_matrix: Dict[str, Any],
    column_names: Optional[List[str]] = None,
    ensure_1d: bool = True
) -> Dict[str, np.ndarray]:
    """
    Convenience function for extracting columns using the default DataFrameTransformer.
    
    This function maintains backward compatibility while providing access to enhanced
    validation and error handling capabilities.
    
    Args:
        exp_matrix: Dictionary containing experimental data
        column_names: List of column names to extract. If None, extracts all keys.
        ensure_1d: Whether to convert arrays to 1D (True) or keep original dimensions
        
    Returns:
        Dictionary mapping column names to processed arrays
        
    Raises:
        TypeError: If exp_matrix is not a dictionary
        ValueError: If column extraction fails
    """
    # For backward compatibility, use a simple implementation that matches the original behavior
    if not isinstance(exp_matrix, dict):
        raise ValueError(f"exp_matrix must be a dictionary, got {type(exp_matrix)}")
    
    if column_names is None:
        column_names = list(exp_matrix.keys())
    
    result = {}
    for col_name in column_names:
        if col_name in exp_matrix:
            if col_name == 'signal_disp':
                continue  # Skip special columns
                
            value = exp_matrix[col_name]
            if ensure_1d and hasattr(value, 'ndim') and value.ndim > 1:
                try:
                    value = ensure_1d_array(value, col_name)
                except ValueError as e:
                    logger.warning(f"Could not convert '{col_name}' to 1D: {e}")
                    if ensure_1d:
                        continue
            
            result[col_name] = value
    
    return result


def handle_signal_disp(exp_matrix: Dict[str, Any]) -> pd.Series:
    """
    Convenience function for handling signal display data using the default DataFrameTransformer.
    
    This function maintains backward compatibility while providing access to enhanced
    validation and error handling capabilities.
    
    Args:
        exp_matrix: Dictionary containing experimental data including signal_disp and t
        
    Returns:
        Series where each row contains an array of signal intensities
        
    Raises:
        TypeError: If exp_matrix is not a dictionary
        ValueError: If signal_disp dimensions don't match expected format
    """
    # Validate required keys
    if 'signal_disp' not in exp_matrix:
        raise ValueError("exp_matrix missing required 'signal_disp' key")

    if 't' not in exp_matrix:
        raise ValueError("exp_matrix missing required 't' key")

    # Get the signal_disp array and time array
    sd = exp_matrix['signal_disp']
    t = exp_matrix['t']
    T = len(t)  # Length of time dimension

    # Validate array dimensions
    if sd.ndim != 2:
        raise ValueError(f"signal_disp must be 2D, got {sd.shape}")

    # Determine orientation and transpose if needed
    n, m = sd.shape
    if n == T:
        # Already in correct orientation (T, X)
        logger.debug(f"signal_disp in (T, X) orientation with shape {sd.shape}")
        # No transpose needed
    elif m == T:
        # Need to transpose from (X, T) to (T, X)
        logger.debug(f"Transposing signal_disp from (X, T) orientation {sd.shape} to (T, X)")
        sd = sd.T
    else:
        # Neither dimension matches time dimension length
        raise ValueError(f"No dimension of signal_disp {sd.shape} matches time dimension length T={T}")

    return pd.Series(
        [sd[i, :] for i in range(T)],  # One row per timepoint
        index=range(T),  # Use time indices
        name='signal_disp',  # Name the series
    )


def make_dataframe_from_config(
    exp_matrix: Dict[str, Any],
    config_source: Union[str, Dict[str, Any], ColumnConfigDict, None] = None,
    metadata: Optional[Dict[str, Any]] = None,
    skip_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Enhanced convenience function for DataFrame creation with comprehensive configuration support.
    
    This function maintains backward compatibility while providing access to the enhanced
    ConfigurableDataFrameBuilder capabilities for dependency injection and testing.
    
    Args:
        exp_matrix: Dictionary containing experimental data
        config_source: Configuration source (path, dict, ColumnConfigDict, or None)
        metadata: Optional dictionary with metadata to add to the DataFrame
        skip_columns: Optional list of columns to exclude from processing
    
    Returns:
        pd.DataFrame: DataFrame containing the data with correct types
        
    Raises:
        ValueError: If required columns are missing or configuration is invalid
        ValidationError: If the configuration is invalid
        TypeError: If the config_source type is invalid
    """
    # Load and validate configuration
    config = get_config_from_source(config_source)
    skip_columns = skip_columns or []
    
    # Simple validation for required columns (backward compatibility)
    missing_columns = []
    for col_name, col_config in config.columns.items():
        if col_name in skip_columns or not col_config.required or col_config.is_metadata:
            continue
            
        if col_name in exp_matrix:
            continue
            
        # Check if there's an alias for this column
        alias = col_config.alias
        if alias and alias in exp_matrix:
            logger.debug(f"Required column '{col_name}' found via alias '{alias}'")
            continue
            
        missing_columns.append(col_name)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Process columns
    data_dict = {}
    for col_name, col_config in config.columns.items():
        if col_name in skip_columns:
            if col_config.required:
                logger.warning(f"Skipping required column '{col_name}' as requested")
            continue
            
        # Skip metadata columns
        if col_config.is_metadata:
            continue

        # Handle aliases
        source_col = col_name
        if col_config.alias and col_config.alias in exp_matrix:
            source_col = col_config.alias
            logger.debug(f"Using alias '{source_col}' for column '{col_name}'")

        # If column is not in the matrix
        if source_col not in exp_matrix:
            if not col_config.required and hasattr(col_config, 'default_value'):
                logger.debug(f"Using default value for missing column '{col_name}'")
                data_dict[col_name] = col_config.default_value
            continue
                
        # Get the value from the exp_matrix
        value = exp_matrix[source_col]

        # Apply special handling if configured
        if col_config.special_handling:
            if col_config.special_handling == SpecialHandlerType.EXTRACT_FIRST_COLUMN:
                logger.debug(f"Extracting first column from 2D array for '{col_name}'")
                if hasattr(value, 'ndim') and value.ndim == 2:
                    value = value[:, 0] if value.shape[1] > 0 else value
            elif col_config.special_handling == SpecialHandlerType.TRANSFORM_TIME_DIMENSION:
                logger.debug(f"Transforming '{col_name}' to match time dimension")
                value = handle_signal_disp(exp_matrix)

        # Ensure correct dimensionality for numpy arrays
        if isinstance(value, np.ndarray) and col_config.dimension:
            try:
                if col_config.dimension.value == 1:
                    value = ensure_1d_array(value, col_name)
            except ValueError as e:
                logger.warning(f"Could not convert '{col_name}' to 1D: {e}")

        data_dict[col_name] = value
    
    # Create DataFrame
    df = pd.DataFrame(data_dict)
    
    # Add metadata
    if metadata:
        for key, value in metadata.items():
            if key in config.columns and config.columns[key].is_metadata:
                logger.debug(f"Adding metadata field '{key}'")
                df[key] = value
    
    return df


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


def create_test_dataframe_transformer(
    filesystem_provider: Optional[FileSystemProvider] = None,
    compression_provider: Optional[CompressionProvider] = None,
    pickle_provider: Optional[PickleProvider] = None,
    dataframe_provider: Optional[DataFrameProvider] = None
) -> DataFrameTransformer:
    """
    Create a DataFrameTransformer instance with test-specific dependencies.
    
    This factory function enables easy creation of DataFrameTransformer instances with
    mocked dependencies for comprehensive unit testing scenarios.
    
    Args:
        filesystem_provider: Mock filesystem operations
        compression_provider: Mock compression operations
        pickle_provider: Mock pickle operations
        dataframe_provider: Mock DataFrame operations
        
    Returns:
        DataFrameTransformer instance configured with test dependencies
    """
    test_deps = DependencyContainer(
        filesystem_provider=filesystem_provider,
        compression_provider=compression_provider,
        pickle_provider=pickle_provider,
        dataframe_provider=dataframe_provider
    )
    return DataFrameTransformer(test_deps)