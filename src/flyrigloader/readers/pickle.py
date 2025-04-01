"""
pickle.py - Module for loading data from various pickle file formats.

This module specializes in handling different types of pickle files from fly behavior rigs, 
including compressed files, with robust error handling and standardized output.
"""

import gzip
import pickle
import numpy as np
import pandas as pd
import tempfile
from typing import Dict, Any, Optional, Union, Tuple, List
from loguru import logger
import contextlib

from ..core.utils import ensure_path, ensure_path_exists, PathLike
from ..core.utils.schema_utils import ensure_1d as ensure_1d_array

def read_pickle_any_format(path: PathLike) -> Union[Dict[str, Any], pd.DataFrame]:
    """
    Read a pickle file in any common format, with automatic detection.
    
    This function attempts multiple approaches:
    1. If file is gzipped, uses gzip.open + pickle.load
    2. If not gzipped, uses normal open + pickle.load
    3. If above fail or result isn't a dict/DataFrame, tries pd.read_pickle
    
    Args:
        path: Path to the pickle file to read
        
    Returns:
        Dictionary data (exp_matrix style) or DataFrame
        
    Raises:
        FileNotFoundError: If the file does not exist
        IOError: If there are I/O errors
        PermissionError: If the file cannot be accessed due to permissions
        pickle.UnpicklingError: If the pickle format is invalid
        TypeError: If the file content is of an unexpected type
        RuntimeError: If all approaches to read the file fail
    """
    # Validate and normalize the path
    try:
        path = ensure_path(path)
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid path format: {str(e)}")
        raise ValueError(f"Invalid path format: {str(e)}") from e
    
    # Check if file exists
    if not path.exists():
        error_msg = f"File not found: {path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Track attempted approaches for detailed error reporting
    errors = []

    # Approach 1: Try to read as gzipped pickle
    try:
        with gzip.open(path, 'rb') as f:
            obj = pickle.load(f)
        logger.debug(f"Successfully loaded gzipped pickle: {path}")
        return obj
    except gzip.BadGzipFile as e:
        errors.append(f"Not a valid gzipped file: {e}")
    except OSError as e:
        errors.append(f"OSError with gzipped file: {e}")
    except pickle.UnpicklingError as e:
        errors.append(f"Invalid pickle format in gzipped file: {e}")
    except PermissionError as e:
        logger.error(f"Permission denied when accessing file {path}: {e}")
        raise PermissionError(f"Cannot access file due to permissions: {e}") from e
    except EOFError as e:
        errors.append(f"Unexpected end of file in gzipped format: {e}")
    
    # Approach 2: Try to read as regular pickle
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        logger.debug(f"Successfully loaded regular pickle: {path}")
        return obj
    except pickle.UnpicklingError as e:
        errors.append(f"Invalid pickle format in regular file: {e}")
    except PermissionError as e:
        logger.error(f"Permission denied when accessing file {path}: {e}")
        raise PermissionError(f"Cannot access file due to permissions: {e}") from e
    except EOFError as e:
        errors.append(f"Unexpected end of file in regular format: {e}")
    except OSError as e:
        errors.append(f"OSError with regular file: {e}")

    # Approach 3: Try using pandas read_pickle
    try:
        obj = pd.read_pickle(path)
        logger.debug(f"Successfully loaded with pd.read_pickle: {path}")
        return obj
    except pickle.UnpicklingError as e:
        errors.append(f"Invalid pickle format for pandas: {e}")
    except Exception as e:
        # We keep a broader exception here since pandas might raise various errors
        # that we can't easily enumerate, but we log them specifically
        errors.append(f"Error with pd.read_pickle: {type(e).__name__}: {e}")

    # If we get here, all approaches failed
    error_details = "; ".join(errors)
    error_msg = f"Failed to load pickle file using all approaches: {error_details}"
    logger.error(error_msg)
    raise RuntimeError(error_msg)


def handle_signal_disp(exp_matrix: Dict[str, Any]) -> pd.Series:
    """
    Convert exp_matrix['signal_disp'] to a pandas Series of arrays.
    
    Handles different orientations of the signal_disp array:
    - (T, X) where T is the time dimension length
    - (X, T) where the time dimension is transposed
    
    Args:
        exp_matrix: Dictionary containing experimental data including signal_disp and t
        
    Returns:
        Series where each row contains an array of signal intensities
        
    Raises:
        ValueError: If signal_disp dimensions don't match expected format
    """
    if 'signal_disp' not in exp_matrix:
        raise ValueError("exp_matrix missing required 'signal_disp' key")
        
    if 't' not in exp_matrix:
        raise ValueError("exp_matrix missing required 't' key")
    
    sd = exp_matrix['signal_disp']
    t = exp_matrix['t']
    T = len(t)

    if sd.ndim != 2:
        raise ValueError(f"signal_disp must be 2D, got {sd.shape}")

    n, m = sd.shape
    if n == T:
        # Already in correct orientation
        pass
    elif m == T:
        # Transpose to match time dimension
        sd = sd.T
    else:
        # Neither dimension matches time dimension length
        raise ValueError(f"No dimension of signal_disp {sd.shape} matches time dimension length T={T}")

    # Create a Series where each element is the signal array for one timepoint
    return pd.Series(list(sd), index=range(T), name='signal_disp')


def extract_columns_from_matrix(
    exp_matrix: Dict[str, Any],
    column_names: Optional[List[str]] = None,
    ensure_1d: bool = True
) -> Dict[str, np.ndarray]:
    """
    Extract columns from an exp_matrix as 1D arrays.
    
    Args:
        exp_matrix: Dictionary containing experimental data
        column_names: List of column names to extract. If None, extracts all keys.
        ensure_1d: Whether to convert arrays to 1D (True) or keep original dimensions
        
    Returns:
        Dictionary mapping column names to 1D arrays
        
    Raises:
        ValueError: If the input is not a valid dictionary or critical columns are missing
    """
    if not isinstance(exp_matrix, dict):
        raise ValueError(f"exp_matrix must be a dictionary, got {type(exp_matrix)}")
    
    # Use all keys if column_names not specified
    if column_names is None:
        column_names = list(exp_matrix.keys())
    
    # Initialize result dictionary
    result = {}
    
    # Extract each column
    for col_name in column_names:
        if col_name in exp_matrix:
            if col_name == 'signal_disp':
                # Special handling for signal_disp
                continue  # We'll handle this separately
                
            # Extract and potentially convert to 1D
            value = exp_matrix[col_name]
            if ensure_1d and hasattr(value, 'ndim') and value.ndim > 1:
                try:
                    value = ensure_1d_array(value, col_name)
                except ValueError as e:
                    logger.warning(f"Could not convert '{col_name}' to 1D: {e}")
                    # Skip this column if we can't convert it and ensure_1d is required
                    if ensure_1d:
                        continue
            
            result[col_name] = value
    
    return result


def make_dataframe_from_matrix(
    exp_matrix: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    include_signal_disp: bool = True,
    column_list: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convert an exp_matrix dictionary to a DataFrame with metadata.
    
    Args:
        exp_matrix: Dictionary containing experimental data
        metadata: Dictionary with metadata to add as columns
        include_signal_disp: Whether to include the signal_disp column
        column_list: Optional list of specific columns to include
        
    Returns:
        DataFrame with time series and metadata columns
        
    Raises:
        ValueError: If the exp_matrix is missing required keys or has invalid structure
        RuntimeError: If the conversion process fails for other reasons
    """
    try:
        # Get time dimension for validation
        if 't' not in exp_matrix:
            raise ValueError("exp_matrix must contain a 't' key for time values")
        
        # Extract regular columns
        data_dict = extract_columns_from_matrix(exp_matrix, column_list)
        
        # Add signal_disp if requested and available
        if include_signal_disp and 'signal_disp' in exp_matrix:
            signal_disp = handle_signal_disp(exp_matrix)
            data_dict['signal_disp'] = signal_disp
        
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        
        # Add metadata columns if provided
        if metadata:
            for key, value in metadata.items():
                df[key] = value
        
        return df
    except ValueError as e:
        # Re-raise ValueError exceptions with the original message
        logger.error(f"Value error when converting matrix to DataFrame: {e}")
        raise ValueError(f"Invalid matrix structure: {str(e)}") from e
    except Exception as e:
        # Catch and convert other exceptions to RuntimeError
        logger.error(f"Error converting matrix to DataFrame: {e}")
        raise RuntimeError(f"Failed to convert matrix to DataFrame: {str(e)}") from e


def _add_metadata_to_dataframe(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
    """
    Add metadata columns to a DataFrame.
    
    Args:
        df: DataFrame to add metadata to
        metadata: Dictionary of metadata key-value pairs
        
    Returns:
        DataFrame with metadata columns added
    """
    df_copy = df.copy()
    for key, value in metadata.items():
        df_copy[key] = value
    return df_copy


def extract_metadata_from_path(path: PathLike) -> Dict[str, Any]:
    """
    Extract metadata from a file path based on its structure and naming.
    
    This function parses path components to extract meaningful metadata
    such as experiment date, fly ID, genotype, etc.
    
    Args:
        path: Path to extract metadata from
        
    Returns:
        Dictionary of extracted metadata
        
    Raises:
        ValueError: If the path cannot be parsed properly
    """
    try:
        path = ensure_path(path)
        
        return {
            'file_path': str(path),
            'file_name': path.name
        }
    except Exception as e:
        logger.error(f"Error extracting metadata from path {path}: {e}")
        raise ValueError(f"Failed to extract metadata from path: {str(e)}") from e


def load_pickle_to_dataframe(
    path: PathLike,
    metadata: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Load a pickle file and convert to a DataFrame with metadata.
    
    This function handles the complete workflow:
    1. Load pickle file in any format
    2. Extract data columns and metadata
    3. Create a DataFrame with appropriate structure
    
    Args:
        path: Path to the pickle file
        metadata: Optional additional metadata to include
        
    Returns:
        DataFrame with the pickle contents and metadata
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the path cannot be parsed for metadata or the file content is invalid
        TypeError: If the file content is of an unexpected type
        pickle.UnpicklingError: If the pickle format is invalid
        RuntimeError: If the file cannot be converted to a DataFrame for other reasons
    """
    # Validate the path
    try:
        path_obj = ensure_path_exists(path)
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid path format: {str(e)}")
        raise ValueError(f"Invalid path format: {str(e)}") from e
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise FileNotFoundError(f"File not found: {str(e)}") from e
    
    # Initialize metadata if not provided
    if metadata is None:
        metadata = {}
    
    # Add path and filename to metadata
    metadata.update({
        'source_path': str(path_obj),
        'filename': path_obj.name
    })
    
    # Try to extract additional metadata from the path
    try:
        path_metadata = extract_metadata_from_path(path_obj)
        metadata.update(path_metadata)
    except ValueError as e:
        # Log but don't fail if metadata extraction fails
        logger.warning(f"Could not extract metadata from path {path_obj}: {str(e)}")
    
    # Load the pickle file
    try:
        data = read_pickle_any_format(path_obj)
    except (FileNotFoundError, PermissionError) as e:
        # These exceptions should be propagated directly
        raise
    except pickle.UnpicklingError as e:
        logger.error(f"Invalid pickle format in {path_obj}: {str(e)}")
        raise pickle.UnpicklingError(f"Invalid pickle format: {str(e)}") from e
    except RuntimeError as e:
        logger.error(f"Failed to read pickle file {path_obj}: {str(e)}")
        raise RuntimeError(f"Failed to read pickle file: {str(e)}") from e
    
    # Handle different data types
    try:
        # If already a DataFrame, add metadata and return
        if isinstance(data, pd.DataFrame):
            logger.debug(f"Data from {path_obj} already in DataFrame format")
            return _add_metadata_to_dataframe(data, metadata)
            
        # If dictionary (exp_matrix format), convert to DataFrame
        if isinstance(data, dict):
            logger.debug(f"Converting dictionary data from {path_obj} to DataFrame")
            return make_dataframe_from_matrix(data, metadata)
            
        # Unsupported data type
        raise TypeError(f"Unsupported data type from pickle: {type(data).__name__}")
        
    except KeyError as e:
        logger.error(f"Missing required key in data from {path_obj}: {str(e)}")
        raise ValueError(f"Missing required key in data: {str(e)}") from e
    except (TypeError, ValueError) as e:
        logger.error(f"Error converting data from {path_obj} to DataFrame: {str(e)}")
        raise
    except Exception as e:
        # Last resort for truly unexpected errors
        logger.error(f"Unexpected error converting data from {path_obj} to DataFrame: {type(e).__name__}: {str(e)}")
        raise RuntimeError(f"Failed to convert pickle data to DataFrame: {str(e)}") from e