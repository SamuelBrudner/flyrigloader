"""
Module for loading data from various pickle file formats.

This module specializes in handling different types of pickle files from fly behavior rigs, 
including compressed files, with robust error handling and standardized output.
"""

import gzip
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, Optional, List, Tuple
from loguru import logger


def read_pickle_any_format(path) -> Union[Dict[str, Any], pd.DataFrame]:
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
        ValueError: If the path format is invalid
        IOError: If there are I/O errors
        PermissionError: If the file cannot be accessed due to permissions
        pickle.UnpicklingError: If the pickle format is invalid
        TypeError: If the file content is of an unexpected type
        RuntimeError: If all approaches to read the file fail
    """
    # Validate and normalize the path
    if not isinstance(path, (str, Path)):
        raise ValueError(f"Invalid path format: expected string or Path, got {type(path)}")

    path = Path(path)

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


def ensure_1d_array(array, name):
    """Stub for ensure_1d_array function that will be imported from schema_utils."""
    # This is just a stub - in reality, this will be imported from core.utils.schema_utils
    return array


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
        ValueError: If the input is not a valid dictionary
    """
    # Input validation
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
                # Special handling for signal_disp - we'll skip it here
                # as it's handled separately in other functions
                continue
                
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


def _validate_time_dimension(exp_matrix: Dict[str, Any]) -> None:
    """
    Validate that the exp_matrix contains a time dimension.
    
    Args:
        exp_matrix: Dictionary containing experimental data
        
    Raises:
        ValueError: If the exp_matrix is missing the 't' key for time values
    """
    if 't' not in exp_matrix:
        raise ValueError("exp_matrix must contain a 't' key for time values")


def _create_dataframe_direct(data_dict: Dict[str, Any]) -> Tuple[pd.DataFrame, bool]:
    """
    Attempt to create a DataFrame directly from the data dictionary.
    
    Args:
        data_dict: Dictionary of column data
        
    Returns:
        Tuple of (DataFrame or None, success flag)
    """
    try:
        df = pd.DataFrame(data_dict)
        logger.debug("Created DataFrame directly from extracted data dictionary")
        return df, True
    except ValueError as e:
        logger.debug(f"Direct DataFrame creation failed: {e}")
        return None, False


def _create_dataframe_from_time_index(data_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a DataFrame using time index as reference.
    
    Args:
        data_dict: Dictionary of column data that includes 't' key
        
    Returns:
        DataFrame with proper time indexing
    """
    if 't' not in data_dict:
        return pd.DataFrame({k: [v] for k, v in data_dict.items()})
    
    t_length = len(data_dict['t'])
    
    # First create a DataFrame with just the time column
    df = pd.DataFrame({'t': data_dict['t']})
    
    # Add each column individually
    for col, values in data_dict.items():
        if col == 't':
            continue  # Already added
        
        if isinstance(values, np.ndarray) and values.ndim > 0:
            try:
                # Try to add the column directly
                df[col] = values
            except ValueError:
                # If direct addition fails due to length mismatch, handle as object
                logger.debug(f"Column '{col}' with shape {getattr(values, 'shape', None)} doesn't match time length {t_length}, storing as object array")
                # Create a Series with objects that can be added to the DataFrame
                df[col] = pd.Series([values] * t_length, index=df.index)
        else:
            # Scalars can be broadcast automatically
            df[col] = values
            
    return df


def _create_dataframe_processed(data_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a DataFrame with careful processing of arrays.
    
    Args:
        data_dict: Dictionary of column data
        
    Returns:
        DataFrame with processed arrays
    """
    if 't' not in data_dict:
        return pd.DataFrame({k: [v] for k, v in data_dict.items()})
    
    t_length = len(data_dict['t'])
    index = range(t_length)
    
    # Pre-process the data to ensure all arrays have the right length
    processed_data = {}
    for col, values in data_dict.items():
        if col == 't':
            # Time column used directly
            processed_data[col] = data_dict['t']
        elif isinstance(values, np.ndarray) and values.ndim > 0:
            # Check if any dimension matches time length
            if len(values) == t_length:
                # First dimension matches, use directly
                processed_data[col] = values
            elif values.ndim > 1 and values.shape[1] == t_length:
                # Second dimension matches, transpose and use
                processed_data[col] = values.T
            else:
                # No dimension matches time, but include anyway as object array
                # This handles columns explicitly included by the user or test
                logger.debug(f"Column '{col}' with shape {values.shape} doesn't match time length {t_length}, storing as object")
                processed_data[col] = [values] * t_length
        else:
            # For scalars, broadcast to the entire index
            processed_data[col] = [values] * t_length
    
    # Create DataFrame with the processed data
    df = pd.DataFrame(index=index)
    
    # Add each column individually to handle any potential issues
    for col, values in processed_data.items():
        df[col] = values
    
    return df


def _add_signal_disp_to_dataframe(df: pd.DataFrame, exp_matrix: Dict[str, Any]) -> pd.DataFrame:
    """
    Add signal_disp column to DataFrame if available.
    
    Args:
        df: DataFrame to add signal_disp to
        exp_matrix: Dictionary containing signal_disp data
        
    Returns:
        DataFrame with signal_disp added (if available)
    """
    if 'signal_disp' not in exp_matrix:
        return df
    
    try:
        signal_series = handle_signal_disp(exp_matrix)
        
        # Handle different alignment cases
        if len(df) == 0:
            return pd.DataFrame({'signal_disp': signal_series})
        elif len(signal_series) == len(df):
            df['signal_disp'] = signal_series
        elif len(df) == 1:
            df.at[0, 'signal_disp'] = signal_series
        else:
            logger.warning(f"signal_disp length {len(signal_series)} doesn't match DataFrame length {len(df)}")
    except ValueError as e:
        logger.warning(f"Could not process signal_disp: {e}")
    
    return df


def _add_metadata_to_dataframe(df: pd.DataFrame, metadata: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """
    Add metadata columns to DataFrame.
    
    Args:
        df: DataFrame to add metadata to
        metadata: Dictionary of metadata to add
        
    Returns:
        DataFrame with metadata added (if provided)
    """
    if not metadata:
        return df
    
    for key, value in metadata.items():
        df[key] = value
    
    return df


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
                   or if arrays don't have a dimension matching the time dimension
        RuntimeError: If the conversion process fails for other reasons
    """
    try:
        # Validate time dimension
        _validate_time_dimension(exp_matrix)
        
        # Extract data columns
        data_dict = extract_columns_from_matrix(exp_matrix, column_names=column_list, ensure_1d=True)
        
        # Verify that all array columns match the time dimension
        if 't' in data_dict:
            t_length = len(data_dict['t'])
            for col, values in data_dict.items():
                if col == 't':
                    continue
                
                if isinstance(values, np.ndarray) and values.ndim > 0 and len(values) != t_length and all(dim != t_length for dim in values.shape):
                    raise ValueError(
                        f"Column '{col}' has shape {values.shape} with no dimension "
                        f"matching time dimension length {t_length}"
                    )
        
        # Try different DataFrame creation strategies
        try:
            # First, try direct DataFrame creation
            df = pd.DataFrame(data_dict)
            logger.debug("Created DataFrame directly from data dictionary")
        except ValueError as e:
            logger.debug(f"Direct DataFrame creation failed: {e}")
            
            # If direct creation fails, try to create with index
            if 't' in data_dict:
                t_length = len(data_dict['t'])
                df = pd.DataFrame({'t': data_dict['t']})
                
                # Add each column separately, handling dimension mismatches
                for col, values in data_dict.items():
                    if col == 't':
                        continue  # Already added
                    
                    # For arrays, check if they can be added directly
                    if isinstance(values, np.ndarray) and values.ndim > 1 and values.shape[1] == t_length:
                        df[col] = values.T
                    else:
                        df[col] = values
            else:
                # No time dimension to align with, create simple DataFrame
                df = pd.DataFrame({k: [v] for k, v in data_dict.items()})
        
        # Add signal_disp if requested
        if include_signal_disp:
            df = _add_signal_disp_to_dataframe(df, exp_matrix)
        
        # Add metadata
        df = _add_metadata_to_dataframe(df, metadata)
        
        return df
        
    except ValueError as e:
        logger.error(f"Value error when converting matrix to DataFrame: {e}")
        raise ValueError(f"Invalid matrix structure: {str(e)}") from e
    except Exception as e:
        logger.error(f"Error converting matrix to DataFrame: {e}")
        raise RuntimeError(f"Failed to convert matrix to DataFrame: {str(e)}") from e
