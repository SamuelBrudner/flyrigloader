"""
pickle.py - Module for loading data from various pickle file formats.

This module specializes in handling different types of pickle files from fly behavior rigs, 
including compressed files, with robust error handling and standardized output.
"""

import gzip
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
from loguru import logger


def read_pickle_any_format(path: Union[str, Path]) -> Optional[Union[Dict[str, Any], pd.DataFrame]]:
    """
    Read a pickle file in any common format, with automatic detection.
    
    This function attempts multiple approaches:
    1. If file is gzipped, uses gzip.open + pickle.load
    2. If not gzipped, uses normal open + pickle.load
    3. If above fail or result isn't a dict/DataFrame, tries pd.read_pickle
    
    Args:
        path: Path to the pickle file to read
        
    Returns:
        Dictionary data (exp_matrix style) or DataFrame, or None if all approaches fail
    """
    path = Path(path)
    if not path.exists():
        logger.error(f"File not found: {path}")
        return None

    # Track attempted approaches for detailed error reporting
    errors = []

    # Approach 1: Try to read as gzipped pickle
    try:
        with gzip.open(path, 'rb') as f:
            obj = pickle.load(f)
        logger.debug(f"Successfully loaded gzipped pickle: {path}")
        return obj
    except OSError as e:
        errors.append(f"Not a gzipped file: {e}")
    except Exception as e:
        errors.append(f"Error reading as gzipped pickle: {e}")

    # Approach 2: Try to read as regular pickle
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        logger.debug(f"Successfully loaded regular pickle: {path}")
        return obj
    except Exception as e:
        errors.append(f"Error reading as regular pickle: {e}")

    # Approach 3: Try pandas read_pickle as last resort
    try:
        obj = pd.read_pickle(path)
        logger.debug(f"Successfully loaded with pd.read_pickle: {path}")
        return obj
    except Exception as e:
        errors.append(f"Error with pd.read_pickle: {e}")

    # All approaches failed
    logger.error(f"Failed to read pickle file {path} after multiple attempts: {errors}")
    return None


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
        # Neither dimension matches time array length
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
    """
    from ..schema.operations import ensure_1d as make_1d
    
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
                    value = make_1d(value, col_name)
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
    """
    # Get time dimension for validation
    if 't' not in exp_matrix:
        raise ValueError("exp_matrix must contain a 't' key for time values")
    
    # Extract regular columns
    data_dict = extract_columns_from_matrix(exp_matrix, column_list)
    
    # Add signal_disp if requested and available
    if include_signal_disp and 'signal_disp' in exp_matrix:
        data_dict['signal_disp'] = handle_signal_disp(exp_matrix)
    
    # Create DataFrame
    df = pd.DataFrame(data_dict)
    
    # Add metadata columns if provided
    if metadata:
        for key, value in metadata.items():
            df[key] = value
    
    return df


def extract_metadata_from_path(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract metadata from a file path.
    
    Args:
        path: Path to the file
        
    Returns:
        Dictionary with metadata extracted from the path
    """
    path = Path(path)
    
    # Basic metadata that can be extracted from any path
    metadata = {
        'file_path': str(path.absolute()),
        'file_name': path.name,
        'folder_name': path.parent.name,
        'date': 'unknown'  # Set default date value
    }
    
    # Extract date from parent directory name if it follows YYYY-MM-DD format
    parent_name = path.parent.name
    if parent_name and len(parent_name) >= 10:
        date_candidate = parent_name[:10]  # Extract first 10 characters
        if len(date_candidate.split('-')) == 3:  # Check if it has two hyphens
            try:
                # Validate it's a proper date format
                pd.to_datetime(date_candidate)  
                metadata['date'] = date_candidate
            except ValueError:
                # Keep using the default 'unknown' set above
                logger.debug(f"Failed to parse date from path: {parent_name}")
    
    return metadata


def load_pickle_to_dataframe(
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[pd.DataFrame]:
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
        DataFrame or None if loading fails
    """
    # Load the pickle file
    obj = read_pickle_any_format(path)
    if obj is None:
        return None
    
    # Combine metadata from path and provided metadata
    combined_metadata = extract_metadata_from_path(path)
    if metadata:
        combined_metadata.update(metadata)
    
    # If object is already a DataFrame, add metadata and return
    if isinstance(obj, pd.DataFrame):
        for key, value in combined_metadata.items():
            obj[key] = value
        return obj
    
    # If object is a dictionary, convert to DataFrame
    if isinstance(obj, dict):
        try:
            return make_dataframe_from_matrix(obj, combined_metadata)
        except Exception as e:
            logger.error(f"Error converting dictionary to DataFrame: {e}")
            return None
    
    # Unhandled object type
    logger.error(f"Unhandled object type in pickle file: {type(obj)}")
    return None