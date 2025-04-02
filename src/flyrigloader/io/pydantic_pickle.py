"""
Data processing functions for experimental data using Pydantic models.

This module provides functions for converting experimental data matrices
into pandas DataFrames based on column configurations defined with Pydantic.
"""

from typing import Dict, Any, Optional, Union, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from flyrigloader.io.column_models import (
    load_column_config, 
    ColumnConfigDict, 
    ColumnConfig, 
    SpecialHandlerType
)


def _validate_required_columns(exp_matrix: Dict[str, Any], config: ColumnConfigDict) -> List[str]:
    """
    Validate that all required columns are present in the exp_matrix.
    
    Args:
        exp_matrix: Dictionary containing experimental data.
        config: Column configuration dictionary.
        
    Returns:
        list: Missing required columns.
    """
    missing_columns = []
    
    for col_name, col_config in config.columns.items():
        if not col_config.required or col_config.is_metadata:
            continue
            
        if col_name in exp_matrix:
            continue
            
        # Check if there's an alias for this column
        alias = col_config.alias
        if alias and alias in exp_matrix:
            logger.debug(f"Required column '{col_name}' found via alias '{alias}'")
            continue
            
        missing_columns.append(col_name)
    
    return missing_columns


def _extract_first_column(array: np.ndarray) -> np.ndarray:
    """
    Extract the first column from a 2D array.
    
    Args:
        array: NumPy array to process.
        
    Returns:
        np.ndarray: First column of the array if 2D, otherwise the original array.
    """
    arr = np.asarray(array)
    return arr[:, 0] if arr.ndim == 2 and arr.shape[1] > 1 else arr


def _ensure_1d(array: np.ndarray, name: str) -> np.ndarray:
    """
    Convert array to a 1D NumPy array if possible. 
    Raises ValueError if the array has more than 1 column.
    
    Args:
        array: NumPy array to process.
        name: Name of the column being processed.
        
    Returns:
        np.ndarray: 1D array.
        
    Raises:
        ValueError: If the array cannot be converted to 1D.
    """
    arr = np.asarray(array)
    # If shape is (N, 1) or (1, N), ravel to (N,)
    if arr.ndim == 2 and 1 in arr.shape:
        return arr.ravel()
    # If it's 1D, just return it
    elif arr.ndim == 1:
        return arr
    else:
        raise ValueError(
            f"Column '{name}' has shape {arr.shape}, which is not strictly 1D. "
            "If you need NxM data, store each column separately or flatten the data explicitly."
        )


def _handle_signal_disp(exp_matrix: Dict[str, Any], col_name: str = "signal_disp") -> pd.Series:
    """
    Process signal_disp array to match the time dimension.
    
    We detect which axis matches the time dimension, transpose if necessary,
    and store the 'other' axis as a small array. For example, if shape is
    (15, 54953), we transpose to (54953, 15). The resulting Series has 54953
    entries, each of which is an array of length 15.
    
    Args:
        exp_matrix: Dictionary containing experimental data.
        col_name: Name of the column to process.
        
    Returns:
        pd.Series: Series with arrays as values.
        
    Raises:
        ValueError: If the data cannot be transformed to match the time dimension.
    """
    sd = exp_matrix[col_name]
    t = exp_matrix['t']
    T = len(t)

    if sd.ndim != 2:
        logger.warning(f"Expected 2D data for '{col_name}', got shape {sd.shape}. Attempting to convert.")
        sd = np.atleast_2d(sd)
        
    n, m = sd.shape
    if n == T:
        # shape is (T, leftover), which is already correct
        pass
    elif m == T:
        # shape is (leftover, T) -> transpose so time is the first axis
        logger.debug(f"Transposing {col_name} to match time dimension")
        sd = sd.T
    else:
        raise ValueError(
            f"Neither dimension matches t.size={T} for '{col_name}'. "
            f"Shape is {sd.shape}."
        )

    # Now sd has shape (T, leftover) or (T,). We create a Series of length T,
    # each entry is an array from that row.
    if sd.ndim == 2:
        return pd.Series(list(sd), index=range(T), name=col_name)
    else:
        return pd.Series(sd, index=range(T), name=col_name)


def _apply_special_handler(
    exp_matrix: Dict[str, Any], 
    col_name: str, 
    value: Any, 
    special_handler: SpecialHandlerType,
    handler_name: Optional[str] = None
) -> Any:
    """
    Apply a special handler to a column value.
    
    Args:
        exp_matrix: Dictionary containing experimental data.
        col_name: Name of the column being processed.
        value: Current value for the column.
        special_handler: Type of special handling to apply.
        handler_name: Optional name of handler function.
        
    Returns:
        Processed value.
    """
    if handler_name and handler_name.startswith('_'):
        if handler_func := globals().get(handler_name):
            logger.debug(f"Applying special handler '{handler_name}' to column '{col_name}'")
            if handler_name == '_handle_signal_disp':
                return handler_func(exp_matrix, col_name)
            return handler_func(value)
        else:
            logger.warning(f"Special handler '{handler_name}' not found, using raw value for '{col_name}'")
            return value

    # Use the enum value to determine the handler
    if special_handler == SpecialHandlerType.EXTRACT_FIRST_COLUMN:
        logger.debug(f"Extracting first column from 2D array for '{col_name}'")
        return _extract_first_column(value)
    elif special_handler == SpecialHandlerType.TRANSFORM_TIME_DIMENSION:
        logger.debug(f"Transforming '{col_name}' to match time dimension")
        return _handle_signal_disp(exp_matrix, col_name)

    return value


def _process_column(
    exp_matrix: Dict[str, Any], 
    col_name: str, 
    col_config: ColumnConfig,
    config: ColumnConfigDict
) -> Optional[Tuple[str, Any]]:
    """
    Process a single column according to its configuration.
    
    Args:
        exp_matrix: Dictionary containing experimental data.
        col_name: Name of the column to process.
        col_config: Configuration for the column.
        config: Complete column configuration dictionary.
        
    Returns:
        tuple: (column_name, processed_value) or None if column should be skipped.
    """
    # Skip metadata columns
    if col_config.is_metadata:
        return None
        
    # Handle aliases
    source_col = col_name
    if col_config.alias and col_config.alias in exp_matrix:
        source_col = col_config.alias
        logger.debug(f"Using alias '{source_col}' for column '{col_name}'")
        
    # If column is not in the matrix
    if source_col not in exp_matrix:
        if not col_config.required and hasattr(col_config, 'default_value'):
            logger.debug(f"Using default value for missing column '{col_name}'")
            return col_name, col_config.default_value
        return None
            
    # Get the value from the exp_matrix
    value = exp_matrix[source_col]
    
    # Apply any special handling
    if col_config.special_handling:
        handler_name = config.special_handlers.get(col_config.special_handling.value)
        value = _apply_special_handler(
            exp_matrix, 
            col_name, 
            value, 
            col_config.special_handling, 
            handler_name
        )
    
    # Ensure correct dimensionality for numpy arrays
    if isinstance(value, np.ndarray) and col_config.dimension:
        try:
            if col_config.dimension.value == 1:
                value = _ensure_1d(value, col_name)
        except ValueError as e:
            logger.warning(f"Could not ensure 1D for column '{col_name}': {e}")
        
    return col_name, value


def make_dataframe_from_pydantic_config(
    exp_matrix: Dict[str, Any], 
    config_path: str, 
    metadata: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Convert an exp_matrix dictionary to a DataFrame based on Pydantic column configuration.
    
    Args:
        exp_matrix: Dictionary containing experimental data.
        config_path: Path to the column configuration YAML file.
        metadata: Optional dictionary with metadata to add to the DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame containing the data with correct types.
        
    Raises:
        ValueError: If required columns are missing.
        ValidationError: If the configuration is invalid.
    """
    # Load and validate configuration
    config = load_column_config(config_path)

    if missing_columns := _validate_required_columns(exp_matrix, config):
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    # Process columns
    data_dict = {}
    for col_name, col_config in config.columns.items():
        if result := _process_column(exp_matrix, col_name, col_config, config):
            col_name, value = result
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
