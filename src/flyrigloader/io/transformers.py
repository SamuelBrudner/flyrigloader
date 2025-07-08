"""
Dedicated transformation module providing optional DataFrame utilities for the decoupled I/O architecture.

This module extracts DataFrame transformation utilities from pickle.py to create a clean separation
between data loading and transformation operations. It provides optional utilities for converting
raw experimental data into standardized Pandas DataFrames with comprehensive configuration support
and schema validation.

The transformation utilities support:
- Configurable DataFrame creation from raw data structures
- Domain-specific column transformations with metadata preservation  
- Signal display data handling with proper dimensionality transformation
- Column extraction and validation utilities
- Dimension transformation functions for array processing

This separation enables the new manifest-based workflow where data discovery, loading, and
transformation are decoupled for better control over memory usage and processing pipelines.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Optional, List

# Internal imports for configuration and logging
from flyrigloader import logger
from flyrigloader.io.column_models import get_config_from_source, ColumnConfigDict, SpecialHandlerType


class DataFrameTransformer:
    """
    Modular DataFrame transformation component providing optional utilities for converting
    raw experimental data into standardized Pandas DataFrames.
    
    This class implements the decoupled transformation architecture per TST-REF-002 requirements,
    separating DataFrame creation and transformation operations from data loading to reduce
    coupling and enable comprehensive control over the transformation pipeline.
    
    The transformer supports configurable column handling, metadata preservation, and 
    domain-specific transformations while maintaining backward compatibility with existing APIs.
    """
    
    def __init__(self):
        """Initialize DataFrameTransformer with logging support."""
        logger.debug("Initialized DataFrameTransformer for optional DataFrame transformation")
    
    def validate_matrix_input(self, exp_matrix: Any) -> Dict[str, Any]:
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
    
    def validate_time_dimension(self, exp_matrix: Dict[str, Any]) -> int:
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

    def transform_data(
        self,
        exp_matrix: Dict[str, Any],
        config_source: Union[str, Dict[str, Any], ColumnConfigDict, None] = None,
        metadata: Optional[Dict[str, Any]] = None,
        skip_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Transform experimental data into a standardized DataFrame with comprehensive configuration support.
        
        This is the primary transformation method that provides configurable DataFrame creation
        with enhanced validation, column handling, and metadata integration per the new
        decoupled architecture requirements.
        
        Args:
            exp_matrix: Dictionary containing experimental data
            config_source: Configuration source (path, dict, ColumnConfigDict, or None)
            metadata: Optional dictionary with metadata to add to the DataFrame
            skip_columns: Optional list of columns to exclude from processing
        
        Returns:
            pd.DataFrame: DataFrame containing the transformed data with correct types
            
        Raises:
            ValueError: If required columns are missing or configuration is invalid
            TypeError: If the input types are invalid
        """
        # Validate input matrix
        validated_matrix = self.validate_matrix_input(exp_matrix)
        
        # Load and validate configuration
        config = get_config_from_source(config_source)
        skip_columns = skip_columns or []
        
        logger.debug(f"Starting DataFrame transformation with {len(config.columns)} configured columns")
        
        # Simple validation for required columns (backward compatibility)
        missing_columns = []
        for col_name, col_config in config.columns.items():
            if col_name in skip_columns or not col_config.required or col_config.is_metadata:
                continue
                
            if col_name in validated_matrix:
                continue
                
            # Check if there's an alias for this column
            alias = col_config.alias
            if alias and alias in validated_matrix:
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
            if col_config.alias and col_config.alias in validated_matrix:
                source_col = col_config.alias
                logger.debug(f"Using alias '{source_col}' for column '{col_name}'")

            # If column is not in the matrix
            if source_col not in validated_matrix:
                if not col_config.required and hasattr(col_config, 'default_value'):
                    logger.debug(f"Using default value for missing column '{col_name}'")
                    data_dict[col_name] = col_config.default_value
                continue
                    
            # Get the value from the exp_matrix
            value = validated_matrix[source_col]

            # Apply special handling if configured
            if col_config.special_handling:
                if col_config.special_handling == SpecialHandlerType.EXTRACT_FIRST_COLUMN:
                    logger.debug(f"Extracting first column from 2D array for '{col_name}'")
                    if hasattr(value, 'ndim') and value.ndim == 2:
                        value = value[:, 0] if value.shape[1] > 0 else value
                elif col_config.special_handling == SpecialHandlerType.TRANSFORM_TIME_DIMENSION:
                    logger.debug(f"Transforming '{col_name}' to match time dimension")
                    value = handle_signal_disp(validated_matrix)

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
        
        logger.info(f"Successfully transformed data to DataFrame with shape {df.shape}")
        return df


def make_dataframe_from_config(
    exp_matrix: Dict[str, Any],
    config_source: Union[str, Dict[str, Any], ColumnConfigDict, None] = None,
    metadata: Optional[Dict[str, Any]] = None,
    skip_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Enhanced convenience function for DataFrame creation with comprehensive configuration support.
    
    This function provides a simple interface for creating DataFrames from experimental data
    using the DataFrameTransformer while maintaining backward compatibility with existing APIs.
    
    Args:
        exp_matrix: Dictionary containing experimental data
        config_source: Configuration source (path, dict, ColumnConfigDict, or None)
        metadata: Optional dictionary with metadata to add to the DataFrame
        skip_columns: Optional list of columns to exclude from processing
    
    Returns:
        pd.DataFrame: DataFrame containing the data with correct types
        
    Raises:
        ValueError: If required columns are missing or configuration is invalid
        TypeError: If the config_source type is invalid
    """
    transformer = DataFrameTransformer()
    return transformer.transform_data(exp_matrix, config_source, metadata, skip_columns)


def handle_signal_disp(exp_matrix: Dict[str, Any]) -> pd.Series:
    """
    Handle signal display data with proper dimensionality transformation.
    
    This function specializes in processing signal_disp arrays from experimental data,
    ensuring proper orientation and creating a Series where each row contains an array
    of signal intensities corresponding to a time point.
    
    Args:
        exp_matrix: Dictionary containing experimental data including signal_disp and t
        
    Returns:
        Series where each row contains an array of signal intensities
        
    Raises:
        ValueError: If signal_disp dimensions don't match expected format or required keys are missing
    """
    logger.debug("Processing signal_disp data for time dimension transformation")
    
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

    result = pd.Series(
        [sd[i, :] for i in range(T)],  # One row per timepoint
        index=range(T),  # Use time indices
        name='signal_disp',  # Name the series
    )
    
    logger.debug(f"Created signal_disp Series with {len(result)} time points")
    return result


def extract_columns_from_matrix(
    exp_matrix: Dict[str, Any],
    column_names: Optional[List[str]] = None,
    ensure_1d: bool = True
) -> Dict[str, np.ndarray]:
    """
    Extract and process specific columns from experimental data matrix.
    
    This utility function provides column extraction capabilities with optional 1D conversion,
    supporting the decoupled architecture by allowing selective processing of data columns
    without requiring full DataFrame transformation.
    
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
    if not isinstance(exp_matrix, dict):
        raise ValueError(f"exp_matrix must be a dictionary, got {type(exp_matrix)}")
    
    if column_names is None:
        column_names = list(exp_matrix.keys())
    
    logger.debug(f"Extracting {len(column_names)} columns from matrix")
    
    result = {}
    for col_name in column_names:
        if col_name in exp_matrix:
            if col_name == 'signal_disp':
                logger.debug(f"Skipping special column '{col_name}' in basic extraction")
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
            logger.debug(f"Extracted column '{col_name}' with shape {getattr(value, 'shape', 'scalar')}")
    
    logger.debug(f"Successfully extracted {len(result)} columns")
    return result


def ensure_1d_array(array, name):
    """
    Enhanced utility function for ensuring 1D array format with detailed error reporting.
    
    This function provides improved error handling and logging for better transformation
    observability and validation capabilities per Section 2.2.8 requirements.
    
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


def transform_to_dataframe(
    exp_matrix: Dict[str, Any],
    config_source: Union[str, Dict[str, Any], ColumnConfigDict, None] = None,
    metadata: Optional[Dict[str, Any]] = None,
    skip_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Primary entry point for transforming raw experimental data to DataFrame format.
    
    This function serves as the main interface for the new decoupled transformation workflow,
    providing comprehensive DataFrame creation with configuration support, metadata integration,
    and optional column processing as part of the manifest-based data loading architecture.
    
    Args:
        exp_matrix: Dictionary containing raw experimental data
        config_source: Configuration source (path, dict, ColumnConfigDict, or None)
        metadata: Optional metadata to add to the resulting DataFrame
        skip_columns: Optional list of columns to exclude from processing
        
    Returns:
        pd.DataFrame: Fully transformed DataFrame with validated columns and metadata
        
    Raises:
        ValueError: If transformation fails or required data is missing
        TypeError: If input types are invalid
    """
    logger.info("Starting DataFrame transformation with decoupled architecture")
    
    transformer = DataFrameTransformer()
    result = transformer.transform_data(exp_matrix, config_source, metadata, skip_columns)
    
    logger.info(f"Completed DataFrame transformation: {result.shape} DataFrame with {len(result.columns)} columns")
    return result