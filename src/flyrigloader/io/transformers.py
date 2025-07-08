"""
Module for DataFrame transformation utilities extracted from pickle.py.

This module provides optional DataFrame creation and transformation utilities
that are decoupled from the core data loading operations. It includes the
DataFrameTransformer class and various utility functions for converting
raw experimental data into pandas DataFrames with proper schema validation
and metadata handling.

Key transformations include:
- Converting experimental data matrices to DataFrames
- Handling signal display data with proper dimensionality
- Extracting columns from complex data structures
- Applying column configurations and validations
- Managing metadata integration
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, Optional, List, Tuple, Protocol, runtime_checkable

from flyrigloader import logger
from flyrigloader.io.column_models import (
    get_config_from_source, 
    ColumnConfigDict, 
    ColumnConfig,
    SpecialHandlerType
)


# Dependency Injection Protocols for Testing

@runtime_checkable
class DataFrameProvider(Protocol):
    """Protocol for pandas DataFrame operations to enable dependency injection."""
    
    def create_dataframe(self, data: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """Create a DataFrame from data."""
        ...
    
    def create_series(self, data: Any, **kwargs) -> pd.Series:
        """Create a Series from data."""
        ...


@runtime_checkable
class ArrayProvider(Protocol):
    """Protocol for numpy array operations to enable dependency injection."""
    
    def asarray(self, data: Any) -> np.ndarray:
        """Convert data to numpy array."""
        ...
    
    def ensure_1d(self, array: np.ndarray, name: Optional[str] = None) -> np.ndarray:
        """Ensure array is 1D."""
        ...


class DefaultDataFrameProvider:
    """Default implementation of DataFrameProvider using pandas."""
    
    def create_dataframe(self, data: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """Create a DataFrame using pandas."""
        return pd.DataFrame(data, **kwargs)
    
    def create_series(self, data: Any, **kwargs) -> pd.Series:
        """Create a Series using pandas."""
        return pd.Series(data, **kwargs)


class DefaultArrayProvider:
    """Default implementation of ArrayProvider using numpy."""
    
    def asarray(self, data: Any) -> np.ndarray:
        """Convert data to numpy array."""
        return np.asarray(data)
    
    def ensure_1d(self, array: np.ndarray, name: Optional[str] = None) -> np.ndarray:
        """Ensure array is 1D by flattening if necessary."""
        if array.ndim > 1:
            logger.debug(f"Flattening {array.ndim}D array {name or 'unknown'} to 1D")
            return array.flatten()
        return array


class TransformerDependencyContainer:
    """Container for managing transformer dependencies."""
    
    def __init__(
        self,
        dataframe_provider: Optional[DataFrameProvider] = None,
        array_provider: Optional[ArrayProvider] = None
    ):
        self.dataframe = dataframe_provider or DefaultDataFrameProvider()
        self.array = array_provider or DefaultArrayProvider()
        logger.debug("Initialized TransformerDependencyContainer")


# Global default dependency container
_default_transformer_deps = TransformerDependencyContainer()


class DataFrameTransformer:
    """
    Enhanced DataFrame transformation class with dependency injection support.
    
    This class provides utilities for converting raw experimental data
    into pandas DataFrames with proper schema validation, metadata handling,
    and special transformations for signal processing.
    """
    
    def __init__(self, dependencies: Optional[TransformerDependencyContainer] = None):
        """
        Initialize DataFrameTransformer with configurable dependencies.
        
        Args:
            dependencies: Dependency container for injectable providers
        """
        self.deps = dependencies or _default_transformer_deps
        logger.debug("Initialized DataFrameTransformer with dependency injection")
    
    def transform_data(
        self,
        exp_matrix: Dict[str, Any],
        config_source: Union[str, Path, Dict[str, Any], ColumnConfigDict],
        metadata: Optional[Dict[str, Any]] = None,
        include_file_path: bool = False,
        file_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Transform experimental data matrix to DataFrame with configuration.
        
        Args:
            exp_matrix: Raw experimental data dictionary
            config_source: Column configuration source
            metadata: Optional metadata to add to DataFrame
            include_file_path: Whether to include file_path column
            file_path: Path to add if include_file_path is True
            
        Returns:
            Configured pandas DataFrame
        """
        # Load and validate configuration
        config = get_config_from_source(config_source)
        
        # Only extract columns that are defined in the configuration
        columns_to_extract = [col_name for col_name in config.columns.keys() 
                            if col_name in exp_matrix and not config.columns[col_name].special_handling]
        extracted_data = extract_columns_from_matrix(exp_matrix, columns_to_extract)
        
        # Apply special handlers for columns that have special handling
        for col_name, col_config in config.columns.items():
            if col_config.special_handling and col_name in exp_matrix:
                if col_config.special_handling == SpecialHandlerType.TRANSFORM_TIME_DIMENSION:
                    # Apply signal display transformation
                    extracted_data[col_name] = handle_signal_disp(exp_matrix)
                elif col_config.special_handling == SpecialHandlerType.EXTRACT_FIRST_COLUMN:
                    # Apply first column extraction - get raw data and transform it
                    raw_data = exp_matrix[col_name]
                    extracted_data[col_name] = _extract_first_column(raw_data)
        
        # Create DataFrame
        df = self.deps.dataframe.create_dataframe(extracted_data)
        
        # Add metadata if provided
        if metadata:
            for key, value in metadata.items():
                df[key] = value
        
        # Add file path if requested
        if include_file_path and file_path:
            df['file_path'] = str(file_path)
        
        logger.debug(f"Transformed data to DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def validate_matrix_input(self, exp_matrix: Dict[str, Any]) -> None:
        """
        Validate that experimental matrix has required structure.
        
        Args:
            exp_matrix: Experimental data matrix to validate
            
        Raises:
            ValueError: If matrix structure is invalid
        """
        if not isinstance(exp_matrix, dict):
            raise ValueError("exp_matrix must be a dictionary")
        
        if not exp_matrix:
            raise ValueError("exp_matrix cannot be empty")
        
        # Check for time dimension
        if 't' not in exp_matrix:
            logger.warning("exp_matrix missing 't' (time) column")
    
    def validate_time_dimension(self, exp_matrix: Dict[str, Any]) -> int:
        """
        Validate and return the time dimension length.
        
        Args:
            exp_matrix: Experimental data matrix
            
        Returns:
            Length of time dimension
            
        Raises:
            ValueError: If time dimension is invalid
        """
        if 't' not in exp_matrix:
            raise ValueError("exp_matrix missing required 't' key")
        
        time_data = exp_matrix['t']
        if not isinstance(time_data, np.ndarray):
            time_data = np.asarray(time_data)
        
        if time_data.ndim != 1:
            raise ValueError("Time dimension 't' must be 1D array")
        
        return len(time_data)


def ensure_1d_array(array: np.ndarray, name: Optional[str] = None) -> np.ndarray:
    """
    Ensure array is 1-dimensional by flattening if necessary.
    
    Args:
        array: Input array to ensure is 1D
        name: Optional name for logging
        
    Returns:
        1D numpy array
    """
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    
    if array.ndim > 1:
        logger.debug(f"Flattening {array.ndim}D array {name or 'unknown'} to 1D")
        return array.flatten()
    
    return array


def extract_columns_from_matrix(
    exp_matrix: Dict[str, Any], 
    columns_to_extract: Optional[List[str]] = None,
    ensure_1d: bool = False
) -> Dict[str, Any]:
    """
    Extract specified columns from experimental data matrix.
    
    Args:
        exp_matrix: Experimental data dictionary
        columns_to_extract: List of column names to extract, or None for all
        ensure_1d: Whether to ensure arrays are 1D
        
    Returns:
        Dictionary with extracted columns
        
    Raises:
        ValueError: If exp_matrix is not a dictionary
    """
    if not isinstance(exp_matrix, dict):
        raise ValueError("exp_matrix must be a dictionary")
    
    # If no columns specified, extract all except signal_disp (handled separately)
    if columns_to_extract is None:
        columns_to_extract = [k for k in exp_matrix.keys() if k != 'signal_disp']
    
    result = {}
    for col_name in columns_to_extract:
        if col_name in exp_matrix:
            value = exp_matrix[col_name]
            if isinstance(value, np.ndarray) and ensure_1d:
                result[col_name] = ensure_1d_array(value, col_name)
            else:
                result[col_name] = value
    
    logger.debug(f"Extracted {len(result)} columns from matrix")
    return result


def handle_signal_disp(exp_matrix: Dict[str, Any]) -> pd.Series:
    """
    Handle signal display data transformation with proper dimensionality.
    
    This function transforms 2D signal display data into a pandas Series
    where each element is a 1D array representing signal data at that time point.
    
    Args:
        exp_matrix: Experimental data matrix containing 'signal_disp' and 't' keys
        
    Returns:
        pandas Series with name 'signal_disp' containing arrays for each time point
        
    Raises:
        ValueError: If required keys are missing or dimensions are invalid
    """
    # Validate required keys
    if 'signal_disp' not in exp_matrix:
        raise ValueError("exp_matrix missing required 'signal_disp' key")
    
    if 't' not in exp_matrix:
        raise ValueError("exp_matrix missing required 't' key")
    
    signal_disp = exp_matrix['signal_disp']
    time_data = exp_matrix['t']
    
    # Ensure signal_disp is numpy array
    if not isinstance(signal_disp, np.ndarray):
        signal_disp = np.asarray(signal_disp)
    
    # Ensure time_data is numpy array
    if not isinstance(time_data, np.ndarray):
        time_data = np.asarray(time_data)
    
    # Validate dimensions
    if signal_disp.ndim != 2:
        raise ValueError("signal_disp must be 2D array")
    
    time_length = len(time_data)
    
    # Determine orientation: (T, X) or (X, T)
    if signal_disp.shape[0] == time_length:
        # (T, X) orientation - time is first dimension
        logger.debug(f"Processing signal_disp in (T, X) orientation: {signal_disp.shape}")
        signal_arrays = [signal_disp[i, :] for i in range(time_length)]
    elif signal_disp.shape[1] == time_length:
        # (X, T) orientation - time is second dimension
        logger.debug(f"Processing signal_disp in (X, T) orientation: {signal_disp.shape}")
        signal_arrays = [signal_disp[:, i] for i in range(time_length)]
    else:
        raise ValueError(
            f"No dimension of signal_disp {signal_disp.shape} matches time dimension {time_length}"
        )
    
    # Create Series with proper name
    result = pd.Series(signal_arrays, name='signal_disp')
    logger.debug(f"Created signal_disp Series with {len(result)} time points")
    return result


def _extract_first_column(data: Any) -> Any:
    """
    Extract first column from 2D arrays, pass through other data types.
    
    Args:
        data: Input data (array or other)
        
    Returns:
        First column if 2D array, otherwise original data
    """
    if isinstance(data, np.ndarray) and data.ndim == 2:
        logger.debug(f"Extracting first column from 2D array shape {data.shape}")
        return data[:, 0]
    return data


def _handle_signal_disp(exp_matrix: Dict[str, Any]) -> pd.Series:
    """
    Alias for handle_signal_disp for backward compatibility.
    
    This function name is referenced in column_config.yaml special_handlers.
    """
    return handle_signal_disp(exp_matrix)


def make_dataframe_from_config(
    exp_matrix: Dict[str, Any],
    config_source: Union[str, Path, Dict[str, Any], ColumnConfigDict],
    metadata: Optional[Dict[str, Any]] = None,
    include_file_path: bool = False,
    file_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Create a pandas DataFrame from experimental data using column configuration.
    
    Args:
        exp_matrix: Raw experimental data dictionary
        config_source: Column configuration source (file path, dict, or model)
        metadata: Optional metadata to add to DataFrame
        include_file_path: Whether to include file_path column
        file_path: Path to add if include_file_path is True
        
    Returns:
        Configured pandas DataFrame with proper column handling
        
    Raises:
        ValueError: If input validation fails
    """
    transformer = DataFrameTransformer()
    return transformer.transform_data(
        exp_matrix=exp_matrix,
        config_source=config_source,
        metadata=metadata,
        include_file_path=include_file_path,
        file_path=file_path
    )


def transform_to_dataframe(
    data: Dict[str, Any],
    config_source: Optional[Union[str, Path, Dict[str, Any], ColumnConfigDict]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Transform raw data to DataFrame with optional configuration.
    
    This is a convenience function for the new decoupled workflow.
    
    Args:
        data: Raw data dictionary
        config_source: Optional column configuration
        metadata: Optional metadata to add
        
    Returns:
        pandas DataFrame
    """
    if config_source is None:
        # Simple DataFrame creation without column configuration
        return pd.DataFrame(data)
    
    return make_dataframe_from_config(
        exp_matrix=data,
        config_source=config_source,
        metadata=metadata
    )


# Test helper functions

def create_test_dataframe_transformer(
    dataframe_provider: Optional[DataFrameProvider] = None,
    array_provider: Optional[ArrayProvider] = None
) -> DataFrameTransformer:
    """
    Create a DataFrameTransformer instance with test-specific dependencies.
    
    Args:
        dataframe_provider: Mock DataFrame operations
        array_provider: Mock array operations
        
    Returns:
        DataFrameTransformer configured with test dependencies
    """
    test_deps = TransformerDependencyContainer(
        dataframe_provider=dataframe_provider,
        array_provider=array_provider
    )
    return DataFrameTransformer(test_deps)


def set_global_transformer_dependencies(
    dependencies: TransformerDependencyContainer
) -> TransformerDependencyContainer:
    """
    Set global transformer dependencies for testing.
    
    Args:
        dependencies: New dependency container
        
    Returns:
        Previous dependency container
    """
    global _default_transformer_deps
    previous = _default_transformer_deps
    _default_transformer_deps = dependencies
    logger.debug("Updated global transformer dependencies")
    return previous


def reset_global_transformer_dependencies() -> None:
    """
    Reset global transformer dependencies to defaults.
    """
    global _default_transformer_deps
    _default_transformer_deps = TransformerDependencyContainer()
    logger.debug("Reset global transformer dependencies")