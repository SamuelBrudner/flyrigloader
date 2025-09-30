"""
Dedicated transformation module providing optional DataFrame utilities for the decoupled I/O architecture.

This module consolidates all transformation logic from various modules to create a clean separation
between data discovery, loading, and transformation operations. It provides optional utilities for
converting raw experimental data into standardized Pandas DataFrames with comprehensive configuration
support, schema validation, and pluggable transformation handlers.

The transformation utilities support:
- Configurable DataFrame creation from raw data structures
- Domain-specific column transformations with metadata preservation  
- Signal display data handling with proper dimensionality transformation
- Column extraction and validation utilities
- Dimension transformation functions for array processing
- Pluggable transformation handler registry for extensibility
- Transformation chain integrity validation
- Registry-based transformation pipeline management

This separation enables the new decoupled pipeline workflow: discover() → load() → transform(),
where each stage can be tested and used independently for better control over memory usage and
processing pipelines.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Optional, List, Callable, Protocol, runtime_checkable
from functools import wraps
from abc import ABC, abstractmethod
import warnings
from itertools import chain
from time import perf_counter
from datetime import datetime

# Internal imports for configuration and logging
from flyrigloader import logger
from flyrigloader.io.column_models import get_config_from_source, ColumnConfigDict, SpecialHandlerType
from flyrigloader.migration.versions import detect_config_version


@runtime_checkable
class TransformationHandler(Protocol):
    """
    Protocol defining the interface for pluggable transformation handlers.
    
    This protocol enables extensible transformation processing through registry-based
    handler registration, supporting the new decoupled architecture requirements.
    """
    
    def can_handle(self, data_type: str, column_config: Any) -> bool:
        """
        Check if this handler can process the given data type and column configuration.
        
        Args:
            data_type: Type identifier for the data to be processed
            column_config: Column configuration object
            
        Returns:
            True if handler can process this data type, False otherwise
        """
        ...
    
    def transform(self, data: Any, column_config: Any, context: Dict[str, Any]) -> Any:
        """
        Transform the input data according to the column configuration.
        
        Args:
            data: Raw data to transform
            column_config: Column configuration object
            context: Transformation context including metadata and settings
            
        Returns:
            Transformed data ready for DataFrame integration
        """
        ...
    
    def validate_chain_integrity(self, previous_handler: Optional['TransformationHandler'], 
                               next_handler: Optional['TransformationHandler']) -> bool:
        """
        Validate that this handler can be chained with adjacent handlers.
        
        Args:
            previous_handler: Handler that precedes this one in the chain
            next_handler: Handler that follows this one in the chain
            
        Returns:
            True if chain integrity is maintained, False otherwise
        """
        ...


class TransformationHandlerRegistry:
    """
    Registry for pluggable transformation handlers supporting the decoupled architecture.
    
    This registry enables dynamic registration of transformation handlers without modifying
    core code, supporting the extensibility requirements outlined in the technical specification.
    """
    
    def __init__(self):
        """Initialize the transformation handler registry."""
        self._handlers: Dict[str, TransformationHandler] = {}
        self._handler_priority: Dict[str, int] = {}
        logger.debug("Initialized TransformationHandlerRegistry")
    
    def register(self, name: str, handler: TransformationHandler, priority: int = 0) -> None:
        """
        Register a transformation handler with optional priority.
        
        Args:
            name: Unique identifier for the handler
            handler: Handler instance implementing TransformationHandler protocol
            priority: Priority level (higher values = higher priority)
            
        Raises:
            ValueError: If handler doesn't implement required protocol
        """
        if not isinstance(handler, TransformationHandler):
            raise ValueError(
                f"Handler '{name}' must implement TransformationHandler protocol",
                recovery_hint="Ensure handler implements can_handle() and transform() methods from TransformationHandler protocol."
            )
        
        self._handlers[name] = handler
        self._handler_priority[name] = priority
        logger.debug(f"Registered transformation handler '{name}' with priority {priority}")
    
    def get_handler(self, name: str) -> Optional[TransformationHandler]:
        """
        Get a registered transformation handler by name.
        
        Args:
            name: Handler identifier
            
        Returns:
            Handler instance or None if not found
        """
        return self._handlers.get(name)
    
    def get_compatible_handlers(self, data_type: str, column_config: Any) -> List[TransformationHandler]:
        """
        Get all handlers compatible with the given data type and column configuration.
        
        Args:
            data_type: Type identifier for the data to be processed
            column_config: Column configuration object
            
        Returns:
            List of compatible handlers sorted by priority (highest first)
        """
        compatible = []
        for name, handler in self._handlers.items():
            if handler.can_handle(data_type, column_config):
                compatible.append((self._handler_priority[name], handler))
        
        # Sort by priority (highest first) and return handlers only
        compatible.sort(key=lambda x: x[0], reverse=True)
        return [handler for _, handler in compatible]
    
    def validate_transformation_chain(self, handlers: List[TransformationHandler]) -> bool:
        """
        Validate the integrity of a transformation handler chain.
        
        Args:
            handlers: List of handlers in chain order
            
        Returns:
            True if chain integrity is maintained, False otherwise
        """
        if not handlers:
            return True
        
        logger.debug(f"Validating transformation chain with {len(handlers)} handlers")
        
        for i, handler in enumerate(handlers):
            prev_handler = handlers[i-1] if i > 0 else None
            next_handler = handlers[i+1] if i < len(handlers) - 1 else None
            
            if not handler.validate_chain_integrity(prev_handler, next_handler):
                logger.error(f"Chain integrity validation failed at handler {i}")
                return False
        
        logger.debug("Transformation chain integrity validated successfully")
        return True
    
    def list_handlers(self) -> Dict[str, int]:
        """
        List all registered handlers with their priorities.
        
        Returns:
            Dictionary mapping handler names to their priority levels
        """
        return self._handler_priority.copy()


# Global registry instance for transformation handlers
_transformation_registry = TransformationHandlerRegistry()


def get_transformation_registry() -> TransformationHandlerRegistry:
    """
    Get the global transformation handler registry instance.
    
    Returns:
        Global TransformationHandlerRegistry instance
    """
    return _transformation_registry


class DefaultTransformationHandler:
    """
    Default transformation handler for standard data processing operations.
    
    This handler provides the core transformation logic that was previously embedded
    in various modules, now consolidated into the decoupled transformation architecture.
    """
    
    def can_handle(self, data_type: str, column_config: Any) -> bool:
        """Check if this handler can process standard data types."""
        return data_type in ['standard', 'numeric', 'array', 'default']
    
    def transform(self, data: Any, column_config: Any, context: Dict[str, Any]) -> Any:
        """
        Apply standard transformation logic to the input data.
        
        Args:
            data: Raw data to transform
            column_config: Column configuration object
            context: Transformation context
            
        Returns:
            Transformed data
        """
        logger.debug(f"Applying default transformation to data type: {type(data)}")
        
        # Apply special handling if configured
        if hasattr(column_config, 'special_handling') and column_config.special_handling:
            if column_config.special_handling == SpecialHandlerType.EXTRACT_FIRST_COLUMN:
                if hasattr(data, 'ndim') and data.ndim == 2:
                    data = data[:, 0] if data.shape[1] > 0 else data
                    logger.debug("Applied EXTRACT_FIRST_COLUMN transformation")
            elif column_config.special_handling == SpecialHandlerType.TRANSFORM_TIME_DIMENSION:
                exp_matrix = context.get('exp_matrix', {})
                if exp_matrix:
                    data = handle_signal_disp(exp_matrix)
                    logger.debug("Applied TRANSFORM_TIME_DIMENSION transformation")
        
        # Ensure correct dimensionality for numpy arrays
        if isinstance(data, np.ndarray) and hasattr(column_config, 'dimension') and column_config.dimension:
            if column_config.dimension.value == 1:
                data = ensure_1d_array(data, context.get('column_name', 'unknown'))
                logger.debug("Applied 1D array transformation")
        
        return data
    
    def validate_chain_integrity(self, previous_handler: Optional[TransformationHandler], 
                               next_handler: Optional[TransformationHandler]) -> bool:
        """Default handler is compatible with all chain positions."""
        return True


# Register the default handler
_transformation_registry.register('default', DefaultTransformationHandler(), priority=0)


def validate_transformation_chain_integrity(handlers: List[TransformationHandler]) -> bool:
    """
    Validate the integrity of a transformation handler chain.
    
    This function ensures that transformation handlers can be chained together
    without data type conflicts or processing incompatibilities.
    
    Args:
        handlers: List of transformation handlers in chain order
        
    Returns:
        True if chain integrity is maintained, False otherwise
    """
    return _transformation_registry.validate_transformation_chain(handlers)


def register_transformation_handler(name: str, handler: TransformationHandler, priority: int = 0) -> None:
    """
    Register a new transformation handler with the global registry.
    
    Args:
        name: Unique identifier for the handler
        handler: Handler instance implementing TransformationHandler protocol
        priority: Priority level (higher values = higher priority)
        
    Raises:
        ValueError: If handler doesn't implement required protocol
    """
    _transformation_registry.register(name, handler, priority)


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
                "with keys representing column names and values containing the data arrays.",
                recovery_hint="Provide a valid experimental data dictionary. Example: {'t': time_array, 'signal': signal_array}"
            )
        
        if not isinstance(exp_matrix, dict):
            raise TypeError(
                f"exp_matrix must be a dictionary, got {type(exp_matrix).__name__}. "
                f"Expected format: Dict[str, Any] where keys are column names and values are data arrays. "
                f"Received value: {repr(exp_matrix)}",
                recovery_hint="Load data as a dictionary. Check data loading function returns Dict[str, Any] format."
            )
        
        if not exp_matrix:
            raise ValueError(
                "exp_matrix cannot be empty. Expected a dictionary with at least one key-value pair "
                "representing experimental data columns.",
                recovery_hint="Ensure experimental data contains at least one column. Check data loading and processing."
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
                f"Please ensure your experimental data includes time information.",
                recovery_hint="Add a 't' key with time values to exp_matrix. Example: exp_matrix['t'] = np.arange(0, 10, 0.1)"
            )
        
        time_data = exp_matrix['t']
        if not hasattr(time_data, '__len__'):
            raise ValueError(
                f"Time data 't' must be array-like with length, got {type(time_data).__name__}. "
                f"Expected numpy array, list, or pandas Series with time values.",
                recovery_hint="Convert time data to array-like format. Use numpy array, list, or pandas Series."
            )
        
        time_length = len(time_data)
        if time_length == 0:
            raise ValueError(
                "Time dimension 't' cannot be empty. Expected array of time values.",
                recovery_hint="Provide non-empty time array. Example: exp_matrix['t'] = np.linspace(0, 10, 100)"
            )
        
        logger.debug(f"Validated time dimension with length: {time_length}")
        return time_length

    def transform_data(
        self,
        exp_matrix: Dict[str, Any],
        config_source: Union[str, Dict[str, Any], ColumnConfigDict, None] = None,
        metadata: Optional[Dict[str, Any]] = None,
        skip_columns: Optional[List[str]] = None,
        use_pluggable_handlers: bool = True
    ) -> pd.DataFrame:
        """
        Transform experimental data into a standardized DataFrame with comprehensive configuration support.
        
        This is the primary transformation method that provides configurable DataFrame creation
        with enhanced validation, column handling, metadata integration, and pluggable transformation
        handlers per the new decoupled architecture requirements.
        
        Args:
            exp_matrix: Dictionary containing experimental data
            config_source: Configuration source (path, dict, ColumnConfigDict, or None)
            metadata: Optional dictionary with metadata to add to the DataFrame
            skip_columns: Optional list of columns to exclude from processing
            use_pluggable_handlers: Whether to use registered transformation handlers
        
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
            raise ValueError(
                f"Missing required columns: {', '.join(missing_columns)}",
                recovery_hint=f"Add missing columns to exp_matrix: {', '.join(missing_columns)}"
            )
        
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

            # Apply transformation handlers if enabled
            if use_pluggable_handlers:
                # Determine data type for handler selection
                data_type = self._determine_data_type(value, col_config)
                
                # Get compatible handlers
                registry = get_transformation_registry()
                compatible_handlers = registry.get_compatible_handlers(data_type, col_config)
                
                if compatible_handlers:
                    # Validate handler chain integrity
                    if not validate_transformation_chain_integrity(compatible_handlers):
                        logger.warning(f"Transformation chain integrity validation failed for column '{col_name}', using default processing")
                        compatible_handlers = [registry.get_handler('default')]
                    
                    # Apply handlers in priority order
                    transformation_context = {
                        'exp_matrix': validated_matrix,
                        'column_name': col_name,
                        'metadata': metadata or {},
                        'skip_columns': skip_columns or []
                    }
                    
                    for handler in compatible_handlers:
                        try:
                            value = handler.transform(value, col_config, transformation_context)
                            logger.debug(f"Applied transformation handler to column '{col_name}'")
                            break  # Use first successful handler
                        except Exception as e:
                            logger.warning(f"Transformation handler failed for column '{col_name}': {e}")
                            continue
                else:
                    logger.debug(f"No compatible transformation handlers found for column '{col_name}', using default processing")
                    # Fall back to default processing
                    value = self._apply_default_transformations(value, col_config, validated_matrix, col_name)
            else:
                # Apply legacy transformation logic
                value = self._apply_default_transformations(value, col_config, validated_matrix, col_name)

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
    
    def _determine_data_type(self, value: Any, col_config: Any) -> str:
        """
        Determine the data type for transformation handler selection.
        
        Args:
            value: Data value to analyze
            col_config: Column configuration object
            
        Returns:
            Data type identifier for handler selection
        """
        if hasattr(col_config, 'special_handling') and col_config.special_handling:
            if col_config.special_handling == SpecialHandlerType.EXTRACT_FIRST_COLUMN:
                return 'extract_first_column'
            elif col_config.special_handling == SpecialHandlerType.TRANSFORM_TIME_DIMENSION:
                return 'transform_time_dimension'
        
        if isinstance(value, np.ndarray):
            return 'array'
        elif isinstance(value, (int, float)):
            return 'numeric'
        elif isinstance(value, str):
            return 'string'
        else:
            return 'standard'
    
    def _apply_default_transformations(self, value: Any, col_config: Any, exp_matrix: Dict[str, Any], col_name: str) -> Any:
        """
        Apply default transformation logic for backward compatibility.
        
        Args:
            value: Data value to transform
            col_config: Column configuration object
            exp_matrix: Full experimental data matrix
            col_name: Column name for error reporting
            
        Returns:
            Transformed value
        """
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
        
        return value


def make_dataframe_from_config(
    exp_matrix: Dict[str, Any],
    config_source: Union[str, Dict[str, Any], ColumnConfigDict, None] = None,
    metadata: Optional[Dict[str, Any]] = None,
    skip_columns: Optional[List[str]] = None,
    use_pluggable_handlers: bool = True,
    experiment_name: Optional[str] = None,
    file_path: Optional[str] = None,
    enable_kedro_compatibility: bool = False
) -> pd.DataFrame:
    """
    Enhanced convenience function for DataFrame creation with comprehensive configuration support.
    
    This function provides a simple interface for creating DataFrames from experimental data
    using the DataFrameTransformer with pluggable transformation handlers while maintaining
    backward compatibility with existing APIs. Optional Kedro compatibility can be enabled
    for pipeline integration.
    
    Args:
        exp_matrix: Dictionary containing experimental data
        config_source: Configuration source (path, dict, ColumnConfigDict, or None)
        metadata: Optional dictionary with metadata to add to the DataFrame
        skip_columns: Optional list of columns to exclude from processing
        use_pluggable_handlers: Whether to use registered transformation handlers
        experiment_name: Experiment name for Kedro lineage tracking (if Kedro enabled)
        file_path: Source file path for provenance tracking (if Kedro enabled)
        enable_kedro_compatibility: Whether to add Kedro metadata columns
    
    Returns:
        pd.DataFrame: DataFrame containing the data with correct types and optional Kedro metadata
        
    Raises:
        ValueError: If required columns are missing or configuration is invalid
        TypeError: If the config_source type is invalid
    """
    transformer = DataFrameTransformer()
    result = transformer.transform_data(exp_matrix, config_source, metadata, skip_columns, use_pluggable_handlers)
    
    # Add Kedro compatibility if requested
    if enable_kedro_compatibility:
        result = add_kedro_metadata(
            df=result,
            config_source=config_source,
            experiment_name=experiment_name,
            file_path=file_path,
            custom_metadata=metadata
        )
    
    return result


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
        raise ValueError(
            "exp_matrix missing required 'signal_disp' key",
            recovery_hint="Add 'signal_disp' array to exp_matrix. This should be a 2D array with signal displacement data."
        )

    if 't' not in exp_matrix:
        raise ValueError(
            "exp_matrix missing required 't' key",
            recovery_hint="Add 't' time array to exp_matrix. Example: exp_matrix['t'] = np.arange(0, duration, dt)"
        )

    # Get the signal_disp array and time array
    sd = exp_matrix['signal_disp']
    t = exp_matrix['t']
    T = len(t)  # Length of time dimension

    # Validate array dimensions
    if sd.ndim != 2:
        raise ValueError(
            f"signal_disp must be 2D array, got shape {sd.shape} ({sd.ndim}D)",
            recovery_hint="Reshape signal_disp to 2D array. Example: signal_disp.reshape(time_length, num_signals)"
        )

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
        raise ValueError(
            f"No dimension of signal_disp {sd.shape} matches time dimension length T={T}",
            recovery_hint=f"Reshape signal_disp so one dimension equals {T} (time length). Current shape: {sd.shape}"
        )

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
        raise ValueError(
            f"exp_matrix must be a dictionary, got {type(exp_matrix).__name__}",
            recovery_hint="Load data as a dictionary format. Check data loading returns Dict[str, Any]."
        )
    
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
        raise ValueError(
            f"Array '{name}' cannot be None. Expected a valid array-like input.",
            recovery_hint=f"Provide valid data for column '{name}'. Check data loading and ensure column exists in source data."
        )
    
    try:
        arr = np.asarray(array)
    except Exception as e:
        raise ValueError(
            f"Failed to convert '{name}' to numpy array: {e}",
            recovery_hint=f"Ensure data in column '{name}' is array-like (list, numpy array, or pandas Series)."
        ) from e
    
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
            f"If you need multi-dimensional data, consider storing each dimension separately ",
            recovery_hint=f"Reshape column '{name}' to 1D or split into separate columns. Current shape: {arr.shape}"
            f"or flattening the data explicitly before processing."
        )


def monitor_transformation_performance(
    transformation_func: Callable,
    performance_targets: Optional[Dict[str, float]] = None
) -> Callable:
    """
    Monitor transformation performance to maintain <2x data size memory overhead target.
    
    This decorator function provides high-resolution timing utilities for performance
    monitoring during Kedro-compatible DataFrame transformation to maintain the
    performance targets specified in Section 5.2.4 scaling considerations.
    
    Args:
        transformation_func: Function to monitor
        performance_targets: Dictionary of performance targets to validate against
        
    Returns:
        Callable: Wrapped function with performance monitoring
        
    Example:
        >>> @monitor_transformation_performance
        ... def my_transform(data):
        ...     return transform_to_dataframe(data)
        >>> result = my_transform(exp_matrix)
    """
    if performance_targets is None:
        performance_targets = {
            'max_memory_overhead_ratio': 2.0,  # <2x data size
            'max_duration_per_100mb': 1.0,     # <1s per 100MB
            'max_manifest_time': 0.1           # <100ms for manifest operations
        }
    
    @wraps(transformation_func)
    def performance_monitored_transform(*args, **kwargs):
        """Wrapper function with comprehensive performance monitoring."""
        start_time = perf_counter()
        
        # Estimate input data size
        input_size_bytes = 0
        if args:
            first_arg = args[0]
            if isinstance(first_arg, dict):
                for value in first_arg.values():
                    if hasattr(value, 'nbytes'):
                        input_size_bytes += value.nbytes
                    elif hasattr(value, '__len__'):
                        input_size_bytes += len(str(value).encode('utf-8'))
        
        input_size_mb = input_size_bytes / (1024 * 1024)
        logger.debug(f"Monitoring transformation of {input_size_mb:.2f} MB input data")
        
        # Execute transformation
        try:
            result = transformation_func(*args, **kwargs)
            end_time = perf_counter()
            duration = end_time - start_time
            
            # Calculate performance metrics
            metrics = {
                'duration_seconds': duration,
                'input_size_mb': input_size_mb,
                'duration_per_100mb': (duration / max(input_size_mb / 100, 0.01)),  # Avoid division by zero
            }
            
            # Estimate output size for memory overhead calculation
            if isinstance(result, pd.DataFrame):
                output_size_bytes = result.memory_usage(deep=True).sum()
                output_size_mb = output_size_bytes / (1024 * 1024)
                memory_overhead_ratio = output_size_mb / max(input_size_mb, 0.01)
                
                metrics.update({
                    'output_size_mb': output_size_mb,
                    'memory_overhead_ratio': memory_overhead_ratio
                })
                
                logger.info(f"Transformation performance: {duration:.3f}s, "
                          f"memory overhead: {memory_overhead_ratio:.2f}x, "
                          f"rate: {input_size_mb/duration:.2f} MB/s")
            
            # Check performance targets
            warnings_issued = []
            
            if 'memory_overhead_ratio' in metrics:
                if metrics['memory_overhead_ratio'] > performance_targets['max_memory_overhead_ratio']:
                    warning_msg = (f"Memory overhead {metrics['memory_overhead_ratio']:.2f}x exceeds "
                                 f"target {performance_targets['max_memory_overhead_ratio']:.2f}x")
                    warnings_issued.append(warning_msg)
                    logger.warning(warning_msg)
            
            if metrics['duration_per_100mb'] > performance_targets['max_duration_per_100mb']:
                warning_msg = (f"Processing rate {metrics['duration_per_100mb']:.2f}s/100MB exceeds "
                             f"target {performance_targets['max_duration_per_100mb']:.2f}s/100MB")
                warnings_issued.append(warning_msg)
                logger.warning(warning_msg)
            
            # Add performance metadata to result if it's a DataFrame
            if isinstance(result, pd.DataFrame) and not warnings_issued:
                logger.debug("Performance targets met, transformation successful")
            
            return result
            
        except Exception as e:
            end_time = perf_counter()
            duration = end_time - start_time
            logger.error(f"Transformation failed after {duration:.3f}s: {e}")
            raise
    
    return performance_monitored_transform


@monitor_transformation_performance
def transform_to_dataframe(
    exp_matrix: Dict[str, Any],
    config_source: Union[str, Dict[str, Any], ColumnConfigDict, None] = None,
    metadata: Optional[Dict[str, Any]] = None,
    skip_columns: Optional[List[str]] = None,
    use_pluggable_handlers: bool = True,
    experiment_name: Optional[str] = None,
    file_path: Optional[str] = None,
    enable_kedro_compatibility: bool = True,
    lazy_threshold_mb: float = 100.0
) -> pd.DataFrame:
    """
    Primary entry point for transforming raw experimental data to Kedro-compatible DataFrame format.
    
    This function serves as the main interface for the new decoupled transformation workflow,
    providing comprehensive DataFrame creation with configuration support, metadata integration,
    pluggable transformation handlers, Kedro compatibility, and memory-efficient processing
    as part of the manifest-based data loading architecture.
    
    Enhanced with Kedro compatibility features per Section 2.2.11 requirements:
    - Mandatory metadata columns for Kedro pipeline integration
    - Versioning and lineage tracking metadata
    - Memory-efficient lazy transformation for large datasets
    - Performance monitoring to maintain <2x data size memory overhead
    
    Args:
        exp_matrix: Dictionary containing raw experimental data
        config_source: Configuration source (path, dict, ColumnConfigDict, or None)
        metadata: Optional metadata to add to the resulting DataFrame
        skip_columns: Optional list of columns to exclude from processing
        use_pluggable_handlers: Whether to use registered transformation handlers
        experiment_name: Experiment name for Kedro lineage tracking
        file_path: Source file path for provenance tracking
        enable_kedro_compatibility: Whether to add Kedro metadata columns
        lazy_threshold_mb: Memory threshold for lazy transformation (MB)
        
    Returns:
        pd.DataFrame: Fully transformed DataFrame with validated columns, metadata, and
                     Kedro compatibility features
        
    Raises:
        ValueError: If transformation fails or required data is missing
        TypeError: If input types are invalid
        
    Example:
        >>> config = {"project": {"major_data_directory": "/data"}}
        >>> df = transform_to_dataframe(
        ...     exp_matrix={"t": [1, 2, 3], "data": [4, 5, 6]},
        ...     config_source=config,
        ...     experiment_name="baseline_study",
        ...     enable_kedro_compatibility=True
        ... )
        >>> assert "_kedro_version" in df.columns
        >>> assert "_kedro_experiment" in df.columns
    """
    logger.info("Starting DataFrame transformation with Kedro-compatible decoupled architecture")
    
    # Estimate data size for lazy transformation decision
    total_elements = sum(
        np.asarray(value).size if hasattr(value, 'size') else 1 
        for value in exp_matrix.values()
    )
    estimated_memory_mb = (total_elements * 8) / (1024 * 1024)  # Assume 8 bytes per element
    
    if estimated_memory_mb > lazy_threshold_mb:
        logger.info(f"Large dataset detected ({estimated_memory_mb:.2f} MB), using lazy transformation")
        lazy_transformer = create_lazy_dataframe_transformer(
            memory_threshold_mb=lazy_threshold_mb,
            enable_streaming=True
        )
        result = lazy_transformer(
            exp_matrix=exp_matrix,
            config_source=config_source,
            metadata=metadata,
            skip_columns=skip_columns,
            experiment_name=experiment_name,
            file_path=file_path,
            use_pluggable_handlers=use_pluggable_handlers
        )
    else:
        logger.debug("Using standard transformation for smaller dataset")
        # Standard transformation path
        transformer = DataFrameTransformer()
        result = transformer.transform_data(
            exp_matrix, 
            config_source, 
            metadata, 
            skip_columns, 
            use_pluggable_handlers
        )
        
        # Add Kedro compatibility metadata if enabled
        if enable_kedro_compatibility:
            result = add_kedro_metadata(
                df=result,
                config_source=config_source,
                experiment_name=experiment_name,
                file_path=file_path,
                custom_metadata=metadata
            )
    
    logger.info(f"Completed Kedro-compatible DataFrame transformation: {result.shape} DataFrame with {len(result.columns)} columns")
    return result


class SignalDisplayTransformationHandler:
    """
    Specialized transformation handler for signal display data processing.
    
    This handler demonstrates the pluggable transformation system by providing
    domain-specific processing for signal display arrays with time dimension alignment.
    """
    
    def can_handle(self, data_type: str, column_config: Any) -> bool:
        """Check if this handler can process signal display data."""
        return data_type == 'transform_time_dimension' or (
            hasattr(column_config, 'special_handling') and 
            column_config.special_handling == SpecialHandlerType.TRANSFORM_TIME_DIMENSION
        )
    
    def transform(self, data: Any, column_config: Any, context: Dict[str, Any]) -> Any:
        """
        Transform signal display data to match time dimension.
        
        Args:
            data: Signal display data array
            column_config: Column configuration object
            context: Transformation context including exp_matrix
            
        Returns:
            Transformed signal display data as pandas Series
        """
        exp_matrix = context.get('exp_matrix', {})
        if not exp_matrix:
            logger.warning("No experimental data matrix available for signal display transformation")
            return data
        
        logger.debug("Applying signal display transformation with time dimension alignment")
        return handle_signal_disp(exp_matrix)
    
    def validate_chain_integrity(self, previous_handler: Optional[TransformationHandler], 
                               next_handler: Optional[TransformationHandler]) -> bool:
        """Signal display handler is compatible with most chain positions."""
        return True


class ArrayExtractionTransformationHandler:
    """
    Specialized transformation handler for array extraction operations.
    
    This handler demonstrates extraction of specific columns or dimensions from
    multi-dimensional arrays, supporting the decoupled transformation architecture.
    """
    
    def can_handle(self, data_type: str, column_config: Any) -> bool:
        """Check if this handler can process array extraction."""
        return data_type == 'extract_first_column' or (
            hasattr(column_config, 'special_handling') and 
            column_config.special_handling == SpecialHandlerType.EXTRACT_FIRST_COLUMN
        )
    
    def transform(self, data: Any, column_config: Any, context: Dict[str, Any]) -> Any:
        """
        Extract first column from 2D array data.
        
        Args:
            data: Multi-dimensional array data
            column_config: Column configuration object
            context: Transformation context
            
        Returns:
            Extracted column data
        """
        column_name = context.get('column_name', 'unknown')
        logger.debug(f"Extracting first column from 2D array for '{column_name}'")
        
        if hasattr(data, 'ndim') and data.ndim == 2:
            if data.shape[1] > 0:
                return data[:, 0]
            else:
                logger.warning(f"Array for column '{column_name}' has zero columns, returning unchanged")
                return data
        else:
            logger.warning(f"Data for column '{column_name}' is not 2D, returning unchanged")
            return data
    
    def validate_chain_integrity(self, previous_handler: Optional[TransformationHandler], 
                               next_handler: Optional[TransformationHandler]) -> bool:
        """Array extraction handler is compatible with most chain positions."""
        return True


class DimensionNormalizationTransformationHandler:
    """
    Specialized transformation handler for array dimension normalization.
    
    This handler ensures arrays conform to expected dimensionality requirements
    as part of the pluggable transformation system.
    """
    
    def can_handle(self, data_type: str, column_config: Any) -> bool:
        """Check if this handler can process dimension normalization."""
        return data_type == 'array' and (
            hasattr(column_config, 'dimension') and 
            column_config.dimension and 
            column_config.dimension.value == 1
        )
    
    def transform(self, data: Any, column_config: Any, context: Dict[str, Any]) -> Any:
        """
        Normalize array dimensions to expected format.
        
        Args:
            data: Array data to normalize
            column_config: Column configuration object
            context: Transformation context
            
        Returns:
            Dimension-normalized array data
        """
        column_name = context.get('column_name', 'unknown')
        
        if isinstance(data, np.ndarray) and column_config.dimension.value == 1:
            try:
                normalized_data = ensure_1d_array(data, column_name)
                logger.debug(f"Applied dimension normalization to column '{column_name}'")
                return normalized_data
            except ValueError as e:
                logger.warning(f"Could not normalize dimensions for column '{column_name}': {e}")
                return data
        
        return data
    
    def validate_chain_integrity(self, previous_handler: Optional[TransformationHandler], 
                               next_handler: Optional[TransformationHandler]) -> bool:
        """Dimension normalization should typically be last in chain."""
        return next_handler is None or not hasattr(next_handler, 'requires_specific_dimensions')


# Register specialized transformation handlers
_transformation_registry.register('signal_display', SignalDisplayTransformationHandler(), priority=10)
_transformation_registry.register('array_extraction', ArrayExtractionTransformationHandler(), priority=8)
_transformation_registry.register('dimension_normalization', DimensionNormalizationTransformationHandler(), priority=5)


def create_transformation_context(
    exp_matrix: Dict[str, Any],
    column_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    skip_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create a standardized transformation context for handler processing.
    
    This function provides a consistent way to create transformation contexts
    that are passed to pluggable handlers, ensuring all necessary information
    is available for transformation operations.
    
    Args:
        exp_matrix: Full experimental data matrix
        column_name: Name of the column being processed
        metadata: Optional metadata dictionary
        skip_columns: Optional list of columns to skip
        
    Returns:
        Dictionary containing standardized transformation context
    """
    return {
        'exp_matrix': exp_matrix,
        'column_name': column_name,
        'metadata': metadata or {},
        'skip_columns': skip_columns or []
    }


def validate_transformation_pipeline(
    exp_matrix: Dict[str, Any],
    config_source: Union[str, Dict[str, Any], ColumnConfigDict, None] = None,
    skip_columns: Optional[List[str]] = None
) -> bool:
    """
    Validate that a transformation pipeline can be executed successfully.
    
    This function performs comprehensive validation of the transformation pipeline
    including data integrity checks, handler compatibility validation, and
    configuration consistency verification.
    
    Args:
        exp_matrix: Dictionary containing experimental data
        config_source: Configuration source for validation
        skip_columns: Optional list of columns to skip
        
    Returns:
        True if pipeline validation passes, False otherwise
    """
    logger.info("Starting transformation pipeline validation")
    
    try:
        # Validate input data
        if not isinstance(exp_matrix, dict):
            logger.error("exp_matrix must be a dictionary")
            return False
        
        if not exp_matrix:
            logger.error("exp_matrix cannot be empty")
            return False
        
        # Load and validate configuration
        config = get_config_from_source(config_source)
        skip_columns = skip_columns or []
        
        # Check for required columns
        required_columns = []
        for col_name, col_config in config.columns.items():
            if col_name in skip_columns or not col_config.required:
                continue
            required_columns.append(col_name)
        
        # Validate transformation handlers for each column
        registry = get_transformation_registry()
        transformer = DataFrameTransformer()
        
        for col_name, col_config in config.columns.items():
            if col_name in skip_columns:
                continue
            
            # Check if column exists in data
            source_col = col_name
            if col_config.alias and col_config.alias in exp_matrix:
                source_col = col_config.alias
            
            if source_col not in exp_matrix:
                if col_config.required:
                    logger.error(f"Required column '{col_name}' not found in data")
                    return False
                continue
            
            # Validate transformation handlers
            value = exp_matrix[source_col]
            data_type = transformer._determine_data_type(value, col_config)
            compatible_handlers = registry.get_compatible_handlers(data_type, col_config)
            
            if compatible_handlers:
                if not validate_transformation_chain_integrity(compatible_handlers):
                    logger.error(f"Transformation chain validation failed for column '{col_name}'")
                    return False
            
            logger.debug(f"Column '{col_name}' validation passed")
        
        logger.info("Transformation pipeline validation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Transformation pipeline validation failed: {e}")
        return False


def get_transformation_pipeline_info() -> Dict[str, Any]:
    """
    Get information about the current transformation pipeline configuration.
    
    This function provides insight into the registered transformation handlers,
    their priorities, and the overall pipeline configuration for debugging
    and monitoring purposes.
    
    Returns:
        Dictionary containing pipeline configuration information
    """
    registry = get_transformation_registry()
    
    return {
        'registered_handlers': registry.list_handlers(),
        'total_handlers': len(registry.list_handlers()),
        'handler_details': {
            name: {
                'priority': priority,
                'handler_type': type(registry.get_handler(name)).__name__
            }
            for name, priority in registry.list_handlers().items()
        }
    }


def legacy_transform_data(
    exp_matrix: Dict[str, Any],
    config_source: Union[str, Dict[str, Any], ColumnConfigDict, None] = None,
    metadata: Optional[Dict[str, Any]] = None,
    skip_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Legacy transformation function for backward compatibility.
    
    This function provides the old transformation interface while internally
    using the new decoupled architecture. It is maintained for backward
    compatibility during the transition period.
    
    Args:
        exp_matrix: Dictionary containing experimental data
        config_source: Configuration source (path, dict, ColumnConfigDict, or None)
        metadata: Optional dictionary with metadata to add to the DataFrame
        skip_columns: Optional list of columns to exclude from processing
        
    Returns:
        pd.DataFrame: DataFrame containing the transformed data
        
    Raises:
        ValueError: If transformation fails or required data is missing
        TypeError: If input types are invalid
    """
    warnings.warn(
        "legacy_transform_data is deprecated and will be removed in a future version. "
        "Use transform_to_dataframe instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    logger.debug("Using legacy transformation interface (deprecated)")
    return transform_to_dataframe(exp_matrix, config_source, metadata, skip_columns, use_pluggable_handlers=False)


def add_kedro_metadata(
    df: pd.DataFrame,
    config_source: Union[str, Dict[str, Any], ColumnConfigDict, None] = None,
    experiment_name: Optional[str] = None,
    file_path: Optional[str] = None,
    custom_metadata: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Add Kedro-compatible metadata columns to DataFrame for pipeline integration.
    
    This function ensures DataFrames are Kedro-compatible by adding mandatory metadata
    columns required by Kedro pipelines. It includes versioning and lineage tracking
    metadata per Section 2.2.11 Kedro Integration Layer requirements.
    
    Args:
        df: Input DataFrame to enhance with Kedro metadata
        config_source: Configuration source for version detection
        experiment_name: Name of the experiment for lineage tracking
        file_path: Source file path for provenance tracking
        custom_metadata: Additional metadata to include
        
    Returns:
        pd.DataFrame: Enhanced DataFrame with Kedro-compatible metadata columns
        
    Example:
        >>> df = pd.DataFrame({'data': [1, 2, 3]})
        >>> kedro_df = add_kedro_metadata(df, experiment_name="baseline_study")
        >>> assert '_kedro_version' in kedro_df.columns
        >>> assert '_kedro_timestamp' in kedro_df.columns
    """
    logger.debug(f"Adding Kedro metadata to DataFrame with shape {df.shape}")
    
    # Create a copy to avoid modifying the original DataFrame
    enhanced_df = df.copy()
    
    # Add mandatory Kedro metadata columns
    current_timestamp = datetime.now().isoformat()
    
    # Version tracking - detect configuration version for lineage
    if config_source is not None:
        try:
            if isinstance(config_source, str):
                # For file paths, read and detect version
                with open(config_source, 'r') as f:
                    config_content = f.read()
                detected_version = detect_config_version(config_content)
            elif isinstance(config_source, dict):
                detected_version = detect_config_version(config_source)
            else:
                # For ColumnConfigDict, assume current version
                detected_version = detect_config_version({"schema_version": "1.0.0"})
            
            config_version = str(detected_version)
        except Exception as e:
            logger.warning(f"Could not detect configuration version: {e}")
            config_version = "unknown"
    else:
        config_version = "default"
    
    # Core Kedro metadata columns (mandatory for pipeline compatibility)
    enhanced_df['_kedro_version'] = config_version
    enhanced_df['_kedro_timestamp'] = current_timestamp
    enhanced_df['_kedro_dataset_id'] = f"{experiment_name or 'unnamed'}_{hash(current_timestamp) % 10000:04d}"
    
    # Lineage tracking metadata
    if experiment_name:
        enhanced_df['_kedro_experiment'] = experiment_name
    
    if file_path:
        enhanced_df['_kedro_source_file'] = file_path
        enhanced_df['_kedro_file_hash'] = hash(file_path) % 100000  # Simple hash for tracking
    
    # Kedro versioning support
    enhanced_df['_kedro_created_at'] = current_timestamp
    enhanced_df['_kedro_pipeline_version'] = "1.0.0"  # FlyRigLoader version
    
    # Add custom metadata if provided
    if custom_metadata:
        for key, value in custom_metadata.items():
            # Prefix with _kedro_ if not already prefixed
            kedro_key = key if key.startswith('_kedro_') else f'_kedro_{key}'
            enhanced_df[kedro_key] = value
    
    logger.info(f"Enhanced DataFrame with {len(enhanced_df.columns) - len(df.columns)} Kedro metadata columns")
    return enhanced_df


def create_lazy_dataframe_transformer(
    memory_threshold_mb: float = 100.0,
    chunk_size: int = 1000,
    enable_streaming: bool = True
) -> Callable[[Dict[str, Any], ...], pd.DataFrame]:
    """
    Create a memory-efficient lazy DataFrame transformer for Kedro pipeline performance.
    
    This function implements lazy transformation for memory efficiency to support Kedro
    pipeline performance requirements per Section 0.2.2 component impact analysis.
    It maintains the <2x data size memory overhead performance target while adding
    Kedro compatibility.
    
    Args:
        memory_threshold_mb: Memory threshold in MB to trigger lazy loading
        chunk_size: Number of rows to process in each chunk
        enable_streaming: Whether to enable streaming processing for large datasets
        
    Returns:
        Callable: Lazy transformer function with configured memory management
        
    Example:
        >>> lazy_transformer = create_lazy_dataframe_transformer(memory_threshold_mb=50.0)
        >>> df = lazy_transformer(exp_matrix, config_source="config.yaml")
    """
    logger.debug(f"Creating lazy DataFrame transformer with {memory_threshold_mb}MB threshold")
    
    def lazy_transform(
        exp_matrix: Dict[str, Any],
        config_source: Union[str, Dict[str, Any], ColumnConfigDict, None] = None,
        metadata: Optional[Dict[str, Any]] = None,
        skip_columns: Optional[List[str]] = None,
        experiment_name: Optional[str] = None,
        file_path: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Lazy transformation function with memory management and Kedro compatibility.
        
        Args:
            exp_matrix: Dictionary containing experimental data
            config_source: Configuration source for validation and versioning
            metadata: Optional metadata to include
            skip_columns: Columns to skip during processing
            experiment_name: Experiment name for Kedro lineage tracking
            file_path: Source file path for provenance
            **kwargs: Additional transformation parameters
            
        Returns:
            pd.DataFrame: Kedro-compatible DataFrame with memory-efficient processing
        """
        start_time = perf_counter()
        logger.debug("Starting lazy DataFrame transformation")
        
        # Estimate memory requirements
        total_elements = sum(
            np.asarray(value).size if hasattr(value, 'size') else 1 
            for value in exp_matrix.values()
        )
        estimated_memory_mb = (total_elements * 8) / (1024 * 1024)  # Assume 8 bytes per element
        
        logger.debug(f"Estimated memory requirement: {estimated_memory_mb:.2f} MB")
        
        if estimated_memory_mb > memory_threshold_mb and enable_streaming:
            logger.info(f"Large dataset detected ({estimated_memory_mb:.2f} MB), using streaming transformation")
            return _stream_transform_data(
                exp_matrix, config_source, metadata, skip_columns, 
                chunk_size, experiment_name, file_path, **kwargs
            )
        else:
            logger.debug("Using standard transformation for smaller dataset")
            # Use standard transformation
            transformer = DataFrameTransformer()
            df = transformer.transform_data(exp_matrix, config_source, metadata, skip_columns, **kwargs)
            
            # Add Kedro metadata
            df = add_kedro_metadata(df, config_source, experiment_name, file_path, metadata)
            
            end_time = perf_counter()
            logger.info(f"Lazy transformation completed in {end_time - start_time:.3f} seconds")
            return df
    
    return lazy_transform


def _stream_transform_data(
    exp_matrix: Dict[str, Any],
    config_source: Union[str, Dict[str, Any], ColumnConfigDict, None],
    metadata: Optional[Dict[str, Any]],
    skip_columns: Optional[List[str]],
    chunk_size: int,
    experiment_name: Optional[str],
    file_path: Optional[str],
    **kwargs
) -> pd.DataFrame:
    """
    Internal streaming transformation for large datasets.
    
    Args:
        exp_matrix: Experimental data dictionary
        config_source: Configuration source
        metadata: Metadata dictionary
        skip_columns: Columns to skip
        chunk_size: Chunk size for processing
        experiment_name: Experiment name
        file_path: Source file path
        **kwargs: Additional parameters
        
    Returns:
        pd.DataFrame: Processed DataFrame with memory-efficient streaming
    """
    logger.debug(f"Starting streaming transformation with chunk size {chunk_size}")
    
    # For streaming, we need to process data in chunks
    # This is a simplified implementation - in practice, would need more sophisticated chunking
    
    # Get array length for chunking (assume all arrays have same length)
    array_lengths = {}
    for key, value in exp_matrix.items():
        if hasattr(value, '__len__') and not isinstance(value, str):
            array_lengths[key] = len(value)
    
    if not array_lengths:
        logger.warning("No arrays found for streaming, falling back to standard transformation")
        transformer = DataFrameTransformer()
        df = transformer.transform_data(exp_matrix, config_source, metadata, skip_columns, **kwargs)
        return add_kedro_metadata(df, config_source, experiment_name, file_path, metadata)
    
    # Use the most common array length
    target_length = max(array_lengths.values(), key=array_lengths.values().count)
    logger.debug(f"Target array length for streaming: {target_length}")
    
    # Process in chunks
    chunk_dataframes = []
    for start_idx in range(0, target_length, chunk_size):
        end_idx = min(start_idx + chunk_size, target_length)
        logger.debug(f"Processing chunk {start_idx}:{end_idx}")
        
        # Create chunk matrix
        chunk_matrix = {}
        for key, value in exp_matrix.items():
            if hasattr(value, '__getitem__') and len(value) == target_length:
                # Slice array data
                chunk_matrix[key] = value[start_idx:end_idx]
            else:
                # Use full value for non-array data
                chunk_matrix[key] = value
        
        # Transform chunk
        transformer = DataFrameTransformer()
        chunk_df = transformer.transform_data(chunk_matrix, config_source, metadata, skip_columns, **kwargs)
        chunk_dataframes.append(chunk_df)
    
    # Combine chunks
    logger.debug(f"Combining {len(chunk_dataframes)} chunks")
    if chunk_dataframes:
        combined_df = pd.concat(chunk_dataframes, ignore_index=True)
    else:
        # Fallback to empty DataFrame
        combined_df = pd.DataFrame()
    
    # Add Kedro metadata to final result
    final_df = add_kedro_metadata(combined_df, config_source, experiment_name, file_path, metadata)
    
    logger.info(f"Streaming transformation completed, final shape: {final_df.shape}")
    return final_df


def create_test_dataframe_transformer() -> DataFrameTransformer:
    """
    Test utility factory for creating DataFrameTransformer instances.
    
    This function provides a clean way to create DataFrameTransformer instances
    for testing purposes, ensuring consistent initialization across test suites.
    
    Returns:
        DataFrameTransformer: New instance configured for testing
    """
    logger.debug("Creating test DataFrameTransformer instance")
    return DataFrameTransformer()