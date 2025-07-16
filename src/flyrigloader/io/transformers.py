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

# Internal imports for configuration and logging
from flyrigloader import logger
from flyrigloader.io.column_models import get_config_from_source, ColumnConfigDict, SpecialHandlerType


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
            raise ValueError(f"Handler '{name}' must implement TransformationHandler protocol")
        
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
    use_pluggable_handlers: bool = True
) -> pd.DataFrame:
    """
    Enhanced convenience function for DataFrame creation with comprehensive configuration support.
    
    This function provides a simple interface for creating DataFrames from experimental data
    using the DataFrameTransformer with pluggable transformation handlers while maintaining
    backward compatibility with existing APIs.
    
    Args:
        exp_matrix: Dictionary containing experimental data
        config_source: Configuration source (path, dict, ColumnConfigDict, or None)
        metadata: Optional dictionary with metadata to add to the DataFrame
        skip_columns: Optional list of columns to exclude from processing
        use_pluggable_handlers: Whether to use registered transformation handlers
    
    Returns:
        pd.DataFrame: DataFrame containing the data with correct types
        
    Raises:
        ValueError: If required columns are missing or configuration is invalid
        TypeError: If the config_source type is invalid
    """
    transformer = DataFrameTransformer()
    return transformer.transform_data(exp_matrix, config_source, metadata, skip_columns, use_pluggable_handlers)


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
    skip_columns: Optional[List[str]] = None,
    use_pluggable_handlers: bool = True
) -> pd.DataFrame:
    """
    Primary entry point for transforming raw experimental data to DataFrame format.
    
    This function serves as the main interface for the new decoupled transformation workflow,
    providing comprehensive DataFrame creation with configuration support, metadata integration,
    pluggable transformation handlers, and optional column processing as part of the manifest-based
    data loading architecture.
    
    Args:
        exp_matrix: Dictionary containing raw experimental data
        config_source: Configuration source (path, dict, ColumnConfigDict, or None)
        metadata: Optional metadata to add to the resulting DataFrame
        skip_columns: Optional list of columns to exclude from processing
        use_pluggable_handlers: Whether to use registered transformation handlers
        
    Returns:
        pd.DataFrame: Fully transformed DataFrame with validated columns and metadata
        
    Raises:
        ValueError: If transformation fails or required data is missing
        TypeError: If input types are invalid
    """
    logger.info("Starting DataFrame transformation with decoupled architecture")
    
    transformer = DataFrameTransformer()
    result = transformer.transform_data(
        exp_matrix, 
        config_source, 
        metadata, 
        skip_columns, 
        use_pluggable_handlers
    )
    
    logger.info(f"Completed DataFrame transformation: {result.shape} DataFrame with {len(result.columns)} columns")
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