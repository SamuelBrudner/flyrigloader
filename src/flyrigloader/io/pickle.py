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
from flyrigloader.io.column_models import (
    load_column_config, 
    ColumnConfigDict, 
    ColumnConfig, 
    SpecialHandlerType
)


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


def _filter_matrix_columns(
    exp_matrix: Dict[str, Any],
    include_signal_disp: bool,
    column_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Filter columns from the exp_matrix based on inclusion criteria."""
    filtered_matrix = {}
    
    # Determine the columns to process
    columns_to_process = column_list if column_list is not None else exp_matrix.keys()

    for col in columns_to_process:
        if col not in exp_matrix:
            logger.warning(f"Column '{col}' not found in exp_matrix.") 
            continue

        # Handle signal_disp exclusion
        if col == 'signal_disp' and not include_signal_disp:
            continue
            
        filtered_matrix[col] = exp_matrix[col]
        
    return filtered_matrix


def _validate_column_dimension(col: str, values: Any, t_length: int) -> None:
    """Validate the dimension of a column against the time dimension length."""
    if (
        isinstance(values, np.ndarray)
        and values.ndim > 0
        and all(dim != t_length for dim in values.shape)
        and (values.ndim != 1 or len(values) != t_length)
    ):
        if values.ndim == 1:
            raise ValueError(
               f"1D Column '{col}' has length {len(values)}, which does not "
               f"match time dimension length {t_length}"
            )
        elif all(dim != t_length for dim in values.shape):
            raise ValueError(
               f"Column '{col}' has shape {values.shape} with no dimension "
               f"matching time dimension length {t_length}"
            )


def make_dataframe_from_matrix(
    exp_matrix: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    include_signal_disp: bool = True,
    column_list: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convert an exp_matrix dictionary to a DataFrame with metadata, validating dimensions.

    Args:
        exp_matrix: Dictionary containing experimental data.
        metadata: Dictionary with metadata to add as columns.
        include_signal_disp: Whether to include the 'signal_disp' column if it exists.
        column_list: Optional list of columns to include. If None, all columns are considered.

    Returns:
        pandas.DataFrame: DataFrame containing the selected and validated data and metadata.

    Raises:
        ValueError: If 't' column is missing or not a 1D array.
        ValueError: If any included array column has no dimension matching the length of 't'.
        UserWarning: If a column specified in column_list is not found in exp_matrix.
    """
    if 't' not in exp_matrix or not isinstance(exp_matrix['t'], np.ndarray) or exp_matrix['t'].ndim != 1:
        raise ValueError("exp_matrix must contain a 1D numpy array named 't'.")
    t_length = len(exp_matrix['t'])

    # Filter columns first
    filtered_exp_matrix = _filter_matrix_columns(
        exp_matrix, include_signal_disp, column_list
    )

    df_data = {}
    # Validate dimensions and prepare data for DataFrame
    for col, values in filtered_exp_matrix.items():
        # Skip validation for 't' as it's already checked and defines the length
        if col != 't':
             _validate_column_dimension(col, values, t_length)

        # Ensure data added to df_data is suitable for DataFrame construction
        # Convert multi-dimensional arrays into a list of arrays along the first axis
        # This allows pandas to handle them as an object-dtype column.
        if isinstance(values, np.ndarray) and values.ndim > 1:
            # Assuming the validation ensured the first dimension matches t_length
            df_data[col] = list(values)
        else:
            # Use scalars, 1D arrays, or other types directly
            df_data[col] = values

    # Create DataFrame from validated data
    # Handle potential length mismatches if validation missed something
    try:
        df = pd.DataFrame(df_data)
    except ValueError as e:
        if "All arrays must be of the same length" not in str(e):
            raise # Re-raise other ValueErrors

        lengths = {k: len(v) if hasattr(v, '__len__') else 'scalar' for k, v in df_data.items()}
        raise ValueError(
            f"Error creating DataFrame: Mismatched lengths detected after validation. "
            f"Column lengths: {lengths}. Original error: {e}"
        ) from e
    # Add metadata if provided
    if metadata:
        df = _add_metadata_to_dataframe(df, metadata)

    return df


def _extract_first_column(array):
    """Extract the first column from a 2D array."""
    import numpy as np
    arr = np.asarray(array)
    return arr[:, 0] if arr.ndim == 2 and arr.shape[1] > 1 else arr


def _ensure_1d(array, name):
    """
    Convert array to a 1D NumPy array if possible. 
    Raises ValueError if the array has more than 1 column.
    """
    import numpy as np
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


def _handle_signal_disp(exp_matrix, col_name=None):
    """
    Return a pd.Series of length len(t), where each entry is an array
    from exp_matrix['signal_disp'].
    
    We detect which axis matches the time dimension, transpose if necessary,
    and store the 'other' axis as a small array. For example, if shape is
    (15, 54953), we transpose to (54953, 15). The resulting Series has 54953
    entries, each of which is an array of length 15.
    """
    import numpy as np
    import pandas as pd
    from loguru import logger
    
    col_name = col_name or "signal_disp"
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


def _load_column_config(config_path):
    """
    Load column configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        tuple: (column_config, special_handlers)
    """
    import yaml
    from loguru import logger
    
    logger.debug(f"Loading column configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config['columns'], config.get('special_handlers', {})


def _validate_required_columns(exp_matrix, column_config):
    """
    Validate that all required columns are present in the exp_matrix.
    
    Args:
        exp_matrix: Dictionary containing experimental data.
        column_config: Dictionary of column configurations.
        
    Returns:
        list: Missing required columns.
    """
    from loguru import logger
    
    missing_columns = []
    for col, settings in column_config.items():
        if not settings.get('required', False) or settings.get('is_metadata', False):
            continue
            
        if col in exp_matrix:
            continue
            
        # Check if there's an alias for this column
        alias = settings.get('alias')
        if alias and alias in exp_matrix:
            logger.debug(f"Required column '{col}' found via alias '{alias}'")
            continue
            
        missing_columns.append(col)
    
    return missing_columns


def _apply_special_handler(exp_matrix, col, value, handler_name, special_handler):
    """
    Apply a special handler function to a column value.
    
    Args:
        exp_matrix: Dictionary containing experimental data.
        col: Column name.
        value: Current value for the column.
        handler_name: Name of handler function.
        special_handler: Type of special handling to apply.
        
    Returns:
        Processed value.
    """
    from loguru import logger

    if handler_name and handler_name.startswith('_'):
        if handler_func := getattr(__name__, handler_name, None):
            logger.debug(f"Applying special handler '{handler_name}' to column '{col}'")
            return handler_func(exp_matrix, col) if handler_name == '_handle_signal_disp' else handler_func(value)
        else:
            logger.warning(f"Special handler '{handler_name}' not found, using raw value for '{col}'")
            return value

    # Apply built-in handlers by name
    if special_handler == 'extract_first_column_if_2d':
        logger.debug(f"Extracting first column from 2D array for '{col}'")
        return _extract_first_column(value)
    elif special_handler == 'transform_to_match_time_dimension':
        logger.debug(f"Transforming '{col}' to match time dimension")
        return _handle_signal_disp(exp_matrix, col)

    return value


def _process_column(exp_matrix, col_name, col_config, special_handlers):
    """
    Process a single column according to its configuration.
    
    Args:
        exp_matrix: Dictionary containing experimental data.
        col_name: Column name to process.
        col_config: Configuration settings for this column.
        special_handlers: Dictionary of special handler functions.
        
    Returns:
        tuple: (column_name, processed_value) or None if column should be skipped.
    """
    import numpy as np
    from loguru import logger

    # Skip metadata columns
    if col_config.get('is_metadata', False):
        return None

    # Handle aliases (e.g., dtheta_smooth -> dtheta)
    source_col = col_name
    if col_config.get('alias') and col_config.get('alias') in exp_matrix:
        source_col = col_config.get('alias')
        logger.debug(f"Using alias '{source_col}' for column '{col_name}'")

    # If column is not in the matrix
    if source_col not in exp_matrix:
        # For non-required columns, include them if they have a default value defined
        # (even if that default value is None)
        if not col_config.get('required', True) and hasattr(col_config, 'default_value'):
            logger.debug(f"Using default value for missing column '{col_name}'")
            return col_name, col_config.default_value
        return None
            
    # Get the value from the exp_matrix
    value = exp_matrix[source_col]

    if special_handler := col_config.get('special_handling'):
        handler_name = special_handlers.get(special_handler)
        value = _apply_special_handler(exp_matrix, col_name, value, special_handler, handler_name)

    # Ensure correct dimensionality for numpy arrays
    if isinstance(value, np.ndarray) and col_config.get('dimension'):
        try:
            if col_config.dimension.value == 1:
                value = _ensure_1d(value, col_name)
        except ValueError as e:
            logger.warning(f"Could not convert '{col_name}' to 1D: {e}")

    return col_name, value


def _add_metadata_columns(df, metadata, column_config):
    """
    Add metadata fields to the DataFrame based on configuration.
    
    Args:
        df: DataFrame to update.
        metadata: Dictionary of metadata values.
        column_config: Dictionary of column configurations.
        
    Returns:
        DataFrame with metadata added.
    """
    from loguru import logger
    
    if not metadata:
        return df
        
    for key, value in metadata.items():
        if key in column_config and column_config[key].get('is_metadata', False):
            logger.debug(f"Adding metadata field '{key}'")
            df[key] = value
    
    return df


def make_dataframe_from_config(
    exp_matrix: Dict[str, Any], 
    config_path: str, 
    metadata: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Convert an exp_matrix dictionary to a DataFrame based on column configuration.
    
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
    
    # Validate required columns
    if missing_columns := _validate_required_columns(exp_matrix, config.columns):
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Process columns
    data_dict = {}
    for col_name, col_config in config.columns.items():
        if result := _process_column(exp_matrix, col_name, col_config, config.special_handlers):
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


def _extract_first_column(array: np.ndarray) -> np.ndarray:
    """
    Extract the first column from a 2D array.
    
    Args:
        array: NumPy array to process.
        
    Returns:
        np.ndarray: First column of the array if 2D, otherwise the original array.
    """
    arr = np.asarray(array)
    if arr.ndim == 2 and arr.shape[1] > 1:
        return arr[:, 0]
    return arr


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


def _validate_required_columns(exp_matrix: Dict[str, Any], columns: Dict[str, ColumnConfig]) -> List[str]:
    """
    Validate that all required columns are present in the exp_matrix.
    
    Args:
        exp_matrix: Dictionary containing experimental data.
        columns: Dictionary of column configurations.
        
    Returns:
        list: Missing required columns.
    """
    missing_columns = []
    
    for col_name, col_config in columns.items():
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
        # Try to find a handler function in this module
        import sys
        current_module = sys.modules[__name__]
        handler_func = getattr(current_module, handler_name, None)
        
        if handler_func:
            logger.debug(f"Applying special handler '{handler_name}' to column '{col_name}'")
            return handler_func(exp_matrix, col_name) if handler_name == '_handle_signal_disp' else handler_func(value)
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
    special_handlers: Dict[str, str]
) -> Optional[Tuple[str, Any]]:
    """
    Process a single column according to its configuration.
    
    Args:
        exp_matrix: Dictionary containing experimental data.
        col_name: Name of the column to process.
        col_config: Configuration for the column.
        special_handlers: Dictionary mapping handler types to handler function names.
        
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
        # For non-required columns, include them if they have a default value defined
        # (even if that default value is None)
        if not col_config.required and hasattr(col_config, 'default_value'):
            logger.debug(f"Using default value for missing column '{col_name}'")
            return col_name, col_config.default_value
        return None
            
    # Get the value from the exp_matrix
    value = exp_matrix[source_col]
    
    # Apply any special handling
    if col_config.special_handling:
        handler_name = special_handlers.get(col_config.special_handling.value)
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
