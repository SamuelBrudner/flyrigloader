"""
schema_utils.py - Core schema utilities.

This module provides low-level schema operations that are used across different parts of the codebase,
including:
1. Type mapping (string to pandas dtypes)
2. Column operations (renaming, type conversion)
3. Schema validation helpers

These utilities are designed to be used with both dictionary-based schemas and Pandera schemas.
They are low-level utilities that throw exceptions directly.

For higher-level schema validation with the tuple return pattern, use schema/validator.py.
"""

import contextlib
import numpy as np
import pandas as pd
from loguru import logger
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from flyrigloader.core.utils.value_utils import coerce_to_list as coerce_to_list_value_utils


# Define type sets at module level for use across the codebase
INT_TYPES = {"int", "int32", "int64", "integer"}
FLOAT_TYPES = {"float", "float32", "float64", "numeric"}
STR_TYPES = {"str", "string", "text"}
BOOL_TYPES = {"bool", "boolean"}
LIST_TYPES = {"list", "array"}
DATE_TYPES = {"datetime", "date"}
CATEGORY_TYPES = {"category"}
OBJECT_TYPES = {"object", "any"}

# Dictionary for mapping string type names to pandas dtypes
# This is used for consistent type conversion across the codebase
TYPE_MAPPING = {
    **{t: pd.Int64Dtype() for t in INT_TYPES},
    **{t: pd.Float64Dtype() for t in FLOAT_TYPES},
    **{t: pd.StringDtype() for t in STR_TYPES},
    **{t: pd.BooleanDtype() for t in BOOL_TYPES},
    **{t: pd.ArrowDtype(storage="list") for t in LIST_TYPES},
    **{t: pd.DatetimeTZDtype(tz=None) for t in DATE_TYPES},
    **{t: pd.CategoricalDtype() for t in CATEGORY_TYPES},
    **{t: pd.ObjectDtype() for t in OBJECT_TYPES},
}


def is_type_in_category(dtype: str, category: Set[str]) -> bool:
    """
    Check if a type string belongs to a particular type category.
    
    Args:
        dtype: Type name as string
        category: Set of type names that belong to the category
        
    Returns:
        True if the type belongs to the category, False otherwise
    """
    return dtype.lower() in category


def extract_column_info(col_def: Union[str, Dict[str, Any]]) -> Tuple[Optional[str], bool, str, Optional[str]]:
    """
    Extract column information from a column definition.
    
    Args:
        col_def: Column definition as a string or dictionary
        
    Returns:
        Tuple of (source_column, nullable, dtype, target_column)
        
    Raises:
        TypeError: If col_def is neither a string nor a dictionary
    """
    source_col = None
    nullable = True
    dtype = "object"
    target_col = None
    
    # Handle string type definitions (simple case)
    if isinstance(col_def, str):
        dtype = col_def
    # Handle dictionary definitions (complex case)
    elif isinstance(col_def, dict):
        # Handle renames - safely check if key exists
        if "rename_to" in col_def:
            target_col = col_def["rename_to"]
        # Handle source columns - safely check if key exists
        if "source" in col_def:
            source_col = col_def["source"]
        # Extract type information with safe defaults
        dtype = col_def.get("dtype", "object")
        nullable = col_def.get("nullable", True)
        
        # Validate data types of extracted values
        if not isinstance(nullable, bool):
            logger.warning(f"Invalid 'nullable' value in column definition: {nullable}. Using default: True")
            nullable = True
            
        if not isinstance(dtype, str):
            logger.warning(f"Invalid 'dtype' value in column definition: {dtype}. Using default: 'object'")
            dtype = "object"
            
        if target_col is not None and not isinstance(target_col, str):
            logger.warning(f"Invalid 'rename_to' value in column definition: {target_col}. Ignoring rename.")
            target_col = None
            
        if source_col is not None and not isinstance(source_col, str):
            logger.warning(f"Invalid 'source' value in column definition: {source_col}. Using None.")
            source_col = None
    else:
        # Invalid input type
        logger.error(f"Invalid column definition type: {type(col_def)}. Expected string or dictionary.")
        raise TypeError(f"Column definition must be a string or dictionary, got {type(col_def)}")
    
    return source_col, nullable, dtype, target_col


def apply_column_operations(
    df: pd.DataFrame,
    schema: Dict[str, Any],
    warn_on_overwrite: bool = True
) -> pd.DataFrame:
    """
    Apply column operations defined in a schema to a DataFrame.
    
    This function centralizes the logic for:
    1. Column renaming
    2. Type conversion
    3. Default value handling
    
    Args:
        df: DataFrame to transform
        schema: Schema dictionary with column definitions
        warn_on_overwrite: Whether to log warnings when columns are overwritten
        
    Returns:
        Transformed DataFrame
    
    Raises:
        KeyError: If required source columns are missing
        TypeError: If column definitions are invalid
        ValueError: If type conversion fails
    """
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Extract column definitions
    column_defs = get_column_definitions(schema)
    if not column_defs:
        return result  # Return original if no column definitions
        
    # Check for any source columns that might be missing
    for col_name, col_def in column_defs.items():
        # Extract column info
        source_col, nullable, dtype, target_col = extract_column_info(col_def)
        
        # Handle source column mapping
        if source_col is not None:
            if source_col not in result.columns:
                logger.error(f"Source column '{source_col}' not found in DataFrame for '{col_name}'")
                raise KeyError(f"Source column '{source_col}' not found in DataFrame")
                
            # Create target column from source
            if target_col is None:
                target_col = col_name  # Use col_name from schema if no rename specified
                
            # Check if we're overwriting existing columns
            if target_col in result.columns and warn_on_overwrite:
                logger.warning(f"Column '{target_col}' already exists and will be overwritten")
                
            # Copy data from source to target column
            result[target_col] = result[source_col]
            
        # If it's just a type definition for an existing column, ensure it exists
        elif col_name not in result.columns:
            # Column doesn't exist, check if it's nullable
            if not nullable:
                logger.error(f"Non-nullable column '{col_name}' missing from DataFrame")
                raise KeyError(f"Required column '{col_name}' missing from DataFrame")
            else:
                # Create empty column with appropriate type
                result[col_name] = pd.Series(dtype=map_type_string_to_pandas(dtype))
                
        # Convert column to specified type
        if dtype != "object":
            result = convert_column(result, target_col or col_name, dtype)
        
    return result


def convert_column(df: pd.DataFrame, column_name: str, dtype: str) -> pd.DataFrame:
    """
    Convert a DataFrame column to the specified type.
    
    Args:
        df: DataFrame to modify
        column_name: Column to convert
        dtype: Type to convert to as string ('int', 'float', etc.)
        
    Returns:
        Modified DataFrame with any conversion failures tracked in attrs
    
    Raises:
        KeyError: If column_name doesn't exist in DataFrame
        ValueError: If type conversion fails and is critical
    """
    # Verify column exists
    if column_name not in df.columns:
        logger.error(f"Cannot convert column '{column_name}': not found in DataFrame")
        raise KeyError(f"Column '{column_name}' not found in DataFrame")

    # Make a copy to avoid modifying the original
    result = df.copy()
    
    try:
        # Handle categorical type specially
        if is_type_in_category(dtype, CATEGORY_TYPES):
            result[column_name] = result[column_name].astype('category')
        # Handle list type specially
        elif is_type_in_category(dtype, LIST_TYPES):
            result[column_name] = result[column_name].apply(coerce_to_list)
        # Handle other types using our mapping function
        else:
            pandas_dtype = map_type_string_to_pandas(dtype)
            result[column_name] = result[column_name].astype(pandas_dtype)
    except (ValueError, TypeError) as e:
        # Log error and track in DataFrame attrs
        error_msg = f"Failed to convert column '{column_name}' to {dtype}: {str(e)}"
        logger.warning(error_msg)
        _track_conversion_failure(result, column_name, dtype, error_msg)
        
        # Re-raise for non-nullable types
        if not is_type_in_category(dtype, STR_TYPES + OBJECT_TYPES):
            raise ValueError(error_msg) from e
            
    return result


def _track_conversion_failure(df: pd.DataFrame, column_name: str, target_type: str, error_msg: str) -> None:
    """
    Track a type conversion failure in DataFrame attributes.
    
    Args:
        df: DataFrame to modify
        column_name: Name of column that failed conversion
        target_type: Type that was attempted
        error_msg: Error message from the conversion attempt
    """
    # Initialize conversion_failures attribute if it doesn't exist
    if 'conversion_failures' not in df.attrs:
        df.attrs['conversion_failures'] = {}
        
    # Add failure record
    df.attrs['conversion_failures'][column_name] = {
        'target_type': target_type,
        'error': error_msg,
        'timestamp': pd.Timestamp.now().isoformat()
    }


def coerce_to_list(val: Any) -> List[Any]:
    """
    Convert 'val' into a Python list, row by row.
    - If val is already a list, return it.
    - If val is a numpy array, call val.tolist().
    - If val is a string representation of a list, try parsing it. 
    - Otherwise, wrap val in a list.
    
    Args:
        val: Value to convert to a list
        
    Returns:
        List representation of the value
        
    Note:
        This function is a wrapper around the more comprehensive implementation 
        in value_utils.py, maintained for backward compatibility.
    """
    return coerce_to_list_value_utils(val)


def map_type_string_to_pandas(type_str: str) -> Any:
    """
    Map string type names to pandas/Python types for use with Pandera schemas.
    
    This is a central function for mapping type names to pandas dtypes.
    This ensures consistent type handling across the codebase.
    
    Args:
        type_str: String representation of a type
        
    Returns:
        Pandas dtype object for consistent schema validation
    
    Raises:
        ValueError: If type_str is not a supported type name
    """
    if not isinstance(type_str, str):
        logger.error(f"Type must be a string, got {type(type_str)}")
        raise TypeError(f"Type must be a string, got {type(type_str)}")
        
    type_lower = type_str.lower()
    
    # Use the centralized type mapping dictionary
    if type_lower in TYPE_MAPPING:
        return TYPE_MAPPING[type_lower]
        
    # Use direct numpy types if needed
    if type_lower in ['float32']:
        return np.float32
    if type_lower in ['int32']:
        return np.int32
        
    # Default to object type for unknown types
    logger.warning(f"Unknown type '{type_str}', using 'object' type")
    return pd.ObjectDtype()


def ensure_1d(array, name: str) -> np.ndarray:
    """
    Ensure an array is one-dimensional.
    
    Args:
        array: Array to check
        name: Name of the array for error messages
    
    Returns:
        1D numpy array
    
    Raises:
        ValueError: If array cannot be converted to 1D
    """
    # Handle None or empty
    if array is None:
        return np.array([])
    
    # Convert to numpy array
    arr = np.asarray(array)
    
    # Handle scalar
    if arr.ndim == 0:
        return np.array([arr.item()])
    
    # Handle 1D
    if arr.ndim == 1:
        return arr
    
    # Handle 2D with one dimension of size 1
    if arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
        return arr.flatten()
    
    # Handle array-like objects
    try:
        if hasattr(array, 'flatten'):
            flattened = array.flatten()
            return flattened
        elif hasattr(array, 'ravel'):
            raveled = array.ravel()
            return raveled
    except Exception as e:
        logger.error(f"Failed to flatten {name}: {e}")
    
    # If we get here, the array is not convertible to 1D
    raise ValueError(f"{name} must be 1D or convertible to 1D, got shape {arr.shape}")


def get_column_definitions(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract column definitions from a schema.
    
    Args:
        schema: Schema dictionary
        
    Returns:
        Dictionary mapping column names to their definitions
    
    Raises:
        TypeError: If schema is not a dictionary
        KeyError: If schema is missing required keys
    """
    if not isinstance(schema, dict):
        logger.error(f"Schema must be a dictionary, got {type(schema)}")
        raise TypeError(f"Schema must be a dictionary, got {type(schema)}")
        
    # Check for 'column_mappings' key
    if 'column_mappings' not in schema:
        logger.warning("Schema does not contain 'column_mappings' key")
        return {}
        
    column_mappings = schema['column_mappings']
    
    # Combine data and metadata columns
    data_columns = column_mappings.get('data_columns', {})
    metadata_columns = column_mappings.get('metadata_columns', {})
    
    # Return combined dictionary
    return {**data_columns, **metadata_columns}


def check_types(df: pd.DataFrame, types_dict: Dict[str, str]) -> Dict[str, bool]:
    """
    Check if DataFrame columns match the specified types.
    
    Args:
        df: DataFrame to check
        types_dict: Dictionary mapping column names to expected type strings
        
    Returns:
        Dictionary mapping column names to boolean results (True if type matches)
    
    Raises:
        KeyError: If a column in types_dict is not in the DataFrame
    """
    results = {}
    
    for col_name, expected_type in types_dict.items():
        if col_name not in df.columns:
            logger.error(f"Column '{col_name}' not found in DataFrame")
            raise KeyError(f"Column '{col_name}' not found in DataFrame")
            
        # Get actual pandas dtype
        actual_dtype = df[col_name].dtype
        
        # Convert expected type string to pandas dtype for comparison
        expected_dtype = map_type_string_to_pandas(expected_type)
        
        # Compare types, handling special cases
        if is_type_in_category(expected_type, INT_TYPES):
            results[col_name] = pd.api.types.is_integer_dtype(actual_dtype)
        elif is_type_in_category(expected_type, FLOAT_TYPES):
            results[col_name] = pd.api.types.is_float_dtype(actual_dtype)
        elif is_type_in_category(expected_type, STR_TYPES):
            results[col_name] = pd.api.types.is_string_dtype(actual_dtype)
        elif is_type_in_category(expected_type, BOOL_TYPES):
            results[col_name] = pd.api.types.is_bool_dtype(actual_dtype)
        elif is_type_in_category(expected_type, DATE_TYPES):
            results[col_name] = pd.api.types.is_datetime64_dtype(actual_dtype)
        elif is_type_in_category(expected_type, LIST_TYPES):
            # List type is difficult to check directly, so we just verify it's object type
            results[col_name] = pd.api.types.is_object_dtype(actual_dtype)
        elif is_type_in_category(expected_type, CATEGORY_TYPES):
            results[col_name] = pd.api.types.is_categorical_dtype(actual_dtype)
        else:
            # For unknown types, default to True to be permissive
            results[col_name] = True
            
    return results
