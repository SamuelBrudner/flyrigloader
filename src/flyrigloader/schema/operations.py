"""
operations.py - Unified schema operations module.

This module centralizes all column transformation operations (renaming, type conversion, etc.)
providing a clean, consistent interface for:
1. Column renaming
2. Type conversions 
3. Schema transformations

Works with both dictionary-based schemas and Pandera schemas.
"""

import contextlib
import numpy as np
import pandas as pd
from loguru import logger
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# Define type sets at module level to avoid repeated creation
INT_TYPES = {"int", "int32", "int64", "integer"}
FLOAT_TYPES = {"float", "float32", "float64", "numeric"}
STR_TYPES = {"str", "string", "text"}
BOOL_TYPES = {"bool", "boolean"}
LIST_TYPES = {"list", "array"}
DATE_TYPES = {"datetime", "date"}
CATEGORY_TYPES = {"category"}


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
        # Handle renames
        if "rename_to" in col_def:
            target_col = col_def["rename_to"]
        # Handle source columns
        if "source" in col_def:
            source_col = col_def["source"]
        # Extract type information
        dtype = col_def.get("dtype", "object")
        nullable = col_def.get("nullable", True)
    
    return source_col, nullable, dtype, target_col


def create_empty_column(dtype: str, length: int) -> pd.Series:
    """
    Create an empty column with the appropriate dtype.
    
    Args:
        dtype: Type name as string
        length: Length of the Series to create
        
    Returns:
        Series with appropriate null values
    """
    if is_type_in_category(dtype, FLOAT_TYPES):
        return pd.Series([np.nan] * length)
    elif is_type_in_category(dtype, INT_TYPES):
        # Can't use NaN for integer types, use None/null instead
        return pd.Series([None] * length, dtype=pd.Int64Dtype())
    elif is_type_in_category(dtype, BOOL_TYPES):
        return pd.Series([None] * length, dtype=pd.BooleanDtype())
    else:
        # For string/object columns
        return pd.Series([None] * length)


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
    """
    result_df = df.copy()
    overwritten_columns = []
    
    # Extract column mappings from schema
    column_mappings = schema.get("column_mappings", {})
    
    # Process both data columns and metadata columns
    for section in ["data_columns", "metadata_columns"]:
        section_mappings = column_mappings.get(section, {})
        
        for col_name, col_def in section_mappings.items():
            # Extract column information
            source_col, nullable, dtype, target_col = extract_column_info(col_def)
            
            # Set defaults if not provided
            if source_col is None:
                source_col = col_name
            if target_col is None:
                target_col = col_name
            
            # Skip if this column should be built from a different source column
            # and we're just processing the target
            if source_col != col_name and source_col in result_df.columns:
                continue
                
            # Check if source column exists in DataFrame
            if source_col in result_df.columns:
                # Check for overwrite condition
                if target_col != source_col and target_col in result_df.columns:
                    overwritten_columns.append(target_col)
                    if warn_on_overwrite:
                        logger.warning(f"Column '{target_col}' already exists and will be overwritten by data from '{source_col}'")
                
                # Rename the column if needed
                if target_col != source_col:
                    result_df[target_col] = result_df[source_col]
                    
                # Apply type conversion if specified
                if dtype:
                    result_df = convert_column(result_df, target_col, dtype)
            else:
                # Handle missing columns that should exist according to schema
                if not nullable:
                    logger.warning(f"Required column '{source_col}' missing, schema expected it to be non-nullable")
                
                # Create empty column with appropriate dtype
                result_df[target_col] = create_empty_column(dtype, len(result_df))
    
    # Add a warning attribute for traceability if columns were overwritten
    if overwritten_columns and warn_on_overwrite:
        result_df.attrs['overwritten_columns'] = overwritten_columns
        logger.warning(f"Column operation resulted in {len(overwritten_columns)} overwritten columns: {overwritten_columns}")
    
    return result_df


def convert_column(df: pd.DataFrame, column_name: str, dtype: str) -> pd.DataFrame:
    """
    Convert a DataFrame column to the specified type.
    
    Args:
        df: DataFrame to modify
        column_name: Column to convert
        dtype: Type to convert to as string ('int', 'float', etc.)
        
    Returns:
        Modified DataFrame
    """
    result_df = df.copy()
    
    try:
        # Convert the column based on the type category
        if is_type_in_category(dtype, INT_TYPES):
            # Use pandas nullable integer type for columns that might have NaN values
            result_df[column_name] = pd.to_numeric(result_df[column_name], errors='coerce').astype(pd.Int64Dtype())
        elif is_type_in_category(dtype, FLOAT_TYPES):
            result_df[column_name] = pd.to_numeric(result_df[column_name], errors='coerce')
        elif is_type_in_category(dtype, STR_TYPES):
            # Handle potential nan values by converting them to None first
            result_df[column_name] = result_df[column_name].map(lambda x: str(x) if pd.notna(x) else None)
        elif is_type_in_category(dtype, BOOL_TYPES):
            # Use pandas nullable boolean type
            result_df[column_name] = result_df[column_name].astype(pd.BooleanDtype())
        elif is_type_in_category(dtype, LIST_TYPES):
            # Convert to list, handling various input formats
            result_df[column_name] = result_df[column_name].map(coerce_to_list)
        elif is_type_in_category(dtype, CATEGORY_TYPES):
            result_df[column_name] = result_df[column_name].astype('category')
        elif is_type_in_category(dtype, DATE_TYPES):
            result_df[column_name] = pd.to_datetime(result_df[column_name], errors='coerce')
        else:
            logger.warning(f"Unknown dtype '{dtype}' for column '{column_name}', leaving as is")
    except Exception as e:
        logger.warning(f"Error converting column '{column_name}' to {dtype}: {e}")
    
    return result_df


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
    """
    import ast
    
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, str):
        # Try to parse string as a list
        with contextlib.suppress(SyntaxError, ValueError):
            # Strip whitespace before checking for list-like structure
            cleaned_val = val.strip()
            if cleaned_val.startswith('[') and cleaned_val.endswith(']'):
                return ast.literal_eval(cleaned_val)
    # Default: wrap in a list
    return [val]


def map_type_string_to_pandas(type_str: str) -> Any:
    """
    Map string type names to pandas/Python types for use with Pandera schemas.
    
    Args:
        type_str: String representation of a type
        
    Returns:
        Pandas dtype object for consistent schema validation
    """
    if is_type_in_category(type_str, INT_TYPES):
        return pd.Int64Dtype()  # Use nullable integer type
    elif is_type_in_category(type_str, FLOAT_TYPES):
        return pd.Float64Dtype()  # Use nullable float type instead of float
    elif is_type_in_category(type_str, STR_TYPES):
        return pd.StringDtype()  # Use pandas string type instead of str
    elif is_type_in_category(type_str, BOOL_TYPES):
        return pd.BooleanDtype()  # Use nullable boolean type
    elif is_type_in_category(type_str, LIST_TYPES):
        return pd.ArrowDtype(storage="list")  # Use Arrow-backed list type if available
    elif is_type_in_category(type_str, CATEGORY_TYPES):
        return pd.CategoricalDtype()
    elif is_type_in_category(type_str, DATE_TYPES):
        return pd.DatetimeTZDtype(tz=None)
    else:
        return pd.StringDtype()  # Default to string type instead of object


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
    arr = np.asarray(array)
    if arr.ndim == 2 and 1 in arr.shape:
        return arr.ravel()
    elif arr.ndim == 1:
        return arr
    elif name == 'signal' and arr.ndim == 2 and arr.shape[1] == 3:
        return arr[:,0]
    else:
        raise ValueError(
            f"Column '{name}' has shape {arr.shape}, not strictly 1D. "
            "Customize 'ensure_1d' if needed."
        )


def get_column_definitions(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract column definitions from a schema.
    
    Args:
        schema: Schema dictionary
        
    Returns:
        Dictionary mapping column names to their definitions
    """
    column_defs = {}
    
    # Process data columns
    for col_name, col_def in schema.get("column_mappings", {}).get("data_columns", {}).items():
        column_defs[col_name] = col_def if isinstance(col_def, dict) else {"dtype": col_def}
    
    # Process metadata columns
    for col_name, col_def in schema.get("column_mappings", {}).get("metadata_columns", {}).items():
        column_defs[col_name] = col_def if isinstance(col_def, dict) else {"dtype": col_def}
    
    return column_defs