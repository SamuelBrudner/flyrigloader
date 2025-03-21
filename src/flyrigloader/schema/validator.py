"""
validator.py - Module for schema validation using Pandera with fallbacks.

This module provides functionality for validating DataFrames against schemas,
with robust fallbacks when Pandera is not available. It handles schema creation,
validation, and application with a consistent API regardless of the underlying
implementation.
"""


from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd
from loguru import logger

from ..utils.imports import has_pandera, import_pandera
from .operations import map_type_string_to_pandas, apply_column_operations

pa = import_pandera() if has_pandera() else None


def create_schema_from_dict(schema_dict: Dict[str, Any]) -> Optional[Any]:
    """
    Create a Pandera schema from a dictionary definition.
    
    Args:
        schema_dict: Dictionary with column definitions structured as follows:
            {
                "column_mappings": {
                    "data_columns": {
                        "column_name": <type_or_config>,  # e.g., "int64" or {"dtype": "int64", ...}
                        ...
                    },
                    "metadata_columns": {
                        "column_name": <type_or_config>,
                        ...
                    }
                },
                "strict": bool  # Whether to enforce that DataFrame contains only specified columns
            }
            
            where <type_or_config> can be either:
            1. A string specifying the data type (e.g., "int64", "float", "string")
            2. A dictionary with configuration options:
                {
                    "dtype": str,           # Required: Data type (e.g., "int64", "float", "string")
                    "nullable": bool,       # Optional: Whether column can contain nulls (default: True)
                    "unique": bool,         # Optional: Whether column values must be unique (default: False)
                    "optional": bool,       # Optional: Whether column is required (default: False)
                    "checks": List[Dict],   # Optional: Advanced validation checks (see examples)
                }
    
    Returns:
        Pandera DataFrameSchema or None if pandera is not installed
    """
    if not has_pandera():
        logger.warning("Pandera is not installed. Cannot create schema.")
        return None
        
    # Create Pandera columns for data and metadata columns using dictionary comprehensions
    data_columns = {
        col_name: _create_pandera_column(col_def) 
        for col_name, col_def in schema_dict.get("column_mappings", {}).get("data_columns", {}).items()
    }
    
    metadata_columns = {
        col_name: _create_pandera_column(col_def)
        for col_name, col_def in schema_dict.get("column_mappings", {}).get("metadata_columns", {}).items()
    }
    
    # Create and return the schema with combined columns using the dictionary union operator
    return pa.DataFrameSchema(
        columns=data_columns | metadata_columns,
        strict=schema_dict.get("strict", False)
    )


def _create_pandera_column(col_def: Union[str, Dict[str, Any]]) -> Any:
    """
    Create a Pandera Column from a column definition.
    
    Args:
        col_def: String type name or dictionary with column configuration
        
    Returns:
        Configured Pandera Column
    """
    # Handle string type definitions
    if isinstance(col_def, str):
        # Map string types to pandas types using our centralized mapping function
        dtype = map_type_string_to_pandas(col_def)
        return pa.Column(dtype)
    
    # Handle dictionary definitions
    # Get the data type and convert from string if needed
    dtype = map_type_string_to_pandas(col_def.get("dtype", "object"))
    
    # Basic column configuration
    nullable = col_def.get("nullable", True)
    unique = col_def.get("unique", False)
    required = not col_def.get("optional", False)
    
    # Process advanced validation checks if provided
    checks = _process_checks(col_def)
    
    # Create the column with all configurations
    return pa.Column(
        dtype,
        nullable=nullable,
        unique=unique,
        required=required,
        checks=checks or None
    )


def _process_checks(col_def: Dict[str, Any]) -> List[Any]:
    """
    Process checks configuration from a column definition.
    
    Args:
        col_def: Column definition dictionary containing checks
        
    Returns:
        List of Pandera Check objects
    """
    checks = []
    
    # Process advanced validation checks if provided
    if "checks" not in col_def:
        return checks
        
    for check_def in col_def["checks"]:
        check_type = check_def.get("check_type")
        
        # Handle different types of checks
        if check_type == "in_range":
            # Numeric range check
            min_val = check_def.get("min_value")
            max_val = check_def.get("max_value")
            
            if min_val is not None and max_val is not None:
                checks.append(pa.Check.in_range(min_value=min_val, max_value=max_val))
            elif min_val is not None:
                checks.append(pa.Check.ge(min_val))
            elif max_val is not None:
                checks.append(pa.Check.le(max_val))
        
        elif check_type == "str_matches" and (pattern := check_def.get("pattern")):
            # String regex pattern check
            checks.append(pa.Check.str_matches(pattern))
        
        elif check_type == "isin" and (allowed_values := check_def.get("values", [])):
            # Value is in a set of allowed values
            checks.append(pa.Check.isin(allowed_values))
        
        elif check_type == "str_length":
            # String length check
            min_len = check_def.get("min_length")
            max_len = check_def.get("max_length")
            
            if min_len is not None and max_len is not None:
                checks.append(pa.Check.str_length(min_value=min_len, max_value=max_len))
            elif min_len is not None:
                checks.append(pa.Check.str_length(min_value=min_len))
            elif max_len is not None:
                checks.append(pa.Check.str_length(max_value=max_len))
    
    return checks


def validate_dataframe(
    df: pd.DataFrame, 
    schema: Union[Dict[str, Any], Any],
    schema_name: str = "unnamed_schema"
) -> Tuple[bool, List[str]]:
    """
    Validate a DataFrame against a schema.
    
    Args:
        df: DataFrame to validate
        schema: Dictionary schema definition or Pandera DataFrameSchema
        schema_name: Name of the schema for error messages
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    if not has_pandera():
        return False, ["Pandera is not installed. Cannot validate DataFrame."]
    
    # Convert dict schema to Pandera schema if needed
    if not isinstance(schema, pa.DataFrameSchema):
        pa_schema = create_schema_from_dict(schema)
        if pa_schema is None:
            return False, [f"Failed to create Pandera schema from dictionary ({schema_name})"]
    else:
        pa_schema = schema
    
    # Validate the DataFrame
    try:
        pa_schema.validate(df)
        return True, []
    except Exception as e:
        error_message = str(e)
        logger.warning(f"DataFrame validation failed: {error_message}")
        return False, [error_message]


def validate_schema_file(
    df: pd.DataFrame,
    schema_path: str
) -> Tuple[bool, List[str]]:
    """
    Validate a DataFrame against a schema loaded from a file.
    
    Args:
        df: DataFrame to validate
        schema_path: Path to the schema file (YAML)
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    from ..utils.files import safe_load_yaml
    
    # Load schema from file
    try:
        schema_dict = safe_load_yaml(schema_path)
        if not schema_dict:
            return False, [f"Failed to load schema from {schema_path}"]
    except Exception as e:
        return False, [f"Error loading schema from {schema_path}: {str(e)}"]
    
    # Validate using the loaded schema
    return validate_dataframe(df, schema_dict, schema_name=schema_path)


def _try_create_pandera_schema(schema: Union[Dict[str, Any], Any]) -> Optional[Any]:
    """
    Try to create a Pandera schema from a schema dictionary or return the existing schema.
    
    Args:
        schema: Dictionary schema definition or Pandera DataFrameSchema
        
    Returns:
        Pandera DataFrameSchema or None if creation fails
    """
    if has_pandera() and isinstance(schema, pa.DataFrameSchema):
        return schema
    return create_schema_from_dict(schema)


def apply_schema(df: pd.DataFrame, schema: Union[Dict[str, Any], Any]) -> pd.DataFrame:
    """
    Apply a schema to a DataFrame, converting columns to their specified types.
    
    This function transforms a DataFrame according to the schema, applying type
    conversions and validations. It attempts to use Pandera's coerce feature if available.
    
    Args:
        df: DataFrame to apply schema to
        schema: Dictionary schema definition or Pandera DataFrameSchema
        
    Returns:
        DataFrame with applied schema
    """
    result_df = df.copy()

    # If Pandera is not available, use our fallback apply_column_operations
    if not has_pandera():
        logger.warning("Pandera not installed. Using basic schema transformations.")
        return _extracted_from_apply_schema_20(
            schema,
            result_df,
            "Cannot apply non-dictionary schema without Pandera. Returning original DataFrame.",
        )
    # Use Pandera for column transformations if available
    try:
        # Convert dict schema to Pandera schema if needed
        pa_schema = _try_create_pandera_schema(schema)
        if pa_schema is None:
            return _extracted_from_apply_schema_20(
                schema,
                result_df,
                "Cannot create Pandera schema. Returning original DataFrame.",
            )
        # Use Pandera's coerce feature to transform the DataFrame
        result_df = pa_schema.validate(result_df, lazy=True, inplace=False)
        return result_df
    except Exception as e:
        logger.warning(f"Error applying schema with Pandera: {str(e)}")

        # Fall back to our centralized column operations if Pandera fails
        if isinstance(schema, dict):
            return apply_column_operations(result_df, schema)

    return result_df


# TODO Rename this here and in `apply_schema`
def _extracted_from_apply_schema_20(schema, result_df, arg2):
    if isinstance(schema, dict):
        return apply_column_operations(result_df, schema)
    logger.warning(arg2)
    return result_df


def quick_validate(df: pd.DataFrame, 
                  schema: Union[Dict[str, Any], Any] = None,
                  schema_path: Optional[str] = None,
                  schema_name: str = "default") -> bool:
    """
    Quickly validate a DataFrame in a notebook with nicely formatted output.
    
    Args:
        df: DataFrame to validate
        schema: Dictionary schema definition or Pandera DataFrameSchema
        schema_path: Path to a schema file (YAML) to load
        schema_name: Human-readable name for error reporting
        
    Returns:
        True if validation passed, False otherwise
    """
    from ..utils.files import safe_load_yaml
    
    # First get the schema if path is provided
    if schema_path:
        try:
            schema = safe_load_yaml(schema_path)
            schema_name = schema_path
        except Exception as e:
            logger.error(f"Error loading schema from {schema_path}: {str(e)}")
            return False
    
    # Cannot proceed without a schema
    if schema is None:
        logger.error("No schema provided for validation. Please provide either schema or schema_path.")
        return False
    
    # Validate the DataFrame
    is_valid, error_messages = validate_dataframe(df, schema, schema_name)
    
    # Create a nice output for notebooks
    if is_valid:
        logger.success(f"✅ DataFrame validation passed against schema '{schema_name}'")
    else:
        logger.error(f"❌ DataFrame validation failed against schema '{schema_name}'")
        for i, msg in enumerate(error_messages, 1):
            logger.error(f"  Error {i}: {msg}")
    
    return is_valid


def get_schema_by_name(
    schema_name: Optional[str] = None,
    schema_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Get a schema by name or path.
    
    This function provides a central way to get schemas, with fallbacks and
    config-based resolution.
    
    Args:
        schema_name: Name of a schema defined in config
        schema_path: Direct path to a schema file
        config: Configuration dictionary that may contain schema definitions
        
    Returns:
        Schema dictionary or None if not found
    """
    from ..utils.files import safe_load_yaml

    # If direct path is provided, load it
    if schema_path:
        return safe_load_yaml(schema_path)

    # If name and config are provided, look up in config
    if schema_name and config:
        schema_info = config.get('schemas', {}).get(schema_name)
        if schema_info and 'path' in schema_info:
            return safe_load_yaml(schema_info['path'])

    # If just a name but no config, or if lookup failed
    if schema_name:
        # Try common locations
        common_paths = [
            "conf/schemas/" + schema_name + ".yaml",
            "conf/" + schema_name + ".yaml",
            "schemas/" + schema_name + ".yaml",
            "conf/column_definitions.yaml"  # Default fallback
        ]

        for path in common_paths:
            if schema := safe_load_yaml(path):
                return schema

    # Last resort: return a minimal default schema
    return {
        "schema_name": "default",
        "schema_version": "1.0.0",
        "column_mappings": {
            "data_columns": {
                "t": "float",
                "x": "float", 
                "y": "float",
                "signal": "float"
            },
            "metadata_columns": {
                "rig": "str",
                "date": "str"
            }
        }
    }