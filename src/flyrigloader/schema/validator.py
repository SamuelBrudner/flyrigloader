"""
validator.py - Module for schema validation using Pandera with fallbacks.

This module provides high-level functionality for validating DataFrames against schemas,
with robust fallbacks when Pandera is not available. It handles schema creation,
validation, and application with a consistent API regardless of the underlying
implementation.

This module uses the tuple return pattern (result, metadata) for consistent error handling
at the orchestration level, while core utilities throw exceptions directly.

Separation of concerns:
- core/utils/schema_utils.py: Low-level schema utilities with minimal dependencies
  (used by both this module and other parts of the codebase)
- schema/validator.py (this module): Higher-level schema validation with Pandera integration
  (used by pipeline and data processing modules)
"""


from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
import json
import os
import re
from copy import deepcopy
from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union, cast

import jsonschema
import numpy as np
import pandas as pd
from loguru import logger

from ..core.utils import (
    schema_utils,
    import_utils,
    create_metadata,
    create_error_metadata
)
from ..core.utils.file_utils import safe_load_yaml

# Import Pandera if available
has_pandera = import_utils.has_pandera
import_pandera = import_utils.import_pandera

pa = import_pandera() if has_pandera() else None


def create_schema_from_dict(schema_dict: Dict[str, Any]) -> Tuple[Optional[Any], Dict[str, Any]]:
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
                "strict": bool,  # Whether to enforce that DataFrame contains only specified columns
                "custom_validators": [  # List of custom validation function definitions
                    {
                        "name": str,     # Name of the validation rule
                        "description": str,  # Description of what the validation checks
                        "columns": List[str],  # Columns to validate or ["*"] for all columns
                        "error_message": str  # Template for error message
                    }
                ]
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
                    "description": str,     # Human-readable description of the column
                    "error_templates": Dict # Custom error message templates for this column
                }
    
    Returns:
        Tuple of (schema, metadata) where schema is a Pandera DataFrameSchema or None 
        if pandera is not installed, and metadata contains status information and any error details
    """
    metadata = create_metadata()
    
    try:
        if not has_pandera():
            logger.warning("Pandera is not installed. Cannot create schema.")
            return None, create_error_metadata(
                metadata, 
                "PanderaMissing", 
                "Pandera package is not installed"
            )
            
        # Create Pandera columns for data and metadata columns using dictionary comprehensions
        data_columns = {
            col_name: _create_pandera_column(col_name, col_def) 
            for col_name, col_def in schema_dict.get("column_mappings", {}).get("data_columns", {}).items()
        }
        
        metadata_columns = {
            col_name: _create_pandera_column(col_name, col_def)
            for col_name, col_def in schema_dict.get("column_mappings", {}).get("metadata_columns", {}).items()
        }
        
        # Extract custom validators if they exist
        custom_validators = schema_dict.get("custom_validators", [])
        
        # Create DataFrameSchema with combined columns
        schema = pa.DataFrameSchema(
            columns={**data_columns, **metadata_columns},
            strict=schema_dict.get("strict", False)
        )
        
        # Store custom validators in schema object for later use
        # We'll attach them as a custom attribute that our validation
        # functions will check for
        setattr(schema, "_custom_validators", custom_validators)
        
        return schema, metadata
    except (TypeError, AttributeError, KeyError) as e:
        logger.error(f"Error creating schema from dictionary: {e}")
        return None, create_error_metadata(
            metadata,
            "SchemaCreationError",
            f"Failed to create schema: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error creating schema: {e}")
        return None, create_error_metadata(
            metadata,
            "UnexpectedError",
            f"Unexpected error creating schema: {str(e)}"
        )


def _create_pandera_column(col_name: str, col_def: Union[str, Dict[str, Any]]) -> Any:
    """
    Create a Pandera Column from a column definition.
    
    Args:
        col_name: Name of the column (for improved error messages)
        col_def: String type name or dictionary with column configuration
        
    Returns:
        Configured Pandera Column or None if an error occurs
    """
    try:
        # Handle string type definitions
        if isinstance(col_def, str):
            # Map string types to pandas types using our centralized mapping function
            dtype = schema_utils.map_type_string_to_pandas(col_def)
            return pa.Column(dtype)
        
        # Handle dictionary definitions
        # Get the data type and convert from string if needed
        dtype = schema_utils.map_type_string_to_pandas(col_def.get("dtype", "object"))
        
        # Basic column configuration
        nullable = col_def.get("nullable", True)
        unique = col_def.get("unique", False)
        required = not col_def.get("optional", False)
        
        # Extract column description for error messages
        description = col_def.get("description", f"Column '{col_name}'")
        
        # Process advanced validation checks if provided
        checks = _process_checks(col_name, col_def)
        
        # Create the column with all configurations
        column = pa.Column(
            dtype,
            nullable=nullable,
            unique=unique,
            required=required,
            checks=checks or None
        )
        
        # Store the column description and any custom error templates as custom attributes
        setattr(column, "_description", description)
        setattr(column, "_error_templates", col_def.get("error_templates", {}))
        
        return column
    except TypeError as e:
        logger.error(f"Type error creating column '{col_name}': {e}")
        return None
    except ValueError as e:
        logger.error(f"Value error creating column '{col_name}': {e}")
        return None
    except Exception as e:
        logger.error(f"Error creating Pandera column '{col_name}': {e}")
        return None


def _process_checks(col_name: str, col_def: Dict[str, Any]) -> List[Any]:
    """
    Process checks configuration from a column definition.
    
    Args:
        col_name: Name of the column (for improved error messages)
        col_def: Column definition dictionary containing checks
        
    Returns:
        List of Pandera Check objects or empty list if an error occurs
    """
    try:
        checks = []
        
        # Process advanced validation checks if provided
        if "checks" not in col_def:
            return checks
        
        # Get custom error templates if available
        error_templates = col_def.get("error_templates", {})
        description = col_def.get("description", f"Column '{col_name}'")
            
        for check_def in col_def["checks"]:
            check_type = check_def.get("check_type")
            
            # Extract custom error message if provided
            error_message = check_def.get("error_message")
            
            # Handle different types of checks
            if check_type == "in_range":
                # Numeric range check
                min_val = check_def.get("min_value")
                max_val = check_def.get("max_value")
                
                # Use custom error message if provided, otherwise use templates
                if error_message is None:
                    if min_val is not None and max_val is not None:
                        error_message = error_templates.get(
                            "in_range", 
                            f"{description} must be between {min_val} and {max_val}"
                        )
                    elif min_val is not None:
                        error_message = error_templates.get(
                            "min_value", 
                            f"{description} must be at least {min_val}"
                        )
                    elif max_val is not None:
                        error_message = error_templates.get(
                            "max_value", 
                            f"{description} cannot exceed {max_val}"
                        )
                
                if min_val is not None and max_val is not None:
                    checks.append(pa.Check.in_range(
                        min_value=min_val, 
                        max_value=max_val,
                        error=error_message
                    ))
                elif min_val is not None:
                    checks.append(pa.Check.ge(
                        min_val, 
                        error=error_message
                    ))
                elif max_val is not None:
                    checks.append(pa.Check.le(
                        max_val,
                        error=error_message
                    ))
            
            elif check_type == "str_matches" and (pattern := check_def.get("pattern")):
                # String regex pattern check
                if error_message is None:
                    error_message = error_templates.get(
                        "str_matches", 
                        f"{description} must match pattern: {pattern}"
                    )
                checks.append(pa.Check.str_matches(
                    pattern,
                    error=error_message
                ))
            
            elif check_type == "str_length":
                # String length check
                min_len = check_def.get("min_length")
                max_len = check_def.get("max_length")
                
                if error_message is None:
                    if min_len is not None and max_len is not None:
                        error_message = error_templates.get(
                            "str_length", 
                            f"{description} length must be between {min_len} and {max_len} characters"
                        )
                    elif min_len is not None:
                        error_message = error_templates.get(
                            "min_length", 
                            f"{description} must be at least {min_len} characters"
                        )
                    elif max_len is not None:
                        error_message = error_templates.get(
                            "max_length", 
                            f"{description} cannot exceed {max_len} characters"
                        )
                
                if min_len is not None:
                    checks.append(pa.Check.str_length(
                        min_value=min_len,
                        error=error_message
                    ))
                elif max_len is not None:
                    checks.append(pa.Check.str_length(
                        max_value=max_len,
                        error=error_message
                    ))
            
            elif check_type == "isin" and (allowed_values := check_def.get("values")):
                # Check if value is in a set of allowed values
                if error_message is None:
                    # Format the allowed values for display in error message
                    values_str = ", ".join([str(v) for v in allowed_values[:5]])
                    if len(allowed_values) > 5:
                        values_str += f" ... and {len(allowed_values) - 5} more options"
                    
                    error_message = error_templates.get(
                        "isin", 
                        f"{description} must be one of: {values_str}"
                    )
                checks.append(pa.Check.isin(
                    allowed_values,
                    error=error_message
                ))
                
            elif check_type == "custom" and (check_func := check_def.get("check_function")):
                # Custom validation function
                if error_message is None:
                    func_name = getattr(check_func, "__name__", "custom check")
                    error_message = error_templates.get(
                        "custom", 
                        f"{description} failed validation: {func_name}"
                    )
                checks.append(pa.Check(
                    check_func,
                    error=error_message
                ))
        
        return checks
    except (TypeError, KeyError, ValueError) as e:
        logger.error(f"Error processing checks for column '{col_name}': {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error processing checks for column '{col_name}': {e}")
        return []


def _run_custom_validators(
    df: pd.DataFrame, 
    schema: Any,
    schema_name: str
) -> List[str]:
    """
    Run custom validators defined in the schema.
    
    Args:
        df: DataFrame to validate
        schema: Pandera DataFrameSchema with custom validators
        schema_name: Name of the schema for error messages
        
    Returns:
        List of error messages from custom validators
    """
    try:
        error_messages = []
        
        # Check if schema has custom validators
        if not hasattr(schema, "_custom_validators"):
            return error_messages
        
        # Get custom validators from schema
        custom_validators = getattr(schema, "_custom_validators", [])
        
        # Run each validator
        for validator in custom_validators:
            validator_name = validator.get("name", "Unnamed validator")
            columns = validator.get("columns", [])
            validator_func = validator.get("function")
            
            # Skip if no function is defined (this happens when loaded from YAML)
            # In that case, the validator should be registered separately
            if validator_func is None:
                continue
            
            # If columns is ["*"], validate all columns
            if columns == ["*"]:
                columns = df.columns.tolist()
            
            # Only validate columns that exist in the DataFrame
            valid_columns = [c for c in columns if c in df.columns]
            
            # Skip if none of the specified columns exist
            if not valid_columns and columns:
                logger.warning(f"Schema '{schema_name}' custom validator '{validator_name}' " +
                             f"specified columns {columns} but none exist in DataFrame")
                continue
                
            # Select only the columns to validate
            df_to_validate = df[valid_columns] if valid_columns else df
            
            # Apply the validation function
            try:
                is_valid = validator_func(df_to_validate)
                
                # If validation fails, add the error message
                if not is_valid:
                    error_message = validator.get("error_message", 
                                                f"Custom validator '{validator_name}' failed")
                    error_messages.append(error_message)
            except Exception as e:
                logger.warning(f"Error in custom validator '{validator_name}': {e}")
                error_messages.append(
                    f"Error in custom validator '{validator_name}': {str(e)}"
                )
        
        return error_messages
    except (TypeError, AttributeError) as e:
        logger.error(f"Error running custom validators for schema '{schema_name}': {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error running custom validators: {e}")
        return []


def validate_dataframe(
    df: pd.DataFrame, 
    schema: Union[Dict[str, Any], Any],
    schema_name: str = "unnamed_schema"
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate a DataFrame against a schema.
    
    Args:
        df: DataFrame to validate
        schema: Dictionary schema definition or Pandera DataFrameSchema
        schema_name: Name of the schema for error messages
        
    Returns:
        Tuple of (is_valid, error_messages, metadata) where is_valid is a boolean
        indicating if validation passed, error_messages is a list of validation errors,
        and metadata contains status information and any error details
    """
    metadata = create_metadata()
    error_messages = []
    is_valid = True
    
    # Return early if DataFrame is empty
    if df.empty:
        logger.warning(f"Empty DataFrame passed to validate_dataframe for schema '{schema_name}'")
        metadata["status"] = "warning"
        metadata["message"] = "Empty DataFrame passed to validate_dataframe"
        return True, [], metadata
    
    try:
        # Handle case where schema is already a Pandera schema
        if has_pandera() and schema is not None and not isinstance(schema, dict):
            try:
                # Check if it has the validate method (is a Pandera schema)
                if hasattr(schema, 'validate'):
                    schema.validate(df)
                    
                    # Run any custom validators
                    if hasattr(schema, '_custom_validators'):
                        custom_errors = _run_custom_validators(df, schema, schema_name)
                        if custom_errors:
                            error_messages.extend(custom_errors)
                            is_valid = False
                            
                    # If we got here and there are no errors, it's valid
                    if not error_messages:
                        return True, [], metadata
                else:
                    logger.warning(f"Object passed as schema for '{schema_name}' does not appear to be a valid Pandera schema")
                    error_messages.append(f"Object passed as schema for '{schema_name}' is not a valid Pandera schema")
                    is_valid = False
                    
            except Exception as e:
                # If it's a Pandera SchemaError, format it nicely
                if hasattr(pa, 'errors') and isinstance(e, pa.errors.SchemaError):
                    error_messages = _format_pandera_errors(e, schema_name)
                else:
                    error_messages.append(f"Error validating DataFrame with schema '{schema_name}': {e}")
                is_valid = False
        
        # Handle dictionary-based schema
        elif isinstance(schema, dict):
            # Try to create a Pandera schema first
            schema_obj, schema_meta = create_schema_from_dict(schema)
            
            if schema_obj is not None:
                try:
                    schema_obj.validate(df)
                    
                    # Run any custom validators
                    if hasattr(schema_obj, '_custom_validators'):
                        custom_errors = _run_custom_validators(df, schema_obj, schema_name)
                        if custom_errors:
                            error_messages.extend(custom_errors)
                            is_valid = False
                except Exception as e:
                    # If it's a Pandera SchemaError, format it nicely
                    if hasattr(pa, 'errors') and isinstance(e, pa.errors.SchemaError):
                        error_messages = _format_pandera_errors(e, schema_name)
                    else:
                        error_messages.append(f"Error validating DataFrame with schema '{schema_name}': {e}")
                    is_valid = False
            else:
                # Fallback to basic validation if we couldn't create a Pandera schema
                logger.warning(f"Using fallback validation for schema '{schema_name}'")
                
                # Get column definitions using schema_utils
                column_defs = schema_utils.get_column_definitions(schema)
                
                # Check if required columns exist
                for col_name, col_def in column_defs.items():
                    # Extract nullable info
                    _, nullable, dtype, _ = schema_utils.extract_column_info(col_def)
                    
                    # If it's not nullable, it must exist
                    if not nullable and col_name not in df.columns:
                        error_messages.append(f"Required column '{col_name}' missing from DataFrame")
                        is_valid = False
                
                # Check column types for existing columns
                type_checks = {}
                for col_name, col_def in column_defs.items():
                    if col_name in df.columns:
                        _, _, dtype, _ = schema_utils.extract_column_info(col_def)
                        type_checks[col_name] = dtype
                
                # Use schema_utils to check types
                if type_checks:
                    try:
                        type_results = schema_utils.check_types(df, type_checks)
                        for col_name, is_match in type_results.items():
                            if not is_match:
                                error_messages.append(
                                    f"Column '{col_name}' has incorrect type. "
                                    f"Expected: {type_checks[col_name]}, "
                                    f"Found: {df[col_name].dtype}"
                                )
                                is_valid = False
                    except Exception as e:
                        error_messages.append(f"Error checking column types: {e}")
                        is_valid = False
        else:
            logger.error(f"Invalid schema type passed to validate_dataframe: {type(schema)}")
            error_messages.append(f"Invalid schema type: {type(schema)}. Expected dict or Pandera schema.")
            is_valid = False
    
    except Exception as e:
        logger.error(f"Unexpected error in validate_dataframe: {e}")
        error_messages.append(f"Unexpected error during validation: {e}")
        is_valid = False
    
    # Update metadata with validation results
    if not is_valid:
        metadata = create_error_metadata(
            metadata,
            "ValidationError",
            f"DataFrame validation failed for schema '{schema_name}'"
        )
        metadata["validation_errors"] = error_messages
    
    return is_valid, error_messages, metadata


def validate_schema_file(
    df: pd.DataFrame,
    schema_path: str
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate a DataFrame against a schema loaded from a file.
    
    Args:
        df: DataFrame to validate
        schema_path: Path to the schema file (YAML)
        
    Returns:
        Tuple of (is_valid, error_messages, metadata)
    """
    try:
        # Load schema from file
        schema_dict, metadata = get_schema_by_name(schema_path=schema_path)
        if schema_dict is None:
            return False, [], metadata
        
        # Use the filename as the schema name for better error messages
        import os
        schema_name = os.path.basename(schema_path)
        
        # Validate using the loaded schema
        return validate_dataframe(df, schema_dict, schema_name=schema_name)
    except (ValueError, TypeError) as e:
        logger.error(f"Error validating DataFrame against schema from '{schema_path}': {e}")
        return False, [f"Error validating DataFrame: {e}"], create_error_metadata(e)
    except Exception as e:
        logger.error(f"Unexpected error during schema validation: {e}")
        return False, [f"Unexpected error during validation: {e}"], create_error_metadata(e)


def register_custom_validator(
    schema: Any,
    name: str,
    validator_func: Callable[[pd.DataFrame], bool],
    columns: List[str] = None,
    error_message: str = None,
    description: str = None
) -> Any:
    """
    Register a custom validator function with a schema.
    
    Args:
        schema: Pandera DataFrameSchema to add validator to
        name: Name of the validator
        validator_func: Function that takes a DataFrame and returns True if valid, False if invalid
        columns: List of column names to validate or ["*"] for all columns
        error_message: Custom error message if validation fails
        description: Description of what the validator checks
        
    Returns:
        Updated schema with custom validator
    """
    try:
        if not hasattr(schema, "_custom_validators"):
            setattr(schema, "_custom_validators", [])
        
        custom_validators = getattr(schema, "_custom_validators")
        
        validator = {
            "name": name,
            "function": validator_func,
            "columns": columns or ["*"],
            "error_message": error_message or f"Custom validation '{name}' failed",
            "description": description or f"Custom validator: {name}"
        }
        
        custom_validators.append(validator)
        return schema
    except (AttributeError, TypeError) as e:
        logger.error(f"Failed to register custom validator '{name}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error registering custom validator '{name}': {e}")
        return None


def quick_validate(
    df: pd.DataFrame, 
    schema: Union[Dict[str, Any], Any], 
    schema_name: str = "unnamed_schema"
) -> bool:
    """
    Validate a DataFrame against a schema and print results.
    
    This is a quick utility function that validates a DataFrame and prints
    the results to the console in a user-friendly format.
    
    Args:
        df: DataFrame to validate
        schema: Schema to validate against (dict or Pandera schema)
        schema_name: Name to use for the schema in error messages
        
    Returns:
        True if validation passed, False otherwise
    """
    # Validate the DataFrame
    valid, error_messages, metadata = validate_dataframe(df, schema, schema_name=schema_name)
    
    # Print nicely formatted results
    if not valid:
        _print_validation_failures(df, error_messages, schema_name)
        return False
        
    print(f"✅ DataFrame successfully validated against schema '{schema_name}'")
    return True


def _print_validation_failures(
    df: pd.DataFrame,
    error_messages: List[str],
    schema_name: str
) -> None:
    """
    Print validation failures in a user-friendly format.
    
    Args:
        df: The DataFrame that failed validation
        error_messages: List of error messages from validation
        schema_name: Name of the schema used for validation
    """
    print(f"❌ DataFrame validation against schema '{schema_name}' failed:")
    for i, message in enumerate(error_messages, 1):
        # Format multi-line error messages with proper indentation
        lines = message.split('\n')
        if len(lines) > 1:
            print(f"  {i}. {lines[0]}")
            for line in lines[1:]:
                print(f"     {line}")
        else:
            print(f"  {i}. {message}")
    
    print("\nDataFrame information:")
    print(f"  - Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("  - Columns:")
    for col, dtype in df.dtypes.items():
        sample = "N/A" if df.empty else str(df[col].iloc[0])
        sample = sample if len(sample) <= 50 else sample[:47] + "..."
        print(f"    • {col} ({dtype}): {sample}")


def get_schema_by_name(
    schema_name: Optional[str] = None,
    schema_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Get a schema by name or path.
    
    This function provides a central way to get schemas, with fallbacks and
    config-based resolution.
    
    Args:
        schema_name: Name of a schema defined in config
        schema_path: Direct path to a schema file
        config: Configuration dictionary that may contain schema definitions
        
    Returns:
        Tuple of (schema, metadata) where schema is the schema dictionary or None if not found,
        and metadata contains status information and any error details
    """
    metadata = create_metadata()
    
    # Option 1: Load directly from a specified file path
    if schema_path is not None:
        try:
            schema_dict = _load_schema_file(schema_path)
            if schema_dict is None:
                return None, create_error_metadata(
                    metadata,
                    "SchemaFileLoadError",
                    f"Failed to load schema file: {schema_path}"
                )
            
            metadata["schema_source"] = f"file:{schema_path}"
            return schema_dict, metadata
        except Exception as e:
            logger.error(f"Error loading schema file '{schema_path}': {e}")
            return None, create_error_metadata(
                metadata,
                "SchemaFileLoadError",
                f"Error loading schema file '{schema_path}': {str(e)}"
            )
    
    # Option 2: Look up schema by name in config
    if schema_name is not None:
        try:
            # Case 2.1: Schema reference could be a direct file path
            if schema_name.endswith(('.yaml', '.yml', '.json')):
                schema_dict = _load_schema_file(schema_name)
                if schema_dict is not None:
                    metadata["schema_source"] = f"file:{schema_name}"
                    return schema_dict, metadata
            
            # Case 2.2: Schema reference is a name in the config
            if config is not None:
                # Look in 'schemas' section of config
                schemas = config.get('schemas', {})
                if schema_name in schemas:
                    metadata["schema_source"] = f"config:schemas.{schema_name}"
                    return schemas[schema_name], metadata
                
                # Look directly at the top level
                if schema_name in config:
                    schema_def = config[schema_name]
                    if isinstance(schema_def, dict) and "column_mappings" in schema_def:
                        metadata["schema_source"] = f"config:{schema_name}"
                        return schema_def, metadata
        except Exception as e:
            logger.error(f"Error looking up schema '{schema_name}': {e}")
            return None, create_error_metadata(
                metadata,
                "SchemaLookupError",
                f"Error looking up schema '{schema_name}': {str(e)}"
            )
    
    # Option 3: Look for a default schema
    if config is not None:
        try:
            if "default_schema" in config:
                schema_def = config["default_schema"]
                if isinstance(schema_def, dict) and "column_mappings" in schema_def:
                    metadata["schema_source"] = "config:default_schema"
                    return schema_def, metadata
        except Exception as e:
            logger.error(f"Error getting default schema: {e}")
            return None, create_error_metadata(
                metadata,
                "DefaultSchemaError",
                f"Error getting default schema: {str(e)}"
            )
    
    # No schema found
    return None, create_error_metadata(
        metadata,
        "SchemaNotFound",
        f"Schema not found: {schema_name or 'default'}"
    )


def _load_schema_file(schema_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a schema from a file.
    
    Args:
        schema_path: Path to schema file
        
    Returns:
        Schema dictionary or None if loading fails
        
    Raises:
        FileNotFoundError: If the schema file does not exist
        ValueError: If the schema file has an unsupported format
    """
    try:
        if not os.path.exists(schema_path):
            logger.error(f"Schema file not found: {schema_path}")
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
            
        if schema_path.endswith(('.yaml', '.yml')):
            return safe_load_yaml(schema_path)
        elif schema_path.endswith('.json'):
            with open(schema_path, 'r') as f:
                return json.load(f)
        else:
            logger.error(f"Unsupported schema file format: {schema_path}")
            raise ValueError(f"Unsupported schema file format: {schema_path}")
    except Exception as e:
        logger.error(f"Error loading schema file '{schema_path}': {e}")
        raise


# Define base schema for configuration validation
CONFIG_BASE_SCHEMA = {
    "type": "object",
    "properties": {
        "paths": {
            "type": "object",
            "description": "File system paths used by the application"
        },
        "experiments": {
            "type": "object",
            "description": "Experiment definitions",
            "additionalProperties": {
                "type": "object",
                "required": ["datasets"],
                "properties": {
                    "datasets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of dataset names used in this experiment"
                    },
                    "schema": {
                        "type": "string",
                        "description": "Name of schema to validate experiment data"
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description of the experiment"
                    }
                }
            }
        },
        "datasets": {
            "type": "object",
            "description": "Dataset definitions",
            "additionalProperties": {
                "type": "object",
                "required": ["path_pattern"],
                "properties": {
                    "path_pattern": {
                        "type": "string",
                        "description": "File path pattern to match data files for this dataset"
                    },
                    "format": {
                        "type": "string",
                        "description": "File format (e.g., csv, h5, etc.)"
                    },
                    "loader": {
                        "type": "string",
                        "description": "Name of the loader function to use for this dataset"
                    }
                }
            }
        },
        "schemas": {
            "type": "object",
            "description": "Schema definitions for validation",
            "additionalProperties": {
                "oneOf": [
                    {"type": "string"},  # Path to a schema file
                    {"type": "object"}   # Inline schema definition
                ]
            }
        },
        "rigs": {
            "type": "object",
            "description": "Rig definitions and configurations"
        }
    }
}

# Schema for specific config types
SCHEMA_CONFIG_SCHEMA = {
    "type": "object",
    "required": ["column_mappings"],
    "properties": {
        "column_mappings": {
            "type": "object",
            "properties": {
                "data_columns": {
                    "type": "object",
                    "additionalProperties": {
                        "oneOf": [
                            {"type": "string"},  # Simple dtype string
                            {  # Column config object
                                "type": "object",
                                "properties": {
                                    "dtype": {"type": "string"},
                                    "nullable": {"type": "boolean"},
                                    "description": {"type": "string"},
                                    "checks": {
                                        "type": "array",
                                        "items": {"type": "object"}
                                    },
                                    "error_templates": {"type": "object"}
                                }
                            }
                        ]
                    }
                },
                "index_columns": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        },
        "strict": {"type": "boolean"}
    }
}

# Registry of schema validation functions for different config types
CONFIG_SCHEMA_REGISTRY = {
    "base_config": CONFIG_BASE_SCHEMA,
    "schema_config": SCHEMA_CONFIG_SCHEMA,
}


def validate_config(
    config: Dict[str, Any], 
    schema_type: str = "base_config", 
    custom_schema: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate a configuration dictionary against a predefined schema.
    
    Args:
        config: Configuration dictionary to validate
        schema_type: Type of schema to validate against (e.g., "base_config", "schema_config")
        custom_schema: Optional custom schema to use instead of a predefined one
        
    Returns:
        Tuple of (is_valid, error_messages, metadata)
    """
    # Select the schema to use
    if custom_schema is not None:
        schema = custom_schema
    elif schema_type in CONFIG_SCHEMA_REGISTRY:
        schema = CONFIG_SCHEMA_REGISTRY[schema_type]
    else:
        return False, [f"Unknown schema type: {schema_type}"], create_error_metadata(ValueError("Unknown schema type"))
    
    # Validate the config
    try:
        jsonschema.validate(instance=config, schema=schema)
        return True, [], create_metadata(success=True)
    except jsonschema.ValidationError as e:
        # Format the error message for better readability
        error_path = " -> ".join([str(p) for p in e.path]) if e.path else "root"
        return False, [f"Config validation error at {error_path}: {e.message}"], create_error_metadata(e)
    except Exception as e:
        return False, [f"Unexpected error during config validation: {str(e)}"], create_error_metadata(e)


def quick_validate_config(
    config: Dict[str, Any],
    schema_type: str = "base_config",
    custom_schema: Optional[Dict[str, Any]] = None,
    config_name: str = "configuration"
) -> bool:
    """
    Validate a configuration and print results in a user-friendly format.
    
    Args:
        config: Configuration dictionary to validate
        schema_type: Type of schema to validate against
        custom_schema: Optional custom schema to use
        config_name: Name to use for the configuration in error messages
        
    Returns:
        True if validation passed, False otherwise
    """
    valid, errors, metadata = validate_config(config, schema_type, custom_schema)
    
    if not valid:
        print(f"❌ {config_name} validation failed:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        
        _print_config_debug_info(config)
        return False
    
    print(f"✅ {config_name} successfully validated")
    return True


def _print_config_debug_info(config: Dict[str, Any]) -> None:
    """
    Print helpful debugging information about a configuration dictionary.
    
    Args:
        config: Configuration dictionary to print information about
    """
    print("\nConfiguration information:")
    print(f"  - Top-level keys: {', '.join(config.keys())}")
    
    # Print some nested structure information if available
    if "experiments" in config:
        print(f"  - Experiments: {', '.join(config['experiments'].keys())}")
    if "datasets" in config:
        print(f"  - Datasets: {', '.join(config['datasets'].keys())}")


def validate_config_file(
    config_path: str, 
    schema_type: str = "base_config",
    custom_schema: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Load and validate a configuration file.
    
    Args:
        config_path: Path to the configuration file (YAML)
        schema_type: Type of schema to validate against
        custom_schema: Optional custom schema to use
        
    Returns:
        Tuple of (is_valid, error_messages, metadata)
    """
    # Load the configuration
    config = safe_load_yaml(config_path)
    if config is None:
        return False, [f"Failed to load configuration from {config_path}"], create_error_metadata(IOError("Failed to load configuration"))
    
    # Validate the configuration
    return validate_config(config, schema_type, custom_schema)


def ensure_valid_config(
    config: Dict[str, Any],
    schema_type: str = "base_config",
    custom_schema: Optional[Dict[str, Any]] = None,
    error_behavior: str = "raise"
) -> Dict[str, Any]:
    """
    Ensure a configuration is valid, with options for handling invalid configs.
    
    Args:
        config: Configuration dictionary to validate
        schema_type: Type of schema to validate against
        custom_schema: Optional custom schema to use
        error_behavior: How to handle validation errors:
            - "raise": Raise a ValueError (default)
            - "warn": Log a warning and continue
            - "silent": Silently continue
            
    Returns:
        The validated configuration (unchanged)
        
    Raises:
        ValueError: If validation fails and error_behavior is "raise"
    """
    try:
        valid, errors, metadata = validate_config(config, schema_type, custom_schema)
        
        if not valid:
            error_message = "\n".join(errors)
            
            if error_behavior == "raise":
                raise ValueError(f"Invalid configuration: {error_message}")
            elif error_behavior == "warn":
                logger.warning(f"Configuration validation failed: {error_message}")
            # Silent mode: do nothing
        
        return config
    except Exception as e:
        # If any unexpected error occurs during validation
        if error_behavior == "raise":
            raise ValueError(f"Error validating configuration: {str(e)}")
        elif error_behavior == "warn":
            logger.warning(f"Configuration validation error: {str(e)}")
        return config


def register_config_schema(schema_type: str, schema: Dict[str, Any]) -> None:
    """
    Register a JSON schema for validating configuration files.
    
    Args:
        schema_type: Schema type identifier
        schema: JSON Schema dictionary
    """
    try:
        # Validate that the schema is a valid JSON Schema
        jsonschema.Draft7Validator.check_schema(schema)
    except Exception as e:
        raise ValueError(f"Invalid JSON Schema: {str(e)}")
    
    CONFIG_SCHEMA_REGISTRY[schema_type] = schema
    logger.info(f"Registered config schema: {schema_type}")


def apply_schema(
    df: pd.DataFrame, 
    schema: Union[Dict[str, Any], Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply a schema to a DataFrame, converting columns to their specified types.
    
    This function transforms a DataFrame according to the schema, applying type
    conversions and validations. It attempts to use Pandera's coerce feature if available.
    
    Args:
        df: DataFrame to apply schema to
        schema: Dictionary schema definition or Pandera DataFrameSchema
        
    Returns:
        Tuple of (result_df, metadata) where result_df is the DataFrame with schema applied
        and metadata contains status information and any error details
    """
    metadata = create_metadata()
    
    if df.empty:
        logger.warning("Empty DataFrame passed to apply_schema")
        metadata["status"] = "warning"
        metadata["message"] = "Empty DataFrame passed to apply_schema"
        return df.copy(), metadata
    
    try:
        # For Pandera schemas, try to use the built-in coerce feature
        if has_pandera() and schema is not None and not isinstance(schema, dict):
            try:
                if hasattr(schema, 'validate'):
                    # Use Pandera's coerce feature which handles type conversion
                    result_df = schema.validate(df, lazy=True, inplace=False)
                    metadata["schema_type"] = "pandera"
                    return result_df, metadata
                else:
                    logger.warning("Schema object does not have validate method, using fallback")
            except Exception as e:
                logger.warning(f"Error applying Pandera schema: {e}. Using fallback method.")
                metadata["warning"] = f"Error applying Pandera schema: {str(e)}"
        
        # For dictionary schemas, try to convert to Pandera first
        if isinstance(schema, dict):
            # Try to create a Pandera schema
            pandera_schema, schema_meta = create_schema_from_dict(schema)
            
            # If we successfully created a Pandera schema, use it
            if pandera_schema is not None and has_pandera():
                try:
                    result_df = pandera_schema.validate(df, lazy=True, inplace=False)
                    metadata["schema_type"] = "pandera_from_dict"
                    return result_df, metadata
                except Exception as e:
                    logger.warning(f"Error applying created Pandera schema: {e}. Using fallback method.")
                    metadata["warning"] = f"Error applying created Pandera schema: {str(e)}"
            
            # Fallback: Use schema_utils.apply_column_operations for type conversion
            try:
                result_df = schema_utils.apply_column_operations(df, schema)
                metadata["schema_type"] = "schema_utils"
                return result_df, metadata
            except Exception as e:
                logger.error(f"Error applying schema using schema_utils: {e}")
                return df.copy(), create_error_metadata(
                    metadata,
                    "SchemaApplicationError",
                    f"Failed to apply schema: {str(e)}"
                )
        
        # If we got here, we couldn't apply the schema
        logger.warning("Could not apply schema, returning original DataFrame")
        metadata["warning"] = "Could not apply schema"
        return df.copy(), metadata
        
    except Exception as e:
        logger.error(f"Unexpected error in apply_schema: {e}")
        return df.copy(), create_error_metadata(
            metadata,
            "UnexpectedError",
            f"Unexpected error applying schema: {str(e)}"
        )


def _format_pandera_errors(schema_errors: Any, schema_name: str) -> List[str]:
    """
    Format and enhance Pandera SchemaErrors into user-friendly messages.
    
    Args:
        schema_errors: Pandera SchemaErrors object
        schema_name: Name of the schema for context
        
    Returns:
        List of formatted error messages
    """
    try:
        error_messages = [f"DataFrame failed validation against schema '{schema_name}'"]
        
        # Group errors by failure case
        grouped_errors = {}
        
        # Format and add each error
        for _, row in schema_errors.failure_cases.iterrows():
            column = row.get('column')
            check = row.get('check')
            failure_case = row.get('failure_case', 'Unknown')
            
            # Get the index value as a string
            index = str(row.get('index', 'Unknown'))
            
            # Try to extract a more descriptive message from check description if available
            if isinstance(check, str) and "Check.Lambda" in check:
                # For lambda checks, use the custom error message if available
                check_info = "Custom check"
            else:
                check_info = str(check)
            
            # Create a key for grouping similar errors
            error_key = f"{column}:{check_info}"
            
            if error_key not in grouped_errors:
                grouped_errors[error_key] = {
                    'column': column,
                    'check': check_info,
                    'description': f"Failed check: {check_info}",
                    'indexes': set(),
                    'values': set()
                }
            
            # Add this specific instance to the group
            grouped_errors[error_key]['indexes'].add(index)
            grouped_errors[error_key]['values'].add(str(failure_case))
        
        # Format each group of errors
        for error_info in grouped_errors.values():
            column = error_info['column']
            check = error_info['check']
            
            # Format indexes and values for display
            indexes = error_info['indexes']
            if len(indexes) > 5:
                indexes_str = f"{', '.join(list(indexes)[:5])}, ... ({len(indexes)-5} more)"
            else:
                indexes_str = ', '.join(indexes)
            
            # Format unique failing values for display
            values = error_info['values']
            if len(values) > 5:
                values_str = f"{', '.join(list(values)[:5])}, ... ({len(values)-5} more distinct values)"
            else:
                values_str = ', '.join(values)
            
            # Create a detailed error message
            message = (
                f"Column '{column}' failed validation:\n"
                f"  - Check: {check}\n"
                f"  - Failing rows (indexes): {indexes_str}\n"
                f"  - Invalid values: {values_str}"
            )
            
            error_messages.append(message)
        
        # Add summary
        error_count = len(schema_errors.failure_cases)
        error_messages.append(f"Total validation errors: {error_count}")
        
        return error_messages
    except (AttributeError, KeyError, TypeError) as e:
        logger.error(f"Error formatting schema errors for '{schema_name}': {e}")
        return [f"Validation failed for schema '{schema_name}' but error details could not be processed: {e}"]
    except Exception as e:
        logger.error(f"Unexpected error formatting schema errors: {e}")
        return [f"Schema validation failed with error: {e}"]


def schema_exists(
    schema_name: str, 
    config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if a schema exists by name.
    
    Args:
        schema_name: Name of schema to check
        config: Configuration dictionary that may contain schema definitions
        
    Returns:
        Tuple of (exists, metadata) where exists is a boolean indicating if the
        schema exists, and metadata contains status information
    """
    metadata = create_metadata()
    
    # Get the schema by name
    schema, schema_metadata = get_schema_by_name(schema_name=schema_name, config=config)
    
    # Check if schema was found
    exists = schema is not None
    
    # Update metadata
    metadata.update({
        "schema_name": schema_name,
        "exists": exists
    })
    
    # Include source information if schema was found
    if exists and "schema_source" in schema_metadata:
        metadata["schema_source"] = schema_metadata["schema_source"]
    
    return exists, metadata