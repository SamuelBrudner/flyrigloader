"""
Value utilities for normalizing and standardizing data values.

This module provides standardized functions for coercing, normalizing and
transforming values to ensure consistent data types and formats across
the codebase, which is particularly useful for configuration management.
"""

from typing import Any, Dict, List, Optional, Union, TypeVar, Callable, cast
import math
from enum import Enum, auto
from datetime import datetime, date
from contextlib import suppress

from loguru import logger


class ValueType(Enum):
    """Enum defining the standard value types for coercion."""
    NUMERIC = auto()
    BOOLEAN = auto()
    FLOAT = auto()
    STRING = auto()
    DATE = auto()
    LIST = auto()


# Common value patterns for reuse
COMMON_VALUES = {
    'boolean_true': [True, 'true', 'True', 'TRUE', '1', 1, 'yes', 'Yes', 'YES', 'y', 'Y', 'on', 'On', 'ON'],
    'boolean_false': [False, 'false', 'False', 'FALSE', '0', 0, 'none', 'None', 'NONE', None, 'no', 'No', 'NO', 'n', 'N', 'off', 'Off', 'OFF'],
    'null_values': ['na', 'NA', 'n/a', 'N/A', 'unknown', 'Unknown', 'UNKNOWN', 'null', 'Null', 'NULL', None, 'none', 'None', 'NONE'],
}


def coerce_value(
    value: Any,
    target_type: Union[ValueType, str],
    **options
) -> Any:
    """
    Coerce a value to a specific type with standardized rules.
    
    Args:
        value: Value to coerce
        target_type: Target type (ValueType enum or string name of the enum)
        **options: Additional options for coercion:
            - default: Default value if coercion fails
            - min_value: Minimum value for numeric types
            - max_value: Maximum value for numeric types
            - true_values: List of values to consider as True for boolean types
            - false_values: List of values to consider as False for boolean types
            - null_values: List of values to consider as null/nan for numeric types
            - format: Format string for date/time types
            - mapping: Dictionary mapping of source values to target values
            
    Returns:
        Coerced value of the specified type
        
    Example:
        >>> coerce_value("1.5", ValueType.FLOAT)
        1.5
        >>> coerce_value("yes", ValueType.BOOLEAN)
        True
        >>> coerce_value("unknown", ValueType.FLOAT, default=0.0)
        0.0
    """
    # Handle string target_type by converting to enum
    if isinstance(target_type, str):
        try:
            target_type = ValueType[target_type.upper()]
        except KeyError:
            supported_types = [t.name for t in ValueType]
            logger.error(f"Invalid target_type: {target_type}. Supported types: {supported_types}")
            return options.get('default', value)
    
    # Dispatch to the appropriate handler
    try:
        if target_type == ValueType.NUMERIC:
            return coerce_to_numeric(value, **options)
        elif target_type == ValueType.BOOLEAN:
            return coerce_to_boolean(value, **options)
        elif target_type == ValueType.FLOAT:
            return coerce_to_float(value, **options)
        elif target_type == ValueType.STRING:
            return coerce_to_string(value, **options)
        elif target_type == ValueType.DATE:
            return coerce_to_date(value, **options)
        elif target_type == ValueType.LIST:
            return coerce_to_list(value, **options)
        else:
            logger.warning(f"Unknown target_type: {target_type}")
            return options.get('default', value)
    except Exception as e:
        logger.error(f"Error coercing value '{value}' to {target_type}: {str(e)}")
        return options.get('default', value)


def coerce_to_numeric(
    value: Any,
    force_type: Optional[type] = None,
    default: Any = 0,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    zero_values: Optional[List[Any]] = None,
    one_values: Optional[List[Any]] = None,
    **kwargs
) -> Union[int, float]:
    """
    Coerce a value to a numeric type (int or float).
    
    Args:
        value: Value to coerce
        force_type: Force the output to be this type (int or float)
        default: Default value if coercion fails
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        zero_values: List of values to consider as 0
        one_values: List of values to consider as 1
        **kwargs: Additional options (ignored)
        
    Returns:
        Numeric value (int or float)
    """
    # Handle values that should be treated as 0 or 1
    if zero_values and value in zero_values:
        result = 0
    elif one_values and value in one_values:
        result = 1
    elif isinstance(value, str):
        # Try to convert numeric string
        if value.replace('.', '', 1).isdigit():
            result = float(value) if '.' in value else int(value)
        else:
            return default
    elif isinstance(value, (int, float)):
        result = value
    else:
        try:
            # Attempt to convert to the target numeric type
            result = int(value) if isinstance(value, bool) else float(value)
        except (ValueError, TypeError):
            return default
    
    # Apply min/max constraints if specified
    if min_value is not None and result < min_value:
        result = min_value
    if max_value is not None and result > max_value:
        result = max_value
    
    # Force the type if specified
    if force_type == int:
        return int(result)
    elif force_type == float:
        return float(result)
    
    # Otherwise return the natural type
    return result


def coerce_to_boolean(
    value: Any,
    true_values: Optional[List[Any]] = None,
    false_values: Optional[List[Any]] = None,
    default: bool = False,
    **kwargs
) -> bool:
    """
    Coerce a value to a boolean type.
    
    Args:
        value: Value to coerce
        true_values: List of values to consider as True
        false_values: List of values to consider as False
        default: Default value if coercion fails
        **kwargs: Additional options (ignored)
        
    Returns:
        Boolean value
    """
    # Use default lists if not provided
    if true_values is None:
        true_values = COMMON_VALUES['boolean_true']
    if false_values is None:
        false_values = COMMON_VALUES['boolean_false']
    
    # Check if value matches true or false values
    if value in true_values:
        return True
    elif value in false_values:
        return False
    
    # Handle unexpected values
    logger.warning(
        f"Unexpected value '{value}' for boolean conversion. "
        f"Expected one of: {true_values + false_values}. "
        f"Using default: {default}"
    )
    return default


def coerce_to_float(
    value: Any,
    default: float = 0.0,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    nan_values: Optional[List[Any]] = None,
    **kwargs
) -> float:
    """
    Coerce a value to a float type.
    
    Args:
        value: Value to coerce
        default: Default value if coercion fails
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        nan_values: List of values to interpret as NaN
        **kwargs: Additional options (ignored)
        
    Returns:
        Float value
    """
    # Use default list if not provided
    if nan_values is None:
        nan_values = COMMON_VALUES['null_values']
    
    # Handle values that should be treated as NaN
    if value in nan_values:
        return float('nan')
    
    try:
        # Try to convert to float
        result = float(value)
        
        # Apply min/max constraints if specified
        if min_value is not None and result < min_value:
            result = min_value
        if max_value is not None and result > max_value:
            result = max_value
            
        return result
    except (ValueError, TypeError):
        return default


def coerce_to_string(
    value: Any,
    default: str = "",
    mapping: Optional[Dict[str, str]] = None,
    strip: bool = True,
    lower: bool = False,
    upper: bool = False,
    **kwargs
) -> str:
    """
    Coerce a value to a string type with optional transformations.
    
    Args:
        value: Value to coerce
        default: Default value if coercion fails
        mapping: Dictionary mapping input values to output values
        strip: Whether to strip whitespace
        lower: Whether to convert to lowercase
        upper: Whether to convert to uppercase
        **kwargs: Additional options (ignored)
        
    Returns:
        String value
    """
    if value is None:
        return default
    
    with suppress(Exception):
        # Convert to string
        result = str(value)
        
        # Apply transformations
        if strip:
            result = result.strip()
        if lower:
            result = result.lower()
        if upper:
            result = result.upper()
        
        # Apply mapping if provided
        if mapping and (mapped_value := mapping.get(result)):
            result = mapped_value
            
        return result
    
    return default


def coerce_to_date(
    value: Any,
    default: Optional[date] = None,
    format_str: str = "%Y-%m-%d",
    **kwargs
) -> Optional[date]:
    """
    Coerce a value to a date type.
    
    Args:
        value: Value to coerce
        default: Default value if coercion fails
        format_str: Format string for parsing date strings
        **kwargs: Additional options (ignored)
        
    Returns:
        Date value or default
    """
    if value is None:
        return default
    
    # Handle datetime objects
    if isinstance(value, datetime):
        return value.date()
    
    # Handle date objects
    if isinstance(value, date):
        return value
    
    # Handle string values
    if isinstance(value, str):
        try:
            dt = datetime.strptime(value, format_str)
            return dt.date()
        except ValueError:
            pass
    
    # Return default if conversion fails
    return default


def coerce_to_list(
    value: Any,
    default: Optional[List[Any]] = None,
    item_type: Optional[Union[ValueType, str]] = None,
    delimiter: str = ",",
    **kwargs
) -> List[Any]:
    """
    Coerce a value to a list type.
    
    Args:
        value: Value to coerce
        default: Default value if coercion fails
        item_type: Type to coerce each item to
        delimiter: Delimiter for splitting strings
        **kwargs: Additional options passed to the item coercion
        
    Returns:
        List of values
    """
    if default is None:
        default = []
    
    if value is None:
        return default
    
    with suppress(Exception):
        # Handle different input types
        if isinstance(value, list):
            result = value
        elif isinstance(value, str):
            result = [item.strip() for item in value.split(delimiter) if item.strip()]
        elif isinstance(value, (set, tuple)):
            result = list(value)
        else:
            # Single item becomes a list with one element
            result = [value]
        
        # Coerce individual items if needed
        if item_type:
            result = [coerce_value(item, item_type, **kwargs) for item in result]
            
        return result
        
    return default


def coerce_dict_values(
    data: Dict[str, Any],
    rules: Dict[str, Dict[str, Any]],
    recursive: bool = True
) -> Dict[str, Any]:
    """
    Standardize values in a dictionary according to specified rules.
    
    This function recursively walks through a dictionary and applies coercion
    rules to values for specific keys.
    
    Args:
        data: Dictionary to update
        rules: Dictionary mapping keys to coercion rules
        recursive: Whether to recursively process nested dictionaries
        
    Returns:
        Updated dictionary with standardized values
        
    Example:
        >>> rules = {
        ...     'temperature': {'type': 'FLOAT', 'min_value': 0, 'max_value': 100},
        ...     'enabled': {'type': 'BOOLEAN'}
        ... }
        >>> data = {'temperature': '75.5', 'enabled': 'yes'}
        >>> coerce_dict_values(data, rules)
        {'temperature': 75.5, 'enabled': True}
    """
    result = data.copy()
    
    # Process each key in the dictionary
    for key, value in list(result.items()):
        # Handle nested dictionaries recursively
        if recursive and isinstance(value, dict):
            result[key] = coerce_dict_values(value, rules, recursive)
            continue
            
        # Apply coercion rule if one exists for this key
        if key in rules and (rule := rules[key].copy()):
            target_type = rule.pop('type', None)
            if target_type:
                result[key] = coerce_value(value, target_type, **rule)
    
    return result
