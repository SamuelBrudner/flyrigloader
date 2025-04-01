"""
Environment variable utilities for configuration and runtime settings.

This module provides standardized functions for working with environment variables,
including type conversion, default values, and consistent error handling.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TypeVar, Callable, cast

from loguru import logger

from .path_utils import PathLike, ensure_path

T = TypeVar('T')


def get_env_var(
    name: str, 
    default: Optional[str] = None, 
    required: bool = False
) -> Optional[str]:
    """
    Get an environment variable with consistent error handling.
    
    Args:
        name: Name of the environment variable
        default: Default value to return if not found
        required: If True, raise an error if not found
        
    Returns:
        Value of the environment variable or default
        
    Raises:
        ValueError: If required is True and the variable is not set
        
    Example:
        >>> get_env_var("HOME")  # Returns home directory
        >>> get_env_var("NONEXISTENT_VAR", default="/tmp")  # Returns "/tmp"
    """
    value = os.environ.get(name)
    
    if value is None:
        if required:
            raise ValueError(f"Required environment variable {name} is not set")
        return default
        
    return value


def get_env_var_as_type(
    name: str, 
    default: Optional[T] = None, 
    required: bool = False,
    converter: Callable[[str], T] = str,
    error_msg: Optional[str] = None
) -> Optional[T]:
    """
    Get an environment variable and convert it to a specific type.
    
    Args:
        name: Name of the environment variable
        default: Default value to return if not found
        required: If True, raise an error if not found
        converter: Function to convert the string value to the desired type
        error_msg: Custom error message for conversion failures
        
    Returns:
        Converted value of the environment variable or default
        
    Raises:
        ValueError: If required is True and the variable is not set
        ValueError: If conversion fails
        
    Example:
        >>> get_env_var_as_type("PORT", default=8080, converter=int)  # Returns port as integer
    """
    value = get_env_var(name, default=None, required=required)
    
    if value is None:
        return default
        
    try:
        return converter(value)
    except Exception as e:
        msg = error_msg or f"Failed to convert environment variable {name}={value} using {converter.__name__}"
        logger.error(f"{msg}: {str(e)}")
        raise ValueError(f"{msg}: {str(e)}") from e


def get_env_bool(
    name: str, 
    default: Optional[bool] = None,
    required: bool = False
) -> Optional[bool]:
    """
    Get an environment variable as a boolean value.
    
    Treats the following as True: "1", "true", "yes", "y", "on" (case insensitive)
    Treats the following as False: "0", "false", "no", "n", "off" (case insensitive)
    
    Args:
        name: Name of the environment variable
        default: Default value to return if not found
        required: If True, raise an error if not found
        
    Returns:
        Boolean value of the environment variable or default
        
    Raises:
        ValueError: If required is True and the variable is not set
        ValueError: If the value cannot be interpreted as a boolean
        
    Example:
        >>> get_env_bool("DEBUG", default=False)  # Returns True if DEBUG=1 or DEBUG=true
    """
    value = get_env_var(name, default=None, required=required)
    
    if value is None:
        return default
        
    value = value.lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    elif value in ("0", "false", "no", "n", "off"):
        return False
    else:
        raise ValueError(
            f"Cannot convert environment variable {name}={value} to boolean. "
            f"Expected one of: 1, true, yes, y, on, 0, false, no, n, off"
        )


def get_env_int(
    name: str, 
    default: Optional[int] = None,
    required: bool = False,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None
) -> Optional[int]:
    """
    Get an environment variable as an integer with optional range validation.
    
    Args:
        name: Name of the environment variable
        default: Default value to return if not found
        required: If True, raise an error if not found
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        
    Returns:
        Integer value of the environment variable or default
        
    Raises:
        ValueError: If required is True and the variable is not set
        ValueError: If the value cannot be converted to an integer
        ValueError: If the value is outside the specified range
        
    Example:
        >>> get_env_int("TIMEOUT", default=30, min_value=1, max_value=3600)
    """
    def convert_and_validate(value: str) -> int:
        try:
            result = int(value)
        except ValueError:
            raise ValueError(f"Cannot convert {value} to integer")
            
        if min_value is not None and result < min_value:
            raise ValueError(f"Value {result} is less than minimum {min_value}")
            
        if max_value is not None and result > max_value:
            raise ValueError(f"Value {result} is greater than maximum {max_value}")
            
        return result
        
    return get_env_var_as_type(
        name, 
        default=default,
        required=required,
        converter=convert_and_validate,
        error_msg=f"Failed to get valid integer from environment variable {name}"
    )


def get_env_float(
    name: str, 
    default: Optional[float] = None,
    required: bool = False,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> Optional[float]:
    """
    Get an environment variable as a float with optional range validation.
    
    Args:
        name: Name of the environment variable
        default: Default value to return if not found
        required: If True, raise an error if not found
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        
    Returns:
        Float value of the environment variable or default
        
    Raises:
        ValueError: If required is True and the variable is not set
        ValueError: If the value cannot be converted to a float
        ValueError: If the value is outside the specified range
        
    Example:
        >>> get_env_float("SCALING_FACTOR", default=1.0, min_value=0.1, max_value=10.0)
    """
    def convert_and_validate(value: str) -> float:
        try:
            result = float(value)
        except ValueError:
            raise ValueError(f"Cannot convert {value} to float")
            
        if min_value is not None and result < min_value:
            raise ValueError(f"Value {result} is less than minimum {min_value}")
            
        if max_value is not None and result > max_value:
            raise ValueError(f"Value {result} is greater than maximum {max_value}")
            
        return result
        
    return get_env_var_as_type(
        name, 
        default=default,
        required=required,
        converter=convert_and_validate,
        error_msg=f"Failed to get valid float from environment variable {name}"
    )


def get_env_path(
    name: str, 
    default: Optional[PathLike] = None,
    required: bool = False,
    must_exist: bool = False,
    create_if_missing: bool = False
) -> Optional[Path]:
    """
    Get an environment variable as a Path with existence validation.
    
    Args:
        name: Name of the environment variable
        default: Default path to return if not found
        required: If True, raise an error if not found
        must_exist: If True, raise an error if the path doesn't exist
        create_if_missing: If True, create the directory if it doesn't exist
        
    Returns:
        Path object from the environment variable or default
        
    Raises:
        ValueError: If required is True and the variable is not set
        FileNotFoundError: If must_exist is True and the path doesn't exist
        
    Example:
        >>> get_env_path("CONFIG_DIR", default=Path("config"), create_if_missing=True)
    """
    value = get_env_var(name, default=None, required=required)
    
    if value is None:
        if default is None:
            return None
        path = ensure_path(default)
    else:
        path = ensure_path(value)
    
    if create_if_missing and not path.exists():
        try:
            # If it has an extension, create parent directory
            if path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory structure for {path}")
            else:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory {path}")
        except Exception as e:
            logger.error(f"Failed to create directory for {path}: {str(e)}")
            if must_exist:
                raise
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path from environment variable {name}={path} does not exist")
        
    return path


def get_env_list(
    name: str,
    default: Optional[List[str]] = None,
    required: bool = False,
    separator: str = ","
) -> Optional[List[str]]:
    """
    Get an environment variable as a list of strings.
    
    Args:
        name: Name of the environment variable
        default: Default list to return if not found
        required: If True, raise an error if not found
        separator: String separator to split the value
        
    Returns:
        List of strings from the environment variable or default
        
    Raises:
        ValueError: If required is True and the variable is not set
        
    Example:
        >>> get_env_list("ALLOWED_HOSTS", default=["localhost"])
        >>> # If ALLOWED_HOSTS="localhost,127.0.0.1" returns ["localhost", "127.0.0.1"]
    """
    value = get_env_var(name, default=None, required=required)
    
    if value is None:
        return default
        
    # Split, strip, and filter empty strings
    return [item.strip() for item in value.split(separator) if item.strip()]


def expand_env_vars(
    text: str
) -> str:
    """
    Expand environment variables in a string.
    
    Supports both ${VAR} and $VAR formats.
    
    Args:
        text: String containing environment variables to expand
        
    Returns:
        String with environment variables expanded
        
    Example:
        >>> expand_env_vars("Config file at ${HOME}/config.json")
        "Config file at /home/user/config.json"
    """
    # First, handle ${VAR} format
    pattern = r"\${([a-zA-Z0-9_]+)}"
    
    def replace_var(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))
        
    result = re.sub(pattern, replace_var, text)
    
    # Then, handle $VAR format
    pattern = r"\$([a-zA-Z0-9_]+)"
    result = re.sub(pattern, replace_var, result)
    
    return result
