"""
validation.py - Common validation logic for lineage tracking.

This module provides utilities for validating lineage data, ensuring
that the tracking system operates with consistent and valid input.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union, TypeVar, cast

import pandas as pd
from loguru import logger

# Type variables for better type hints
T = TypeVar('T')


def validate_dataframe(df: Any) -> bool:
    """
    Verify that an object is a valid pandas DataFrame.
    
    Args:
        df: Object to validate
        
    Returns:
        True if the object is a valid DataFrame, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        logger.warning(f"Expected pandas DataFrame, got {type(df)}")
        return False
    
    # Additional validation could be added here in the future
    # For example, checking for minimum columns or valid dtypes
    
    return True


def ensure_dataframe(df: Any) -> pd.DataFrame:
    """
    Ensure an object is a valid pandas DataFrame.
    
    Args:
        df: Object to validate and ensure
        
    Returns:
        The validated DataFrame
        
    Raises:
        ValueError: If the object is not a valid DataFrame
    """
    if not validate_dataframe(df):
        raise ValueError(f"Expected pandas DataFrame, got {type(df)}")
    
    return df


def validate_lineage_storage_params(storage_type: str, **kwargs) -> bool:
    """
    Validate parameters for creating a lineage storage backend.
    
    Args:
        storage_type: Type of storage to validate
        **kwargs: Additional parameters specific to the storage type
        
    Returns:
        True if parameters are valid, False otherwise
    """
    valid_storage_types = {"attribute", "registry", "null"}
    
    if storage_type not in valid_storage_types:
        logger.warning(f"Invalid storage type: {storage_type}. Valid options: {', '.join(valid_storage_types)}")
        return False
    
    # Additional validation could be added here in the future
    # based on the specific storage type
    
    return True


def validate_lineage_complexity(complexity: str) -> bool:
    """
    Validate the lineage complexity level.
    
    Args:
        complexity: Complexity level to validate
        
    Returns:
        True if the complexity level is valid, False otherwise
    """
    valid_complexity_levels = {"minimal", "standard", "full"}
    
    if complexity not in valid_complexity_levels:
        logger.warning(f"Invalid complexity level: {complexity}. Valid options: {', '.join(valid_complexity_levels)}")
        return False
    
    return True


def verify_callable(func: Any, name: str = "callback") -> Callable:
    """
    Verify that an object is callable.
    
    Args:
        func: Object to verify
        name: Name to use in error messages
        
    Returns:
        The validated callable
        
    Raises:
        ValueError: If the object is not callable
    """
    if not callable(func):
        raise ValueError(f"Expected {name} to be callable, got {type(func)}")
    
    return func


def verify_instance(obj: Any, cls: Type[T], name: str = "object") -> T:
    """
    Verify that an object is an instance of a specific class.
    
    Args:
        obj: Object to verify
        cls: Class to check against
        name: Name to use in error messages
        
    Returns:
        The validated instance, cast to the correct type
        
    Raises:
        ValueError: If the object is not an instance of the specified class
    """
    if not isinstance(obj, cls):
        raise ValueError(f"Expected {name} to be {cls.__name__}, got {type(obj).__name__}")
    
    return cast(T, obj)


def _get_signature(func: Callable) -> inspect.Signature:
    """
    Get the signature of a callable.
    
    Args:
        func: Callable to get signature for
        
    Returns:
        Signature object
        
    Raises:
        ValueError: If signature cannot be determined
    """
    try:
        return inspect.signature(func)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Cannot determine signature for {func}: {str(e)}")


def verify_callable_signature(
    func: Callable, 
    expected_args: Optional[List[str]] = None,
    positional_only: bool = False
) -> Callable:
    """
    Verify that a callable has the expected argument signature.
    
    Args:
        func: Callable to verify
        expected_args: List of expected argument names
        positional_only: If True, only verify positional arguments
        
    Returns:
        The validated callable
        
    Raises:
        ValueError: If the callable doesn't have the expected signature
    """
    verify_callable(func)
    
    if expected_args is not None:
        sig = _get_signature(func)
        actual_args = list(sig.parameters.keys())
        
        if positional_only:
            actual_args = [arg for arg in actual_args 
                          if sig.parameters[arg].kind in (
                              inspect.Parameter.POSITIONAL_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD
                          )]
        
        missing_args = [arg for arg in expected_args if arg not in actual_args]
        
        if missing_args:
            missing_str = ", ".join(missing_args)
            raise ValueError(f"Callable {func.__name__} is missing required arguments: {missing_str}")
    
    return func
