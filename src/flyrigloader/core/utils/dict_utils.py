"""
Dictionary utilities for common operations on dictionaries.

This module provides standardized functions for dictionary operations
that are used throughout the codebase, with an emphasis on immutability
and predictable behavior.
"""

import copy
from typing import Dict, Any, List, Optional, Union, TypeVar, Generic, Mapping

from loguru import logger

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


def deep_update(target: Dict[K, V], source: Dict[K, V], copy_values: bool = True, copy_target: bool = True) -> Dict[K, V]:
    """
    Deep update a nested dictionary with proper immutability.
    
    Args:
        target: Target dictionary to update
        source: Source dictionary with updates
        copy_values: If True, use deepcopy on values to avoid mutable references
        copy_target: If True, create a copy of target before updating (preserves original)
            
    Returns:
        Updated target dictionary (new instance if copy_target=True, otherwise modified original)
        
    Example:
        >>> target = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> source = {"b": {"c": 4, "e": 5}, "f": 6}
        >>> result = deep_update(target, source)
        >>> result
        {'a': 1, 'b': {'c': 4, 'd': 3, 'e': 5}, 'f': 6}
        >>> target is result  # target and result are not the same object
        False
    """
    # Create a copy of the target if requested
    result = copy.deepcopy(target) if copy_target else target
    
    for key, value in source.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            # Recursively update nested dictionaries, preserving copy_target setting
            # The nested call should respect the same immutability settings
            result[key] = deep_update(result[key], value, copy_values, copy_target=False)
        else:
            # Replace or add value
            result[key] = copy.deepcopy(value) if copy_values else value
    
    return result


def deep_merge(*dicts: Dict[K, V], copy_result: bool = True) -> Dict[K, V]:
    """
    Merge multiple dictionaries with deep updating.
    
    This combines multiple dictionaries, with later dictionaries
    taking precedence over earlier ones for conflicting keys.
    
    Args:
        *dicts: Dictionaries to merge
        copy_result: If True, return a new dictionary
            
    Returns:
        New merged dictionary
        
    Example:
        >>> dict1 = {"a": 1, "b": {"c": 2}}
        >>> dict2 = {"b": {"d": 3}}
        >>> dict3 = {"b": {"c": 4}}
        >>> deep_merge(dict1, dict2, dict3)
        {'a': 1, 'b': {'c': 4, 'd': 3}}
    """
    if not dicts:
        return {}
        
    if len(dicts) == 1:
        return copy.deepcopy(dicts[0]) if copy_result else dicts[0]
        
    result = copy.deepcopy(dicts[0]) if copy_result else dicts[0]
    for d in dicts[1:]:
        if d:  # Skip empty dictionaries
            deep_update(result, d, copy_values=copy_result)
            
    return result


def get_nested_value(
    data: Dict[str, Any], 
    path: Union[str, List[str]], 
    default: Any = None, 
    path_separator: str = "."
) -> Any:
    """
    Get a nested value from a dictionary using a path string or list.
    
    Args:
        data: Dictionary to get value from
        path: Path to the value as a dot-separated string or list of keys
        default: Default value to return if path is not found
        path_separator: Separator to use if path is a string
            
    Returns:
        Value at the specified path, or default if not found
        
    Example:
        >>> data = {"a": {"b": {"c": 42}}}
        >>> get_nested_value(data, "a.b.c")
        42
        >>> get_nested_value(data, ["a", "b", "c"])
        42
        >>> get_nested_value(data, "a.b.d", default="not found")
        'not found'
    """
    if not data:
        return default
        
    # Convert string path to list of parts
    parts = path.split(path_separator) if isinstance(path, str) else path
        
    # Navigate through the nested dictionary
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
            
    return current


def set_nested_value(
    data: Dict[str, Any], 
    path: Union[str, List[str]], 
    value: Any, 
    create_missing: bool = True,
    path_separator: str = "."
) -> Dict[str, Any]:
    """
    Set a nested value in a dictionary using a path string or list.
    
    Args:
        data: Dictionary to modify
        path: Path to the value as a dot-separated string or list of keys
        value: Value to set
        create_missing: If True, create missing dictionary levels
        path_separator: Separator to use if path is a string
            
    Returns:
        Modified dictionary
        
    Example:
        >>> data = {"a": {"b": {}}}
        >>> set_nested_value(data, "a.b.c", 42)
        {'a': {'b': {'c': 42}}}
        >>> set_nested_value(data, ["a", "x", "y"], "new", create_missing=True)
        {'a': {'b': {'c': 42}, 'x': {'y': 'new'}}}
    """
    if not data:
        data = {}
        
    # Convert string path to list of parts
    parts = path.split(path_separator) if isinstance(path, str) else path
        
    if not parts:
        return data
        
    # Navigate to the parent of the target
    current = data
    for part in parts[:-1]:
        if part not in current:
            if create_missing:
                current[part] = {}
            else:
                raise ValueError(f"Cannot set nested value: parent path {'.'.join(parts[:parts.index(part)+1])} does not exist")
        current = current[part]
            
    # Set the value at the target location
    current[parts[-1]] = value
    return data


def filter_dict(
    data: Dict[str, Any],
    include_keys: Optional[List[Union[str, List[str]]]] = None,
    exclude_keys: Optional[List[Union[str, List[str]]]] = None,
    path_separator: str = "."
) -> Dict[str, Any]:
    """
    Filter a dictionary by including or excluding keys.
    
    Args:
        data: Dictionary to filter
        include_keys: List of keys or paths to include (if None, include all)
        exclude_keys: List of keys or paths to exclude
        path_separator: Separator for path strings
            
    Returns:
        Filtered dictionary
        
    Example:
        >>> data = {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}
        >>> filter_dict(data, include_keys=["a", "b.c"])
        {'a': 1, 'b': {'c': 2}}
        >>> filter_dict(data, exclude_keys=["b.d", "e"])
        {'a': 1, 'b': {'c': 2}}
    """
    if not data:
        return {}
        
    # Initialize result with a copy of the input
    result = copy.deepcopy(data)
    
    # Process exclude keys first
    if exclude_keys:
        for key_path in exclude_keys:
            # Convert string to list if needed
            parts = key_path.split(path_separator) if isinstance(key_path, str) else key_path
            
            # Skip empty paths
            if not parts:
                continue
                
            # Handle root level keys
            if len(parts) == 1:
                if parts[0] in result:
                    del result[parts[0]]
                continue
                
            # Navigate to parent of target
            current = result
            found = True
            
            for part in parts[:-1]:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    found = False
                    break
                    
            # Remove the key if found
            if found and isinstance(current, dict) and parts[-1] in current:
                del current[parts[-1]]
    
    # If include_keys is None, we're done (keep everything not excluded)
    if include_keys is None:
        return result
        
    # Process include keys
    if include_keys:
        filtered = {}
        
        for key_path in include_keys:
            # Convert string to list if needed
            parts = key_path.split(path_separator) if isinstance(key_path, str) else key_path
            
            # Skip empty paths
            if not parts:
                continue
                
            # Handle root level keys
            if len(parts) == 1:
                if parts[0] in result:
                    if parts[0] not in filtered:
                        filtered[parts[0]] = {}
                    filtered[parts[0]] = result[parts[0]]
                continue
                
            # Navigate to target
            current = result
            target = filtered
            found = True
            
            # Create path in filtered result and copy nested value
            for i, part in enumerate(parts[:-1]):
                if isinstance(current, dict) and part in current:
                    current = current[part]
                    
                    # Create path in filtered result
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                else:
                    found = False
                    break
                    
            # Set the value if found
            if found and isinstance(current, dict) and parts[-1] in current:
                target[parts[-1]] = current[parts[-1]]
        
        return filtered
        
    return result


def ensure_dict(data: Any) -> Dict[str, Any]:
    """
    Ensure that a value is a dictionary.
    
    Args:
        data: Value to convert to dictionary
            
    Returns:
        Dictionary version of data, or empty dict if conversion not possible
        
    Raises:
        ValueError: If data cannot be converted to a dictionary
    """
    try:
        if data is None:
            return {}
            
        if isinstance(data, dict):
            return data
            
        if isinstance(data, str):
            # Strings are technically iterable but not what we want
            raise ValueError(f"Cannot convert string to dictionary")
            
        if hasattr(data, 'keys') and callable(getattr(data, 'keys')):
            # If it has a keys method, it's dict-like
            return dict(data)
            
        if hasattr(data, '__iter__') and not isinstance(data, str):
            # If it's iterable, try to convert to dict
            try:
                return dict(data)
            except (ValueError, TypeError):
                raise ValueError(f"Cannot convert iterable to dictionary")
                
        raise ValueError(f"Cannot convert {type(data).__name__} to dictionary")
    except Exception as e:
        logger.error(f"Failed to ensure dictionary: {str(e)}")
        raise ValueError(f"Failed to ensure dictionary: {str(e)}") from e


def safe_dict_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely get a value from a dictionary with robust error handling.
    
    Args:
        data: Dictionary to get value from
        key: Key to lookup
        default: Default value to return if key not found or data is not a dict
            
    Returns:
        Value for the key, or default if not found
    """
    try:
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict but got {type(data).__name__} in safe_dict_get")
            
        return data.get(key, default)
    except Exception as e:
        logger.error(f"Error in safe_dict_get: {str(e)}")
        raise ValueError(f"Error in safe_dict_get: {str(e)}") from e
