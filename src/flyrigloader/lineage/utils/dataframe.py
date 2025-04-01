"""
dataframe.py - DataFrame-specific utility functions for lineage tracking.

This module provides helper functions for working with DataFrames in the
context of lineage tracking, including functions to safely access and modify
lineage attributes in DataFrames.
"""

from typing import Any, Dict, List, Optional, Set, Union, cast

import pandas as pd
from loguru import logger

from ..core import (
    ATTR_MAIN_LINEAGE,
    ATTR_LINEAGE_IDS,
    ATTR_LINEAGES_DICT,
    ATTR_MINIMAL_FLAG,
    DF_LINEAGE_ID_ATTR,
    ensure_dataframe
)


def get_lineage_attribute(
    df: pd.DataFrame, 
    attr_name: str, 
    default: Any = None
) -> Any:
    """
    Safely retrieve a lineage-related attribute from a DataFrame.
    
    This function uses defensive programming to prevent KeyError exceptions
    when accessing potentially missing attributes.
    
    Args:
        df: DataFrame to get attribute from
        attr_name: Name of the attribute to retrieve
        default: Default value to return if attribute is missing
        
    Returns:
        Attribute value if found, default otherwise
    
    Raises:
        TypeError: If df is not a DataFrame
        ValueError: If the DataFrame validation fails
        KeyError: If attribute exists but has an invalid structure
        AttributeError: If pandas attributes cannot be accessed
        
    Example:
        >>> lineage = get_lineage_attribute(df, ATTR_MAIN_LINEAGE)
        >>> if lineage is not None:
        >>>     print(f"Found lineage: {lineage.name}")
    """
    # First validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
        
    if not isinstance(attr_name, str):
        raise TypeError(f"Expected string for attr_name, got {type(attr_name).__name__}")
    
    # Handle missing attributes dictionary
    try:
        # Ensure we have a valid DataFrame
        df = ensure_dataframe(df)
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid DataFrame when accessing attribute {attr_name}: {str(e)}")
        raise ValueError(f"Cannot access DataFrame attributes: {str(e)}") from e
    
    # If DataFrame doesn't have attributes or they're empty, return default
    if not hasattr(df, 'attrs') or not df.attrs:
        logger.debug(f"DataFrame has no attrs dictionary or empty attrs, returning default for {attr_name}")
        return default
    
    # Check if specific attribute exists and return it if found
    try:
        return df.attrs.get(attr_name, default)
    except KeyError as e:
        logger.error(f"Key error accessing attribute {attr_name}: {str(e)}")
        raise KeyError(f"Cannot access attribute {attr_name}: {str(e)}") from e
    except AttributeError as e:
        logger.error(f"Attribute error accessing DataFrame attrs: {str(e)}")
        raise AttributeError(f"DataFrame attributes not accessible: {str(e)}") from e
    except Exception as e:
        # For unexpected errors, provide clear context
        logger.error(f"Unexpected error accessing attribute {attr_name}: {str(e)}")
        raise RuntimeError(f"Failed to access lineage attribute: {str(e)}") from e


def set_lineage_attribute(
    df: pd.DataFrame, 
    attr_name: str, 
    value: Any
) -> pd.DataFrame:
    """
    Safely set a lineage-related attribute on a DataFrame.
    
    Args:
        df: DataFrame to set attribute on
        attr_name: Name of the attribute to set
        value: Value to set for the attribute
        
    Returns:
        DataFrame with attribute set
        
    Raises:
        TypeError: If df is not a DataFrame or attr_name is not a string
        ValueError: If DataFrame validation fails
        AttributeError: If pandas attributes cannot be accessed
        
    Example:
        >>> df = set_lineage_attribute(df, ATTR_MAIN_LINEAGE, lineage_record)
    """
    # Validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
        
    if not isinstance(attr_name, str):
        raise TypeError(f"Expected string for attr_name, got {type(attr_name).__name__}")
    
    # First ensure we have a valid DataFrame
    try:
        df = ensure_dataframe(df)
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid DataFrame when setting attribute {attr_name}: {str(e)}")
        raise ValueError(f"Cannot set DataFrame attribute: {str(e)}") from e
    
    # Set the attribute safely
    try:
        # Ensure attributes dictionary exists
        if not hasattr(df, 'attrs'):
            logger.debug(f"Creating attrs dictionary for DataFrame to set {attr_name}")
            df.attrs = {}
        
        # Set attribute value
        df.attrs[attr_name] = value
        return df
    except AttributeError as e:
        logger.error(f"Attribute error setting {attr_name}: {str(e)}")
        raise AttributeError(f"Cannot set DataFrame attribute: {str(e)}") from e
    except TypeError as e:
        logger.error(f"Type error setting attribute {attr_name}: {str(e)}")
        raise TypeError(f"Value has incompatible type for pandas attrs: {str(e)}") from e
    except Exception as e:
        # For unexpected errors, provide clear context
        logger.error(f"Unexpected error setting attribute {attr_name}: {str(e)}")
        raise RuntimeError(f"Failed to set lineage attribute: {str(e)}") from e


def has_lineage(df: pd.DataFrame) -> bool:
    """
    Check if a DataFrame has lineage information.
    
    Args:
        df: DataFrame to check
        
    Returns:
        True if DataFrame has lineage, False otherwise
        
    Raises:
        TypeError: If df is not a DataFrame
        ValueError: If DataFrame validation fails
        
    Example:
        >>> if has_lineage(df):
        >>>     print("This DataFrame has lineage information")
    """
    # First validate input parameter
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
    
    try:
        # Ensure we have a valid DataFrame
        df = ensure_dataframe(df)
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid DataFrame when checking for lineage: {str(e)}")
        raise ValueError(f"Cannot check lineage: {str(e)}") from e
    
    try:
        # Check for main lineage attribute
        lineage_attr = get_lineage_attribute(df, ATTR_MAIN_LINEAGE, None)
        return lineage_attr is not None
    except (KeyError, AttributeError) as e:
        logger.debug(f"Error checking for lineage (likely no lineage): {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking for lineage: {str(e)}")
        # Don't raise here, as has_lineage is meant to be a safe check
        return False


def is_minimal_lineage(df: pd.DataFrame) -> bool:
    """
    Check if a DataFrame has minimal lineage tracking.
    
    Args:
        df: DataFrame to check
        
    Returns:
        True if DataFrame has minimal lineage, False otherwise
        
    Raises:
        TypeError: If df is not a DataFrame
        ValueError: If DataFrame validation fails
        
    Example:
        >>> if is_minimal_lineage(df):
        >>>     print("This DataFrame uses minimal lineage tracking")
    """
    # First validate input parameter
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
    
    try:
        # Ensure we have a valid DataFrame
        df = ensure_dataframe(df)
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid DataFrame when checking for minimal lineage: {str(e)}")
        raise ValueError(f"Cannot check minimal lineage: {str(e)}") from e
    
    try:
        # First check if the DataFrame has any lineage at all
        if not has_lineage(df):
            return False
            
        # Check for explicit minimal flag
        if get_lineage_attribute(df, ATTR_MINIMAL_FLAG, False):
            return True
            
        # Check lineage record metadata for minimal flag
        lineage = get_lineage_attribute(df, ATTR_MAIN_LINEAGE, None)
        return (lineage is not None and 
                hasattr(lineage, 'metadata') and 
                lineage.metadata.get('minimal', False))
    except (KeyError, AttributeError) as e:
        logger.debug(f"Error checking for minimal lineage (likely not minimal): {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking for minimal lineage: {str(e)}")
        return False


def extract_lineage_dict(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Extract lineage information from a DataFrame as a dictionary.
    
    Args:
        df: DataFrame to extract lineage from
        
    Returns:
        Dictionary with lineage information, or None if no lineage found
        
    Raises:
        TypeError: If df is not a DataFrame
        ValueError: If DataFrame validation fails
        
    Example:
        >>> lineage_dict = extract_lineage_dict(df)
        >>> if lineage_dict:
        >>>     print(f"Found lineage with ID: {lineage_dict.get('id')}")
    """
    # First validate input parameter
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
    
    try:
        # Ensure we have a valid DataFrame
        df = ensure_dataframe(df)
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid DataFrame when extracting lineage: {str(e)}")
        raise ValueError(f"Cannot extract lineage: {str(e)}") from e
    
    # First check if the DataFrame has any lineage
    if not has_lineage(df):
        logger.debug("No lineage found in DataFrame")
        return None
    
    try:
        return _extract_lineage_from_attributes(df)
    except (KeyError, AttributeError, TypeError) as e:
        logger.warning(f"Error parsing lineage data from DataFrame: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error extracting lineage dictionary: {str(e)}")
        return None


def _extract_lineage_from_attributes(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Helper function to extract lineage information from DataFrame attributes.
    
    Args:
        df: DataFrame to extract lineage from
        
    Returns:
        Dictionary with lineage information, or None if no valid lineage found
        
    Raises:
        KeyError, AttributeError, TypeError: If extraction fails due to missing or invalid data
    """
    # Try to get lineage from main attribute first
    lineage_obj = get_lineage_attribute(df, ATTR_MAIN_LINEAGE, None)
    if lineage_obj is not None:
        # Convert lineage object to dictionary
        if hasattr(lineage_obj, 'to_dict') and callable(lineage_obj.to_dict):
            return lineage_obj.to_dict()
        if hasattr(lineage_obj, '__dict__'):
            return lineage_obj.__dict__
        if isinstance(lineage_obj, dict):
            return lineage_obj
    
    # Try lineages dictionary next
    lineages_dict = get_lineage_attribute(df, ATTR_LINEAGES_DICT, None)
    if isinstance(lineages_dict, dict) and lineages_dict:
        # If multiple lineages exist, use the first one
        first_key = next(iter(lineages_dict))
        return lineages_dict[first_key]
    
    # As a last resort, check for lineage IDs
    lineage_ids = get_lineage_attribute(df, ATTR_LINEAGE_IDS, None)
    if isinstance(lineage_ids, list) and lineage_ids:
        # Return a minimal dictionary with just the ID
        return {"id": lineage_ids[0], "minimal": True}
    
    # No valid lineage information found
    logger.debug("DataFrame has lineage attributes, but no valid lineage data could be extracted")
    return None


def initialize_lineage_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Initialize lineage attribute structures in a DataFrame.
    
    Args:
        df: DataFrame to initialize
        
    Returns:
        DataFrame with initialized attributes
        
    Raises:
        TypeError: If df is not a DataFrame
        ValueError: If DataFrame validation fails
        AttributeError: If pandas attributes cannot be accessed
        RuntimeError: If initialization fails for other reasons
        
    Example:
        >>> df = initialize_lineage_attributes(df)
    """
    # First validate input parameter
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
        
    try:
        # Ensure we have a valid DataFrame
        df = ensure_dataframe(df)
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid DataFrame when initializing lineage attributes: {str(e)}")
        raise ValueError(f"Cannot initialize lineage attributes: {str(e)}") from e
    
    try:
        # Ensure attributes dictionary exists
        if not hasattr(df, 'attrs'):
            df.attrs = {}
        
        # Initialize lineage attributes if not already present
        if ATTR_LINEAGE_IDS not in df.attrs:
            df.attrs[ATTR_LINEAGE_IDS] = []
            
        if ATTR_LINEAGES_DICT not in df.attrs:
            df.attrs[ATTR_LINEAGES_DICT] = {}
        
        return df
    except AttributeError as e:
        logger.error(f"Attribute error initializing lineage attributes: {str(e)}")
        raise AttributeError(f"Cannot set DataFrame attributes: {str(e)}") from e
    except TypeError as e:
        logger.error(f"Type error initializing lineage attributes: {str(e)}")
        raise TypeError(f"Invalid attribute type: {str(e)}") from e
    except Exception as e:
        # For unexpected errors, provide clear context
        logger.error(f"Unexpected error initializing lineage attributes: {str(e)}")
        raise RuntimeError(f"Failed to initialize lineage attributes: {str(e)}") from e


def clean_lineage_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all lineage attributes from a DataFrame.
    
    Args:
        df: DataFrame to clean
        
    Returns:
        DataFrame with lineage attributes removed
        
    Raises:
        TypeError: If df is not a DataFrame
        ValueError: If DataFrame validation fails
        AttributeError: If pandas attributes cannot be accessed
        RuntimeError: If cleaning fails for other reasons
        
    Example:
        >>> df = clean_lineage_attributes(df)
    """
    # First validate input parameter
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
        
    try:
        # Ensure we have a valid DataFrame
        df = ensure_dataframe(df)
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid DataFrame when cleaning lineage attributes: {str(e)}")
        raise ValueError(f"Cannot clean lineage attributes: {str(e)}") from e
    
    # If DataFrame doesn't have attributes, nothing to clean
    if not hasattr(df, 'attrs'):
        logger.debug("DataFrame has no attrs dictionary, nothing to clean")
        return df
    
    try:
        # Define common lineage attribute names to remove
        lineage_attrs = [
            ATTR_MAIN_LINEAGE,
            ATTR_LINEAGE_IDS,
            ATTR_LINEAGES_DICT,
            ATTR_MINIMAL_FLAG,
            DF_LINEAGE_ID_ATTR
        ]
        
        # Remove all lineage-related attributes
        for attr in lineage_attrs:
            if attr in df.attrs:
                del df.attrs[attr]
                
        return df
    except KeyError as e:
        logger.warning(f"Key error removing lineage attribute: {str(e)}, continuing...")
        # Don't fail completely if one attribute can't be removed
        return df
    except AttributeError as e:
        logger.error(f"Attribute error cleaning lineage attributes: {str(e)}")
        raise AttributeError(f"Cannot modify DataFrame attributes: {str(e)}") from e
    except Exception as e:
        # For unexpected errors, provide clear context
        logger.error(f"Unexpected error cleaning lineage attributes: {str(e)}")
        raise RuntimeError(f"Failed to clean lineage attributes: {str(e)}") from e
