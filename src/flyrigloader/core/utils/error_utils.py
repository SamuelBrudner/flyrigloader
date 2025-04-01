"""
error_utils.py - Standardized error handling utilities.

This module provides shared functions for creating standardized error metadata
and managing error handling across the flyrigloader package.
"""

from datetime import datetime
from typing import Any, Dict, Optional, Type


def create_metadata(success: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Create a standardized metadata dictionary following the project's conventions.
    
    Args:
        success: Whether the operation was successful
        **kwargs: Additional metadata fields to include
    
    Returns:
        Metadata dictionary with standard fields
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "success": success,
    } | kwargs


def create_error_metadata(error: Exception) -> Dict[str, Any]:
    """
    Create a standardized error metadata dictionary from an exception.
    
    Args:
        error: The exception to convert
        
    Returns:
        Dictionary with standardized error information
    """
    return create_metadata(
        success=False,
        error=str(error),
        error_type=type(error).__name__
    )


def update_metadata(
    target_metadata: Dict[str, Any], 
    source_metadata: Dict[str, Any], 
    exclude_fields: Optional[list] = None
) -> Dict[str, Any]:
    """
    Update a target metadata dictionary with values from a source metadata.
    
    Args:
        target_metadata: The metadata dictionary to update
        source_metadata: The metadata dictionary to copy values from
        exclude_fields: Fields to exclude from copying (timestamp is always excluded)
        
    Returns:
        Updated metadata dictionary (modifies target_metadata in-place)
    """
    exclude = set(exclude_fields or [])
    exclude.add("timestamp")  # Always exclude timestamp
    
    for key, value in source_metadata.items():
        if key not in exclude:
            target_metadata[key] = value
    
    return target_metadata
