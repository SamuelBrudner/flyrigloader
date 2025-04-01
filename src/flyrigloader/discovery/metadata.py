"""
Metadata discovery and assignment module.

This module provides utilities for extracting and assigning metadata 
to dataframes based on file attributes, path patterns, and user-provided info.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from loguru import logger
import re

from ..readers.pickle import extract_metadata_from_path
from ..core.utils import PathLike, ensure_path
from .files import discover_files_in_folder, build_file_items


def assign_metadata_to_dataframe(
    df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
    path: Optional[PathLike] = None,
    overwrite: bool = False
) -> pd.DataFrame:
    """
    Assign metadata to a DataFrame as columns.
    
    This function adds metadata as columns to a DataFrame. It can extract
    metadata from a file path and combine it with explicitly provided metadata.
    
    Args:
        df: DataFrame to assign metadata to
        metadata: Dictionary of metadata to assign
        path: Path to extract additional metadata from
        overwrite: Whether to overwrite existing columns with the same name
    
    Returns:
        DataFrame with metadata columns added
    """
    if df is None:
        logger.warning("Cannot assign metadata to None DataFrame")
        return df
        
    # Initialize an empty metadata dict if none provided
    metadata = metadata or {}
    
    # Extract metadata from path if provided
    if path:
        path_obj = ensure_path(path)
        path_metadata = extract_metadata_from_path(path_obj)
        metadata.update(path_metadata)
    
    # Skip if no metadata to assign
    if not metadata:
        return df
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Add metadata as columns
    for key, value in metadata.items():
        if key in result_df.columns and not overwrite:
            logger.debug(f"Skipping metadata column '{key}' (already exists)")
            continue
        result_df[key] = value
        
    return result_df


def parse_folder_name(folder_name: str) -> Dict[str, Any]:
    """
    Parse a folder name into metadata fields.
    
    This is a placeholder implementation that extracts basic information.
    Projects should override this with their own specific parsing logic.
    
    Args:
        folder_name: Folder name to parse
        
    Returns:
        Dictionary with parsed metadata fields
    """
    # Basic implementation that tries to extract common patterns
    metadata = {}
    
    # Try to extract key-value pairs (format: key-value or key_value)
    for separator in ['-', '_']:
        if separator in folder_name:
            parts = folder_name.split(separator, 1)
            if len(parts) == 2:
                metadata[parts[0]] = parts[1]
                return metadata
    
    # If no patterns matched, just use the folder name as-is
    metadata["folder"] = folder_name
    return metadata


def assign_metadata_to_item(
    folder_path: PathLike, 
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a metadata dictionary for a folder including its parsed name.
    
    Args:
        folder_path: Path to the folder
        additional_metadata: Additional metadata to include
        
    Returns:
        Dictionary with metadata fields
    """
    folder_path_obj = ensure_path(folder_path)
    folder_name = folder_path_obj.name
    
    # Parse the folder name into metadata fields
    metadata = parse_folder_name(folder_name)
    
    # Add the folder path as metadata
    metadata["folder_path"] = str(folder_path_obj)
    metadata["folder_name"] = folder_name
    
    # Add additional metadata if provided
    if additional_metadata:
        metadata.update(additional_metadata)
        
    return metadata


def build_file_items_with_metadata(
    folder_path: PathLike,
    pattern: str = "*",
    recursive: bool = False,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Build a list of item dictionaries for all matching files in a folder.
    
    Args:
        folder_path: Path to the folder containing files
        pattern: File glob pattern to match
        recursive: Whether to search recursively
        additional_metadata: Additional metadata to include
        
    Returns:
        List of dictionaries, each containing path and metadata for a file
    """
    folder_path_obj = ensure_path(folder_path)
    
    # Get folder-level metadata
    folder_metadata = assign_metadata_to_item(folder_path_obj, additional_metadata)
    
    # Discover all matching files and build item dictionaries
    return build_file_items(
        discover_files_in_folder(folder_path_obj, pattern, recursive), 
        folder_metadata
    )
