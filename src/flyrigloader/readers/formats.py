"""
formats.py - Module for reading data from various file formats.

This module provides a unified interface for loading data from different file formats
including CSV, Parquet, Feather, and other common formats, with consistent error 
handling and output structure.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable, List, Tuple
from loguru import logger

from .pickle import extract_metadata_from_path


# Dictionary mapping file extensions to pandas read functions
FORMAT_READERS = {
    # CSV formats
    '.csv': pd.read_csv,
    '.txt': lambda path, **kwargs: pd.read_csv(path, sep=None, engine='python', **kwargs),
    
    # Excel formats
    '.xlsx': pd.read_excel,
    '.xls': pd.read_excel,
    
    # Optimized formats
    '.parquet': pd.read_parquet,
    '.pq': pd.read_parquet,
    '.feather': pd.read_feather,
    '.ftr': pd.read_feather,
    
    # JSON formats
    '.json': pd.read_json,
    '.jsonl': lambda path, **kwargs: pd.read_json(path, lines=True, **kwargs),
    
    # Other formats
    '.h5': lambda path, **kwargs: pd.read_hdf(path, **kwargs),
    '.hdf5': lambda path, **kwargs: pd.read_hdf(path, **kwargs),
    
    # Pickle formats - Using None as placeholder since they have special handling
    '.pkl': None,
    '.pickle': None
}


def detect_format(file_path: Union[str, Path]) -> Optional[str]:
    """
    Detect file format from file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Format name or None if format not recognized
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    return extension[1:] if extension in FORMAT_READERS else None


def get_reader_for_format(format_name: str) -> Optional[Callable]:
    """
    Get the appropriate reader function for a file format.
    
    Args:
        format_name: Name of the format (csv, parquet, etc.)
        
    Returns:
        Reader function or None if format not supported
    """
    # Normalize format name
    format_name = format_name if format_name.startswith('.') else f".{format_name}"
    return FORMAT_READERS.get(format_name)


def read_csv(file_path: Union[str, Path], **kwargs) -> Optional[pd.DataFrame]:
    """
    Read a CSV file with robust error handling.
    
    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments to pass to pd.read_csv
        
    Returns:
        DataFrame or None if loading fails
    """
    try:
        return pd.read_csv(file_path, **kwargs)
    except pd.errors.EmptyDataError:
        logger.error(f"Empty CSV file: {file_path}")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error in {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        return None


def read_parquet(file_path: Union[str, Path], **kwargs) -> Optional[pd.DataFrame]:
    """
    Read a Parquet file with robust error handling.
    
    Args:
        file_path: Path to the Parquet file
        **kwargs: Additional arguments to pass to pd.read_parquet
        
    Returns:
        DataFrame or None if loading fails
    """
    try:
        return pd.read_parquet(file_path, **kwargs)
    except Exception as e:
        logger.error(f"Error reading Parquet file {file_path}: {e}")
        return None


def read_feather(file_path: Union[str, Path], **kwargs) -> Optional[pd.DataFrame]:
    """
    Read a Feather file with robust error handling.
    
    Args:
        file_path: Path to the Feather file
        **kwargs: Additional arguments to pass to pd.read_feather
        
    Returns:
        DataFrame or None if loading fails
    """
    try:
        return pd.read_feather(file_path, **kwargs)
    except Exception as e:
        logger.error(f"Error reading Feather file {file_path}: {e}")
        return None


def read_excel(file_path: Union[str, Path], **kwargs) -> Optional[pd.DataFrame]:
    """
    Read an Excel file with robust error handling.
    
    Args:
        file_path: Path to the Excel file
        **kwargs: Additional arguments to pass to pd.read_excel
        
    Returns:
        DataFrame or None if loading fails
    """
    try:
        return pd.read_excel(file_path, **kwargs)
    except Exception as e:
        logger.error(f"Error reading Excel file {file_path}: {e}")
        return None


def read_file(
    file_path: Union[str, Path],
    format_name: Optional[str] = None,
    add_metadata: bool = True,
    reader_kwargs: Optional[Dict[str, Any]] = None
) -> Optional[pd.DataFrame]:
    """
    Read a file in any supported format.
    
    Args:
        file_path: Path to the file
        format_name: Optional format name to override auto-detection
        add_metadata: Whether to add file metadata as columns
        reader_kwargs: Additional arguments to pass to the reader function
        
    Returns:
        DataFrame or None if loading fails
    """
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None

    # Determine format
    if not format_name:
        format_name = detect_format(file_path)
    if not format_name:
        logger.error(f"Unsupported file format: {file_path}")
        return None

    # Special case for pickle files
    if format_name in ('pkl', 'pickle'):
        from .pickle import load_pickle_to_dataframe
        return load_pickle_to_dataframe(file_path)

    # Get reader function for non-pickle formats
    reader = get_reader_for_format(format_name)
    if not reader:
        logger.error(f"No reader available for format: {format_name}")
        return None

    # Read the file
    kwargs = reader_kwargs or {}
    try:
        df = reader(file_path, **kwargs)

        # Add metadata if requested
        if add_metadata and df is not None:
            metadata = extract_metadata_from_path(file_path)
            for key, value in metadata.items():
                df[key] = value

        return df
    except Exception as e:
        logger.error(f"Error reading {format_name} file {file_path}: {e}")
        return None


def read_directory(
    directory_path: Union[str, Path],
    pattern: str = "*.*",
    recursive: bool = False,
    combine: bool = True,
    formats: Optional[List[str]] = None
) -> Union[pd.DataFrame, List[Tuple[Path, pd.DataFrame]]]:
    """
    Read all matching files in a directory.
    
    Args:
        directory_path: Path to the directory
        pattern: Glob pattern to match files
        recursive: Whether to search subdirectories
        combine: Whether to combine all data into a single DataFrame
        formats: Optional list of formats to include (e.g., ['csv', 'parquet'])
        
    Returns:
        If combine=True: Combined DataFrame with all data
        If combine=False: List of (path, DataFrame) tuples
    """
    import glob
    
    directory_path = Path(directory_path)
    
    # Check if directory exists
    if not directory_path.exists() or not directory_path.is_dir():
        logger.error(f"Directory not found: {directory_path}")
        return pd.DataFrame() if combine else []
    
    # Find matching files
    if recursive:
        glob_pattern = str(directory_path / "**" / pattern)
        matching_files = glob.glob(glob_pattern, recursive=True)
    else:
        glob_pattern = str(directory_path / pattern)
        matching_files = glob.glob(glob_pattern)
    
    # Filter by format if specified
    if formats:
        format_extensions = [fmt if fmt.startswith('.') else f".{fmt}" for fmt in formats]
        matching_files = [f for f in matching_files if Path(f).suffix.lower() in format_extensions]
    
    # Sort files for consistency
    matching_files.sort()
    
    # Read each file
    results = []
    for file_path in matching_files:
        df = read_file(file_path)
        if df is not None and not df.empty:
            results.append((Path(file_path), df))
    
    # Combine if requested
    if combine and results:
        return pd.concat([df for _, df in results], ignore_index=True)
    
    return results


def save_dataframe(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    format_name: Optional[str] = None,
    **kwargs
) -> bool:
    """
    Save a DataFrame to a file in the specified format.
    
    Args:
        df: DataFrame to save
        file_path: Path to save the file
        format_name: Format name (if None, inferred from file extension)
        **kwargs: Additional arguments to pass to the writer function
        
    Returns:
        True if successful, False otherwise
    """
    file_path = Path(file_path)
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Infer format from extension if not provided
    if not format_name:
        format_name = file_path.suffix.lower()[1:]  # Remove the leading dot
    
    # Map format to writer function
    format_writers = {
        'csv': df.to_csv,
        'parquet': df.to_parquet,
        'pq': df.to_parquet,
        'feather': df.to_feather,
        'ftr': df.to_feather,
        'pkl': df.to_pickle,
        'pickle': df.to_pickle,
        'json': df.to_json,
        'xlsx': df.to_excel,
        'xls': df.to_excel,
        'h5': df.to_hdf,
        'hdf5': df.to_hdf
    }
    
    # Get writer function
    writer = format_writers.get(format_name)
    if not writer:
        logger.error(f"Unsupported output format: {format_name}")
        return False
    
    # CSV-specific default kwargs
    if format_name == 'csv' and 'index' not in kwargs:
        kwargs['index'] = False
    
    # HDF5-specific default kwargs
    if format_name in ('h5', 'hdf5') and 'key' not in kwargs:
        kwargs['key'] = 'data'
    
    # Write the file
    try:
        writer(file_path, **kwargs)
        logger.info(f"Saved DataFrame to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving DataFrame to {file_path}: {e}")
        return False