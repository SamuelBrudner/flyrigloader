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
from typing import Dict, Any, Optional, Union, Tuple, List, Callable
from loguru import logger

from .pickle import extract_metadata_from_path
from ..core.utils import ensure_path, ensure_path_exists, PathLike


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


def detect_format(file_path: PathLike) -> Optional[str]:
    """
    Detect file format from file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Format name or None if format not recognized
    """
    file_path = ensure_path(file_path)
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


def _get_file_reader(file_path: PathLike, format_name: Optional[str] = None) -> Callable:
    """
    Helper function to get the appropriate reader function for a file.
    
    Args:
        file_path: Path to the file
        format_name: Optional format name to override auto-detection
        
    Returns:
        Reader function to use
        
    Raises:
        ValueError: If format is not supported or no reader is available
    """
    path = ensure_path(file_path)
    
    # Use provided format_name or detect from file extension
    if format_name is None:
        format_name = detect_format(path)
        
    if format_name is None:
        raise ValueError(f"Unsupported or unknown file format for {path}")
    
    # Get the appropriate reader function
    reader_func = get_reader_for_format(format_name)
    
    if reader_func is None:
        raise ValueError(f"No reader available for format: {format_name}")
        
    return reader_func


def _add_metadata_to_dataframe(df: pd.DataFrame, path: PathLike) -> pd.DataFrame:
    """
    Add file metadata as columns to a DataFrame.
    
    Args:
        df: DataFrame to add metadata to
        path: Path to extract metadata from
        
    Returns:
        DataFrame with metadata columns added
    """
    if df is None or df.empty:
        return df
        
    metadata = extract_metadata_from_path(path)
    for col_name, value in metadata.items():
        if col_name not in df.columns:
            df[col_name] = value
    
    return df


def _read_file_with_error_handling(
    read_func: Callable,
    file_path: PathLike,
    file_type: str,
    **kwargs
) -> pd.DataFrame:
    """
    Read a file with consistent error handling.
    
    Args:
        read_func: Pandas read function to use (e.g., pd.read_csv)
        file_path: Path to the file
        file_type: Type of file for error messages (e.g., 'CSV', 'Excel')
        **kwargs: Additional arguments to pass to the read function
        
    Returns:
        DataFrame with the file contents
        
    Raises:
        FileNotFoundError: If the file does not exist
        PermissionError: If the file cannot be accessed due to permissions
        IOError: If there are general I/O errors when reading the file
        ValueError: If the file content is invalid or cannot be parsed
        TypeError: If parameters are of incorrect type
        pd.errors.EmptyDataError: If the file is empty
        pd.errors.ParserError: If pandas fails to parse the file
        RuntimeError: If there are any other unhandled errors
    """
    try:
        path = ensure_path(file_path)
        if not path.exists():
            logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"File does not exist: {path}")
        
        return read_func(path, **kwargs)
    except FileNotFoundError as e:
        logger.error(f"File not found when reading {file_type} from {path}: {e}")
        raise
    except PermissionError as e:
        logger.error(f"Permission denied when reading {file_type} from {path}: {e}")
        raise PermissionError(f"Permission denied when reading {file_type} file: {str(e)}") from e
    except (IOError, OSError) as e:
        logger.error(f"I/O error when reading {file_type} from {path}: {e}")
        raise IOError(f"Failed to read {file_type} file: {str(e)}") from e
    except ValueError as e:
        logger.error(f"Value error when reading {file_type} from {path}: {e}")
        raise ValueError(f"Invalid content in {file_type} file: {str(e)}") from e
    except TypeError as e:
        logger.error(f"Type error when reading {file_type} from {path}: {e}")
        raise TypeError(f"Type error when reading {file_type} file: {str(e)}") from e
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data error when reading {file_type} from {path}: {e}")
        raise pd.errors.EmptyDataError(f"The {file_type} file is empty: {str(e)}") from e
    except pd.errors.ParserError as e:
        logger.error(f"Parser error when reading {file_type} from {path}: {e}")
        raise pd.errors.ParserError(f"Failed to parse {file_type} file: {str(e)}") from e
    except Exception as e:
        # For truly unexpected errors that we can't anticipate
        logger.error(f"Unexpected error reading {file_type} from {path}: {type(e).__name__}: {e}")
        raise RuntimeError(f"Failed to read {file_type} file: {str(e)}") from e


def _create_reader_function(reader_func: Callable, file_type: str) -> Callable:
    """
    Create a standardized reader function with proper error handling.
    
    Args:
        reader_func: The pandas reader function (pd.read_csv, pd.read_parquet, etc.)
        file_type: The file type name for error messages (CSV, Parquet, etc.)
        
    Returns:
        A function that reads files with standardized error handling
    """
    def reader(file_path: PathLike, **kwargs) -> pd.DataFrame:
        return _read_file_with_error_handling(reader_func, file_path, file_type, **kwargs)
    return reader


# Create reader functions using the factory pattern
read_csv = _create_reader_function(pd.read_csv, "CSV")
read_csv.__doc__ = """
Read a CSV file with error handling.

Args:
    file_path: Path to the CSV file
    **kwargs: Additional arguments to pass to pd.read_csv
    
Returns:
    DataFrame with the CSV contents
    
Raises:
    FileNotFoundError: If the file does not exist
    PermissionError: If the file cannot be accessed due to permissions
    IOError: If there are general I/O errors when reading the file
    ValueError: If the file content is invalid or cannot be parsed
    TypeError: If parameters are of incorrect type
    pd.errors.EmptyDataError: If the file is empty
    pd.errors.ParserError: If pandas fails to parse the file
    RuntimeError: If there are any other unhandled errors
"""

read_parquet = _create_reader_function(pd.read_parquet, "Parquet")
read_parquet.__doc__ = """
Read a Parquet file with error handling.

Args:
    file_path: Path to the Parquet file
    **kwargs: Additional arguments to pass to pd.read_parquet
    
Returns:
    DataFrame with the Parquet contents
    
Raises:
    FileNotFoundError: If the file does not exist
    PermissionError: If the file cannot be accessed due to permissions
    IOError: If there are general I/O errors when reading the file
    ValueError: If the file content is invalid or cannot be parsed
    TypeError: If parameters are of incorrect type
    pd.errors.EmptyDataError: If the file is empty
    pd.errors.ParserError: If pandas fails to parse the file
    RuntimeError: If there are any other unhandled errors
"""

read_feather = _create_reader_function(pd.read_feather, "Feather")
read_feather.__doc__ = """
Read a Feather file with error handling.

Args:
    file_path: Path to the Feather file
    **kwargs: Additional arguments to pass to pd.read_feather
    
Returns:
    DataFrame with the Feather contents
    
Raises:
    FileNotFoundError: If the file does not exist
    PermissionError: If the file cannot be accessed due to permissions
    IOError: If there are general I/O errors when reading the file
    ValueError: If the file content is invalid or cannot be parsed
    TypeError: If parameters are of incorrect type
    pd.errors.EmptyDataError: If the file is empty
    pd.errors.ParserError: If pandas fails to parse the file
    RuntimeError: If there are any other unhandled errors
"""

read_excel = _create_reader_function(pd.read_excel, "Excel")
read_excel.__doc__ = """
Read an Excel file with error handling.

Args:
    file_path: Path to the Excel file
    **kwargs: Additional arguments to pass to pd.read_excel
    
Returns:
    DataFrame with the Excel contents
    
Raises:
    FileNotFoundError: If the file does not exist
    PermissionError: If the file cannot be accessed due to permissions
    IOError: If there are general I/O errors when reading the file
    ValueError: If the file content is invalid or cannot be parsed
    TypeError: If parameters are of incorrect type
    pd.errors.EmptyDataError: If the file is empty
    pd.errors.ParserError: If pandas fails to parse the file
    RuntimeError: If there are any other unhandled errors
"""


def read_file(
    file_path: PathLike,
    format_name: Optional[str] = None,
    add_metadata: bool = True,
    reader_kwargs: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Read a file in any supported format.
    
    Args:
        file_path: Path to the file
        format_name: Optional format name to override auto-detection
        add_metadata: Whether to add file metadata as columns
        reader_kwargs: Additional arguments to pass to the reader function
        
    Returns:
        DataFrame with the file contents
        
    Raises:
        FileNotFoundError: If the file does not exist
        PermissionError: If the file cannot be accessed due to permissions
        IOError: If there are general I/O errors when reading the file
        ValueError: If the format is not supported or the file content is invalid
        TypeError: If parameters are of incorrect type
        pd.errors.EmptyDataError: If the file is empty
        pd.errors.ParserError: If pandas fails to parse the file
        RuntimeError: If there are any other unhandled errors
    """
    try:
        path = ensure_path_exists(file_path)
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid path format: {str(e)}")
        raise ValueError(f"Invalid path format: {str(e)}") from e
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise FileNotFoundError(f"File not found: {str(e)}") from e
    
    reader_kwargs = reader_kwargs or {}
    
    # Handle pickle files separately
    if path.suffix.lower() in ['.pkl', '.pickle']:
        from .pickle import read_pickle_auto
        try:
            df = read_pickle_auto(path, **reader_kwargs)
        except (FileNotFoundError, PermissionError):
            # Let these propagate directly
            raise
        except Exception as e:
            logger.error(f"Error reading pickle file {path}: {type(e).__name__}: {e}")
            raise ValueError(f"Error reading pickle file: {str(e)}") from e
            
        if df is None:
            logger.error(f"Pickle file {path} was read but contained None")
            raise ValueError(f"Pickle file {path} contained None")
    else:
        # Get reader function for the file format
        try:
            reader_func = _get_file_reader(path, format_name)
        except ValueError as e:
            logger.error(f"Failed to get reader for {path}: {str(e)}")
            raise ValueError(f"Unsupported file format: {str(e)}") from e
            
        # Read the file using the reader function with error handling
        df = _read_file_with_error_handling(
            reader_func, 
            path, 
            format_name or path.suffix[1:].upper(),
            **reader_kwargs
        )
    
    # Add file metadata if requested
    if add_metadata and df is not None:
        df = _add_metadata_to_dataframe(df, path)
        
    return df


def read_directory(
    directory_path: PathLike,
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
        
    Raises:
        FileNotFoundError: If the directory does not exist
        NotADirectoryError: If the path exists but is not a directory
        PermissionError: If the directory cannot be accessed due to permissions
        ValueError: If the pattern is invalid or no supported formats are found
        IOError: If there are file I/O errors
        RuntimeError: If all files fail to load or other unhandled errors occur
    """
    try:
        # Validate and normalize directory path
        dir_path = ensure_path(directory_path)
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid directory path format: {str(e)}")
        raise ValueError(f"Invalid directory path format: {str(e)}") from e
        
    # Validate directory exists
    if not dir_path.exists():
        error_msg = f"Directory not found: {dir_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Validate it's actually a directory
    if not dir_path.is_dir():
        error_msg = f"Path exists but is not a directory: {dir_path}"
        logger.error(error_msg)
        raise NotADirectoryError(error_msg)
    
    # Validate pattern
    if not pattern:
        error_msg = "Pattern cannot be empty"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # Find all matching files
        glob_pattern = f"**/{pattern}" if recursive else pattern
        try:
            matched_files = list(dir_path.glob(glob_pattern))
        except Exception as e:
            logger.error(f"Error while globbing for pattern '{glob_pattern}' in {dir_path}: {e}")
            raise ValueError(f"Invalid glob pattern '{pattern}': {str(e)}") from e
        
        if not matched_files:
            logger.warning(f"No files matching pattern '{pattern}' found in {dir_path}")
            return pd.DataFrame() if combine else []
        
        # Filter by format if specified
        if formats:
            # Normalize formats (remove dots, convert to lowercase)
            try:
                normalized_formats = [fmt.lower().lstrip('.') for fmt in formats]
                matched_files = [
                    f for f in matched_files 
                    if f.is_file() and f.suffix.lower().lstrip('.') in normalized_formats
                ]
            except Exception as e:
                logger.error(f"Error filtering files by format: {e}")
                raise ValueError(f"Invalid format specification: {str(e)}") from e
                
            if not matched_files:
                logger.warning(f"No files with specified formats {formats} found in {dir_path}")
                return pd.DataFrame() if combine else []
            
        # Attempt to read each file
        results = []
        errors = []
        
        for file_path in matched_files:
            if not file_path.is_file():
                continue
                
            try:
                df = read_file(file_path)
                if df is not None and not df.empty:
                    results.append((file_path, df))
            except FileNotFoundError as e:
                # File might have been deleted between listing and reading
                logger.warning(f"File disappeared during processing: {file_path}: {e}")
                errors.append((file_path, f"FileNotFoundError: {str(e)}"))
            except PermissionError as e:
                logger.warning(f"Permission denied for file {file_path}: {e}")
                errors.append((file_path, f"PermissionError: {str(e)}"))
            except pd.errors.EmptyDataError as e:
                logger.warning(f"Empty data in file {file_path}: {e}")
                errors.append((file_path, f"EmptyDataError: {str(e)}"))
            except pd.errors.ParserError as e:
                logger.warning(f"Failed to parse file {file_path}: {e}")
                errors.append((file_path, f"ParserError: {str(e)}"))
            except ValueError as e:
                logger.warning(f"Invalid content in file {file_path}: {e}")
                errors.append((file_path, f"ValueError: {str(e)}"))
            except (IOError, OSError) as e:
                logger.warning(f"I/O error with file {file_path}: {e}")
                errors.append((file_path, f"IOError: {str(e)}"))
            except Exception as e:
                # For truly unexpected errors
                logger.warning(f"Unexpected error reading file {file_path}: {type(e).__name__}: {e}")
                errors.append((file_path, f"{type(e).__name__}: {str(e)}"))
                
        if not results and errors:
            # If all files failed to load, raise an error with details
            error_details = "; ".join([f"{path}: {err}" for path, err in errors[:5]])
            if len(errors) > 5:
                error_details += f"; and {len(errors) - 5} more errors"
                
            logger.error(f"Failed to read any files from directory {dir_path}")
            raise RuntimeError(f"Failed to read any files from directory. Sample errors: {error_details}")
                
        # Return appropriate result based on combine flag
        if combine:
            try:
                return _combine_dataframes(results) if results else pd.DataFrame()
            except Exception as e:
                logger.error(f"Error combining dataframes: {e}")
                raise RuntimeError(f"Failed to combine dataframes: {str(e)}") from e
        else:
            return results
    except (FileNotFoundError, NotADirectoryError, PermissionError, ValueError) as e:
        # Let these propagate directly
        raise
    except Exception as e:
        # Handle any truly unexpected errors
        logger.error(f"Unexpected error in read_directory for {dir_path}: {type(e).__name__}: {e}")
        raise RuntimeError(f"Failed to read directory: {str(e)}") from e


def _combine_dataframes(results: List[Tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    """
    Combine multiple DataFrames into one.
    
    Args:
        results: List of (path, DataFrame) tuples to combine
        
    Returns:
        Combined DataFrame
    """
    df_list = [df for _, df in results]
    return pd.concat(df_list, ignore_index=True)


def _save_dataframe_by_format(df: pd.DataFrame, path: Path, format_name: str, **kwargs) -> bool:
    """
    Save a DataFrame in the specified format.
    
    Args:
        df: DataFrame to save
        path: Path to save the file
        format_name: Format name with leading dot
        **kwargs: Additional arguments to pass to the writer function
        
    Returns:
        True if saving was successful
        
    Raises:
        ValueError: If the format is not supported or the DataFrame is invalid
        TypeError: If the input parameters are of incorrect types
        IOError: If there are file I/O errors
        PermissionError: If write access is denied
        OSError: If there are file system errors
    """
    # Define sets for format groups
    csv_formats = {'.csv', '.txt'}
    excel_formats = {'.xlsx', '.xls'}
    parquet_formats = {'.parquet', '.pq'}
    feather_formats = {'.feather', '.ftr'}
    hdf_formats = {'.h5', '.hdf5'}
    pickle_formats = {'.pkl', '.pickle'}
    
    try:
        # Ensure DataFrame is valid and matches expectations
        if df is None:
            raise ValueError("Cannot save None as DataFrame")
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
            
        # Check for empty DataFrame - not an error but worth logging
        if df.empty:
            logger.warning(f"Saving an empty DataFrame to {path}")
        
        # CSV formats
        if format_name in csv_formats:
            df.to_csv(path, **kwargs)
            logger.debug(f"Successfully saved DataFrame to CSV format: {path}")
            return True
            
        # Excel formats
        if format_name in excel_formats:
            df.to_excel(path, **kwargs)
            logger.debug(f"Successfully saved DataFrame to Excel format: {path}")
            return True
            
        # Optimized formats
        if format_name in parquet_formats:
            df.to_parquet(path, **kwargs)
            logger.debug(f"Successfully saved DataFrame to Parquet format: {path}")
            return True
            
        if format_name in feather_formats:
            df.to_feather(path, **kwargs)
            logger.debug(f"Successfully saved DataFrame to Feather format: {path}")
            return True
            
        # JSON formats
        if format_name == '.json':
            df.to_json(path, **kwargs)
            logger.debug(f"Successfully saved DataFrame to JSON format: {path}")
            return True
            
        if format_name == '.jsonl':
            df.to_json(path, orient='records', lines=True, **kwargs)
            logger.debug(f"Successfully saved DataFrame to JSONL format: {path}")
            return True
            
        # HDF formats
        if format_name in hdf_formats:
            key = kwargs.pop('key', 'data')
            df.to_hdf(path, key=key, **kwargs)
            logger.debug(f"Successfully saved DataFrame to HDF format: {path}")
            return True
        
        # Other formats like pickle
        if format_name in pickle_formats:
            df.to_pickle(path, **kwargs)
            logger.debug(f"Successfully saved DataFrame to Pickle format: {path}")
            return True
            
        # If we get here, something went wrong with our validation
        raise ValueError(f"Unsupported format: {format_name}")
        
    except ValueError as e:
        logger.error(f"Value error saving DataFrame to {path}: {e}")
        raise ValueError(f"Failed to save DataFrame: {str(e)}") from e
    except TypeError as e:
        logger.error(f"Type error saving DataFrame to {path}: {e}")
        raise TypeError(f"Type error when saving DataFrame: {str(e)}") from e
    except PermissionError as e:
        logger.error(f"Permission denied when saving to {path}: {e}")
        raise PermissionError(f"Permission denied when saving file: {str(e)}") from e
    except (IOError, OSError) as e:
        logger.error(f"I/O error when saving to {path}: {e}")
        raise IOError(f"Failed to save file due to I/O error: {str(e)}") from e
    except Exception as e:
        logger.error(f"Unexpected error saving DataFrame to {path}: {type(e).__name__}: {e}")
        # Re-raise with added context but don't hide the original exception type
        raise


def save_dataframe(
    df: pd.DataFrame,
    file_path: PathLike,
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
        True if saving was successful
        
    Raises:
        FileNotFoundError: If the parent directory does not exist and cannot be created
        PermissionError: If there are permission issues with the path
        ValueError: If the format is not supported or parameters are invalid
        TypeError: If the DataFrame or parameters are of incorrect types
        IOError: If there are file I/O errors
        OSError: If there are file system errors
        RuntimeError: If there are any other unhandled errors
    """
    # Validate DataFrame first
    if df is None:
        logger.error("Cannot save None as DataFrame")
        raise ValueError("Cannot save None as DataFrame")
        
    if not isinstance(df, pd.DataFrame):
        logger.error(f"Expected pandas DataFrame, got {type(df).__name__}")
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        
    # Normalize and validate path
    try:
        path = ensure_path(file_path)
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid path format: {str(e)}")
        raise ValueError(f"Invalid path format: {str(e)}") from e
    
    # Create parent directories if they don't exist
    try:
        if not path.parent.exists():
            logger.debug(f"Creating parent directories for {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        logger.error(f"Permission denied when creating directories for {path}: {e}")
        raise PermissionError(f"Cannot create directories due to permissions: {str(e)}") from e
    except OSError as e:
        logger.error(f"OS error when creating directories for {path}: {e}")
        raise OSError(f"Failed to create directories: {str(e)}") from e
    
    # Detect format if not specified
    if format_name is None:
        format_name = path.suffix.lower()
    
    # Ensure format has a leading dot
    if not format_name.startswith('.'):
        format_name = f".{format_name}"
    
    # Check for unsupported format
    supported_formats = {".csv", ".txt", ".xlsx", ".xls", ".parquet", ".pq", 
                        ".feather", ".ftr", ".json", ".jsonl", ".h5", ".hdf5", 
                        ".pkl", ".pickle"}
    if format_name not in supported_formats:
        logger.error(f"Unsupported format: {format_name}")
        raise ValueError(f"Unsupported format: {format_name}")
    
    # Save the DataFrame using the appropriate method
    try:
        return _save_dataframe_by_format(df, path, format_name, **kwargs)
    except (ValueError, TypeError, PermissionError, IOError, OSError) as e:
        # Let these propagate directly as they're already properly logged
        raise
    except Exception as e:
        # Handle truly unexpected errors
        logger.error(f"Unexpected error saving DataFrame to {path}: {type(e).__name__}: {e}")
        raise RuntimeError(f"Failed to save DataFrame: {str(e)}") from e