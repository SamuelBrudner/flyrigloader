"""
File statistics functionality.

Utilities for collecting and working with file system metadata.
"""
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
from datetime import datetime
import os
import stat


def get_file_stats(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get file statistics for a given file path.
    
    Args:
        path: Path to the file
        
    Returns:
        Dictionary with comprehensive file stats including size, modification time,
        creation time, permissions, and other metadata.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
        
    stats = path.stat()

    # Determine creation time in a platform-independent manner
    creation_timestamp = getattr(stats, "st_birthtime", stats.st_ctime)

    return {
        "size": stats.st_size,
        "size_bytes": stats.st_size,  # Alias for backward compatibility
        "mtime": datetime.fromtimestamp(stats.st_mtime),
        "modified_time": stats.st_mtime,  # Timestamp version
        "creation_time": datetime.fromtimestamp(creation_timestamp),
        "ctime": datetime.fromtimestamp(stats.st_ctime),
        "created_time": stats.st_ctime,  # Timestamp version
        "is_directory": path.is_dir(),
        "is_file": path.is_file(),
        "is_symlink": path.is_symlink(),
        "filename": path.name,
        "extension": path.suffix[1:] if path.suffix else "",
        "permissions": stats.st_mode & 0o777,
        "is_readable": os.access(path, os.R_OK),
        "is_writable": os.access(path, os.W_OK),
        "is_executable": os.access(path, os.X_OK)
    }


def attach_file_stats(
    file_data: Union[List[str], Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """
    Attach file statistics to discovery results.
    
    Args:
        file_data: Either a list of file paths or a dictionary mapping file paths to metadata
        
    Returns:
        Dictionary mapping file paths to metadata including file statistics
    """
    # Handle list of file paths
    if isinstance(file_data, list):
        # Convert for loop to dictionary comprehension for better efficiency
        return {file_path: get_file_stats(file_path) for file_path in file_data}
    
    # Handle dictionary with existing metadata
    return {
        file_path: {**metadata, **get_file_stats(file_path)}
        for file_path, metadata in file_data.items()
    }
