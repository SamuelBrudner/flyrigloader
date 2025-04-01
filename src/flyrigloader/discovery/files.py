"""
File discovery functionality.

Basic utilities for finding files based on patterns.
"""
from typing import List, Optional, Iterable
from pathlib import Path


def discover_files(
    directory: str, 
    pattern: str, 
    recursive: bool = False,
    extensions: Optional[List[str]] = None
) -> List[str]:
    """
    Discover files matching a pattern in the specified directory.
    
    Args:
        directory: The directory to search in
        pattern: File pattern to match (glob format)
        recursive: If True, search recursively through subdirectories
        extensions: Optional list of file extensions to filter by (without the dot)
        
    Returns:
        List of file paths matching the pattern and extensions
    """
    directory_path = Path(directory)
    
    # Handle file discovery based on recursion needs
    if recursive and "**" not in pattern:
        # Convert simple pattern to recursive search
        clean_pattern = pattern.lstrip("./")
        matched_files = directory_path.rglob(clean_pattern)
    else:
        # Use glob for non-recursive or patterns already containing **
        matched_files = directory_path.glob(pattern)
    
    # Convert to list of strings
    file_paths = [str(file) for file in matched_files]
    
    # Filter by extensions if specified
    if extensions:
        # Add dot prefix to extensions if not already there
        ext_filters = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]
        # Filter files by extensions
        file_paths = [f for f in file_paths if any(f.endswith(ext) for ext in ext_filters)]
    
    return file_paths
