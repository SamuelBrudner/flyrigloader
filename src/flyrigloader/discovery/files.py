"""
File discovery functionality.

Basic utilities for finding files based on patterns.
"""
from typing import List, Optional, Iterable, Union
from pathlib import Path


def discover_files(
    directory: Union[str, List[str]], 
    pattern: str, 
    recursive: bool = False,
    extensions: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    mandatory_substrings: Optional[List[str]] = None
) -> List[str]:
    """
    Discover files matching a pattern in the specified directory or directories.
    
    Args:
        directory: The directory or list of directories to search in
        pattern: File pattern to match (glob format)
        recursive: If True, search recursively through subdirectories
        extensions: Optional list of file extensions to filter by (without the dot)
        ignore_patterns: Optional list of substring patterns to ignore
        mandatory_substrings: Optional list of substrings that must be present in files
        
    Returns:
        List of file paths matching the pattern and extensions, excluding ignored patterns
        and including only paths with mandatory substrings (if specified)
    """
    # Handle single directory or multiple directories
    directories = [directory] if isinstance(directory, str) else directory
    
    # Collect all matching files
    all_matched_files = []
    
    for dir_path in directories:
        directory_path = Path(dir_path)
        
        # Handle file discovery based on recursion needs
        if recursive and "**" not in pattern:
            # Convert simple pattern to recursive search
            clean_pattern = pattern.lstrip("./")
            matched_files = directory_path.rglob(clean_pattern)
        else:
            # Use glob for non-recursive or patterns already containing **
            matched_files = directory_path.glob(pattern)
        
        # Add matched files to the result list
        all_matched_files.extend([str(file) for file in matched_files])
    
    # Filter by extensions if specified
    if extensions:
        # Add dot prefix to extensions if not already there
        ext_filters = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]
        # Filter files by extensions
        all_matched_files = [f for f in all_matched_files if any(f.endswith(ext) for ext in ext_filters)]
    
    # Apply ignore patterns if specified
    if ignore_patterns:
        # Filter out files matching any ignore pattern
        all_matched_files = [
            f for f in all_matched_files
            if all(pattern not in f for pattern in ignore_patterns)
        ]
    
    # Apply mandatory substrings if specified
    if mandatory_substrings:
        # Keep only files containing at least one of the mandatory substrings (OR logic)
        all_matched_files = [
            f for f in all_matched_files
            if any(pattern in f for pattern in mandatory_substrings)
        ]
    
    return all_matched_files
