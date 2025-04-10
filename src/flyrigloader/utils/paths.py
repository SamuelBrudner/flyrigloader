"""
Path utilities for working with file paths.

Utilities for manipulating, converting, and validating file paths.
"""
from typing import Union, List, Optional
from pathlib import Path


def get_relative_path(path: Union[str, Path], base_dir: Union[str, Path]) -> Path:
    """
    Get a path relative to a base directory.
    
    Args:
        path: Path to convert
        base_dir: Base directory
        
    Returns:
        Relative path
        
    Raises:
        ValueError: If the path is not within the base directory
    """
    path = Path(path).resolve()
    base_dir = Path(base_dir).resolve()

    try:
        return path.relative_to(base_dir)
    except ValueError as e:
        raise ValueError(f"Path {path} is not within base directory {base_dir}") from e


def get_absolute_path(path: Union[str, Path], base_dir: Union[str, Path]) -> Path:
    """
    Convert a relative path to an absolute path.
    
    Args:
        path: Relative path
        base_dir: Base directory
        
    Returns:
        Absolute path
    """
    path = Path(path)
    base_dir = Path(base_dir)

    return path if path.is_absolute() else (base_dir / path).resolve()


def find_common_base_directory(paths: List[Union[str, Path]]) -> Optional[Path]:
    """
    Find the common base directory for a list of paths.
    
    Args:
        paths: List of paths to analyze
        
    Returns:
        Common base directory or None if no common base can be found
    """
    if not paths:
        return None

    # Convert all paths to Path objects and resolve them
    resolved_paths = [Path(path).resolve() for path in paths]

    # Find the common parts
    parts_list = [path.parts for path in resolved_paths]
    min_length = min(len(parts) for parts in parts_list)

    # Find the common prefix
    common_parts = []
    for i in range(min_length):
        if all(parts[i] == parts_list[0][i] for parts in parts_list):
            common_parts.append(parts_list[0][i])
        else:
            break

    return Path(*common_parts) if common_parts else None


def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path to the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_path(path: Union[str, Path]) -> Path:
    """
    Normalize a path, resolving symlinks and removing redundant elements.
    
    Args:
        path: Path to normalize
        
    Returns:
        Normalized path
    """
    return Path(path).resolve()


def get_file_extension(path: Union[str, Path]) -> str:
    """
    Get the file extension from a path.
    
    Args:
        path: Path to a file
        
    Returns:
        File extension (without the dot)
    """
    return Path(path).suffix[1:] if Path(path).suffix else ""


def get_filename(path: Union[str, Path]) -> str:
    """
    Get the filename (without directory) from a path.
    
    Args:
        path: Path to a file
        
    Returns:
        Filename without directory
    """
    return Path(path).name


def get_parent_directory(path: Union[str, Path]) -> Path:
    """
    Get the parent directory from a path.
    
    Args:
        path: Path to a file or directory
        
    Returns:
        Parent directory path
    """
    return Path(path).parent


def check_file_exists(file_path: Union[str, Path]) -> bool:
    """
    Check if a file exists.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file exists, False otherwise
    """
    # Using is_file() alone is sufficient as it returns False if the path doesn't exist
    return Path(file_path).is_file()
