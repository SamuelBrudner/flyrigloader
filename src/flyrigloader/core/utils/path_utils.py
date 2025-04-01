"""
Path utilities for standardized file path handling across the codebase.

This module provides consistent path handling functions to eliminate
inconsistencies between str and Path objects throughout the project.
"""

import os
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict, Any
from loguru import logger

PathLike = Union[str, Path]


def ensure_path(path: Optional[PathLike], default_path: Optional[PathLike] = None) -> Path:
    """
    Convert a string or Path object to a Path object.
    
    Args:
        path: Path as string or Path object
        default_path: Default path to use if path is None
        
    Returns:
        Path object
        
    Raises:
        ValueError: If path is None and no default path was provided
    """
    try:
        if path is not None:
            return path if isinstance(path, Path) else Path(path)
        elif default_path is not None:
            return default_path if isinstance(default_path, Path) else Path(default_path)
        raise ValueError("Path cannot be None and no default path was provided")
    except Exception as e:
        logger.error(f"Failed to ensure path: {str(e)}")
        raise ValueError(f"Failed to ensure path: {str(e)}") from e


def ensure_path_exists(path: Optional[PathLike], create_parents: bool = False, default_path: Optional[PathLike] = None) -> Path:
    """
    Ensure a path exists and return it as a Path object.
    
    Args:
        path: Path as string or Path object
        create_parents: If True, create parent directories if they don't exist
        default_path: Default path to use if path is None
        
    Returns:
        Path object
        
    Raises:
        FileNotFoundError: If the path doesn't exist and create_parents is False
        ValueError: If path is None and no default_path is provided
    """
    try:
        path_obj = ensure_path(path, default_path)

        # For directories
        if path_obj.is_dir() or (not path_obj.exists() and not path_obj.suffix):
            if not path_obj.exists():
                if not create_parents:
                    raise FileNotFoundError(f"Directory not found: {path_obj}")
                path_obj.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {path_obj}")
        elif not path_obj.exists():
            if create_parents and not path_obj.parent.exists():
                path_obj.parent.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created parent directories for: {path_obj}")
            # We don't create the file itself, just ensure parent dirs exist

        return path_obj
    except ValueError:
        # Re-raise ValueError to preserve error messages from ensure_path
        raise
    except Exception as e:
        logger.error(f"Failed to ensure path exists: {str(e)}")
        raise ValueError(f"Failed to ensure path exists: {str(e)}") from e


def get_absolute_path(path: Optional[PathLike], default_path: Optional[PathLike] = None) -> Path:
    """
    Convert a path to an absolute path.
    
    Args:
        path: Path as string or Path object
        default_path: Default path to use if path is None
        
    Returns:
        Absolute Path object
        
    Raises:
        ValueError: If path is None and no default_path is provided
    """
    try:
        return ensure_path(path, default_path).absolute()
    except ValueError:
        # Re-raise ValueError to preserve error messages from ensure_path
        raise
    except Exception as e:
        logger.error(f"Failed to get absolute path: {str(e)}")
        raise ValueError(f"Failed to get absolute path: {str(e)}") from e


def normalize_path(path: Optional[PathLike], default_path: Optional[PathLike] = None) -> Path:
    """
    Normalize a path (expand user directory, resolve symlinks).
    
    Args:
        path: Path as string or Path object
        default_path: Default path to use if path is None
        
    Returns:
        Normalized Path object
        
    Raises:
        ValueError: If path is None and no default_path is provided
    """
    try:
        path_obj = ensure_path(path, default_path)
        return path_obj.expanduser().resolve()
    except ValueError:
        # Re-raise ValueError to preserve error messages from ensure_path
        raise
    except Exception as e:
        logger.error(f"Failed to normalize path: {str(e)}")
        raise ValueError(f"Failed to normalize path: {str(e)}") from e


def get_relative_path(path: Optional[PathLike], base_path: Optional[PathLike], 
                     default_path: Optional[PathLike] = None, 
                     default_base_path: Optional[PathLike] = None) -> Path:
    """
    Get a path relative to a base path.
    
    Args:
        path: Path as string or Path object
        base_path: Base path as string or Path object
        default_path: Default path to use if path is None
        default_base_path: Default base path to use if base_path is None
        
    Returns:
        Relative Path object
        
    Raises:
        ValueError: If path or base_path is None and no default is provided
    """
    try:
        return ensure_path(path, default_path).relative_to(ensure_path(base_path, default_base_path))
    except ValueError:
        # Re-raise ValueError to preserve error messages from ensure_path
        raise
    except Exception as e:
        logger.error(f"Failed to get relative path: {str(e)}")
        raise ValueError(f"Failed to get relative path: {str(e)}") from e


def split_path(path: Optional[PathLike], default_path: Optional[PathLike] = None) -> Tuple[Path, str]:
    """
    Split a path into directory and filename.
    
    Args:
        path: Path as string or Path object
        default_path: Default path to use if path is None
        
    Returns:
        Tuple of (directory Path, filename)
        
    Raises:
        ValueError: If path is None and no default_path is provided
    """
    try:
        path_obj = ensure_path(path, default_path)
        return path_obj.parent, path_obj.name
    except ValueError:
        # Re-raise ValueError to preserve error messages from ensure_path
        raise
    except Exception as e:
        logger.error(f"Failed to split path: {str(e)}")
        raise ValueError(f"Failed to split path: {str(e)}") from e


def join_paths(base_path: Optional[PathLike], *paths, default_base_path: Optional[PathLike] = None) -> Path:
    """
    Join paths together.
    
    Args:
        base_path: Base path as string or Path object
        *paths: Additional path components
        default_base_path: Default base path to use if base_path is None
        
    Returns:
        Joined Path object
        
    Raises:
        ValueError: If base_path is None and no default_base_path is provided
    """
    try:
        result = ensure_path(base_path, default_base_path)
        for path in paths:
            if path is None:
                continue
            result = result / ensure_path(path).name if isinstance(path, (str, Path)) else result / str(path)
        return result
    except ValueError:
        # Re-raise ValueError to preserve error messages from ensure_path
        raise
    except Exception as e:
        logger.error(f"Failed to join paths: {str(e)}")
        raise ValueError(f"Failed to join paths: {str(e)}") from e


def change_extension(path: Optional[PathLike], new_extension: str, default_path: Optional[PathLike] = None) -> Path:
    """
    Change the extension of a path.
    
    Args:
        path: Path as string or Path object
        new_extension: New extension (with or without leading dot)
        default_path: Default path to use if path is None
        
    Returns:
        Path object with new extension
        
    Raises:
        ValueError: If path is None and no default_path is provided
    """
    try:
        path_obj = ensure_path(path, default_path)
        
        # Ensure the extension starts with a dot
        if not new_extension.startswith('.'):
            new_extension = f".{new_extension}"
            
        return path_obj.with_suffix(new_extension)
    except ValueError:
        # Re-raise ValueError to preserve error messages from ensure_path
        raise
    except Exception as e:
        logger.error(f"Failed to change extension: {str(e)}")
        raise ValueError(f"Failed to change extension: {str(e)}") from e


def get_related_path(path: Optional[PathLike], suffix: str, new_extension: Optional[str] = None, default_path: Optional[PathLike] = None) -> Path:
    """
    Get a related path by adding a suffix and optionally changing the extension.
    
    Args:
        path: Path as string or Path object
        suffix: Suffix to add to the filename (before the extension)
        new_extension: Optional new extension (with or without leading dot)
        default_path: Default path to use if path is None
        
    Returns:
        Related Path object
        
    Raises:
        ValueError: If path is None and no default_path is provided
    """
    try:
        path_obj = ensure_path(path, default_path)

        # Split the path into parts
        parent, name = path_obj.parent, path_obj.stem

        # Create the new filename with suffix
        new_name = f"{name}{suffix}"

        if not new_extension:
            return parent / f"{new_name}{path_obj.suffix}"

        if not new_extension.startswith('.'):
            new_extension = f".{new_extension}"
        return parent / f"{new_name}{new_extension}"
    except ValueError:
        # Re-raise ValueError to preserve error messages from ensure_path
        raise
    except Exception as e:
        logger.error(f"Failed to get related path: {str(e)}")
        raise ValueError(f"Failed to get related path: {str(e)}") from e
