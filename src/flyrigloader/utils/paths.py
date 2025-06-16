"""
Path utilities for working with file paths.

Utilities for manipulating, converting, and validating file paths with enhanced
testability through dependency injection and configurable filesystem providers.

Enhanced Features for Testing:
- Dependency injection patterns for filesystem access components
- Configurable filesystem providers for pytest.monkeypatch scenarios
- Test-specific entry points for controlled utility behavior
- Cross-platform abstraction for comprehensive testing isolation
"""
from typing import Union, List, Optional, Protocol, runtime_checkable
from pathlib import Path
from abc import ABC, abstractmethod
import os
import sys
from flyrigloader import logger


@runtime_checkable
class FileSystemProvider(Protocol):
    """
    Protocol defining filesystem operations for dependency injection.
    
    Enables comprehensive testing through configurable filesystem providers
    supporting pytest.monkeypatch scenarios and cross-platform testing isolation.
    """
    
    def resolve_path(self, path: Path) -> Path:
        """Resolve a path to its absolute canonical form."""
        ...
    
    def make_relative(self, path: Path, base: Path) -> Path:
        """Make path relative to base directory."""
        ...
    
    def is_absolute(self, path: Path) -> bool:
        """Check if path is absolute."""
        ...
    
    def join_paths(self, base: Path, *parts: Union[str, Path]) -> Path:
        """Join path components."""
        ...
    
    def create_directory(self, path: Path, parents: bool = True, exist_ok: bool = True) -> Path:
        """Create directory with specified options."""
        ...
    
    def check_file_exists(self, path: Path) -> bool:
        """Check if file exists and is a regular file."""
        ...
    
    def get_path_parts(self, path: Path) -> tuple:
        """Get path components as tuple."""
        ...


class StandardFileSystemProvider:
    """
    Standard filesystem provider using pathlib operations.
    
    Implements FileSystemProvider interface for production use with
    comprehensive error handling and logging for test observability.
    """
    
    def resolve_path(self, path: Path) -> Path:
        """Resolve a path to its absolute canonical form with error handling."""
        try:
            resolved = path.resolve()
            logger.debug(f"Resolved path {path} to {resolved}")
            return resolved
        except (OSError, RuntimeError) as e:
            logger.error(f"Failed to resolve path {path}: {e}")
            raise ValueError(f"Cannot resolve path {path}: {e}") from e
    
    def make_relative(self, path: Path, base: Path) -> Path:
        """Make path relative to base directory with enhanced error context."""
        try:
            relative = path.relative_to(base)
            logger.debug(f"Made path {path} relative to {base}: {relative}")
            return relative
        except ValueError as e:
            logger.error(f"Path {path} is not within base directory {base}")
            raise ValueError(f"Path {path} is not within base directory {base}") from e
    
    def is_absolute(self, path: Path) -> bool:
        """Check if path is absolute with debug logging."""
        result = path.is_absolute()
        logger.debug(f"Path {path} is_absolute: {result}")
        return result
    
    def join_paths(self, base: Path, *parts: Union[str, Path]) -> Path:
        """Join path components with validation and logging."""
        try:
            result = base
            for part in parts:
                result = result / part
            logger.debug(f"Joined paths {base} + {parts} = {result}")
            return result
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to join paths {base} + {parts}: {e}")
            raise ValueError(f"Cannot join paths: {e}") from e
    
    def create_directory(self, path: Path, parents: bool = True, exist_ok: bool = True) -> Path:
        """Create directory with comprehensive error handling and logging."""
        try:
            path.mkdir(parents=parents, exist_ok=exist_ok)
            logger.info(f"Created directory: {path} (parents={parents}, exist_ok={exist_ok})")
            return path
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to create directory {path}: {e}")
            raise ValueError(f"Cannot create directory {path}: {e}") from e
    
    def check_file_exists(self, path: Path) -> bool:
        """Check if file exists with debug logging."""
        try:
            result = path.is_file()
            logger.debug(f"File existence check for {path}: {result}")
            return result
        except (OSError, PermissionError) as e:
            logger.warning(f"Cannot check file existence for {path}: {e}")
            return False
    
    def get_path_parts(self, path: Path) -> tuple:
        """Get path components as tuple with logging."""
        try:
            parts = path.parts
            logger.debug(f"Path parts for {path}: {parts}")
            return parts
        except (AttributeError, ValueError) as e:
            logger.error(f"Failed to get path parts for {path}: {e}")
            raise ValueError(f"Cannot get path parts: {e}") from e


# Global default filesystem provider for production use
_default_filesystem_provider = StandardFileSystemProvider()


def normalize_path_separators(path: Union[str, Path]) -> str:
    """
    Normalize path separators in a path string to be consistent across platforms.
    
    This function ensures that path separators are consistent with the current
    operating system, converting forward slashes to backslashes on Windows and
    vice versa on Unix-like systems.
    
    Args:
        path: The path to normalize (string or Path object)
        
    Returns:
        String with normalized path separators
        
    Example:
        >>> # On Unix
        >>> normalize_path_separators('C:\\Windows\\Path')
        'C:/Windows/Path'
        
        >>> # On Windows
        >>> normalize_path_separators('C:/Windows/Path')
        'C:\\Windows\\Path'
    """
    path_str = str(path)
    if os.sep == '\\':
        # On Windows, convert forward slashes to backslashes
        return path_str.replace('/', '\\')
    else:
        # On Unix, convert backslashes to forward slashes
        return path_str.replace('\\', '/')


def resolve_path(
    path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
    fs_provider: Optional[FileSystemProvider] = None
) -> Path:
    """
    Resolve a path to its absolute form with symlinks resolved.
    
    This function provides a high-level interface for path resolution with
    dependency injection support for testing and error handling.
    
    Args:
        path: The path to resolve (can be relative or absolute)
        base_dir: Optional base directory for relative paths
        fs_provider: Optional filesystem provider for dependency injection
        
    Returns:
        Resolved absolute Path object
        
    Raises:
        ValueError: If the path cannot be resolved
        
    Example:
        >>> resolve_path("relative/path", "/base/dir")
        Path("/base/dir/relative/path")
        
        >>> resolve_path("~/file.txt")
        Path("/home/username/file.txt")
    """
    # Use provided filesystem provider or default
    provider = fs_provider or _default_filesystem_provider
    
    try:
        # Convert input to Path if it's a string
        path_obj = Path(path) if isinstance(path, str) else path
        
        # If path is already absolute, just resolve it
        if path_obj.is_absolute():
            return provider.resolve_path(path_obj)
            
        # Handle relative paths with base directory
        if base_dir is not None:
            base_dir_obj = Path(base_dir) if isinstance(base_dir, str) else base_dir
            if not base_dir_obj.is_absolute():
                base_dir_obj = base_dir_obj.resolve()
            path_obj = base_dir_obj / path_obj
        
        # Resolve the final path (handles symlinks, . and ..)
        return provider.resolve_path(path_obj)
        
    except Exception as e:
        error_msg = f"Failed to resolve path '{path}': {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def get_relative_path(
    path: Union[str, Path], 
    base_dir: Union[str, Path],
    *,
    fs_provider: Optional[FileSystemProvider] = None
) -> Path:
    """
    Get a path relative to a base directory with configurable filesystem provider.
    
    Enhanced for testability with dependency injection support and improved error handling.
    
    Args:
        path: Path to convert
        base_dir: Base directory
        fs_provider: Optional filesystem provider for dependency injection (test hook)
        
    Returns:
        Relative path
        
    Raises:
        ValueError: If the path is not within the base directory or filesystem operations fail
    """
    provider = fs_provider or _default_filesystem_provider
    
    try:
        # Log operation start for test observability
        logger.debug(f"Converting path {path} to relative path from base {base_dir}")
        
        # Convert to Path objects and resolve using provider
        path_obj = Path(path)
        base_dir_obj = Path(base_dir)
        
        resolved_path = provider.resolve_path(path_obj)
        resolved_base = provider.resolve_path(base_dir_obj)
        
        # Use provider's relative path calculation
        return provider.make_relative(resolved_path, resolved_base)
        
    except Exception as e:
        # Enhanced error context for debugging
        logger.error(f"get_relative_path failed: path={path}, base_dir={base_dir}, error={e}")
        raise ValueError(f"Failed to get relative path for {path} from {base_dir}: {e}") from e


def get_absolute_path(
    path: Union[str, Path], 
    base_dir: Union[str, Path],
    *,
    fs_provider: Optional[FileSystemProvider] = None
) -> Path:
    """
    Convert a relative path to an absolute path with configurable filesystem provider.
    
    Enhanced for testability with dependency injection support and improved error handling.
    
    Args:
        path: Relative path to convert
        base_dir: Base directory for relative path resolution
        fs_provider: Optional filesystem provider for dependency injection (test hook)
        
    Returns:
        Absolute path
        
    Raises:
        ValueError: If filesystem operations fail
    """
    provider = fs_provider or _default_filesystem_provider
    
    try:
        # Log operation start for test observability
        logger.debug(f"Converting path {path} to absolute path with base {base_dir}")
        
        # Convert to Path objects
        path_obj = Path(path)
        base_dir_obj = Path(base_dir)
        
        # Check if already absolute using provider
        if provider.is_absolute(path_obj):
            logger.debug(f"Path {path} is already absolute")
            return provider.resolve_path(path_obj)
        
        # Join with base directory and resolve
        joined = provider.join_paths(base_dir_obj, path_obj)
        return provider.resolve_path(joined)
        
    except Exception as e:
        # Enhanced error context for debugging
        logger.error(f"get_absolute_path failed: path={path}, base_dir={base_dir}, error={e}")
        raise ValueError(f"Failed to get absolute path for {path} with base {base_dir}: {e}") from e


def find_common_base_directory(
    paths: List[Union[str, Path]],
    *,
    fs_provider: Optional[FileSystemProvider] = None
) -> Optional[Path]:
    """
    Find the common base directory for a list of paths with configurable filesystem provider.
    
    Enhanced for testability with dependency injection support and improved error handling.
    
    Args:
        paths: List of paths to analyze
        fs_provider: Optional filesystem provider for dependency injection (test hook)
        
    Returns:
        Common base directory or None if no common base can be found
        
    Raises:
        ValueError: If filesystem operations fail
    """
    provider = fs_provider or _default_filesystem_provider
    
    try:
        # Log operation start for test observability
        logger.debug(f"Finding common base directory for {len(paths)} paths")
        
        if not paths:
            logger.debug("Empty paths list, returning None")
            return None

        # Convert all paths to Path objects and resolve them using provider
        resolved_paths = []
        for path in paths:
            path_obj = Path(path)
            resolved = provider.resolve_path(path_obj)
            resolved_paths.append(resolved)
            logger.debug(f"Resolved path {path} to {resolved}")

        # Get path parts using provider
        parts_list = [provider.get_path_parts(path) for path in resolved_paths]
        min_length = min(len(parts) for parts in parts_list)

        # Find the common prefix
        common_parts = []
        for i in range(min_length):
            if all(parts[i] == parts_list[0][i] for parts in parts_list):
                common_parts.append(parts_list[0][i])
            else:
                break

        result = Path(*common_parts) if common_parts else None
        logger.debug(f"Common base directory found: {result}")
        return result
        
    except Exception as e:
        # Enhanced error context for debugging
        logger.error(f"find_common_base_directory failed: paths={paths}, error={e}")
        raise ValueError(f"Failed to find common base directory: {e}") from e


def ensure_directory_exists(
    path: Union[str, Path],
    *,
    fs_provider: Optional[FileSystemProvider] = None,
    parents: bool = True,
    exist_ok: bool = True
) -> Path:
    """
    Ensure that a directory exists with configurable filesystem provider.
    
    Enhanced for testability with dependency injection support and improved error handling.
    
    Args:
        path: Directory path to create
        fs_provider: Optional filesystem provider for dependency injection (test hook)
        parents: If True, create parent directories as needed
        exist_ok: If True, don't raise error if directory already exists
        
    Returns:
        Path to the directory
        
    Raises:
        ValueError: If directory creation fails
    """
    provider = fs_provider or _default_filesystem_provider
    
    try:
        # Log operation start for test observability
        logger.debug(f"Ensuring directory exists: {path} (parents={parents}, exist_ok={exist_ok})")
        
        # Convert to Path object
        path_obj = Path(path)
        
        # Use provider to create directory
        return provider.create_directory(path_obj, parents=parents, exist_ok=exist_ok)
        
    except Exception as e:
        # Enhanced error context for debugging
        logger.error(f"ensure_directory_exists failed: path={path}, error={e}")
        raise ValueError(f"Failed to ensure directory exists at {path}: {e}") from e


def ensure_directory(directory: Path, exist_ok: bool = True) -> None:
    """Ensure a directory exists, creating it if necessary.

    Args:
        directory (Path): The directory path to create if missing.
        exist_ok (bool, optional): If ``True`` (default), no exception is raised
            when the directory already exists.

    Raises:
        PermissionError: If there are insufficient permissions to create the
            directory.
        FileExistsError: If ``exist_ok`` is ``False`` and the directory already
            exists.
        OSError: For other filesystem-related errors.

    Example:
        >>> from pathlib import Path
        >>> ensure_directory(Path('output'))
    """

    try:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=exist_ok)
        logger.debug(f"Directory ensured: {dir_path} (exist_ok={exist_ok})")
    except PermissionError as e:
        logger.error(f"Permission denied creating directory {dir_path}: {e}")
        raise
    except FileExistsError as e:
        logger.error(f"Directory already exists at {dir_path}: {e}")
        raise
    except OSError as e:
        logger.error(f"Error creating directory {dir_path}: {e}")
        raise


def check_file_exists(
    file_path: Union[str, Path],
    *,
    fs_provider: Optional[FileSystemProvider] = None
) -> bool:
    """
    Check if a file exists with configurable filesystem provider.
    
    Enhanced for testability with dependency injection support and improved error handling.
    
    Args:
        file_path: Path to the file to check
        fs_provider: Optional filesystem provider for dependency injection (test hook)
        
    Returns:
        True if the file exists and is a regular file, False otherwise
        
    Note:
        This function returns False for directories and other non-regular files.
        Filesystem errors are logged but don't raise exceptions.
    """
    provider = fs_provider or _default_filesystem_provider
    
    try:
        # Log operation start for test observability
        logger.debug(f"Checking file existence: {file_path}")
        
        # Convert to Path object
        path_obj = Path(file_path)
        
        # Use provider to check file existence
        return provider.check_file_exists(path_obj)
        
    except Exception as e:
        # Enhanced error logging but don't raise exception for existence checks
        logger.warning(f"check_file_exists failed for {file_path}: {e}")
        return False


# Test-specific utility functions for enhanced testability

def set_filesystem_provider_for_testing(provider: FileSystemProvider) -> FileSystemProvider:
    """
    Set a custom filesystem provider for testing scenarios.
    
    This is a test-specific entry point that allows complete control over
    filesystem operations during testing through dependency injection.
    
    Args:
        provider: Custom filesystem provider implementing FileSystemProvider protocol
        
    Returns:
        Previously active filesystem provider (for restoration)
        
    Note:
        This function is intended for test use only. Production code should not
        call this function as it modifies global state.
    """
    global _default_filesystem_provider
    previous_provider = _default_filesystem_provider
    _default_filesystem_provider = provider
    logger.info(f"Filesystem provider changed for testing: {type(provider).__name__}")
    return previous_provider


def restore_filesystem_provider(provider: FileSystemProvider) -> None:
    """
    Restore a previous filesystem provider after testing.
    
    This is a test-specific entry point that restores filesystem provider
    state after test completion.
    
    Args:
        provider: Previously active filesystem provider to restore
        
    Note:
        This function is intended for test use only. Production code should not
        call this function as it modifies global state.
    """
    global _default_filesystem_provider
    _default_filesystem_provider = provider
    logger.info(f"Filesystem provider restored after testing: {type(provider).__name__}")


def get_current_filesystem_provider() -> FileSystemProvider:
    """
    Get the currently active filesystem provider.
    
    This is a test-specific entry point that allows tests to inspect
    the current filesystem provider configuration.
    
    Returns:
        Currently active filesystem provider
        
    Note:
        This function is primarily intended for test verification.
    """
    return _default_filesystem_provider