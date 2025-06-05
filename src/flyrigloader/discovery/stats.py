"""
File statistics functionality.

Utilities for collecting and working with file system metadata.
Enhanced with dependency injection patterns for improved testability.
"""
from typing import Dict, Any, List, Union, Optional, Protocol, Callable
from pathlib import Path
from datetime import datetime
import os
import stat
from abc import ABC, abstractmethod


class FileSystemProvider(Protocol):
    """Protocol for filesystem operations to enable dependency injection."""
    
    def path_exists(self, path: Path) -> bool:
        """Check if path exists."""
        ...
    
    def get_stat(self, path: Path) -> os.stat_result:
        """Get file statistics."""
        ...
    
    def check_access(self, path: Path, mode: int) -> bool:
        """Check file access permissions."""
        ...
    
    def is_dir(self, path: Path) -> bool:
        """Check if path is a directory."""
        ...
    
    def is_file(self, path: Path) -> bool:
        """Check if path is a file."""
        ...
    
    def is_symlink(self, path: Path) -> bool:
        """Check if path is a symbolic link."""
        ...


class TimestampProvider(Protocol):
    """Protocol for timestamp handling to enable cross-platform testing."""
    
    def get_creation_timestamp(self, stat_result: os.stat_result) -> float:
        """Extract creation timestamp from stat result."""
        ...
    
    def timestamp_to_datetime(self, timestamp: float) -> datetime:
        """Convert timestamp to datetime object."""
        ...


class DefaultFileSystemProvider:
    """Default implementation using standard OS operations."""
    
    def path_exists(self, path: Path) -> bool:
        """Check if path exists using Path.exists()."""
        return path.exists()
    
    def get_stat(self, path: Path) -> os.stat_result:
        """Get file statistics using Path.stat()."""
        return path.stat()
    
    def check_access(self, path: Path, mode: int) -> bool:
        """Check file access permissions using os.access()."""
        return os.access(path, mode)
    
    def is_dir(self, path: Path) -> bool:
        """Check if path is a directory using Path.is_dir()."""
        return path.is_dir()
    
    def is_file(self, path: Path) -> bool:
        """Check if path is a file using Path.is_file()."""
        return path.is_file()
    
    def is_symlink(self, path: Path) -> bool:
        """Check if path is a symbolic link using Path.is_symlink()."""
        return path.is_symlink()


class DefaultTimestampProvider:
    """Default implementation for platform-specific timestamp handling."""
    
    def get_creation_timestamp(self, stat_result: os.stat_result) -> float:
        """
        Extract creation timestamp from stat result.
        
        Uses st_birthtime on systems that support it (macOS), 
        falls back to st_ctime on other systems (Linux, Windows).
        """
        return getattr(stat_result, "st_birthtime", stat_result.st_ctime)
    
    def timestamp_to_datetime(self, timestamp: float) -> datetime:
        """Convert timestamp to datetime object."""
        return datetime.fromtimestamp(timestamp)


class FileStatsError(Exception):
    """Enhanced exception for file statistics operations with structured context."""
    
    def __init__(self, message: str, path: Union[str, Path], context: Optional[Dict[str, Any]] = None):
        """
        Initialize FileStatsError with structured context.
        
        Args:
            message: Error description
            path: File path that caused the error
            context: Additional context for debugging
        """
        super().__init__(message)
        self.path = str(path)
        self.context = context or {}
        self.context.update({
            "path": self.path,
            "error_type": self.__class__.__name__
        })


def get_file_stats(
    path: Union[str, Path],
    fs_provider: Optional[FileSystemProvider] = None,
    timestamp_provider: Optional[TimestampProvider] = None,
    enable_test_hooks: bool = False
) -> Dict[str, Any]:
    """
    Get file statistics for a given file path with configurable providers.
    
    Enhanced with dependency injection patterns to support pytest.monkeypatch scenarios
    for os.stat and os.access operations per F-016 requirements.
    
    Args:
        path: Path to the file
        fs_provider: Optional filesystem provider for dependency injection
        timestamp_provider: Optional timestamp provider for cross-platform testing
        enable_test_hooks: Enable test-specific entry points for controlled testing
        
    Returns:
        Dictionary with comprehensive file stats including size, modification time,
        creation time, permissions, and other metadata.
        
    Raises:
        FileStatsError: Enhanced exception with structured context for better test observability
    """
    # Use default providers if none specified
    if fs_provider is None:
        fs_provider = DefaultFileSystemProvider()
    if timestamp_provider is None:
        timestamp_provider = DefaultTimestampProvider()
    
    path = Path(path)
    
    # Enhanced error handling with structured exception context
    try:
        if not fs_provider.path_exists(path):
            raise FileStatsError(
                f"File not found: {path}",
                path,
                context={
                    "operation": "path_exists_check",
                    "fs_provider": fs_provider.__class__.__name__,
                    "test_hooks_enabled": enable_test_hooks
                }
            )
        
        # Get file statistics through configurable provider
        stats = fs_provider.get_stat(path)
        
        # Platform-specific timestamp handling through dependency injection
        creation_timestamp = timestamp_provider.get_creation_timestamp(stats)
        
        # Collect file statistics with error context
        result = {
            "size": stats.st_size,
            "size_bytes": stats.st_size,  # Alias for backward compatibility
            "mtime": timestamp_provider.timestamp_to_datetime(stats.st_mtime),
            "modified_time": stats.st_mtime,  # Timestamp version
            "creation_time": timestamp_provider.timestamp_to_datetime(creation_timestamp),
            "ctime": timestamp_provider.timestamp_to_datetime(stats.st_ctime),
            "created_time": stats.st_ctime,  # Timestamp version
            "is_directory": fs_provider.is_dir(path),
            "is_file": fs_provider.is_file(path),
            "is_symlink": fs_provider.is_symlink(path),
            "filename": path.name,
            "extension": path.suffix[1:] if path.suffix else "",
            "permissions": stats.st_mode & 0o777,
            "is_readable": fs_provider.check_access(path, os.R_OK),
            "is_writable": fs_provider.check_access(path, os.W_OK),
            "is_executable": fs_provider.check_access(path, os.X_OK)
        }
        
        # Add test-specific metadata when test hooks are enabled
        if enable_test_hooks:
            result.update({
                "_test_metadata": {
                    "fs_provider": fs_provider.__class__.__name__,
                    "timestamp_provider": timestamp_provider.__class__.__name__,
                    "stat_mode": stats.st_mode,
                    "stat_uid": stats.st_uid,
                    "stat_gid": stats.st_gid
                }
            })
        
        return result
        
    except (OSError, IOError) as e:
        # Wrap OS errors with structured context
        raise FileStatsError(
            f"Failed to get file statistics for {path}: {e}",
            path,
            context={
                "operation": "file_stat_collection",
                "original_error": str(e),
                "error_code": getattr(e, "errno", None),
                "fs_provider": fs_provider.__class__.__name__,
                "timestamp_provider": timestamp_provider.__class__.__name__,
                "test_hooks_enabled": enable_test_hooks
            }
        ) from e


def attach_file_stats(
    file_data: Union[List[str], Dict[str, Dict[str, Any]]],
    fs_provider: Optional[FileSystemProvider] = None,
    timestamp_provider: Optional[TimestampProvider] = None,
    enable_test_hooks: bool = False,
    error_handler: Optional[Callable[[Exception, str], Any]] = None,
    batch_size: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Attach file statistics to discovery results with configurable providers.
    
    Enhanced with test-specific entry points allowing controlled statistics collection
    during test execution per TST-REF-003 requirements.
    
    Args:
        file_data: Either a list of file paths or a dictionary mapping file paths to metadata
        fs_provider: Optional filesystem provider for dependency injection
        timestamp_provider: Optional timestamp provider for cross-platform testing
        enable_test_hooks: Enable test-specific entry points for controlled testing
        error_handler: Optional custom error handler for failed file stat operations
        batch_size: Optional batch size for processing large file lists (for testing)
        
    Returns:
        Dictionary mapping file paths to metadata including file statistics
        
    Raises:
        FileStatsError: Enhanced exception with batch processing context
    """
    def default_error_handler(error: Exception, file_path: str) -> Dict[str, Any]:
        """Default error handler that re-raises with additional context."""
        if isinstance(error, FileStatsError):
            # Add batch processing context to existing error
            error.context.update({
                "batch_processing": True,
                "current_file": file_path,
                "total_files": len(file_paths) if isinstance(file_data, list) else len(file_data)
            })
            raise error
        else:
            # Wrap other exceptions in FileStatsError
            raise FileStatsError(
                f"Unexpected error processing file {file_path}: {error}",
                file_path,
                context={
                    "batch_processing": True,
                    "original_error_type": error.__class__.__name__,
                    "total_files": len(file_paths) if isinstance(file_data, list) else len(file_data)
                }
            ) from error
    
    # Use default error handler if none provided
    if error_handler is None:
        error_handler = default_error_handler
    
    # Extract file paths based on input type
    if isinstance(file_data, list):
        file_paths = file_data
        existing_metadata = {}
    else:
        file_paths = list(file_data.keys())
        existing_metadata = file_data
    
    result = {}
    processed_count = 0
    
    # Process files in batches if batch_size is specified (useful for testing)
    if batch_size is not None and enable_test_hooks:
        batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]
    else:
        batches = [file_paths]  # Single batch
    
    for batch_index, batch in enumerate(batches):
        for file_path in batch:
            try:
                # Get file statistics with configurable providers
                file_stats = get_file_stats(
                    file_path,
                    fs_provider=fs_provider,
                    timestamp_provider=timestamp_provider,
                    enable_test_hooks=enable_test_hooks
                )
                
                # Merge with existing metadata if available
                if file_path in existing_metadata:
                    result[file_path] = {**existing_metadata[file_path], **file_stats}
                else:
                    result[file_path] = file_stats
                
                processed_count += 1
                
                # Add batch processing metadata when test hooks are enabled
                if enable_test_hooks:
                    result[file_path].setdefault("_test_metadata", {}).update({
                        "batch_index": batch_index,
                        "batch_size": len(batch),
                        "processed_order": processed_count,
                        "total_batches": len(batches)
                    })
                    
            except Exception as e:
                # Use custom error handler for processing errors
                try:
                    fallback_result = error_handler(e, file_path)
                    if fallback_result is not None:
                        result[file_path] = fallback_result
                except Exception:
                    # If error handler also fails, propagate the original error
                    raise e
    
    return result


# Modular interface for statistics collection to reduce coupling
class FileStatsCollector:
    """
    Modular file statistics collection interface reducing coupling with filesystem operations.
    
    Implements modular file statistics collection interfaces reducing coupling with filesystem 
    operations per TST-REF-002 requirements.
    """
    
    def __init__(
        self,
        fs_provider: Optional[FileSystemProvider] = None,
        timestamp_provider: Optional[TimestampProvider] = None
    ):
        """
        Initialize FileStatsCollector with configurable providers.
        
        Args:
            fs_provider: Filesystem provider for dependency injection
            timestamp_provider: Timestamp provider for cross-platform compatibility
        """
        self.fs_provider = fs_provider or DefaultFileSystemProvider()
        self.timestamp_provider = timestamp_provider or DefaultTimestampProvider()
    
    def collect_single_file_stats(
        self, 
        path: Union[str, Path], 
        enable_test_hooks: bool = False
    ) -> Dict[str, Any]:
        """
        Collect statistics for a single file.
        
        Args:
            path: File path to analyze
            enable_test_hooks: Enable test-specific metadata collection
            
        Returns:
            File statistics dictionary
        """
        return get_file_stats(
            path,
            fs_provider=self.fs_provider,
            timestamp_provider=self.timestamp_provider,
            enable_test_hooks=enable_test_hooks
        )
    
    def collect_batch_stats(
        self,
        file_data: Union[List[str], Dict[str, Dict[str, Any]]],
        enable_test_hooks: bool = False,
        error_handler: Optional[Callable[[Exception, str], Any]] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Collect statistics for multiple files.
        
        Args:
            file_data: Files to process
            enable_test_hooks: Enable test-specific metadata collection
            error_handler: Custom error handler for failed operations
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping file paths to statistics
        """
        return attach_file_stats(
            file_data,
            fs_provider=self.fs_provider,
            timestamp_provider=self.timestamp_provider,
            enable_test_hooks=enable_test_hooks,
            error_handler=error_handler,
            batch_size=batch_size
        )