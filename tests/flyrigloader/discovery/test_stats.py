"""
Tests for file statistics functionality.

Tests the functionality for collecting and working with file system metadata.
"""

import os
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import time
import itertools
from typing import List, Dict, Any, Callable

import pytest

from flyrigloader.discovery.stats import (
    get_file_stats,
    attach_file_stats,
)
from flyrigloader.discovery.files import discover_files


def test_get_file_stats():
    """Test getting file statistics."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"test content")
        tmp_path = tmp.name

    try:
        verify_file_stats(tmp_path)
    finally:
        # Clean up
        safe_remove_file(tmp_path)


def verify_file_stats(file_path: str) -> None:
    """Verify file statistics properties for a given file."""
    # Get file stats
    stats = get_file_stats(file_path)

    # Verify the stats dictionary has the expected keys
    assert "size" in stats
    assert "mtime" in stats
    assert "ctime" in stats
    assert "creation_time" in stats

    # Verify the values are of expected types
    assert isinstance(stats["size"], int)
    assert isinstance(stats["mtime"], datetime)
    assert isinstance(stats["ctime"], datetime)
    assert isinstance(stats["creation_time"], datetime)

    # Optional check: creation time is close to current time
    assert abs((stats["creation_time"] - datetime.now()).total_seconds()) < 5

    # Verify the size is correct
    assert stats["size"] == 12  # "test content" is 12 bytes


def safe_remove_file(file_path: str) -> None:
    """Safely remove a file if it exists."""
    if os.path.exists(file_path):
        os.unlink(file_path)


def test_get_file_stats_nonexistent_file():
    """Test getting stats for a nonexistent file raises an error."""
    # Use a path that definitely doesn't exist
    nonexistent_path = "/path/that/definitely/does/not/exist/12345.txt"
    
    # Verify that it raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        get_file_stats(nonexistent_path)


def create_temp_files(num_files: int, content_template: str = "content {}") -> List[str]:
    """Create temporary test files with given content template."""
    temp_files = []
    for i in range(num_files):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(content_template.format(i).encode())
            temp_files.append(tmp.name)
    return temp_files


def create_temp_files_in_dir(dir_path: str, num_files: int, name_template: str = "file_{}.txt", 
                         content_template: str = "content {}") -> List[str]:
    """Create temporary files in a specified directory."""
    file_paths = []
    for i in range(num_files):
        file_path = os.path.join(dir_path, name_template.format(i))
        with open(file_path, 'w') as f:
            f.write(content_template.format(i))
        file_paths.append(file_path)
    return file_paths


def safe_remove_files(file_paths: List[str]) -> None:
    """Safely remove multiple files."""
    for file_path in file_paths:
        safe_remove_file(file_path)


def verify_stats_keys_in_result(result: Dict[str, Any], file_paths: List[str]) -> None:
    """Verify that file stats keys exist in the result dictionary."""
    assert isinstance(result, dict)
    assert len(result) == len(file_paths)
    
    for file_path in file_paths:
        assert file_path in result
        assert "size" in result[file_path]
        assert "mtime" in result[file_path]
        assert "ctime" in result[file_path]
        assert "creation_time" in result[file_path]


def verify_all_files_have_stats(result: Dict[str, Dict[str, Any]]) -> None:
    """Verify that all files in the result have stats fields."""
    for file_data in result.values():
        assert "size" in file_data
        assert "mtime" in file_data
        assert "ctime" in file_data
        assert "creation_time" in file_data


def verify_metadata_preserved(result: Dict[str, Dict[str, Any]], file_paths: List[str], 
                             expected_keys: List[str]) -> None:
    """Verify that metadata keys are preserved in the result."""
    for file_path, key in itertools.product(file_paths, expected_keys):
        assert key in result[file_path], f"Key '{key}' missing from metadata for {file_path}"


def test_attach_file_stats_list():
    """Test attaching file stats to a list of files."""
    # Create temporary files
    temp_files = create_temp_files(3)
    
    try:
        # Attach file stats
        result = attach_file_stats(temp_files)
        
        # Verify the result
        verify_stats_keys_in_result(result, temp_files)
    finally:
        # Clean up
        safe_remove_files(temp_files)


def create_metadata_dict(file_paths: List[str], metadata_creator: Callable[[int], Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Create a metadata dictionary for testing."""
    return {
        file_path: metadata_creator(i) 
        for i, file_path in enumerate(file_paths)
    }


def test_attach_file_stats_dict():
    """Test attaching file stats to a dict with metadata."""
    # Create temporary files
    temp_files = create_temp_files(3)
    
    try:
        # Create a dict with metadata
        file_data = create_metadata_dict(
            temp_files, 
            lambda i: {"type": "text", "index": i}
        )
        
        # Attach file stats
        result = attach_file_stats(file_data)
        
        # Verify the result
        verify_stats_keys_in_result(result, temp_files)
        
        # Verify original metadata is preserved
        verify_metadata_preserved(result, temp_files, ["type", "index"])
    finally:
        # Clean up
        safe_remove_files(temp_files)


def test_integration_with_discover_files():
    """Test integration with discover_files function."""
    # Create a temporary directory with files
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create files in the temporary directory
        temp_files = create_temp_files_in_dir(temp_dir, 3)
        
        # Use discover_files with include_stats
        result = discover_files(temp_dir, "*.txt", include_stats=True)
        
        # Verify the result has file stats
        assert isinstance(result, dict)
        assert len(result) == 3
        verify_all_files_have_stats(result)
    finally:
        # Clean up
        safe_remove_files(temp_files)
        safe_remove_directory(temp_dir)


def safe_remove_directory(dir_path: str) -> None:
    """Safely remove a directory if it exists."""
    if os.path.exists(dir_path):
        os.rmdir(dir_path)
