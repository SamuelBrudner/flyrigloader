"""
Tests for path utilities.

Tests the functionality for manipulating and converting file paths.
"""
import os
import tempfile
import sys
from pathlib import Path

import pytest

from flyrigloader.utils.paths import (
    get_relative_path,
    get_absolute_path,
    find_common_base_directory,
    ensure_directory_exists
)


def test_get_relative_path():
    """Test getting a relative path from an absolute path."""
    # Test with string paths
    base_dir = "/home/user/data"
    path = "/home/user/data/subdir/file.txt"
    rel_path = get_relative_path(path, base_dir)
    assert str(rel_path) == "subdir/file.txt"
    
    # Test with Path objects
    base_dir = Path("/home/user/data")
    path = Path("/home/user/data/subdir/file.txt")
    rel_path = get_relative_path(path, base_dir)
    assert rel_path == Path("subdir/file.txt")
    
    # Test with relative path input (should be resolved first)
    base_dir = Path("/home/user/data").resolve()
    path = Path("/home/user/data/subdir/../other/file.txt").resolve()
    rel_path = get_relative_path(path, base_dir)
    assert rel_path == Path("other/file.txt")
    
    # Test with path not in base_dir
    base_dir = "/home/user/data"
    path = "/home/otheruser/data/file.txt"
    with pytest.raises(ValueError):
        get_relative_path(path, base_dir)


def test_get_absolute_path():
    """Test getting an absolute path from a relative path."""
    # Create a temporary directory to use real paths
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with relative path
        rel_path = "subdir/file.txt"
        abs_path = get_absolute_path(rel_path, temp_dir)
        expected_path = Path(temp_dir) / rel_path
        assert abs_path.resolve() == expected_path.resolve()
        
        # Test with Path objects
        rel_path = Path("other/file.txt")
        abs_path = get_absolute_path(rel_path, Path(temp_dir))
        expected_path = Path(temp_dir) / rel_path
        assert abs_path.resolve() == expected_path.resolve()
        
        # Test with already absolute path (should be returned as is)
        existing_path = Path(temp_dir) / "absolute.txt"
        with open(existing_path, 'w') as f:
            f.write("test")
        abs_path = get_absolute_path(str(existing_path), temp_dir)
        assert abs_path.resolve() == existing_path.resolve()


def test_find_common_base_directory():
    """Test finding the common base directory for a list of paths."""
    # Test with temporary directory to use real paths
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a directory structure
        temp_base = Path(temp_dir) / "project1"
        temp_base.mkdir()
        (temp_base / "subdir").mkdir()
        (temp_base / "other").mkdir()
        
        paths = [
            str(temp_base / "file1.txt"),
            str(temp_base / "subdir/file2.txt"),
            str(temp_base / "other/file3.txt")
        ]
        
        # Create the files
        for path in paths:
            with open(path, 'w') as f:
                f.write("test")
                
        common_base = find_common_base_directory(paths)
        assert common_base.resolve() == temp_base.resolve()
        
    # Test with partial common base
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create directory structure
        base_dir = Path(temp_dir)
        (base_dir / "project1").mkdir()
        (base_dir / "project2").mkdir()
        (base_dir / "project3").mkdir()
        
        file1 = base_dir / "project1/file1.txt"
        file2 = base_dir / "project2/file2.txt"
        file3 = base_dir / "project3/file3.txt"
        
        for file_path in [file1, file2, file3]:
            with open(file_path, 'w') as f:
                f.write("test")
                
        paths = [str(file1), str(file2), str(file3)]
        common_base = find_common_base_directory(paths)
        assert common_base.resolve() == base_dir.resolve()
        
    # Test with empty list
    assert find_common_base_directory([]) is None


def test_ensure_directory_exists():
    """Test ensuring a directory exists."""
    # Test with temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = os.path.join(temp_dir, "test_dir")
        
        # Directory should not exist yet
        assert not os.path.exists(test_dir)
        
        # Ensure directory exists
        result = ensure_directory_exists(test_dir)
        
        # Directory should now exist
        assert os.path.exists(test_dir)
        assert os.path.isdir(test_dir)
        assert result == Path(test_dir)
        
        # Test with nested directories
        nested_dir = os.path.join(test_dir, "nested1", "nested2")
        
        # Nested directories should not exist yet
        assert not os.path.exists(nested_dir)
        
        # Ensure nested directories exist
        result = ensure_directory_exists(nested_dir)
        
        # Nested directories should now exist
        assert os.path.exists(nested_dir)
        assert os.path.isdir(nested_dir)
        assert result == Path(nested_dir)
        
        # Test with existing directory (should not raise an error)
        result = ensure_directory_exists(test_dir)
        assert result == Path(test_dir)
