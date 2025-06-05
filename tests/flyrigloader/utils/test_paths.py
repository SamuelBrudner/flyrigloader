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


class TestEnsureDirectoryExistsComprehensive:
    """Comprehensive test suite for ensure_directory_exists with edge cases."""

    @pytest.mark.parametrize("input_type", ["string", "path_object"])
    def test_ensure_directory_exists_basic(self, input_type):
        """Test basic directory creation with different input types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir_path = Path(temp_dir) / "new_directory"
            test_input = str(test_dir_path) if input_type == "string" else test_dir_path
            
            # Directory should not exist initially
            assert not test_dir_path.exists()
            
            result = ensure_directory_exists(test_input)
            
            # Verify directory was created
            assert test_dir_path.exists()
            assert test_dir_path.is_dir()
            assert result == test_dir_path

    def test_ensure_directory_exists_idempotent(self):
        """Test that ensure_directory_exists is idempotent (can be called multiple times)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "idempotent_test"
            
            # Create directory first time
            result1 = ensure_directory_exists(test_dir)
            assert test_dir.exists()
            assert test_dir.is_dir()
            
            # Call again - should not raise error
            result2 = ensure_directory_exists(test_dir)
            assert result1 == result2
            assert test_dir.exists()
            assert test_dir.is_dir()

    @pytest.mark.parametrize("depth", [1, 3, 5, 10])
    def test_ensure_directory_exists_nested_creation(self, depth):
        """Test creation of deeply nested directory structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested path with specified depth
            nested_parts = [f"level_{i}" for i in range(depth)]
            nested_path = Path(temp_dir).joinpath(*nested_parts)
            
            # Verify none of the nested directories exist
            current_path = Path(temp_dir)
            for part in nested_parts:
                current_path = current_path / part
                assert not current_path.exists()
            
            result = ensure_directory_exists(nested_path)
            
            # Verify all nested directories were created
            assert nested_path.exists()
            assert nested_path.is_dir()
            assert result == nested_path
            
            # Verify all intermediate directories exist
            current_path = Path(temp_dir)
            for part in nested_parts:
                current_path = current_path / part
                assert current_path.exists()
                assert current_path.is_dir()

    def test_ensure_directory_exists_special_characters(self):
        """Test directory creation with special characters in path names."""
        special_names = [
            "dir with spaces",
            "dir-with-dashes",
            "dir_with_underscores",
            "dir.with.dots",
            "UPPERCASE_DIR",
            "MixedCase_Dir"
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for special_name in special_names:
                test_dir = Path(temp_dir) / special_name
                
                result = ensure_directory_exists(test_dir)
                
                assert test_dir.exists()
                assert test_dir.is_dir()
                assert result == test_dir

    def test_ensure_directory_exists_unicode_characters(self):
        """Test directory creation with unicode characters."""
        unicode_names = [
            "директория",  # Cyrillic
            "目录",        # Chinese
            "ディレクトリ",  # Japanese
            "φάκελος"      # Greek
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for unicode_name in unicode_names:
                test_dir = Path(temp_dir) / unicode_name
                
                try:
                    result = ensure_directory_exists(test_dir)
                    assert test_dir.exists()
                    assert test_dir.is_dir()
                    assert result == test_dir
                except (UnicodeError, OSError):
                    # Skip if filesystem doesn't support unicode
                    pytest.skip(f"Filesystem doesn't support unicode directory: {unicode_name}")

    def test_ensure_directory_exists_long_paths(self):
        """Test directory creation with very long path names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a very long directory name (but within filesystem limits)
            long_name = "very_long_directory_name_" * 10  # Should be manageable on most systems
            test_dir = Path(temp_dir) / long_name
            
            try:
                result = ensure_directory_exists(test_dir)
                assert test_dir.exists()
                assert test_dir.is_dir()
                assert result == test_dir
            except OSError:
                # Skip if path is too long for filesystem
                pytest.skip("Path too long for filesystem")

    def test_ensure_directory_exists_existing_file_conflict(self):
        """Test behavior when path exists but is a file, not a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file at the target path
            file_path = Path(temp_dir) / "conflicting_file"
            file_path.touch()
            
            # Attempt to create directory with same name should raise error
            with pytest.raises(FileExistsError):
                ensure_directory_exists(file_path)

    def test_ensure_directory_exists_permission_handling(self):
        """Test directory creation with permission considerations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            
            # Test normal directory creation (should work)
            normal_dir = base_path / "normal_directory"
            result = ensure_directory_exists(normal_dir)
            assert normal_dir.exists()
            assert result == normal_dir
            
            # Test creating subdirectory in existing directory
            sub_dir = normal_dir / "subdirectory"
            result = ensure_directory_exists(sub_dir)
            assert sub_dir.exists()
            assert result == sub_dir

    @pytest.mark.parametrize("parent_exists", [True, False])
    def test_ensure_directory_exists_parent_scenarios(self, parent_exists):
        """Test directory creation when parent directory may or may not exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            parent_dir = Path(temp_dir) / "parent"
            child_dir = parent_dir / "child"
            
            if parent_exists:
                parent_dir.mkdir()
            
            result = ensure_directory_exists(child_dir)
            
            assert parent_dir.exists()
            assert child_dir.exists()
            assert child_dir.is_dir()
            assert result == child_dir

    def test_ensure_directory_exists_relative_paths(self):
        """Test directory creation with relative paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                # Test relative path creation
                relative_dir = Path("relative_directory")
                result = ensure_directory_exists(relative_dir)
                
                assert relative_dir.exists()
                assert relative_dir.is_dir()
                assert result == relative_dir
                
            finally:
                os.chdir(original_cwd)

    def test_ensure_directory_exists_dot_segments(self):
        """Test directory creation with paths containing dot segments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            
            # Create intermediate directory
            intermediate = base_path / "intermediate"
            intermediate.mkdir()
            
            # Test path with dot segments
            dot_path = base_path / "intermediate" / ".." / "final_directory"
            result = ensure_directory_exists(dot_path)
            
            # Should resolve to base_path / "final_directory"
            expected = base_path / "final_directory"
            assert expected.exists()
            assert expected.is_dir()
            assert result.resolve() == expected.resolve()

    def test_ensure_directory_exists_concurrent_creation(self):
        """Test directory creation under concurrent conditions (simulate race conditions)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "concurrent_test"
            
            # Simulate concurrent creation by calling multiple times rapidly
            results = []
            for _ in range(10):
                result = ensure_directory_exists(test_dir)
                results.append(result)
            
            # All results should be the same
            assert all(r == test_dir for r in results)
            assert test_dir.exists()
            assert test_dir.is_dir()


class TestCheckFileExistsComprehensive:
    """Comprehensive test suite for check_file_exists with edge cases."""

    def test_check_file_exists_basic_functionality(self):
        """Test basic file existence checking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_file.txt"
            
            # File doesn't exist initially
            assert not check_file_exists(file_path)
            assert not check_file_exists(str(file_path))
            
            # Create file
            file_path.touch()
            
            # File exists now
            assert check_file_exists(file_path)
            assert check_file_exists(str(file_path))

    @pytest.mark.parametrize("input_type", ["string", "path_object"])
    def test_check_file_exists_input_types(self, input_type):
        """Test file existence checking with different input types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_file.txt"
            file_path.touch()
            
            test_input = str(file_path) if input_type == "string" else file_path
            
            assert check_file_exists(test_input) is True

    def test_check_file_exists_directory_vs_file(self):
        """Test that check_file_exists correctly distinguishes files from directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a directory
            dir_path = Path(temp_dir) / "test_directory"
            dir_path.mkdir()
            
            # Create a file
            file_path = Path(temp_dir) / "test_file.txt"
            file_path.touch()
            
            # Directory should return False (not a file)
            assert check_file_exists(dir_path) is False
            
            # File should return True
            assert check_file_exists(file_path) is True

    def test_check_file_exists_nonexistent_paths(self):
        """Test file existence checking for various non-existent paths."""
        nonexistent_paths = [
            "/path/that/does/not/exist.txt",
            "relative/nonexistent/file.txt",
            "simply_nonexistent.txt"
        ]
        
        for path in nonexistent_paths:
            assert check_file_exists(path) is False

    def test_check_file_exists_special_characters(self):
        """Test file existence checking with special characters in filenames."""
        special_filenames = [
            "file with spaces.txt",
            "file-with-dashes.txt",
            "file_with_underscores.txt",
            "file.with.multiple.dots.txt",
            "UPPERCASE_FILE.TXT",
            "MixedCase_File.txt"
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for filename in special_filenames:
                file_path = Path(temp_dir) / filename
                
                # Initially doesn't exist
                assert check_file_exists(file_path) is False
                
                # Create file
                file_path.touch()
                
                # Now exists
                assert check_file_exists(file_path) is True

    def test_check_file_exists_unicode_filenames(self):
        """Test file existence checking with unicode characters in filenames."""
        unicode_filenames = [
            "файл.txt",     # Cyrillic
            "文件.txt",     # Chinese
            "ファイル.txt",   # Japanese
            "αρχείο.txt"   # Greek
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for filename in unicode_filenames:
                file_path = Path(temp_dir) / filename
                
                try:
                    # Initially doesn't exist
                    assert check_file_exists(file_path) is False
                    
                    # Create file
                    file_path.touch()
                    
                    # Now exists
                    assert check_file_exists(file_path) is True
                    
                except (UnicodeError, OSError):
                    # Skip if filesystem doesn't support unicode
                    pytest.skip(f"Filesystem doesn't support unicode filename: {filename}")

    def test_check_file_exists_absolute_vs_relative_paths(self):
        """Test file existence checking with absolute and relative paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_file.txt"
            file_path.touch()
            
            # Test absolute path
            absolute_path = file_path.resolve()
            assert check_file_exists(absolute_path) is True
            
            # Test relative path (change to temp directory)
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                relative_path = "test_file.txt"
                assert check_file_exists(relative_path) is True
            finally:
                os.chdir(original_cwd)

    def test_check_file_exists_symlinks(self):
        """Test file existence checking with symbolic links."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create original file
            original_file = Path(temp_dir) / "original.txt"
            original_file.touch()
            
            # Create symlink
            try:
                link_file = Path(temp_dir) / "link.txt"
                link_file.symlink_to(original_file)
                
                # Both should exist
                assert check_file_exists(original_file) is True
                assert check_file_exists(link_file) is True
                
                # Remove original file
                original_file.unlink()
                
                # Original doesn't exist, link becomes broken
                assert check_file_exists(original_file) is False
                assert check_file_exists(link_file) is False  # Broken symlink
                
            except (OSError, NotImplementedError):
                pytest.skip("Symbolic links not supported or permission denied")

    def test_check_file_exists_different_file_types(self):
        """Test file existence checking with different file types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_types = [
                "text_file.txt",
                "data_file.csv",
                "config_file.yaml",
                "script_file.py",
                "no_extension"
            ]
            
            for filename in file_types:
                file_path = Path(temp_dir) / filename
                
                # Initially doesn't exist
                assert check_file_exists(file_path) is False
                
                # Create file with some content
                file_path.write_text("test content")
                
                # Now exists
                assert check_file_exists(file_path) is True

    def test_check_file_exists_empty_vs_non_empty_files(self):
        """Test that file existence checking works regardless of file content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Empty file
            empty_file = Path(temp_dir) / "empty.txt"
            empty_file.touch()
            assert check_file_exists(empty_file) is True
            
            # Non-empty file
            content_file = Path(temp_dir) / "content.txt"
            content_file.write_text("This file has content")
            assert check_file_exists(content_file) is True

    def test_check_file_exists_case_sensitivity(self):
        """Test file existence checking case sensitivity (platform dependent)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create file with specific case
            original_file = Path(temp_dir) / "TestFile.txt"
            original_file.touch()
            
            # Test same case
            assert check_file_exists(original_file) is True
            
            # Test different case (behavior depends on filesystem)
            different_case = Path(temp_dir) / "testfile.txt"
            case_sensitive_result = check_file_exists(different_case)
            
            # Result depends on filesystem case sensitivity
            # On case-insensitive filesystems (Windows, macOS default), should be True
            # On case-sensitive filesystems (Linux), should be False
            if os.name == "nt":  # Windows
                assert case_sensitive_result is True
            # For other platforms, we can't make assumptions about case sensitivity

    def test_check_file_exists_long_paths(self):
        """Test file existence checking with very long file paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a long filename (but within reasonable limits)
            long_filename = "very_long_filename_" * 10 + ".txt"
            long_file_path = Path(temp_dir) / long_filename
            
            try:
                # Initially doesn't exist
                assert check_file_exists(long_file_path) is False
                
                # Create file
                long_file_path.touch()
                
                # Now exists
                assert check_file_exists(long_file_path) is True
                
            except OSError:
                # Skip if filename is too long for filesystem
                pytest.skip("Filename too long for filesystem")

    def test_check_file_exists_performance_large_directory(self):
        """Test performance of file existence checking in directories with many files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create many files
            file_count = 100
            target_file = None
            
            for i in range(file_count):
                file_path = Path(temp_dir) / f"file_{i:03d}.txt"
                file_path.touch()
                if i == 50:  # Remember middle file for testing
                    target_file = file_path
            
            # Test existence of specific file
            assert check_file_exists(target_file) is True
            
            # Test non-existent file in same directory
            nonexistent = Path(temp_dir) / "nonexistent_file.txt"
            assert check_file_exists(nonexistent) is False
