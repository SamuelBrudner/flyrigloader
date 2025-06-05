"""
Tests for file statistics functionality.

Tests the functionality for collecting and working with file system metadata.
Implements comprehensive edge case testing, cross-platform validation, and 
modern pytest practices with fixtures, parametrization, and property-based testing.
"""

import os
import sys
import stat
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import time
import platform
from typing import List, Dict, Any, Union, Tuple
from unittest.mock import Mock, patch, MagicMock

import pytest
import hypothesis
from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import composite

from flyrigloader.discovery.stats import (
    get_file_stats,
    attach_file_stats,
)
from flyrigloader.discovery.files import discover_files


# === Session-scoped Fixtures for Modern Pytest Practice ===

@pytest.fixture(scope="session")
def temp_workspace():
    """
    Session-scoped fixture creating a temporary workspace for all tests.
    
    This replaces basic file creation patterns with a managed workspace
    that persists across all tests in the session for better performance.
    """
    workspace = tempfile.mkdtemp(prefix="flyrigloader_test_")
    yield workspace
    # Cleanup after session
    if os.path.exists(workspace):
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.fixture
def test_dir(temp_workspace):
    """Function-scoped fixture providing a clean test directory."""
    test_subdir = os.path.join(temp_workspace, f"test_{os.getpid()}_{int(time.time() * 1000)}")
    os.makedirs(test_subdir, exist_ok=True)
    yield test_subdir
    # Cleanup after each test
    if os.path.exists(test_subdir):
        shutil.rmtree(test_subdir, ignore_errors=True)


@pytest.fixture
def sample_files(test_dir):
    """
    Fixture creating a set of sample files with known properties.
    
    Creates files with different sizes, timestamps, and content types
    for comprehensive testing scenarios.
    """
    files = {}
    
    # Small text file
    small_file = os.path.join(test_dir, "small.txt")
    with open(small_file, 'w') as f:
        f.write("Small content")
    files['small'] = small_file
    
    # Larger binary file
    large_file = os.path.join(test_dir, "large.bin")
    with open(large_file, 'wb') as f:
        f.write(b"X" * 10240)  # 10KB
    files['large'] = large_file
    
    # Empty file
    empty_file = os.path.join(test_dir, "empty.dat")
    Path(empty_file).touch()
    files['empty'] = empty_file
    
    # Unicode filename
    unicode_file = os.path.join(test_dir, "测试文件.txt")
    with open(unicode_file, 'w', encoding='utf-8') as f:
        f.write("Unicode content 测试")
    files['unicode'] = unicode_file
    
    return files


@pytest.fixture
def mock_filesystem(mocker):
    """
    Comprehensive filesystem mocking fixture for isolated unit testing.
    
    Uses pytest-mock for standardized mocking of os.stat and filesystem
    operations enabling isolated unit testing per TST-MOD-003 requirements.
    """
    mock_stat = mocker.patch('os.stat')
    mock_path_exists = mocker.patch('pathlib.Path.exists')
    mock_path_is_file = mocker.patch('pathlib.Path.is_file')
    mock_path_is_dir = mocker.patch('pathlib.Path.is_dir')
    mock_path_is_symlink = mocker.patch('pathlib.Path.is_symlink')
    mock_access = mocker.patch('os.access')
    
    return {
        'stat': mock_stat,
        'exists': mock_path_exists,
        'is_file': mock_path_is_file,
        'is_dir': mock_path_is_dir,
        'is_symlink': mock_path_is_symlink,
        'access': mock_access
    }


# === Core File Statistics Testing ===

class TestGetFileStats:
    """Test suite for get_file_stats function with comprehensive coverage."""
    
    def test_basic_file_stats(self, sample_files):
        """Test basic file statistics collection."""
        # Test small file
        stats = get_file_stats(sample_files['small'])
        
        # Verify required keys are present
        required_keys = {
            'size', 'size_bytes', 'mtime', 'modified_time', 'creation_time',
            'ctime', 'created_time', 'is_directory', 'is_file', 'is_symlink',
            'filename', 'extension', 'permissions', 'is_readable', 
            'is_writable', 'is_executable'
        }
        assert required_keys.issubset(stats.keys())
        
        # Verify data types
        assert isinstance(stats['size'], int)
        assert isinstance(stats['size_bytes'], int)
        assert isinstance(stats['mtime'], datetime)
        assert isinstance(stats['modified_time'], (int, float))
        assert isinstance(stats['creation_time'], datetime)
        assert isinstance(stats['ctime'], datetime)
        assert isinstance(stats['created_time'], (int, float))
        assert isinstance(stats['is_directory'], bool)
        assert isinstance(stats['is_file'], bool)
        assert isinstance(stats['is_symlink'], bool)
        assert isinstance(stats['filename'], str)
        assert isinstance(stats['extension'], str)
        assert isinstance(stats['permissions'], int)
        assert isinstance(stats['is_readable'], bool)
        assert isinstance(stats['is_writable'], bool)
        assert isinstance(stats['is_executable'], bool)
        
        # Verify specific values for known file
        assert stats['size'] == stats['size_bytes']  # Aliases should match
        assert stats['filename'] == "small.txt"
        assert stats['extension'] == "txt"
        assert stats['is_file'] == True
        assert stats['is_directory'] == False
        assert stats['is_readable'] == True  # Should be readable
    
    def test_file_nonexistent_raises_error(self):
        """Test that accessing nonexistent file raises FileNotFoundError."""
        nonexistent_path = "/absolutely/nonexistent/path/file.txt"
        
        with pytest.raises(FileNotFoundError, match="File not found"):
            get_file_stats(nonexistent_path)
    
    @pytest.mark.parametrize("file_type,expected_size", [
        ("small", 13),    # "Small content" = 13 characters
        ("large", 10240), # 10KB binary content
        ("empty", 0),     # Empty file
    ])
    def test_file_sizes(self, sample_files, file_type, expected_size):
        """Test file size reporting for different file types."""
        stats = get_file_stats(sample_files[file_type])
        assert stats['size'] == expected_size
        assert stats['size_bytes'] == expected_size
    
    def test_unicode_filename_handling(self, sample_files):
        """Test handling of Unicode filenames across platforms."""
        stats = get_file_stats(sample_files['unicode'])
        assert stats['filename'] == "测试文件.txt"
        assert stats['extension'] == "txt"
        assert stats['size'] > 0  # Should have content
    
    def test_pathlib_path_input(self, sample_files):
        """Test that pathlib.Path objects are handled correctly."""
        path_obj = Path(sample_files['small'])
        stats = get_file_stats(path_obj)
        
        assert stats['filename'] == "small.txt"
        assert stats['size'] > 0


# === Cross-Platform Timestamp Testing ===

class TestCrossPlatformTimestamps:
    """
    Test cross-platform OS-specific timestamp handling.
    
    Validates Windows, Linux, and macOS filesystem behaviors per Section 3.6.1 requirements.
    """
    
    @pytest.mark.parametrize("platform_name,has_birthtime", [
        ("Windows", True),
        ("Darwin", True),   # macOS
        ("Linux", False),
    ])
    def test_creation_time_platform_behavior(self, mock_filesystem, platform_name, has_birthtime):
        """Test platform-specific creation time handling."""
        with patch('platform.system', return_value=platform_name):
            # Mock stat result
            mock_stat_result = Mock()
            mock_stat_result.st_size = 1024
            mock_stat_result.st_mtime = 1609459200.0  # 2021-01-01 00:00:00
            mock_stat_result.st_ctime = 1609459100.0  # 100 seconds earlier
            
            if has_birthtime:
                mock_stat_result.st_birthtime = 1609459050.0  # Birth time even earlier
            
            # Configure mocks
            mock_filesystem['exists'].return_value = True
            mock_filesystem['stat'].return_value = mock_stat_result
            mock_filesystem['is_file'].return_value = True
            mock_filesystem['is_dir'].return_value = False
            mock_filesystem['is_symlink'].return_value = False
            mock_filesystem['access'].return_value = True
            
            with patch('pathlib.Path.stat', return_value=mock_stat_result):
                stats = get_file_stats("/test/file.txt")
                
                if has_birthtime:
                    # Should use st_birthtime
                    expected_creation = datetime.fromtimestamp(1609459050.0)
                else:
                    # Should fall back to st_ctime
                    expected_creation = datetime.fromtimestamp(1609459100.0)
                
                assert abs((stats['creation_time'] - expected_creation).total_seconds()) < 1
    
    def test_timestamp_precision(self, sample_files):
        """Test timestamp precision across different platforms."""
        stats = get_file_stats(sample_files['small'])
        
        # Timestamps should be reasonably recent (within last hour)
        now = datetime.now()
        time_diff = abs((stats['mtime'] - now).total_seconds())
        assert time_diff < 3600  # Within 1 hour
        
        time_diff = abs((stats['creation_time'] - now).total_seconds())
        assert time_diff < 3600  # Within 1 hour


# === Permission and Access Testing ===

class TestFilePermissions:
    """Test file permission and access checking."""
    
    def test_permission_calculation(self, sample_files):
        """Test permission calculation and access checks."""
        stats = get_file_stats(sample_files['small'])
        
        # Permissions should be a valid octal value
        assert 0 <= stats['permissions'] <= 0o777
        
        # Access checks should be boolean
        assert isinstance(stats['is_readable'], bool)
        assert isinstance(stats['is_writable'], bool)
        assert isinstance(stats['is_executable'], bool)
        
        # For a regular file we created, it should be readable
        assert stats['is_readable'] == True
    
    @pytest.mark.skipif(platform.system() == "Windows", 
                       reason="POSIX permissions not applicable on Windows")
    def test_posix_permission_modes(self, test_dir):
        """Test POSIX-specific permission modes."""
        # Create file with specific permissions
        test_file = os.path.join(test_dir, "perm_test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        
        # Set specific permissions (readable/writable by owner only)
        os.chmod(test_file, 0o600)
        
        stats = get_file_stats(test_file)
        assert stats['permissions'] == 0o600
        assert stats['is_readable'] == True
        assert stats['is_writable'] == True
        # Executable should be False for 0o600
        assert stats['is_executable'] == False


# === Symbolic Link Testing ===

@pytest.mark.skipif(platform.system() == "Windows", 
                   reason="Symbolic link testing may require admin privileges on Windows")
class TestSymbolicLinks:
    """Test handling of symbolic links."""
    
    def test_symlink_detection(self, test_dir, sample_files):
        """Test detection and handling of symbolic links."""
        # Create a symbolic link
        link_path = os.path.join(test_dir, "test_link.txt")
        
        try:
            os.symlink(sample_files['small'], link_path)
            
            stats = get_file_stats(link_path)
            assert stats['is_symlink'] == True
            assert stats['is_file'] == True  # Symlink to file should report as file
            assert stats['filename'] == "test_link.txt"
            
        except (OSError, NotImplementedError):
            # Skip if symlinks not supported
            pytest.skip("Symbolic links not supported on this system")


# === Property-Based Testing with Hypothesis ===

@composite
def file_sizes(draw):
    """Generate realistic file sizes for property-based testing."""
    # Most files are small, but occasionally large
    return draw(st.one_of(
        st.integers(min_value=0, max_value=1024),      # Small files (0-1KB)
        st.integers(min_value=1024, max_value=102400), # Medium files (1KB-100KB)
        st.integers(min_value=102400, max_value=1048576) # Large files (100KB-1MB)
    ))


@composite
def file_content(draw, size):
    """Generate file content of specified size."""
    if size == 0:
        return b""
    
    # Generate random bytes
    return draw(st.binary(min_size=size, max_size=size))


class TestPropertyBasedFileStats:
    """
    Property-based testing using Hypothesis for robust validation.
    
    Tests file statistics collection across diverse file types and sizes
    per Section 3.6.3 requirements.
    """
    
    @given(size=file_sizes())
    @settings(max_examples=50, deadline=5000)  # Limit examples for CI performance
    def test_file_size_property(self, test_dir, size):
        """Property test: reported file size should match actual file size."""
        assume(size >= 0)  # Ensure non-negative size
        
        # Create file with specific size
        test_file = os.path.join(test_dir, f"prop_test_{size}.dat")
        content = b"X" * size
        
        with open(test_file, 'wb') as f:
            f.write(content)
        
        stats = get_file_stats(test_file)
        assert stats['size'] == size
        assert stats['size_bytes'] == size
    
    @given(filename=st.text(min_size=1, max_size=50, 
                           alphabet=st.characters(whitelist_categories=['Lu', 'Ll', 'Nd', 'Pc'])))
    @settings(max_examples=30, deadline=3000)
    def test_filename_handling_property(self, test_dir, filename):
        """Property test: filename handling should be robust."""
        assume(filename.strip())  # Ensure non-empty after stripping
        assume(not any(char in filename for char in ['/', '\\', '\0']))  # Invalid chars
        
        # Create file with generated filename
        safe_filename = filename.replace(' ', '_')  # Replace spaces for safety
        test_file = os.path.join(test_dir, f"{safe_filename}.txt")
        
        try:
            with open(test_file, 'w') as f:
                f.write("test content")
            
            stats = get_file_stats(test_file)
            assert stats['filename'] == f"{safe_filename}.txt"
            assert stats['extension'] == "txt"
            
        except (OSError, UnicodeError):
            # Some filenames may not be valid on all systems
            assume(False)


# === Attach File Stats Testing ===

class TestAttachFileStats:
    """Test the attach_file_stats function with various input scenarios."""
    
    def test_attach_to_file_list(self, sample_files):
        """Test attaching stats to a list of file paths."""
        file_list = list(sample_files.values())
        result = attach_file_stats(file_list)
        
        assert isinstance(result, dict)
        assert len(result) == len(file_list)
        
        for file_path in file_list:
            assert file_path in result
            assert 'size' in result[file_path]
            assert 'mtime' in result[file_path]
            assert isinstance(result[file_path]['size'], int)
    
    def test_attach_to_metadata_dict(self, sample_files):
        """Test attaching stats to existing metadata dictionary."""
        initial_metadata = {
            sample_files['small']: {'experiment': 'exp1', 'condition': 'control'},
            sample_files['large']: {'experiment': 'exp2', 'condition': 'treatment'}
        }
        
        result = attach_file_stats(initial_metadata)
        
        assert isinstance(result, dict)
        assert len(result) == 2
        
        for file_path, metadata in result.items():
            # Original metadata should be preserved
            assert 'experiment' in metadata
            assert 'condition' in metadata
            
            # File stats should be added
            assert 'size' in metadata
            assert 'mtime' in metadata
            assert 'is_file' in metadata
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        # Empty list
        result = attach_file_stats([])
        assert result == {}
        
        # Empty dict
        result = attach_file_stats({})
        assert result == {}
    
    @pytest.mark.parametrize("invalid_input", [
        None,
        "single_string",
        123,
        {"file.txt": "not_a_dict"}  # Dict with non-dict values
    ])
    def test_invalid_input_handling(self, invalid_input):
        """Test handling of invalid inputs."""
        if invalid_input is None or isinstance(invalid_input, (str, int)):
            with pytest.raises((TypeError, AttributeError)):
                attach_file_stats(invalid_input)
        else:
            # For the dict with non-dict values case
            with pytest.raises((TypeError, AttributeError)):
                attach_file_stats(invalid_input)


# === Integration Testing ===

class TestFileStatsIntegration:
    """
    Enhanced integration testing with discover_files function.
    
    Validates end-to-end statistics attachment workflows per F-008 
    and TST-INTEG-001 requirements.
    """
    
    def test_discover_files_with_stats(self, test_dir):
        """Test integration between discover_files and file statistics."""
        # Create test files with known patterns
        test_files = [
            "mouse_20240101_control_1.csv",
            "mouse_20240102_treatment_1.csv", 
            "rat_20240103_control_2.csv"
        ]
        
        for filename in test_files:
            file_path = os.path.join(test_dir, filename)
            with open(file_path, 'w') as f:
                f.write(f"# {filename}\ndata,value\n1,2\n3,4\n")
        
        # Discover files with stats
        result = discover_files(test_dir, "*.csv", include_stats=True)
        
        assert isinstance(result, dict)
        assert len(result) == 3
        
        # Verify all files have stats
        for file_path, metadata in result.items():
            assert 'size' in metadata
            assert 'mtime' in metadata
            assert 'filename' in metadata
            assert 'is_file' in metadata
            assert metadata['is_file'] == True
            assert metadata['size'] > 0  # Should have content
    
    def test_discover_files_with_metadata_and_stats(self, test_dir):
        """Test discovery with both metadata extraction and file stats."""
        # Create files with extractable metadata
        test_files = [
            "exp001_mouse_baseline.csv",
            "exp002_rat_treatment.csv"
        ]
        
        for filename in test_files:
            file_path = os.path.join(test_dir, filename)
            with open(file_path, 'w') as f:
                f.write(f"timestamp,x,y\n1.0,10.5,20.3\n2.0,11.2,21.1\n")
        
        # Use extraction patterns and stats
        extract_patterns = [r".*/(?P<experiment_id>exp\d+)_(?P<animal>\w+)_(?P<condition>\w+)\.csv"]
        
        from flyrigloader.discovery.files import FileDiscoverer
        discoverer = FileDiscoverer(
            extract_patterns=extract_patterns,
            include_stats=True
        )
        
        result = discoverer.discover(test_dir, "*.csv")
        
        assert isinstance(result, dict)
        assert len(result) == 2
        
        # Verify metadata extraction and stats are both present
        for file_path, metadata in result.items():
            # Extracted metadata
            assert 'experiment_id' in metadata
            assert 'animal' in metadata 
            assert 'condition' in metadata
            
            # File statistics
            assert 'size' in metadata
            assert 'mtime' in metadata
            assert 'filename' in metadata
    
    def test_large_file_set_performance(self, test_dir):
        """Test performance with larger numbers of files."""
        # Create a moderate number of files to test performance
        num_files = 50
        file_paths = []
        
        for i in range(num_files):
            file_path = os.path.join(test_dir, f"data_{i:03d}.txt")
            with open(file_path, 'w') as f:
                f.write(f"File {i} content with some data")
            file_paths.append(file_path)
        
        # Time the stats attachment
        import time
        start_time = time.time()
        result = attach_file_stats(file_paths)
        end_time = time.time()
        
        # Should complete reasonably quickly
        execution_time = end_time - start_time
        assert execution_time < 5.0  # Should complete within 5 seconds
        
        # Verify all files processed
        assert len(result) == num_files
        for file_path in file_paths:
            assert file_path in result
            assert 'size' in result[file_path]


# === Error Handling and Edge Cases ===

class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge case scenarios."""
    
    def test_permission_denied_handling(self, mock_filesystem):
        """Test handling of permission denied scenarios."""
        # Mock a permission denied scenario
        mock_filesystem['exists'].return_value = True
        mock_filesystem['stat'].side_effect = PermissionError("Permission denied")
        
        with pytest.raises(PermissionError):
            get_file_stats("/restricted/file.txt")
    
    def test_corrupted_filesystem_handling(self, mock_filesystem):
        """Test handling of corrupted filesystem scenarios."""
        # Mock filesystem corruption
        mock_filesystem['exists'].return_value = True
        mock_filesystem['stat'].side_effect = OSError("I/O error")
        
        with pytest.raises(OSError):
            get_file_stats("/corrupted/file.txt")
    
    @pytest.mark.parametrize("special_path", [
        "",           # Empty string
        ".",          # Current directory
        "..",         # Parent directory
        "/dev/null",  # Special device (Unix)
    ])
    def test_special_path_handling(self, special_path):
        """Test handling of special filesystem paths."""
        if special_path in ["", ".", ".."]:
            # These should raise FileNotFoundError or similar
            with pytest.raises((FileNotFoundError, OSError, PermissionError)):
                get_file_stats(special_path)
        elif special_path == "/dev/null" and platform.system() != "Windows":
            # On Unix systems, /dev/null exists but may behave differently
            try:
                stats = get_file_stats(special_path)
                # /dev/null should be a special file
                assert stats['size'] == 0
            except (OSError, PermissionError):
                # May not be accessible in some environments
                pass
    
    def test_very_long_filename_handling(self, test_dir):
        """Test handling of very long filenames."""
        # Most filesystems have limits around 255 characters for filenames
        long_name = "a" * 240 + ".txt"  # Stay under typical limits
        long_path = os.path.join(test_dir, long_name)
        
        try:
            with open(long_path, 'w') as f:
                f.write("test content")
            
            stats = get_file_stats(long_path)
            assert stats['filename'] == long_name
            assert len(stats['filename']) == 244  # 240 + ".txt"
            
        except OSError:
            # Some systems may reject very long filenames
            pytest.skip("Filesystem doesn't support long filenames")


# === Mock-Based Unit Testing ===

class TestMockedFileStats:
    """
    Mocked unit tests for isolated testing of stats functionality.
    
    Uses pytest-mock for standardized mocking enabling isolated unit testing
    per TST-MOD-003 requirements.
    """
    
    def test_mocked_stat_call(self, mock_filesystem):
        """Test file stats with mocked os.stat call."""
        # Create a mock stat result
        mock_stat_result = Mock()
        mock_stat_result.st_size = 2048
        mock_stat_result.st_mode = 0o100644  # Regular file, rw-r--r--
        mock_stat_result.st_mtime = 1609459200.0
        mock_stat_result.st_ctime = 1609459100.0
        mock_stat_result.st_birthtime = 1609459050.0  # macOS/Windows creation time
        
        # Configure mocks
        mock_filesystem['exists'].return_value = True
        mock_filesystem['stat'].return_value = mock_stat_result
        mock_filesystem['is_file'].return_value = True
        mock_filesystem['is_dir'].return_value = False
        mock_filesystem['is_symlink'].return_value = False
        mock_filesystem['access'].return_value = True
        
        with patch('pathlib.Path.stat', return_value=mock_stat_result):
            stats = get_file_stats("/mocked/file.txt")
            
            assert stats['size'] == 2048
            assert stats['permissions'] == 0o644  # Should extract permission bits
            assert stats['filename'] == "file.txt"
            assert stats['is_file'] == True
            assert stats['is_directory'] == False
    
    def test_cross_platform_birthtime_fallback(self, mock_filesystem):
        """Test birthtime fallback behavior across platforms."""
        mock_stat_result = Mock()
        mock_stat_result.st_size = 1024
        mock_stat_result.st_mtime = 1609459200.0
        mock_stat_result.st_ctime = 1609459100.0
        # No st_birthtime attribute (Linux behavior)
        
        mock_filesystem['exists'].return_value = True
        mock_filesystem['stat'].return_value = mock_stat_result
        mock_filesystem['is_file'].return_value = True
        mock_filesystem['is_dir'].return_value = False
        mock_filesystem['is_symlink'].return_value = False
        mock_filesystem['access'].return_value = True
        
        with patch('pathlib.Path.stat', return_value=mock_stat_result):
            stats = get_file_stats("/test/file.txt")
            
            # Should fall back to st_ctime
            expected_creation = datetime.fromtimestamp(1609459100.0)
            assert abs((stats['creation_time'] - expected_creation).total_seconds()) < 1
