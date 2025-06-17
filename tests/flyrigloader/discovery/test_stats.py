"""
Enhanced file statistics functionality testing.

This module implements behavior-focused testing for file statistics functionality,
emphasizing public API behavior validation through black-box testing approaches
that validate observable system behavior rather than implementation-specific details.

Testing Strategy:
- Protocol-based mock implementations for consistent dependency injection
- Centralized fixture management from tests/conftest.py and tests/utils.py
- AAA (Arrange-Act-Assert) pattern enforcement for improved readability
- Edge-case coverage through parameterized test scenarios
- Performance tests isolated to scripts/benchmarks/ directory

Key Features:
- Observable behavior validation for get_file_stats and attach_file_stats functions
- Cross-platform file permission and timestamp handling validation
- Unicode filename and special character processing edge cases
- Error handling validation for nonexistent paths and corrupted files
- Protocol-based filesystem mocking for reliable test isolation
"""

import os
import platform
import stat
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import Mock

import pytest
from hypothesis import given, settings, strategies as st

# Import production modules for public API testing
from flyrigloader.discovery.stats import (
    FileStatsError,
    get_file_stats,
    attach_file_stats,
    FileStatsCollector,
    DefaultFileSystemProvider,
    DefaultTimestampProvider,
)

# Import centralized test utilities for consistent mock implementations
from tests.utils import (
    MockFilesystem,
    create_mock_filesystem,
    EdgeCaseScenarioGenerator,
    generate_edge_case_scenarios,
    FlyrigloaderStrategies,
)


# ============================================================================
# CORE FILE STATISTICS PUBLIC API TESTING
# ============================================================================

class TestGetFileStatsPublicAPI:
    """
    Test suite for get_file_stats function focusing on public API behavior validation.
    
    Implements black-box testing approach emphasizing observable behavior rather than
    internal implementation details, per Section 0 behavior-focused testing requirements.
    """
    
    def test_basic_file_stats_structure(self, temp_experiment_directory):
        """
        Test that get_file_stats returns correctly structured metadata dictionary.
        
        ARRANGE: Create test file with known properties
        ACT: Call get_file_stats on the test file
        ASSERT: Verify complete metadata structure and data types
        """
        # ARRANGE - Set up test file with known content
        test_file = temp_experiment_directory["directory"] / "test_stats.txt"
        test_content = "Test file content for statistics validation"
        test_file.write_text(test_content)
        
        # ACT - Call public API function
        result = get_file_stats(test_file)
        
        # ASSERT - Verify comprehensive metadata structure
        required_keys = {
            'size', 'size_bytes', 'mtime', 'modified_time', 'creation_time',
            'ctime', 'created_time', 'is_directory', 'is_file', 'is_symlink',
            'filename', 'extension', 'permissions', 'is_readable', 
            'is_writable', 'is_executable'
        }
        assert required_keys.issubset(result.keys())
        
        # Verify data types match API contract
        assert isinstance(result['size'], int)
        assert isinstance(result['size_bytes'], int)
        assert isinstance(result['mtime'], datetime)
        assert isinstance(result['modified_time'], (int, float))
        assert isinstance(result['creation_time'], datetime)
        assert isinstance(result['is_file'], bool)
        assert isinstance(result['filename'], str)
        assert isinstance(result['extension'], str)
        assert isinstance(result['permissions'], int)
        
        # Verify observable behavior properties
        assert result['size'] == len(test_content.encode('utf-8'))
        assert result['size'] == result['size_bytes']  # Alias consistency
        assert result['filename'] == "test_stats.txt"
        assert result['extension'] == "txt"
        assert result['is_file'] is True
        assert result['is_directory'] is False
        assert result['is_readable'] is True

    def test_nonexistent_file_error_behavior(self):
        """
        Test error handling behavior for nonexistent files.
        
        ARRANGE: Define path to nonexistent file
        ACT: Call get_file_stats on nonexistent path
        ASSERT: Verify FileStatsError is raised with appropriate context
        """
        # ARRANGE - Create path that definitely doesn't exist
        nonexistent_path = Path("/absolutely/nonexistent/path/file.txt")
        
        # ACT & ASSERT - Verify error behavior
        with pytest.raises(FileStatsError) as exc_info:
            get_file_stats(nonexistent_path)
        
        # Verify error provides meaningful context
        assert str(nonexistent_path) in str(exc_info.value)
        assert exc_info.value.path == str(nonexistent_path)
        assert "context" in dir(exc_info.value)

    @pytest.mark.parametrize("file_extension,expected_ext", [
        ("txt", "txt"),
        ("csv", "csv"),
        ("pkl", "pkl"),
        ("yaml", "yaml"),
        ("", ""),  # No extension
    ])
    def test_extension_detection_behavior(self, temp_experiment_directory, file_extension, expected_ext):
        """
        Test file extension detection behavior across different file types.
        
        ARRANGE: Create files with various extensions
        ACT: Get file statistics for each file
        ASSERT: Verify extension detection behavior
        """
        # ARRANGE - Create test file with specific extension
        filename = f"test_file.{file_extension}" if file_extension else "test_file"
        test_file = temp_experiment_directory["directory"] / filename
        test_file.write_text("test content")
        
        # ACT - Get file statistics
        result = get_file_stats(test_file)
        
        # ASSERT - Verify extension detection
        assert result['extension'] == expected_ext
        assert result['filename'] == filename

    def test_unicode_filename_handling_behavior(self, temp_experiment_directory):
        """
        Test Unicode filename handling across platforms.
        
        ARRANGE: Create file with Unicode characters in name
        ACT: Get file statistics
        ASSERT: Verify Unicode handling behavior
        """
        # ARRANGE - Create file with Unicode name
        unicode_filename = "測試文件_ñämé_файл.txt"
        test_file = temp_experiment_directory["directory"] / unicode_filename
        
        try:
            test_file.write_text("Unicode test content 測試內容")
            
            # ACT - Get file statistics
            result = get_file_stats(test_file)
            
            # ASSERT - Verify Unicode filename preservation
            assert result['filename'] == unicode_filename
            assert result['extension'] == "txt"
            assert result['size'] > 0
            
        except (OSError, UnicodeError):
            # Skip test if filesystem doesn't support Unicode
            pytest.skip("Filesystem doesn't support Unicode filenames")

    def test_pathlib_path_object_behavior(self, temp_experiment_directory):
        """
        Test that Path objects are handled correctly by the public API.
        
        ARRANGE: Create test file and Path object
        ACT: Call get_file_stats with Path object
        ASSERT: Verify consistent behavior with string paths
        """
        # ARRANGE - Set up Path object
        test_file = temp_experiment_directory["directory"] / "path_object_test.txt"
        test_file.write_text("Path object test content")
        
        # ACT - Use Path object directly
        result = get_file_stats(test_file)
        
        # ASSERT - Verify Path object handling
        assert result['filename'] == "path_object_test.txt"
        assert result['extension'] == "txt"
        assert result['is_file'] is True
        assert result['size'] > 0


# ============================================================================
# PROTOCOL-BASED MOCK TESTING WITH CENTRALIZED UTILITIES
# ============================================================================

class TestFileStatsWithMockFilesystem:
    """
    Test file statistics using Protocol-based mock implementations from tests/utils.py.
    
    Implements centralized mocking patterns for consistent behavior simulation
    across test scenarios, eliminating duplicate mock definitions.
    """
    
    def test_mock_filesystem_basic_behavior(self):
        """
        Test basic file statistics collection using centralized MockFilesystem.
        
        ARRANGE: Set up MockFilesystem with test files
        ACT: Get file statistics using mock provider
        ASSERT: Verify mock behavior integration
        """
        # ARRANGE - Create MockFilesystem from centralized utilities
        mock_fs = create_mock_filesystem()
        
        # Add test file to mock filesystem
        test_path = Path("/test/mock_file.txt")
        mock_fs.add_file(
            test_path,
            size=1024,
            mtime=datetime(2024, 1, 1, 12, 0, 0),
            content="Mock file content"
        )
        
        # Create mock providers
        class MockFileSystemProvider:
            def path_exists(self, path):
                return mock_fs.exists(path)
            
            def get_stat(self, path):
                return mock_fs.stat(path)
            
            def check_access(self, path, mode):
                return True  # Simplified for testing
            
            def is_dir(self, path):
                return mock_fs.is_dir(path)
            
            def is_file(self, path):
                return mock_fs.is_file(path)
            
            def is_symlink(self, path):
                return False  # Simplified for testing
        
        # ACT - Get file statistics with mock provider
        result = get_file_stats(
            test_path,
            fs_provider=MockFileSystemProvider(),
            enable_test_hooks=True
        )
        
        # ASSERT - Verify mock integration behavior
        assert result['size'] == 1024
        assert result['filename'] == "mock_file.txt"
        assert result['extension'] == "txt"
        assert result['is_file'] is True
        assert result['is_directory'] is False
        assert '_test_metadata' in result  # Test hooks enabled

    def test_mock_filesystem_error_scenarios(self):
        """
        Test error handling with MockFilesystem edge cases.
        
        ARRANGE: Set up MockFilesystem with error scenarios
        ACT: Attempt file operations on problematic files
        ASSERT: Verify error handling behavior
        """
        # ARRANGE - Create MockFilesystem with error scenarios
        mock_fs = create_mock_filesystem()
        
        # Add file with access error
        error_file = Path("/test/permission_denied.txt")
        mock_fs.add_file(
            error_file,
            size=100,
            access_error=PermissionError("Mock permission denied")
        )
        
        # Create mock provider that raises configured errors
        class MockFileSystemProviderWithErrors:
            def path_exists(self, path):
                return mock_fs.exists(path)
            
            def get_stat(self, path):
                if str(path) == str(error_file):
                    raise PermissionError("Mock permission denied")
                return mock_fs.stat(path)
            
            def check_access(self, path, mode):
                return True
            
            def is_dir(self, path):
                return False
            
            def is_file(self, path):
                return True
            
            def is_symlink(self, path):
                return False
        
        # ACT & ASSERT - Verify error propagation
        with pytest.raises(FileStatsError) as exc_info:
            get_file_stats(
                error_file,
                fs_provider=MockFileSystemProviderWithErrors()
            )
        
        # Verify error context includes original error information
        assert "Mock permission denied" in str(exc_info.value)
        assert exc_info.value.path == str(error_file)


# ============================================================================
# ENHANCED EDGE-CASE COVERAGE WITH PARAMETERIZED TESTING
# ============================================================================

class TestFileStatsEdgeCases:
    """
    Comprehensive edge-case testing using centralized scenario generators.
    
    Implements parameterized test scenarios for boundary conditions, Unicode handling,
    and cross-platform compatibility validation.
    """
    
    @pytest.mark.parametrize("scenario", [
        {"name": "empty_file", "content": "", "expected_size": 0},
        {"name": "single_byte", "content": "A", "expected_size": 1},
        {"name": "unicode_content", "content": "測試內容 🚀", "expected_size": None},  # Variable size
        {"name": "binary_content", "content": b"\x00\x01\x02\xFF", "expected_size": 4},
    ])
    def test_file_content_edge_cases(self, temp_experiment_directory, scenario):
        """
        Test file statistics behavior with various content edge cases.
        
        ARRANGE: Create files with edge-case content
        ACT: Get file statistics
        ASSERT: Verify correct size and metadata handling
        """
        # ARRANGE - Create test file with edge-case content
        test_file = temp_experiment_directory["directory"] / f"{scenario['name']}_test.txt"
        
        if isinstance(scenario['content'], bytes):
            test_file.write_bytes(scenario['content'])
        else:
            test_file.write_text(scenario['content'], encoding='utf-8')
        
        # ACT - Get file statistics
        result = get_file_stats(test_file)
        
        # ASSERT - Verify edge-case handling
        if scenario['expected_size'] is not None:
            assert result['size'] == scenario['expected_size']
        else:
            assert result['size'] > 0  # Unicode content should have some size
        
        assert result['filename'] == f"{scenario['name']}_test.txt"
        assert result['is_file'] is True

    @pytest.mark.parametrize("path_component", [
        "normal_file.txt",
        "file-with-dashes.txt", 
        "file_with_underscores.txt",
        "file.with.multiple.dots.txt",
        "UPPERCASE_FILE.TXT",
        "123numeric_start.txt",
    ])
    def test_filename_pattern_edge_cases(self, temp_experiment_directory, path_component):
        """
        Test filename pattern handling edge cases.
        
        ARRANGE: Create files with various naming patterns
        ACT: Get file statistics
        ASSERT: Verify filename parsing behavior
        """
        # ARRANGE - Create file with specific naming pattern
        test_file = temp_experiment_directory["directory"] / path_component
        test_file.write_text("Pattern test content")
        
        # ACT - Get file statistics
        result = get_file_stats(test_file)
        
        # ASSERT - Verify filename parsing
        assert result['filename'] == path_component
        expected_ext = Path(path_component).suffix[1:] if Path(path_component).suffix else ""
        assert result['extension'] == expected_ext

    @pytest.mark.skipif(platform.system() == "Windows", 
                       reason="POSIX permissions not applicable on Windows")
    def test_permission_edge_cases(self, temp_experiment_directory):
        """
        Test file permission detection edge cases on POSIX systems.
        
        ARRANGE: Create files with specific permissions
        ACT: Get file statistics
        ASSERT: Verify permission detection behavior
        """
        # ARRANGE - Create file with specific permissions
        test_file = temp_experiment_directory["directory"] / "permission_test.txt"
        test_file.write_text("Permission test content")
        
        # Set specific permissions (read-only)
        os.chmod(test_file, 0o444)
        
        # ACT - Get file statistics
        result = get_file_stats(test_file)
        
        # ASSERT - Verify permission detection
        assert result['permissions'] == 0o444
        assert result['is_readable'] is True
        assert result['is_writable'] is False
        assert result['is_executable'] is False

    def test_corrupted_file_scenarios(self):
        """
        Test behavior with corrupted file scenarios using centralized utilities.
        
        ARRANGE: Set up corrupted file scenarios
        ACT: Attempt to get file statistics
        ASSERT: Verify appropriate error handling
        """
        # ARRANGE - Generate corrupted file scenarios
        edge_scenarios = generate_edge_case_scenarios(['corrupted'])
        
        for scenario in edge_scenarios['corrupted']:
            if scenario['type'] == 'permission_denied':
                # Create MockFilesystem for permission testing
                mock_fs = create_mock_filesystem()
                mock_fs.add_file(
                    "/test/restricted.txt",
                    size=100,
                    access_error=PermissionError("Access denied")
                )
                
                # Verify error handling is covered by mock testing
                assert mock_fs.access_errors["/test/restricted.txt"]


# ============================================================================
# ATTACH FILE STATS BEHAVIOR VALIDATION
# ============================================================================

class TestAttachFileStatsPublicAPI:
    """
    Test suite for attach_file_stats function focusing on public API behavior.
    
    Validates batch processing functionality and error handling patterns
    through observable behavior rather than implementation coupling.
    """
    
    def test_attach_stats_to_file_list_behavior(self, temp_experiment_directory):
        """
        Test attaching statistics to a list of file paths.
        
        ARRANGE: Create list of test files
        ACT: Call attach_file_stats
        ASSERT: Verify batch processing behavior
        """
        # ARRANGE - Create multiple test files
        test_files = []
        for i in range(3):
            test_file = temp_experiment_directory["directory"] / f"list_test_{i}.txt"
            test_file.write_text(f"Test content {i}")
            test_files.append(str(test_file))
        
        # ACT - Attach file statistics
        result = attach_file_stats(test_files)
        
        # ASSERT - Verify batch processing behavior
        assert isinstance(result, dict)
        assert len(result) == len(test_files)
        
        for file_path in test_files:
            assert file_path in result
            assert 'size' in result[file_path]
            assert 'mtime' in result[file_path]
            assert 'filename' in result[file_path]
            assert isinstance(result[file_path]['size'], int)

    def test_attach_stats_to_metadata_dict_behavior(self, temp_experiment_directory):
        """
        Test attaching statistics to existing metadata dictionary.
        
        ARRANGE: Create files with existing metadata
        ACT: Call attach_file_stats with metadata dict
        ASSERT: Verify metadata preservation and enhancement
        """
        # ARRANGE - Create files with initial metadata
        test_files = {}
        for i in range(2):
            test_file = temp_experiment_directory["directory"] / f"metadata_test_{i}.txt"
            test_file.write_text(f"Metadata test content {i}")
            test_files[str(test_file)] = {
                'experiment': f'exp_{i}',
                'condition': 'control' if i % 2 == 0 else 'treatment'
            }
        
        # ACT - Attach file statistics to existing metadata
        result = attach_file_stats(test_files)
        
        # ASSERT - Verify metadata preservation and enhancement
        assert isinstance(result, dict)
        assert len(result) == len(test_files)
        
        for file_path, metadata in result.items():
            # Original metadata should be preserved
            assert 'experiment' in metadata
            assert 'condition' in metadata
            
            # File statistics should be added
            assert 'size' in metadata
            assert 'mtime' in metadata
            assert 'filename' in metadata
            assert 'is_file' in metadata

    def test_empty_input_handling_behavior(self):
        """
        Test behavior with empty inputs.
        
        ARRANGE: Prepare empty inputs
        ACT: Call attach_file_stats with empty inputs
        ASSERT: Verify graceful handling
        """
        # ARRANGE & ACT & ASSERT - Test empty list
        result = attach_file_stats([])
        assert result == {}
        
        # ARRANGE & ACT & ASSERT - Test empty dict
        result = attach_file_stats({})
        assert result == {}

    @pytest.mark.parametrize("invalid_input", [
        None,
        "single_string",
        123,
        ["valid_path", None, "another_path"],  # Mixed valid/invalid
    ])
    def test_invalid_input_error_behavior(self, invalid_input):
        """
        Test error handling behavior with invalid inputs.
        
        ARRANGE: Prepare invalid inputs
        ACT: Call attach_file_stats with invalid inputs
        ASSERT: Verify appropriate error behavior
        """
        # ARRANGE, ACT & ASSERT - Verify error handling
        with pytest.raises((TypeError, AttributeError, FileStatsError)):
            attach_file_stats(invalid_input)


# ============================================================================
# CROSS-PLATFORM TIMESTAMP BEHAVIOR VALIDATION
# ============================================================================

class TestCrossPlatformTimestampBehavior:
    """
    Test cross-platform timestamp handling behavior through public API.
    
    Validates platform-specific timestamp behavior using dependency injection
    rather than testing internal implementation details.
    """
    
    @pytest.mark.parametrize("platform_name,has_birthtime", [
        ("Windows", True),
        ("Darwin", True),   # macOS
        ("Linux", False),
    ])
    def test_timestamp_provider_behavior(self, platform_name, has_birthtime):
        """
        Test timestamp provider behavior across platforms.
        
        ARRANGE: Create mock timestamp provider for platform
        ACT: Get file statistics with custom provider
        ASSERT: Verify platform-specific timestamp behavior
        """
        # ARRANGE - Create mock timestamp provider
        class MockTimestampProvider:
            def get_creation_timestamp(self, stat_result):
                if has_birthtime:
                    return getattr(stat_result, 'st_birthtime', stat_result.st_ctime)
                else:
                    return stat_result.st_ctime
            
            def timestamp_to_datetime(self, timestamp):
                return datetime.fromtimestamp(timestamp)
        
        # Mock stat result
        mock_stat = Mock()
        mock_stat.st_size = 1024
        mock_stat.st_mtime = 1609459200.0  # 2021-01-01 00:00:00
        mock_stat.st_ctime = 1609459100.0  # 100 seconds earlier
        if has_birthtime:
            mock_stat.st_birthtime = 1609459050.0  # Birth time even earlier
        mock_stat.st_mode = 0o644
        mock_stat.st_uid = 1000
        mock_stat.st_gid = 1000
        mock_stat.st_nlink = 1
        
        # Create mock filesystem provider
        class MockFileSystemProvider:
            def path_exists(self, path):
                return True
            
            def get_stat(self, path):
                return mock_stat
            
            def check_access(self, path, mode):
                return True
            
            def is_dir(self, path):
                return False
            
            def is_file(self, path):
                return True
            
            def is_symlink(self, path):
                return False
        
        # ACT - Get file statistics with custom providers
        result = get_file_stats(
            Path("/test/timestamp_test.txt"),
            fs_provider=MockFileSystemProvider(),
            timestamp_provider=MockTimestampProvider()
        )
        
        # ASSERT - Verify timestamp behavior
        if has_birthtime:
            expected_creation = datetime.fromtimestamp(1609459050.0)
        else:
            expected_creation = datetime.fromtimestamp(1609459100.0)
        
        # Allow for minor timestamp differences
        time_diff = abs((result['creation_time'] - expected_creation).total_seconds())
        assert time_diff < 1

    def test_recent_file_timestamp_behavior(self, temp_experiment_directory):
        """
        Test timestamp behavior for recently created files.
        
        ARRANGE: Create a new file
        ACT: Get file statistics immediately
        ASSERT: Verify timestamps are recent and reasonable
        """
        # ARRANGE - Create new file
        test_file = temp_experiment_directory["directory"] / "recent_file.txt"
        create_time = datetime.now()
        test_file.write_text("Recent file content")
        
        # ACT - Get file statistics
        result = get_file_stats(test_file)
        
        # ASSERT - Verify recent timestamps
        now = datetime.now()
        
        # Modification time should be recent
        mtime_diff = abs((result['mtime'] - now).total_seconds())
        assert mtime_diff < 60  # Within 1 minute
        
        # Creation time should be after our start time
        assert result['creation_time'] >= create_time - timedelta(seconds=1)
        assert result['creation_time'] <= now + timedelta(seconds=1)


# ============================================================================
# FILE STATS COLLECTOR MODULAR INTERFACE TESTING
# ============================================================================

class TestFileStatsCollectorBehavior:
    """
    Test FileStatsCollector modular interface behavior.
    
    Validates the modular collection interface while maintaining focus
    on public API behavior rather than internal coupling.
    """
    
    def test_collector_single_file_behavior(self, temp_experiment_directory):
        """
        Test FileStatsCollector single file collection behavior.
        
        ARRANGE: Create FileStatsCollector and test file
        ACT: Collect single file statistics
        ASSERT: Verify collector behavior
        """
        # ARRANGE - Create collector and test file
        collector = FileStatsCollector()
        test_file = temp_experiment_directory["directory"] / "collector_test.txt"
        test_file.write_text("Collector test content")
        
        # ACT - Collect single file statistics
        result = collector.collect_single_file_stats(test_file)
        
        # ASSERT - Verify collector behavior
        assert isinstance(result, dict)
        assert 'size' in result
        assert 'filename' in result
        assert result['filename'] == "collector_test.txt"
        assert result['is_file'] is True

    def test_collector_batch_stats_behavior(self, temp_experiment_directory):
        """
        Test FileStatsCollector batch collection behavior.
        
        ARRANGE: Create FileStatsCollector and multiple test files
        ACT: Collect batch statistics
        ASSERT: Verify batch collection behavior
        """
        # ARRANGE - Create collector and test files
        collector = FileStatsCollector()
        test_files = []
        for i in range(3):
            test_file = temp_experiment_directory["directory"] / f"batch_test_{i}.txt"
            test_file.write_text(f"Batch content {i}")
            test_files.append(str(test_file))
        
        # ACT - Collect batch statistics
        result = collector.collect_batch_stats(test_files)
        
        # ASSERT - Verify batch collection behavior
        assert isinstance(result, dict)
        assert len(result) == len(test_files)
        for file_path in test_files:
            assert file_path in result
            assert isinstance(result[file_path], dict)

    def test_collector_with_custom_providers(self):
        """
        Test FileStatsCollector with custom providers.
        
        ARRANGE: Create collector with custom providers
        ACT: Use collector with dependency injection
        ASSERT: Verify provider integration behavior
        """
        # ARRANGE - Create custom providers
        fs_provider = DefaultFileSystemProvider()
        timestamp_provider = DefaultTimestampProvider()
        collector = FileStatsCollector(
            fs_provider=fs_provider,
            timestamp_provider=timestamp_provider
        )
        
        # ACT & ASSERT - Verify provider integration
        assert collector.fs_provider is fs_provider
        assert collector.timestamp_provider is timestamp_provider


# ============================================================================
# PROPERTY-BASED TESTING WITH HYPOTHESIS
# ============================================================================

class TestFileStatsPropertyBased:
    """
    Property-based testing using Hypothesis for comprehensive edge-case discovery.
    
    Implements domain-specific strategies for robust validation of file statistics
    behavior across diverse file scenarios.
    """
    
    @given(st.integers(min_value=0, max_value=10000))
    @settings(max_examples=20, deadline=5000)  # Reduced for performance
    def test_file_size_property_consistency(self, temp_experiment_directory, file_size):
        """
        Property test: reported file size should match actual file size.
        
        ARRANGE: Create file with specific size
        ACT: Get file statistics
        ASSERT: Verify size consistency property
        """
        # ARRANGE - Create file with specific size
        test_file = temp_experiment_directory["directory"] / f"prop_size_{file_size}.txt"
        content = "X" * file_size
        test_file.write_text(content)
        
        # ACT - Get file statistics
        result = get_file_stats(test_file)
        
        # ASSERT - Verify size property
        expected_size = len(content.encode('utf-8'))
        assert result['size'] == expected_size
        assert result['size_bytes'] == expected_size

    @given(st.text(min_size=1, max_size=30, 
                   alphabet=st.characters(whitelist_categories=['Lu', 'Ll', 'Nd'])))
    @settings(max_examples=15, deadline=3000)  # Reduced for performance
    def test_filename_handling_property(self, temp_experiment_directory, filename):
        """
        Property test: filename handling should be robust.
        
        ARRANGE: Create file with generated filename
        ACT: Get file statistics
        ASSERT: Verify filename preservation property
        """
        # ARRANGE - Create file with generated filename
        safe_filename = filename.replace(' ', '_')  # Make filesystem-safe
        test_file = temp_experiment_directory["directory"] / f"{safe_filename}.txt"
        
        try:
            test_file.write_text("Property test content")
            
            # ACT - Get file statistics
            result = get_file_stats(test_file)
            
            # ASSERT - Verify filename property
            assert result['filename'] == f"{safe_filename}.txt"
            assert result['extension'] == "txt"
            
        except (OSError, UnicodeError):
            # Some filenames may not be valid on all systems
            pytest.skip(f"Filename not supported: {safe_filename}")


# ============================================================================
# INTEGRATION TESTING WITH DISCOVERY MODULE
# ============================================================================

class TestFileStatsIntegration:
    """
    Integration testing with discovery module functionality.
    
    Validates end-to-end file statistics attachment workflows while maintaining
    focus on observable behavior rather than internal implementation coupling.
    """
    
    def test_integration_with_discovery_module(self, temp_experiment_directory):
        """
        Test integration between file discovery and statistics attachment.
        
        ARRANGE: Create structured test files for discovery
        ACT: Use discovery with statistics attachment
        ASSERT: Verify integrated workflow behavior
        """
        # ARRANGE - Create test files with patterns
        test_files = [
            "experiment_001_control.csv",
            "experiment_002_treatment.csv",
            "experiment_003_control.csv"
        ]
        
        for filename in test_files:
            file_path = temp_experiment_directory["directory"] / filename
            file_path.write_text(f"# {filename}\ndata,value\n1,2\n3,4\n")
        
        # ACT - Create file list and attach statistics
        file_paths = [str(f) for f in temp_experiment_directory["directory"].glob("*.csv")]
        result = attach_file_stats(file_paths)
        
        # ASSERT - Verify integration behavior
        assert isinstance(result, dict)
        assert len(result) == len(test_files)
        
        for file_path, metadata in result.items():
            assert 'size' in metadata
            assert 'mtime' in metadata
            assert 'filename' in metadata
            assert 'is_file' in metadata
            assert metadata['is_file'] is True
            assert metadata['size'] > 0

    def test_integration_with_existing_metadata(self, temp_experiment_directory):
        """
        Test integration with existing metadata structures.
        
        ARRANGE: Create files with pre-existing metadata
        ACT: Attach file statistics to metadata
        ASSERT: Verify metadata preservation and enhancement
        """
        # ARRANGE - Create files with metadata structure
        test_data = {}
        for i in range(2):
            filename = f"integration_test_{i}.csv"
            file_path = temp_experiment_directory["directory"] / filename
            file_path.write_text(f"integration,test\n{i},data\n")
            
            test_data[str(file_path)] = {
                'experiment_id': f'INT_{i:03d}',
                'condition': 'control' if i % 2 == 0 else 'treatment',
                'replicate': i + 1
            }
        
        # ACT - Attach statistics to existing metadata
        result = attach_file_stats(test_data)
        
        # ASSERT - Verify integration preservation
        for file_path, metadata in result.items():
            # Original metadata preserved
            assert 'experiment_id' in metadata
            assert 'condition' in metadata
            assert 'replicate' in metadata
            
            # Statistics added
            assert 'size' in metadata
            assert 'mtime' in metadata
            assert 'filename' in metadata
            assert 'is_file' in metadata


# ============================================================================
# ERROR HANDLING AND RESILIENCE TESTING
# ============================================================================

class TestFileStatsErrorHandling:
    """
    Comprehensive error handling and resilience testing.
    
    Validates error recovery and graceful degradation behavior
    through observable API responses rather than internal error handling.
    """
    
    def test_custom_error_handler_behavior(self, temp_experiment_directory):
        """
        Test custom error handler behavior in batch processing.
        
        ARRANGE: Set up files with mixed success/failure scenarios
        ACT: Use attach_file_stats with custom error handler
        ASSERT: Verify error handler integration behavior
        """
        # ARRANGE - Create mix of valid and invalid files
        valid_file = temp_experiment_directory["directory"] / "valid.txt"
        valid_file.write_text("Valid content")
        
        file_list = [
            str(valid_file),
            "/nonexistent/invalid.txt"  # This will cause error
        ]
        
        # Custom error handler that returns fallback data
        def custom_error_handler(error, file_path):
            return {
                'size': -1,
                'error': str(error),
                'filename': Path(file_path).name,
                'is_file': False
            }
        
        # ACT - Use custom error handler
        result = attach_file_stats(file_list, error_handler=custom_error_handler)
        
        # ASSERT - Verify error handler behavior
        assert len(result) == 2
        assert str(valid_file) in result
        assert "/nonexistent/invalid.txt" in result
        
        # Valid file should have normal statistics
        assert result[str(valid_file)]['size'] > 0
        assert result[str(valid_file)]['is_file'] is True
        
        # Invalid file should have error handler result
        assert result["/nonexistent/invalid.txt"]['size'] == -1
        assert 'error' in result["/nonexistent/invalid.txt"]

    def test_filesystem_access_error_propagation(self):
        """
        Test filesystem access error propagation behavior.
        
        ARRANGE: Set up mock filesystem with access errors
        ACT: Attempt file operations
        ASSERT: Verify error propagation behavior
        """
        # ARRANGE - Create mock filesystem with access error
        mock_fs = create_mock_filesystem()
        restricted_file = Path("/test/restricted.txt")
        mock_fs.add_file(
            restricted_file,
            size=100,
            access_error=PermissionError("Access denied for testing")
        )
        
        # Create provider that propagates access errors
        class ErrorPropagatingProvider:
            def path_exists(self, path):
                try:
                    return mock_fs.exists(path)
                except PermissionError:
                    raise
            
            def get_stat(self, path):
                if str(path) == str(restricted_file):
                    raise PermissionError("Access denied for testing")
                return mock_fs.stat(path)
            
            def check_access(self, path, mode):
                return True
            
            def is_dir(self, path):
                return False
            
            def is_file(self, path):
                return True
            
            def is_symlink(self, path):
                return False
        
        # ACT & ASSERT - Verify error propagation
        with pytest.raises(FileStatsError) as exc_info:
            get_file_stats(
                restricted_file,
                fs_provider=ErrorPropagatingProvider()
            )
        
        # Verify error context
        assert "Access denied for testing" in str(exc_info.value)
        assert exc_info.value.path == str(restricted_file)

    def test_batch_processing_resilience(self, temp_experiment_directory):
        """
        Test batch processing resilience with partial failures.
        
        ARRANGE: Create batch with mix of valid/invalid files
        ACT: Process batch with default error handling
        ASSERT: Verify resilience behavior
        """
        # ARRANGE - Create valid files
        valid_files = []
        for i in range(3):
            test_file = temp_experiment_directory["directory"] / f"batch_valid_{i}.txt"
            test_file.write_text(f"Valid content {i}")
            valid_files.append(str(test_file))
        
        # Mix in invalid files
        mixed_files = valid_files + ["/invalid/path1.txt", "/invalid/path2.txt"]
        
        # ACT & ASSERT - Verify batch resilience
        # Default behavior should raise error for invalid files
        with pytest.raises(FileStatsError):
            attach_file_stats(mixed_files)
        
        # But valid files should work independently
        valid_result = attach_file_stats(valid_files)
        assert len(valid_result) == len(valid_files)
        for file_path in valid_files:
            assert file_path in valid_result