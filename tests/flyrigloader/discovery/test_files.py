"""
Comprehensive test suite for file discovery functionality.

This module provides modern pytest-based testing for the flyrigloader.discovery.files
module, implementing comprehensive edge case testing, property-based validation,
performance benchmarking, and cross-platform compatibility verification.

Requirements covered:
- F-002-RQ-001: Recursive directory traversal testing
- F-002-RQ-002: Multi-extension filtering validation  
- F-002-RQ-003: Pattern-based file exclusion testing
- F-002-RQ-004: Mandatory substring filtering validation
- F-002-RQ-005: Date-based directory resolution testing
- TST-PERF-001: Performance validation (<5s for 10,000 files)
- TST-MOD-001: Modern pytest fixture usage
- TST-MOD-002: Comprehensive parametrization strategies
- TST-MOD-003: Standardized mocking with pytest-mock
"""

import os
import platform
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock

import pytest
from hypothesis import given, strategies as st, assume, settings

# Import the functions under test
from flyrigloader.discovery.files import discover_files, get_latest_file, FileDiscoverer


class TestFileDiscoveryCore:
    """Core file discovery functionality tests with modern pytest patterns."""

    # =========================================================================
    # Fixture Management - Modern pytest fixture usage (TST-MOD-001)
    # =========================================================================

    @pytest.fixture(scope="function")
    def temp_filesystem(self, tmp_path):
        """
        Create a comprehensive temporary filesystem structure for testing.
        
        Returns:
            Dict[str, Path]: Mapping of logical names to filesystem paths
        """
        # Create directory structure
        root = tmp_path / "test_data"
        subdirs = {
            "root": root,
            "level1": root / "level1",
            "level2": root / "level1" / "level2", 
            "parallel": root / "parallel",
            "dates": root / "dates",
            "batch1": root / "batch1",
            "batch2": root / "batch2"
        }
        
        for dir_path in subdirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        return subdirs

    @pytest.fixture(scope="function") 
    def sample_files(self, temp_filesystem):
        """
        Generate comprehensive test files across the filesystem structure.
        
        Returns:
            Dict[str, List[Path]]: Categorized file paths for test validation
        """
        dirs = temp_filesystem
        files = {
            "txt_files": [],
            "csv_files": [],
            "json_files": [],
            "hidden_files": [],
            "dated_files": [],
            "experiment_files": []
        }
        
        # Standard text files across directories
        txt_data = [
            (dirs["root"] / "root_file_1.txt", "Root content 1"),
            (dirs["root"] / "root_file_2.txt", "Root content 2"),
            (dirs["level1"] / "level1_data.txt", "Level 1 content"),
            (dirs["level2"] / "deep_analysis.txt", "Deep analysis data"),
            (dirs["parallel"] / "parallel_test.txt", "Parallel processing data")
        ]
        
        for file_path, content in txt_data:
            file_path.write_text(content)
            files["txt_files"].append(file_path)
            
        # CSV data files with experimental patterns
        csv_data = [
            (dirs["root"] / "experiment_20240315_control_1.csv", "date,condition,value\n20240315,control,1.0"),
            (dirs["root"] / "experiment_20240316_treatment_1.csv", "date,condition,value\n20240316,treatment,2.0"),
            (dirs["level1"] / "mouse_20240320_baseline_2.csv", "animal,date,condition\nmouse,20240320,baseline"),
            (dirs["batch1"] / "batch1_exp_smoke_2a_001.csv", "batch,experiment,replicate\n1,smoke_2a,1"),
            (dirs["batch2"] / "batch2_exp_nagel_002.csv", "batch,experiment,replicate\n2,nagel,2")
        ]
        
        for file_path, content in csv_data:
            file_path.write_text(content)
            files["csv_files"].append(file_path)
            files["experiment_files"].append(file_path)
            
        # JSON configuration files
        json_data = [
            (dirs["root"] / "config_main.json", '{"type": "main", "version": 1}'),
            (dirs["level1"] / "config_analysis.json", '{"type": "analysis", "params": {}}'),
            (dirs["parallel"] / "settings.json", '{"parallel": true, "workers": 4}')
        ]
        
        for file_path, content in json_data:
            file_path.write_text(content)
            files["json_files"].append(file_path)
            
        # Hidden files and ignore patterns
        hidden_data = [
            (dirs["root"] / "._hidden_system.txt", "Hidden system file"),
            (dirs["level1"] / "._temp_cache.csv", "Temporary cache"),
            (dirs["parallel"] / "temp_processing_data.json", "Temporary processing"),
            (dirs["batch1"] / "static_horiz_ribbon_data.csv", "Static ribbon data")
        ]
        
        for file_path, content in hidden_data:
            file_path.write_text(content)
            files["hidden_files"].append(file_path)
            
        # Date-structured files for temporal testing
        date_data = [
            (dirs["dates"] / "data_20240301.csv", "Early March data"),
            (dirs["dates"] / "data_20240315.csv", "Mid March data"),
            (dirs["dates"] / "data_20240331.csv", "Late March data"),
            (dirs["dates"] / "analysis_2024-04-01.json", "April analysis"),
            (dirs["dates"] / "experiment_v1_20240201.csv", "Version 1"),
            (dirs["dates"] / "experiment_v2_20240215.csv", "Version 2"),
            (dirs["dates"] / "experiment_v3_20240301.csv", "Version 3")
        ]
        
        for file_path, content in date_data:
            file_path.write_text(content)
            files["dated_files"].append(file_path)
            
        return files

    @pytest.fixture(scope="function")
    def performance_filesystem(self, tmp_path):
        """
        Create large filesystem structure for performance testing.
        
        Generates filesystem with configurable number of files for performance validation.
        """
        perf_root = tmp_path / "performance_test"
        perf_root.mkdir()
        
        # Create directory structure with nested levels
        for i in range(50):  # 50 directories
            dir_path = perf_root / f"dir_{i:03d}"
            dir_path.mkdir()
            
            # Create files in each directory
            for j in range(20):  # 20 files per directory = 1000 total files
                file_path = dir_path / f"file_{j:03d}.txt"
                file_path.write_text(f"Content for file {i}_{j}")
                
        return perf_root

    # =========================================================================
    # Basic File Discovery Tests - Parametrized edge cases (TST-MOD-002)
    # =========================================================================

    @pytest.mark.parametrize("pattern,expected_count,recursive", [
        ("*.txt", 1, False),  # Only root level txt files
        ("*.csv", 2, False),  # Only root level csv files  
        ("*.json", 1, False), # Only root level json files
        ("**/*.txt", 5, True), # All txt files recursively
        ("**/*.csv", 5, True), # All csv files recursively
        ("**/*.json", 3, True), # All json files recursively
        ("**/level1*", 2, True), # Files with level1 in name
        ("**/batch*/*", 2, True), # Files in batch directories
    ])
    def test_basic_discovery_patterns(self, temp_filesystem, sample_files, pattern, expected_count, recursive):
        """Test basic file discovery with various glob patterns and recursion settings."""
        result = discover_files(
            directory=str(temp_filesystem["root"]),
            pattern=pattern,
            recursive=recursive
        )
        
        assert len(result) == expected_count, f"Expected {expected_count} files for pattern '{pattern}', got {len(result)}"
        assert all(isinstance(f, str) for f in result), "All results should be string paths"

    @pytest.mark.parametrize("extensions,expected_extensions", [
        (["txt"], [".txt"]),
        (["csv"], [".csv"]), 
        (["json"], [".json"]),
        (["txt", "csv"], [".txt", ".csv"]),
        (["TXT"], [".txt"]),  # Case insensitive
        (["CSV", "Json"], [".csv", ".json"]),  # Mixed case
    ])
    def test_extension_filtering_comprehensive(self, temp_filesystem, sample_files, extensions, expected_extensions):
        """Test extension-based filtering with case insensitivity and multiple extensions."""
        result = discover_files(
            directory=str(temp_filesystem["root"]),
            pattern="**/*",
            recursive=True,
            extensions=extensions
        )
        
        # Verify all returned files have expected extensions
        for file_path in result:
            file_ext = Path(file_path).suffix.lower()
            assert file_ext in expected_extensions, f"File {file_path} has unexpected extension {file_ext}"

    @pytest.mark.parametrize("ignore_patterns,should_exclude", [
        (["._*"], ["._hidden_system.txt", "._temp_cache.csv"]),
        (["temp*"], ["temp_processing_data.json"]),
        (["*ribbon*"], ["static_horiz_ribbon_data.csv"]),
        (["._*", "temp*"], ["._hidden_system.txt", "._temp_cache.csv", "temp_processing_data.json"]),
        (["*smoke_2a*"], ["batch1_exp_smoke_2a_001.csv"]),
    ])
    def test_ignore_patterns_globbing(self, temp_filesystem, sample_files, ignore_patterns, should_exclude):
        """Test ignore patterns using glob matching instead of substring matching."""
        result = discover_files(
            directory=str(temp_filesystem["root"]),
            pattern="**/*",
            recursive=True,
            ignore_patterns=ignore_patterns
        )
        
        # Verify excluded files are not in results
        result_basenames = [Path(f).name for f in result]
        for excluded_file in should_exclude:
            assert excluded_file not in result_basenames, f"File {excluded_file} should be excluded but was found"

    @pytest.mark.parametrize("mandatory_substrings,should_include", [
        (["experiment"], ["experiment_20240315_control_1.csv", "experiment_20240316_treatment_1.csv"]),
        (["mouse"], ["mouse_20240320_baseline_2.csv"]),
        (["batch1"], ["batch1_exp_smoke_2a_001.csv"]),
        (["smoke_2a", "nagel"], ["batch1_exp_smoke_2a_001.csv", "batch2_exp_nagel_002.csv"]),  # OR logic
    ])
    def test_mandatory_substrings_filtering(self, temp_filesystem, sample_files, mandatory_substrings, should_include):
        """Test mandatory substring filtering with OR logic."""
        result = discover_files(
            directory=str(temp_filesystem["root"]),
            pattern="**/*",
            recursive=True,
            mandatory_substrings=mandatory_substrings
        )
        
        result_basenames = [Path(f).name for f in result]
        for required_file in should_include:
            assert required_file in result_basenames, f"File {required_file} should be included but was not found"

    # =========================================================================
    # Advanced Pattern Extraction and Metadata Tests
    # =========================================================================

    @pytest.mark.parametrize("extract_patterns,expected_metadata_keys", [
        ([r".*/(?P<animal>mouse)_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.csv"],
         ["animal", "date", "condition", "replicate"]),
        ([r".*/(?P<date>\d{8})_(?P<animal>rat)_(?P<condition>\w+)_(?P<replicate>\d+)\.csv"],
         ["date", "animal", "condition", "replicate"]),
        ([r".*/(?P<experiment_id>exp\d+)_(?P<animal>\w+)_(?P<condition>\w+)\.csv"],
         ["experiment_id", "animal", "condition"]),
    ])
    def test_pattern_extraction_metadata(self, temp_filesystem, extract_patterns, expected_metadata_keys):
        """Test metadata extraction from filenames using regex patterns."""
        # Create test files matching the patterns
        test_files = [
            temp_filesystem["root"] / "mouse_20240315_control_1.csv",
            temp_filesystem["root"] / "20240316_rat_treatment_2.csv", 
            temp_filesystem["root"] / "exp001_mouse_baseline.csv"
        ]
        
        for file_path in test_files:
            file_path.write_text("test,data\n1,2")
            
        result = discover_files(
            directory=str(temp_filesystem["root"]),
            pattern="*.csv",
            extract_patterns=extract_patterns
        )
        
        assert isinstance(result, dict), "Result should be dictionary when extract_patterns is provided"
        
        # Check that at least one file has the expected metadata
        metadata_found = False
        for file_path, metadata in result.items():
            if any(key in metadata for key in expected_metadata_keys):
                metadata_found = True
                break
                
        assert metadata_found, f"Expected metadata keys {expected_metadata_keys} not found in any file"

    def test_multiple_extract_patterns_combined(self, tmp_path):
        folder = tmp_path / "smoke_1a_vial2_vflipTrue_PSI40_intensity1_0"
        folder.mkdir()
        file_path = folder / "exp_matrix.pklz"
        file_path.write_text("dummy")

        patterns = [
            r"parent::^(?P<experiment_type>[^_]+)",
            r"parent::^[^_]+_(?P<genotype>[0-9]+[a-zA-Z])",
            r"parent::.*vial(?P<vial>\d+)",
            r"parent::.*vflip(?P<vflip>True|False)",
            r"parent::.*PSI(?P<PSI>\d+)",
            r"parent::.*intensity(?P<intensity>\d+)",
            r"parent::.*_(?P<replicate>\d+)$",
        ]

        result = discover_files(
            directory=str(tmp_path),
            pattern="*.pklz",
            extract_patterns=patterns,
        )

        assert isinstance(result, dict)
        md = result[str(file_path)]
        assert md["experiment_type"] == "smoke"
        assert md["genotype"] == "1a"
        assert md["vial"] == "2"
        assert md["vflip"] == "True"
        assert md["PSI"] == "40"
        assert md["intensity"] == "1"
        assert md["replicate"] == "0"

    def test_date_parsing_comprehensive(self, temp_filesystem):
        """Test date parsing from various filename formats."""
        # Create files with different date formats
        date_files = [
            temp_filesystem["root"] / "data_20240315.csv",  # YYYYMMDD
            temp_filesystem["root"] / "data_2024-03-16.csv",  # YYYY-MM-DD  
            temp_filesystem["root"] / "data_03-17-2024.csv",  # MM-DD-YYYY
            temp_filesystem["root"] / "data_20240318_142030.csv",  # YYYYMMDD_HHMMSS
        ]
        
        for file_path in date_files:
            file_path.write_text("date,value\n2024-03-15,1.0")
            
        result = discover_files(
            directory=str(temp_filesystem["root"]),
            pattern="*.csv",
            parse_dates=True
        )
        
        assert isinstance(result, dict), "Result should be dictionary when parse_dates is True"
        
        # Verify all files have parsed_date
        for file_path, metadata in result.items():
            if "data_" in Path(file_path).name:
                assert "parsed_date" in metadata, f"File {file_path} should have parsed_date"
                assert isinstance(metadata["parsed_date"], datetime), "parsed_date should be datetime object"

    # =========================================================================
    # Property-Based Testing with Hypothesis (Section 3.6.3)
    # =========================================================================

    @given(
        num_files=st.integers(min_value=1, max_value=100),
        num_dirs=st.integers(min_value=1, max_value=10),
        file_extensions=st.lists(st.sampled_from([".txt", ".csv", ".json", ".pkl"]), min_size=1, max_size=3)
    )
    @settings(max_examples=20, deadline=5000)
    def test_discovery_property_based(self, tmp_path, num_files, num_dirs, file_extensions):
        """Property-based testing for robust validation across diverse directory structures."""
        assume(num_files > 0 and num_dirs > 0)
        
        # Generate random filesystem structure
        root = tmp_path / "property_test"
        root.mkdir()
        
        created_files = []
        for i in range(num_dirs):
            dir_path = root / f"dir_{i}"
            dir_path.mkdir()
            
            files_per_dir = max(1, num_files // num_dirs)
            for j in range(files_per_dir):
                ext = file_extensions[j % len(file_extensions)]
                file_path = dir_path / f"file_{j}{ext}"
                file_path.write_text(f"Content {i}_{j}")
                created_files.append(file_path)
                
        # Test discovery with various patterns
        all_files = discover_files(str(root), "**/*", recursive=True)
        
        # Properties that should always hold
        assert len(all_files) > 0, "Should discover at least some files"
        assert len(all_files) <= len(created_files), "Should not discover more files than created"
        assert all(os.path.exists(f) for f in all_files), "All discovered files should exist"

    @given(
        ignore_pattern=st.text(alphabet="abcdefghijklmnopqrstuvwxyz_*", min_size=1, max_size=10),
        mandatory_substring=st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=8)
    )
    @settings(max_examples=15, deadline=3000)
    def test_filtering_properties(self, tmp_path, ignore_pattern, mandatory_substring):
        """Property-based testing for filtering logic consistency."""
        assume("*" in ignore_pattern or len(ignore_pattern) > 2)
        assume(ignore_pattern != mandatory_substring)
        
        # Create test files
        root = tmp_path / "filter_test" 
        root.mkdir()
        
        # Create files that should match/not match patterns
        test_files = [
            root / f"match_{mandatory_substring}_file.txt",
            root / f"nomatch_file.txt",
            root / f"{ignore_pattern.replace('*', 'ignore')}_file.txt"
        ]
        
        for file_path in test_files:
            file_path.write_text("test content")
            
        # Test with ignore patterns
        ignored_result = discover_files(
            str(root),
            "*.txt", 
            ignore_patterns=[ignore_pattern]
        )
        
        # Test with mandatory substrings
        mandatory_result = discover_files(
            str(root),
            "*.txt",
            mandatory_substrings=[mandatory_substring]
        )
        
        # Properties
        assert len(ignored_result) <= 3, "Ignore patterns should not increase file count"
        assert len(mandatory_result) <= 3, "Mandatory patterns should not increase file count"

    # =========================================================================
    # Cross-Platform Compatibility Tests (Section 3.6.1)
    # =========================================================================

    @pytest.mark.parametrize("path_style", ["posix", "windows"])
    def test_cross_platform_path_handling(self, tmp_path, path_style, mocker):
        """Test cross-platform path handling for Windows, Linux, and macOS."""
        # Mock platform-specific path handling
        if path_style == "windows":
            mocker.patch("os.sep", "\\")
            # Removed patch to pathlib.Path._flavour which caused teardown errors in pytest
        else:
            mocker.patch("os.sep", "/")
            
        # Create cross-platform test structure
        root = tmp_path / "cross_platform"
        nested = root / "nested" / "deep"
        nested.mkdir(parents=True)
        
        test_file = nested / "test_file.txt"
        test_file.write_text("cross-platform content")
        
        # Test discovery works regardless of platform
        result = discover_files(str(root), "**/*.txt", recursive=True)
        
        assert len(result) == 1, "Should find file regardless of platform path style"
        assert os.path.exists(result[0]), "Discovered path should be valid on current platform"

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix-specific permissions test")
    def test_permission_handling_unix(self, tmp_path):
        """Test file discovery with restricted permissions on Unix systems."""
        # Create directory with restricted permissions
        restricted_dir = tmp_path / "restricted"
        restricted_dir.mkdir(mode=0o700)
        
        test_file = restricted_dir / "test.txt"
        test_file.write_text("restricted content")
        
        # Change permissions
        restricted_dir.chmod(0o000)
        
        try:
            # Should handle permission errors gracefully
            result = discover_files(str(tmp_path), "**/*.txt", recursive=True)
            # May or may not find the file depending on system, but shouldn't crash
            assert isinstance(result, list), "Should return list even with permission issues"
        finally:
            # Restore permissions for cleanup
            restricted_dir.chmod(0o700)

    # =========================================================================
    # Performance Validation Tests (TST-PERF-001)
    # =========================================================================

    @pytest.mark.benchmark(group="file_discovery")
    def test_discovery_performance_small_dataset(self, performance_filesystem, benchmark):
        """Benchmark file discovery performance for small datasets."""
        result = benchmark(
            discover_files,
            str(performance_filesystem),
            "**/*.txt",
            recursive=True
        )
        
        assert len(result) == 1000, "Should discover all 1000 test files"
        # pytest-benchmark will automatically validate performance

    @pytest.mark.benchmark(group="file_discovery") 
    @pytest.mark.slow
    def test_discovery_performance_large_dataset(self, tmp_path, benchmark):
        """Benchmark file discovery for large datasets approaching TST-PERF-001 requirements."""
        # Create larger dataset for performance testing (5000 files)
        large_root = tmp_path / "large_performance"
        large_root.mkdir()
        
        for i in range(100):  # 100 directories
            dir_path = large_root / f"large_dir_{i:03d}"
            dir_path.mkdir()
            
            for j in range(50):  # 50 files per directory = 5000 total
                file_path = dir_path / f"large_file_{j:03d}.txt"
                file_path.write_text(f"Large dataset content {i}_{j}")
                
        # Benchmark the discovery
        result = benchmark(
            discover_files,
            str(large_root),
            "**/*.txt", 
            recursive=True
        )
        
        assert len(result) == 5000, "Should discover all 5000 test files"

    def test_performance_sla_compliance(self, tmp_path):
        """Validate that file discovery meets TST-PERF-001 SLA (<5s for 10,000 files)."""
        # Note: Creating 10,000 actual files would be slow for regular testing
        # This test validates the SLA requirement with a smaller representative dataset
        perf_root = tmp_path / "sla_test"
        perf_root.mkdir()
        
        # Create 500 files as representative sample
        for i in range(50):
            dir_path = perf_root / f"sla_dir_{i}"
            dir_path.mkdir()
            for j in range(10):
                file_path = dir_path / f"sla_file_{j}.txt"
                file_path.write_text(f"SLA test content {i}_{j}")
                
        start_time = time.time()
        result = discover_files(str(perf_root), "**/*.txt", recursive=True)
        elapsed_time = time.time() - start_time
        
        assert len(result) == 500, "Should discover all test files"
        
        # Extrapolate performance: if 500 files take X seconds, 10,000 should take ~20X seconds
        estimated_time_for_10k = elapsed_time * 20
        assert estimated_time_for_10k < 5.0, f"Estimated time for 10,000 files ({estimated_time_for_10k:.2f}s) exceeds 5s SLA"

    # =========================================================================
    # Mock-Based Testing (TST-MOD-003)
    # =========================================================================

    def test_filesystem_mocking_with_pytest_mock(self, mocker):
        """Test file discovery with mocked filesystem operations."""
        # Mock pathlib.Path.glob to return controlled results
        mock_glob = mocker.patch("pathlib.Path.glob")
        mock_glob.return_value = [
            Path("/mocked/path/file1.txt"),
            Path("/mocked/path/file2.csv"),
            Path("/mocked/path/file3.json")
        ]
        
        # Mock pathlib.Path.rglob for recursive operations
        mock_rglob = mocker.patch("pathlib.Path.rglob") 
        mock_rglob.return_value = [
            Path("/mocked/path/deep/file4.txt"),
            Path("/mocked/path/deep/file5.csv")
        ]
        
        # Test non-recursive discovery
        result = discover_files("/mocked/path", "*.txt")
        mock_glob.assert_called_once_with("*.txt")
        assert len(result) == 3, "Should return all mocked files"
        
        # Reset mocks
        mock_glob.reset_mock()
        
        # Test recursive discovery
        result_recursive = discover_files("/mocked/path", "*.txt", recursive=True)
        mock_rglob.assert_called_once_with("*.txt")
        assert len(result_recursive) == 2, "Should return recursive mocked files"

    def test_error_handling_with_mocks(self, mocker):
        """Test error handling scenarios using pytest-mock."""
        # Mock filesystem operations to raise exceptions
        mock_glob = mocker.patch("pathlib.Path.glob")
        mock_glob.side_effect = PermissionError("Mocked permission denied")
        
        # Test that permission errors are handled gracefully
        with pytest.raises(PermissionError):
            discover_files("/restricted/path", "*.txt")

    def test_file_stats_mocking(self, mocker):
        """Test file statistics collection with mocked filesystem."""
        # Mock Path.stat() to return controlled file stats
        mock_stat = mocker.MagicMock()
        mock_stat.st_mtime = 1640995200.0  # Fixed timestamp
        mock_stat.st_size = 1024  # Fixed file size
        
        mocker.patch("pathlib.Path.stat", return_value=mock_stat)
        mocker.patch("pathlib.Path.glob", return_value=[Path("/mocked/file.txt")])
        
        # Test that stats are properly integrated
        result = discover_files("/mocked", "*.txt", include_stats=True)
        
        assert isinstance(result, dict), "Should return dict when include_stats=True"

    # =========================================================================
    # Get Latest File Tests
    # =========================================================================

    @pytest.mark.parametrize("file_count,time_delays", [
        (2, [0.01, 0.02]),
        (3, [0.01, 0.02, 0.03]),
        (5, [0.01, 0.02, 0.03, 0.04, 0.05])
    ])
    def test_get_latest_file_timing(self, tmp_path, file_count, time_delays):
        """Test get_latest_file with various file counts and timing scenarios."""
        files = []
        
        for i, delay in enumerate(time_delays):
            if i > 0:
                time.sleep(delay)
                
            file_path = tmp_path / f"timed_file_{i}.txt"
            file_path.write_text(f"Content {i}")
            files.append(str(file_path))
            
        latest = get_latest_file(files)
        
        # Should return the last created file
        assert latest == files[-1], f"Expected {files[-1]}, got {latest}"

    def test_get_latest_file_edge_cases(self):
        """Test get_latest_file edge cases and error conditions."""
        # Empty list
        assert get_latest_file([]) is None, "Empty list should return None"
        
        # Single file
        single_file = ["/path/to/single/file.txt"]
        with pytest.raises((FileNotFoundError, OSError)):
            # Should raise error for non-existent file
            get_latest_file(single_file)

    def test_get_latest_file_with_real_files(self, tmp_path):
        """Test get_latest_file with real files and modification times."""
        # Create files with different modification times
        file1 = tmp_path / "old_file.txt"
        file1.write_text("Old content")
        
        time.sleep(0.01)  # Ensure different modification times
        
        file2 = tmp_path / "new_file.txt" 
        file2.write_text("New content")
        
        files = [str(file1), str(file2)]
        latest = get_latest_file(files)
        
        assert latest == str(file2), "Should return the newer file"

    # =========================================================================
    # Integration Tests with FileDiscoverer Class
    # =========================================================================

    def test_file_discoverer_class_integration(self, temp_filesystem, sample_files):
        """Test FileDiscoverer class with various configuration options."""
        # Test with extract patterns
        patterns = [r".*/(?P<animal>mouse)_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.csv"]
        discoverer = FileDiscoverer(extract_patterns=patterns)
        
        # Create mouse file for pattern matching
        mouse_file = temp_filesystem["root"] / "mouse_20240315_control_1.csv"
        mouse_file.write_text("test,data\n1,2")
        
        result = discoverer.discover(
            str(temp_filesystem["root"]),
            "*.csv"
        )
        
        assert isinstance(result, dict), "Should return dict when patterns configured"
        
        # Test with date parsing
        date_discoverer = FileDiscoverer(parse_dates=True)
        date_result = date_discoverer.discover(
            str(temp_filesystem["root"]),
            "*.csv"
        )
        
        assert isinstance(date_result, dict), "Should return dict when parse_dates=True"

    def test_file_discoverer_combined_options(self, temp_filesystem):
        """Test FileDiscoverer with multiple options combined.""" 
        # Create test files
        test_files = [
            temp_filesystem["root"] / "exp001_mouse_20240315.csv",
            temp_filesystem["root"] / "exp002_rat_20240316.csv"
        ]
        
        for file_path in test_files:
            file_path.write_text("experiment,animal,date\ntest,mouse,20240315")
            
        # Configure discoverer with all options
        discoverer = FileDiscoverer(
            extract_patterns=[r".*/(?P<experiment_id>exp\d+)_(?P<animal>\w+)_(?P<date>\d{8})\.csv"],
            parse_dates=True,
            include_stats=True
        )
        
        result = discoverer.discover(
            str(temp_filesystem["root"]),
            "*.csv"
        )
        
        assert isinstance(result, dict), "Should return dict with combined options"
        
        # Verify metadata structure
        for file_path, metadata in result.items():
            assert "path" in metadata, "Should include path in metadata"
            if "exp0" in Path(file_path).name:
                assert "parsed_date" in metadata, "Should include parsed date"
                # File stats should be included
                assert any(key in metadata for key in ["size", "mtime", "ctime"]), "Should include file stats"

    # =========================================================================
    # Boundary Condition and Error Tests  
    # =========================================================================

    def test_boundary_conditions(self, tmp_path):
        """Test boundary conditions and edge cases."""
        # Empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        result = discover_files(str(empty_dir), "*", recursive=True)
        assert result == [], "Empty directory should return empty list"
        
        # Non-existent directory
        with pytest.raises((FileNotFoundError, OSError)):
            discover_files("/non/existent/path", "*")
            
        # Invalid pattern
        result_invalid = discover_files(str(empty_dir), "", recursive=True) 
        assert result_invalid == [], "Invalid pattern should return empty list"

    @pytest.mark.parametrize("invalid_input", [
        None,
        123, 
        [],
        {}
    ])
    def test_invalid_input_handling(self, invalid_input):
        """Test handling of invalid input parameters."""
        with pytest.raises((TypeError, ValueError, AttributeError)):
            discover_files(invalid_input, "*")

    def test_large_filename_handling(self, tmp_path):
        """Test handling of files with very long names."""
        # Create file with long name (within filesystem limits)
        long_name = "a" * 200 + ".txt"
        long_file = tmp_path / long_name
        long_file.write_text("Long filename content")
        
        result = discover_files(str(tmp_path), "*.txt")
        assert len(result) == 1, "Should handle long filenames"
        assert long_name in result[0], "Should find file with long name"

    def test_unicode_filename_handling(self, tmp_path):
        """Test handling of files with unicode characters."""
        # Create files with unicode names
        unicode_files = [
            tmp_path / "测试文件.txt",  # Chinese characters
            tmp_path / "файл_тест.txt",  # Cyrillic characters
            tmp_path / "archivo_español.txt",  # Spanish characters
            tmp_path / "émoji_🎉_file.txt"  # Emoji characters
        ]

        for file_path in unicode_files:
            try:
                file_path.write_text("Unicode content")
            except (UnicodeError, OSError):
                # Skip if filesystem doesn't support unicode
                continue

        result = discover_files(str(tmp_path), "*.txt")
        # Should handle unicode filenames without crashing
        assert isinstance(result, list), "Should return list for unicode filenames"

    def test_pathlike_directory_inputs(self, tmp_path):
        """Ensure discover_files accepts Path objects and Path iterables."""
        txt_files = [
            tmp_path / "alpha.txt",
            tmp_path / "beta.txt"
        ]

        for file_path in txt_files:
            file_path.write_text("sample data")

        expected = {str(path) for path in txt_files}

        # Single Path instance
        single_path_result = discover_files(tmp_path, "*.txt")
        assert set(single_path_result) == expected, "Single Path input should discover all files"

        # Iterable of Path instances
        list_path_result = discover_files([tmp_path], "*.txt")
        assert set(list_path_result) == expected, "Iterable Path input should discover all files"


class TestFileDiscoverySpecialCases:
    """Special case testing for complex scenarios and edge conditions."""

    def test_symlink_handling(self, tmp_path):
        """Test file discovery with symbolic links.""" 
        if platform.system() == "Windows":
            pytest.skip("Symlink test requires Unix-like system")
            
        # Create original file and symlink
        original = tmp_path / "original.txt"
        original.write_text("Original content")
        
        symlink = tmp_path / "link.txt"
        symlink.symlink_to(original)
        
        result = discover_files(str(tmp_path), "*.txt")
        
        # Should handle symlinks appropriately
        assert len(result) >= 1, "Should discover files through symlinks"

    def test_concurrent_file_modification(self, tmp_path):
        """Test file discovery behavior during concurrent file operations."""
        import threading
        import time
        
        def create_files():
            """Create files during discovery operation."""
            time.sleep(0.01)  # Small delay
            for i in range(5):
                file_path = tmp_path / f"concurrent_{i}.txt"
                file_path.write_text(f"Concurrent content {i}")
                time.sleep(0.001)
                
        # Start file creation in background
        thread = threading.Thread(target=create_files)
        thread.start()
        
        # Perform discovery while files are being created
        result = discover_files(str(tmp_path), "*.txt", recursive=True)
        
        thread.join()  # Wait for completion
        
        # Should handle concurrent operations gracefully
        assert isinstance(result, list), "Should return list even during concurrent operations"

    def test_very_deep_directory_structure(self, tmp_path):
        """Test discovery in very deeply nested directory structures."""
        # Create deep nested structure (10 levels)
        current = tmp_path
        for i in range(10):
            current = current / f"level_{i}"
            current.mkdir()
            
        # Create file at deepest level
        deep_file = current / "deep_file.txt"
        deep_file.write_text("Deep content")
        
        result = discover_files(str(tmp_path), "**/*.txt", recursive=True)
        
        assert len(result) == 1, "Should find file in deeply nested structure"
        assert "deep_file.txt" in result[0], "Should find correct deep file"

    @pytest.mark.parametrize("special_chars", [
        "file with spaces.txt",
        "file-with-dashes.txt", 
        "file_with_underscores.txt",
        "file.with.dots.txt",
        "file(with)parentheses.txt",
        "file[with]brackets.txt"
    ])
    def test_special_character_filenames(self, tmp_path, special_chars):
        """Test discovery of files with special characters in names."""
        special_file = tmp_path / special_chars
        special_file.write_text("Special character content")
        
        result = discover_files(str(tmp_path), "*.txt")
        
        assert len(result) == 1, f"Should find file with special characters: {special_chars}"
        assert special_chars in result[0], "Should find correct special character file"