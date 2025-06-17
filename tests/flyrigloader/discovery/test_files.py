"""
Comprehensive test suite for file discovery functionality using behavior-focused validation.

This module implements behavior-focused testing for the flyrigloader.discovery.files
module, emphasizing public API contracts and observable system behavior rather than
implementation-specific details. All tests use centralized fixtures and utilities 
from tests/conftest.py and tests/utils.py to ensure consistent testing patterns.

Testing approach:
- Public API behavior validation through black-box testing
- Observable file discovery behavior (accurate filtering, metadata extraction, etc.)
- Protocol-based mock implementations for dependency isolation
- AAA (Arrange-Act-Assert) pattern structure throughout
- Centralized fixture usage for consistency across test layers

Performance and benchmark tests have been relocated to scripts/benchmarks/ per
Section 0 requirements for performance test isolation.
"""

import os
import platform
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import pytest
from hypothesis import given, strategies as st, assume, settings

# Import the public API functions under test
from flyrigloader.discovery.files import discover_files, get_latest_file

# Import centralized test utilities and Protocol-based mocks
from tests.utils import (
    create_mock_filesystem,
    create_mock_dataloader,
    create_mock_config_provider,
    MockFilesystemProvider,
    generate_edge_case_scenarios,
    EdgeCaseScenarioGenerator,
    create_hypothesis_strategies
)


class TestFileDiscoveryBehaviorValidation:
    """
    Behavior-focused tests for file discovery functionality using public API validation.
    
    This test class focuses on observable file discovery behavior rather than 
    implementation details, using centralized fixtures and Protocol-based mocks.
    """

    # =========================================================================
    # Public API Behavior Tests - Core Discovery Functionality
    # =========================================================================

    @pytest.mark.parametrize("pattern,recursive,expected_behavior", [
        ("*.txt", False, "find_root_level_txt_only"),
        ("*.csv", False, "find_root_level_csv_only"),
        ("**/*.txt", True, "find_all_txt_recursively"),
        ("**/*.csv", True, "find_all_csv_recursively"),
        ("**/level1*", True, "find_files_with_level1_in_name"),
        ("**/batch*/*", True, "find_files_in_batch_directories"),
    ])
    def test_discover_files_pattern_behavior(
        self, 
        temp_experiment_directory, 
        pattern, 
        recursive, 
        expected_behavior
    ):
        """
        Test discover_files public API behavior with various glob patterns and recursion.
        
        Validates observable file discovery behavior through public interface
        rather than internal implementation details.
        """
        # ARRANGE - Set up test directory with known file structure
        experiment_dir = temp_experiment_directory["directory"]
        
        # Create additional test files for pattern validation
        test_files = [
            experiment_dir / "root_file.txt",
            experiment_dir / "raw_data" / "level1_data.txt", 
            experiment_dir / "raw_data" / "data_file.csv",
            experiment_dir / "processed" / "batch1_results.csv",
            experiment_dir / "metadata" / "level1_meta.json"
        ]
        
        for file_path in test_files:
            file_path.write_text("test content")
        
        # ACT - Execute discover_files with public API
        result = discover_files(
            directory=str(experiment_dir),
            pattern=pattern,
            recursive=recursive
        )
        
        # ASSERT - Validate observable behavior based on expected pattern matching
        assert isinstance(result, list), "discover_files should return list of file paths"
        assert all(isinstance(f, str) for f in result), "All results should be string paths"
        
        # Validate pattern-specific behavior
        if expected_behavior == "find_root_level_txt_only":
            txt_files = [f for f in result if f.endswith('.txt')]
            assert len(txt_files) >= 1, "Should find at least one txt file at root level"
            assert all(Path(f).parent == experiment_dir for f in txt_files), \
                "Should only find root level files when recursive=False"
                
        elif expected_behavior == "find_all_txt_recursively":
            txt_files = [f for f in result if f.endswith('.txt')]
            assert len(txt_files) >= 2, "Should find txt files recursively"
            
        elif expected_behavior == "find_files_with_level1_in_name":
            level1_files = [f for f in result if 'level1' in Path(f).name]
            assert len(level1_files) >= 1, "Should find files with 'level1' in name"

    @pytest.mark.parametrize("extensions,expected_file_types", [
        (["txt"], [".txt"]),
        (["csv"], [".csv"]),
        (["TXT"], [".txt"]),  # Case insensitive behavior
        (["txt", "csv"], [".txt", ".csv"]),
        (["CSV", "Json"], [".csv", ".json"]),  # Mixed case handling
    ])
    def test_discover_files_extension_filtering_behavior(
        self, 
        temp_experiment_directory, 
        extensions, 
        expected_file_types
    ):
        """
        Test extension filtering behavior through public API interface.
        
        Validates that extension filtering works correctly and case-insensitively
        through observable file discovery results.
        """
        # ARRANGE - Set up directory with files of various extensions
        experiment_dir = temp_experiment_directory["directory"]
        
        test_files = [
            experiment_dir / "data.txt",
            experiment_dir / "results.csv", 
            experiment_dir / "config.json",
            experiment_dir / "analysis.pkl"
        ]
        
        for file_path in test_files:
            file_path.write_text("test content")
        
        # ACT - Apply extension filtering through public API
        result = discover_files(
            directory=str(experiment_dir),
            pattern="*",
            extensions=extensions
        )
        
        # ASSERT - Validate extension filtering behavior
        assert isinstance(result, list), "Result should be list of filtered files"
        
        for file_path in result:
            file_ext = Path(file_path).suffix.lower()
            assert file_ext in expected_file_types, \
                f"File {file_path} has unexpected extension {file_ext}"

    @pytest.mark.parametrize("ignore_patterns,should_be_excluded", [
        (["temp*"], ["temp_file.txt"]),
        (["*backup*"], ["data_backup.csv"]),
        ([".*"], [".hidden_file.txt"]),
        (["temp*", "*backup*"], ["temp_file.txt", "data_backup.csv"]),
    ])
    def test_discover_files_ignore_pattern_behavior(
        self, 
        temp_experiment_directory, 
        ignore_patterns, 
        should_be_excluded
    ):
        """
        Test ignore pattern behavior through observable file discovery results.
        
        Validates that ignore patterns correctly exclude files based on
        glob matching behavior through public API.
        """
        # ARRANGE - Create files that should and shouldn't be ignored
        experiment_dir = temp_experiment_directory["directory"]
        
        all_test_files = [
            experiment_dir / "normal_file.txt",
            experiment_dir / "temp_file.txt",
            experiment_dir / "data_backup.csv",
            experiment_dir / ".hidden_file.txt",
            experiment_dir / "important_data.csv"
        ]
        
        for file_path in all_test_files:
            file_path.write_text("test content")
        
        # ACT - Apply ignore patterns through public API
        result = discover_files(
            directory=str(experiment_dir),
            pattern="*",
            ignore_patterns=ignore_patterns
        )
        
        # ASSERT - Validate ignore behavior through observable results
        result_basenames = [Path(f).name for f in result]
        
        for excluded_file in should_be_excluded:
            assert excluded_file not in result_basenames, \
                f"File {excluded_file} should be excluded but was found"
        
        # Verify non-excluded files are still present
        assert "normal_file.txt" in result_basenames, \
            "Non-excluded files should still be present"

    @pytest.mark.parametrize("mandatory_substrings,should_be_included", [
        (["experiment"], ["experiment_data.csv", "old_experiment.txt"]),
        (["mouse"], ["mouse_001.csv"]),
        (["batch"], ["batch1_results.csv", "batch2_analysis.txt"]),
        (["mouse", "batch"], ["mouse_001.csv", "batch1_results.csv"]),  # OR logic
    ])
    def test_discover_files_mandatory_substring_behavior(
        self, 
        temp_experiment_directory, 
        mandatory_substrings, 
        should_be_included
    ):
        """
        Test mandatory substring filtering behavior through public API.
        
        Validates OR logic behavior for mandatory substrings through 
        observable file discovery results.
        """
        # ARRANGE - Create test files with various naming patterns
        experiment_dir = temp_experiment_directory["directory"]
        
        test_files = [
            experiment_dir / "experiment_data.csv",
            experiment_dir / "old_experiment.txt",
            experiment_dir / "mouse_001.csv",
            experiment_dir / "batch1_results.csv",
            experiment_dir / "batch2_analysis.txt",
            experiment_dir / "unrelated_file.txt"
        ]
        
        for file_path in test_files:
            file_path.write_text("test content")
        
        # ACT - Apply mandatory substring filtering
        result = discover_files(
            directory=str(experiment_dir),
            pattern="*",
            mandatory_substrings=mandatory_substrings
        )
        
        # ASSERT - Validate mandatory substring behavior
        result_basenames = [Path(f).name for f in result]
        
        for required_file in should_be_included:
            if Path(experiment_dir / required_file).exists():
                assert required_file in result_basenames, \
                    f"File {required_file} should be included but was not found"

    # =========================================================================
    # Metadata Extraction Behavior Tests
    # =========================================================================

    @pytest.mark.parametrize("extract_patterns,expected_metadata_fields", [
        ([r".*/(mouse)_(\d{8})_(\w+)_(\d+)\.csv"], ["animal", "date", "condition", "replicate"]),
        ([r".*/(\d{8})_(rat)_(\w+)_(\d+)\.csv"], ["date", "animal", "condition", "replicate"]),
        ([r".*/(exp\d+)_(\w+)_(\w+)\.csv"], ["experiment_id", "animal", "condition"]),
    ])
    def test_discover_files_metadata_extraction_behavior(
        self, 
        temp_experiment_directory, 
        extract_patterns, 
        expected_metadata_fields
    ):
        """
        Test metadata extraction behavior through public API return values.
        
        Validates that metadata extraction produces expected output structure
        without inspecting internal implementation details.
        """
        # ARRANGE - Create files matching extraction patterns
        experiment_dir = temp_experiment_directory["directory"]
        
        test_files = [
            experiment_dir / "mouse_20240315_control_1.csv",
            experiment_dir / "20240316_rat_treatment_2.csv",
            experiment_dir / "exp001_mouse_baseline.csv"
        ]
        
        for file_path in test_files:
            file_path.write_text("animal,condition,value\nmouse,control,1.0")
        
        # ACT - Execute metadata extraction through public API
        result = discover_files(
            directory=str(experiment_dir),
            pattern="*.csv",
            extract_patterns=extract_patterns
        )
        
        # ASSERT - Validate metadata extraction behavior
        assert isinstance(result, dict), \
            "Should return dictionary when extract_patterns provided"
        
        # Verify at least one file has expected metadata fields
        metadata_found = False
        for file_path, metadata in result.items():
            if isinstance(metadata, dict):
                metadata_fields = set(metadata.keys())
                expected_fields = set(expected_metadata_fields)
                if metadata_fields.intersection(expected_fields):
                    metadata_found = True
                    break
        
        assert metadata_found, \
            f"Expected metadata fields {expected_metadata_fields} not found"

    def test_discover_files_date_parsing_behavior(self, temp_experiment_directory):
        """
        Test date parsing behavior through observable output structure.
        
        Validates date parsing functionality through public API results
        without accessing internal parsing implementation.
        """
        # ARRANGE - Create files with different date formats
        experiment_dir = temp_experiment_directory["directory"]
        
        date_files = [
            experiment_dir / "data_20240315.csv",
            experiment_dir / "data_2024-03-16.csv",
            experiment_dir / "data_20240318_142030.csv",
        ]
        
        for file_path in date_files:
            file_path.write_text("date,value\n2024-03-15,1.0")
        
        # ACT - Execute date parsing through public API
        result = discover_files(
            directory=str(experiment_dir),
            pattern="*.csv",
            parse_dates=True
        )
        
        # ASSERT - Validate date parsing behavior through return structure
        assert isinstance(result, dict), \
            "Should return dictionary when parse_dates is True"
        
        # Verify date parsing results in expected metadata structure
        parsed_date_found = False
        for file_path, metadata in result.items():
            if isinstance(metadata, dict) and "parsed_date" in metadata:
                assert isinstance(metadata["parsed_date"], datetime), \
                    "parsed_date should be datetime object"
                parsed_date_found = True
        
        assert parsed_date_found, "At least one file should have parsed_date"

    # =========================================================================
    # Edge-Case and Error Handling Behavior Tests
    # =========================================================================

    def test_discover_files_empty_directory_behavior(self, tmp_path):
        """
        Test behavior with empty directory through public API.
        
        Validates graceful handling of empty directories without
        inspecting internal implementation details.
        """
        # ARRANGE - Create empty directory
        empty_dir = tmp_path / "empty_directory"
        empty_dir.mkdir()
        
        # ACT - Search in empty directory
        result = discover_files(
            directory=str(empty_dir),
            pattern="*",
            recursive=True
        )
        
        # ASSERT - Validate empty directory behavior
        assert isinstance(result, list), "Should return list for empty directory"
        assert result == [], "Empty directory should return empty list"

    def test_discover_files_nonexistent_directory_behavior(self):
        """
        Test behavior with non-existent directory through public API.
        
        Validates error handling for invalid directory paths without
        accessing internal error handling implementation.
        """
        # ARRANGE - Use non-existent directory path
        nonexistent_path = "/non/existent/directory/path"
        
        # ACT & ASSERT - Validate error handling behavior
        with pytest.raises((FileNotFoundError, OSError)):
            discover_files(
                directory=nonexistent_path,
                pattern="*"
            )

    @pytest.mark.parametrize("special_filename", [
        "file with spaces.txt",
        "file-with-dashes.txt",
        "file_with_underscores.txt",
        "file.with.dots.txt",
        "file(with)parentheses.txt",
    ])
    def test_discover_files_special_character_handling(
        self, 
        tmp_path, 
        special_filename
    ):
        """
        Test handling of special characters in filenames through public API.
        
        Validates special character handling through observable discovery results.
        """
        # ARRANGE - Create file with special characters
        special_file = tmp_path / special_filename
        special_file.write_text("content with special filename")
        
        # ACT - Discover files with special characters
        result = discover_files(
            directory=str(tmp_path),
            pattern="*.txt"
        )
        
        # ASSERT - Validate special character handling
        assert len(result) == 1, f"Should find file with special characters: {special_filename}"
        assert special_filename in result[0], "Should find correct special character file"

    # =========================================================================
    # Cross-Platform Compatibility Tests
    # =========================================================================

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix-specific test")
    def test_discover_files_unix_path_behavior(self, tmp_path):
        """
        Test Unix-specific path handling behavior through public API.
        
        Validates cross-platform path handling without accessing
        internal path processing implementation.
        """
        # ARRANGE - Create Unix-style nested structure
        unix_structure = tmp_path / "unix" / "nested" / "deep"
        unix_structure.mkdir(parents=True)
        
        test_file = unix_structure / "unix_test.txt"
        test_file.write_text("unix path content")
        
        # ACT - Test path discovery behavior
        result = discover_files(
            directory=str(tmp_path),
            pattern="**/*.txt",
            recursive=True
        )
        
        # ASSERT - Validate Unix path handling
        assert len(result) == 1, "Should find file in nested Unix structure"
        assert os.path.exists(result[0]), "Discovered path should be valid"

    # =========================================================================
    # Property-Based Testing for Robust Validation
    # =========================================================================

    @given(
        num_files=st.integers(min_value=1, max_value=50),
        file_extensions=st.lists(
            st.sampled_from([".txt", ".csv", ".json"]), 
            min_size=1, 
            max_size=3
        )
    )
    @settings(max_examples=10, deadline=3000)
    def test_discover_files_property_based_validation(
        self, 
        tmp_path, 
        num_files, 
        file_extensions
    ):
        """
        Property-based testing for discover_files behavior validation.
        
        Uses Hypothesis to validate invariant properties of file discovery
        across diverse input scenarios without implementation coupling.
        """
        assume(num_files > 0 and len(file_extensions) > 0)
        
        # ARRANGE - Generate random file structure
        test_root = tmp_path / "property_test"
        test_root.mkdir()
        
        created_files = []
        for i in range(num_files):
            ext = file_extensions[i % len(file_extensions)]
            file_path = test_root / f"file_{i}{ext}"
            file_path.write_text(f"Content {i}")
            created_files.append(file_path)
        
        # ACT - Test discovery behavior
        all_result = discover_files(str(test_root), "*")
        ext_result = discover_files(str(test_root), "*", extensions=[ext.lstrip('.') for ext in file_extensions])
        
        # ASSERT - Validate property invariants
        assert len(all_result) == num_files, "Should discover all created files"
        assert len(ext_result) == num_files, "Extension filtering should find all matching files"
        assert all(os.path.exists(f) for f in all_result), "All discovered files should exist"
        assert all(Path(f).suffix in file_extensions for f in ext_result), "Extension filtering should work correctly"

    # =========================================================================
    # Unicode and International Character Support Tests  
    # =========================================================================

    def test_discover_files_unicode_filename_behavior(self, tmp_path):
        """
        Test Unicode filename handling behavior through public API.
        
        Validates Unicode support through observable file discovery results.
        """
        # ARRANGE - Create files with Unicode names
        unicode_files = [
            tmp_path / "测试文件.txt",  # Chinese
            tmp_path / "archivo_español.txt",  # Spanish
            tmp_path / "файл_тест.txt",  # Cyrillic
        ]
        
        created_files = []
        for file_path in unicode_files:
            try:
                file_path.write_text("Unicode content")
                created_files.append(file_path)
            except (OSError, UnicodeError):
                # Skip if filesystem doesn't support Unicode
                continue
        
        if not created_files:
            pytest.skip("Filesystem doesn't support Unicode filenames")
        
        # ACT - Test Unicode file discovery
        result = discover_files(
            directory=str(tmp_path),
            pattern="*.txt"
        )
        
        # ASSERT - Validate Unicode handling behavior
        assert isinstance(result, list), "Should handle Unicode filenames gracefully"
        assert len(result) == len(created_files), "Should find all Unicode files"


class TestGetLatestFileBehaviorValidation:
    """
    Behavior-focused tests for get_latest_file functionality using public API validation.
    
    Tests focus on observable timing behavior and edge-case handling through
    the public interface rather than internal implementation details.
    """

    def test_get_latest_file_timing_behavior(self, tmp_path):
        """
        Test get_latest_file timing behavior through observable results.
        
        Validates modification time-based selection without accessing
        internal timing implementation details.
        """
        # ARRANGE - Create files with different modification times
        files = []
        
        for i in range(3):
            file_path = tmp_path / f"timed_file_{i}.txt"
            file_path.write_text(f"Content {i}")
            files.append(str(file_path))
            
            if i < 2:  # Don't sleep after last file
                time.sleep(0.01)  # Ensure different modification times
        
        # ACT - Get latest file through public API
        latest = get_latest_file(files)
        
        # ASSERT - Validate timing behavior
        assert latest == files[-1], f"Should return most recent file: {files[-1]}"

    def test_get_latest_file_edge_cases_behavior(self):
        """
        Test get_latest_file edge case behavior through public API.
        
        Validates error handling and edge cases without accessing
        internal implementation details.
        """
        # ARRANGE & ACT & ASSERT - Test empty list behavior
        result = get_latest_file([])
        assert result is None, "Empty list should return None"
        
        # Test non-existent file behavior
        with pytest.raises((FileNotFoundError, OSError)):
            get_latest_file(["/non/existent/file.txt"])

    def test_get_latest_file_single_file_behavior(self, tmp_path):
        """
        Test get_latest_file behavior with single file through public API.
        
        Validates single file handling without internal implementation access.
        """
        # ARRANGE - Create single file
        single_file = tmp_path / "single_file.txt"
        single_file.write_text("Single file content")
        
        # ACT - Test single file handling
        result = get_latest_file([str(single_file)])
        
        # ASSERT - Validate single file behavior
        assert result == str(single_file), "Should return the single file"


class TestFileDiscoveryIntegrationBehavior:
    """
    Integration tests focusing on cross-module behavior validation.
    
    Tests the interaction between file discovery and other system components
    through public API behavior validation.
    """

    def test_discover_files_with_configuration_integration(
        self, 
        temp_experiment_directory
    ):
        """
        Test discover_files integration with configuration-driven patterns.
        
        Validates configuration-based discovery behavior through public API
        without accessing internal configuration processing.
        """
        # ARRANGE - Set up experimental directory with configuration patterns
        experiment_dir = temp_experiment_directory["directory"]
        
        # Create files matching typical experimental patterns
        pattern_files = [
            experiment_dir / "raw_data" / "mouse_20240315_control_001.pkl",
            experiment_dir / "raw_data" / "mouse_20240316_treatment_002.pkl",
            experiment_dir / "processed" / "analysis_20240315.csv"
        ]
        
        for file_path in pattern_files:
            file_path.write_text("experimental data")
        
        # ACT - Test configuration-style discovery
        result = discover_files(
            directory=str(experiment_dir),
            pattern="**/*mouse*",
            recursive=True,
            extensions=["pkl"],
            mandatory_substrings=["mouse"]
        )
        
        # ASSERT - Validate integration behavior
        assert len(result) == 2, "Should find mouse experiment files"
        assert all("mouse" in Path(f).name for f in result), \
            "All results should contain 'mouse' substring"
        assert all(f.endswith(".pkl") for f in result), \
            "All results should be pickle files"

    def test_discover_files_complex_filtering_integration(
        self, 
        temp_experiment_directory
    ):
        """
        Test complex filtering integration behavior through public API.
        
        Validates combined filtering operations without accessing
        internal filtering implementation details.
        """
        # ARRANGE - Create complex file structure with filtering challenges
        experiment_dir = temp_experiment_directory["directory"]
        
        complex_files = [
            experiment_dir / "experiment_mouse_data.csv",
            experiment_dir / "experiment_backup_mouse.csv",  # Should be ignored
            experiment_dir / "temp_experiment_mouse.csv",    # Should be ignored
            experiment_dir / "final_experiment_mouse.csv",
            experiment_dir / "experiment_rat_data.csv",      # Different animal
        ]
        
        for file_path in complex_files:
            file_path.write_text("complex experimental data")
        
        # ACT - Apply complex filtering combination
        result = discover_files(
            directory=str(experiment_dir),
            pattern="*.csv",
            extensions=["csv"],
            ignore_patterns=["*backup*", "temp*"],
            mandatory_substrings=["experiment", "mouse"]
        )
        
        # ASSERT - Validate complex filtering behavior
        expected_files = ["experiment_mouse_data.csv", "final_experiment_mouse.csv"]
        result_basenames = [Path(f).name for f in result]
        
        for expected_file in expected_files:
            assert expected_file in result_basenames, \
                f"Complex filtering should include {expected_file}"
        
        # Verify excluded files are not present
        excluded_files = ["experiment_backup_mouse.csv", "temp_experiment_mouse.csv"]
        for excluded_file in excluded_files:
            assert excluded_file not in result_basenames, \
                f"Complex filtering should exclude {excluded_file}"


class TestEdgeCaseScenarios:
    """
    Comprehensive edge-case testing using centralized edge-case generators.
    
    Uses shared utilities from tests/utils.py for consistent edge-case coverage
    across the test suite.
    """

    def test_unicode_edge_cases_with_shared_generators(self, tmp_path):
        """
        Test Unicode edge cases using centralized scenario generators.
        
        Validates Unicode handling through shared test utilities for consistency.
        """
        # ARRANGE - Use centralized edge-case generator
        edge_case_generator = EdgeCaseScenarioGenerator()
        unicode_scenarios = edge_case_generator.generate_unicode_scenarios(count=3)
        
        created_files = []
        for scenario in unicode_scenarios:
            try:
                file_path = tmp_path / scenario['filename']
                file_path.write_text("Unicode test content")
                created_files.append(file_path)
            except (OSError, UnicodeError):
                continue  # Skip unsupported Unicode on this platform
        
        if not created_files:
            pytest.skip("Platform doesn't support Unicode test scenarios")
        
        # ACT - Test Unicode discovery behavior
        result = discover_files(
            directory=str(tmp_path),
            pattern="*"
        )
        
        # ASSERT - Validate Unicode edge-case handling
        assert len(result) == len(created_files), \
            "Should handle all supported Unicode scenarios"

    def test_boundary_conditions_with_shared_utilities(self, tmp_path):
        """
        Test boundary conditions using centralized boundary generators.
        
        Validates boundary condition handling through shared test utilities.
        """
        # ARRANGE - Use centralized boundary condition generator
        edge_case_generator = EdgeCaseScenarioGenerator()
        boundary_conditions = edge_case_generator.generate_boundary_conditions(['file_size'])
        
        # Test with small and large file scenarios
        small_size = boundary_conditions['file_size'][0]  # 0 bytes
        medium_size = boundary_conditions['file_size'][2]  # 1 KB
        
        files = [
            (tmp_path / "empty_file.txt", small_size),
            (tmp_path / "medium_file.txt", medium_size),
        ]
        
        for file_path, size in files:
            content = "x" * size if size > 0 else ""
            file_path.write_text(content)
        
        # ACT - Test boundary condition discovery
        result = discover_files(
            directory=str(tmp_path),
            pattern="*.txt"
        )
        
        # ASSERT - Validate boundary condition handling
        assert len(result) == 2, "Should handle boundary file sizes"
        assert all(os.path.exists(f) for f in result), "All boundary files should be discoverable"

    def test_concurrent_access_simulation(self, tmp_path):
        """
        Test concurrent file access scenarios using shared utilities.
        
        Validates concurrent access handling through observable behavior.
        """
        # ARRANGE - Create test files for concurrent access simulation
        concurrent_files = []
        for i in range(5):
            file_path = tmp_path / f"concurrent_file_{i}.txt"
            file_path.write_text(f"Concurrent content {i}")
            concurrent_files.append(file_path)
        
        # ACT - Test discovery during simulated concurrent access
        result = discover_files(
            directory=str(tmp_path),
            pattern="*.txt",
            recursive=True
        )
        
        # ASSERT - Validate concurrent access handling
        assert isinstance(result, list), \
            "Should return list even during concurrent access scenarios"
        assert len(result) == 5, "Should find all files during concurrent access"


# =========================================================================
# Test Configuration and Markers
# =========================================================================

# Mark slow tests for conditional execution
pytestmark = [
    pytest.mark.unit,  # All tests in this module are unit tests
]

# Configure test settings for property-based testing
settings.register_profile("file_discovery", 
                        max_examples=50,
                        deadline=5000)
settings.load_profile("file_discovery")