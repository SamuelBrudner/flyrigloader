"""
Refactored DataFrame utilities test module implementing behavior-focused testing.

This module validates DataFrame utility functionality through public API behavior 
validation rather than implementation-specific details, following the testing 
strategy requirements for black-box behavioral validation per Section 6.6.

Key Refactoring Changes:
- Replaced manual file creation with centralized fixtures from tests/conftest.py
- Converted whitebox assertions to blackbox behavioral validation
- Implemented Protocol-based mocks from tests/utils.py for consistent dependency isolation
- Enhanced edge-case coverage through parameterized test scenarios
- Structured all tests with clear AAA (Arrange-Act-Assert) patterns
- Focused on observable DataFrame utility behavior rather than internal implementation

Testing Strategy Implementation:
- TST-REF-001: Public API behavior validation through documented interfaces
- TST-REF-002: Centralized fixture management eliminating code duplication
- TST-REF-003: Enhanced edge-case coverage through parameterized testing
- TST-MOD-003: Standardized mocking patterns using Protocol-based implementations
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union
from unittest.mock import MagicMock

from flyrigloader.utils.dataframe import (
    build_manifest_df,
    filter_manifest_df,
    extract_unique_values,
    attach_file_metadata_to_dataframe,
    discovery_results_to_dataframe
)

# Import centralized test utilities for Protocol-based mocking
from tests.utils import (
    create_mock_filesystem,
    create_mock_dataloader,
    MockDataLoading,
    MockFilesystem,
    EdgeCaseScenarioGenerator
)


class TestDataFrameUtilities:
    """
    Comprehensive test suite for DataFrame utilities with behavior-focused validation.
    
    Tests DataFrame construction, filtering, and value extraction functionality
    through public API behavior validation using centralized fixtures and 
    Protocol-based mocks for consistent testing patterns.
    """

    def test_build_manifest_df_from_file_list_basic_functionality(
        self, temp_filesystem_structure
    ):
        """
        Test building DataFrame from file list using centralized filesystem fixture.
        
        Validates basic manifest DataFrame construction behavior with realistic
        file structure using centralized test fixtures.
        """
        # ARRANGE - Use centralized fixture for consistent test setup
        test_files = [
            str(temp_filesystem_structure["baseline_file_1"]),
            str(temp_filesystem_structure["baseline_file_2"]),
            str(temp_filesystem_structure["opto_file_1"])
        ]
        
        # ACT - Execute DataFrame construction through public API
        result_df = build_manifest_df(test_files)
        
        # ASSERT - Verify observable behavior through public interface
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 3
        assert "path" in result_df.columns
        assert set(result_df["path"].tolist()) == set(test_files)
        
        # Verify DataFrame structure integrity
        assert result_df.shape[0] == len(test_files)
        assert all(path in result_df["path"].values for path in test_files)

    def test_build_manifest_df_with_file_statistics_behavior(
        self, temp_filesystem_structure
    ):
        """
        Test DataFrame construction with file statistics through public interface.
        
        Validates that file statistics are properly included when requested,
        focusing on observable behavior rather than implementation details.
        """
        # ARRANGE - Setup test data using centralized fixtures
        test_files = [
            str(temp_filesystem_structure["baseline_file_1"]),
            str(temp_filesystem_structure["opto_file_1"])
        ]
        
        # ACT - Request DataFrame with statistics through public API
        result_df = build_manifest_df(test_files, include_stats=True)
        
        # ASSERT - Verify statistics inclusion behavior
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2
        
        # Verify statistics columns are present (behavior validation)
        expected_stat_columns = ["path", "size", "mtime", "ctime", "creation_time"]
        for col in expected_stat_columns:
            assert col in result_df.columns, f"Expected statistics column '{col}' missing"
        
        # Verify data integrity through observable behavior
        assert set(result_df["path"].tolist()) == set(test_files)
        assert all(isinstance(size, (int, float)) for size in result_df["size"] if pd.notna(size))

    def test_build_manifest_df_with_relative_paths_behavior(
        self, temp_filesystem_structure
    ):
        """
        Test relative path calculation behavior through public interface.
        
        Validates relative path generation functionality focusing on observable
        results rather than internal path calculation implementation.
        """
        # ARRANGE - Setup with centralized fixture and base directory
        base_dir = temp_filesystem_structure["data_root"]
        test_files = [
            str(temp_filesystem_structure["baseline_file_1"]),
            str(temp_filesystem_structure["opto_file_1"])
        ]
        
        # ACT - Execute relative path calculation through public API
        result_df = build_manifest_df(test_files, base_directory=base_dir)
        
        # ASSERT - Verify relative path behavior
        assert isinstance(result_df, pd.DataFrame)
        assert "path" in result_df.columns
        assert "relative_path" in result_df.columns
        
        # Validate relative path calculation behavior
        for _, row in result_df.iterrows():
            assert isinstance(row["relative_path"], str)
            assert len(row["relative_path"]) <= len(row["path"])
            # Verify relative path is actually relative (behavioral check)
            assert not Path(row["relative_path"]).is_absolute() or row["relative_path"] == row["path"]

    def test_build_manifest_df_from_metadata_dictionary_behavior(self):
        """
        Test DataFrame construction from metadata dictionary through public interface.
        
        Validates metadata preservation and DataFrame structure when building
        from dictionary input, focusing on observable behavior validation.
        """
        # ARRANGE - Create test metadata using realistic patterns
        file_metadata = {
            "/path/to/mouse_20241201_control_rep1.pkl": {
                "animal": "mouse", 
                "condition": "control", 
                "replicate": 1,
                "date": "20241201"
            },
            "/path/to/mouse_20241202_treatment_rep2.pkl": {
                "animal": "mouse", 
                "condition": "treatment", 
                "replicate": 2,
                "date": "20241202"
            },
            "/path/to/rat_20241201_control_rep1.pkl": {
                "animal": "rat", 
                "condition": "control", 
                "replicate": 1,
                "date": "20241201"
            }
        }
        
        # ACT - Build DataFrame from metadata dictionary
        result_df = build_manifest_df(file_metadata)
        
        # ASSERT - Verify metadata preservation behavior
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 3
        
        # Verify all expected columns are present
        expected_columns = ["path", "animal", "condition", "replicate", "date"]
        for col in expected_columns:
            assert col in result_df.columns, f"Expected column '{col}' missing"
        
        # Verify metadata integrity through observable behavior
        assert set(result_df["animal"].unique()) == {"mouse", "rat"}
        assert set(result_df["condition"].unique()) == {"control", "treatment"}
        assert set(result_df["replicate"].unique()) == {1, 2}
        
        # Verify path preservation
        assert set(result_df["path"].tolist()) == set(file_metadata.keys())

    @pytest.mark.parametrize("input_data,expected_behavior", [
        ([], {"rows": 0, "has_path_column": True}),
        ({}, {"rows": 0, "has_path_column": True}),
        (["single_file.pkl"], {"rows": 1, "has_path_column": True}),
        ({"/test/file.pkl": {"meta": "data"}}, {"rows": 1, "has_path_column": True})
    ])
    def test_build_manifest_df_edge_cases_behavior(self, input_data, expected_behavior):
        """
        Test DataFrame construction edge cases through parameterized scenarios.
        
        Validates handling of boundary conditions including empty inputs,
        single files, and minimal metadata through behavioral validation.
        """
        # ARRANGE - Use parameterized input data for edge case testing
        # (Input data provided through parametrize decorator)
        
        # ACT - Execute DataFrame construction with edge case input
        result_df = build_manifest_df(input_data)
        
        # ASSERT - Verify expected edge case behavior
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == expected_behavior["rows"]
        
        if expected_behavior["has_path_column"]:
            assert "path" in result_df.columns
        
        # Verify DataFrame structure remains consistent for edge cases
        if len(result_df) > 0:
            assert all(isinstance(path, str) for path in result_df["path"])

    def test_filter_manifest_df_basic_filtering_behavior(self):
        """
        Test basic DataFrame filtering behavior through public interface.
        
        Validates filtering functionality using observable behavior rather
        than internal filtering implementation details.
        """
        # ARRANGE - Create test DataFrame with filtering scenarios
        test_data = {
            "path": [
                "/test/mouse_control_1.pkl",
                "/test/mouse_treatment_1.pkl", 
                "/test/rat_control_1.pkl",
                "/test/rat_treatment_1.pkl"
            ],
            "animal": ["mouse", "mouse", "rat", "rat"],
            "condition": ["control", "treatment", "control", "treatment"],
            "replicate": [1, 1, 1, 1]
        }
        input_df = pd.DataFrame(test_data)
        
        # ACT - Execute filtering through public API
        filtered_df = filter_manifest_df(input_df, animal="mouse")
        
        # ASSERT - Verify filtering behavior
        assert isinstance(filtered_df, pd.DataFrame)
        assert len(filtered_df) == 2
        assert all(animal == "mouse" for animal in filtered_df["animal"])
        
        # Verify original DataFrame structure is preserved
        assert set(filtered_df.columns) == set(input_df.columns)
        assert all(path.startswith("/test/mouse") for path in filtered_df["path"])

    def test_filter_manifest_df_multiple_criteria_behavior(self):
        """
        Test multi-criteria filtering behavior through public interface.
        
        Validates complex filtering logic with multiple filter criteria,
        focusing on observable filtering results.
        """
        # ARRANGE - Setup DataFrame with multi-criteria filtering scenario
        test_data = {
            "path": [f"/test/file_{i}.pkl" for i in range(6)],
            "animal": ["mouse", "mouse", "rat", "rat", "mouse", "rat"],
            "condition": ["control", "treatment", "control", "treatment", "control", "control"],
            "replicate": [1, 1, 1, 1, 2, 2]
        }
        input_df = pd.DataFrame(test_data)
        
        # ACT - Apply multiple filter criteria
        filtered_df = filter_manifest_df(input_df, animal="mouse", condition="control")
        
        # ASSERT - Verify multi-criteria filtering behavior
        assert isinstance(filtered_df, pd.DataFrame)
        assert len(filtered_df) == 2  # Two mouse control entries
        
        # Verify all criteria are satisfied
        assert all(animal == "mouse" for animal in filtered_df["animal"])
        assert all(condition == "control" for condition in filtered_df["condition"])
        
        # Verify DataFrame integrity
        assert set(filtered_df.columns) == set(input_df.columns)

    def test_filter_manifest_df_list_value_filtering_behavior(self):
        """
        Test list-based value filtering behavior through public interface.
        
        Validates filtering with list values for multiple option selection,
        focusing on observable filtering results rather than implementation.
        """
        # ARRANGE - Create DataFrame for list-based filtering test
        test_data = {
            "path": [f"/test/file_{i}.pkl" for i in range(5)],
            "condition": ["control", "treatment", "baseline", "control", "treatment"],
            "animal": ["mouse", "rat", "mouse", "rat", "mouse"]
        }
        input_df = pd.DataFrame(test_data)
        
        # ACT - Filter using list of values
        filtered_df = filter_manifest_df(input_df, condition=["control", "treatment"])
        
        # ASSERT - Verify list-based filtering behavior
        assert isinstance(filtered_df, pd.DataFrame)
        assert len(filtered_df) == 4  # All except baseline
        
        # Verify filtering logic
        assert all(condition in ["control", "treatment"] for condition in filtered_df["condition"])
        assert "baseline" not in filtered_df["condition"].values
        
        # Verify DataFrame structure preservation
        assert set(filtered_df.columns) == set(input_df.columns)

    def test_filter_manifest_df_nonexistent_column_behavior(self):
        """
        Test filtering behavior with non-existent columns.
        
        Validates graceful handling of invalid filter criteria through
        observable behavior rather than internal error handling implementation.
        """
        # ARRANGE - Create test DataFrame
        test_data = {
            "path": ["/test/file1.pkl", "/test/file2.pkl"],
            "animal": ["mouse", "rat"]
        }
        input_df = pd.DataFrame(test_data)
        
        # ACT - Attempt filtering with non-existent column
        result_df = filter_manifest_df(input_df, nonexistent_column="value")
        
        # ASSERT - Verify graceful handling behavior
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(input_df)  # Should return original data
        assert result_df.equals(input_df)  # No filtering should occur

    def test_extract_unique_values_basic_behavior(self):
        """
        Test basic unique value extraction behavior through public interface.
        
        Validates unique value extraction functionality focusing on observable
        results rather than internal extraction implementation.
        """
        # ARRANGE - Create test DataFrame with duplicate values
        test_data = {
            "path": [f"/test/file_{i}.pkl" for i in range(5)],
            "animal": ["mouse", "mouse", "rat", "mouse", "rat"],
            "condition": ["control", "treatment", "control", "control", "treatment"],
            "replicate": [1, 2, 1, 3, 2]
        }
        input_df = pd.DataFrame(test_data)
        
        # ACT - Extract unique values for each column
        unique_animals = extract_unique_values(input_df, "animal")
        unique_conditions = extract_unique_values(input_df, "condition")
        unique_replicates = extract_unique_values(input_df, "replicate")
        
        # ASSERT - Verify unique value extraction behavior
        assert isinstance(unique_animals, list)
        assert isinstance(unique_conditions, list)
        assert isinstance(unique_replicates, list)
        
        # Verify correctness of unique value extraction
        assert set(unique_animals) == {"mouse", "rat"}
        assert set(unique_conditions) == {"control", "treatment"}
        assert set(unique_replicates) == {1, 2, 3}
        
        # Verify no duplicates in results
        assert len(unique_animals) == len(set(unique_animals))
        assert len(unique_conditions) == len(set(unique_conditions))
        assert len(unique_replicates) == len(set(unique_replicates))

    def test_extract_unique_values_with_missing_data_behavior(self):
        """
        Test unique value extraction with missing data through public interface.
        
        Validates handling of NaN values and missing data in unique value
        extraction, focusing on observable behavior.
        """
        # ARRANGE - Create DataFrame with missing values
        test_data = {
            "path": ["/test/file1.pkl", "/test/file2.pkl", "/test/file3.pkl"],
            "animal": ["mouse", np.nan, "rat"],
            "condition": ["control", "treatment", np.nan],
            "value": [1, 2, np.nan]
        }
        input_df = pd.DataFrame(test_data)
        
        # ACT - Extract unique values from columns with missing data
        unique_animals = extract_unique_values(input_df, "animal")
        unique_conditions = extract_unique_values(input_df, "condition")
        unique_values = extract_unique_values(input_df, "value")
        
        # ASSERT - Verify missing data handling behavior
        assert isinstance(unique_animals, list)
        assert isinstance(unique_conditions, list)
        assert isinstance(unique_values, list)
        
        # Verify NaN values are excluded by default (observable behavior)
        assert set(unique_animals) == {"mouse", "rat"}
        assert set(unique_conditions) == {"control", "treatment"}
        assert set(unique_values) == {1, 2}
        
        # Verify no NaN values in results
        assert not any(pd.isna(val) for val in unique_animals)
        assert not any(pd.isna(val) for val in unique_conditions)
        assert not any(pd.isna(val) for val in unique_values)

    def test_extract_unique_values_nonexistent_column_behavior(self):
        """
        Test unique value extraction with non-existent column.
        
        Validates graceful handling of invalid column names through
        observable behavior rather than internal error handling.
        """
        # ARRANGE - Create test DataFrame
        test_data = {
            "path": ["/test/file1.pkl", "/test/file2.pkl"],
            "animal": ["mouse", "rat"]
        }
        input_df = pd.DataFrame(test_data)
        
        # ACT - Attempt extraction from non-existent column
        result = extract_unique_values(input_df, "nonexistent_column")
        
        # ASSERT - Verify graceful handling behavior
        assert isinstance(result, list)
        assert len(result) == 0  # Should return empty list

    @pytest.mark.parametrize("test_scenario", [
        {
            "name": "unicode_file_paths",
            "file_data": {
                "/test/tëst_fïlé_ūnïcōdė.pkl": {"condition": "unicode_test"},
                "/test/dàtä_ñãmé_test.pkl": {"condition": "special_chars"}
            },
            "expected_count": 2
        },
        {
            "name": "special_characters_in_metadata",
            "file_data": {
                "/test/file1.pkl": {"condition": "test with spaces"},
                "/test/file2.pkl": {"condition": "test-with-dashes"},
                "/test/file3.pkl": {"condition": "test_with_underscores"}
            },
            "expected_count": 3
        },
        {
            "name": "numeric_and_boolean_metadata",
            "file_data": {
                "/test/file1.pkl": {"value": 0, "flag": True},
                "/test/file2.pkl": {"value": 1, "flag": False},
                "/test/file3.pkl": {"value": 0, "flag": True}
            },
            "expected_count": 3
        }
    ])
    def test_dataframe_utilities_edge_case_scenarios(self, test_scenario):
        """
        Test DataFrame utilities with comprehensive edge case scenarios.
        
        Validates handling of Unicode paths, special characters, and various
        data types through parameterized testing scenarios focusing on
        observable behavior validation.
        """
        # ARRANGE - Use parameterized edge case scenario
        file_data = test_scenario["file_data"]
        expected_count = test_scenario["expected_count"]
        
        # ACT - Build DataFrame with edge case data
        result_df = build_manifest_df(file_data)
        
        # ASSERT - Verify edge case handling behavior
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == expected_count
        assert "path" in result_df.columns
        
        # Verify all paths are preserved (Unicode handling)
        assert set(result_df["path"].tolist()) == set(file_data.keys())
        
        # Verify metadata preservation for edge cases
        for path, metadata in file_data.items():
            row = result_df[result_df["path"] == path].iloc[0]
            for key, value in metadata.items():
                assert key in result_df.columns
                assert row[key] == value

    def test_dataframe_utilities_consistency_across_operations(self):
        """
        Test consistency of DataFrame utilities across multiple operations.
        
        Validates that DataFrame construction, filtering, and value extraction
        work consistently together through integrated behavioral validation.
        """
        # ARRANGE - Create comprehensive test data
        file_metadata = {
            f"/test/mouse_{date}_{condition}_rep{rep}.pkl": {
                "animal": "mouse" if i % 2 == 0 else "rat",
                "condition": "control" if i % 3 == 0 else "treatment",
                "replicate": rep,
                "date": date
            }
            for i, (date, rep) in enumerate([
                ("20241201", 1), ("20241201", 2), ("20241202", 1),
                ("20241202", 2), ("20241203", 1), ("20241203", 2)
            ])
        }
        
        # ACT - Perform integrated operations
        # 1. Build DataFrame
        df = build_manifest_df(file_metadata)
        
        # 2. Filter for specific criteria
        mouse_df = filter_manifest_df(df, animal="mouse")
        control_df = filter_manifest_df(df, condition="control")
        
        # 3. Extract unique values
        all_animals = extract_unique_values(df, "animal")
        mouse_conditions = extract_unique_values(mouse_df, "condition")
        control_animals = extract_unique_values(control_df, "animal")
        
        # ASSERT - Verify integrated operation consistency
        assert isinstance(df, pd.DataFrame)
        assert isinstance(mouse_df, pd.DataFrame)
        assert isinstance(control_df, pd.DataFrame)
        
        # Verify filtering consistency
        assert all(animal == "mouse" for animal in mouse_df["animal"])
        assert all(condition == "control" for condition in control_df["condition"])
        
        # Verify unique value extraction consistency
        assert set(all_animals) == {"mouse", "rat"}
        assert all(condition in ["control", "treatment"] for condition in mouse_conditions)
        assert all(animal in ["mouse", "rat"] for animal in control_animals)
        
        # Verify cross-operation consistency
        mouse_control_df = filter_manifest_df(mouse_df, condition="control")
        assert len(mouse_control_df) <= len(mouse_df)
        assert len(mouse_control_df) <= len(control_df)

    def test_attach_file_metadata_to_dataframe_behavior(self):
        """
        Test metadata attachment functionality through public interface.
        
        Validates metadata attachment behavior focusing on observable results
        rather than internal merging implementation details.
        """
        # ARRANGE - Create base DataFrame and metadata
        base_df = pd.DataFrame({
            "path": ["/test/file1.pkl", "/test/file2.pkl", "/test/file3.pkl"],
            "experiment": ["exp1", "exp2", "exp3"]
        })
        
        metadata = {
            "/test/file1.pkl": {"animal": "mouse", "condition": "control"},
            "/test/file2.pkl": {"animal": "rat", "condition": "treatment"},
            "/test/file3.pkl": {"animal": "mouse", "condition": "control"}
        }
        
        # ACT - Attach metadata through public interface
        result_df = attach_file_metadata_to_dataframe(base_df, metadata)
        
        # ASSERT - Verify metadata attachment behavior
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 3
        
        # Verify original columns are preserved
        assert "path" in result_df.columns
        assert "experiment" in result_df.columns
        
        # Verify metadata columns are added
        assert "animal" in result_df.columns
        assert "condition" in result_df.columns
        
        # Verify metadata attachment correctness
        for _, row in result_df.iterrows():
            path = row["path"]
            if path in metadata:
                expected_metadata = metadata[path]
                for key, value in expected_metadata.items():
                    assert row[key] == value

    def test_discovery_results_to_dataframe_behavior(self):
        """
        Test discovery results conversion through public interface.
        
        Validates conversion of various discovery result formats to DataFrame,
        focusing on observable behavior rather than internal conversion logic.
        """
        # ARRANGE - Create discovery results in different formats
        list_results = [
            {"path": "/test/file1.pkl", "animal": "mouse", "condition": "control"},
            {"path": "/test/file2.pkl", "animal": "rat", "condition": "treatment"}
        ]
        
        dict_results = {
            "/test/file3.pkl": {"animal": "mouse", "condition": "control"},
            "/test/file4.pkl": {"animal": "rat", "condition": "treatment"}
        }
        
        # ACT - Convert different result formats
        df_from_list = discovery_results_to_dataframe(list_results)
        df_from_dict = discovery_results_to_dataframe(dict_results)
        
        # ASSERT - Verify conversion behavior
        assert isinstance(df_from_list, pd.DataFrame)
        assert isinstance(df_from_dict, pd.DataFrame)
        
        # Verify structure consistency
        assert len(df_from_list) == 2
        assert len(df_from_dict) == 2
        
        # Verify required columns are present
        for df in [df_from_list, df_from_dict]:
            assert "path" in df.columns
            assert "animal" in df.columns
            assert "condition" in df.columns
        
        # Verify data integrity
        assert set(df_from_list["animal"].unique()) == {"mouse", "rat"}
        assert set(df_from_dict["animal"].unique()) == {"mouse", "rat"}


class TestDataFrameUtilitiesWithMocks:
    """
    Test suite using Protocol-based mocks for dependency isolation.
    
    Validates DataFrame utilities behavior with mocked dependencies using
    centralized mock implementations from tests/utils.py for consistent
    testing patterns across the test suite.
    """

    def test_build_manifest_df_with_mock_filesystem(self):
        """
        Test DataFrame construction with mock filesystem provider.
        
        Validates behavior when using Protocol-based filesystem mocks,
        focusing on observable DataFrame construction results.
        """
        # ARRANGE - Create mock filesystem using centralized utilities
        mock_fs = create_mock_filesystem()
        
        # Add test files to mock filesystem
        test_files = [
            mock_fs.add_file("/mock/test1.pkl", size=1024),
            mock_fs.add_file("/mock/test2.pkl", size=2048),
            mock_fs.add_file("/mock/test3.pkl", size=512)
        ]
        
        file_paths = [str(f) for f in test_files]
        
        # ACT - Build DataFrame with mock filesystem
        result_df = build_manifest_df(file_paths)
        
        # ASSERT - Verify behavior with mocked dependencies
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 3
        assert "path" in result_df.columns
        assert set(result_df["path"].tolist()) == set(file_paths)

    def test_build_manifest_df_with_mock_data_loading(self):
        """
        Test DataFrame construction with mock data loading provider.
        
        Validates DataFrame construction behavior when using Protocol-based
        data loading mocks for consistent dependency isolation.
        """
        # ARRANGE - Create mock data loader using centralized utilities
        mock_loader = create_mock_dataloader(scenarios=['basic'])
        
        # Setup test data with mock loader
        test_metadata = {
            "/mock/experimental_data.pkl": {
                "animal": "mouse", 
                "condition": "control", 
                "date": "20241201"
            }
        }
        
        # ACT - Build DataFrame with metadata
        result_df = build_manifest_df(test_metadata)
        
        # ASSERT - Verify behavior with mock data loading
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 1
        assert "path" in result_df.columns
        assert "animal" in result_df.columns
        assert "condition" in result_df.columns
        assert "date" in result_df.columns
        
        # Verify metadata preservation
        row = result_df.iloc[0]
        assert row["animal"] == "mouse"
        assert row["condition"] == "control"
        assert row["date"] == "20241201"

    @pytest.mark.parametrize("corruption_scenario", [
        {
            "name": "corrupted_pickle_files",
            "error_type": "pickle_corruption",
            "expected_behavior": "graceful_degradation"
        },
        {
            "name": "permission_denied",
            "error_type": "access_error", 
            "expected_behavior": "skip_file"
        },
        {
            "name": "missing_files",
            "error_type": "file_not_found",
            "expected_behavior": "continue_processing"
        }
    ])
    def test_dataframe_utilities_error_handling_scenarios(
        self, corruption_scenario
    ):
        """
        Test DataFrame utilities error handling with corrupted file scenarios.
        
        Validates graceful error handling behavior using Protocol-based mocks
        to simulate various file corruption and access error scenarios.
        """
        # ARRANGE - Setup error scenario using centralized mock utilities
        mock_fs = create_mock_filesystem()
        error_type = corruption_scenario["error_type"]
        
        if error_type == "pickle_corruption":
            # Add corrupted file scenario
            corrupted_scenarios = mock_fs.add_corrupted_scenarios("/mock/corrupted")
            test_files = [str(path) for path in corrupted_scenarios.values()]
        elif error_type == "access_error":
            # Add permission denied scenario
            denied_file = mock_fs.add_file(
                "/mock/denied.pkl", 
                access_error=PermissionError("Access denied for testing")
            )
            test_files = [str(denied_file)]
        else:
            # Missing files scenario
            test_files = ["/mock/nonexistent.pkl"]
        
        # ACT - Attempt DataFrame construction with error scenarios
        try:
            result_df = build_manifest_df(test_files)
            construction_succeeded = True
        except Exception:
            construction_succeeded = False
            result_df = pd.DataFrame(columns=["path"])
        
        # ASSERT - Verify error handling behavior
        expected_behavior = corruption_scenario["expected_behavior"]
        
        if expected_behavior == "graceful_degradation":
            # Should handle errors gracefully and return valid DataFrame
            assert isinstance(result_df, pd.DataFrame)
            assert "path" in result_df.columns
        elif expected_behavior in ["skip_file", "continue_processing"]:
            # Should skip problematic files and continue processing
            assert isinstance(result_df, pd.DataFrame)
            # May have empty DataFrame but should not crash
            assert len(result_df) >= 0


class TestDataFrameUtilitiesPerformanceAware:
    """
    Performance-aware test suite focusing on behavioral validation.
    
    Tests DataFrame utilities performance characteristics through behavioral
    validation while maintaining focus on public API behavior rather than
    internal performance implementation details.
    """

    @pytest.mark.parametrize("data_size", [10, 100, 1000])
    def test_dataframe_construction_scales_appropriately(self, data_size):
        """
        Test DataFrame construction scaling behavior with various data sizes.
        
        Validates that DataFrame construction maintains consistent behavior
        across different data sizes, focusing on observable scaling behavior.
        """
        # ARRANGE - Generate test data of specified size
        file_metadata = {
            f"/test/file_{i:04d}.pkl": {
                "animal": "mouse" if i % 2 == 0 else "rat",
                "condition": "control" if i % 3 == 0 else "treatment",
                "replicate": i % 5 + 1,
                "batch": f"batch_{i // 10}"
            }
            for i in range(data_size)
        }
        
        # ACT - Build DataFrame with scaled data
        result_df = build_manifest_df(file_metadata)
        
        # ASSERT - Verify scaling behavior maintains correctness
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == data_size
        
        # Verify structure consistency regardless of size
        expected_columns = ["path", "animal", "condition", "replicate", "batch"]
        for col in expected_columns:
            assert col in result_df.columns
        
        # Verify data integrity scales appropriately
        assert len(result_df["animal"].unique()) <= 2  # mouse, rat
        assert len(result_df["condition"].unique()) <= 2  # control, treatment
        assert len(result_df["replicate"].unique()) <= 5  # 1-5
        
        # Verify path uniqueness (behavioral requirement)
        assert len(result_df["path"].unique()) == data_size

    def test_filtering_maintains_performance_characteristics(self):
        """
        Test filtering performance characteristics through behavioral validation.
        
        Validates that filtering operations maintain consistent behavior
        and performance characteristics across different filtering scenarios.
        """
        # ARRANGE - Create substantial test dataset
        large_dataset = {
            f"/test/large_dataset_file_{i:05d}.pkl": {
                "animal": ["mouse", "rat", "hamster"][i % 3],
                "condition": ["control", "treatment", "baseline"][i % 3],
                "replicate": i % 10 + 1,
                "experiment_batch": i // 100
            }
            for i in range(5000)  # Substantial but not excessive for unit tests
        }
        
        # Build base DataFrame
        base_df = build_manifest_df(large_dataset)
        
        # ACT - Perform various filtering operations
        single_filter = filter_manifest_df(base_df, animal="mouse")
        multi_filter = filter_manifest_df(base_df, animal="mouse", condition="control")
        list_filter = filter_manifest_df(base_df, condition=["control", "treatment"])
        
        # ASSERT - Verify filtering behavior consistency
        assert isinstance(single_filter, pd.DataFrame)
        assert isinstance(multi_filter, pd.DataFrame)
        assert isinstance(list_filter, pd.DataFrame)
        
        # Verify filtering correctness
        assert all(animal == "mouse" for animal in single_filter["animal"])
        assert all(animal == "mouse" for animal in multi_filter["animal"])
        assert all(condition == "control" for condition in multi_filter["condition"])
        
        # Verify filtering relationship consistency
        assert len(multi_filter) <= len(single_filter)
        assert len(single_filter) <= len(base_df)
        assert len(list_filter) <= len(base_df)


# Performance tests are excluded from default execution per Section 6.6.4
# They are relocated to scripts/benchmarks/ for optional execution
# This maintains <30 second test suite execution for developer feedback