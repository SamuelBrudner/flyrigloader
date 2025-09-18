"""
Tests for DataFrame utilities.

Tests the functionality for working with discovery results as DataFrames.
"""
import os
import tempfile
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

from flyrigloader.utils.dataframe import (
    build_manifest_df,
    filter_manifest_df,
    extract_unique_values
)
from flyrigloader.discovery.files import discover_files


@pytest.fixture
def temp_test_files(tmp_path):
    files = []
    for index in range(3):
        file_path = tmp_path / f"file_{index}.txt"
        file_path.write_text(f"content {index}", encoding="utf-8")
        files.append(str(file_path))
    return str(tmp_path), files


def test_build_manifest_df_from_list():
    """Test building a DataFrame from a list of files."""
    # Create temporary files
    temp_dir = tempfile.mkdtemp()
    temp_files = []
    
    try:
        # Create some files
        for i in range(3):
            file_path = os.path.join(temp_dir, f"file_{i}.txt")
            with open(file_path, 'w') as f:
                f.write(f"content {i}")
            temp_files.append(file_path)
        
        # Build a DataFrame
        df = build_manifest_df(temp_files)
        
        # Verify the result
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "path" in df.columns
        assert set(df["path"].tolist()) == set(temp_files)
    finally:
        # Clean up
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


def test_build_manifest_df_with_stats():
    """Test building a DataFrame with file statistics."""
    # Create temporary files
    temp_dir = tempfile.mkdtemp()
    temp_files = []
    
    try:
        # Create some files
        for i in range(3):
            file_path = os.path.join(temp_dir, f"file_{i}.txt")
            with open(file_path, 'w') as f:
                f.write(f"content {i}")
            temp_files.append(file_path)
        
        # Build a DataFrame with stats
        df = build_manifest_df(temp_files, include_stats=True)
        
        # Verify the result
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "path" in df.columns
        assert "size" in df.columns
        assert "mtime" in df.columns
        assert "ctime" in df.columns
        assert "creation_time" in df.columns
        assert set(df["path"].tolist()) == set(temp_files)
    finally:
        # Clean up
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


def test_build_manifest_df_with_relative_paths():
    """Test building a DataFrame with relative paths."""
    # Create temporary files
    temp_dir = tempfile.mkdtemp()
    temp_files = []
    
    try:
        # Create some files
        for i in range(3):
            file_path = os.path.join(temp_dir, f"file_{i}.txt")
            with open(file_path, 'w') as f:
                f.write(f"content {i}")
            temp_files.append(file_path)
        
        # Build a DataFrame with relative paths
        df = build_manifest_df(temp_files, base_directory=temp_dir)
        
        # Verify the result
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "path" in df.columns
        assert "relative_path" in df.columns
        
        # Check that relative paths are correct
        for i, row in df.iterrows():
            expected_rel_path = os.path.basename(row["path"])
            assert row["relative_path"] == expected_rel_path
    finally:
        # Clean up
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


def test_build_manifest_df_from_dict():
    """Test building a DataFrame from a dictionary with metadata."""
    # Create a dictionary with file metadata
    file_data = {
        "/path/to/file1.txt": {"animal": "mouse", "condition": "control", "replicate": 1},
        "/path/to/file2.txt": {"animal": "mouse", "condition": "test", "replicate": 2},
        "/path/to/file3.txt": {"animal": "rat", "condition": "control", "replicate": 1}
    }
    
    # Build a DataFrame
    df = build_manifest_df(file_data)
    
    # Verify the result
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "path" in df.columns
    assert "animal" in df.columns
    assert "condition" in df.columns
    assert "replicate" in df.columns
    
    # Check that metadata was preserved
    assert set(df["animal"].unique()) == {"mouse", "rat"}
    assert set(df["condition"].unique()) == {"control", "test"}
    assert set(df["replicate"].unique()) == {1, 2}


def test_build_manifest_df_empty_input():
    """Test building a DataFrame from empty input."""
    # Empty list
    df_list = build_manifest_df([])
    assert isinstance(df_list, pd.DataFrame)
    assert len(df_list) == 0
    assert "path" in df_list.columns
    
    # Empty dict
    df_dict = build_manifest_df({})
    assert isinstance(df_dict, pd.DataFrame)
    assert len(df_dict) == 0
    assert "path" in df_dict.columns


def test_filter_manifest_df():
    """Test filtering a manifest DataFrame."""
    # Create a sample DataFrame
    data = {
        "path": ["/path/to/file1.txt", "/path/to/file2.txt", "/path/to/file3.txt"],
        "animal": ["mouse", "mouse", "rat"],
        "condition": ["control", "test", "control"],
        "replicate": [1, 2, 1]
    }
    df = pd.DataFrame(data)
    
    # Filter by a single value
    filtered_df = filter_manifest_df(df, animal="mouse")
    assert len(filtered_df) == 2
    assert set(filtered_df["animal"].unique()) == {"mouse"}
    
    # Filter by multiple values
    filtered_df = filter_manifest_df(df, animal="mouse", condition="control")
    assert len(filtered_df) == 1
    assert filtered_df.iloc[0]["path"] == "/path/to/file1.txt"
    
    # Filter using a list
    filtered_df = filter_manifest_df(df, condition=["control", "test"])
    assert len(filtered_df) == 3  # All rows match
    
    # Filter with non-existent column
    filtered_df = filter_manifest_df(df, non_existent="value")
    assert len(filtered_df) == 3  # No filtering performed


def test_extract_unique_values():
    """Test extracting unique values from a DataFrame."""
    # Create a sample DataFrame
    data = {
        "path": ["/path/to/file1.txt", "/path/to/file2.txt", "/path/to/file3.txt"],
        "animal": ["mouse", "mouse", "rat"],
        "condition": ["control", "test", "control"],
        "replicate": [1, 2, 1]
    }
    df = pd.DataFrame(data)
    
    # Extract unique values
    animals = extract_unique_values(df, "animal")
    assert set(animals) == {"mouse", "rat"}
    
    conditions = extract_unique_values(df, "condition")
    assert set(conditions) == {"control", "test"}
    
    replicates = extract_unique_values(df, "replicate")
    assert set(replicates) == {1, 2}
    
    # Test with non-existent column
    non_existent = extract_unique_values(df, "non_existent")
    assert non_existent == []
    
    # Test with column containing NaN values
    df["optional"] = [1, np.nan, 3]
    optional = extract_unique_values(df, "optional")
    assert set(optional) == {1, 3}  # NaN values are dropped


# --- Edge Case and Regression Tests ---

class TestDataFrameEdgeCases:
    """Edge case testing for comprehensive coverage."""

    def test_build_manifest_df_special_characters(self):
        """Test handling of special characters in paths and metadata."""
        
        special_char_data = {
            "/path/with spaces/file name.csv": {"condition": "test with spaces"},
            "/path/with-dashes/file-name.csv": {"condition": "test-with-dashes"},
            "/path/with_underscores/file_name.csv": {"condition": "test_with_underscores"},
            "/path/with.dots/file.name.csv": {"condition": "test.with.dots"},
            "/path/with(parens)/file(name).csv": {"condition": "test(with)parens"},
            "/path/with[brackets]/file[name].csv": {"condition": "test[with]brackets"}
        }
        
        df = build_manifest_df(special_char_data)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6
        
        # Verify all special characters are preserved
        for original_path in special_char_data.keys():
            assert original_path in df["path"].values

    def test_filter_manifest_df_edge_cases(self):
        """Test filtering with edge case values."""
        
        # DataFrame with edge case values
        edge_case_df = pd.DataFrame({
            "path": ["/a", "/b", "/c", "/d", "/e"],
            "empty_string": ["", "value", "", "another", ""],
            "zero_value": [0, 1, 0, 2, 0],
            "none_equivalent": [None, "value", None, "another", None],
            "boolean_like": [True, False, True, False, True]
        })
        
        # Test filtering by empty string
        empty_filtered = filter_manifest_df(edge_case_df, empty_string="")
        assert len(empty_filtered) == 3
        
        # Test filtering by zero
        zero_filtered = filter_manifest_df(edge_case_df, zero_value=0)
        assert len(zero_filtered) == 3
        
        # Test filtering by boolean values
        true_filtered = filter_manifest_df(edge_case_df, boolean_like=True)
        assert len(true_filtered) == 3
        
        false_filtered = filter_manifest_df(edge_case_df, boolean_like=False)
        assert len(false_filtered) == 2

    def test_extract_unique_values_edge_cases(self):
        """Test extraction with edge case data types."""
        
        # DataFrame with various edge case data types
        edge_case_df = pd.DataFrame({
            "path": ["/a", "/b", "/c", "/d"],
            "booleans": [True, False, True, False],
            "zeros_and_ones": [0, 1, 0, 1],
            "empty_strings": ["", "value", "", "value2"],
            "mixed_numeric": [1, 1.0, 2, 2.0]  # Mix of int and float
        })
        
        # Test boolean extraction
        bool_values = extract_unique_values(edge_case_df, "booleans")
        assert set(bool_values) == {True, False}
        
        # Test zero/one extraction
        zero_one_values = extract_unique_values(edge_case_df, "zeros_and_ones")
        assert set(zero_one_values) == {0, 1}
        
        # Test empty string extraction
        empty_str_values = extract_unique_values(edge_case_df, "empty_strings")
        assert set(empty_str_values) == {"", "value", "value2"}
        
        # Test mixed numeric types
        mixed_values = extract_unique_values(edge_case_df, "mixed_numeric")
        assert len(set(mixed_values)) <= 2  # Should deduplicate 1/1.0 and 2/2.0

    def test_build_manifest_df_unicode_handling(self):
        """Test handling of Unicode characters in paths and metadata."""
        
        unicode_data = {
            "/path/français/fichier.csv": {"language": "français", "type": "européen"},
            "/path/中文/文件.csv": {"language": "中文", "type": "亚洲"},
            "/path/العربية/ملف.csv": {"language": "العربية", "type": "عربي"},
            "/path/русский/файл.csv": {"language": "русский", "type": "кириллица"},
            "/path/emoji😀/file🎉.csv": {"mood": "happy😀", "celebration": "party🎉"}
        }
        
        df = build_manifest_df(unicode_data)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        
        # Verify Unicode is preserved
        languages = extract_unique_values(df, "language")
        expected_languages = {"français", "中文", "العربية", "русский"}
        assert set(languages) == expected_languages

    def test_dataframe_operations_consistency(self):
        """Test consistency across multiple operations on the same data."""
        
        # Create consistent test data
        test_data = {
            f"/path/file_{i}.csv": {
                "animal": "mouse" if i % 2 == 0 else "rat",
                "condition": "control" if i % 3 == 0 else "test",
                "replicate": i % 5 + 1,
                "batch": f"batch_{i // 10}"
            }
            for i in range(50)
        }
        
        # Build DataFrame
        df = build_manifest_df(test_data)
        
        # Test multiple filtering operations
        mouse_files = filter_manifest_df(df, animal="mouse")
        control_files = filter_manifest_df(df, condition="control")
        mouse_control = filter_manifest_df(df, animal="mouse", condition="control")
        
        # Verify consistency
        mouse_control_alt = filter_manifest_df(mouse_files, condition="control")
        assert len(mouse_control) == len(mouse_control_alt)
        assert set(mouse_control["path"]) == set(mouse_control_alt["path"])
        
        # Test unique value extraction consistency
        all_animals = extract_unique_values(df, "animal")
        mouse_animals = extract_unique_values(mouse_files, "animal")
        
        assert set(all_animals) == {"mouse", "rat"}
        assert set(mouse_animals) == {"mouse"}


# --- Backward Compatibility Tests ---

class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing code."""

    def test_legacy_api_compatibility(self, temp_test_files):
        """Test that legacy API usage patterns still work."""
        temp_dir, file_paths = temp_test_files
        
        # Legacy usage: simple file list
        df1 = build_manifest_df(file_paths)
        assert isinstance(df1, pd.DataFrame)
        assert "path" in df1.columns
        
        # Legacy usage: with stats
        df2 = build_manifest_df(file_paths, include_stats=True)
        assert isinstance(df2, pd.DataFrame)
        assert "size" in df2.columns
        
        # Legacy usage: simple filtering
        filtered = filter_manifest_df(df1, path=file_paths[0])
        assert len(filtered) <= 1
        
        # Legacy usage: simple extraction
        paths = extract_unique_values(df1, "path")
        assert isinstance(paths, list)

    def test_return_type_consistency(self):
        """Test that return types are consistent across different inputs."""
        
        # All these should return DataFrames
        inputs = [
            [],  # Empty list
            {},  # Empty dict
            ["/path/file.csv"],  # Single file list
            {"/path/file.csv": {"meta": "data"}},  # Single file dict
        ]
        
        for input_data in inputs:
            result = build_manifest_df(input_data)
            assert isinstance(result, pd.DataFrame), f"Failed for input: {input_data}"
            assert "path" in result.columns, f"Missing path column for input: {input_data}"
        
        # All filter results should be DataFrames
        test_df = pd.DataFrame({"path": ["/a"], "col": ["val"]})
        filter_result = filter_manifest_df(test_df, col="val")
        assert isinstance(filter_result, pd.DataFrame)
        
        # All extract results should be lists
        extract_result = extract_unique_values(test_df, "col")
        assert isinstance(extract_result, list)
