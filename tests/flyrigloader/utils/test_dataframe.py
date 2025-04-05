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


def test_integration_with_discover_files():
    """Test integration with discover_files."""
    # Create temporary files
    temp_dir = tempfile.mkdtemp()
    try:
        # Create files with metadata in the filenames
        for animal in ["mouse", "rat"]:
            for condition in ["control", "test"]:
                for replicate in [1, 2]:
                    filename = f"{animal}_{condition}_{replicate}.txt"
                    file_path = os.path.join(temp_dir, filename)
                    with open(file_path, 'w') as f:
                        f.write(f"content for {filename}")
        
        # Define extraction pattern
        pattern = r"(?P<animal>\w+)_(?P<condition>\w+)_(?P<replicate>\d+)\.txt"
        
        # Discover files with metadata extraction
        files = discover_files(
            temp_dir,
            "*.txt",
            extract_patterns=[pattern]
        )
        
        # Convert to DataFrame
        df = build_manifest_df(files)
        
        # Verify the result
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 8  # 2 animals × 2 conditions × 2 replicates
        assert "animal" in df.columns
        assert "condition" in df.columns
        assert "replicate" in df.columns
        
        # Check that metadata was correctly extracted
        assert set(df["animal"].unique()) == {"mouse", "rat"}
        assert set(df["condition"].unique()) == {"control", "test"}
        assert set(df["replicate"].unique()) == {"1", "2"}  # Note: replicate is a string
    finally:
        # Clean up
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.unlink(os.path.join(root, name))
        os.rmdir(temp_dir)
