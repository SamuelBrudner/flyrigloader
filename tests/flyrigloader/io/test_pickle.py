"""Tests for the pickle.py module."""

import os
import gzip
import pickle
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from unittest.mock import patch
import logging

# This will be the module we're testing - we'll have to implement it later
from flyrigloader.io.pickle import (
    read_pickle_any_format, 
    extract_columns_from_matrix, 
    handle_signal_disp,
    make_dataframe_from_matrix
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def sample_data():
    """Create sample data dictionary for test pickles."""
    return {
        't': np.arange(0, 10),
        'x': np.arange(10, 20),
        'metadata': {
            'experiment': 'test',
            'date': '2025-04-01'
        }
    }


@pytest.fixture
def exp_matrix_data():
    """Sample experimental matrix data with different array dimensions."""
    return {
        't': np.arange(0, 10),                     # 1D array
        'x': np.arange(10, 20),                    # 1D array
        'multi_dim': np.ones((5, 3)),              # 2D array
        'complex_array': np.ones((3, 4, 2)),       # 3D array
        'signal_disp': np.random.rand(10, 5),      # 2D array with special handling
        'scalar': 42,                               # Scalar value
        'string': np.array(['test_string'] * 10),  # Array of strings with length matching t
        'list_data': list(range(10))               # List value with length matching t
    }


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        'fly_id': 'fly-123',
        'genotype': 'wild-type',
        'date': '2025-04-01',
        'experiment_type': 'behavior'
    }


@pytest.fixture
def signal_disp_T_X():
    """Create sample exp_matrix with signal_disp in time-first orientation (T, X)."""
    T = 10  # Time dimension
    X = 5   # Signal dimension
    return {
        't': np.arange(0, T),
        'signal_disp': np.random.rand(T, X)  # Shape is (time, signals)
    }


@pytest.fixture
def signal_disp_X_T():
    """Create sample exp_matrix with signal_disp in signal-first orientation (X, T)."""
    T = 10  # Time dimension
    X = 5   # Signal dimension
    return {
        't': np.arange(0, T),
        'signal_disp': np.random.rand(X, T)  # Shape is (signals, time)
    }


@pytest.fixture
def regular_pickle_file(temp_dir, sample_data):
    """Create a regular pickle file for testing."""
    filepath = temp_dir / "regular_data.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(sample_data, f)
    yield filepath
    # No need for cleanup as temp_dir fixture handles it


@pytest.fixture
def gzipped_pickle_file(temp_dir, sample_data):
    """Create a gzipped pickle file for testing."""
    filepath = temp_dir / "gzipped_data.pkl.gz"
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(sample_data, f)
    yield filepath
    # No need for cleanup as temp_dir fixture handles it


@pytest.fixture
def pandas_pickle_file(temp_dir):
    """Create a pandas-specific pickle file for testing."""
    filepath = temp_dir / "pandas_data.pkl"
    df = pd.DataFrame({
        't': np.arange(0, 10),
        'x': np.arange(10, 20)
    })
    df.to_pickle(filepath)
    yield filepath
    # No need for cleanup as temp_dir fixture handles it


def test_read_pickle_any_format_regular(regular_pickle_file, sample_data):
    """Test reading a regular pickle file."""
    result = read_pickle_any_format(regular_pickle_file)
    
    # Verify the result is a dictionary matching sample_data
    assert isinstance(result, dict)
    assert 't' in result
    assert 'x' in result
    assert 'metadata' in result
    
    # Verify array content equality (using numpy's assertions for arrays)
    np.testing.assert_array_equal(result['t'], sample_data['t'])
    np.testing.assert_array_equal(result['x'], sample_data['x'])
    
    # Verify metadata
    assert result['metadata'] == sample_data['metadata']


def test_read_pickle_any_format_gzipped(gzipped_pickle_file, sample_data):
    """Test reading a gzipped pickle file."""
    result = read_pickle_any_format(gzipped_pickle_file)
    
    # Verify the result is a dictionary matching sample_data
    assert isinstance(result, dict)
    assert 't' in result
    assert 'x' in result
    assert 'metadata' in result
    
    # Verify array content equality
    np.testing.assert_array_equal(result['t'], sample_data['t'])
    np.testing.assert_array_equal(result['x'], sample_data['x'])
    
    # Verify metadata
    assert result['metadata'] == sample_data['metadata']


def test_read_pickle_any_format_pandas(pandas_pickle_file):
    """Test reading a pandas-specific pickle file."""
    result = read_pickle_any_format(pandas_pickle_file)
    
    # Verify the result is a DataFrame with expected columns
    assert isinstance(result, pd.DataFrame)
    assert 't' in result.columns
    assert 'x' in result.columns
    
    # Verify content matches what we expect
    assert len(result) == 10
    np.testing.assert_array_equal(result['t'].values, np.arange(0, 10))
    np.testing.assert_array_equal(result['x'].values, np.arange(10, 20))


def test_read_pickle_any_format_file_not_found():
    """Test that an appropriate error is raised for a non-existent file."""
    with pytest.raises(FileNotFoundError):
        read_pickle_any_format("non_existent_file.pkl")


def test_read_pickle_any_format_invalid_path():
    """Test that an appropriate error is raised for an invalid path."""
    with pytest.raises(ValueError):
        read_pickle_any_format(123)  # Not a valid path


def test_extract_columns_from_matrix_all_columns(exp_matrix_data):
    """Test extracting all columns from an exp_matrix."""
    # Mock the ensure_1d function to return the input unchanged
    with patch('flyrigloader.io.pickle.ensure_1d_array', side_effect=lambda x, *args: x):
        result = extract_columns_from_matrix(exp_matrix_data)
    
    # Check that all columns except signal_disp are extracted
    assert len(result) == len(exp_matrix_data) - 1  # signal_disp should be skipped
    assert 'signal_disp' not in result  # Verify signal_disp is skipped
    
    # Check the content of extracted columns
    for key in exp_matrix_data:
        if key != 'signal_disp':  # signal_disp is handled separately
            assert key in result
            if isinstance(exp_matrix_data[key], np.ndarray):
                np.testing.assert_array_equal(result[key], exp_matrix_data[key])
            else:
                assert result[key] == exp_matrix_data[key]


def test_extract_columns_from_matrix_specific_columns(exp_matrix_data):
    """Test extracting specific columns from an exp_matrix."""
    columns_to_extract = ['t', 'x', 'scalar']
    
    # Mock the ensure_1d function to return the input unchanged
    with patch('flyrigloader.io.pickle.ensure_1d_array', side_effect=lambda x, *args: x):
        result = extract_columns_from_matrix(exp_matrix_data, columns_to_extract)
    
    # Check that only the specified columns are extracted
    assert set(result.keys()) == set(columns_to_extract)
    
    # Check the content of extracted columns
    for key in columns_to_extract:
        if isinstance(exp_matrix_data[key], np.ndarray):
            np.testing.assert_array_equal(result[key], exp_matrix_data[key])
        else:
            assert result[key] == exp_matrix_data[key]


def test_extract_columns_from_matrix_without_1d_conversion(exp_matrix_data):
    """Test extracting columns without converting to 1D."""
    # Extract without ensuring 1D arrays
    result = extract_columns_from_matrix(exp_matrix_data, ensure_1d=False)
    
    # Check that multi-dimensional arrays remain multi-dimensional
    assert result['multi_dim'].ndim == 2
    assert result['complex_array'].ndim == 3
    np.testing.assert_array_equal(result['multi_dim'], exp_matrix_data['multi_dim'])
    np.testing.assert_array_equal(result['complex_array'], exp_matrix_data['complex_array'])


def test_extract_columns_from_matrix_with_1d_conversion(exp_matrix_data):
    """Test extracting columns with conversion to 1D."""
    # Create a mock for ensure_1d_array that flattens arrays
    def mock_ensure_1d(arr, name):
        return arr.flatten() if hasattr(arr, 'ndim') and arr.ndim > 1 else arr

    # Use the mock to test conversion
    with patch('flyrigloader.io.pickle.ensure_1d_array', side_effect=mock_ensure_1d):
        result = extract_columns_from_matrix(exp_matrix_data)

    # Check that all arrays in the result are 1D
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            assert value.ndim == 1, f"Array for key '{key}' should be 1D"

    # Verify the multi-dimensional arrays were flattened
    assert len(result['multi_dim']) == 15  # 5x3 array flattened
    assert len(result['complex_array']) == 24  # 3x4x2 array flattened


def test_extract_columns_from_matrix_invalid_input():
    """Test that appropriate error is raised for invalid input."""
    with pytest.raises(ValueError):
        extract_columns_from_matrix("not_a_dictionary")


def test_extract_columns_from_matrix_nonexistent_column(exp_matrix_data):
    """Test behavior with non-existent column names."""
    columns_to_extract = ['t', 'nonexistent_column']
    
    # Mock ensure_1d to avoid side effects
    with patch('flyrigloader.io.pickle.ensure_1d_array', side_effect=lambda x, *args: x):
        result = extract_columns_from_matrix(exp_matrix_data, columns_to_extract)
    
    # Should only include existing columns
    assert 't' in result
    assert 'nonexistent_column' not in result


def test_handle_signal_disp_T_X(signal_disp_T_X):
    """Test handling signal_disp in (T, X) orientation."""
    result = handle_signal_disp(signal_disp_T_X)
    
    # Verify the result is a pandas Series
    assert isinstance(result, pd.Series)
    
    # Verify the length matches the time dimension
    T = len(signal_disp_T_X['t'])
    assert len(result) == T
    
    # Verify the content type
    for i in range(T):
        assert isinstance(result.iloc[i], np.ndarray)
        assert result.iloc[i].ndim == 1  # Each element should be a 1D array
        assert len(result.iloc[i]) == signal_disp_T_X['signal_disp'].shape[1]


def test_handle_signal_disp_X_T(signal_disp_X_T):
    """Test handling signal_disp in (X, T) orientation."""
    result = handle_signal_disp(signal_disp_X_T)
    
    # Verify the result is a pandas Series
    assert isinstance(result, pd.Series)
    
    # Verify the length matches the time dimension
    T = len(signal_disp_X_T['t'])
    assert len(result) == T
    
    # Verify the content type
    for i in range(T):
        assert isinstance(result.iloc[i], np.ndarray)
        assert result.iloc[i].ndim == 1  # Each element should be a 1D array
        assert len(result.iloc[i]) == signal_disp_X_T['signal_disp'].shape[0]


def test_handle_signal_disp_missing_keys():
    """Test that appropriate errors are raised when required keys are missing."""
    # Missing signal_disp
    with pytest.raises(ValueError, match="missing required 'signal_disp' key"):
        handle_signal_disp({'t': np.arange(10)})
    
    # Missing t
    with pytest.raises(ValueError, match="missing required 't' key"):
        handle_signal_disp({'signal_disp': np.ones((10, 5))})


def test_handle_signal_disp_wrong_dimensions():
    """Test that appropriate errors are raised when dimensions don't match."""
    # 1D signal_disp (invalid)
    with pytest.raises(ValueError, match="signal_disp must be 2D"):
        handle_signal_disp({'t': np.arange(10), 'signal_disp': np.ones(10)})
    
    # 3D signal_disp (invalid)
    with pytest.raises(ValueError, match="signal_disp must be 2D"):
        handle_signal_disp({'t': np.arange(10), 'signal_disp': np.ones((10, 5, 2))})
    
    # Neither dimension matches time
    with pytest.raises(ValueError, match="No dimension of signal_disp .* matches time dimension"):
        handle_signal_disp({'t': np.arange(10), 'signal_disp': np.ones((15, 20))})


def test_make_dataframe_from_matrix_basic(exp_matrix_data):
    """Test basic conversion of an exp_matrix to DataFrame without signal_disp."""
    # Create a copy and remove the columns that fail validation
    valid_matrix_data = exp_matrix_data.copy()
    valid_matrix_data.pop('multi_dim', None)
    valid_matrix_data.pop('mismatched_array', None)
    valid_matrix_data.pop('complex_array', None)
    
    df = make_dataframe_from_matrix(valid_matrix_data, include_signal_disp=False, metadata=None)
    
    # Expect 't', 'x', 'scalar', 'list_data', 'string'
    expected_cols = {'t', 'x', 'scalar', 'list_data', 'string'}
    assert set(df.columns) == expected_cols
    assert len(df) == len(exp_matrix_data['t'])
    assert df['scalar'].iloc[0] == 42


def test_make_dataframe_from_matrix_with_signal_disp(exp_matrix_data):
    """Test basic conversion including signal_disp."""
    # Create a copy and remove the columns that fail validation
    valid_matrix_data = exp_matrix_data.copy()
    valid_matrix_data.pop('multi_dim', None)
    valid_matrix_data.pop('mismatched_array', None)
    valid_matrix_data.pop('complex_array', None)
    
    df = make_dataframe_from_matrix(valid_matrix_data, include_signal_disp=True, metadata=None)
    
    # Expect same as basic + 'signal_disp'
    expected_cols = {'t', 'x', 'scalar', 'list_data', 'string', 'signal_disp'}
    assert set(df.columns) == expected_cols
    assert 'signal_disp' in df.columns
    assert len(df) == len(exp_matrix_data['t'])


def test_make_dataframe_from_matrix_with_metadata(exp_matrix_data, sample_metadata):
    """Test conversion with metadata added."""
    # Create a copy and remove the columns that fail validation
    valid_matrix_data = exp_matrix_data.copy()
    valid_matrix_data.pop('multi_dim', None)
    valid_matrix_data.pop('mismatched_array', None)
    valid_matrix_data.pop('complex_array', None)
    
    df = make_dataframe_from_matrix(valid_matrix_data, include_signal_disp=False, metadata=sample_metadata)
    
    # Expect basic columns (excluding multi_dim) + metadata keys
    basic_cols = {'t', 'x', 'scalar', 'list_data', 'string'}
    expected_cols = basic_cols.union(set(sample_metadata.keys()))
    assert set(df.columns) == expected_cols
    assert 'fly_id' in df.columns
    assert df['fly_id'].iloc[0] == sample_metadata['fly_id']
    assert len(df) == len(exp_matrix_data['t'])


def test_make_dataframe_from_matrix_with_column_list(exp_matrix_data):
    """Test conversion using an explicit column list."""
    column_list = ['t', 'x', 'scalar'] # Explicitly select these
    df = make_dataframe_from_matrix(
        exp_matrix_data, 
        metadata=None, 
        include_signal_disp=False, # Exclude signal_disp unless in list
        column_list=column_list
    )

    # Expect only columns from the list
    assert set(df.columns) == set(column_list)
    assert len(df) == len(exp_matrix_data['t'])


def test_make_dataframe_from_matrix_missing_time_column():
    """Test that appropriate error is raised when t column is missing."""
    invalid_exp_matrix = {'x': np.arange(10)}

    with pytest.raises(ValueError, match=r"exp_matrix must contain a 1D numpy array named 't'"):
        make_dataframe_from_matrix(invalid_exp_matrix)


def test_make_dataframe_from_matrix_complex_case(exp_matrix_data, sample_metadata):
    """Test a complex case with specific columns, signal_disp, and metadata."""
    # Define specific columns to include, including signal_disp
    column_list = ['t', 'x', 'scalar', 'signal_disp'] 
    
    # Call the function requesting these columns, signal_disp, and metadata
    df = make_dataframe_from_matrix(
        exp_matrix_data,
        metadata=sample_metadata,
        include_signal_disp=True, # Redundant if signal_disp in list, but good practice
        column_list=column_list
    )

    # Verify DataFrame has exactly the requested data columns + metadata columns
    expected_columns = set(column_list).union(set(sample_metadata.keys()))
    assert set(df.columns) == expected_columns
    assert 'signal_disp' in df.columns
    assert 'fly_id' in df.columns
    assert len(df) == len(exp_matrix_data['t'])


def test_make_dataframe_from_matrix_missing_column_warning(
    exp_matrix_data, sample_metadata, caplog
):
    """Test that a warning is logged when a requested column is missing."""
    non_existent_col = "ghost_column"
    column_list = ['t', 'x', non_existent_col] # Include a column not in the fixture
    
    # Set caplog level to capture WARNING messages
    caplog.set_level(logging.WARNING)
    
    # Call the function that should trigger the warning via _filter_matrix_columns
    df = make_dataframe_from_matrix(
        exp_matrix_data, 
        metadata=sample_metadata, 
        column_list=column_list
    )

    # Assert that the warning was logged
    assert any(
        non_existent_col in record.message and record.levelno == logging.WARNING
        for record in caplog.records
    ), f"Expected WARNING log for missing column '{non_existent_col}' not found."
    
    # Also assert that the valid columns were still processed
    assert 't' in df.columns
    assert 'x' in df.columns
    assert non_existent_col not in df.columns # The non-existent column shouldn't be added
    assert 'fly_id' in df.columns # Check for an actual metadata column


# === Error Handling Tests ===
