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
import logging  # Keep for log level constants used with pytest caplog
from loguru import logger
import contextlib

# This will be the module we're testing
from flyrigloader.io.pickle import (
    read_pickle_any_format, 
    extract_columns_from_matrix, 
    handle_signal_disp,
    make_dataframe_from_config
)
from flyrigloader.io.column_models import ColumnDimension, get_default_config_path, load_column_config


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


def test_make_dataframe_from_config_basic(exp_matrix_data):
    """Test basic conversion of an exp_matrix to DataFrame without signal_disp."""
    # Create a copy and remove the columns that fail validation
    valid_matrix_data = exp_matrix_data.copy()
    valid_matrix_data.pop('multi_dim', None)
    valid_matrix_data.pop('mismatched_array', None)
    valid_matrix_data.pop('complex_array', None)
    
    # Create minimal custom config with all required fields
    custom_config = {
        "columns": {
            "t": {
                "type": "numpy.ndarray",
                "dimension": 1,
                "required": True,
                "description": "Time values"
            },
            "x": {
                "type": "numpy.ndarray",
                "dimension": 1,
                "required": False,
                "description": "X position"
            },
            "scalar": {
                "type": "int",
                "required": False,
                "description": "Scalar value"
            }
        },
        "special_handlers": {}
    }
    
    df = make_dataframe_from_config(valid_matrix_data, config_source=custom_config, metadata=None)
    
    # Verify the DataFrame has expected columns and properties
    assert 't' in df.columns
    assert 'x' in df.columns
    assert len(df) == len(exp_matrix_data['t'])
    assert 'scalar' in df.columns and df['scalar'].iloc[0] == 42


def test_make_dataframe_from_config_with_signal_disp(exp_matrix_data):
    """Test basic conversion including signal_disp."""
    # Create a copy and remove the columns that fail validation
    valid_matrix_data = exp_matrix_data.copy()
    valid_matrix_data.pop('multi_dim', None)
    valid_matrix_data.pop('mismatched_array', None)
    valid_matrix_data.pop('complex_array', None)
    
    # Create a config that includes signal_disp
    custom_config = {
        "columns": {
            "t": {
                "type": "numpy.ndarray",
                "dimension": 1,
                "required": True,
                "description": "Time values"
            },
            "x": {
                "type": "numpy.ndarray",
                "dimension": 1,
                "required": False,
                "description": "X position"
            },
            "signal_disp": {
                "type": "numpy.ndarray",
                "dimension": 2,
                "required": False,
                "description": "Signal display data",
                "special_handling": "transform_to_match_time_dimension"
            }
        },
        "special_handlers": {
            "transform_to_match_time_dimension": "signal_disp"
        }
    }
    
    df = make_dataframe_from_config(valid_matrix_data, config_source=custom_config, metadata=None)
    
    # Verify signal_disp is included
    assert 'signal_disp' in df.columns
    assert len(df) == len(exp_matrix_data['t'])


def test_make_dataframe_from_config_with_metadata(exp_matrix_data, sample_metadata):
    """Test conversion with metadata added."""
    # Create a copy and remove the columns that fail validation
    valid_matrix_data = exp_matrix_data.copy()
    valid_matrix_data.pop('multi_dim', None)
    valid_matrix_data.pop('mismatched_array', None)
    valid_matrix_data.pop('complex_array', None)
    
    # Create a config with metadata fields
    custom_config = {
        "columns": {
            "t": {
                "type": "numpy.ndarray",
                "dimension": 1,
                "required": True,
                "description": "Time values"
            },
            "x": {
                "type": "numpy.ndarray",
                "dimension": 1,
                "required": False,
                "description": "X position"
            },
            "fly_id": {  # Match name to sample_metadata key
                "type": "str",
                "required": False,
                "description": "Fly ID",
                "is_metadata": True
            }
        },
        "special_handlers": {}
    }
    
    df = make_dataframe_from_config(valid_matrix_data, config_source=custom_config, metadata=sample_metadata)
    
    # Verify metadata fields are added to the DataFrame for those defined in config
    assert 'fly_id' in df.columns
    assert df['fly_id'].iloc[0] == sample_metadata['fly_id']


def test_make_dataframe_from_config_with_custom_config(exp_matrix_data):
    """Test conversion with custom column selection."""
    # Create a copy and remove the columns that fail validation
    valid_matrix_data = exp_matrix_data.copy()
    valid_matrix_data.pop('multi_dim', None)
    valid_matrix_data.pop('mismatched_array', None)
    valid_matrix_data.pop('complex_array', None)
    
    # Create minimal custom config that only includes certain columns
    custom_config = {
        "columns": {
            "t": {
                "type": "numpy.ndarray",
                "dimension": 1,
                "required": True,
                "description": "Time values"
            },
            "x": {
                "type": "numpy.ndarray",
                "dimension": 1,
                "required": False,
                "description": "X position"
            },
            "scalar": {
                "type": "int",
                "required": False,
                "description": "Scalar value"
            }
        },
        "special_handlers": {}
    }
    
    df = make_dataframe_from_config(valid_matrix_data, config_source=custom_config, metadata=None)
    
    # Verify only specified columns are included
    # Note: only columns in both the config and the data will be included
    assert 't' in df.columns
    assert 'x' in df.columns
    assert 'scalar' in df.columns
    assert len(df) == len(exp_matrix_data['t'])


def test_make_dataframe_from_config_missing_time_column():
    """Test handling of missing required time column."""
    # Create a matrix without the required 't' column
    invalid_exp_matrix = {
        'x': np.array([1, 2, 3])
    }
    
    # Use a minimal config that requires 't'
    custom_config = {
        "columns": {
            "t": {
                "type": "numpy.ndarray",
                "dimension": 1,
                "required": True,
                "description": "Time values"
            },
            "x": {
                "type": "numpy.ndarray",
                "dimension": 1,
                "required": False,
                "description": "X position"
            }
        },
        "special_handlers": {}
    }
    
    # This should raise ValueError due to missing required 't' column
    with pytest.raises(ValueError, match="Missing required columns: t"):
        make_dataframe_from_config(invalid_exp_matrix, config_source=custom_config)


def test_make_dataframe_from_config_complex_case(exp_matrix_data, sample_metadata):
    """Test conversion with complex combinations of options."""
    # Create a copy and remove the columns that fail validation
    valid_matrix_data = exp_matrix_data.copy()
    valid_matrix_data.pop('multi_dim', None)
    valid_matrix_data.pop('mismatched_array', None)
    valid_matrix_data.pop('complex_array', None)
    
    # Create a detailed custom config
    custom_config = {
        "columns": {
            "t": {
                "type": "numpy.ndarray",
                "dimension": 1,
                "required": True,
                "description": "Time values"
            },
            "x": {
                "type": "numpy.ndarray",
                "dimension": 1,
                "required": True,
                "description": "X position"
            },
            "signal_disp": {
                "type": "numpy.ndarray",
                "dimension": 2,
                "required": False,
                "description": "Signal display data",
                "special_handling": "transform_to_match_time_dimension"
            },
            "scalar": {
                "type": "int",
                "required": False,
                "description": "Scalar value"
            },
            "experiment_id": {
                "type": "str",
                "required": False,
                "description": "Experiment ID",
                "is_metadata": True
            },
            "date": {
                "type": "str",
                "required": False,
                "description": "Experiment date",
                "is_metadata": True
            }
        },
        "special_handlers": {
            "transform_to_match_time_dimension": "signal_disp"
        }
    }
    
    df = make_dataframe_from_config(
        valid_matrix_data,
        config_source=custom_config,
        metadata=sample_metadata
    )
    
    # Verify DataFrame has expected structure
    assert 't' in df.columns
    assert 'x' in df.columns
    assert 'signal_disp' in df.columns
    assert 'scalar' in df.columns
    
    # Verify metadata fields
    for key, value in sample_metadata.items():
        if key in custom_config["columns"] and custom_config["columns"][key].get("is_metadata", False):
            assert key in df.columns
            assert df[key].iloc[0] == value


def test_make_dataframe_from_config_missing_columns(exp_matrix_data, caplog):
    """Test warning for non-existent columns in config."""
    # Create a copy with standard columns
    valid_matrix_data = exp_matrix_data.copy()
    valid_matrix_data.pop('multi_dim', None)
    valid_matrix_data.pop('mismatched_array', None)
    valid_matrix_data.pop('complex_array', None)
    
    # Create a config referencing a non-existent column
    non_existent_col = "this_column_does_not_exist"
    custom_config = {
        "columns": {
            "t": {
                "type": "numpy.ndarray",
                "dimension": 1,
                "required": True,
                "description": "Time values"
            },
            "x": {
                "type": "numpy.ndarray",
                "dimension": 1,
                "required": True,
                "description": "X position"
            },
            non_existent_col: {
                "type": "numpy.ndarray", 
                "dimension": 1,
                "required": False,
                "description": "Non-existent column"
            }
        },
        "special_handlers": {}
    }
    
    # Set logging level to capture all messages including DEBUG
    caplog.set_level(logging.DEBUG)
    
    # Function should run but log a warning
    df = make_dataframe_from_config(
        valid_matrix_data,
        config_source=custom_config
    )
    
    # Verify the non-existent column is added with NULL values
    assert non_existent_col in df.columns
    assert df[non_existent_col].isna().all(), "Expected non-existent column to contain NULL values"
    
    # Verify debug message about using default value was logged
    assert any(
        (non_existent_col in record.message and "Using default value" in record.message)
        for record in caplog.records
    ), "Expected debug message about using default value for missing column"
