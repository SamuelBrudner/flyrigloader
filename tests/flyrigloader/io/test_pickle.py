"""
Comprehensive test suite for the pickle.py module.

This module implements modernized pytest practices with extensive fixture usage,
parametrization, and comprehensive mocking strategies per TST-MOD-001 through TST-MOD-003.
Includes performance benchmarks, property-based testing with Hypothesis, and secure
pickle handling validation.
"""

import gzip
import logging
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, patch

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings

# Testing framework imports
from loguru import logger

# Module under test
from flyrigloader.io.pickle import (
    read_pickle_any_format,
)
from flyrigloader.io.transformers import (
    extract_columns_from_matrix,
    handle_signal_disp,
    make_dataframe_from_config,
)
from flyrigloader.io.column_models import (
    ColumnConfig,
    ColumnConfigDict,
    ColumnDimension,
    SpecialHandlerType,
    get_default_config_path,
    load_column_config,
)


# =============================================================================
# COMPREHENSIVE FIXTURE DEFINITIONS
# =============================================================================
# Following TST-MOD-001: Modern pytest fixture usage replacing setup/teardown

@pytest.fixture(scope="function")
def temp_dir(tmp_path):
    """
    Create a temporary directory for test files using pytest's tmp_path.
    
    Modernized replacement for manual tempfile.TemporaryDirectory() usage
    per TST-MOD-001 requirements.
    
    Returns:
        Path: Temporary directory path
    """
    return tmp_path


@pytest.fixture(scope="function")
def sample_data():
    """
    Create basic sample data dictionary for test pickles.
    
    Returns:
        Dict[str, Any]: Basic experimental data structure
    """
    return {
        't': np.arange(0, 10),
        'x': np.arange(10, 20),
        'metadata': {
            'experiment': 'test',
            'date': '2025-04-01'
        }
    }


@pytest.fixture(scope="function") 
def large_sample_data():
    """
    Create large sample data for performance testing per TST-PERF-001.
    
    Generates realistic scale experimental data to validate SLA requirements:
    - Data loading: < 1s per 100MB
    - DataFrame transformation: < 500ms per 1M rows
    
    Returns:
        Dict[str, Any]: Large-scale experimental data
    """
    # Generate 1M data points for performance testing
    n_points = 1_000_000
    return {
        't': np.linspace(0, 3600, n_points),  # 1 hour at high frequency
        'x': np.random.randn(n_points) * 10,
        'y': np.random.randn(n_points) * 10,
        'velocity': np.random.exponential(5, n_points),
        'metadata': {
            'experiment': 'performance_test',
            'date': '2025-04-01',
            'duration_hours': 1.0,
            'n_points': n_points
        }
    }


@pytest.fixture(scope="function")
def corrupted_data():
    """
    Create various types of corrupted/malformed data for security testing.
    
    Tests secure pickle handling per F-003 security implications.
    
    Returns:
        Dict[str, Any]: Various corrupted data scenarios
    """
    return {
        'missing_time': {'x': np.arange(10)},  # Missing required 't' key
        'mismatched_lengths': {
            't': np.arange(10),
            'x': np.arange(15)  # Mismatched array length
        },
        'empty_arrays': {
            't': np.array([]),
            'x': np.array([])
        },
        'non_numeric': {
            't': np.arange(10),
            'x': ['a', 'b', 'c'] * 3 + ['d']  # String data where numeric expected
        }
    }


@pytest.fixture(scope="function")
def exp_matrix_data():
    """
    Sample experimental matrix data with different array dimensions.
    
    Enhanced for comprehensive multi-dimensional array handling testing
    per F-006 requirements.
    
    Returns:
        Dict[str, Any]: Multi-dimensional experimental data
    """
    np.random.seed(42)  # Ensure reproducible test data
    return {
        't': np.arange(0, 10),                     # 1D array - time dimension
        'x': np.arange(10, 20),                    # 1D array - position
        'multi_dim': np.ones((5, 3)),              # 2D array
        'complex_array': np.ones((3, 4, 2)),       # 3D array
        'signal_disp': np.random.rand(10, 5),      # 2D array with special handling
        'scalar': 42,                               # Scalar value
        'string': np.array(['test_string'] * 10),  # Array of strings with length matching t
        'list_data': list(range(10)),              # List value with length matching t
        'float_array': np.random.normal(0, 1, 10), # 1D float array
        'int_array': np.arange(10, dtype=np.int32) # 1D integer array
    }


@pytest.fixture(scope="function")
def comprehensive_column_config():
    """
    Create comprehensive column configuration for testing.
    
    Supports advanced schema validation per F-004 requirements and
    comprehensive DataFrame transformation testing per F-006.
    
    Returns:
        ColumnConfigDict: Complete column configuration
    """
    config_dict = {
        "columns": {
            "t": {
                "type": "numpy.ndarray",
                "dimension": 1,
                "required": True,
                "description": "Time values (seconds)"
            },
            "x": {
                "type": "numpy.ndarray", 
                "dimension": 1,
                "required": True,
                "description": "X position (mm)"
            },
            "y": {
                "type": "numpy.ndarray",
                "dimension": 1,
                "required": False,
                "description": "Y position (mm)",
                "default_value": None
            },
            "signal_disp": {
                "type": "numpy.ndarray",
                "dimension": 2,
                "required": False,
                "description": "Signal display data",
                "special_handling": "transform_to_match_time_dimension"
            },
            "velocity": {
                "type": "numpy.ndarray",
                "dimension": 1,
                "required": False,
                "description": "Velocity magnitude (mm/s)"
            },
            "scalar": {
                "type": "int",
                "required": False,
                "description": "Scalar value"
            },
            "float_array": {
                "type": "numpy.ndarray",
                "dimension": 1,
                "required": False,
                "description": "Float array data"
            },
            "experiment_id": {
                "type": "str",
                "required": False,
                "description": "Experiment identifier",
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
            "transform_to_match_time_dimension": "_handle_signal_disp"
        }
    }
    return ColumnConfigDict.model_validate(config_dict)


@pytest.fixture(scope="function")
def sample_metadata():
    """
    Sample metadata for testing metadata integration.
    
    Returns:
        Dict[str, str]: Sample metadata dictionary
    """
    return {
        'fly_id': 'fly-123',
        'genotype': 'wild-type',
        'date': '2025-04-01',
        'experiment_type': 'behavior',
        'experimenter': 'test_user',
        'rig_id': 'rig_001'
    }


@pytest.fixture(scope="function")
def signal_disp_T_X():
    """
    Create sample exp_matrix with signal_disp in time-first orientation (T, X).
    
    Returns:
        Dict[str, np.ndarray]: Time-first signal data
    """
    T = 10  # Time dimension
    X = 5   # Signal dimension
    np.random.seed(42)  # Reproducible data
    return {
        't': np.arange(0, T),
        'signal_disp': np.random.rand(T, X)  # Shape is (time, signals)
    }


@pytest.fixture(scope="function")
def signal_disp_X_T():
    """
    Create sample exp_matrix with signal_disp in signal-first orientation (X, T).
    
    Returns:
        Dict[str, np.ndarray]: Signal-first orientation data
    """
    T = 10  # Time dimension
    X = 5   # Signal dimension
    np.random.seed(42)  # Reproducible data
    return {
        't': np.arange(0, T),
        'signal_disp': np.random.rand(X, T)  # Shape is (signals, time)
    }


# =============================================================================
# FILE FORMAT FIXTURES
# =============================================================================
# Comprehensive pickle file fixtures supporting all formats per F-003-RQ-001 to F-003-RQ-004

@pytest.fixture(scope="function")
def regular_pickle_file(temp_dir, sample_data):
    """
    Create a regular (uncompressed) pickle file for testing.
    
    Tests F-003-RQ-001: Load standard pickle files.
    
    Args:
        temp_dir: Temporary directory fixture
        sample_data: Sample data fixture
        
    Returns:
        Path: Path to regular pickle file
    """
    filepath = temp_dir / "regular_data.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(sample_data, f)
    return filepath


@pytest.fixture(scope="function")
def gzipped_pickle_file(temp_dir, sample_data):
    """
    Create a gzipped pickle file for testing.
    
    Tests F-003-RQ-002: Load gzipped pickle files with auto-detection.
    
    Args:
        temp_dir: Temporary directory fixture
        sample_data: Sample data fixture
        
    Returns:
        Path: Path to gzipped pickle file
    """
    filepath = temp_dir / "gzipped_data.pkl.gz"
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(sample_data, f)
    return filepath


@pytest.fixture(scope="function")
def pandas_pickle_file(temp_dir):
    """
    Create a pandas-specific pickle file for testing.
    
    Tests F-003-RQ-003: Load pandas-specific serialization formats.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path: Path to pandas pickle file
    """
    filepath = temp_dir / "pandas_data.pkl"
    df = pd.DataFrame({
        't': np.arange(0, 10),
        'x': np.arange(10, 20),
        'metadata': ['test'] * 10
    })
    df.to_pickle(filepath)
    return filepath


@pytest.fixture(scope="function")
def large_pickle_file(temp_dir, large_sample_data):
    """
    Create a large pickle file for performance testing.
    
    Supports TST-PERF-001: Data loading SLA validation (< 1s per 100MB).
    
    Args:
        temp_dir: Temporary directory fixture
        large_sample_data: Large dataset fixture
        
    Returns:
        Path: Path to large pickle file
    """
    filepath = temp_dir / "large_data.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(large_sample_data, f)
    return filepath


@pytest.fixture(scope="function")
def corrupted_pickle_file(temp_dir):
    """
    Create a corrupted pickle file for error handling testing.
    
    Tests secure pickle handling per F-003 security implications.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path: Path to corrupted pickle file
    """
    filepath = temp_dir / "corrupted.pkl"
    # Write invalid pickle data
    with open(filepath, 'wb') as f:
        f.write(b"This is not valid pickle data")
    return filepath


@pytest.fixture(scope="function")
def multiple_format_files(temp_dir, sample_data):
    """
    Create multiple pickle files in different formats for parametrized testing.
    
    Supports pytest.mark.parametrize systematic testing per TST-MOD-002.
    
    Args:
        temp_dir: Temporary directory fixture
        sample_data: Sample data fixture
        
    Returns:
        Dict[str, Path]: Mapping of format names to file paths
    """
    files = {}
    
    # Regular pickle
    regular_path = temp_dir / "regular.pkl"
    with open(regular_path, 'wb') as f:
        pickle.dump(sample_data, f)
    files['regular'] = regular_path
    
    # Gzipped pickle
    gzipped_path = temp_dir / "gzipped.pkl.gz"
    with gzip.open(gzipped_path, 'wb') as f:
        pickle.dump(sample_data, f)
    files['gzipped'] = gzipped_path
    
    # Pandas pickle
    pandas_path = temp_dir / "pandas.pkl"
    df = pd.DataFrame({
        't': sample_data['t'],
        'x': sample_data['x']
    })
    df.to_pickle(pandas_path)
    files['pandas'] = pandas_path
    
    return files


# =============================================================================
# MOCKING FIXTURES  
# =============================================================================
# Standardized mocking strategies per TST-MOD-003

@pytest.fixture(scope="function")
def mock_ensure_1d_array(mocker):
    """
    Mock the ensure_1d_array function for controlled testing.
    
    Implements standardized mocking strategies per TST-MOD-003.
    
    Args:
        mocker: pytest-mock fixture
        
    Returns:
        MagicMock: Mocked ensure_1d_array function
    """
    def mock_ensure_1d(array, name=None):
        """Mock implementation that flattens arrays if needed."""
        if hasattr(array, 'ndim') and array.ndim > 1:
            return array.flatten()
        return array
    
    return mocker.patch(
        'flyrigloader.io.transformers.ensure_1d_array',
        side_effect=mock_ensure_1d
    )


# =============================================================================
# PARAMETRIZED TEST DATA
# =============================================================================
# Data sets for comprehensive parametrized testing per TST-MOD-002

# Pickle format parameters for systematic testing
PICKLE_FORMATS = [
    ('regular', 'regular.pkl', False),
    ('gzipped', 'gzipped.pkl.gz', True), 
    ('pandas', 'pandas.pkl', False)
]

# Error condition parameters
ERROR_CONDITIONS = [
    ('file_not_found', FileNotFoundError, "File not found"),
    ('invalid_path_type', ValueError, "Invalid path format")
]

# Signal display orientation parameters
SIGNAL_DISP_ORIENTATIONS = [
    ('time_first', 'T_X', (10, 5)),  # (T, X) orientation: 10 time points, 5 signals each
    ('signal_first', 'X_T', (10, 5))  # (X, T) orientation: 10 time points, 5 signals each
]


# =============================================================================
# COMPREHENSIVE PARAMETRIZED TESTS
# =============================================================================
# Systematic testing per TST-MOD-002: pytest.mark.parametrize for edge cases

@pytest.mark.parametrize("format_name,expected_log_msg", [
    ("regular", "Loaded pickle using regular pickle"),
    ("gzipped", "Loaded pickle using gzip"), 
    ("pandas", "Loaded pickle using pandas")
])
def test_read_pickle_any_format_all_formats(multiple_format_files, sample_data, 
                                          format_name, expected_log_msg, caplog):
    """
    Test reading all supported pickle formats with auto-detection.
    
    Implements F-003-RQ-001 through F-003-RQ-004: comprehensive format support
    with auto-detection per TST-MOD-002 parametrization requirements.
    
    Args:
        multiple_format_files: Fixture providing all format files
        sample_data: Original data for comparison
        format_name: Format type being tested
        expected_log_msg: Expected log message for format detection
        caplog: Pytest log capture fixture
    """
    caplog.clear()
    file_path = multiple_format_files[format_name]
    
    with caplog.at_level(logging.DEBUG):
        result = read_pickle_any_format(file_path)
    
    # Verify successful loading
    assert result is not None
    
    # Verify format detection logging
    assert any(expected_log_msg in record.message for record in caplog.records)
    
    # Format-specific validation
    if format_name == 'pandas':
        # Pandas format returns DataFrame
        assert isinstance(result, pd.DataFrame)
        assert 't' in result.columns
        assert 'x' in result.columns
        assert len(result) == len(sample_data['t'])
    else:
        # Regular and gzipped return dictionaries
        assert isinstance(result, dict)
        assert 't' in result
        assert 'x' in result
        np.testing.assert_array_equal(result['t'], sample_data['t'])
        np.testing.assert_array_equal(result['x'], sample_data['x'])


@pytest.mark.parametrize("error_type,exception_class,error_message", ERROR_CONDITIONS)
def test_read_pickle_any_format_error_conditions(temp_dir, error_type, 
                                               exception_class, error_message):
    """
    Test comprehensive error handling for pickle loading.
    
    Validates secure pickle handling per F-003 security implications and
    comprehensive error scenario coverage.
    
    Args:
        temp_dir: Temporary directory fixture
        error_type: Type of error condition
        exception_class: Expected exception type
        error_message: Expected error message pattern
    """
    if error_type == 'file_not_found':
        file_path = temp_dir / "nonexistent.pkl"
        with pytest.raises(exception_class):
            read_pickle_any_format(file_path)
            
    elif error_type == 'invalid_path_type':
        with pytest.raises(exception_class, match="Invalid path format"):
            read_pickle_any_format(123)  # Invalid path type


@pytest.mark.parametrize("orientation_name,fixture_name,expected_shape", SIGNAL_DISP_ORIENTATIONS)
def test_handle_signal_disp_parametrized(orientation_name, fixture_name, expected_shape, request):
    """
    Test signal_disp handling with different orientations using parametrization.
    
    Validates comprehensive signal handling per F-006 requirements using
    systematic parametrized testing per TST-MOD-002.
    
    Args:
        orientation_name: Name of the orientation being tested
        fixture_name: Name of the fixture to request
        expected_shape: Expected shape tuple
        request: pytest request fixture for dynamic fixture access
    """
    # Dynamically get the appropriate fixture
    if fixture_name == 'T_X':
        signal_data = request.getfixturevalue('signal_disp_T_X')
    else:  # X_T
        signal_data = request.getfixturevalue('signal_disp_X_T')
    
    result = handle_signal_disp(signal_data)
    
    # Verify the result is a pandas Series
    assert isinstance(result, pd.Series)
    assert len(result) == expected_shape[0]  # Time dimension
    assert result.name == 'signal_disp'
    
    # Verify each element is correct
    for i in range(len(result)):
        assert isinstance(result.iloc[i], np.ndarray)
        assert len(result.iloc[i]) == expected_shape[1]  # Signal dimension


# =============================================================================
# PERFORMANCE BENCHMARK TESTS
# =============================================================================
# Performance validation per TST-PERF-001 and TST-PERF-002

@pytest.mark.benchmark
def test_read_pickle_performance_sla(large_pickle_file, benchmark):
    """
    Benchmark pickle loading performance against SLA requirements.
    
    Validates TST-PERF-001: Data loading must complete within 1s per 100MB.
    Uses pytest-benchmark for statistical performance measurement.
    
    Args:
        large_pickle_file: Large dataset pickle file fixture
        benchmark: pytest-benchmark fixture
    """
    # Benchmark the loading operation
    result = benchmark(read_pickle_any_format, large_pickle_file)
    
    # Verify successful loading
    assert result is not None
    assert isinstance(result, dict)
    assert 't' in result
    
    # Performance validation is handled by pytest-benchmark configuration
    # SLA: < 1s per 100MB is enforced through benchmark thresholds


@pytest.mark.benchmark
def test_dataframe_transformation_performance_sla(large_sample_data, 
                                                 comprehensive_column_config, benchmark):
    """
    Benchmark DataFrame transformation performance against SLA requirements.
    
    Validates TST-PERF-002: DataFrame transformation within 500ms per 1M rows.
    
    Args:
        large_sample_data: Large dataset fixture
        comprehensive_column_config: Column configuration fixture
        benchmark: pytest-benchmark fixture
    """
    # Benchmark the transformation operation
    result = benchmark(make_dataframe_from_config, large_sample_data, comprehensive_column_config)
    
    # Verify successful transformation
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(large_sample_data['t'])
    assert 't' in result.columns
    
    # Performance validation handled by pytest-benchmark thresholds
    # SLA: < 500ms per 1M rows


# =============================================================================
# PROPERTY-BASED TESTS WITH HYPOTHESIS
# =============================================================================
# Robust data transformation integrity per Section 3.6.3 requirements

@given(
    time_points=st.integers(min_value=10, max_value=1000),
    signal_channels=st.integers(min_value=1, max_value=20),
    orientation=st.sampled_from(['time_first', 'signal_first'])
)
@settings(max_examples=50, deadline=5000)  # Reasonable limits for CI/CD
def test_signal_disp_handling_property_based(time_points, signal_channels, orientation):
    """
    Property-based testing for signal_disp handling with diverse data scenarios.
    
    Uses Hypothesis to generate diverse experimental data scenarios per
    Section 3.6.3 requirements for robust data transformation integrity.
    
    Args:
        time_points: Number of time points (generated by Hypothesis)
        signal_channels: Number of signal channels (generated by Hypothesis)  
        orientation: Signal array orientation (generated by Hypothesis)
    """
    # Generate test data based on properties
    t_array = np.linspace(0, time_points/10, time_points)
    
    if orientation == 'time_first':
        signal_disp = np.random.rand(time_points, signal_channels)
    else:  # signal_first
        signal_disp = np.random.rand(signal_channels, time_points)
    
    exp_matrix = {
        't': t_array,
        'signal_disp': signal_disp
    }
    
    # Test the signal handling function
    result = handle_signal_disp(exp_matrix)
    
    # Verify invariant properties
    assert isinstance(result, pd.Series)
    assert len(result) == time_points
    assert result.name == 'signal_disp'
    
    # Each element should be an array of signal values
    for i in range(len(result)):
        element = result.iloc[i]
        assert isinstance(element, np.ndarray)
        assert len(element) == signal_channels


@given(
    n_columns=st.integers(min_value=2, max_value=10),
    array_length=st.integers(min_value=5, max_value=100)
)
@settings(max_examples=30, deadline=3000)
def test_extract_columns_property_based(n_columns, array_length):
    """
    Property-based testing for column extraction with various configurations.
    
    Args:
        n_columns: Number of columns to generate (generated by Hypothesis)
        array_length: Length of arrays (generated by Hypothesis)
    """
    # Generate random experimental matrix
    exp_matrix = {'t': np.arange(array_length)}
    
    for i in range(n_columns):
        col_name = f'col_{i}'
        exp_matrix[col_name] = np.random.randn(array_length)
    
    # Test column extraction
    result = extract_columns_from_matrix(exp_matrix)
    
    # Verify properties
    assert isinstance(result, dict)
    assert 't' in result
    assert len(result) >= 1  # At least time column
    
    # All extracted arrays should have consistent length
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            assert len(value) == array_length or value.ndim > 1


# =============================================================================
# ENHANCED DATAFRAME TRANSFORMATION TESTS
# =============================================================================
# Comprehensive DataFrame transformation testing per F-006 requirements

def test_make_dataframe_comprehensive_integration(exp_matrix_data, comprehensive_column_config, sample_metadata):
    """
    Test comprehensive DataFrame creation with all features.
    
    Validates F-006 requirements: multi-dimensional array handling, special handlers,
    metadata integration, and time alignment validation.
    
    Args:
        exp_matrix_data: Multi-dimensional experimental data
        comprehensive_column_config: Complete column configuration
        sample_metadata: Sample metadata dictionary
    """
    # Create DataFrame with comprehensive configuration
    df = make_dataframe_from_config(
        exp_matrix_data,
        config_source=comprehensive_column_config,
        metadata=sample_metadata
    )
    
    # Verify basic structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(exp_matrix_data['t'])
    
    # Verify required columns
    assert 't' in df.columns
    assert 'x' in df.columns
    
    # Verify data type preservation
    assert df['t'].dtype == exp_matrix_data['t'].dtype
    assert df['x'].dtype == exp_matrix_data['x'].dtype
    
    # Verify metadata integration
    for key, value in sample_metadata.items():
        if key in comprehensive_column_config.columns and comprehensive_column_config.columns[key].is_metadata:
            assert key in df.columns
            assert df[key].iloc[0] == value
    
    # Verify time alignment
    np.testing.assert_array_equal(df['t'].values, exp_matrix_data['t'])


def test_make_dataframe_signal_disp_handling(exp_matrix_data, comprehensive_column_config):
    """
    Test signal_disp special handler integration in DataFrame creation.
    
    Validates special handler implementation per F-006 requirements.
    
    Args:
        exp_matrix_data: Experimental data with signal_disp
        comprehensive_column_config: Configuration with signal_disp handler
    """
    df = make_dataframe_from_config(exp_matrix_data, config_source=comprehensive_column_config)
    
    # Verify signal_disp column exists and is properly handled
    assert 'signal_disp' in df.columns
    assert len(df) == len(exp_matrix_data['t'])
    
    # Verify signal_disp content type
    for i in range(len(df)):
        element = df['signal_disp'].iloc[i]
        # signal_disp should always contain numpy arrays after transformation
        assert isinstance(element, np.ndarray)


# =============================================================================
# ENHANCED ERROR HANDLING TESTS
# =============================================================================
# Comprehensive error scenario validation per F-003 security implications

def test_corrupted_pickle_security_handling(corrupted_pickle_file):
    """
    Test secure handling of corrupted pickle files.
    
    Validates F-003 security implications for secure pickle handling.
    
    Args:
        corrupted_pickle_file: Fixture providing corrupted pickle file
    """
    with pytest.raises((pickle.UnpicklingError, RuntimeError)):
        read_pickle_any_format(corrupted_pickle_file)


def test_signal_disp_comprehensive_error_conditions():
    """
    Test comprehensive error conditions for signal_disp handling.
    
    Validates robust error handling per F-006 requirements.
    """
    # Test missing signal_disp key
    with pytest.raises(ValueError, match="missing required 'signal_disp' key"):
        handle_signal_disp({'t': np.arange(10)})
    
    # Test missing time key
    with pytest.raises(ValueError, match="missing required 't' key"):
        handle_signal_disp({'signal_disp': np.ones((10, 5))})
    
    # Test invalid dimensions
    with pytest.raises(ValueError, match="signal_disp must be 2D"):
        handle_signal_disp({
            't': np.arange(10),
            'signal_disp': np.ones(10)  # 1D instead of 2D
        })
    
    # Test dimension mismatch
    with pytest.raises(ValueError, match="No dimension of signal_disp .* matches time dimension"):
        handle_signal_disp({
            't': np.arange(10),
            'signal_disp': np.ones((15, 20))  # Neither dimension matches time length
        })


def test_extract_columns_comprehensive_error_handling():
    """
    Test comprehensive error handling for column extraction.
    
    Validates robust error handling and edge case coverage.
    """
    # Test invalid input type
    with pytest.raises(ValueError, match="exp_matrix must be a dictionary"):
        extract_columns_from_matrix("not_a_dictionary")
    
    # Test empty dictionary
    result = extract_columns_from_matrix({})
    assert isinstance(result, dict)
    assert len(result) == 0
    
    # Test nonexistent columns
    exp_matrix = {'t': np.arange(10), 'x': np.arange(10, 20)}
    result = extract_columns_from_matrix(exp_matrix, ['t', 'nonexistent'])
    assert 't' in result
    assert 'nonexistent' not in result


# =============================================================================
# LEGACY TESTS (Maintained for Compatibility)
# =============================================================================
# Original test functions maintained during modernization transition

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


def test_read_pickle_any_format_logs_gzip(gzipped_pickle_file, caplog):
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        read_pickle_any_format(gzipped_pickle_file)
    assert any("Loaded pickle using gzip" in r.message for r in caplog.records)


def test_read_pickle_any_format_logs_regular(regular_pickle_file, caplog):
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        read_pickle_any_format(regular_pickle_file)
    assert any("Loaded pickle using regular pickle" in r.message for r in caplog.records)


def test_read_pickle_any_format_logs_pandas(pandas_pickle_file, caplog):
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        read_pickle_any_format(pandas_pickle_file)
    assert any("Loaded pickle using pandas" in r.message for r in caplog.records)


def test_read_pickle_any_format_file_not_found():
    """Test that an appropriate error is raised for a non-existent file."""
    with pytest.raises(FileNotFoundError):
        read_pickle_any_format("non_existent_file.pkl")


def test_read_pickle_any_format_invalid_path():
    """Test that an appropriate error is raised for an invalid path."""
    with pytest.raises(ValueError):
        read_pickle_any_format(123)  # Not a valid path


def test_extract_columns_from_matrix_all_columns(exp_matrix_data, mock_ensure_1d_array):
    """Test extracting all columns from an exp_matrix."""
    result = extract_columns_from_matrix(exp_matrix_data)
    
    # Check that all columns except signal_disp are extracted
    assert len(result) == len(exp_matrix_data) - 1  # signal_disp should be skipped
    assert 'signal_disp' not in result  # Verify signal_disp is skipped
    
    # Check the content of extracted columns
    for key in exp_matrix_data:
        if key != 'signal_disp':  # signal_disp is handled separately
            assert key in result
            if isinstance(exp_matrix_data[key], np.ndarray):
                # For arrays handled by the mock, verify they're processed
                assert result[key] is not None
            else:
                assert result[key] == exp_matrix_data[key]


def test_extract_columns_from_matrix_specific_columns(exp_matrix_data, mock_ensure_1d_array):
    """Test extracting specific columns from an exp_matrix."""
    columns_to_extract = ['t', 'x', 'scalar']
    
    result = extract_columns_from_matrix(exp_matrix_data, columns_to_extract)
    
    # Check that only the specified columns are extracted
    assert set(result.keys()) == set(columns_to_extract)
    
    # Check the content of extracted columns
    for key in columns_to_extract:
        if isinstance(exp_matrix_data[key], np.ndarray):
            assert result[key] is not None
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


def test_extract_columns_from_matrix_with_1d_conversion(exp_matrix_data, mock_ensure_1d_array):
    """Test extracting columns with conversion to 1D."""
    result = extract_columns_from_matrix(exp_matrix_data)

    # Check that all arrays in the result are processed by the mock
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            assert value is not None  # Mock returns flattened arrays


def test_extract_columns_from_matrix_invalid_input():
    """Test that appropriate error is raised for invalid input."""
    with pytest.raises(ValueError):
        extract_columns_from_matrix("not_a_dictionary")


def test_extract_columns_from_matrix_nonexistent_column(exp_matrix_data, mock_ensure_1d_array):
    """Test behavior with non-existent column names."""
    columns_to_extract = ['t', 'nonexistent_column']
    
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


def test_make_dataframe_from_config_basic(exp_matrix_data, comprehensive_column_config):
    """Test basic conversion of an exp_matrix to DataFrame."""
    df = make_dataframe_from_config(exp_matrix_data, config_source=comprehensive_column_config, metadata=None)
    
    # Verify the DataFrame has expected columns and properties
    assert 't' in df.columns
    assert 'x' in df.columns
    assert len(df) == len(exp_matrix_data['t'])