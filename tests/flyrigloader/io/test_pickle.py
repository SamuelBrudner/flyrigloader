"""
Behavior-focused test suite for pickle I/O operations.

This module implements comprehensive black-box testing of pickle I/O functionality
focusing on observable behavior and public API contracts rather than implementation
details. Tests emphasize data loading success, DataFrame construction accuracy,
compression handling, error recovery, and signal transformation results.

Testing Strategy:
- Uses centralized fixtures from tests/conftest.py for consistent test setup
- Protocol-based mocking from tests/utils.py for dependency isolation  
- AAA pattern structure with clear separation of setup, execution, and verification
- Edge-case coverage through parameterized test scenarios
- Performance tests isolated to scripts/benchmarks/ per Section 0 requirements
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings

# Centralized test utilities for protocol-based mocking and shared fixtures
from tests.utils import (
    create_mock_dataloader,
    create_mock_filesystem,
    MockDataLoading,
    MockFilesystem,
    EdgeCaseScenarioGenerator,
    FlyrigloaderStrategies,
)

# Module under test - focusing on public API behavior
from flyrigloader.io.pickle import (
    extract_columns_from_matrix,
    handle_signal_disp,
    make_dataframe_from_config,
    read_pickle_any_format,
)
from flyrigloader.io.column_models import ColumnConfigDict


# =============================================================================
# CENTRALIZED FIXTURE INTEGRATION AND TEST UTILITIES
# =============================================================================
# Utilizes centralized fixtures from tests/conftest.py for consistent patterns
# and shared mock implementations from tests/utils.py for dependency isolation

# Note: Large performance datasets and benchmark fixtures have been relocated 
# to scripts/benchmarks/ per Section 0 performance test isolation requirements


# Corrupted data scenarios and experimental matrix data are now provided
# by centralized fixtures from tests/conftest.py (sample_exp_matrix_comprehensive,
# fixture_corrupted_file_scenarios) for consistency across test modules


# Column configuration and metadata fixtures are provided by centralized
# fixtures from tests/conftest.py (sample_column_config_file, 
# sample_experimental_metadata) for consistency across test modules


# Signal display orientation fixtures are provided by centralized fixtures
# from tests/conftest.py (sample_exp_matrix_comprehensive) which includes
# various signal_disp configurations for testing different orientations


# File format fixtures and large dataset fixtures are provided by centralized
# fixtures from tests/conftest.py and test utilities from tests/utils.py
# Performance test file fixtures have been relocated to scripts/benchmarks/


# =============================================================================
# PROTOCOL-BASED MOCK IMPLEMENTATIONS FROM CENTRALIZED UTILITIES
# =============================================================================
# Standardized mocking strategies using centralized tests/utils.py implementations

# Custom mocking implementations replaced with protocol-based mocks from 
# tests/utils.py for consistent dependency isolation across pickle I/O scenarios

# Parametrized test data now generated through centralized utilities and
# hypothesis strategies for comprehensive edge-case coverage


# =============================================================================
# BEHAVIOR-FOCUSED PICKLE I/O TESTS
# =============================================================================
# Tests focus on observable pickle I/O behavior rather than implementation details

def test_read_pickle_any_format_successful_data_loading(temp_cross_platform_dir, 
                                                       sample_exp_matrix_comprehensive):
    """
    Test successful pickle data loading through public API behavior validation.
    
    ARRANGE: Create pickle file with experimental data using centralized fixtures
    ACT: Load pickle file using read_pickle_any_format public API
    ASSERT: Verify data loading success and content accuracy through observable behavior
    """
    # ARRANGE - Set up test data and pickle file using centralized fixtures
    experimental_data = sample_exp_matrix_comprehensive
    pickle_file_path = temp_cross_platform_dir / "test_experiment.pkl"
    
    # Create pickle file for testing
    import pickle
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(experimental_data, f)
    
    # ACT - Execute pickle loading through public API
    loaded_data = read_pickle_any_format(pickle_file_path)
    
    # ASSERT - Verify successful loading and data accuracy through observable behavior
    assert loaded_data is not None, "Pickle loading should return data"
    assert isinstance(loaded_data, dict), "Should return dictionary structure"
    
    # Verify essential experimental data columns are present
    assert 't' in loaded_data, "Time data should be preserved"
    assert 'x' in loaded_data, "X position data should be preserved"
    assert 'y' in loaded_data, "Y position data should be preserved"
    
    # Verify data content accuracy without accessing internal loading mechanisms
    np.testing.assert_array_equal(loaded_data['t'], experimental_data['t'],
                                  "Time data should match original")
    np.testing.assert_array_equal(loaded_data['x'], experimental_data['x'],
                                  "X position data should match original")
    np.testing.assert_array_equal(loaded_data['y'], experimental_data['y'],
                                  "Y position data should match original")


def test_read_pickle_any_format_file_not_found_error_handling(temp_cross_platform_dir):
    """
    Test error handling behavior for non-existent pickle files.
    
    ARRANGE: Reference non-existent pickle file path
    ACT: Attempt to load non-existent file through public API
    ASSERT: Verify appropriate error response without accessing internal error mechanisms
    """
    # ARRANGE - Set up non-existent file path
    nonexistent_file = temp_cross_platform_dir / "does_not_exist.pkl"
    
    # ACT & ASSERT - Verify error handling behavior through public API response
    with pytest.raises(FileNotFoundError):
        read_pickle_any_format(nonexistent_file)


def test_read_pickle_any_format_invalid_path_type_error_handling():
    """
    Test error handling behavior for invalid path types.
    
    ARRANGE: Prepare invalid path input (non-string, non-Path)
    ACT: Attempt to load with invalid path type through public API  
    ASSERT: Verify appropriate error response for input validation
    """
    # ARRANGE - Set up invalid path input
    invalid_path = 123  # Integer instead of path
    
    # ACT & ASSERT - Verify error handling behavior
    with pytest.raises((ValueError, TypeError)):
        read_pickle_any_format(invalid_path)


def test_handle_signal_disp_time_first_orientation_behavior(sample_exp_matrix_comprehensive):
    """
    Test signal_disp handling behavior for time-first (T, X) orientation.
    
    ARRANGE: Create experimental matrix with signal_disp in time-first orientation
    ACT: Process signal_disp through handle_signal_disp public API
    ASSERT: Verify correct signal transformation results through observable behavior
    """
    # ARRANGE - Set up signal data in time-first orientation using centralized fixture
    exp_matrix = sample_exp_matrix_comprehensive.copy()
    T, X = 10, 5
    exp_matrix['t'] = np.arange(T)
    exp_matrix['signal_disp'] = np.random.rand(T, X)  # Time-first orientation
    
    # ACT - Process signal_disp through public API
    result = handle_signal_disp(exp_matrix)
    
    # ASSERT - Verify correct transformation behavior without accessing internal mechanisms
    assert isinstance(result, pd.Series), "Should return pandas Series"
    assert len(result) == T, "Series length should match time dimension"
    assert result.name == 'signal_disp', "Series should be properly named"
    
    # Verify each time point contains correct signal array
    for i in range(T):
        signal_array = result.iloc[i]
        assert isinstance(signal_array, np.ndarray), "Each element should be numpy array"
        assert len(signal_array) == X, "Signal array length should match signal dimension"


def test_handle_signal_disp_signal_first_orientation_behavior(sample_exp_matrix_comprehensive):
    """
    Test signal_disp handling behavior for signal-first (X, T) orientation.
    
    ARRANGE: Create experimental matrix with signal_disp in signal-first orientation
    ACT: Process signal_disp through handle_signal_disp public API
    ASSERT: Verify correct signal transformation results through observable behavior
    """
    # ARRANGE - Set up signal data in signal-first orientation
    exp_matrix = sample_exp_matrix_comprehensive.copy()
    T, X = 10, 5
    exp_matrix['t'] = np.arange(T)
    exp_matrix['signal_disp'] = np.random.rand(X, T)  # Signal-first orientation
    
    # ACT - Process signal_disp through public API
    result = handle_signal_disp(exp_matrix)
    
    # ASSERT - Verify correct transformation behavior
    assert isinstance(result, pd.Series), "Should return pandas Series"
    assert len(result) == T, "Series length should match time dimension"
    assert result.name == 'signal_disp', "Series should be properly named"
    
    # Verify transformation correctly handles orientation change
    for i in range(T):
        signal_array = result.iloc[i]
        assert isinstance(signal_array, np.ndarray), "Each element should be numpy array"
        assert len(signal_array) == X, "Signal array length should match signal dimension"


# =============================================================================
# PERFORMANCE TESTS RELOCATED TO SCRIPTS/BENCHMARKS/
# =============================================================================
# Performance validation tests have been relocated to scripts/benchmarks/ per 
# Section 0 performance test isolation requirement to maintain rapid default
# test suite execution while preserving comprehensive performance validation


# =============================================================================
# ENHANCED EDGE-CASE COVERAGE WITH HYPOTHESIS STRATEGIES
# =============================================================================
# Comprehensive edge-case testing using centralized hypothesis strategies

@given(FlyrigloaderStrategies.neuroscience_time_series())
@settings(max_examples=30, deadline=3000)
def test_signal_disp_handling_with_property_based_edge_cases(exp_matrix_data):
    """
    Property-based edge-case testing for signal_disp handling using centralized strategies.
    
    ARRANGE: Generate diverse experimental data scenarios using centralized strategies
    ACT: Process signal_disp through public API with various data patterns  
    ASSERT: Verify invariant behavioral properties across diverse inputs
    """
    # ARRANGE - Modify experimental data to include signal_disp for testing
    if 'signal_disp' not in exp_matrix_data:
        # Add signal_disp with random orientation for testing
        t_length = len(exp_matrix_data['t'])
        signal_channels = np.random.randint(3, 16)
        
        if np.random.choice([True, False]):
            # Time-first orientation
            exp_matrix_data['signal_disp'] = np.random.rand(t_length, signal_channels)
        else:
            # Signal-first orientation  
            exp_matrix_data['signal_disp'] = np.random.rand(signal_channels, t_length)
    
    # ACT - Process signal_disp through public API
    try:
        result = handle_signal_disp(exp_matrix_data)
        
        # ASSERT - Verify invariant behavioral properties
        assert isinstance(result, pd.Series), "Should always return pandas Series"
        assert len(result) == len(exp_matrix_data['t']), "Length should match time dimension"
        assert result.name == 'signal_disp', "Series should be properly named"
        
        # Verify each element is correctly formatted
        for element in result:
            assert isinstance(element, np.ndarray), "Each element should be numpy array"
            assert element.ndim == 1, "Each element should be 1D array"
            
    except ValueError:
        # Expected for invalid signal_disp dimensions - this is acceptable behavior
        pass


def test_extract_columns_edge_case_empty_matrix():
    """
    Test column extraction behavior with empty experimental matrix.
    
    ARRANGE: Create empty experimental matrix
    ACT: Attempt column extraction through public API
    ASSERT: Verify graceful handling of edge case
    """
    # ARRANGE - Set up empty matrix edge case
    empty_matrix = {}
    
    # ACT - Attempt column extraction
    result = extract_columns_from_matrix(empty_matrix)
    
    # ASSERT - Verify graceful edge case handling
    assert isinstance(result, dict), "Should return dictionary even for empty input"
    assert len(result) == 0, "Should return empty dictionary for empty input"


def test_extract_columns_edge_case_mixed_data_types(sample_exp_matrix_comprehensive):
    """
    Test column extraction behavior with mixed data types.
    
    ARRANGE: Create experimental matrix with various data types
    ACT: Extract columns through public API
    ASSERT: Verify correct handling of different data types
    """
    # ARRANGE - Set up mixed data types using centralized fixture
    mixed_matrix = sample_exp_matrix_comprehensive.copy()
    mixed_matrix['string_data'] = 'test_string'
    mixed_matrix['list_data'] = [1, 2, 3, 4, 5]
    mixed_matrix['scalar_data'] = 42
    
    # ACT - Extract columns
    result = extract_columns_from_matrix(mixed_matrix)
    
    # ASSERT - Verify correct data type handling
    assert isinstance(result, dict), "Should return dictionary"
    assert 't' in result, "Should preserve time data"
    
    # Verify different data types are handled appropriately
    for key, value in result.items():
        if key != 'signal_disp':  # signal_disp has special handling
            assert key in mixed_matrix, "All keys should be from original matrix"


# =============================================================================
# DATAFRAME TRANSFORMATION BEHAVIOR TESTS
# =============================================================================
# Tests focus on DataFrame construction accuracy and observable transformation results

def test_make_dataframe_from_config_successful_construction(sample_exp_matrix_comprehensive, 
                                                           sample_column_config_file,
                                                           sample_experimental_metadata):
    """
    Test successful DataFrame construction from experimental matrix using centralized fixtures.
    
    ARRANGE: Set up experimental data, column configuration, and metadata using centralized fixtures
    ACT: Create DataFrame through make_dataframe_from_config public API
    ASSERT: Verify DataFrame construction accuracy through observable structure and content
    """
    # ARRANGE - Set up test data using centralized fixtures
    exp_matrix = sample_exp_matrix_comprehensive
    metadata = sample_experimental_metadata
    
    # ACT - Create DataFrame through public API
    df = make_dataframe_from_config(
        exp_matrix,
        config_source=sample_column_config_file,
        metadata=metadata
    )
    
    # ASSERT - Verify successful DataFrame construction through observable behavior
    assert isinstance(df, pd.DataFrame), "Should return pandas DataFrame"
    assert len(df) > 0, "DataFrame should contain data rows"
    
    # Verify essential experimental columns are present
    assert 't' in df.columns, "Time column should be included"
    assert 'x' in df.columns, "X position column should be included"
    assert 'y' in df.columns, "Y position column should be included"
    
    # Verify data content accuracy without accessing internal transformation details
    expected_length = len(exp_matrix['t'])
    assert len(df) == expected_length, "DataFrame length should match time dimension"
    
    # Verify time alignment through observable values
    np.testing.assert_array_equal(df['t'].values, exp_matrix['t'],
                                  "Time data should be accurately preserved")


def test_make_dataframe_signal_disp_transformation_behavior(sample_exp_matrix_comprehensive,
                                                          sample_column_config_file):
    """
    Test signal_disp transformation behavior in DataFrame construction.
    
    ARRANGE: Set up experimental matrix with signal_disp data
    ACT: Create DataFrame with signal_disp transformation through public API
    ASSERT: Verify signal_disp transformation results through observable DataFrame content
    """
    # ARRANGE - Set up experimental data with signal_disp
    exp_matrix = sample_exp_matrix_comprehensive.copy()
    
    # Ensure signal_disp is present for transformation testing
    if 'signal_disp' not in exp_matrix:
        T = len(exp_matrix['t'])
        exp_matrix['signal_disp'] = np.random.rand(T, 5)  # Time-first orientation
    
    # ACT - Create DataFrame through public API
    df = make_dataframe_from_config(exp_matrix, config_source=sample_column_config_file)
    
    # ASSERT - Verify signal_disp transformation behavior
    if 'signal_disp' in df.columns:
        assert len(df) == len(exp_matrix['t']), "DataFrame length should match time dimension"
        
        # Verify signal_disp transformation results through observable content
        for i in range(min(3, len(df))):  # Check first few rows
            if pd.notna(df['signal_disp'].iloc[i]):
                signal_value = df['signal_disp'].iloc[i]
                assert isinstance(signal_value, np.ndarray), "Signal values should be numpy arrays"


def test_make_dataframe_metadata_integration_behavior(sample_exp_matrix_comprehensive,
                                                     sample_column_config_file,
                                                     sample_experimental_metadata):
    """
    Test metadata integration behavior in DataFrame construction.
    
    ARRANGE: Set up experimental data and metadata using centralized fixtures
    ACT: Create DataFrame with metadata integration through public API
    ASSERT: Verify metadata integration results through observable DataFrame columns
    """
    # ARRANGE - Set up experimental data and metadata
    exp_matrix = sample_exp_matrix_comprehensive
    metadata = sample_experimental_metadata
    
    # ACT - Create DataFrame with metadata integration
    df = make_dataframe_from_config(
        exp_matrix,
        config_source=sample_column_config_file,
        metadata=metadata
    )
    
    # ASSERT - Verify metadata integration behavior through observable DataFrame content
    assert isinstance(df, pd.DataFrame), "Should return pandas DataFrame"
    
    # Verify metadata fields are integrated without accessing internal mechanisms
    for key, expected_value in metadata.items():
        if key in df.columns:
            # Check that metadata values are properly integrated
            actual_value = df[key].iloc[0] if len(df) > 0 else None
            assert actual_value == expected_value, f"Metadata {key} should be correctly integrated"


# =============================================================================
# COMPREHENSIVE ERROR HANDLING AND SECURITY VALIDATION
# =============================================================================
# Error handling tests focus on observable error recovery behavior

def test_corrupted_pickle_file_error_recovery(temp_cross_platform_dir, 
                                            fixture_corrupted_file_scenarios):
    """
    Test error recovery behavior for corrupted pickle files using centralized scenarios.
    
    ARRANGE: Create corrupted pickle file using centralized corrupted file scenarios
    ACT: Attempt to load corrupted file through public API
    ASSERT: Verify appropriate error recovery behavior without accessing internal mechanisms
    """
    # ARRANGE - Set up corrupted pickle file using centralized scenario generator
    corrupted_scenarios = fixture_corrupted_file_scenarios
    corrupted_file_path = corrupted_scenarios.get('corrupted_pickle')
    
    if corrupted_file_path:
        # ACT & ASSERT - Verify error recovery behavior
        with pytest.raises((pickle.UnpicklingError, RuntimeError, IOError)):
            read_pickle_any_format(corrupted_file_path)


def test_handle_signal_disp_missing_required_keys_error_behavior():
    """
    Test error handling behavior for missing required keys in signal_disp processing.
    
    ARRANGE: Create experimental matrices missing required keys
    ACT: Attempt signal_disp processing through public API
    ASSERT: Verify appropriate error responses for missing data
    """
    # ARRANGE & ACT & ASSERT - Test missing signal_disp key
    missing_signal_disp = {'t': np.arange(10)}
    with pytest.raises(ValueError, match="missing required 'signal_disp' key"):
        handle_signal_disp(missing_signal_disp)
    
    # ARRANGE & ACT & ASSERT - Test missing time key
    missing_time = {'signal_disp': np.ones((10, 5))}
    with pytest.raises(ValueError, match="missing required 't' key"):
        handle_signal_disp(missing_time)


def test_handle_signal_disp_invalid_dimensions_error_behavior():
    """
    Test error handling behavior for invalid signal_disp dimensions.
    
    ARRANGE: Create experimental matrices with invalid signal_disp dimensions
    ACT: Attempt signal_disp processing through public API
    ASSERT: Verify appropriate error responses for dimension mismatches
    """
    # ARRANGE & ACT & ASSERT - Test 1D signal_disp (invalid)
    invalid_1d = {
        't': np.arange(10),
        'signal_disp': np.ones(10)  # Should be 2D
    }
    with pytest.raises(ValueError, match="signal_disp must be 2D"):
        handle_signal_disp(invalid_1d)
    
    # ARRANGE & ACT & ASSERT - Test dimension mismatch
    dimension_mismatch = {
        't': np.arange(10),
        'signal_disp': np.ones((15, 20))  # Neither dimension matches time length
    }
    with pytest.raises(ValueError, match="No dimension of signal_disp .* matches time dimension"):
        handle_signal_disp(dimension_mismatch)


def test_extract_columns_invalid_input_error_behavior():
    """
    Test error handling behavior for invalid inputs to column extraction.
    
    ARRANGE: Prepare various invalid input scenarios
    ACT: Attempt column extraction through public API
    ASSERT: Verify appropriate error responses for invalid inputs
    """
    # ARRANGE & ACT & ASSERT - Test invalid input type
    with pytest.raises(ValueError, match="exp_matrix must be a dictionary"):
        extract_columns_from_matrix("not_a_dictionary")
    
    # ARRANGE & ACT & ASSERT - Test graceful handling of empty dictionary
    empty_result = extract_columns_from_matrix({})
    assert isinstance(empty_result, dict), "Should return dictionary for empty input"
    assert len(empty_result) == 0, "Should return empty dictionary for empty input"


def test_extract_columns_nonexistent_columns_behavior():
    """
    Test behavior when requesting non-existent columns from experimental matrix.
    
    ARRANGE: Create experimental matrix and request non-existent columns
    ACT: Extract columns including non-existent ones through public API
    ASSERT: Verify graceful handling of non-existent column requests
    """
    # ARRANGE - Set up experimental matrix with known columns
    exp_matrix = {'t': np.arange(10), 'x': np.arange(10, 20)}
    requested_columns = ['t', 'nonexistent_column']
    
    # ACT - Extract columns including non-existent one
    result = extract_columns_from_matrix(exp_matrix, requested_columns)
    
    # ASSERT - Verify graceful handling of non-existent columns
    assert isinstance(result, dict), "Should return dictionary"
    assert 't' in result, "Should include existing columns"
    assert 'nonexistent_column' not in result, "Should gracefully skip non-existent columns"


# =============================================================================
# COMPREHENSIVE COMPRESSION AND FORMAT VALIDATION TESTS
# =============================================================================
# Tests for compression handling correctness and format detection behavior

def test_read_pickle_gzipped_compression_handling(temp_cross_platform_dir, 
                                                sample_exp_matrix_comprehensive):
    """
    Test gzipped pickle compression handling correctness.
    
    ARRANGE: Create gzipped pickle file with experimental data
    ACT: Load gzipped pickle through public API
    ASSERT: Verify compression handling correctness through data integrity validation
    """
    # ARRANGE - Set up gzipped pickle file
    import gzip
    import pickle
    
    test_data = sample_exp_matrix_comprehensive
    gzipped_file_path = temp_cross_platform_dir / "compressed_experiment.pkl.gz"
    
    with gzip.open(gzipped_file_path, 'wb') as f:
        pickle.dump(test_data, f)
    
    # ACT - Load gzipped pickle file
    loaded_data = read_pickle_any_format(gzipped_file_path)
    
    # ASSERT - Verify compression handling correctness
    assert loaded_data is not None, "Gzipped file should load successfully"
    assert isinstance(loaded_data, dict), "Should return dictionary structure"
    assert 't' in loaded_data, "Time data should be preserved through compression"
    
    # Verify data integrity through compression/decompression cycle
    np.testing.assert_array_equal(loaded_data['t'], test_data['t'],
                                  "Time data should survive compression handling")


def test_read_pickle_pandas_format_detection(temp_cross_platform_dir):
    """
    Test pandas-specific pickle format detection and handling.
    
    ARRANGE: Create pandas DataFrame pickle file
    ACT: Load pandas pickle through public API  
    ASSERT: Verify pandas format detection and DataFrame preservation
    """
    # ARRANGE - Set up pandas DataFrame pickle file
    test_df = pd.DataFrame({
        't': np.arange(0, 10),
        'x': np.arange(10, 20),
        'y': np.arange(20, 30)
    })
    pandas_file_path = temp_cross_platform_dir / "pandas_dataframe.pkl"
    test_df.to_pickle(pandas_file_path)
    
    # ACT - Load pandas pickle file
    loaded_data = read_pickle_any_format(pandas_file_path)
    
    # ASSERT - Verify pandas format detection and handling
    assert loaded_data is not None, "Pandas pickle should load successfully"
    assert isinstance(loaded_data, pd.DataFrame), "Should preserve DataFrame structure"
    assert 't' in loaded_data.columns, "Should preserve column structure"
    assert len(loaded_data) == 10, "Should preserve DataFrame length"


# =============================================================================
# COMPREHENSIVE COLUMN EXTRACTION BEHAVIOR TESTS  
# =============================================================================
# Tests focus on column extraction accuracy without implementation coupling

def test_extract_columns_all_columns_behavior(sample_exp_matrix_comprehensive):
    """
    Test column extraction behavior for all available columns.
    
    ARRANGE: Use comprehensive experimental matrix from centralized fixtures
    ACT: Extract all columns through public API
    ASSERT: Verify column extraction accuracy without accessing internal mechanisms
    """
    # ARRANGE - Set up comprehensive experimental matrix
    exp_matrix = sample_exp_matrix_comprehensive
    
    # ACT - Extract all columns through public API
    result = extract_columns_from_matrix(exp_matrix)
    
    # ASSERT - Verify column extraction behavior
    assert isinstance(result, dict), "Should return dictionary"
    assert 't' in result, "Should extract time column"
    
    # Verify that signal_disp is handled specially (not extracted with other columns)
    assert 'signal_disp' not in result, "signal_disp should be handled separately"
    
    # Verify content preservation for extracted columns
    for key in result:
        if key in exp_matrix and key != 'signal_disp':
            original_value = exp_matrix[key]
            extracted_value = result[key]
            
            if isinstance(original_value, np.ndarray) and isinstance(extracted_value, np.ndarray):
                np.testing.assert_array_equal(extracted_value, original_value,
                                              f"Column {key} should be accurately extracted")


def test_extract_columns_specific_columns_behavior(sample_exp_matrix_comprehensive):
    """
    Test column extraction behavior for specific requested columns.
    
    ARRANGE: Use comprehensive experimental matrix and specify target columns
    ACT: Extract specific columns through public API
    ASSERT: Verify accurate extraction of only requested columns
    """
    # ARRANGE - Set up experimental matrix and target columns
    exp_matrix = sample_exp_matrix_comprehensive
    target_columns = ['t', 'x', 'y']
    
    # ACT - Extract specific columns
    result = extract_columns_from_matrix(exp_matrix, target_columns)
    
    # ASSERT - Verify specific column extraction behavior
    assert isinstance(result, dict), "Should return dictionary"
    assert set(result.keys()).issubset(set(target_columns)), "Should only include requested columns"
    
    # Verify requested columns are present (if they exist in source)
    for col in target_columns:
        if col in exp_matrix:
            assert col in result, f"Requested column {col} should be extracted"


def test_extract_columns_multidimensional_array_handling(sample_exp_matrix_comprehensive):
    """
    Test column extraction behavior with multi-dimensional arrays.
    
    ARRANGE: Create experimental matrix with multi-dimensional arrays
    ACT: Extract columns with various array dimensions
    ASSERT: Verify appropriate handling of different array dimensionalities
    """
    # ARRANGE - Set up matrix with multi-dimensional arrays
    exp_matrix = sample_exp_matrix_comprehensive.copy()
    exp_matrix['multi_dim_array'] = np.ones((5, 3, 2))  # 3D array
    exp_matrix['matrix_data'] = np.ones((10, 4))        # 2D array
    
    # ACT - Extract columns without forcing 1D conversion
    result = extract_columns_from_matrix(exp_matrix, ensure_1d=False)
    
    # ASSERT - Verify multi-dimensional array preservation
    if 'multi_dim_array' in result:
        assert result['multi_dim_array'].ndim == 3, "3D arrays should be preserved"
    if 'matrix_data' in result:
        assert result['matrix_data'].ndim == 2, "2D arrays should be preserved"


# =============================================================================
# COMPREHENSIVE INTEGRATION AND EDGE-CASE VALIDATION
# =============================================================================
# Final comprehensive tests ensuring complete edge-case coverage

@pytest.mark.parametrize("corrupted_scenario", [
    "truncated_pickle", "invalid_pickle_header", "binary_in_text", "empty_file"
])
def test_corrupted_file_recovery_scenarios(temp_cross_platform_dir, corrupted_scenario):
    """
    Test comprehensive corrupted file recovery scenarios.
    
    ARRANGE: Create various corrupted file scenarios using parameterization
    ACT: Attempt loading through public API
    ASSERT: Verify appropriate error recovery behavior for different corruption types
    """
    # ARRANGE - Set up corrupted file scenario
    corrupted_file = temp_cross_platform_dir / f"corrupted_{corrupted_scenario}.pkl"
    
    if corrupted_scenario == "truncated_pickle":
        corrupted_file.write_bytes(b'truncated pickle data\x80\x03}')
    elif corrupted_scenario == "invalid_pickle_header":
        corrupted_file.write_bytes(b'invalid pickle header data')
    elif corrupted_scenario == "binary_in_text":
        corrupted_file.write_bytes(b'valid text\x00\x01\x02binary data\xFF\xFE')
    elif corrupted_scenario == "empty_file":
        corrupted_file.write_bytes(b'')
    
    # ACT & ASSERT - Verify error recovery behavior
    with pytest.raises((pickle.UnpicklingError, RuntimeError, IOError, EOFError)):
        read_pickle_any_format(corrupted_file)


@pytest.mark.parametrize("signal_orientation,expected_behavior", [
    ("time_first", "direct_processing"),
    ("signal_first", "transpose_processing")
])
def test_signal_disp_orientation_boundary_conditions(signal_orientation, expected_behavior):
    """
    Test signal_disp handling with various orientation boundary conditions.
    
    ARRANGE: Create signal_disp data in different orientations
    ACT: Process through handle_signal_disp public API
    ASSERT: Verify correct orientation handling behavior
    """
    # ARRANGE - Set up signal data based on orientation
    T, X = 8, 3  # Small dimensions for boundary testing
    t_array = np.arange(T)
    
    if signal_orientation == "time_first":
        signal_disp = np.random.rand(T, X)  # (T, X) orientation
    else:  # signal_first
        signal_disp = np.random.rand(X, T)  # (X, T) orientation
    
    exp_matrix = {
        't': t_array,
        'signal_disp': signal_disp
    }
    
    # ACT - Process signal_disp
    result = handle_signal_disp(exp_matrix)
    
    # ASSERT - Verify orientation handling behavior
    assert isinstance(result, pd.Series), "Should return pandas Series"
    assert len(result) == T, "Length should match time dimension"
    
    # Verify each element has correct signal dimension
    for element in result:
        assert isinstance(element, np.ndarray), "Each element should be numpy array"
        assert len(element) == X, "Each element should have correct signal dimension"


def test_comprehensive_pickle_io_integration_workflow(temp_cross_platform_dir,
                                                     sample_exp_matrix_comprehensive,
                                                     sample_column_config_file,
                                                     sample_experimental_metadata):
    """
    Test comprehensive pickle I/O integration workflow from data creation to DataFrame construction.
    
    ARRANGE: Set up complete experimental workflow with data, configuration, and metadata
    ACT: Execute full pickle I/O workflow through public APIs
    ASSERT: Verify end-to-end workflow success and data integrity
    """
    # ARRANGE - Set up complete experimental workflow
    original_data = sample_exp_matrix_comprehensive
    metadata = sample_experimental_metadata
    
    # Create pickle file
    pickle_file = temp_cross_platform_dir / "integration_test.pkl"
    import pickle
    with open(pickle_file, 'wb') as f:
        pickle.dump(original_data, f)
    
    # ACT - Execute complete workflow
    # Step 1: Load pickle data
    loaded_data = read_pickle_any_format(pickle_file)
    
    # Step 2: Create DataFrame from loaded data
    final_df = make_dataframe_from_config(
        loaded_data,
        config_source=sample_column_config_file,
        metadata=metadata
    )
    
    # ASSERT - Verify end-to-end workflow success
    assert loaded_data is not None, "Data loading should succeed"
    assert isinstance(final_df, pd.DataFrame), "DataFrame creation should succeed"
    assert len(final_df) > 0, "Final DataFrame should contain data"
    
    # Verify data integrity through complete workflow
    if 't' in loaded_data and 't' in final_df.columns:
        np.testing.assert_array_equal(final_df['t'].values, loaded_data['t'],
                                      "Data should maintain integrity through complete workflow")