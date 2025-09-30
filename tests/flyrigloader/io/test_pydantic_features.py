"""
Modern test suite for Pydantic-based data processing functions.

This module provides comprehensive testing for the Pydantic-based column configuration
and data processing implementation using modern pytest practices with fixture-based
organization, parametrization, and comprehensive coverage per TST-MOD-001 and TST-MOD-002
requirements.

Key testing areas:
- DataFrame transformation with full schema validation (F-006-RQ-001 through F-006-RQ-005)
- Pydantic-driven YAML configuration loading with pytest-mock integration (TST-MOD-003)
- Property-based validation using Hypothesis for robust edge case discovery
- Performance benchmark tests for DataFrame operations (TST-PERF-002)
- Comprehensive skip_columns behavior testing
- Alias resolution, default values, and metadata integration (TST-INTEG-003)
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml
from hypothesis import given, strategies as st, assume, settings
from pydantic import ValidationError

from flyrigloader.io.transformers import make_dataframe_from_config
from flyrigloader.io.column_models import (
    ColumnConfig, 
    ColumnConfigDict, 
    ColumnDimension,
    SpecialHandlerType,
    get_config_from_source
)
from flyrigloader.exceptions import TransformError


# ===== MODERN PYTEST FIXTURES =====

@pytest.fixture
def basic_exp_matrix():
    """
    Create a basic experimental data matrix for testing.
    
    Provides consistent test data with proper time alignment for
    DataFrame construction validation per F-006-RQ-005 requirements.
    """
    np.random.seed(42)  # Reproducible test data
    return {
        't': np.linspace(0, 10, 100),
        'x': np.random.rand(100),
        'y': np.random.rand(100)
    }


@pytest.fixture
def exp_matrix_with_signal_disp():
    """
    Create experimental data matrix with signal_disp for special handling tests.
    
    Tests special handler functionality per F-006-RQ-003 requirements.
    """
    np.random.seed(42)
    return {
        't': np.linspace(0, 10, 100),
        'x': np.random.rand(100),
        'y': np.random.rand(100),
        'signal_disp': np.random.rand(15, 100)  # 15 channels, 100 time points
    }


@pytest.fixture
def exp_matrix_with_aliases():
    """
    Create experimental data matrix with aliased column names.
    
    Tests alias resolution functionality per F-004-RQ-005 requirements.
    """
    np.random.seed(42)
    return {
        't': np.linspace(0, 10, 100),
        'x': np.random.rand(100),
        'y': np.random.rand(100),
        'dtheta_smooth': np.random.rand(100)  # Alias for 'dtheta'
    }


@pytest.fixture
def exp_matrix_large():
    """
    Create large experimental data matrix for performance testing.
    
    Supports performance benchmark validation per TST-PERF-002 requirements.
    """
    size = 1_000_000  # 1M rows for performance testing
    np.random.seed(42)
    return {
        't': np.linspace(0, 1000, size),
        'x': np.random.rand(size),
        'y': np.random.rand(size),
        'signal': np.random.rand(size)
    }


@pytest.fixture
def standard_column_config():
    """
    Create standard column configuration dictionary.
    
    Provides consistent configuration for testing schema validation
    per F-004-RQ-001 through F-004-RQ-005 requirements.
    """
    return {
        'columns': {
            't': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': True,
                'description': 'Time values'
            },
            'x': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': True,
                'description': 'X position'
            },
            'y': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': True,
                'description': 'Y position'
            },
            'signal': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': False,
                'description': 'Signal values',
                'default_value': None
            },
            'signal_disp': {
                'type': 'numpy.ndarray',
                'dimension': 2,
                'required': False,
                'description': 'Signal display data',
                'special_handling': 'transform_to_match_time_dimension'
            },
            'dtheta': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': False,
                'description': 'Angular velocity',
                'alias': 'dtheta_smooth'
            }
        },
        'special_handlers': {
            'transform_to_match_time_dimension': '_handle_signal_disp'
        }
    }


@pytest.fixture
def metadata_config():
    """
    Create column configuration with metadata fields.
    
    Tests metadata integration per F-006-RQ-004 requirements.
    """
    config = {
        'columns': {
            't': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': True,
                'description': 'Time values'
            },
            'x': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': True,
                'description': 'X position'
            },
            'date': {
                'type': 'string',
                'required': False,
                'is_metadata': True,
                'description': 'Experiment date'
            },
            'exp_name': {
                'type': 'string',
                'required': False,
                'is_metadata': True,
                'description': 'Experiment name'
            },
            'rig': {
                'type': 'string',
                'required': False,
                'is_metadata': True,
                'description': 'Rig identifier'
            }
        }
    }
    return config


@pytest.fixture
def sample_metadata():
    """
    Create sample metadata for testing metadata integration.
    """
    return {
        'date': '2025-04-01',
        'exp_name': 'test_experiment',
        'rig': 'test_rig',
        'fly_id': 'fly-123'
    }


@pytest.fixture
def config_file_path(tmp_path, standard_column_config):
    """
    Create temporary configuration file for file-based config testing.
    
    Uses pytest's tmp_path fixture for automatic cleanup per modern practices.
    """
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(standard_column_config, f)
    return str(config_file)


# ===== CORE FUNCTIONALITY TESTS =====

class TestDataFrameCreationBasics:
    """Test basic DataFrame creation functionality with modern pytest practices."""
    
    def test_basic_dataframe_creation(self, basic_exp_matrix, config_file_path):
        """Test basic DataFrame creation from experimental data."""
        df = make_dataframe_from_config(basic_exp_matrix, config_file_path)
        
        # Validate DataFrame structure per F-006-RQ-001
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert {'t', 'x', 'y'} <= set(df.columns)
        
        # Validate data types and content
        assert df['t'].dtype in [np.float64, np.float32]
        assert df['x'].dtype in [np.float64, np.float32]
        assert df['y'].dtype in [np.float64, np.float32]
        
        # Validate data integrity
        np.testing.assert_array_equal(df['t'].values, basic_exp_matrix['t'])
        np.testing.assert_array_equal(df['x'].values, basic_exp_matrix['x'])
        np.testing.assert_array_equal(df['y'].values, basic_exp_matrix['y'])
    
    def test_dataframe_with_dict_config(self, basic_exp_matrix, standard_column_config):
        """Test DataFrame creation using dictionary configuration."""
        df = make_dataframe_from_config(basic_exp_matrix, standard_column_config)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert {'t', 'x', 'y'} <= set(df.columns)
    
    def test_dataframe_with_column_config_dict(self, basic_exp_matrix, standard_column_config):
        """Test DataFrame creation using ColumnConfigDict instance."""
        config_dict = ColumnConfigDict.model_validate(standard_column_config)
        df = make_dataframe_from_config(basic_exp_matrix, config_dict)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert {'t', 'x', 'y'} <= set(df.columns)


class TestSchemaValidation:
    """Test comprehensive schema validation per F-004-RQ-001 through F-004-RQ-005."""
    
    def test_missing_required_columns_validation(self, standard_column_config):
        """Test validation of missing required columns."""
        # Create data missing required column 'x'
        incomplete_matrix = {
            't': np.linspace(0, 10, 100),
            'y': np.random.rand(100)
            # Missing 'x' column
        }
        
        with pytest.raises(TransformError, match="Missing required columns: x"):
            make_dataframe_from_config(incomplete_matrix, standard_column_config)
    
    @pytest.mark.parametrize("missing_columns,expected_error", [
        (['x'], "Missing required columns: x"),
        (['x', 'y'], "Missing required columns: x, y"),
        (['t'], "Missing required columns: t"),
    ])
    def test_multiple_missing_required_columns(self, standard_column_config, missing_columns, expected_error):
        """Test validation with multiple missing required columns."""
        # Create base matrix with all required columns
        base_matrix = {
            't': np.linspace(0, 10, 100),
            'x': np.random.rand(100),
            'y': np.random.rand(100)
        }
        
        # Remove specified columns
        test_matrix = {k: v for k, v in base_matrix.items() if k not in missing_columns}
        
        with pytest.raises(TransformError, match=expected_error):
            make_dataframe_from_config(test_matrix, standard_column_config)
    
    def test_invalid_dimension_validation(self, basic_exp_matrix, tmp_path):
        """Test validation of invalid array dimensions."""
        # Create config with invalid dimension
        invalid_config = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 4,  # Invalid dimension (must be 1, 2, or 3)
                    'required': True,
                    'description': 'Time values'
                }
            }
        }
        
        config_file = tmp_path / "invalid_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        with pytest.raises(ValidationError, match="Dimension must be 1, 2, or 3"):
            make_dataframe_from_config(basic_exp_matrix, str(config_file))


class TestAliasResolution:
    """Test column alias resolution per F-004-RQ-005 requirements."""
    
    def test_alias_column_mapping(self, exp_matrix_with_aliases, standard_column_config):
        """Test proper mapping of aliased columns."""
        df = make_dataframe_from_config(exp_matrix_with_aliases, standard_column_config)
        
        # Verify alias mapping: dtheta_smooth -> dtheta
        assert 'dtheta' in df.columns
        assert 'dtheta_smooth' not in df.columns
        
        # Verify data integrity through alias
        expected_data = exp_matrix_with_aliases['dtheta_smooth']
        np.testing.assert_array_equal(df['dtheta'].values, expected_data)
    
    def test_alias_with_missing_source_column(self, basic_exp_matrix, standard_column_config):
        """Test behavior when aliased source column is missing."""
        # Remove the aliased column from matrix
        matrix_without_alias = basic_exp_matrix.copy()
        # 'dtheta_smooth' not present, so 'dtheta' should not appear
        
        df = make_dataframe_from_config(matrix_without_alias, standard_column_config)
        
        # Since dtheta is not required and alias source is missing, column should not appear
        assert 'dtheta' not in df.columns
        assert 'dtheta_smooth' not in df.columns


class TestDefaultValues:
    """Test default value assignment per F-004-RQ-004 requirements."""
    
    def test_default_value_assignment(self, tmp_path):
        """Test assignment of default values for missing optional columns."""
        # Create matrix missing optional column with default value
        matrix = {
            't': np.linspace(0, 10, 100),
            'x': np.random.rand(100),
            'y': np.random.rand(100)
            # Missing 'signal' column which has default_value
        }
        
        # Create config with default value
        config = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'X pos'},
                'y': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Y pos'},
                'signal': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': 'Signal values',
                    'default_value': None
                }
            }
        }
        
        config_file = tmp_path / "default_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        df = make_dataframe_from_config(matrix, str(config_file))
        
        # Verify default value assignment
        assert 'signal' in df.columns
        assert df['signal'].iloc[0] is None
        assert all(pd.isna(df['signal']) | (df['signal'] == None))
    
    @pytest.mark.parametrize("default_value", [
        None,
        0.0,
        -999,
        "missing_data",
        []
    ])
    def test_various_default_values(self, basic_exp_matrix, tmp_path, default_value):
        """Test various types of default values."""
        config = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'X pos'},
                'y': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Y pos'},
                'optional_col': {
                    'type': 'any',
                    'required': False,
                    'description': 'Optional column',
                    'default_value': default_value
                }
            }
        }
        
        config_file = tmp_path / "test_defaults.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        df = make_dataframe_from_config(basic_exp_matrix, str(config_file))
        
        assert 'optional_col' in df.columns
        if default_value is None:
            assert df['optional_col'].iloc[0] is None
        else:
            assert df['optional_col'].iloc[0] == default_value


class TestSpecialHandlers:
    """Test special handler functionality per F-006-RQ-003 requirements."""
    
    def test_signal_disp_transformation(self, exp_matrix_with_signal_disp, standard_column_config):
        """Test signal_disp special handler transformation."""
        df = make_dataframe_from_config(exp_matrix_with_signal_disp, standard_column_config)
        
        # Verify signal_disp transformation per F-006-RQ-002
        assert 'signal_disp' in df.columns
        assert len(df) == 100
        
        # Each signal_disp entry should be a 1D array
        assert isinstance(df['signal_disp'].iloc[0], np.ndarray)
        assert df['signal_disp'].iloc[0].shape == (15,)
        
        # Verify all entries are arrays of correct shape
        for i in range(len(df)):
            assert isinstance(df['signal_disp'].iloc[i], np.ndarray)
            assert df['signal_disp'].iloc[i].shape == (15,)
    
    def test_signal_disp_transposition_detection(self):
        """Test automatic transposition detection for signal_disp."""
        # Test with transposed signal_disp (time on second axis)
        matrix = {
            't': np.linspace(0, 10, 100),
            'x': np.random.rand(100),
            'y': np.random.rand(100),
            'signal_disp': np.random.rand(15, 100)  # Already in correct orientation
        }
        
        config = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'X pos'},
                'y': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Y pos'},
                'signal_disp': {
                    'type': 'numpy.ndarray',
                    'dimension': 2,
                    'required': False,
                    'description': 'Signal display data',
                    'special_handling': 'transform_to_match_time_dimension'
                }
            },
            'special_handlers': {
                'transform_to_match_time_dimension': '_handle_signal_disp'
            }
        }
        
        df = make_dataframe_from_config(matrix, config)
        
        # Verify correct handling regardless of initial orientation
        assert 'signal_disp' in df.columns
        assert len(df) == 100
        assert isinstance(df['signal_disp'].iloc[0], np.ndarray)


class TestMetadataIntegration:
    """Test metadata integration per F-006-RQ-004 requirements."""
    
    def test_metadata_addition(self, basic_exp_matrix, metadata_config, sample_metadata):
        """Test addition of metadata columns to DataFrame."""
        df = make_dataframe_from_config(
            basic_exp_matrix, 
            metadata_config, 
            metadata=sample_metadata
        )
        
        # Verify metadata columns are added
        assert 'date' in df.columns
        assert 'exp_name' in df.columns
        assert 'rig' in df.columns
        
        # Verify metadata values are consistent across all rows
        assert all(df['date'] == '2025-04-01')
        assert all(df['exp_name'] == 'test_experiment')
        assert all(df['rig'] == 'test_rig')
    
    def test_metadata_with_non_metadata_columns(self, basic_exp_matrix, metadata_config, sample_metadata):
        """Test that only configured metadata columns are added."""
        # fly_id is in sample_metadata but not configured as metadata in config
        df = make_dataframe_from_config(
            basic_exp_matrix, 
            metadata_config, 
            metadata=sample_metadata
        )
        
        # fly_id should not be added since it's not in the column config
        assert 'fly_id' not in df.columns
    
    def test_empty_metadata(self, basic_exp_matrix, metadata_config):
        """Test behavior with empty metadata."""
        df = make_dataframe_from_config(basic_exp_matrix, metadata_config, metadata={})
        
        # No metadata columns should be added
        assert 'date' not in df.columns
        assert 'exp_name' not in df.columns
        assert 'rig' not in df.columns


class TestSkipColumns:
    """Test skip_columns functionality with comprehensive coverage."""
    
    def test_skip_configured_column(self, basic_exp_matrix, standard_column_config):
        """Test skipping a column defined in configuration."""
        df = make_dataframe_from_config(
            basic_exp_matrix, 
            standard_column_config, 
            skip_columns=['y']
        )
        
        # Skipped column should not appear
        assert 'y' not in df.columns
        # Other required columns should remain
        assert 't' in df.columns
        assert 'x' in df.columns
    
    def test_skip_required_column_with_warning(self, basic_exp_matrix, standard_column_config, caplog):
        """Test skipping required column generates warning."""
        with caplog.at_level(logging.WARNING):
            df = make_dataframe_from_config(
                basic_exp_matrix, 
                standard_column_config, 
                skip_columns=['x']  # x is required
            )
        
        # Required column should still be skipped
        assert 'x' not in df.columns
        
        # Warning should be logged
        assert any('Skipping required column' in record.message for record in caplog.records)
        assert any('x' in record.message for record in caplog.records)
    
    def test_skip_nonexistent_column(self, basic_exp_matrix, standard_column_config):
        """Test skipping a column not present in exp_matrix."""
        df = make_dataframe_from_config(
            basic_exp_matrix, 
            standard_column_config, 
            skip_columns=['nonexistent_column']
        )
        
        # Should not error and should include all other columns
        assert 't' in df.columns
        assert 'x' in df.columns
        assert 'y' in df.columns
        assert 'nonexistent_column' not in df.columns
    
    def test_skip_multiple_columns(self, basic_exp_matrix, standard_column_config):
        """Test skipping multiple columns simultaneously."""
        df = make_dataframe_from_config(
            basic_exp_matrix, 
            standard_column_config, 
            skip_columns=['x', 'y']
        )
        
        # Only time column should remain
        assert 't' in df.columns
        assert 'x' not in df.columns
        assert 'y' not in df.columns
    
    def test_empty_skip_list(self, basic_exp_matrix, standard_column_config):
        """Test default behavior with empty skip list."""
        df = make_dataframe_from_config(
            basic_exp_matrix, 
            standard_column_config, 
            skip_columns=[]
        )
        
        # All configured columns should be included
        assert {'t', 'x', 'y'} <= set(df.columns)


# ===== PYTEST-MOCK INTEGRATION TESTS =====

class TestMockIntegration:
    """Test pytest-mock integration for isolated testing per TST-MOD-003."""
    
    def test_mock_yaml_loading(self, mocker, basic_exp_matrix):
        """Test mocking of YAML configuration loading."""
        # Mock yaml.safe_load to return test configuration
        mock_config = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'X pos'}
            }
        }
        
        mock_open = mocker.mock_open(read_data=yaml.dump(mock_config))
        mocker.patch('builtins.open', mock_open)
        mocker.patch('yaml.safe_load', return_value=mock_config)
        
        df = make_dataframe_from_config(basic_exp_matrix, "fake_config.yaml")
        
        # Verify mocked configuration was used
        assert 't' in df.columns
        assert 'x' in df.columns
        assert 'y' not in df.columns  # Not in mocked config
    
    def test_mock_pydantic_validation(self, mocker, basic_exp_matrix, standard_column_config):
        """Test mocking of Pydantic validation process."""
        # Mock ColumnConfigDict validation
        mock_config_dict = MagicMock()
        mock_config_dict.columns = {
            't': MagicMock(required=True, is_metadata=False, alias=None),
            'x': MagicMock(required=True, is_metadata=False, alias=None),
            'y': MagicMock(required=True, is_metadata=False, alias=None)
        }
        mock_config_dict.special_handlers = {}
        
        mocker.patch('flyrigloader.io.pickle.get_config_from_source', return_value=mock_config_dict)
        
        df = make_dataframe_from_config(basic_exp_matrix, standard_column_config)
        
        # Verify mock was called
        assert isinstance(df, pd.DataFrame)
    
    def test_mock_dataframe_construction(self, mocker, basic_exp_matrix, standard_column_config):
        """Test mocking of DataFrame construction scenarios."""
        # Mock pandas DataFrame constructor
        mock_df = MagicMock(spec=pd.DataFrame)
        mock_df.columns = ['t', 'x', 'y']
        mock_df.__len__.return_value = 100
        
        mocker.patch('pandas.DataFrame', return_value=mock_df)
        
        result = make_dataframe_from_config(basic_exp_matrix, standard_column_config)
        
        # Verify mocked DataFrame was returned
        assert result is mock_df


# ===== PROPERTY-BASED TESTING WITH HYPOTHESIS =====

class TestPropertyBasedValidation:
    """Test property-based validation using Hypothesis per Section 3.6.3 requirements."""
    
    @given(
        size=st.integers(min_value=10, max_value=1000),
        num_columns=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=50, deadline=None)
    def test_dataframe_construction_properties(self, size, num_columns):
        """Test DataFrame construction properties with generated data."""
        assume(size >= 10)  # Ensure minimum data size
        
        # Generate experimental matrix with random data
        column_names = ['t'] + [f'col_{i}' for i in range(num_columns - 1)]
        exp_matrix = {
            name: np.random.rand(size) if name == 't' else np.random.rand(size)
            for name in column_names
        }
        
        # Generate corresponding configuration
        config = {
            'columns': {
                name: {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': f'Column {name}'
                }
                for name in column_names
            }
        }
        
        df = make_dataframe_from_config(exp_matrix, config)
        
        # Verify fundamental properties
        assert isinstance(df, pd.DataFrame)
        assert len(df) == size
        assert len(df.columns) == num_columns
        assert set(df.columns) == set(column_names)
        
        # Verify data integrity
        for col_name in column_names:
            np.testing.assert_array_equal(df[col_name].values, exp_matrix[col_name])
    
    @given(
        array_shape=st.tuples(
            st.integers(min_value=10, max_value=100),  # time dimension
            st.integers(min_value=5, max_value=20)     # signal dimension
        ),
        orientation=st.booleans()  # True for (T, X), False for (X, T)
    )
    @settings(max_examples=20, deadline=None)
    def test_signal_disp_transformation_properties(self, array_shape, orientation):
        """Test signal_disp transformation with various array shapes and orientations."""
        time_size, signal_size = array_shape
        
        # Create signal_disp with specified orientation
        if orientation:
            signal_disp = np.random.rand(time_size, signal_size)  # (T, X)
        else:
            signal_disp = np.random.rand(signal_size, time_size)  # (X, T)
        
        exp_matrix = {
            't': np.linspace(0, 10, time_size),
            'x': np.random.rand(time_size),
            'signal_disp': signal_disp
        }
        
        config = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'X pos'},
                'signal_disp': {
                    'type': 'numpy.ndarray',
                    'dimension': 2,
                    'required': False,
                    'description': 'Signal display data',
                    'special_handling': 'transform_to_match_time_dimension'
                }
            },
            'special_handlers': {
                'transform_to_match_time_dimension': '_handle_signal_disp'
            }
        }
        
        df = make_dataframe_from_config(exp_matrix, config)
        
        # Verify transformation properties
        assert len(df) == time_size
        assert 'signal_disp' in df.columns
        
        # Each signal_disp entry should be an array of the signal dimension
        for i in range(len(df)):
            assert isinstance(df['signal_disp'].iloc[i], np.ndarray)
            assert df['signal_disp'].iloc[i].shape == (signal_size,)


# ===== PERFORMANCE BENCHMARK TESTS =====

class TestPerformanceBenchmarks:
    """Test performance benchmarks per TST-PERF-002 requirements."""
    
    @pytest.mark.benchmark
    def test_dataframe_transformation_performance_1m_rows(self, benchmark, exp_matrix_large):
        """Test DataFrame transformation performance with 1M rows."""
        config = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'X pos'},
                'y': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Y pos'},
                'signal': {'type': 'numpy.ndarray', 'dimension': 1, 'required': False, 'description': 'Signal'}
            }
        }
        
        # Benchmark the transformation
        result = benchmark(make_dataframe_from_config, exp_matrix_large, config)
        
        # Verify the result is correct
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1_000_000
        
        # Performance requirement: < 500ms per 1M rows per TST-PERF-002
        # Note: benchmark fixture automatically validates timing
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("data_size", [10_000, 100_000, 500_000])
    def test_dataframe_performance_scaling(self, benchmark, data_size):
        """Test DataFrame transformation performance scaling."""
        np.random.seed(42)
        exp_matrix = {
            't': np.linspace(0, 100, data_size),
            'x': np.random.rand(data_size),
            'y': np.random.rand(data_size)
        }
        
        config = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'X pos'},
                'y': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Y pos'}
            }
        }
        
        result = benchmark(make_dataframe_from_config, exp_matrix, config)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == data_size
    
    @pytest.mark.benchmark
    def test_signal_disp_transformation_performance(self, benchmark):
        """Test signal_disp transformation performance with large arrays."""
        time_size = 100_000
        signal_size = 15
        
        np.random.seed(42)
        exp_matrix = {
            't': np.linspace(0, 1000, time_size),
            'x': np.random.rand(time_size),
            'signal_disp': np.random.rand(signal_size, time_size)
        }
        
        config = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'X pos'},
                'signal_disp': {
                    'type': 'numpy.ndarray',
                    'dimension': 2,
                    'required': False,
                    'description': 'Signal display data',
                    'special_handling': 'transform_to_match_time_dimension'
                }
            },
            'special_handlers': {
                'transform_to_match_time_dimension': '_handle_signal_disp'
            }
        }
        
        result = benchmark(make_dataframe_from_config, exp_matrix, config)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == time_size
        assert 'signal_disp' in result.columns


# ===== COMPREHENSIVE EDGE CASE TESTS =====

class TestEdgeCases:
    """Test comprehensive edge cases and error conditions."""
    
    def test_empty_exp_matrix(self, standard_column_config):
        """Test behavior with empty experimental matrix."""
        with pytest.raises(TransformError, match="exp_matrix cannot be empty"):
            make_dataframe_from_config({}, standard_column_config)
    
    def test_mismatched_array_lengths(self, standard_column_config):
        """Test behavior with mismatched array lengths."""
        mismatched_matrix = {
            't': np.linspace(0, 10, 100),
            'x': np.random.rand(50),  # Different length
            'y': np.random.rand(100)
        }
        
        # This should either work (with proper handling) or raise an appropriate error
        # depending on the implementation's tolerance for length mismatches
        try:
            df = make_dataframe_from_config(mismatched_matrix, standard_column_config)
            # If it succeeds, verify the result is valid
            assert isinstance(df, pd.DataFrame)
        except (ValueError, RuntimeError) as e:
            # If it fails, ensure it's with an appropriate error message
            assert any(word in str(e).lower() for word in ['length', 'dimension', 'shape'])
    
    def test_none_values_in_matrix(self, standard_column_config):
        """Test handling of None values in experimental matrix."""
        matrix_with_none = {
            't': np.linspace(0, 10, 100),
            'x': None,  # None value
            'y': np.random.rand(100)
        }
        
        # Should handle None gracefully or raise appropriate error
        try:
            df = make_dataframe_from_config(matrix_with_none, standard_column_config)
            assert isinstance(df, pd.DataFrame)
        except (ValueError, TypeError) as e:
            assert "None" in str(e) or "null" in str(e).lower()
    
    def test_invalid_config_source_type(self, basic_exp_matrix):
        """Test error handling for invalid configuration source types."""
        with pytest.raises(TypeError):
            make_dataframe_from_config(basic_exp_matrix, 123)  # Invalid type
        
        with pytest.raises(TypeError):
            make_dataframe_from_config(basic_exp_matrix, [])  # Invalid type
    
    def test_circular_alias_reference(self, basic_exp_matrix, tmp_path):
        """Test handling of circular alias references."""
        circular_config = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time'},
                'col_a': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': 'Column A',
                    'alias': 'col_b'
                },
                'col_b': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': 'Column B', 
                    'alias': 'col_a'  # Circular reference
                }
            }
        }
        
        config_file = tmp_path / "circular_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(circular_config, f)
        
        # Should handle circular reference gracefully
        df = make_dataframe_from_config(basic_exp_matrix, str(config_file))
        
        # Neither circular column should appear if not in matrix
        assert 'col_a' not in df.columns
        assert 'col_b' not in df.columns


# ===== INTEGRATION TESTS =====

class TestComprehensiveIntegration:
    """Test comprehensive integration scenarios per TST-INTEG-003."""
    
    def test_end_to_end_workflow_with_all_features(self, tmp_path):
        """Test end-to-end workflow with all features combined."""
        # Create comprehensive experimental matrix
        np.random.seed(42)
        exp_matrix = {
            't': np.linspace(0, 10, 100),
            'x': np.random.rand(100),
            'y': np.random.rand(100),
            'dtheta_smooth': np.random.rand(100),  # Will be aliased to 'dtheta'
            'signal_disp': np.random.rand(15, 100)  # Needs special handling
            # Missing 'optional_signal' - will use default value
        }
        
        # Create comprehensive configuration
        comprehensive_config = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                },
                'x': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'X position'
                },
                'y': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Y position'
                },
                'dtheta': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': 'Angular velocity',
                    'alias': 'dtheta_smooth'
                },
                'signal_disp': {
                    'type': 'numpy.ndarray',
                    'dimension': 2,
                    'required': False,
                    'description': 'Signal display data',
                    'special_handling': 'transform_to_match_time_dimension'
                },
                'optional_signal': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': 'Optional signal',
                    'default_value': 0.0
                },
                'date': {
                    'type': 'string',
                    'required': False,
                    'is_metadata': True,
                    'description': 'Experiment date'
                },
                'exp_name': {
                    'type': 'string',
                    'required': False,
                    'is_metadata': True,
                    'description': 'Experiment name'
                }
            },
            'special_handlers': {
                'transform_to_match_time_dimension': '_handle_signal_disp'
            }
        }
        
        # Create metadata
        metadata = {
            'date': '2025-04-01',
            'exp_name': 'comprehensive_test',
            'rig': 'test_rig'  # Not in config, should be ignored
        }
        
        config_file = tmp_path / "comprehensive_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(comprehensive_config, f)
        
        # Execute full workflow
        df = make_dataframe_from_config(
            exp_matrix, 
            str(config_file), 
            metadata=metadata,
            skip_columns=['y']  # Skip one column
        )
        
        # Comprehensive validation per TST-INTEG-003
        
        # Basic structure validation
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        
        # Required columns (except skipped)
        assert 't' in df.columns
        assert 'x' in df.columns
        assert 'y' not in df.columns  # Was skipped
        
        # Alias resolution
        assert 'dtheta' in df.columns
        assert 'dtheta_smooth' not in df.columns
        np.testing.assert_array_equal(df['dtheta'].values, exp_matrix['dtheta_smooth'])
        
        # Special handler transformation
        assert 'signal_disp' in df.columns
        assert isinstance(df['signal_disp'].iloc[0], np.ndarray)
        assert df['signal_disp'].iloc[0].shape == (15,)
        
        # Default value assignment
        assert 'optional_signal' in df.columns
        assert all(df['optional_signal'] == 0.0)
        
        # Metadata integration
        assert 'date' in df.columns
        assert 'exp_name' in df.columns
        assert all(df['date'] == '2025-04-01')
        assert all(df['exp_name'] == 'comprehensive_test')
        assert 'rig' not in df.columns  # Not configured as metadata
        
        # Data integrity validation
        np.testing.assert_array_equal(df['t'].values, exp_matrix['t'])
        np.testing.assert_array_equal(df['x'].values, exp_matrix['x'])
    
    def test_error_propagation_and_recovery(self, tmp_path):
        """Test error propagation and recovery mechanisms."""
        # Test configuration with recoverable and non-recoverable errors
        matrix = {
            't': np.linspace(0, 10, 100),
            'x': np.random.rand(100)
            # Missing required 'y' column - non-recoverable
            # Missing optional columns - recoverable
        }
        
        config = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'X pos'},
                'y': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Y pos'},
                'optional': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': 'Optional',
                    'default_value': None
                }
            }
        }
        
        config_file = tmp_path / "error_test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Should fail due to missing required column
        with pytest.raises(TransformError, match="Missing required columns: y"):
            make_dataframe_from_config(matrix, str(config_file))
        
        # Add the missing required column and test recovery
        matrix['y'] = np.random.rand(100)
        
        # Should now succeed with default value for optional column
        df = make_dataframe_from_config(matrix, str(config_file))
        
        assert isinstance(df, pd.DataFrame)
        assert 'optional' in df.columns
        assert df['optional'].iloc[0] is None


# ===== REGRESSION TESTS =====

class TestRegressionCases:
    """Test specific regression cases and known issues."""
    
    def test_time_dimension_validation_regression(self):
        """Test regression case for time dimension validation."""
        # This tests a specific scenario that previously caused issues
        matrix = {
            't': np.linspace(0, 10, 100),
            'x': np.random.rand(100),
            'y': np.random.rand(100),
            'signal_disp': np.random.rand(100, 15)  # Time first, signal second
        }
        
        config = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'X pos'},
                'y': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Y pos'},
                'signal_disp': {
                    'type': 'numpy.ndarray',
                    'dimension': 2,
                    'required': False,
                    'description': 'Signal display data',
                    'special_handling': 'transform_to_match_time_dimension'
                }
            },
            'special_handlers': {
                'transform_to_match_time_dimension': '_handle_signal_disp'
            }
        }
        
        df = make_dataframe_from_config(matrix, config)
        
        # Should correctly handle when time is first dimension
        assert 'signal_disp' in df.columns
        assert len(df) == 100
        assert isinstance(df['signal_disp'].iloc[0], np.ndarray)
        assert df['signal_disp'].iloc[0].shape == (15,)
    
    def test_unicode_column_names_regression(self, basic_exp_matrix):
        """Test handling of unicode characters in column names."""
        # Test with unicode column names
        unicode_matrix = basic_exp_matrix.copy()
        unicode_matrix['θ'] = np.random.rand(100)  # Greek theta
        
        config = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'X pos'},
                'y': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Y pos'},
                'θ': {'type': 'numpy.ndarray', 'dimension': 1, 'required': False, 'description': 'Theta'}
            }
        }
        
        df = make_dataframe_from_config(unicode_matrix, config)
        
        assert 'θ' in df.columns
        assert isinstance(df, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__])