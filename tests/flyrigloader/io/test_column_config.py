"""
Modern test suite for column configuration functionality.

This module provides comprehensive testing for column configuration validation,
YAML config loading, Pydantic model enforcement, and DataFrame transformation
according to TST-MOD-001, TST-MOD-002, and TST-MOD-003 requirements.
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pytest
import yaml
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from pydantic import ValidationError
from unittest.mock import MagicMock, patch, mock_open

from flyrigloader.io.column_models import (
    ColumnConfig,
    ColumnConfigDict,
    ColumnDimension,
    SpecialHandlerType,
    load_column_config,
    get_config_from_source,
    get_default_config_path,
)
from flyrigloader.io.transformers import make_dataframe_from_config
from flyrigloader.exceptions import TransformError


# ============================================================================
# PYTEST FIXTURES - Modern fixture-based test architecture per TST-MOD-001
# ============================================================================

@pytest.fixture
def minimal_column_config():
    """
    Minimal valid column configuration for testing.
    
    Returns:
        Dict[str, Any]: Basic column configuration
    """
    return {
        'columns': {
            't': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': True,
                'description': 'Time values'
            }
        },
        'special_handlers': {}
    }


@pytest.fixture
def comprehensive_column_config():
    """
    Comprehensive column configuration with all feature types.
    
    Returns:
        Dict[str, Any]: Complete column configuration for comprehensive testing
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
                'required': False,
                'description': 'Y position',
                'default_value': None
            },
            'signal_disp': {
                'type': 'numpy.ndarray',
                'dimension': 2,
                'required': False,
                'description': 'Signal display data',
                'special_handling': 'transform_to_match_time_dimension'
            },
            'theta_smooth': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': False,
                'description': 'Smoothed heading angle',
                'alias': 'theta_raw'
            },
            'metadata_field': {
                'type': 'string',
                'required': False,
                'is_metadata': True,
                'description': 'Metadata information'
            }
        },
        'special_handlers': {
            'transform_to_match_time_dimension': '_handle_signal_disp'
        }
    }


@pytest.fixture
def sample_exp_matrix_basic():
    """
    Basic experimental data matrix for testing.
    
    Returns:
        Dict[str, np.ndarray]: Sample experimental data
    """
    return {
        't': np.linspace(0, 10, 100),
        'x': np.random.RandomState(42).rand(100),
        'y': np.random.RandomState(42).rand(100)
    }


@pytest.fixture
def sample_exp_matrix_complex():
    """
    Complex experimental data matrix with all data types.
    
    Returns:
        Dict[str, np.ndarray]: Complex experimental data
    """
    np.random.seed(42)  # Ensure reproducible test data
    return {
        't': np.linspace(0, 10, 100),
        'x': np.random.rand(100),
        'y': np.random.rand(100),
        'signal_disp': np.random.rand(15, 100),  # 15 channels, 100 time points
        'theta_raw': np.random.rand(100) * 2 * np.pi,  # Aliased column
        'scalar_value': 42.0
    }


@pytest.fixture
def temp_config_file(comprehensive_column_config):
    """
    Create temporary YAML configuration file for testing.
    
    Args:
        comprehensive_column_config: Configuration data to write
        
    Yields:
        str: Path to temporary configuration file
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(comprehensive_column_config, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_file_system(mocker):
    """
    Mock file system operations for testing file loading scenarios.
    
    Args:
        mocker: pytest-mock fixture
        
    Returns:
        MagicMock: Mock for file operations
    """
    mock_open_func = mocker.mock_open()
    mocker.patch('builtins.open', mock_open_func)
    return mock_open_func


# ============================================================================
# UNIT TESTS - Pydantic Model Validation per F-004-RQ-001 through F-004-RQ-005
# ============================================================================

class TestColumnConfig:
    """Test suite for ColumnConfig Pydantic model validation."""

    def test_column_config_basic_creation(self):
        """Test basic ColumnConfig creation with minimal required fields."""
        config = ColumnConfig(
            type='numpy.ndarray',
            description='Test column'
        )
        assert config.type == 'numpy.ndarray'
        assert config.description == 'Test column'
        assert config.required is False  # Default value
        assert config.dimension is None
        assert config.alias is None

    def test_column_config_full_creation(self):
        """Test ColumnConfig creation with all fields specified."""
        config = ColumnConfig(
            type='numpy.ndarray',
            dimension=ColumnDimension.TWO_D,
            required=True,
            description='Test 2D array column',
            alias='alternative_name',
            is_metadata=False,
            default_value=None,
            special_handling=SpecialHandlerType.TRANSFORM_TIME_DIMENSION
        )
        
        assert config.type == 'numpy.ndarray'
        assert config.dimension == ColumnDimension.TWO_D
        assert config.required is True
        assert config.description == 'Test 2D array column'
        assert config.alias == 'alternative_name'
        assert config.is_metadata is False
        assert config.default_value is None
        assert config.special_handling == SpecialHandlerType.TRANSFORM_TIME_DIMENSION

    @pytest.mark.parametrize("dimension_input,expected_enum", [
        (1, ColumnDimension.ONE_D),
        (2, ColumnDimension.TWO_D), 
        (3, ColumnDimension.THREE_D),
        (ColumnDimension.ONE_D, ColumnDimension.ONE_D),
        (None, None),
    ], ids=['int_1d', 'int_2d', 'int_3d', 'enum_1d', 'none'])
    def test_dimension_validation_valid_cases(self, dimension_input, expected_enum):
        """Test dimension field validation for valid inputs per TST-MOD-002."""
        config = ColumnConfig(
            type='numpy.ndarray',
            dimension=dimension_input,
            description='Test column'
        )
        assert config.dimension == expected_enum

    @pytest.mark.parametrize("invalid_dimension", [
        0, 4, 5, -1, 2.5, "invalid", [], {}
    ], ids=['zero', 'four', 'five', 'negative', 'float', 'string', 'list', 'dict'])
    def test_dimension_validation_invalid_cases(self, invalid_dimension):
        """Test dimension field validation for invalid inputs per TST-MOD-002."""
        with pytest.raises(ValidationError, match="(Input should be 1, 2 or 3|Dimension must be 1, 2, or 3)"):
            ColumnConfig(
                type='numpy.ndarray',
                dimension=invalid_dimension,
                description='Test column'
            )

    @pytest.mark.parametrize("handler_input,expected_enum", [
        ("extract_first_column_if_2d", SpecialHandlerType.EXTRACT_FIRST_COLUMN),
        ("transform_to_match_time_dimension", SpecialHandlerType.TRANSFORM_TIME_DIMENSION),
        (SpecialHandlerType.EXTRACT_FIRST_COLUMN, SpecialHandlerType.EXTRACT_FIRST_COLUMN),
        (None, None),
    ], ids=['extract_string', 'transform_string', 'enum_direct', 'none'])
    def test_special_handling_validation_valid_cases(self, handler_input, expected_enum):
        """Test special_handling field validation for valid inputs per TST-MOD-002."""
        config = ColumnConfig(
            type='numpy.ndarray',
            special_handling=handler_input,
            description='Test column'
        )
        assert config.special_handling == expected_enum

    @pytest.mark.parametrize("invalid_handler", [
        "invalid_handler", "unknown_transformation", 123, [], {}
    ], ids=['invalid_string', 'unknown_string', 'int', 'list', 'dict'])
    def test_special_handling_validation_invalid_cases(self, invalid_handler):
        """Test special_handling field validation for invalid inputs per TST-MOD-002."""
        with pytest.raises(ValidationError, match="(Input should be|Special handler must be one of)"):
            ColumnConfig(
                type='numpy.ndarray',
                special_handling=invalid_handler,
                description='Test column'
            )

    def test_model_validator_dimension_type_compatibility_warning(self, caplog):
        """Test model validator warns about dimension with non-numpy types."""
        config = ColumnConfig(
            type='string',
            dimension=1,
            description='String column with dimension'
        )
        
        # Verify object is created but warning is logged
        assert config.type == 'string'
        assert config.dimension == ColumnDimension.ONE_D
        # Check warning was logged (exact message checking depends on Loguru setup)

    def test_model_validator_signal_disp_dimension_warning(self, caplog):
        """Test model validator warns about incorrect dimension for signal_disp."""
        config = ColumnConfig(
            type='numpy.ndarray',
            dimension=1,
            special_handling=SpecialHandlerType.TRANSFORM_TIME_DIMENSION,
            description='Signal disp with wrong dimension'
        )
        
        # Verify object is created but warning is logged
        assert config.special_handling == SpecialHandlerType.TRANSFORM_TIME_DIMENSION
        assert config.dimension == ColumnDimension.ONE_D


class TestColumnConfigDict:
    """Test suite for ColumnConfigDict Pydantic model validation."""

    def test_column_config_dict_creation(self, comprehensive_column_config):
        """Test ColumnConfigDict creation from valid configuration."""
        config_dict = ColumnConfigDict.model_validate(comprehensive_column_config)
        
        assert len(config_dict.columns) == 6
        assert 't' in config_dict.columns
        assert 'x' in config_dict.columns
        assert config_dict.columns['t'].required is True
        assert config_dict.columns['y'].required is False
        assert len(config_dict.special_handlers) == 1

    def test_column_config_dict_empty_special_handlers(self, minimal_column_config):
        """Test ColumnConfigDict with empty special_handlers."""
        config_dict = ColumnConfigDict.model_validate(minimal_column_config)
        
        assert len(config_dict.columns) == 1
        assert len(config_dict.special_handlers) == 0

    def test_special_handlers_validation_warning(self, caplog, mocker):
        """Test special handlers validation logs warning for undefined handlers."""
        # Mock the logger to capture warnings
        mock_logger = mocker.patch('flyrigloader.io.column_models.logger')
        
        config_data = {
            'columns': {
                'test_col': {
                    'type': 'numpy.ndarray',
                    'description': 'Test column',
                    'special_handling': 'transform_to_match_time_dimension'
                }
            },
            'special_handlers': {}  # Missing the required handler
        }
        
        config_dict = ColumnConfigDict.model_validate(config_data)
        
        # Verify the warning was logged
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        assert 'transform_to_match_time_dimension' in warning_call
        assert 'not defined in special_handlers' in warning_call


# ============================================================================
# CONFIGURATION LOADING TESTS - YAML and Schema Enforcement per TST-MOD-003  
# ============================================================================

class TestConfigurationLoading:
    """Test suite for configuration loading and validation."""

    def test_load_column_config_from_file(self, temp_config_file):
        """Test loading column configuration from YAML file."""
        config = load_column_config(temp_config_file)
        
        assert isinstance(config, ColumnConfigDict)
        assert len(config.columns) > 0
        assert 't' in config.columns
        assert config.columns['t'].required is True

    def test_load_column_config_file_not_found(self):
        """Test loading column configuration from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_column_config('/nonexistent/path/config.yaml')

    @pytest.mark.parametrize("invalid_yaml_content", [
        "invalid: yaml: content:",  # Invalid YAML syntax
        "not_dict_format",  # Not a dictionary
        "",  # Empty file
    ], ids=['invalid_syntax', 'not_dict', 'empty'])
    def test_load_column_config_invalid_yaml(self, mocker, invalid_yaml_content):
        """Test loading column configuration from invalid YAML files per TST-MOD-002."""
        mock_open_func = mocker.mock_open(read_data=invalid_yaml_content)
        mocker.patch('builtins.open', mock_open_func)
        
        with pytest.raises((yaml.YAMLError, ValidationError, AttributeError)):
            load_column_config('test_config.yaml')

    def test_get_config_from_source_file_path(self, temp_config_file):
        """Test get_config_from_source with file path."""
        config = get_config_from_source(temp_config_file)
        
        assert isinstance(config, ColumnConfigDict)
        assert len(config.columns) > 0

    def test_get_config_from_source_dict(self, comprehensive_column_config):
        """Test get_config_from_source with dictionary."""
        config = get_config_from_source(comprehensive_column_config)
        
        assert isinstance(config, ColumnConfigDict)
        assert len(config.columns) == 6

    def test_get_config_from_source_column_config_dict(self, comprehensive_column_config):
        """Test get_config_from_source with ColumnConfigDict instance."""
        original_config = ColumnConfigDict.model_validate(comprehensive_column_config)
        config = get_config_from_source(original_config)
        
        assert config is original_config  # Should return the same instance

    @pytest.mark.parametrize("invalid_source", [
        123,  # Integer
        [],   # List
        object(),  # Generic object
        True,  # Boolean
    ], ids=['int', 'list', 'object', 'bool'])
    def test_get_config_from_source_invalid_type(self, invalid_source):
        """Test get_config_from_source with invalid source types per TST-MOD-002."""
        with pytest.raises(TypeError, match="config_source must be"):
            get_config_from_source(invalid_source)

    def test_get_config_from_source_none_uses_default(self, mocker):
        """Test get_config_from_source with None uses default configuration."""
        mock_get_default = mocker.patch('flyrigloader.io.column_models.get_default_config_path')
        mock_load_config = mocker.patch('flyrigloader.io.column_models.load_column_config')
        mock_config = ColumnConfigDict(columns={}, special_handlers={})
        mock_load_config.return_value = mock_config
        mock_get_default.return_value = '/default/path/config.yaml'
        
        config = get_config_from_source(None)
        
        mock_get_default.assert_called_once()
        mock_load_config.assert_called_once_with('/default/path/config.yaml')
        assert config is mock_config

    def test_get_default_config_path(self):
        """Test get_default_config_path returns valid path."""
        path = get_default_config_path()
        
        assert isinstance(path, str)
        assert path.endswith('column_config.yaml')
        assert os.path.dirname(path)  # Should have a directory component


# ============================================================================
# PROPERTY-BASED TESTING - Hypothesis integration per Section 3.6.3 requirements
# ============================================================================

class TestPropertyBasedValidation:
    """Property-based tests using Hypothesis for robust validation."""

    @given(
        dimension=st.one_of(st.just(None), st.integers(min_value=1, max_value=3)),
        required=st.booleans(),
        is_metadata=st.booleans(),
        type_str=st.sampled_from(['numpy.ndarray', 'string', 'float', 'int'])
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_column_config_property_based_creation(self, dimension, required, is_metadata, type_str):
        """Property-based test for ColumnConfig creation with various combinations."""
        config = ColumnConfig(
            type=type_str,
            dimension=dimension,
            required=required,
            is_metadata=is_metadata,
            description=f'Test {type_str} column'
        )
        
        assert config.type == type_str
        assert config.required == required
        assert config.is_metadata == is_metadata
        if dimension is not None:
            assert config.dimension == ColumnDimension(dimension)
        else:
            assert config.dimension is None

    @given(
        array_size=st.integers(min_value=10, max_value=1000),
        num_columns=st.integers(min_value=1, max_value=5)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_exp_matrix_property_based_validation(self, array_size, num_columns):
        """Property-based test for experimental matrix data validation."""
        # Generate test data
        exp_matrix = {
            't': np.linspace(0, 10, array_size)
        }
        
        # Add random columns
        np.random.seed(42)
        for i in range(num_columns):
            exp_matrix[f'col_{i}'] = np.random.rand(array_size)
        
        # Create minimal config
        config_data = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                }
            },
            'special_handlers': {}
        }
        
        # Add column configs for generated columns
        for i in range(num_columns):
            config_data['columns'][f'col_{i}'] = {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': False,
                'description': f'Column {i}'
            }
        
        # This should not raise an exception
        config = ColumnConfigDict.model_validate(config_data)
        df = make_dataframe_from_config(exp_matrix, config)
        
        assert len(df) == array_size
        assert 't' in df.columns
        for i in range(num_columns):
            assert f'col_{i}' in df.columns

    @given(
        special_handler=st.sampled_from([
            'extract_first_column_if_2d',
            'transform_to_match_time_dimension'
        ]),
        dimension=st.integers(min_value=1, max_value=3)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_special_handler_property_based_validation(self, special_handler, dimension):
        """Property-based test for special handler validation."""
        config = ColumnConfig(
            type='numpy.ndarray',
            dimension=dimension,
            special_handling=special_handler,
            description='Test special handler column'
        )
        
        assert config.special_handling.value == special_handler
        assert config.dimension == ColumnDimension(dimension)


# ============================================================================
# PERFORMANCE BENCHMARK TESTS - per F-004 technical specifications
# ============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmark tests for configuration loading operations."""

    def test_configuration_loading_performance_benchmark(self, benchmark, temp_config_file):
        """
        Benchmark configuration loading performance - must complete within 100ms per F-004.
        
        Performance requirement: < 100ms for configuration loading operations.
        """
        def load_config():
            return load_column_config(temp_config_file)
        
        result = benchmark(load_config)
        
        # Verify the result is valid
        assert isinstance(result, ColumnConfigDict)
        assert len(result.columns) > 0
        
        # Performance validation is handled by pytest-benchmark
        # The benchmark fixture automatically measures execution time

    def test_column_config_dict_validation_performance_benchmark(self, benchmark, comprehensive_column_config):
        """
        Benchmark ColumnConfigDict validation performance.
        
        Performance requirement: < 50ms for validation operations.
        """
        def validate_config():
            return ColumnConfigDict.model_validate(comprehensive_column_config)
        
        result = benchmark(validate_config)
        
        # Verify the result is valid
        assert isinstance(result, ColumnConfigDict)
        assert len(result.columns) == 6

    def test_make_dataframe_performance_benchmark(self, benchmark, sample_exp_matrix_complex, comprehensive_column_config):
        """
        Benchmark DataFrame creation performance from configuration.
        
        Performance requirement: < 500ms for DataFrame transformation per 1M rows.
        """
        config = ColumnConfigDict.model_validate(comprehensive_column_config)
        
        def create_dataframe():
            return make_dataframe_from_config(sample_exp_matrix_complex, config)
        
        result = benchmark(create_dataframe)
        
        # Verify the result is valid
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 't' in result.columns

    @pytest.mark.parametrize("config_size", [10, 50, 100], ids=['small', 'medium', 'large'])
    def test_large_configuration_performance_benchmark(self, benchmark, config_size):
        """
        Benchmark performance with varying configuration sizes.
        
        Tests scalability of configuration validation with increasing numbers of columns.
        """
        # Generate large configuration
        large_config = {
            'columns': {},
            'special_handlers': {}
        }
        
        for i in range(config_size):
            large_config['columns'][f'col_{i}'] = {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': i < config_size // 2,  # Half required, half optional
                'description': f'Column {i}'
            }
        
        def validate_large_config():
            return ColumnConfigDict.model_validate(large_config)
        
        result = benchmark(validate_large_config)
        
        # Verify the result is valid
        assert isinstance(result, ColumnConfigDict)
        assert len(result.columns) == config_size


# ============================================================================
# ERROR HANDLING AND EDGE CASE TESTS - per F-004 validation rules
# ============================================================================

class TestErrorHandlingAndEdgeCases:
    """Comprehensive error handling tests for invalid configuration scenarios."""

    @pytest.mark.parametrize("missing_field", [
        'type', 'description'
    ], ids=['missing_type', 'missing_description'])
    def test_column_config_missing_required_fields(self, missing_field):
        """Test ColumnConfig validation with missing required fields per TST-MOD-002."""
        config_data = {
            'type': 'numpy.ndarray',
            'description': 'Test column'
        }
        del config_data[missing_field]
        
        with pytest.raises(ValidationError) as exc_info:
            ColumnConfig.model_validate(config_data)
        
        assert missing_field in str(exc_info.value)

    @pytest.mark.parametrize("invalid_config_structure", [
        {},  # Empty dict
        {'columns': {}},  # Empty columns
        {'special_handlers': {}},  # Missing columns key
        {'columns': 'invalid'},  # Columns not a dict
        {'columns': {}, 'special_handlers': 'invalid'},  # Special handlers not a dict
    ], ids=['empty', 'empty_columns', 'missing_columns', 'invalid_columns_type', 'invalid_handlers_type'])
    def test_column_config_dict_invalid_structures(self, invalid_config_structure):
        """Test ColumnConfigDict validation with invalid structures per TST-MOD-002."""
        with pytest.raises(ValidationError):
            ColumnConfigDict.model_validate(invalid_config_structure)

    def test_column_config_dict_invalid_column_config(self):
        """Test ColumnConfigDict validation with invalid column configuration."""
        invalid_config = {
            'columns': {
                'invalid_col': {
                    # Missing required 'type' and 'description' fields
                    'dimension': 1
                }
            },
            'special_handlers': {}
        }
        
        with pytest.raises(ValidationError):
            ColumnConfigDict.model_validate(invalid_config)

    @pytest.mark.parametrize("invalid_yaml_file_scenario", [
        "nonexistent_file.yaml",
        "/invalid/path/config.yaml"
    ], ids=['nonexistent', 'invalid_path'])
    def test_load_column_config_file_errors(self, invalid_yaml_file_scenario):
        """Test load_column_config with various file error scenarios per TST-MOD-002."""
        with pytest.raises(FileNotFoundError):
            load_column_config(invalid_yaml_file_scenario)

    def test_make_dataframe_missing_required_columns(self, sample_exp_matrix_basic):
        """Test make_dataframe_from_config with missing required columns."""
        config_data = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                },
                'missing_col': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Missing column'
                }
            },
            'special_handlers': {}
        }
        
        config = ColumnConfigDict.model_validate(config_data)
        
        with pytest.raises(TransformError, match="Missing required columns"):
            make_dataframe_from_config(sample_exp_matrix_basic, config)

    def test_make_dataframe_with_aliases_missing_both(self):
        """Test make_dataframe_from_config when both main column and alias are missing."""
        exp_matrix = {
            't': np.linspace(0, 10, 100)
        }
        
        config_data = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                },
                'missing_col': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'alias': 'also_missing',
                    'description': 'Missing column with missing alias'
                }
            },
            'special_handlers': {}
        }
        
        config = ColumnConfigDict.model_validate(config_data)
        
        with pytest.raises(TransformError, match="Missing required columns"):
            make_dataframe_from_config(exp_matrix, config)

    def test_make_dataframe_with_default_values(self):
        """Test make_dataframe_from_config applies default values for missing optional columns."""
        exp_matrix = {
            't': np.linspace(0, 10, 100)
        }
        
        config_data = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                },
                'optional_col': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'default_value': None,
                    'description': 'Optional column with default'
                }
            },
            'special_handlers': {}
        }
        
        config = ColumnConfigDict.model_validate(config_data)
        df = make_dataframe_from_config(exp_matrix, config)
        
        assert 't' in df.columns
        assert 'optional_col' in df.columns
        assert df['optional_col'].iloc[0] is None

    def test_special_handler_undefined_behavior(self, caplog, mocker):
        """Test behavior when special handler is referenced but not defined."""
        # Mock the logger to capture warnings
        mock_logger = mocker.patch('flyrigloader.io.column_models.logger')
        
        config_data = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                },
                'special_col': {
                    'type': 'numpy.ndarray',
                    'dimension': 2,
                    'required': False,
                    'special_handling': 'transform_to_match_time_dimension',
                    'description': 'Column with special handling'
                }
            },
            'special_handlers': {}  # Handler not defined
        }
        
        # This should create the config but log a warning
        config = ColumnConfigDict.model_validate(config_data)
        
        # Verify the warning was logged
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        assert 'transform_to_match_time_dimension' in warning_call

    @pytest.mark.parametrize("corrupted_yaml_content", [
        "columns:\n  t:\n    type: numpy.ndarray\n    description incomplete",  # Incomplete YAML
        "columns:\n  t: [invalid, structure]",  # Wrong structure for column
        "columns:\n  't':\n    type: 'numpy.ndarray'\n    dimension: 'invalid_dimension'",  # Invalid dimension type
    ], ids=['incomplete', 'wrong_structure', 'invalid_dimension'])
    def test_corrupted_yaml_configurations(self, mocker, corrupted_yaml_content):
        """Test handling of corrupted YAML configuration files per TST-MOD-002."""
        mock_open_func = mocker.mock_open(read_data=corrupted_yaml_content)
        mocker.patch('builtins.open', mock_open_func)
        
        with pytest.raises((yaml.YAMLError, ValidationError)):
            load_column_config('corrupted_config.yaml')


# ============================================================================
# INTEGRATION TESTS - End-to-end workflow validation per TST-INTEG-001
# ============================================================================

class TestIntegrationWorkflows:
    """Integration tests for complete column configuration workflows."""

    def test_complete_yaml_to_dataframe_workflow(self, temp_config_file, sample_exp_matrix_complex):
        """
        Test complete workflow from YAML configuration to DataFrame output.
        
        This test validates the entire pipeline per TST-INTEG-001 requirements.
        """
        # Load configuration from YAML file
        config = load_column_config(temp_config_file)
        
        # Verify configuration loaded correctly
        assert isinstance(config, ColumnConfigDict)
        assert len(config.columns) > 0
        
        # Create DataFrame from configuration
        df = make_dataframe_from_config(sample_exp_matrix_complex, config)
        
        # Verify DataFrame structure and content
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100  # Based on sample data
        assert 't' in df.columns
        assert 'x' in df.columns
        
        # Verify data types and integrity
        assert df['t'].dtype == np.float64
        assert df['x'].dtype == np.float64
        
        # Verify time dimension consistency
        assert len(df['t']) == 100
        assert df['t'].min() >= 0
        assert df['t'].max() <= 10

    def test_configuration_with_metadata_integration(self, temp_config_file):
        """Test integration of configuration with metadata fields."""
        exp_matrix = {
            't': np.linspace(0, 10, 100),
            'x': np.random.RandomState(42).rand(100)
        }
        
        metadata = {
            'metadata_field': 'test_metadata_value',
            'experiment_id': 'EXP_001'
        }
        
        config = load_column_config(temp_config_file)
        df = make_dataframe_from_config(exp_matrix, config, metadata=metadata)
        
        # Verify metadata integration
        assert isinstance(df, pd.DataFrame)
        if 'metadata_field' in config.columns and config.columns['metadata_field'].is_metadata:
            assert 'metadata_field' in df.columns
            assert df['metadata_field'].iloc[0] == 'test_metadata_value'

    def test_alias_resolution_integration(self, sample_exp_matrix_complex):
        """Test integration of column alias resolution in complete workflow."""
        config_data = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                },
                'theta_smooth': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'alias': 'theta_raw',  # Maps to theta_raw in exp_matrix
                    'description': 'Smoothed theta values'
                }
            },
            'special_handlers': {}
        }
        
        config = ColumnConfigDict.model_validate(config_data)
        df = make_dataframe_from_config(sample_exp_matrix_complex, config)
        
        # Verify alias resolution worked
        assert 'theta_smooth' in df.columns
        assert 'theta_raw' not in df.columns  # Original name should not be present
        assert len(df['theta_smooth']) == 100

    def test_special_handler_integration_workflow(self):
        """Test integration of special handlers in complete workflow."""
        exp_matrix = {
            't': np.linspace(0, 10, 100),
            'signal_disp': np.random.RandomState(42).rand(15, 100)  # 15 channels, 100 time points
        }
        
        config_data = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                },
                'signal_disp': {
                    'type': 'numpy.ndarray',
                    'dimension': 2,
                    'required': False,
                    'special_handling': 'transform_to_match_time_dimension',
                    'description': 'Signal display data'
                }
            },
            'special_handlers': {
                'transform_to_match_time_dimension': '_handle_signal_disp'
            }
        }
        
        config = ColumnConfigDict.model_validate(config_data)
        df = make_dataframe_from_config(exp_matrix, config)
        
        # Verify special handler application
        assert 'signal_disp' in df.columns
        assert len(df) == 100
        
        # Verify signal_disp was transformed correctly
        # Each row should contain an array (transformed from 2D to Series of arrays)
        first_signal = df['signal_disp'].iloc[0]
        assert isinstance(first_signal, np.ndarray)
        assert len(first_signal) == 15  # Should be 15 channels per time point


# ============================================================================
# MOCK-BASED TESTS - pytest-mock integration per TST-MOD-003
# ============================================================================

class TestMockIntegration:
    """Test suite demonstrating pytest-mock integration for isolated testing."""

    def test_yaml_loading_with_mocked_file_operations(self, mocker):
        """Test YAML configuration loading with mocked file operations per TST-MOD-003."""
        # Mock file content
        yaml_content = """
columns:
  t:
    type: numpy.ndarray
    dimension: 1
    required: true
    description: Time values
special_handlers: {}
"""
        
        # Mock file operations
        mock_open_func = mocker.mock_open(read_data=yaml_content)
        mocker.patch('builtins.open', mock_open_func)
        
        # Mock yaml.safe_load to return our test data
        test_config_data = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                }
            },
            'special_handlers': {}
        }
        mock_yaml_load = mocker.patch('yaml.safe_load', return_value=test_config_data)
        
        # Execute the function
        config = load_column_config('test_config.yaml')
        
        # Verify mocks were called correctly
        mock_open_func.assert_called_once_with('test_config.yaml', 'r')
        mock_yaml_load.assert_called_once()
        
        # Verify result
        assert isinstance(config, ColumnConfigDict)
        assert len(config.columns) == 1
        assert 't' in config.columns

    def test_default_config_path_with_mocked_os_operations(self, mocker):
        """Test default configuration path with mocked OS operations per TST-MOD-003."""
        # Mock os.path operations
        mock_dirname = mocker.patch('os.path.dirname', return_value='/mocked/directory')
        mock_abspath = mocker.patch('os.path.abspath', return_value='/mocked/directory/column_models.py')
        mock_join = mocker.patch('os.path.join', return_value='/mocked/directory/column_config.yaml')
        
        # Execute the function
        config_path = get_default_config_path()
        
        # Verify mocks were called correctly
        mock_abspath.assert_called_once()
        mock_dirname.assert_called_once()
        mock_join.assert_called_once_with('/mocked/directory', 'column_config.yaml')
        
        # Verify result
        assert config_path == '/mocked/directory/column_config.yaml'

    def test_make_dataframe_with_mocked_validation(self, mocker, sample_exp_matrix_basic):
        """Test DataFrame creation with mocked validation functions per TST-MOD-003."""
        # Create a mock configuration
        mock_config = MagicMock(spec=ColumnConfigDict)
        mock_config.columns = {
            't': MagicMock(required=True, is_metadata=False, alias=None),
            'x': MagicMock(required=True, is_metadata=False, alias=None)
        }
        mock_config.special_handlers = {}
        
        # Mock the get_config_from_source function
        mock_get_config = mocker.patch(
            'flyrigloader.io.pickle.get_config_from_source',
            return_value=mock_config
        )
        
        # Mock the validation function to return no missing columns
        mock_validate = mocker.patch(
            'flyrigloader.io.pickle._validate_required_columns',
            return_value=[]
        )
        
        # Execute the function
        df = make_dataframe_from_config(sample_exp_matrix_basic, mock_config)
        
        # Verify mocks were called
        mock_get_config.assert_called_once_with(mock_config)
        mock_validate.assert_called_once()
        
        # Verify result
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_configuration_validation_with_mocked_pydantic(self, mocker, comprehensive_column_config):
        """Test configuration validation with mocked Pydantic operations per TST-MOD-003."""
        # Mock the Pydantic model_validate method
        mock_validate = mocker.patch.object(ColumnConfigDict, 'model_validate')
        
        # Create expected return value
        expected_config = ColumnConfigDict(
            columns={'t': ColumnConfig(type='numpy.ndarray', description='Time')},
            special_handlers={}
        )
        mock_validate.return_value = expected_config
        
        # Execute the function
        config = get_config_from_source(comprehensive_column_config)
        
        # Verify mock was called with correct arguments
        mock_validate.assert_called_once_with(comprehensive_column_config)
        
        # Verify result
        assert config is expected_config


# ============================================================================
# TEST EXECUTION CONFIGURATION
# ============================================================================

# Mark slow tests for conditional execution
slow_tests = pytest.mark.slow

# Mark integration tests for conditional execution  
integration_tests = pytest.mark.integration

# Mark benchmark tests for conditional execution
benchmark_tests = pytest.mark.benchmark