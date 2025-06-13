"""
Tests for the Pydantic-based column configuration models.
"""

import os
import tempfile
import yaml
import pytest
import numpy as np
from pydantic import ValidationError
import pandas as pd

from flyrigloader.io.column_models import (
    ColumnConfig,
    ColumnConfigDict,
    ColumnDimension,
    SpecialHandlerType,
    load_column_config,
    get_config_from_source
)
import pydantic


def test_models_inherit_basemodel():
    """Ensure core models derive from Pydantic BaseModel."""
    assert issubclass(ColumnConfig, pydantic.BaseModel)
    assert issubclass(ColumnConfigDict, pydantic.BaseModel)
from flyrigloader.io.pickle import make_dataframe_from_config


def test_column_config_validation():
    """Test basic validation of individual column configurations."""
    # Valid configuration
    config = ColumnConfig(
        type="numpy.ndarray",
        dimension=1,
        required=True,
        description="Time values"
    )
    assert config.type == "numpy.ndarray"
    assert config.dimension == ColumnDimension.ONE_D
    assert config.required is True
    
    # Invalid dimension
    with pytest.raises(ValidationError):
        ColumnConfig(
            type="numpy.ndarray",
            dimension=4,  # Only 1, 2, or 3 are valid
            required=True,
            description="Invalid dimension"
        )
    
    # Invalid special handler
    with pytest.raises(ValidationError):
        ColumnConfig(
            type="numpy.ndarray",
            dimension=1,
            required=True,
            description="Test column",
            special_handling="invalid_handler"  # Not a valid handler type
        )


# =============================================================================
# PERFORMANCE BENCHMARK TESTS
# =============================================================================

class TestColumnConfigPerformance:
    """Performance benchmark tests for column configuration operations."""
    
    @pytest.mark.benchmark
    def test_column_config_validation_performance(self, benchmark):
        """Benchmark ColumnConfig validation to ensure < 1ms per column."""
        def create_column_config():
            return ColumnConfig(
                type="numpy.ndarray",
                dimension=1,
                required=True,
                description="Performance test column"
            )
        
        result = benchmark(create_column_config)
        
        # Validate that the result is correct
        assert isinstance(result, ColumnConfig)
        assert result.type == "numpy.ndarray"
        assert result.dimension == ColumnDimension.ONE_D
        
        # Performance assertion: should be well under 1ms
        assert benchmark.stats.mean < 0.001  # 1ms requirement
    
    @pytest.mark.benchmark
    def test_column_config_dict_validation_performance(self, benchmark, sample_column_configs):
        """Benchmark ColumnConfigDict validation performance."""
        def create_config_dict():
            return ColumnConfigDict(
                columns=sample_column_configs,
                special_handlers={
                    "transform_to_match_time_dimension": "_handle_signal_disp"
                }
            )
        
        result = benchmark(create_config_dict)
        
        # Validate result correctness
        assert isinstance(result, ColumnConfigDict)
        assert len(result.columns) == len(sample_column_configs)
        
        # Performance requirement: proportional to number of columns
        max_allowed_time = len(sample_column_configs) * 0.001  # 1ms per column
        assert benchmark.stats.mean < max_allowed_time
    
    @pytest.mark.benchmark
    def test_yaml_config_loading_performance(self, benchmark, sample_yaml_config_file):
        """Benchmark YAML configuration loading performance."""
        def load_yaml_config():
            return load_column_config(sample_yaml_config_file)
        
        result = benchmark(load_yaml_config)
        
        # Validate result correctness
        assert isinstance(result, ColumnConfigDict)
        assert 't' in result.columns
        
        # Performance requirement: < 100ms for typical configs
        assert benchmark.stats.mean < 0.1  # 100ms requirement
    
    @pytest.mark.benchmark
    def test_get_config_from_source_performance(self, benchmark, sample_yaml_config_file):
        """Benchmark get_config_from_source performance with file path."""
        def get_config_from_file():
            return get_config_from_source(sample_yaml_config_file)
        
        result = benchmark(get_config_from_file)
        
        # Validate result correctness
        assert isinstance(result, ColumnConfigDict)
        
        # Performance requirement: < 100ms for file loading
        assert benchmark.stats.mean < 0.1
    
    @pytest.mark.benchmark
    def test_dataframe_creation_performance(self, benchmark, performance_test_data, sample_column_config_dict):
        """Benchmark DataFrame creation performance with large datasets."""
        def create_large_dataframe():
            return make_dataframe_from_config(performance_test_data, sample_column_config_dict)
        
        result = benchmark(create_large_dataframe)
        
        # Validate result correctness
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(performance_test_data['t'])
        
        # Performance requirement: < 500ms per 1M rows (scaled to actual size)
        actual_rows = len(performance_test_data['t'])
        max_allowed_time = (actual_rows / 1_000_000) * 0.5  # Scale 500ms requirement
        assert benchmark.stats.mean < max_allowed_time


# =============================================================================
# MULTI-SOURCE LOADING COMPREHENSIVE TESTS
# =============================================================================

class TestMultiSourceConfigLoading:
    """Comprehensive tests for loading configurations from various sources."""
    
    def test_get_config_from_source_file_path_comprehensive(self, sample_yaml_config_file):
        """Test comprehensive file path loading scenarios."""
        # Test with string path
        config_str = get_config_from_source(sample_yaml_config_file)
        assert isinstance(config_str, ColumnConfigDict)
        assert 't' in config_str.columns
        
        # Test with Path object
        from pathlib import Path
        config_path = get_config_from_source(Path(sample_yaml_config_file))
        assert isinstance(config_path, ColumnConfigDict)
        assert 't' in config_path.columns
        
        # Ensure both methods yield equivalent results
        assert set(config_str.columns.keys()) == set(config_path.columns.keys())
    
    def test_get_config_from_source_dict_comprehensive(self):
        """Test comprehensive dictionary loading scenarios."""
        # Minimal configuration
        minimal_config = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                }
            }
        }
        
        config = get_config_from_source(minimal_config)
        assert isinstance(config, ColumnConfigDict)
        assert 't' in config.columns
        assert config.columns['t'].required is True
        
        # Complex configuration with all features
        complex_config = {
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
                    'description': 'Signal display data',
                    'special_handling': 'transform_to_match_time_dimension'
                },
                'metadata': {
                    'type': 'string',
                    'required': False,
                    'is_metadata': True,
                    'description': 'Metadata field'
                },
                'aliased_field': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': 'Field with alias',
                    'alias': 'original_name',
                    'default_value': None
                }
            },
            'special_handlers': {
                'transform_to_match_time_dimension': '_handle_signal_disp'
            }
        }
        
        config = get_config_from_source(complex_config)
        assert isinstance(config, ColumnConfigDict)
        assert len(config.columns) == 4
        assert config.columns['signal_disp'].special_handling == SpecialHandlerType.TRANSFORM_TIME_DIMENSION
        assert config.columns['metadata'].is_metadata is True
        assert config.columns['aliased_field'].alias == 'original_name'
        assert 'transform_to_match_time_dimension' in config.special_handlers
    
    def test_get_config_from_source_model_instance_comprehensive(self, sample_column_config_dict):
        """Test comprehensive model instance loading scenarios."""
        # Test with existing instance
        config = get_config_from_source(sample_column_config_dict)
        assert config is sample_column_config_dict  # Should return same instance
        
        # Test modifying returned instance doesn't affect original
        original_columns_count = len(sample_column_config_dict.columns)
        returned_config = get_config_from_source(sample_column_config_dict)
        
        # They should be the same object
        assert id(returned_config) == id(sample_column_config_dict)
        assert len(returned_config.columns) == original_columns_count
    
    def test_get_config_from_source_none_default_comprehensive(self, mocker):
        """Test comprehensive default configuration loading."""
        # Mock default config path to control test environment
        mock_default_config = {
            'columns': {
                'default_time': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Default time column'
                },
                'default_x': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Default X position'
                }
            },
            'special_handlers': {}
        }
        
        # Mock the file operations for default config
        mock_file_content = yaml.dump(mock_default_config)
        
        with patch('builtins.open', mock_open(read_data=mock_file_content)):
            with patch('yaml.safe_load', return_value=mock_default_config):
                config = get_config_from_source(None)
        
        assert isinstance(config, ColumnConfigDict)
        assert 'default_time' in config.columns
        assert 'default_x' in config.columns
        assert config.columns['default_time'].required is True
    
    @pytest.mark.parametrize("invalid_source", [
        [],  # List
        42,  # Integer
        3.14,  # Float
        True,  # Boolean
        {'invalid': 'structure'},  # Dict without 'columns' key
        {'columns': 'not_a_dict'},  # Invalid columns structure
    ])
    def test_get_config_from_source_error_scenarios(self, invalid_source):
        """Test error handling with various invalid source types."""
        with pytest.raises((TypeError, ValidationError)):
            get_config_from_source(invalid_source)
    
    def test_get_config_from_source_file_not_found(self):
        """Test error handling when configuration file doesn't exist."""
        non_existent_path = "/path/that/absolutely/does/not/exist.yaml"
        
        with pytest.raises(FileNotFoundError):
            get_config_from_source(non_existent_path)
    
    def test_get_config_from_source_invalid_yaml_content(self, tmp_path):
        """Test error handling with invalid YAML content."""
        # Create file with invalid YAML
        invalid_yaml_file = tmp_path / "invalid.yaml"
        with open(invalid_yaml_file, 'w') as f:
            f.write("invalid: yaml: [unclosed_bracket")
        
        with pytest.raises(yaml.YAMLError):
            get_config_from_source(str(invalid_yaml_file))


# =============================================================================
# DATAFRAME CONSTRUCTION COMPREHENSIVE TESTS
# =============================================================================

class TestDataFrameConstructionComprehensive:
    """Comprehensive tests for DataFrame construction scenarios."""
    
    def test_make_dataframe_dict_config_comprehensive(self, sample_exp_matrix):
        """Test DataFrame creation with dictionary configuration."""
        config_dict = {
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
                'signal_disp': {
                    'type': 'numpy.ndarray',
                    'dimension': 2,
                    'required': False,
                    'description': 'Signal display data',
                    'special_handling': 'transform_to_match_time_dimension'
                },
                'optional_field': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': 'Optional field',
                    'default_value': None
                }
            },
            'special_handlers': {
                'transform_to_match_time_dimension': '_handle_signal_disp'
            }
        }
        
        df = make_dataframe_from_config(sample_exp_matrix, config_dict)
        
        # Comprehensive validation
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_exp_matrix['t'])
        assert 't' in df.columns
        assert 'x' in df.columns
        assert 'y' in df.columns
        assert 'signal_disp' in df.columns
        
        # Validate data integrity
        np.testing.assert_array_equal(df['t'].values, sample_exp_matrix['t'])
        np.testing.assert_array_equal(df['x'].values, sample_exp_matrix['x'])
        np.testing.assert_array_equal(df['y'].values, sample_exp_matrix['y'])
        
        # Validate signal_disp handling (should be Series of arrays)
        assert isinstance(df['signal_disp'].iloc[0], np.ndarray)
    
    def test_make_dataframe_model_config_comprehensive(self, sample_exp_matrix, sample_column_config_dict):
        """Test DataFrame creation with ColumnConfigDict model."""
        df = make_dataframe_from_config(sample_exp_matrix, sample_column_config_dict)
        
        # Comprehensive validation
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        # Check that all required columns are present
        required_columns = [
            name for name, config in sample_column_config_dict.columns.items() 
            if config.required and not config.is_metadata
        ]
        
        for col in required_columns:
            if col in sample_exp_matrix or (
                sample_column_config_dict.columns[col].alias and 
                sample_column_config_dict.columns[col].alias in sample_exp_matrix
            ):
                assert col in df.columns or sample_column_config_dict.columns[col].alias in df.columns
    
    def test_make_dataframe_default_config_realistic(self):
        """Test DataFrame creation with realistic default configuration."""
        # Create comprehensive test data matching expected default config
        realistic_exp_matrix = {
            't': np.linspace(0, 60, 3600),  # 1 minute at 60 Hz
            'x': np.random.rand(3600) * 50,  # X position in mm
            'y': np.random.rand(3600) * 50,  # Y position in mm
            'theta': np.random.rand(3600) * 2 * np.pi,  # Orientation in radians
            'vx': np.random.randn(3600) * 5,  # X velocity
            'vy': np.random.randn(3600) * 5,  # Y velocity
        }
        
        # Create minimal DataFrame without specifying config
        df = make_dataframe_from_config(realistic_exp_matrix)
        
        # Validate basic structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(realistic_exp_matrix['t'])
        
        # Validate required columns are present
        expected_columns = ['t', 'x', 'y', 'theta', 'vx', 'vy']
        for col in expected_columns:
            if col in realistic_exp_matrix:
                assert col in df.columns
    
    def test_make_dataframe_with_metadata_integration(self, sample_exp_matrix):
        """Test DataFrame creation with metadata integration."""
        config_with_metadata = ColumnConfigDict(
            columns={
                't': ColumnConfig(
                    type="numpy.ndarray",
                    dimension=1,
                    required=True,
                    description="Time values"
                ),
                'x': ColumnConfig(
                    type="numpy.ndarray",
                    dimension=1,
                    required=True,
                    description="X position"
                ),
                'experiment_date': ColumnConfig(
                    type="string",
                    required=False,
                    is_metadata=True,
                    description="Experiment date"
                ),
                'rig_id': ColumnConfig(
                    type="string",
                    required=False,
                    is_metadata=True,
                    description="Rig identifier"
                )
            }
        )
        
        metadata = {
            'experiment_date': '2024-12-20',
            'rig_id': 'opto_rig_1',
            'additional_info': 'Extra metadata'
        }
        
        df = make_dataframe_from_config(sample_exp_matrix, config_with_metadata, metadata=metadata)
        
        # Validate metadata integration
        assert isinstance(df, pd.DataFrame)
        assert 'experiment_date' in df.columns
        assert 'rig_id' in df.columns
        assert df['experiment_date'].iloc[0] == '2024-12-20'
        assert df['rig_id'].iloc[0] == 'opto_rig_1'
        
        # Non-configured metadata should not appear
        assert 'additional_info' not in df.columns
    
    def test_make_dataframe_alias_resolution_comprehensive(self):
        """Test comprehensive alias resolution scenarios."""
        exp_matrix_with_aliases = {
            't': np.linspace(0, 10, 100),
            'pos_x': np.random.rand(100),  # Alias for 'x'
            'pos_y': np.random.rand(100),  # Alias for 'y'
            'old_theta_name': np.random.rand(100)  # Alias for 'theta'
        }
        
        config_with_aliases = ColumnConfigDict(
            columns={
                't': ColumnConfig(
                    type="numpy.ndarray",
                    dimension=1,
                    required=True,
                    description="Time values"
                ),
                'x': ColumnConfig(
                    type="numpy.ndarray",
                    dimension=1,
                    required=True,
                    description="X position",
                    alias="pos_x"
                ),
                'y': ColumnConfig(
                    type="numpy.ndarray",
                    dimension=1,
                    required=True,
                    description="Y position",
                    alias="pos_y"
                ),
                'theta': ColumnConfig(
                    type="numpy.ndarray",
                    dimension=1,
                    required=False,
                    description="Orientation",
                    alias="old_theta_name"
                )
            }
        )
        
        df = make_dataframe_from_config(exp_matrix_with_aliases, config_with_aliases)
        
        # Validate alias resolution
        assert isinstance(df, pd.DataFrame)
        assert 'x' in df.columns  # Should use config name, not alias
        assert 'y' in df.columns
        assert 'theta' in df.columns
        assert 'pos_x' not in df.columns  # Alias shouldn't appear
        assert 'pos_y' not in df.columns
        assert 'old_theta_name' not in df.columns
        
        # Validate data is correctly mapped
        np.testing.assert_array_equal(df['x'].values, exp_matrix_with_aliases['pos_x'])
        np.testing.assert_array_equal(df['y'].values, exp_matrix_with_aliases['pos_y'])
        np.testing.assert_array_equal(df['theta'].values, exp_matrix_with_aliases['old_theta_name'])
    
    def test_make_dataframe_default_values_comprehensive(self):
        """Test comprehensive default value scenarios."""
        minimal_exp_matrix = {
            't': np.linspace(0, 10, 100),
            'x': np.random.rand(100)
        }
        
        config_with_defaults = ColumnConfigDict(
            columns={
                't': ColumnConfig(
                    type="numpy.ndarray",
                    dimension=1,
                    required=True,
                    description="Time values"
                ),
                'x': ColumnConfig(
                    type="numpy.ndarray",
                    dimension=1,
                    required=True,
                    description="X position"
                ),
                'y': ColumnConfig(
                    type="numpy.ndarray",
                    dimension=1,
                    required=False,
                    description="Y position",
                    default_value=np.zeros(100)
                ),
                'optional_scalar': ColumnConfig(
                    type="float",
                    required=False,
                    description="Optional scalar value",
                    default_value=42.0
                ),
                'optional_none': ColumnConfig(
                    type="numpy.ndarray",
                    dimension=1,
                    required=False,
                    description="Optional with None default",
                    default_value=None
                )
            }
        )
        
        df = make_dataframe_from_config(minimal_exp_matrix, config_with_defaults)
        
        # Validate default value application
        assert isinstance(df, pd.DataFrame)
        assert 'y' in df.columns
        assert 'optional_scalar' in df.columns
        assert 'optional_none' in df.columns
        
        # Check default values are applied correctly
        np.testing.assert_array_equal(df['y'].values, np.zeros(100))
        assert df['optional_scalar'].iloc[0] == 42.0
        assert pd.isna(df['optional_none'].iloc[0]) or df['optional_none'].iloc[0] is None
    
    def test_make_dataframe_missing_required_columns_error(self):
        """Test error handling when required columns are missing."""
        incomplete_exp_matrix = {
            't': np.linspace(0, 10, 100)
            # Missing required 'x' and 'y' columns
        }
        
        config_with_required = ColumnConfigDict(
            columns={
                't': ColumnConfig(
                    type="numpy.ndarray",
                    dimension=1,
                    required=True,
                    description="Time values"
                ),
                'x': ColumnConfig(
                    type="numpy.ndarray",
                    dimension=1,
                    required=True,
                    description="X position"
                ),
                'y': ColumnConfig(
                    type="numpy.ndarray",
                    dimension=1,
                    required=True,
                    description="Y position"
                )
            }
        )
        
        with pytest.raises(ValueError) as exc_info:
            make_dataframe_from_config(incomplete_exp_matrix, config_with_required)
        
        error_message = str(exc_info.value)
        assert "Missing required columns" in error_message
        assert "x" in error_message
        assert "y" in error_message


# =============================================================================
# EDGE CASE AND ERROR HANDLING TESTS
# =============================================================================

class TestEdgeCasesAndErrorHandling:
    """Comprehensive edge case and error handling tests."""
    
    def test_column_config_extreme_values(self):
        """Test ColumnConfig with extreme but valid values."""
        # Very long description
        long_description = "A" * 10000
        config = ColumnConfig(
            type="numpy.ndarray",
            dimension=3,
            required=False,
            description=long_description,
            alias="very_long_alias_name_that_exceeds_normal_length"
        )
        
        assert config.description == long_description
        assert config.alias == "very_long_alias_name_that_exceeds_normal_length"
        assert config.dimension == ColumnDimension.THREE_D
    
    def test_column_config_dict_empty_scenarios(self):
        """Test ColumnConfigDict with various empty scenarios."""
        # Completely empty
        empty_config = ColumnConfigDict(columns={})
        assert empty_config.columns == {}
        assert empty_config.special_handlers == {}
        
        # Only special handlers, no columns
        handlers_only = ColumnConfigDict(
            columns={},
            special_handlers={'handler1': 'func1'}
        )
        assert handlers_only.columns == {}
        assert handlers_only.special_handlers == {'handler1': 'func1'}
    
    def test_yaml_loading_edge_cases(self, tmp_path):
        """Test YAML loading with various edge cases."""
        # Empty YAML file
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("")
        
        with pytest.raises((ValidationError, TypeError)):
            load_column_config(str(empty_yaml))
        
        # YAML with null values
        null_yaml = tmp_path / "null.yaml"
        null_yaml.write_text("columns: null\nspecial_handlers: null")
        
        with pytest.raises((ValidationError, TypeError)):
            load_column_config(str(null_yaml))
        
        # YAML with only comments
        comments_yaml = tmp_path / "comments.yaml"
        comments_yaml.write_text("# This is just a comment\n# Another comment")
        
        with pytest.raises((ValidationError, TypeError)):
            load_column_config(str(comments_yaml))
    
    def test_dataframe_creation_edge_cases(self):
        """Test DataFrame creation with edge case data."""
        # Single time point
        single_point_matrix = {
            't': np.array([0.0]),
            'x': np.array([10.0]),
            'y': np.array([20.0])
        }
        
        simple_config = ColumnConfigDict(
            columns={
                't': ColumnConfig(type="numpy.ndarray", dimension=1, required=True, description="Time"),
                'x': ColumnConfig(type="numpy.ndarray", dimension=1, required=True, description="X"),
                'y': ColumnConfig(type="numpy.ndarray", dimension=1, required=True, description="Y")
            }
        )
        
        df = make_dataframe_from_config(single_point_matrix, simple_config)
        assert len(df) == 1
        assert df['t'].iloc[0] == 0.0
        assert df['x'].iloc[0] == 10.0
        assert df['y'].iloc[0] == 20.0
        
        # Empty arrays (should handle gracefully or raise appropriate error)
        empty_matrix = {
            't': np.array([]),
            'x': np.array([]),
            'y': np.array([])
        }
        
        df_empty = make_dataframe_from_config(empty_matrix, simple_config)
        assert len(df_empty) == 0
        assert list(df_empty.columns) == ['t', 'x', 'y']
    
    @pytest.mark.parametrize("problematic_data", [
        {'t': np.array([np.inf, 1, 2])},  # Infinity values
        {'t': np.array([np.nan, 1, 2])},  # NaN values
        {'t': np.array([1e100, 1, 2])},   # Very large values
        {'t': np.array([1e-100, 1, 2])}, # Very small values
    ])
    def test_dataframe_creation_with_problematic_numerical_data(self, problematic_data):
        """Test DataFrame creation with numerically problematic data."""
        simple_config = ColumnConfigDict(
            columns={
                't': ColumnConfig(type="numpy.ndarray", dimension=1, required=True, description="Time")
            }
        )
        
        # Should create DataFrame but preserve the problematic values
        df = make_dataframe_from_config(problematic_data, simple_config)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        
        # Values should be preserved as-is
        np.testing.assert_array_equal(df['t'].values, problematic_data['t'])


# =============================================================================
# INTEGRATION WITH EXISTING FIXTURES
# =============================================================================

class TestIntegrationWithExistingFixtures:
    """Tests that integrate with fixtures from conftest.py"""
    
    def test_with_sample_column_config_file(self, sample_column_config_file):
        """Test using the sample_column_config_file fixture from conftest.py."""
        config = load_column_config(sample_column_config_file)
        
        assert isinstance(config, ColumnConfigDict)
        assert 't' in config.columns
        assert 'x' in config.columns
        assert 'y' in config.columns
        assert 'signal_disp' in config.columns
        
        # Validate specific configurations from fixture
        assert config.columns['signal_disp'].special_handling == SpecialHandlerType.TRANSFORM_TIME_DIMENSION
        assert config.columns['date'].is_metadata is True
    
    def test_with_sample_exp_matrix_fixtures(self, sample_exp_matrix, sample_exp_matrix_with_signal_disp, sample_exp_matrix_with_aliases):
        """Test using various exp_matrix fixtures from conftest.py."""
        simple_config = ColumnConfigDict(
            columns={
                't': ColumnConfig(type="numpy.ndarray", dimension=1, required=True, description="Time"),
                'x': ColumnConfig(type="numpy.ndarray", dimension=1, required=True, description="X"),
                'y': ColumnConfig(type="numpy.ndarray", dimension=1, required=True, description="Y"),
                'signal_disp': ColumnConfig(
                    type="numpy.ndarray", 
                    dimension=2, 
                    required=False, 
                    description="Signal display",
                    special_handling=SpecialHandlerType.TRANSFORM_TIME_DIMENSION
                ),
                'dtheta': ColumnConfig(
                    type="numpy.ndarray",
                    dimension=1,
                    required=False,
                    description="Delta theta",
                    alias="dtheta_smooth"
                )
            },
            special_handlers={
                'transform_to_match_time_dimension': '_handle_signal_disp'
            }
        )
        
        # Test with basic matrix
        df1 = make_dataframe_from_config(sample_exp_matrix, simple_config)
        assert isinstance(df1, pd.DataFrame)
        assert 't' in df1.columns
        assert 'x' in df1.columns
        assert 'y' in df1.columns
        
        # Test with signal_disp matrix
        df2 = make_dataframe_from_config(sample_exp_matrix_with_signal_disp, simple_config)
        assert isinstance(df2, pd.DataFrame)
        assert 'signal_disp' in df2.columns
        
        # Test with aliases matrix
        df3 = make_dataframe_from_config(sample_exp_matrix_with_aliases, simple_config)
        assert isinstance(df3, pd.DataFrame)
        if 'dtheta_smooth' in sample_exp_matrix_with_aliases:
            assert 'dtheta' in df3.columns  # Should be renamed from alias
