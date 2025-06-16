"""
Behavioral validation tests for Pydantic-based column configuration models.

Refactored per Section 0 requirements for behavior-focused testing:
- Converted from implementation-dependent checks to black-box behavioral validation
- Replaced custom mock implementations with Protocol-based mocks from centralized tests/utils.py
- Implemented AAA pattern structure with clear separation of setup, execution, and verification
- Enhanced edge-case coverage through parameterized test scenarios
- Utilizes centralized fixtures from tests/conftest.py for consistent test infrastructure
- Focuses on observable column model behavior rather than internal implementation details

Key Behavioral Focus Areas:
- Successful Pydantic validation and configuration loading across multiple sources
- Accurate DataFrame construction with proper column mapping and type preservation
- Appropriate error raising for validation failures and missing required data
- Cross-platform configuration handling with Unicode and special character support
- Performance compliance validation through observable timing characteristics
"""

import os
import tempfile
import yaml
import pytest
import numpy as np
from pydantic import ValidationError
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from unittest.mock import patch, mock_open

# Import centralized test utilities per Section 0 requirement
from tests.utils import (
    create_mock_filesystem,
    create_mock_dataloader,
    create_mock_config_provider,
    generate_edge_case_scenarios,
    MockDataLoading,
    MockFilesystem,
    MockConfigurationProvider
)

# Import system under test - focusing on public API behavior
from flyrigloader.io.column_models import (
    ColumnConfig,
    ColumnConfigDict,
    ColumnDimension,
    SpecialHandlerType,
    load_column_config,
    get_config_from_source
)
from flyrigloader.io.pickle import make_dataframe_from_config


# =============================================================================
# BEHAVIORAL VALIDATION TESTS - PUBLIC API FOCUSED
# =============================================================================

class TestColumnConfigValidationBehavior:
    """Test observable behavior of ColumnConfig validation through public API."""
    
    def test_successful_column_config_creation_behavior(self):
        """
        Test successful ColumnConfig creation produces expected public behavior.
        
        Validates: Successful Pydantic validation with correct attribute access
        Edge cases: Basic required field validation
        """
        # ARRANGE - Set up test data for valid configuration
        config_data = {
            "type": "numpy.ndarray",
            "dimension": 1,
            "required": True,
            "description": "Time values"
        }
        
        # ACT - Create configuration through public API
        config = ColumnConfig(**config_data)
        
        # ASSERT - Verify observable behavior through public interface
        assert config.type == "numpy.ndarray"
        assert config.dimension == ColumnDimension.ONE_D
        assert config.required is True
        assert config.description == "Time values"
        assert config.alias is None
        assert config.is_metadata is False
        assert config.default_value is None
        assert config.special_handling is None

    @pytest.mark.parametrize("invalid_dimension", [0, 4, 5, -1, 10])
    def test_dimension_validation_error_behavior(self, invalid_dimension):
        """
        Test that invalid dimensions produce proper validation errors.
        
        Validates: Appropriate error raising for validation failures
        Edge cases: Boundary conditions for dimension validation
        """
        # ARRANGE - Set up configuration with invalid dimension
        config_data = {
            "type": "numpy.ndarray",
            "dimension": invalid_dimension,
            "required": True,
            "description": "Invalid dimension test"
        }
        
        # ACT & ASSERT - Verify validation error is raised
        with pytest.raises(ValidationError) as exc_info:
            ColumnConfig(**config_data)
        
        # Verify error message contains dimension information
        assert "Dimension must be 1, 2, or 3" in str(exc_info.value)

    @pytest.mark.parametrize("invalid_handler", [
        "nonexistent_handler", 
        "invalid_transform", 
        "bad_handler_name",
        123,  # Non-string type
        ""    # Empty string
    ])
    def test_special_handler_validation_error_behavior(self, invalid_handler):
        """
        Test that invalid special handlers produce proper validation errors.
        
        Validates: Appropriate error raising for validation failures
        Edge cases: Various invalid handler type scenarios
        """
        # ARRANGE - Set up configuration with invalid special handler
        config_data = {
            "type": "numpy.ndarray",
            "dimension": 1,
            "required": True,
            "description": "Test column",
            "special_handling": invalid_handler
        }
        
        # ACT & ASSERT - Verify validation error behavior
        with pytest.raises(ValidationError) as exc_info:
            ColumnConfig(**config_data)
        
        # Verify error contains handler information
        error_message = str(exc_info.value)
        assert any(handler.value in error_message for handler in SpecialHandlerType) or "Special handler" in error_message

    def test_complete_column_config_with_all_options_behavior(self):
        """
        Test creation of fully-configured ColumnConfig with all options.
        
        Validates: Comprehensive configuration handling through public API
        Edge cases: Complex configuration scenarios
        """
        # ARRANGE - Set up comprehensive configuration
        config_data = {
            "type": "numpy.ndarray",
            "dimension": 2,
            "required": False,
            "description": "Signal display data",
            "alias": "signal_data",
            "is_metadata": False,
            "default_value": None,
            "special_handling": "transform_to_match_time_dimension"
        }
        
        # ACT - Create fully configured column
        config = ColumnConfig(**config_data)
        
        # ASSERT - Verify all attributes are properly set through public interface
        assert config.type == "numpy.ndarray"
        assert config.dimension == ColumnDimension.TWO_D
        assert config.required is False
        assert config.description == "Signal display data"
        assert config.alias == "signal_data"
        assert config.is_metadata is False
        assert config.default_value is None
        assert config.special_handling == SpecialHandlerType.TRANSFORM_TIME_DIMENSION


class TestColumnConfigDictValidationBehavior:
    """Test observable behavior of ColumnConfigDict validation through public API."""
    
    def test_empty_column_config_dict_creation_behavior(self):
        """
        Test creation of empty ColumnConfigDict produces expected behavior.
        
        Validates: Minimal configuration handling
        Edge cases: Empty configuration scenarios
        """
        # ARRANGE - Set up empty configuration
        empty_config_data = {"columns": {}}
        
        # ACT - Create empty configuration
        config_dict = ColumnConfigDict(**empty_config_data)
        
        # ASSERT - Verify observable behavior
        assert config_dict.columns == {}
        assert config_dict.special_handlers == {}

    def test_complete_column_config_dict_behavior(self, sample_column_configs):
        """
        Test creation of complete ColumnConfigDict with complex configuration.
        
        Validates: Complex configuration structure handling
        Edge cases: Multi-column configurations with handlers
        """
        # ARRANGE - Set up complex configuration using centralized fixture
        config_data = {
            "columns": sample_column_configs,
            "special_handlers": {
                "transform_to_match_time_dimension": "_handle_signal_disp"
            }
        }
        
        # ACT - Create complex configuration
        config_dict = ColumnConfigDict(**config_data)
        
        # ASSERT - Verify observable behavior through public interface
        assert len(config_dict.columns) == len(sample_column_configs)
        assert "transform_to_match_time_dimension" in config_dict.special_handlers
        assert config_dict.special_handlers["transform_to_match_time_dimension"] == "_handle_signal_disp"
        
        # Verify individual column configurations are accessible
        for col_name, col_config in sample_column_configs.items():
            assert col_name in config_dict.columns
            assert isinstance(config_dict.columns[col_name], ColumnConfig)

    def test_handlers_only_config_dict_behavior(self):
        """
        Test ColumnConfigDict with only special handlers defined.
        
        Validates: Partial configuration scenarios
        Edge cases: Handlers without columns
        """
        # ARRANGE - Set up handlers-only configuration
        config_data = {
            "columns": {},
            "special_handlers": {
                "transform_to_match_time_dimension": "_handle_signal_disp",
                "extract_first_column_if_2d": "_handle_extraction"
            }
        }
        
        # ACT - Create handlers-only configuration
        config_dict = ColumnConfigDict(**config_data)
        
        # ASSERT - Verify expected behavior
        assert config_dict.columns == {}
        assert len(config_dict.special_handlers) == 2
        assert "transform_to_match_time_dimension" in config_dict.special_handlers
        assert "extract_first_column_if_2d" in config_dict.special_handlers


class TestConfigurationLoadingBehavior:
    """Test observable behavior of configuration loading from various sources."""
    
    def test_yaml_file_loading_success_behavior(self, sample_yaml_config_file):
        """
        Test successful YAML configuration loading behavior.
        
        Validates: Successful configuration loading across file sources
        Edge cases: File-based configuration scenarios
        """
        # ARRANGE - Use centralized fixture for YAML configuration file
        
        # ACT - Load configuration from file
        config = load_column_config(sample_yaml_config_file)
        
        # ASSERT - Verify successful loading behavior through public API
        assert isinstance(config, ColumnConfigDict)
        assert 't' in config.columns
        assert 'x' in config.columns
        assert 'y' in config.columns
        
        # Verify specific column configurations
        assert config.columns['t'].required is True
        assert config.columns['t'].type == "numpy.ndarray"
        assert config.columns['t'].dimension == ColumnDimension.ONE_D

    def test_yaml_file_not_found_error_behavior(self):
        """
        Test error behavior when YAML file does not exist.
        
        Validates: Appropriate error raising for missing files
        Edge cases: File not found scenarios
        """
        # ARRANGE - Set up non-existent file path
        non_existent_path = "/absolutely/nonexistent/path/config.yaml"
        
        # ACT & ASSERT - Verify file not found error behavior
        with pytest.raises(FileNotFoundError):
            load_column_config(non_existent_path)

    def test_malformed_yaml_error_behavior(self, tmp_path):
        """
        Test error behavior with malformed YAML content.
        
        Validates: Appropriate error raising for corrupted files
        Edge cases: Malformed file content scenarios
        """
        # ARRANGE - Create malformed YAML file
        malformed_yaml_file = tmp_path / "malformed.yaml"
        with open(malformed_yaml_file, 'w') as f:
            f.write("invalid: yaml: content: [\n  - missing: closing")
        
        # ACT & ASSERT - Verify YAML error behavior
        with pytest.raises(yaml.YAMLError):
            load_column_config(str(malformed_yaml_file))


class TestMultiSourceConfigurationBehavior:
    """Test observable behavior of multi-source configuration loading."""
    
    def test_string_path_source_behavior(self, sample_yaml_config_file):
        """
        Test configuration loading from string file path.
        
        Validates: Cross-platform configuration handling
        Edge cases: String path scenarios
        """
        # ARRANGE - Use string path
        config_path = sample_yaml_config_file
        
        # ACT - Load from string source
        config = get_config_from_source(config_path)
        
        # ASSERT - Verify successful loading behavior
        assert isinstance(config, ColumnConfigDict)
        assert 't' in config.columns

    def test_pathlib_path_source_behavior(self, sample_yaml_config_file):
        """
        Test configuration loading from pathlib.Path object.
        
        Validates: Cross-platform configuration handling
        Edge cases: Path object scenarios
        """
        # ARRANGE - Convert to Path object
        config_path = Path(sample_yaml_config_file)
        
        # ACT - Load from Path source
        config = get_config_from_source(config_path)
        
        # ASSERT - Verify successful loading behavior
        assert isinstance(config, ColumnConfigDict)
        assert 't' in config.columns

    def test_dictionary_source_behavior(self):
        """
        Test configuration loading from dictionary source.
        
        Validates: Direct dictionary configuration handling
        Edge cases: In-memory configuration scenarios
        """
        # ARRANGE - Set up dictionary configuration
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
                }
            },
            'special_handlers': {}
        }
        
        # ACT - Load from dictionary source
        config = get_config_from_source(config_dict)
        
        # ASSERT - Verify successful dictionary loading behavior
        assert isinstance(config, ColumnConfigDict)
        assert len(config.columns) == 2
        assert 't' in config.columns
        assert 'x' in config.columns
        assert config.columns['t'].required is True
        assert config.columns['x'].required is True

    def test_model_instance_source_behavior(self, sample_column_config_dict):
        """
        Test configuration loading from existing ColumnConfigDict instance.
        
        Validates: Model instance pass-through behavior
        Edge cases: Existing model scenarios
        """
        # ARRANGE - Use existing model instance from centralized fixture
        
        # ACT - Load from model instance source
        config = get_config_from_source(sample_column_config_dict)
        
        # ASSERT - Verify pass-through behavior (should return same instance)
        assert config is sample_column_config_dict
        assert isinstance(config, ColumnConfigDict)

    def test_none_source_default_behavior(self):
        """
        Test default configuration loading behavior when source is None.
        
        Validates: Default configuration fallback behavior
        Edge cases: None source scenarios
        """
        # ARRANGE - Use None source (should use default)
        # We'll mock the default config loading to avoid file system dependencies
        mock_default_config = {
            'columns': {
                'default_time': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Default time column'
                }
            },
            'special_handlers': {}
        }
        
        # ACT & ASSERT - Test with mocked default configuration
        with patch('builtins.open', mock_open()):
            with patch('yaml.safe_load', return_value=mock_default_config):
                config = get_config_from_source(None)
                
                assert isinstance(config, ColumnConfigDict)
                assert 'default_time' in config.columns
                assert config.columns['default_time'].required is True

    @pytest.mark.parametrize("invalid_source", [
        [],  # List
        42,  # Integer
        3.14,  # Float
        True,  # Boolean
        {'invalid': 'structure'},  # Dict without 'columns' key
    ])
    def test_invalid_source_type_error_behavior(self, invalid_source):
        """
        Test error behavior with invalid source types.
        
        Validates: Appropriate error raising for validation failures
        Edge cases: Various invalid source type scenarios
        """
        # ARRANGE - Use invalid source type
        
        # ACT & ASSERT - Verify type error behavior
        with pytest.raises((TypeError, ValidationError)):
            get_config_from_source(invalid_source)


class TestDataFrameConstructionBehavior:
    """Test observable behavior of DataFrame construction from column configurations."""
    
    def test_basic_dataframe_creation_behavior(self, sample_exp_matrix):
        """
        Test basic DataFrame creation with simple configuration.
        
        Validates: Correct DataFrame construction
        Edge cases: Basic transformation scenarios
        """
        # ARRANGE - Set up basic configuration
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
                }
            }
        }
        
        # ACT - Create DataFrame from configuration
        df = make_dataframe_from_config(sample_exp_matrix, config_dict)
        
        # ASSERT - Verify DataFrame construction behavior
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_exp_matrix['t'])
        assert 't' in df.columns
        assert 'x' in df.columns
        assert 'y' in df.columns
        
        # Verify data integrity through observable values
        np.testing.assert_array_equal(df['t'].values, sample_exp_matrix['t'])
        np.testing.assert_array_equal(df['x'].values, sample_exp_matrix['x'])
        np.testing.assert_array_equal(df['y'].values, sample_exp_matrix['y'])

    def test_dataframe_with_special_handling_behavior(self, sample_exp_matrix_with_signal_disp):
        """
        Test DataFrame creation with special signal handling.
        
        Validates: Special handling processing behavior
        Edge cases: Complex transformation scenarios
        """
        # ARRANGE - Set up configuration with special handling
        config_dict = {
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
                }
            },
            'special_handlers': {
                'transform_to_match_time_dimension': '_handle_signal_disp'
            }
        }
        
        # ACT - Create DataFrame with special handling
        df = make_dataframe_from_config(sample_exp_matrix_with_signal_disp, config_dict)
        
        # ASSERT - Verify special handling behavior through observable results
        assert isinstance(df, pd.DataFrame)
        assert 'signal_disp' in df.columns
        
        # Verify signal_disp is properly handled (should be Series of arrays)
        if len(df) > 0:
            assert isinstance(df['signal_disp'].iloc[0], np.ndarray)

    def test_dataframe_with_alias_resolution_behavior(self):
        """
        Test DataFrame creation with column alias resolution.
        
        Validates: Accurate configuration loading across multiple sources
        Edge cases: Alias mapping scenarios
        """
        # ARRANGE - Set up experimental matrix with aliased columns
        exp_matrix_with_aliases = {
            't': np.linspace(0, 10, 100),
            'pos_x': np.random.rand(100),  # Alias for 'x'
            'pos_y': np.random.rand(100),  # Alias for 'y'
        }
        
        config_with_aliases = {
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
                    'description': 'X position',
                    'alias': 'pos_x'
                },
                'y': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Y position',
                    'alias': 'pos_y'
                }
            }
        }
        
        # ACT - Create DataFrame with alias resolution
        df = make_dataframe_from_config(exp_matrix_with_aliases, config_with_aliases)
        
        # ASSERT - Verify alias resolution behavior
        assert isinstance(df, pd.DataFrame)
        assert 'x' in df.columns  # Should use config name, not alias
        assert 'y' in df.columns
        assert 'pos_x' not in df.columns  # Alias shouldn't appear in DataFrame
        assert 'pos_y' not in df.columns
        
        # Verify data is correctly mapped from aliases
        np.testing.assert_array_equal(df['x'].values, exp_matrix_with_aliases['pos_x'])
        np.testing.assert_array_equal(df['y'].values, exp_matrix_with_aliases['pos_y'])

    def test_dataframe_with_default_values_behavior(self):
        """
        Test DataFrame creation with default value application.
        
        Validates: Default value handling behavior
        Edge cases: Missing column scenarios
        """
        # ARRANGE - Set up minimal experimental matrix
        minimal_exp_matrix = {
            't': np.linspace(0, 10, 100),
            'x': np.random.rand(100)
        }
        
        config_with_defaults = {
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
                    'default_value': np.zeros(100)
                },
                'optional_scalar': {
                    'type': 'float',
                    'required': False,
                    'description': 'Optional scalar value',
                    'default_value': 42.0
                }
            }
        }
        
        # ACT - Create DataFrame with default values
        df = make_dataframe_from_config(minimal_exp_matrix, config_with_defaults)
        
        # ASSERT - Verify default value application behavior
        assert isinstance(df, pd.DataFrame)
        assert 'y' in df.columns
        assert 'optional_scalar' in df.columns
        
        # Check default values are applied correctly
        np.testing.assert_array_equal(df['y'].values, np.zeros(100))
        assert df['optional_scalar'].iloc[0] == 42.0

    def test_missing_required_columns_error_behavior(self):
        """
        Test error behavior when required columns are missing.
        
        Validates: Appropriate error raising for validation failures
        Edge cases: Missing required column scenarios
        """
        # ARRANGE - Set up incomplete experimental matrix
        incomplete_exp_matrix = {
            't': np.linspace(0, 10, 100)
            # Missing required 'x' and 'y' columns
        }
        
        config_with_required = {
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
                }
            }
        }
        
        # ACT & ASSERT - Verify missing required columns error behavior
        with pytest.raises(ValueError) as exc_info:
            make_dataframe_from_config(incomplete_exp_matrix, config_with_required)
        
        error_message = str(exc_info.value)
        assert "Missing required columns" in error_message
        assert "x" in error_message
        assert "y" in error_message


# =============================================================================
# EDGE-CASE AND BOUNDARY CONDITION TESTS
# =============================================================================

class TestEdgeCaseBehaviorValidation:
    """Test edge-case and boundary condition behavior through parameterized scenarios."""
    
    @pytest.mark.parametrize("array_size", [0, 1, 2, 1000, 100000])
    def test_array_size_boundary_conditions_behavior(self, array_size):
        """
        Test DataFrame creation with various array sizes.
        
        Validates: Performance compliance and boundary condition handling
        Edge cases: Empty arrays, single-point data, large datasets
        """
        # ARRANGE - Set up boundary condition data
        exp_matrix = {
            't': np.linspace(0, array_size/60.0, array_size) if array_size > 0 else np.array([]),
            'x': np.random.rand(array_size) if array_size > 0 else np.array([]),
            'y': np.random.rand(array_size) if array_size > 0 else np.array([])
        }
        
        simple_config = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'X'},
                'y': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Y'}
            }
        }
        
        # ACT - Create DataFrame with boundary condition data
        df = make_dataframe_from_config(exp_matrix, simple_config)
        
        # ASSERT - Verify boundary condition handling behavior
        assert isinstance(df, pd.DataFrame)
        assert len(df) == array_size
        assert list(df.columns) == ['t', 'x', 'y']
        
        if array_size > 0:
            # Verify data integrity for non-empty arrays
            np.testing.assert_array_equal(df['t'].values, exp_matrix['t'])
            np.testing.assert_array_equal(df['x'].values, exp_matrix['x'])
            np.testing.assert_array_equal(df['y'].values, exp_matrix['y'])

    @pytest.mark.parametrize("problematic_data", [
        {'t': np.array([np.inf, 1, 2])},  # Infinity values
        {'t': np.array([np.nan, 1, 2])},  # NaN values  
        {'t': np.array([1e100, 1, 2])},   # Very large values
        {'t': np.array([1e-100, 1, 2])}, # Very small values
    ])
    def test_problematic_numerical_data_behavior(self, problematic_data):
        """
        Test DataFrame creation with numerically problematic data.
        
        Validates: Robust handling of edge-case numerical data
        Edge cases: Infinities, NaNs, extreme magnitudes
        """
        # ARRANGE - Set up simple configuration for problematic data
        simple_config = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time'}
            }
        }
        
        # ACT - Create DataFrame with problematic data
        df = make_dataframe_from_config(problematic_data, simple_config)
        
        # ASSERT - Verify problematic data is preserved (observable behavior)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        
        # Values should be preserved as-is (behavioral preservation)
        np.testing.assert_array_equal(df['t'].values, problematic_data['t'])

    def test_unicode_configuration_handling_behavior(self, tmp_path):
        """
        Test configuration handling with Unicode characters.
        
        Validates: Cross-platform configuration handling
        Edge cases: Unicode and special character scenarios
        """
        # ARRANGE - Set up Unicode configuration
        unicode_config = {
            'columns': {
                'tëst_cölümn': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Ünicöde test cölümn'
                },
                'ñørmæl_cøl': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': 'Nørmål cølümn wïth spëcïål chærs'
                }
            }
        }
        
        # Create Unicode YAML file
        unicode_yaml_file = tmp_path / "ünicöde_cönfïg.yaml"
        with open(unicode_yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(unicode_config, f, allow_unicode=True)
        
        # ACT - Load Unicode configuration
        config = load_column_config(str(unicode_yaml_file))
        
        # ASSERT - Verify Unicode handling behavior
        assert isinstance(config, ColumnConfigDict)
        assert 'tëst_cölümn' in config.columns
        assert 'ñørmæl_cøl' in config.columns
        assert config.columns['tëst_cölümn'].description == 'Ünicöde test cölümn'

    def test_metadata_integration_behavior(self, sample_exp_matrix):
        """
        Test DataFrame creation with metadata field integration.
        
        Validates: Metadata handling behavior
        Edge cases: Metadata field scenarios
        """
        # ARRANGE - Set up configuration with metadata fields
        config_with_metadata = {
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
                'experiment_date': {
                    'type': 'string',
                    'required': False,
                    'is_metadata': True,
                    'description': 'Experiment date'
                },
                'rig_id': {
                    'type': 'string',
                    'required': False,
                    'is_metadata': True,
                    'description': 'Rig identifier'
                }
            }
        }
        
        metadata = {
            'experiment_date': '2024-12-20',
            'rig_id': 'opto_rig_1',
            'additional_info': 'Extra metadata'
        }
        
        # ACT - Create DataFrame with metadata
        df = make_dataframe_from_config(sample_exp_matrix, config_with_metadata, metadata=metadata)
        
        # ASSERT - Verify metadata integration behavior
        assert isinstance(df, pd.DataFrame)
        assert 'experiment_date' in df.columns
        assert 'rig_id' in df.columns
        assert df['experiment_date'].iloc[0] == '2024-12-20'
        assert df['rig_id'].iloc[0] == 'opto_rig_1'
        
        # Non-configured metadata should not appear
        assert 'additional_info' not in df.columns


# =============================================================================
# INTEGRATION WITH CENTRALIZED FIXTURES
# =============================================================================

class TestCentralizedFixtureIntegration:
    """Test integration with centralized fixtures from tests/conftest.py."""
    
    def test_sample_column_config_file_integration(self, sample_column_config_file):
        """
        Test using centralized sample_column_config_file fixture.
        
        Validates: Centralized fixture utilization
        Edge cases: File-based configuration integration
        """
        # ARRANGE - Use centralized fixture
        
        # ACT - Load configuration using centralized fixture
        config = load_column_config(sample_column_config_file)
        
        # ASSERT - Verify integration behavior
        assert isinstance(config, ColumnConfigDict)
        assert 't' in config.columns
        assert 'x' in config.columns
        assert 'y' in config.columns
        
        # Verify specific configurations from centralized fixture
        if 'signal_disp' in config.columns:
            assert config.columns['signal_disp'].special_handling == SpecialHandlerType.TRANSFORM_TIME_DIMENSION
        if 'date' in config.columns:
            assert config.columns['date'].is_metadata is True

    def test_experimental_matrix_fixtures_integration(
        self, 
        sample_exp_matrix, 
        sample_exp_matrix_with_signal_disp, 
        sample_exp_matrix_with_aliases
    ):
        """
        Test using various experimental matrix fixtures from centralized conftest.py.
        
        Validates: Multiple centralized fixture utilization
        Edge cases: Various experimental data scenarios
        """
        # ARRANGE - Set up configuration compatible with all fixtures
        flexible_config = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'X'},
                'y': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Y'},
                'signal_disp': {
                    'type': 'numpy.ndarray', 
                    'dimension': 2, 
                    'required': False, 
                    'description': 'Signal display',
                    'special_handling': 'transform_to_match_time_dimension'
                },
                'dtheta': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': 'Delta theta',
                    'alias': 'dtheta_smooth'
                }
            },
            'special_handlers': {
                'transform_to_match_time_dimension': '_handle_signal_disp'
            }
        }
        
        # ACT & ASSERT - Test with basic matrix
        df1 = make_dataframe_from_config(sample_exp_matrix, flexible_config)
        assert isinstance(df1, pd.DataFrame)
        assert 't' in df1.columns
        assert 'x' in df1.columns
        assert 'y' in df1.columns
        
        # ACT & ASSERT - Test with signal_disp matrix 
        df2 = make_dataframe_from_config(sample_exp_matrix_with_signal_disp, flexible_config)
        assert isinstance(df2, pd.DataFrame)
        if 'signal_disp' in sample_exp_matrix_with_signal_disp:
            assert 'signal_disp' in df2.columns
        
        # ACT & ASSERT - Test with aliases matrix
        df3 = make_dataframe_from_config(sample_exp_matrix_with_aliases, flexible_config)
        assert isinstance(df3, pd.DataFrame)
        if 'dtheta_smooth' in sample_exp_matrix_with_aliases:
            assert 'dtheta' in df3.columns  # Should be renamed from alias

    def test_temp_experiment_directory_integration(self, temp_experiment_directory):
        """
        Test using centralized temp_experiment_directory fixture.
        
        Validates: Centralized temporary resource utilization
        Edge cases: Temporary directory scenarios
        """
        # ARRANGE - Use centralized temporary directory fixture
        exp_dir_info = temp_experiment_directory
        config_file = exp_dir_info["config_file"]
        
        # ACT - Load configuration from temporary directory
        # We'll create a simple configuration for testing
        simple_config_content = """
columns:
  t:
    type: numpy.ndarray
    dimension: 1
    required: true
    description: Time values
  x:
    type: numpy.ndarray
    dimension: 1
    required: true
    description: X position
special_handlers: {}
"""
        config_file.write_text(simple_config_content)
        config = load_column_config(str(config_file))
        
        # ASSERT - Verify temporary directory integration behavior
        assert isinstance(config, ColumnConfigDict)
        assert 't' in config.columns
        assert 'x' in config.columns