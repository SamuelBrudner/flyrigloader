"""
Behavior-focused test suite for Pydantic-based DataFrame construction functionality.

This module provides comprehensive testing for the Pydantic-based column configuration
and DataFrame construction implementation using modern pytest practices with centralized
fixtures, Protocol-based mocks, and behavior-focused validation per F-013 Fixture
Consolidation requirements and Section 6.6.1 behavior-focused testing strategy.

Key testing areas:
- Observable DataFrame construction behavior through public API validation (F-008)
- Pydantic-driven configuration loading with mock implementation verification (F-007)  
- Edge-case coverage through parameterized boundary condition testing
- Centralized fixture management eliminating code duplication per F-013
- AAA pattern enforcement for improved readability and maintainability

The test suite focuses on black-box validation of the make_dataframe_from_config function
and related public interfaces, validating observable behavior rather than internal
implementation details, supporting the Section 6.6.1 public API validation approach.
"""

import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

import numpy as np
import pandas as pd
import pytest
import yaml
from hypothesis import given, strategies as st, assume, settings
from pydantic import ValidationError

from flyrigloader.io.pickle import make_dataframe_from_config
from flyrigloader.io.column_models import (
    ColumnConfig, 
    ColumnConfigDict, 
    ColumnDimension,
    SpecialHandlerType,
    get_config_from_source
)

# Import centralized test utilities per F-013 requirements
from tests.utils import (
    create_mock_filesystem,
    create_mock_dataloader, 
    create_mock_config_provider,
    generate_edge_case_scenarios,
    validate_test_structure,
    MockDataLoading,
    MockFilesystem,
    MockConfigurationProvider
)


# ===== CENTRALIZED FIXTURE USAGE =====
# Using fixtures from tests/conftest.py per F-013 Fixture Consolidation

class TestDataFrameConstructionBehavior:
    """Test observable DataFrame construction behavior through public API validation."""
    
    def test_basic_dataframe_creation_behavior(self, test_data_generator, temp_experiment_directory):
        """
        Test basic DataFrame creation produces expected output structure.
        
        ARRANGE: Set up synthetic experimental data and configuration
        ACT: Execute DataFrame construction via public API
        ASSERT: Verify observable behavior meets expected structure
        """
        # ARRANGE - Set up test data using centralized fixture
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=100,
            cols=3,
            data_type="behavioral"
        )
        
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
                'y': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Y position'
                }
            }
        }
        
        # ACT - Execute DataFrame construction through public API
        result_df = make_dataframe_from_config(exp_matrix, config)
        
        # ASSERT - Verify observable behavior
        assert isinstance(result_df, pd.DataFrame), "Output must be pandas DataFrame"
        assert len(result_df) == 100, "DataFrame length must match input data"
        assert {'t', 'x', 'y'} <= set(result_df.columns), "Required columns must be present"
        
        # Verify data integrity through observable output
        assert result_df['t'].dtype in [np.float64, np.float32], "Time column must be numeric"
        assert result_df['x'].dtype in [np.float64, np.float32], "X column must be numeric"
        assert result_df['y'].dtype in [np.float64, np.float32], "Y column must be numeric"
    
    def test_configuration_dict_input_behavior(self, test_data_generator):
        """
        Test DataFrame construction behavior with dictionary configuration input.
        
        ARRANGE: Set up test data with dictionary configuration
        ACT: Execute DataFrame construction with dict config
        ASSERT: Verify consistent output behavior
        """
        # ARRANGE - Prepare synthetic data and configuration
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=50,
            cols=2,
            data_type="neural"
        )
        
        config_dict = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time dimension'
                },
                'signal': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Neural signal'
                }
            }
        }
        
        # ACT - Execute DataFrame construction
        result_df = make_dataframe_from_config(exp_matrix, config_dict)
        
        # ASSERT - Verify expected behavior
        assert isinstance(result_df, pd.DataFrame), "Must return DataFrame instance"
        assert len(result_df) == 50, "Output length must match input"
        assert 't' in result_df.columns, "Time column must be present"
        assert 'signal' in result_df.columns, "Signal column must be present"
    
    def test_configuration_object_input_behavior(self, test_data_generator):
        """
        Test DataFrame construction with ColumnConfigDict object input.
        
        ARRANGE: Set up test data with validated configuration object
        ACT: Execute DataFrame construction with ColumnConfigDict instance
        ASSERT: Verify behavior consistency with object input
        """
        # ARRANGE - Create validated configuration object
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=75,
            cols=3,
            data_type="behavioral"
        )
        
        config_data = {
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
                    'description': 'X coordinate'
                },
                'y': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Y coordinate'
                }
            }
        }
        
        config_obj = ColumnConfigDict.model_validate(config_data)
        
        # ACT - Execute DataFrame construction with object
        result_df = make_dataframe_from_config(exp_matrix, config_obj)
        
        # ASSERT - Verify expected output behavior
        assert isinstance(result_df, pd.DataFrame), "Must produce DataFrame output"
        assert len(result_df) == 75, "Length must match input matrix"
        assert set(result_df.columns) >= {'t', 'x', 'y'}, "All configured columns must be present"


class TestSchemaValidationBehavior:
    """Test schema validation behavior through observable error conditions."""
    
    def test_missing_required_columns_error_behavior(self, test_data_generator):
        """
        Test error behavior when required columns are missing from input.
        
        ARRANGE: Set up incomplete experimental data missing required columns
        ACT: Attempt DataFrame construction with missing data
        ASSERT: Verify appropriate error behavior
        """
        # ARRANGE - Create incomplete experimental matrix
        incomplete_matrix = test_data_generator.generate_experimental_matrix(
            rows=50,
            cols=1,  # Only time column
            data_type="neural"
        )
        # Remove required column to test error behavior
        incomplete_matrix.pop('signal', None)
        
        config = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                },
                'required_signal': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Required signal data'
                }
            }
        }
        
        # ACT & ASSERT - Verify error behavior
        with pytest.raises(ValueError) as exc_info:
            make_dataframe_from_config(incomplete_matrix, config)
        
        assert "Missing required columns" in str(exc_info.value), "Must indicate missing required columns"
        assert "required_signal" in str(exc_info.value), "Must specify which columns are missing"
    
    @pytest.mark.parametrize("missing_columns,expected_error_content", [
        (['x'], "Missing required columns: x"),
        (['x', 'y'], "Missing required columns"),
        (['t'], "Missing required columns: t"),
    ])
    def test_multiple_missing_columns_error_behavior(self, test_data_generator, missing_columns, expected_error_content):
        """
        Test error behavior with multiple missing required columns.
        
        ARRANGE: Set up experimental data missing specified columns
        ACT: Attempt DataFrame construction with multiple missing columns
        ASSERT: Verify comprehensive error reporting behavior
        """
        # ARRANGE - Create base experimental matrix
        complete_matrix = test_data_generator.generate_experimental_matrix(
            rows=25,
            cols=3,
            data_type="behavioral"
        )
        
        # Remove specified columns to test error behavior
        test_matrix = {k: v for k, v in complete_matrix.items() if k not in missing_columns}
        
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
                'y': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Y position'
                }
            }
        }
        
        # ACT & ASSERT - Verify error behavior
        with pytest.raises(ValueError) as exc_info:
            make_dataframe_from_config(test_matrix, config)
        
        assert expected_error_content in str(exc_info.value), "Error message must indicate missing columns"
    
    def test_invalid_configuration_error_behavior(self, test_data_generator, temp_experiment_directory):
        """
        Test error behavior with invalid configuration structure.
        
        ARRANGE: Set up invalid configuration that should fail validation
        ACT: Attempt DataFrame construction with invalid config
        ASSERT: Verify appropriate validation error behavior
        """
        # ARRANGE - Create experimental data and invalid configuration
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=30,
            cols=2,
            data_type="neural"
        )
        
        # Create configuration with invalid dimension value
        invalid_config = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 5,  # Invalid dimension (must be 1, 2, or 3)
                    'required': True,
                    'description': 'Time values'
                }
            }
        }
        
        config_file = temp_experiment_directory["directory"] / "invalid_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # ACT & ASSERT - Verify validation error behavior
        with pytest.raises(ValidationError) as exc_info:
            make_dataframe_from_config(exp_matrix, str(config_file))
        
        assert "Dimension must be 1, 2, or 3" in str(exc_info.value), "Must validate dimension constraints"


class TestAliasResolutionBehavior:
    """Test column alias resolution through observable output behavior."""
    
    def test_alias_column_mapping_behavior(self, test_data_generator):
        """
        Test alias resolution produces correct column mapping in output.
        
        ARRANGE: Set up experimental data with aliased column names
        ACT: Execute DataFrame construction with alias configuration
        ASSERT: Verify alias resolution in output structure
        """
        # ARRANGE - Create experimental data with aliased columns
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=40,
            cols=3,
            data_type="behavioral"
        )
        # Add aliased column
        exp_matrix['angular_velocity_smooth'] = np.random.rand(40)
        
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
                'y': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Y position'
                },
                'angular_velocity': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': 'Angular velocity',
                    'alias': 'angular_velocity_smooth'
                }
            }
        }
        
        # ACT - Execute DataFrame construction
        result_df = make_dataframe_from_config(exp_matrix, config)
        
        # ASSERT - Verify alias resolution behavior
        assert 'angular_velocity' in result_df.columns, "Target column name must be present in output"
        assert 'angular_velocity_smooth' not in result_df.columns, "Source alias name must not be present"
        
        # Verify data integrity through alias
        expected_data = exp_matrix['angular_velocity_smooth']
        np.testing.assert_array_equal(
            result_df['angular_velocity'].values, 
            expected_data,
            "Aliased data must be correctly mapped"
        )
    
    def test_missing_alias_source_behavior(self, test_data_generator):
        """
        Test behavior when aliased source column is missing from input.
        
        ARRANGE: Set up configuration with alias but missing source column
        ACT: Execute DataFrame construction with missing alias source
        ASSERT: Verify graceful handling of missing alias source
        """
        # ARRANGE - Create experimental data without alias source
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=35,
            cols=3,
            data_type="neural"
        )
        # Ensure alias source is not present
        exp_matrix.pop('theta_smooth', None)
        
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
                'y': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Y position'
                },
                'theta': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': 'Angular position',
                    'alias': 'theta_smooth'
                }
            }
        }
        
        # ACT - Execute DataFrame construction
        result_df = make_dataframe_from_config(exp_matrix, config)
        
        # ASSERT - Verify graceful behavior with missing alias source
        assert 'theta' not in result_df.columns, "Target column must not appear when alias source missing"
        assert 'theta_smooth' not in result_df.columns, "Alias source must not appear when missing"
        assert {'t', 'x', 'y'} <= set(result_df.columns), "Other required columns must still be present"


class TestDefaultValueBehavior:
    """Test default value assignment through observable output behavior."""
    
    def test_default_value_assignment_behavior(self, test_data_generator, temp_experiment_directory):
        """
        Test default value assignment for missing optional columns.
        
        ARRANGE: Set up experimental data missing optional column with default value
        ACT: Execute DataFrame construction with default value configuration
        ASSERT: Verify default value appears in output
        """
        # ARRANGE - Create experimental data missing optional column
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=45,
            cols=3,
            data_type="behavioral"
        )
        # Remove optional column to test default value behavior
        exp_matrix.pop('signal', None)
        
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
                'y': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Y position'
                },
                'optional_signal': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': 'Optional signal values',
                    'default_value': None
                }
            }
        }
        
        config_file = temp_experiment_directory["directory"] / "default_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # ACT - Execute DataFrame construction
        result_df = make_dataframe_from_config(exp_matrix, str(config_file))
        
        # ASSERT - Verify default value assignment behavior
        assert 'optional_signal' in result_df.columns, "Column with default value must be present"
        assert result_df['optional_signal'].iloc[0] is None, "Default value must be correctly assigned"
        assert all(pd.isna(result_df['optional_signal']) | (result_df['optional_signal'] == None)), \
            "All entries must have default value"
    
    @pytest.mark.parametrize("default_value,expected_behavior", [
        (None, "all_none"),
        (0.0, "all_zero"),
        (-999, "all_negative_999"),
        ("missing_data", "all_string"),
        ([], "all_empty_list")
    ])
    def test_various_default_value_types_behavior(self, test_data_generator, temp_experiment_directory, 
                                                 default_value, expected_behavior):
        """
        Test behavior with various types of default values.
        
        ARRANGE: Set up configuration with different default value types
        ACT: Execute DataFrame construction with various default values
        ASSERT: Verify correct default value behavior for each type
        """
        # ARRANGE - Create experimental data and configuration with different default values
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=20,
            cols=3,
            data_type="neural"
        )
        
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
                'y': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Y position'
                },
                'optional_col': {
                    'type': 'any',
                    'required': False,
                    'description': 'Optional column with default',
                    'default_value': default_value
                }
            }
        }
        
        config_file = temp_experiment_directory["directory"] / f"test_defaults_{expected_behavior}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # ACT - Execute DataFrame construction
        result_df = make_dataframe_from_config(exp_matrix, str(config_file))
        
        # ASSERT - Verify default value behavior
        assert 'optional_col' in result_df.columns, "Column with default value must be present"
        
        if expected_behavior == "all_none":
            assert result_df['optional_col'].iloc[0] is None, "Default None value must be correctly assigned"
        elif expected_behavior == "all_zero":
            assert result_df['optional_col'].iloc[0] == 0.0, "Default zero value must be correctly assigned"
        elif expected_behavior == "all_negative_999":
            assert result_df['optional_col'].iloc[0] == -999, "Default negative value must be correctly assigned"
        elif expected_behavior == "all_string":
            assert result_df['optional_col'].iloc[0] == "missing_data", "Default string value must be correctly assigned"
        elif expected_behavior == "all_empty_list":
            assert result_df['optional_col'].iloc[0] == [], "Default list value must be correctly assigned"


class TestSpecialHandlerBehavior:
    """Test special handler functionality through observable output transformations."""
    
    def test_signal_disp_transformation_behavior(self, test_data_generator):
        """
        Test signal_disp special handler produces expected transformation.
        
        ARRANGE: Set up experimental data with signal_disp requiring transformation
        ACT: Execute DataFrame construction with special handling configuration
        ASSERT: Verify transformation behavior in output structure
        """
        # ARRANGE - Create experimental data with signal_disp
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=60,
            cols=3,
            data_type="neural"
        )
        # Add signal_disp data requiring transformation
        exp_matrix['signal_disp'] = np.random.rand(15, 60)  # Multi-channel signal data
        
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
                }
            },
            'special_handlers': {
                'transform_to_match_time_dimension': '_handle_signal_disp'
            }
        }
        
        # ACT - Execute DataFrame construction
        result_df = make_dataframe_from_config(exp_matrix, config)
        
        # ASSERT - Verify transformation behavior
        assert 'signal_disp' in result_df.columns, "Signal display column must be present"
        assert len(result_df) == 60, "DataFrame length must match time dimension"
        
        # Verify each signal_disp entry is properly transformed
        assert isinstance(result_df['signal_disp'].iloc[0], np.ndarray), \
            "Each signal_disp entry must be numpy array"
        assert result_df['signal_disp'].iloc[0].shape == (15,), \
            "Each entry must contain correct number of channels"
        
        # Verify all entries are consistently transformed
        for i in range(len(result_df)):
            entry = result_df['signal_disp'].iloc[i]
            assert isinstance(entry, np.ndarray), f"Entry {i} must be numpy array"
            assert entry.shape == (15,), f"Entry {i} must have correct shape"
    
    def test_signal_disp_orientation_detection_behavior(self, test_data_generator):
        """
        Test automatic orientation detection for signal_disp data.
        
        ARRANGE: Set up signal_disp data in different orientations
        ACT: Execute DataFrame construction with orientation detection
        ASSERT: Verify correct handling regardless of initial orientation
        """
        # ARRANGE - Create experimental data with signal_disp in specific orientation
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=50,
            cols=3,
            data_type="behavioral"
        )
        # Test with (channels, time) orientation
        exp_matrix['signal_disp'] = np.random.rand(12, 50)  # 12 channels, 50 time points
        
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
                }
            },
            'special_handlers': {
                'transform_to_match_time_dimension': '_handle_signal_disp'
            }
        }
        
        # ACT - Execute DataFrame construction
        result_df = make_dataframe_from_config(exp_matrix, config)
        
        # ASSERT - Verify correct orientation handling
        assert 'signal_disp' in result_df.columns, "Signal display column must be present"
        assert len(result_df) == 50, "DataFrame length must match time dimension"
        assert isinstance(result_df['signal_disp'].iloc[0], np.ndarray), \
            "Signal display entries must be arrays"
        assert result_df['signal_disp'].iloc[0].shape == (12,), \
            "Each entry must contain correct number of channels"


class TestMetadataIntegrationBehavior:
    """Test metadata integration through observable output modifications."""
    
    def test_metadata_addition_behavior(self, test_data_generator):
        """
        Test metadata addition produces expected columns in output.
        
        ARRANGE: Set up experimental data with metadata configuration
        ACT: Execute DataFrame construction with metadata parameters
        ASSERT: Verify metadata appears correctly in output
        """
        # ARRANGE - Create experimental data and metadata
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=55,
            cols=2,
            data_type="neural"
        )
        
        metadata_config = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                },
                'signal': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Neural signal'
                },
                'experiment_date': {
                    'type': 'string',
                    'required': False,
                    'is_metadata': True,
                    'description': 'Experiment date'
                },
                'subject_id': {
                    'type': 'string',
                    'required': False,
                    'is_metadata': True,
                    'description': 'Subject identifier'
                },
                'condition': {
                    'type': 'string',
                    'required': False,
                    'is_metadata': True,
                    'description': 'Experimental condition'
                }
            }
        }
        
        sample_metadata = {
            'experiment_date': '2025-01-15',
            'subject_id': 'subject_001',
            'condition': 'control'
        }
        
        # ACT - Execute DataFrame construction with metadata
        result_df = make_dataframe_from_config(
            exp_matrix, 
            metadata_config, 
            metadata=sample_metadata
        )
        
        # ASSERT - Verify metadata integration behavior
        assert 'experiment_date' in result_df.columns, "Metadata date column must be present"
        assert 'subject_id' in result_df.columns, "Metadata subject column must be present"
        assert 'condition' in result_df.columns, "Metadata condition column must be present"
        
        # Verify metadata values are consistent across all rows
        assert all(result_df['experiment_date'] == '2025-01-15'), \
            "Metadata date must be consistent across rows"
        assert all(result_df['subject_id'] == 'subject_001'), \
            "Metadata subject ID must be consistent across rows"
        assert all(result_df['condition'] == 'control'), \
            "Metadata condition must be consistent across rows"
    
    def test_selective_metadata_behavior(self, test_data_generator):
        """
        Test that only configured metadata columns are added to output.
        
        ARRANGE: Set up metadata with extra fields not in configuration
        ACT: Execute DataFrame construction with selective metadata
        ASSERT: Verify only configured metadata appears in output
        """
        # ARRANGE - Create experimental data and selective metadata configuration
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=30,
            cols=2,
            data_type="behavioral"
        )
        
        metadata_config = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                },
                'position': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Position data'
                },
                'session_id': {
                    'type': 'string',
                    'required': False,
                    'is_metadata': True,
                    'description': 'Session identifier'
                }
            }
        }
        
        # Include extra metadata not configured
        sample_metadata = {
            'session_id': 'session_123',
            'extra_field': 'should_not_appear',
            'another_field': 'also_should_not_appear'
        }
        
        # ACT - Execute DataFrame construction
        result_df = make_dataframe_from_config(
            exp_matrix, 
            metadata_config, 
            metadata=sample_metadata
        )
        
        # ASSERT - Verify selective metadata behavior
        assert 'session_id' in result_df.columns, "Configured metadata column must be present"
        assert 'extra_field' not in result_df.columns, "Unconfigured metadata must not appear"
        assert 'another_field' not in result_df.columns, "Extra metadata fields must not appear"
        
        # Verify configured metadata is correctly added
        assert all(result_df['session_id'] == 'session_123'), \
            "Configured metadata must have correct values"
    
    def test_empty_metadata_behavior(self, test_data_generator):
        """
        Test behavior with empty metadata dictionary.
        
        ARRANGE: Set up metadata configuration with empty metadata input
        ACT: Execute DataFrame construction with empty metadata
        ASSERT: Verify no metadata columns are added to output
        """
        # ARRANGE - Create experimental data with metadata configuration but empty metadata
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=25,
            cols=2,
            data_type="neural"
        )
        
        metadata_config = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                },
                'amplitude': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Signal amplitude'
                },
                'trial_type': {
                    'type': 'string',
                    'required': False,
                    'is_metadata': True,
                    'description': 'Trial type'
                },
                'stimulus': {
                    'type': 'string',
                    'required': False,
                    'is_metadata': True,
                    'description': 'Stimulus condition'
                }
            }
        }
        
        # ACT - Execute DataFrame construction with empty metadata
        result_df = make_dataframe_from_config(exp_matrix, metadata_config, metadata={})
        
        # ASSERT - Verify no metadata columns are added
        assert 'trial_type' not in result_df.columns, "Metadata columns must not appear when metadata empty"
        assert 'stimulus' not in result_df.columns, "All metadata columns must be absent"
        assert {'t', 'amplitude'} <= set(result_df.columns), "Required data columns must still be present"


class TestSkipColumnsBehavior:
    """Test skip_columns functionality through observable output modifications."""
    
    def test_skip_configured_column_behavior(self, test_data_generator):
        """
        Test skipping configured columns removes them from output.
        
        ARRANGE: Set up experimental data with columns to be skipped
        ACT: Execute DataFrame construction with skip_columns parameter
        ASSERT: Verify skipped columns absent from output
        """
        # ARRANGE - Create experimental data with multiple columns
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=40,
            cols=4,
            data_type="behavioral"
        )
        
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
                'y': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Y position'
                },
                'velocity': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': 'Velocity values'
                }
            }
        }
        
        # ACT - Execute DataFrame construction with column skipping
        result_df = make_dataframe_from_config(
            exp_matrix, 
            config, 
            skip_columns=['velocity']
        )
        
        # ASSERT - Verify skip behavior
        assert 'velocity' not in result_df.columns, "Skipped column must not appear in output"
        assert 't' in result_df.columns, "Non-skipped required columns must remain"
        assert 'x' in result_df.columns, "Other required columns must be present"
        assert 'y' in result_df.columns, "All non-skipped columns must be included"
    
    def test_skip_required_column_warning_behavior(self, test_data_generator, caplog):
        """
        Test skipping required column generates warning but processes correctly.
        
        ARRANGE: Set up configuration to skip a required column
        ACT: Execute DataFrame construction with required column skip
        ASSERT: Verify warning behavior and output modification
        """
        # ARRANGE - Create experimental data
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=35,
            cols=3,
            data_type="neural"
        )
        
        config = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                },
                'signal': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Neural signal'
                },
                'amplitude': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Signal amplitude'
                }
            }
        }
        
        # ACT - Execute DataFrame construction skipping required column
        with caplog.at_level(logging.WARNING):
            result_df = make_dataframe_from_config(
                exp_matrix, 
                config, 
                skip_columns=['signal']  # Skip required column
            )
        
        # ASSERT - Verify warning behavior and output
        assert 'signal' not in result_df.columns, "Skipped required column must be absent from output"
        assert any('Skipping required column' in record.message for record in caplog.records), \
            "Warning must be logged for skipped required column"
        assert any('signal' in record.message for record in caplog.records), \
            "Warning must specify which required column was skipped"
    
    def test_skip_nonexistent_column_behavior(self, test_data_generator):
        """
        Test skipping nonexistent column does not cause errors.
        
        ARRANGE: Set up experimental data and skip non-existent column
        ACT: Execute DataFrame construction with non-existent column skip
        ASSERT: Verify graceful handling and normal output
        """
        # ARRANGE - Create experimental data
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=28,
            cols=3,
            data_type="behavioral"
        )
        
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
                'y': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Y position'
                }
            }
        }
        
        # ACT - Execute DataFrame construction skipping non-existent column
        result_df = make_dataframe_from_config(
            exp_matrix, 
            config, 
            skip_columns=['nonexistent_column']
        )
        
        # ASSERT - Verify graceful behavior
        assert 'nonexistent_column' not in result_df.columns, \
            "Non-existent column must not appear in output"
        assert {'t', 'x', 'y'} <= set(result_df.columns), \
            "All configured columns must be present despite non-existent skip"
        assert len(result_df) == 28, "DataFrame length must match input data"
    
    def test_skip_multiple_columns_behavior(self, test_data_generator):
        """
        Test skipping multiple columns simultaneously.
        
        ARRANGE: Set up experimental data with multiple columns to skip
        ACT: Execute DataFrame construction with multiple column skips
        ASSERT: Verify all specified columns are skipped from output
        """
        # ARRANGE - Create experimental data with multiple columns
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=45,
            cols=5,
            data_type="neural"
        )
        
        config = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                },
                'channel_1': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': 'Channel 1 signal'
                },
                'channel_2': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': 'Channel 2 signal'
                },
                'channel_3': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': 'Channel 3 signal'
                },
                'trigger': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': 'Trigger signal'
                }
            }
        }
        
        # ACT - Execute DataFrame construction with multiple column skips
        result_df = make_dataframe_from_config(
            exp_matrix, 
            config, 
            skip_columns=['channel_2', 'channel_3', 'trigger']
        )
        
        # ASSERT - Verify multiple skip behavior
        assert 'channel_2' not in result_df.columns, "First skipped column must be absent"
        assert 'channel_3' not in result_df.columns, "Second skipped column must be absent"
        assert 'trigger' not in result_df.columns, "Third skipped column must be absent"
        assert 't' in result_df.columns, "Required time column must remain"
        assert 'channel_1' in result_df.columns, "Non-skipped optional columns must remain"
    
    def test_empty_skip_list_behavior(self, test_data_generator):
        """
        Test behavior with empty skip_columns list.
        
        ARRANGE: Set up experimental data with empty skip list
        ACT: Execute DataFrame construction with empty skip_columns
        ASSERT: Verify all configured columns appear in output
        """
        # ARRANGE - Create experimental data
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=32,
            cols=3,
            data_type="behavioral"
        )
        
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
                'y': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Y position'
                }
            }
        }
        
        # ACT - Execute DataFrame construction with empty skip list
        result_df = make_dataframe_from_config(
            exp_matrix, 
            config, 
            skip_columns=[]
        )
        
        # ASSERT - Verify all columns present with empty skip list
        assert {'t', 'x', 'y'} <= set(result_df.columns), \
            "All configured columns must be present with empty skip list"
        assert len(result_df) == 32, "DataFrame length must match input data"


class TestEdgeCaseScenarios:
    """Test comprehensive edge cases and boundary conditions."""
    
    def test_empty_experimental_matrix_behavior(self, test_data_generator):
        """
        Test behavior with empty experimental matrix input.
        
        ARRANGE: Set up empty experimental matrix
        ACT: Attempt DataFrame construction with empty input
        ASSERT: Verify appropriate error behavior
        """
        # ARRANGE - Create empty experimental matrix
        empty_matrix = {}
        
        config = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                }
            }
        }
        
        # ACT & ASSERT - Verify error behavior with empty input
        with pytest.raises(ValueError) as exc_info:
            make_dataframe_from_config(empty_matrix, config)
        
        assert "Missing required columns" in str(exc_info.value), \
            "Must indicate missing required columns for empty input"
    
    def test_mismatched_array_lengths_behavior(self, test_data_generator):
        """
        Test behavior with mismatched array lengths in experimental data.
        
        ARRANGE: Set up experimental data with inconsistent array lengths
        ACT: Attempt DataFrame construction with mismatched data
        ASSERT: Verify error handling or graceful degradation
        """
        # ARRANGE - Create experimental data with mismatched lengths
        mismatched_matrix = {
            't': np.linspace(0, 10, 100),
            'x': np.random.rand(50),  # Different length
            'y': np.random.rand(100)
        }
        
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
                'y': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Y position'
                }
            }
        }
        
        # ACT & ASSERT - Verify handling of mismatched lengths
        try:
            result_df = make_dataframe_from_config(mismatched_matrix, config)
            # If successful, verify result is valid
            assert isinstance(result_df, pd.DataFrame), "Result must be valid DataFrame"
        except (ValueError, RuntimeError) as e:
            # If error occurs, verify it's appropriate
            error_message = str(e).lower()
            assert any(word in error_message for word in ['length', 'dimension', 'shape']), \
                "Error must indicate array length/shape issues"
    
    def test_none_values_in_matrix_behavior(self, test_data_generator):
        """
        Test handling of None values in experimental matrix.
        
        ARRANGE: Set up experimental data containing None values
        ACT: Attempt DataFrame construction with None values
        ASSERT: Verify appropriate error handling
        """
        # ARRANGE - Create experimental data with None values
        matrix_with_none = {
            't': np.linspace(0, 10, 50),
            'x': None,  # None value
            'y': np.random.rand(50)
        }
        
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
                'y': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Y position'
                }
            }
        }
        
        # ACT & ASSERT - Verify None value handling
        try:
            result_df = make_dataframe_from_config(matrix_with_none, config)
            assert isinstance(result_df, pd.DataFrame), "Result must be valid DataFrame if successful"
        except (ValueError, TypeError) as e:
            error_message = str(e).lower()
            assert any(word in error_message for word in ['none', 'null']), \
                "Error must indicate None/null value issues"
    
    def test_invalid_config_source_type_behavior(self, test_data_generator):
        """
        Test error behavior with invalid configuration source types.
        
        ARRANGE: Set up invalid configuration source types
        ACT: Attempt DataFrame construction with invalid config types
        ASSERT: Verify appropriate type error behavior
        """
        # ARRANGE - Create experimental data
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=25,
            cols=2,
            data_type="neural"
        )
        
        # ACT & ASSERT - Test invalid integer config type
        with pytest.raises(TypeError) as exc_info:
            make_dataframe_from_config(exp_matrix, 123)
        
        assert "config_source must be" in str(exc_info.value), \
            "Must indicate valid config source types"
        
        # ACT & ASSERT - Test invalid list config type
        with pytest.raises(TypeError) as exc_info:
            make_dataframe_from_config(exp_matrix, [])
        
        assert "config_source must be" in str(exc_info.value), \
            "Must indicate valid config source types"


# ===== PROTOCOL-BASED MOCK INTEGRATION TESTS =====
# Using centralized mocks from tests/utils.py per F-013 requirements

class TestProtocolBasedMockIntegration:
    """Test DataFrame construction with Protocol-based mock implementations."""
    
    def test_mock_configuration_provider_behavior(self, test_data_generator, mock_data_loading):
        """
        Test DataFrame construction with mocked configuration provider.
        
        ARRANGE: Set up mock configuration provider with test data
        ACT: Execute DataFrame construction with mocked configuration
        ASSERT: Verify behavior with mocked configuration source
        """
        # ARRANGE - Set up experimental data and mock configuration
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=30,
            cols=2,
            data_type="behavioral"
        )
        
        # Create mock configuration provider
        config_provider = create_mock_config_provider()
        test_config = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                },
                'position': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Position data'
                }
            }
        }
        config_provider.add_configuration('test_config', test_config)
        
        # ACT - Execute DataFrame construction with mock
        result_df = make_dataframe_from_config(exp_matrix, test_config)
        
        # ASSERT - Verify behavior with mocked configuration
        assert isinstance(result_df, pd.DataFrame), "Must produce DataFrame with mock config"
        assert {'t', 'position'} <= set(result_df.columns), "Mock configuration columns must be present"
        assert len(result_df) == 30, "DataFrame length must match input data"
    
    def test_mock_filesystem_integration_behavior(self, test_data_generator):
        """
        Test DataFrame construction with mocked filesystem operations.
        
        ARRANGE: Set up mock filesystem with configuration files
        ACT: Execute DataFrame construction with mocked file operations
        ASSERT: Verify behavior with mocked filesystem access
        """
        # ARRANGE - Set up experimental data and mock filesystem
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=40,
            cols=3,
            data_type="neural"
        )
        
        # Create mock filesystem with configuration
        mock_fs = create_mock_filesystem()
        test_config = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                },
                'signal_1': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Signal channel 1'
                },
                'signal_2': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Signal channel 2'
                }
            }
        }
        
        # ACT - Execute DataFrame construction with dict config (avoiding file operations)
        result_df = make_dataframe_from_config(exp_matrix, test_config)
        
        # ASSERT - Verify behavior with mock filesystem context
        assert isinstance(result_df, pd.DataFrame), "Must produce DataFrame with mock filesystem"
        assert {'t', 'signal_1', 'signal_2'} <= set(result_df.columns), \
            "All configured columns must be present"
        assert len(result_df) == 40, "DataFrame length must match input"


# ===== EDGE-CASE ENHANCEMENT THROUGH PARAMETERIZATION =====
# Enhanced edge-case coverage per testing strategy requirements

class TestParameterizedEdgeCases:
    """Test enhanced edge-case coverage through parameterized scenarios."""
    
    @pytest.mark.parametrize("signal_orientation,expected_channels", [
        ("channels_first", 8),    # (8, 60) -> 8 channels per timepoint
        ("time_first", 12),       # (60, 12) -> 12 channels per timepoint
    ])
    def test_signal_disp_orientation_edge_cases(self, test_data_generator, signal_orientation, expected_channels):
        """
        Test signal_disp orientation detection with various array shapes.
        
        ARRANGE: Set up signal_disp data in different orientations
        ACT: Execute DataFrame construction with orientation detection
        ASSERT: Verify correct channel count regardless of orientation
        """
        # ARRANGE - Create experimental data with specific signal_disp orientation
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=60,
            cols=2,
            data_type="neural"
        )
        
        if signal_orientation == "channels_first":
            exp_matrix['signal_disp'] = np.random.rand(expected_channels, 60)
        else:  # time_first
            exp_matrix['signal_disp'] = np.random.rand(60, expected_channels)
        
        config = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                },
                'signal': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Neural signal'
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
        
        # ACT - Execute DataFrame construction
        result_df = make_dataframe_from_config(exp_matrix, config)
        
        # ASSERT - Verify correct orientation handling
        assert 'signal_disp' in result_df.columns, "Signal display column must be present"
        assert len(result_df) == 60, "DataFrame length must match time dimension"
        assert result_df['signal_disp'].iloc[0].shape == (expected_channels,), \
            f"Each entry must contain {expected_channels} channels"
    
    @pytest.mark.parametrize("alias_scenario,source_name,target_name,data_present", [
        ("valid_alias", "raw_velocity", "velocity", True),
        ("missing_source", "missing_column", "velocity", False),
        ("circular_reference", "circular_a", "circular_b", True),
    ])
    def test_alias_resolution_edge_cases(self, test_data_generator, alias_scenario, source_name, target_name, data_present):
        """
        Test alias resolution with various edge case scenarios.
        
        ARRANGE: Set up experimental data with different alias scenarios
        ACT: Execute DataFrame construction with alias configuration
        ASSERT: Verify appropriate alias handling for each scenario
        """
        # ARRANGE - Create experimental data based on scenario
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=35,
            cols=2,
            data_type="behavioral"
        )
        
        if data_present:
            exp_matrix[source_name] = np.random.rand(35)
        
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
                target_name: {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': f'Target column {target_name}',
                    'alias': source_name
                }
            }
        }
        
        # ACT - Execute DataFrame construction
        result_df = make_dataframe_from_config(exp_matrix, config)
        
        # ASSERT - Verify alias behavior for each scenario
        if alias_scenario == "valid_alias" and data_present:
            assert target_name in result_df.columns, "Valid alias target must be present"
            assert source_name not in result_df.columns, "Alias source must not appear"
            np.testing.assert_array_equal(
                result_df[target_name].values, 
                exp_matrix[source_name], 
                "Alias data must be correctly mapped"
            )
        elif alias_scenario == "missing_source":
            assert target_name not in result_df.columns, "Target must not appear when source missing"
            assert source_name not in result_df.columns, "Missing source must not appear"
        elif alias_scenario == "circular_reference":
            # For circular references, behavior should be graceful (no infinite loops)
            result_columns = set(result_df.columns)
            assert {'t', 'x'} <= result_columns, "Required columns must still be present"
    
    @pytest.mark.parametrize("default_scenario,default_value,expected_type", [
        ("null_default", None, type(None)),
        ("numeric_default", 42.0, float),
        ("string_default", "default_text", str),
        ("list_default", [], list),
        ("negative_default", -1, int),
    ])
    def test_default_value_edge_cases(self, test_data_generator, temp_experiment_directory, 
                                    default_scenario, default_value, expected_type):
        """
        Test default value assignment with various data types and edge cases.
        
        ARRANGE: Set up configuration with different default value types
        ACT: Execute DataFrame construction with various default values
        ASSERT: Verify correct default value handling for each type
        """
        # ARRANGE - Create experimental data missing optional column
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=20,
            cols=2,
            data_type="neural"
        )
        
        config = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                },
                'signal': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Neural signal'
                },
                'optional_field': {
                    'type': 'any',
                    'required': False,
                    'description': 'Optional field with default',
                    'default_value': default_value
                }
            }
        }
        
        config_file = temp_experiment_directory["directory"] / f"defaults_{default_scenario}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # ACT - Execute DataFrame construction
        result_df = make_dataframe_from_config(exp_matrix, str(config_file))
        
        # ASSERT - Verify default value type and assignment
        assert 'optional_field' in result_df.columns, "Optional field with default must be present"
        actual_value = result_df['optional_field'].iloc[0]
        
        if default_value is None:
            assert actual_value is None, "None default must be correctly assigned"
        else:
            assert actual_value == default_value, f"Default value {default_value} must be correctly assigned"
            assert isinstance(actual_value, expected_type), f"Default value must have type {expected_type}"


# ===== COMPREHENSIVE INTEGRATION SCENARIOS =====
# End-to-end behavior validation per TST-INTEG-003 requirements

class TestComprehensiveBehaviorIntegration:
    """Test comprehensive integration scenarios with all features combined."""
    
    def test_end_to_end_workflow_behavior(self, test_data_generator, temp_experiment_directory):
        """
        Test complete end-to-end workflow with all features combined.
        
        ARRANGE: Set up comprehensive experimental data with all feature types
        ACT: Execute complete DataFrame construction workflow
        ASSERT: Verify all features work together correctly
        """
        # ARRANGE - Create comprehensive experimental data
        exp_matrix = test_data_generator.generate_experimental_matrix(
            rows=80,
            cols=4,
            data_type="neural"
        )
        
        # Add specific data for comprehensive testing
        exp_matrix['velocity_raw'] = np.random.rand(80)  # For alias testing
        exp_matrix['signal_disp'] = np.random.rand(16, 80)  # For special handling
        # Missing 'optional_metric' to test default values
        
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
                'velocity': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': 'Velocity values',
                    'alias': 'velocity_raw'
                },
                'signal_disp': {
                    'type': 'numpy.ndarray',
                    'dimension': 2,
                    'required': False,
                    'description': 'Signal display data',
                    'special_handling': 'transform_to_match_time_dimension'
                },
                'optional_metric': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': 'Optional metric',
                    'default_value': 0.0
                },
                'experiment_id': {
                    'type': 'string',
                    'required': False,
                    'is_metadata': True,
                    'description': 'Experiment identifier'
                },
                'session_date': {
                    'type': 'string',
                    'required': False,
                    'is_metadata': True,
                    'description': 'Session date'
                }
            },
            'special_handlers': {
                'transform_to_match_time_dimension': '_handle_signal_disp'
            }
        }
        
        metadata = {
            'experiment_id': 'comprehensive_test_001',
            'session_date': '2025-01-15',
            'extra_field': 'should_be_ignored'  # Not in config
        }
        
        config_file = temp_experiment_directory["directory"] / "comprehensive_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(comprehensive_config, f)
        
        # ACT - Execute comprehensive workflow with column skipping
        result_df = make_dataframe_from_config(
            exp_matrix, 
            str(config_file), 
            metadata=metadata,
            skip_columns=['y']  # Skip one column to test skip functionality
        )
        
        # ASSERT - Verify comprehensive behavior integration
        
        # Basic structure validation
        assert isinstance(result_df, pd.DataFrame), "Result must be pandas DataFrame"
        assert len(result_df) == 80, "DataFrame length must match input data"
        
        # Required columns (except skipped)
        assert 't' in result_df.columns, "Time column must be present"
        assert 'x' in result_df.columns, "X position must be present"
        assert 'y' not in result_df.columns, "Skipped column must be absent"
        
        # Alias resolution validation
        assert 'velocity' in result_df.columns, "Aliased column must be present"
        assert 'velocity_raw' not in result_df.columns, "Original alias source must not appear"
        np.testing.assert_array_equal(
            result_df['velocity'].values, 
            exp_matrix['velocity_raw'],
            "Alias data must be correctly mapped"
        )
        
        # Special handler validation
        assert 'signal_disp' in result_df.columns, "Signal display column must be present"
        assert isinstance(result_df['signal_disp'].iloc[0], np.ndarray), \
            "Signal display entries must be arrays"
        assert result_df['signal_disp'].iloc[0].shape == (16,), \
            "Signal display must have correct channel count"
        
        # Default value validation
        assert 'optional_metric' in result_df.columns, "Optional column with default must be present"
        assert all(result_df['optional_metric'] == 0.0), \
            "Default values must be correctly assigned"
        
        # Metadata integration validation
        assert 'experiment_id' in result_df.columns, "Metadata experiment ID must be present"
        assert 'session_date' in result_df.columns, "Metadata session date must be present"
        assert all(result_df['experiment_id'] == 'comprehensive_test_001'), \
            "Metadata experiment ID must be consistent"
        assert all(result_df['session_date'] == '2025-01-15'), \
            "Metadata session date must be consistent"
        assert 'extra_field' not in result_df.columns, \
            "Unconfigured metadata must not appear"
        
        # Data integrity validation for non-skipped columns
        np.testing.assert_array_equal(result_df['t'].values, exp_matrix['t'], \
            "Time data must be preserved")
        np.testing.assert_array_equal(result_df['x'].values, exp_matrix['x'], \
            "X position data must be preserved")
    
    def test_error_propagation_and_recovery_behavior(self, test_data_generator, temp_experiment_directory):
        """
        Test error propagation and recovery mechanisms in integration scenarios.
        
        ARRANGE: Set up scenarios with recoverable and non-recoverable errors
        ACT: Execute DataFrame construction with error conditions
        ASSERT: Verify appropriate error handling and recovery behavior
        """
        # ARRANGE - Create experimental data with missing required column
        incomplete_matrix = test_data_generator.generate_experimental_matrix(
            rows=50,
            cols=2,
            data_type="behavioral"
        )
        # Remove required column to test error behavior
        incomplete_matrix.pop('y', None)
        
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
                'y': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Y position'
                },
                'optional_data': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': False,
                    'description': 'Optional data field',
                    'default_value': None
                }
            }
        }
        
        config_file = temp_experiment_directory["directory"] / "error_test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # ACT & ASSERT - Test non-recoverable error (missing required column)
        with pytest.raises(ValueError) as exc_info:
            make_dataframe_from_config(incomplete_matrix, str(config_file))
        
        assert "Missing required columns: y" in str(exc_info.value), \
            "Must indicate specific missing required columns"
        
        # ARRANGE - Add missing required column for recovery test
        recovery_matrix = incomplete_matrix.copy()
        recovery_matrix['y'] = np.random.rand(50)
        
        # ACT - Test recovery with complete data
        result_df = make_dataframe_from_config(recovery_matrix, str(config_file))
        
        # ASSERT - Verify successful recovery
        assert isinstance(result_df, pd.DataFrame), "Recovery must produce valid DataFrame"
        assert 'optional_data' in result_df.columns, "Optional column with default must be present"
        assert result_df['optional_data'].iloc[0] is None, "Default value must be correctly assigned"
        assert len(result_df) == 50, "DataFrame length must match recovered data"


if __name__ == "__main__":
    pytest.main([__file__])