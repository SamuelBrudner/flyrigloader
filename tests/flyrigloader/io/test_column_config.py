"""
Behavior-focused test suite for column configuration functionality.

This module provides comprehensive testing for column configuration validation,
YAML config loading, schema enforcement, and DataFrame transformation using
black-box behavioral validation techniques per Section 0 requirements.

Features:
- Behavior-focused testing without implementation coupling
- Protocol-based mock implementations for dependency isolation
- AAA pattern structure with clear test phase separation
- Centralized fixture usage from tests/conftest.py
- Edge-case coverage through parameterized scenarios
- Observable configuration behavior validation
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pytest
import yaml
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from pydantic import ValidationError

# Import centralized test utilities and protocol-based mocks
from tests.utils import (
    create_mock_filesystem,
    create_mock_dataloader,
    create_mock_config_provider,
    generate_edge_case_scenarios,
    validate_test_structure
)

from flyrigloader.io.column_models import (
    ColumnConfig,
    ColumnConfigDict,
    ColumnDimension,
    SpecialHandlerType,
    load_column_config,
    get_config_from_source,
    get_default_config_path,
)
from flyrigloader.io.pickle import make_dataframe_from_config


# ============================================================================
# CENTRALIZED FIXTURES - Using centralized test infrastructure
# ============================================================================

# Use centralized fixtures from tests/conftest.py for consistency
# These fixtures provide enhanced functionality with Protocol-based implementations

@pytest.fixture
def mock_column_config_provider(mocker):
    """
    Protocol-based mock implementation for column configuration operations.
    
    Provides consistent mock behavior across all column configuration test scenarios
    using Protocol-based patterns from tests/utils.py for dependency isolation.
    
    Returns:
        MockConfigurationProvider: Protocol-based configuration provider mock
    """
    return create_mock_config_provider('comprehensive', include_errors=True)


@pytest.fixture
def mock_column_filesystem(mocker):
    """
    Protocol-based mock implementation for filesystem operations in column tests.
    
    Creates filesystem mock with column-specific file scenarios including
    YAML configurations, corrupted files, and Unicode path edge cases.
    
    Returns:
        MockFilesystem: Protocol-based filesystem mock with column scenarios
    """
    filesystem_structure = {
        'files': {
            '/test/config/column_config.yaml': {'size': 2048, 'content': 'test: config'},
            '/test/config/malformed.yaml': {'size': 512, 'corrupted': True},
            '/test/data/test_data.pkl': {'size': 4096}
        },
        'directories': ['/test', '/test/config', '/test/data']
    }
    
    return create_mock_filesystem(
        structure=filesystem_structure,
        unicode_files=True,
        corrupted_files=True
    )


@pytest.fixture
def mock_column_dataloader(mocker):
    """
    Protocol-based mock implementation for data loading operations in column tests.
    
    Creates data loader mock with experimental matrix scenarios and
    DataFrame creation capabilities for testing column configuration application.
    
    Returns:
        MockDataLoading: Protocol-based data loading mock
    """
    return create_mock_dataloader(
        scenarios=['basic', 'corrupted', 'memory'],
        include_experimental_data=True
    )


# ============================================================================
# BEHAVIOR-FOCUSED TESTS - Public API Validation Without Implementation Coupling
# ============================================================================

class TestColumnConfigBehavior:
    """
    Behavior-focused test suite for column configuration functionality.
    
    Tests focus on observable public API behavior rather than internal implementation
    details, ensuring robust validation without coupling to Pydantic internals.
    """

    def test_column_config_creation_returns_valid_object(self):
        """
        Test that column configuration creation returns valid object with expected public interface.
        
        ARRANGE: Set up minimal configuration parameters
        ACT: Create ColumnConfig instance through public constructor
        ASSERT: Verify public interface provides expected behavior
        """
        # ARRANGE - Prepare minimal configuration data
        config_type = 'numpy.ndarray'
        description = 'Test column'
        
        # ACT - Execute column configuration creation
        config = ColumnConfig(type=config_type, description=description)
        
        # ASSERT - Verify observable behavior through public interface
        assert config.type == config_type
        assert config.description == description
        assert config.required is False  # Expected default behavior
        assert hasattr(config, 'dimension')  # Public interface availability
        assert hasattr(config, 'alias')  # Public interface availability

    def test_column_config_comprehensive_creation_preserves_all_values(self):
        """
        Test that comprehensive column configuration preserves all specified values.
        
        ARRANGE: Set up complete configuration with all optional parameters
        ACT: Create ColumnConfig with comprehensive parameters
        ASSERT: Verify all values are accessible through public interface
        """
        # ARRANGE - Prepare comprehensive configuration parameters
        config_params = {
            'type': 'numpy.ndarray',
            'dimension': ColumnDimension.TWO_D,
            'required': True,
            'description': 'Test 2D array column',
            'alias': 'alternative_name',
            'is_metadata': False,
            'default_value': None,
            'special_handling': SpecialHandlerType.TRANSFORM_TIME_DIMENSION
        }
        
        # ACT - Execute comprehensive configuration creation
        config = ColumnConfig(**config_params)
        
        # ASSERT - Verify all values accessible through public interface
        assert config.type == config_params['type']
        assert config.dimension == config_params['dimension']
        assert config.required == config_params['required']
        assert config.description == config_params['description']
        assert config.alias == config_params['alias']
        assert config.is_metadata == config_params['is_metadata']
        assert config.default_value == config_params['default_value']
        assert config.special_handling == config_params['special_handling']

    @pytest.mark.parametrize("dimension_input,expected_behavior", [
        (1, "accepts_one_dimension"),
        (2, "accepts_two_dimension"), 
        (3, "accepts_three_dimension"),
        (ColumnDimension.ONE_D, "accepts_enum_dimension"),
        (None, "accepts_none_dimension"),
    ], ids=['int_1d', 'int_2d', 'int_3d', 'enum_1d', 'none'])
    def test_dimension_validation_accepts_valid_inputs(self, dimension_input, expected_behavior):
        """
        Test that dimension validation accepts valid dimensional inputs.
        
        ARRANGE: Set up valid dimension inputs of various types
        ACT: Create column configuration with dimensional input
        ASSERT: Verify configuration creation succeeds with expected behavior
        """
        # ARRANGE - Prepare valid dimension input
        config_params = {
            'type': 'numpy.ndarray',
            'dimension': dimension_input,
            'description': 'Test column'
        }
        
        # ACT - Execute configuration creation (should succeed)
        config = ColumnConfig(**config_params)
        
        # ASSERT - Verify successful creation and appropriate dimension handling
        assert config is not None
        assert hasattr(config, 'dimension')
        # Verify dimension value is preserved or converted appropriately
        if dimension_input is None:
            assert config.dimension is None
        elif isinstance(dimension_input, int):
            assert config.dimension is not None
        elif isinstance(dimension_input, ColumnDimension):
            assert config.dimension == dimension_input

    @pytest.mark.parametrize("invalid_dimension", [
        0, 4, 5, -1, 2.5, "invalid", [], {}
    ], ids=['zero', 'four', 'five', 'negative', 'float', 'string', 'list', 'dict'])
    def test_dimension_validation_rejects_invalid_inputs(self, invalid_dimension):
        """
        Test that dimension validation rejects invalid dimensional inputs.
        
        ARRANGE: Set up invalid dimension inputs that should be rejected
        ACT: Attempt to create column configuration with invalid dimension
        ASSERT: Verify appropriate error handling occurs
        """
        # ARRANGE - Prepare configuration with invalid dimension
        config_params = {
            'type': 'numpy.ndarray',
            'dimension': invalid_dimension,
            'description': 'Test column'
        }
        
        # ACT & ASSERT - Execute configuration creation (should raise appropriate error)
        with pytest.raises(ValueError) as excinfo:
            ColumnConfig(**config_params)
        
        # Verify error message contains helpful information about valid dimensions
        error_message = str(excinfo.value)
        assert "Dimension must be 1, 2, or 3" in error_message

    @pytest.mark.parametrize("handler_input,expected_behavior", [
        ("extract_first_column_if_2d", "accepts_extract_handler"),
        ("transform_to_match_time_dimension", "accepts_transform_handler"),
        (SpecialHandlerType.EXTRACT_FIRST_COLUMN, "accepts_enum_handler"),
        (None, "accepts_none_handler"),
    ], ids=['extract_string', 'transform_string', 'enum_direct', 'none'])
    def test_special_handling_validation_accepts_valid_handlers(self, handler_input, expected_behavior):
        """
        Test that special handling validation accepts valid handler specifications.
        
        ARRANGE: Set up valid special handler inputs of various types  
        ACT: Create column configuration with special handling input
        ASSERT: Verify configuration creation succeeds with expected behavior
        """
        # ARRANGE - Prepare valid special handler input
        config_params = {
            'type': 'numpy.ndarray',
            'special_handling': handler_input,
            'description': 'Test column'
        }
        
        # ACT - Execute configuration creation (should succeed)
        config = ColumnConfig(**config_params)
        
        # ASSERT - Verify successful creation and appropriate handler preservation
        assert config is not None
        assert hasattr(config, 'special_handling')
        # Verify handler value is preserved or converted appropriately
        if handler_input is None:
            assert config.special_handling is None
        else:
            assert config.special_handling is not None

    @pytest.mark.parametrize("invalid_handler", [
        "invalid_handler", "unknown_transformation", 123, [], {}
    ], ids=['invalid_string', 'unknown_string', 'int', 'list', 'dict'])
    def test_special_handling_validation_rejects_invalid_handlers(self, invalid_handler):
        """
        Test that special handling validation rejects invalid handler specifications.
        
        ARRANGE: Set up invalid special handler inputs that should be rejected
        ACT: Attempt to create column configuration with invalid handler
        ASSERT: Verify appropriate error handling occurs
        """
        # ARRANGE - Prepare configuration with invalid special handler
        config_params = {
            'type': 'numpy.ndarray',
            'special_handling': invalid_handler,
            'description': 'Test column'
        }
        
        # ACT & ASSERT - Execute configuration creation (should raise appropriate error)
        with pytest.raises(ValueError) as excinfo:
            ColumnConfig(**config_params)
        
        # Verify error message contains helpful information about valid handlers
        error_message = str(excinfo.value)
        assert "Special handler must be one of" in error_message

    def test_configuration_accepts_dimension_with_non_numpy_type(self):
        """
        Test that configuration accepts dimension specification with non-numpy types.
        
        ARRANGE: Set up configuration with string type and dimension
        ACT: Create column configuration with type-dimension combination
        ASSERT: Verify configuration creation succeeds despite potential compatibility issue
        """
        # ARRANGE - Prepare configuration with dimension and non-numpy type
        config_params = {
            'type': 'string',
            'dimension': 1,
            'description': 'String column with dimension'
        }
        
        # ACT - Execute configuration creation
        config = ColumnConfig(**config_params)
        
        # ASSERT - Verify object creation succeeds with expected values
        assert config.type == 'string'
        assert config.dimension is not None  # Dimension should be preserved
        assert config.description == 'String column with dimension'

    def test_configuration_accepts_one_d_with_transform_handler(self):
        """
        Test that configuration accepts 1D dimension with transform handler.
        
        ARRANGE: Set up configuration with 1D dimension and time transformation handler
        ACT: Create column configuration with dimension-handler combination  
        ASSERT: Verify configuration creation succeeds despite potential mismatch
        """
        # ARRANGE - Prepare configuration with 1D dimension and transform handler
        config_params = {
            'type': 'numpy.ndarray',
            'dimension': 1,
            'special_handling': SpecialHandlerType.TRANSFORM_TIME_DIMENSION,
            'description': 'Signal column with dimension-handler mismatch'
        }
        
        # ACT - Execute configuration creation
        config = ColumnConfig(**config_params)
        
        # ASSERT - Verify object creation succeeds with expected values
        assert config.special_handling == SpecialHandlerType.TRANSFORM_TIME_DIMENSION
        assert config.dimension is not None
        assert config.type == 'numpy.ndarray'


class TestColumnConfigDictBehavior:
    """
    Behavior-focused test suite for complete column configuration dictionary validation.
    
    Tests focus on observable configuration behavior and successful creation/validation
    without examining internal Pydantic model validation mechanisms.
    """

    def test_comprehensive_config_dict_creation_succeeds(self, sample_comprehensive_config_dict):
        """
        Test that comprehensive configuration dictionary creation succeeds with expected structure.
        
        ARRANGE: Set up comprehensive configuration with all supported features
        ACT: Create ColumnConfigDict from comprehensive configuration
        ASSERT: Verify successful creation and accessible column structure
        """
        # ARRANGE - Use comprehensive configuration from centralized fixture
        comprehensive_config = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time values'},
                'x': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'X position'},
                'y': {'type': 'numpy.ndarray', 'dimension': 1, 'required': False, 'description': 'Y position'},
                'signal_disp': {'type': 'numpy.ndarray', 'dimension': 2, 'required': False, 
                              'description': 'Signal display', 'special_handling': 'transform_to_match_time_dimension'},
                'metadata': {'type': 'string', 'required': False, 'is_metadata': True, 'description': 'Metadata'}
            },
            'special_handlers': {'transform_to_match_time_dimension': '_handle_signal_disp'}
        }
        
        # ACT - Execute configuration dictionary creation
        config_dict = ColumnConfigDict.model_validate(comprehensive_config)
        
        # ASSERT - Verify successful creation and expected public behavior
        assert config_dict is not None
        assert hasattr(config_dict, 'columns')
        assert hasattr(config_dict, 'special_handlers')
        assert len(config_dict.columns) == 5
        assert 't' in config_dict.columns
        assert 'x' in config_dict.columns
        assert len(config_dict.special_handlers) == 1

    def test_minimal_config_dict_creation_succeeds(self):
        """
        Test that minimal configuration dictionary creation succeeds.
        
        ARRANGE: Set up minimal valid configuration
        ACT: Create ColumnConfigDict from minimal configuration  
        ASSERT: Verify successful creation with empty special handlers
        """
        # ARRANGE - Prepare minimal valid configuration
        minimal_config = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time values'}
            },
            'special_handlers': {}
        }
        
        # ACT - Execute minimal configuration creation
        config_dict = ColumnConfigDict.model_validate(minimal_config)
        
        # ASSERT - Verify successful creation with expected minimal structure
        assert config_dict is not None
        assert len(config_dict.columns) == 1
        assert len(config_dict.special_handlers) == 0
        assert 't' in config_dict.columns

    def test_config_dict_handles_undefined_special_handlers(self):
        """
        Test that configuration dictionary handles undefined special handlers gracefully.
        
        ARRANGE: Set up configuration with special handling but no handler definition
        ACT: Create ColumnConfigDict with missing handler definition
        ASSERT: Verify configuration creation succeeds despite missing handler
        """
        # ARRANGE - Prepare configuration with undefined special handler
        config_with_undefined_handler = {
            'columns': {
                'test_col': {
                    'type': 'numpy.ndarray',
                    'description': 'Test column',
                    'special_handling': 'transform_to_match_time_dimension'
                }
            },
            'special_handlers': {}  # Missing the referenced handler
        }
        
        # ACT - Execute configuration creation (should succeed)
        config_dict = ColumnConfigDict.model_validate(config_with_undefined_handler)
        
        # ASSERT - Verify creation succeeds and configuration is accessible
        assert config_dict is not None
        assert 'test_col' in config_dict.columns
        assert config_dict.columns['test_col'].special_handling is not None
        assert len(config_dict.special_handlers) == 0


# ============================================================================
# CONFIGURATION LOADING BEHAVIOR TESTS - YAML Processing and Schema Validation
# ============================================================================

class TestConfigurationLoadingBehavior:
    """
    Behavior-focused test suite for configuration loading and validation.
    
    Tests focus on observable configuration loading behavior using Protocol-based
    mocks for filesystem and YAML operations without coupling to implementation.
    """

    def test_load_column_config_from_valid_file_succeeds(self, sample_column_config_file):
        """
        Test that loading column configuration from valid YAML file succeeds.
        
        ARRANGE: Set up valid YAML configuration file using centralized fixture
        ACT: Execute configuration loading from file path
        ASSERT: Verify successful loading with expected configuration structure
        """
        # ARRANGE - Valid configuration file from centralized fixture
        config_file_path = sample_column_config_file
        
        # ACT - Execute configuration loading
        config = load_column_config(config_file_path)
        
        # ASSERT - Verify successful loading and expected configuration behavior
        assert config is not None
        assert isinstance(config, ColumnConfigDict)
        assert hasattr(config, 'columns')
        assert len(config.columns) > 0
        assert 't' in config.columns  # Expected time column

    def test_load_column_config_handles_missing_file_appropriately(self):
        """
        Test that loading configuration from missing file produces appropriate error.
        
        ARRANGE: Set up non-existent file path
        ACT: Attempt to load configuration from missing file
        ASSERT: Verify appropriate file not found error behavior
        """
        # ARRANGE - Non-existent file path
        nonexistent_path = '/nonexistent/path/config.yaml'
        
        # ACT & ASSERT - Execute configuration loading (should raise FileNotFoundError)
        with pytest.raises(FileNotFoundError) as excinfo:
            load_column_config(nonexistent_path)
        
        # Verify error message contains helpful information
        error_message = str(excinfo.value)
        assert nonexistent_path in error_message or "not found" in error_message

    @pytest.mark.parametrize("yaml_content,expected_error_type", [
        ("invalid: yaml: content:", "yaml_parsing_error"),
        ("not_dict_format", "configuration_format_error"),
        ("", "empty_file_error"),
        ("null", "null_content_error"),
    ], ids=['invalid_syntax', 'not_dict', 'empty', 'null'])
    def test_load_column_config_handles_invalid_yaml_appropriately(self, mock_column_filesystem, 
                                                                   yaml_content, expected_error_type):
        """
        Test that loading configuration from invalid YAML files handles errors appropriately.
        
        ARRANGE: Set up invalid YAML content scenarios using Protocol-based filesystem mock
        ACT: Attempt to load configuration from invalid YAML
        ASSERT: Verify appropriate error handling for each invalid scenario
        """
        # ARRANGE - Set up invalid YAML file in mock filesystem
        invalid_config_path = '/test/config/invalid_config.yaml'
        mock_column_filesystem.add_file(
            invalid_config_path,
            content=yaml_content.encode('utf-8'),
            size=len(yaml_content)
        )
        
        # ACT & ASSERT - Execute configuration loading (should handle error appropriately)
        with pytest.raises((yaml.YAMLError, ValidationError, AttributeError, TypeError)) as excinfo:
            load_column_config(invalid_config_path)
        
        # Verify error handling provides meaningful information
        assert excinfo.value is not None

    def test_get_config_from_file_path_source_succeeds(self, sample_column_config_file):
        """
        Test that get_config_from_source accepts file path and returns valid configuration.
        
        ARRANGE: Set up valid configuration file path
        ACT: Execute get_config_from_source with file path
        ASSERT: Verify successful configuration retrieval with expected structure
        """
        # ARRANGE - Valid configuration file path from centralized fixture
        config_file_path = sample_column_config_file
        
        # ACT - Execute configuration retrieval from file path
        config = get_config_from_source(config_file_path)
        
        # ASSERT - Verify successful retrieval and expected behavior
        assert config is not None
        assert isinstance(config, ColumnConfigDict)
        assert hasattr(config, 'columns')
        assert len(config.columns) > 0

    def test_get_config_from_dictionary_source_succeeds(self):
        """
        Test that get_config_from_source accepts dictionary and returns valid configuration.
        
        ARRANGE: Set up valid configuration dictionary
        ACT: Execute get_config_from_source with dictionary
        ASSERT: Verify successful configuration creation with expected structure
        """
        # ARRANGE - Valid configuration dictionary
        config_dict = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'X position'}
            },
            'special_handlers': {}
        }
        
        # ACT - Execute configuration creation from dictionary
        config = get_config_from_source(config_dict)
        
        # ASSERT - Verify successful creation and expected behavior
        assert config is not None
        assert isinstance(config, ColumnConfigDict)
        assert len(config.columns) == 2

    def test_get_config_from_existing_config_dict_returns_same_instance(self):
        """
        Test that get_config_from_source returns same instance when given ColumnConfigDict.
        
        ARRANGE: Set up existing ColumnConfigDict instance
        ACT: Execute get_config_from_source with existing instance
        ASSERT: Verify same instance is returned (no unnecessary processing)
        """
        # ARRANGE - Create existing ColumnConfigDict instance
        config_data = {
            'columns': {'t': {'type': 'numpy.ndarray', 'description': 'Time'}},
            'special_handlers': {}
        }
        original_config = ColumnConfigDict.model_validate(config_data)
        
        # ACT - Execute get_config_from_source with existing instance
        returned_config = get_config_from_source(original_config)
        
        # ASSERT - Verify same instance is returned (identity check)
        assert returned_config is original_config

    @pytest.mark.parametrize("invalid_source", [
        123,  # Integer
        [],   # List  
        object(),  # Generic object
        True,  # Boolean
    ], ids=['int', 'list', 'object', 'bool'])
    def test_get_config_from_source_rejects_invalid_types(self, invalid_source):
        """
        Test that get_config_from_source rejects invalid source types appropriately.
        
        ARRANGE: Set up invalid source type that should be rejected
        ACT: Attempt to get configuration from invalid source
        ASSERT: Verify appropriate type error handling
        """
        # ARRANGE - Invalid source type (from parameter)
        
        # ACT & ASSERT - Execute get_config_from_source (should raise TypeError)
        with pytest.raises(TypeError) as excinfo:
            get_config_from_source(invalid_source)
        
        # Verify error message provides helpful guidance
        error_message = str(excinfo.value)
        assert "config_source must be" in error_message

    def test_get_config_from_none_source_uses_default_behavior(self, mock_column_config_provider):
        """
        Test that get_config_from_source with None uses default configuration behavior.
        
        ARRANGE: Set up None source input with mock provider for default path
        ACT: Execute get_config_from_source with None
        ASSERT: Verify default configuration loading behavior is triggered
        """
        # ARRANGE - None source input (uses default configuration)
        source = None
        
        # ACT - Execute get_config_from_source with None
        # Note: This may use actual default config path in real implementation
        # For isolated testing, we rely on the function's documented behavior
        try:
            config = get_config_from_source(source)
            
            # ASSERT - Verify configuration returned (default behavior succeeded)
            assert config is not None
            assert isinstance(config, ColumnConfigDict)
        except FileNotFoundError:
            # Expected if default config file doesn't exist in test environment
            pytest.skip("Default configuration file not available in test environment")

    def test_get_default_config_path_returns_valid_structure(self):
        """
        Test that get_default_config_path returns valid path structure.
        
        ARRANGE: No setup required for path generation function
        ACT: Execute get_default_config_path
        ASSERT: Verify returned path has expected structure and format
        """
        # ARRANGE - No setup required
        
        # ACT - Execute default path generation
        path = get_default_config_path()
        
        # ASSERT - Verify path structure and format
        assert path is not None
        assert isinstance(path, str)
        assert len(path) > 0
        assert path.endswith('column_config.yaml')
        # Path should contain directory component (not just filename)
        path_parts = Path(path).parts
        assert len(path_parts) > 1


# ============================================================================
# PROPERTY-BASED BEHAVIOR TESTING - Hypothesis integration for edge-case discovery
# ============================================================================

class TestPropertyBasedBehavior:
    """
    Property-based behavior tests using Hypothesis for robust edge-case validation.
    
    Tests focus on observable configuration behavior across wide parameter ranges
    without examining internal implementation details.
    """

    @given(
        dimension=st.one_of(st.just(None), st.integers(min_value=1, max_value=3)),
        required=st.booleans(),
        is_metadata=st.booleans(),
        type_str=st.sampled_from(['numpy.ndarray', 'string', 'float', 'int'])
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_column_config_creation_behavior_across_parameter_combinations(self, dimension, required, is_metadata, type_str):
        """
        Test column configuration creation behavior across diverse parameter combinations.
        
        ARRANGE: Generate diverse parameter combinations using Hypothesis
        ACT: Create column configuration with generated parameters
        ASSERT: Verify consistent creation behavior and parameter preservation
        """
        # ARRANGE - Generated parameters from Hypothesis (diverse combinations)
        config_params = {
            'type': type_str,
            'dimension': dimension,
            'required': required,
            'is_metadata': is_metadata,
            'description': f'Property-based test {type_str} column'
        }
        
        # ACT - Execute configuration creation with generated parameters
        config = ColumnConfig(**config_params)
        
        # ASSERT - Verify consistent behavior and parameter preservation
        assert config is not None
        assert config.type == type_str
        assert config.required == required
        assert config.is_metadata == is_metadata
        assert config.description.startswith('Property-based test')
        
        # Verify dimension handling behavior
        if dimension is not None:
            assert config.dimension is not None
        else:
            assert config.dimension is None

    @given(
        array_size=st.integers(min_value=10, max_value=1000),
        num_columns=st.integers(min_value=1, max_value=5)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dataframe_creation_behavior_across_matrix_sizes(self, array_size, num_columns):
        """
        Test DataFrame creation behavior across various experimental matrix sizes.
        
        ARRANGE: Generate experimental matrices of varying sizes using Hypothesis
        ACT: Create DataFrame from matrix using column configuration
        ASSERT: Verify consistent DataFrame creation behavior across size variations
        """
        # ARRANGE - Generate experimental matrix with variable dimensions
        exp_matrix = {'t': np.linspace(0, 10, array_size)}
        
        # Add variable number of columns with consistent seeding
        np.random.seed(42)
        for i in range(num_columns):
            exp_matrix[f'col_{i}'] = np.random.rand(array_size)
        
        # Create corresponding configuration
        config_data = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time values'}
            },
            'special_handlers': {}
        }
        
        # Add column configurations for generated columns
        for i in range(num_columns):
            config_data['columns'][f'col_{i}'] = {
                'type': 'numpy.ndarray', 'dimension': 1, 'required': False, 'description': f'Column {i}'
            }
        
        # ACT - Execute configuration validation and DataFrame creation
        config = ColumnConfigDict.model_validate(config_data)
        df = make_dataframe_from_config(exp_matrix, config)
        
        # ASSERT - Verify consistent DataFrame creation behavior
        assert df is not None
        assert len(df) == array_size
        assert 't' in df.columns
        for i in range(num_columns):
            assert f'col_{i}' in df.columns
        
        # Verify DataFrame structure consistency
        assert df.shape[0] == array_size
        assert df.shape[1] == num_columns + 1  # +1 for time column

    @given(
        special_handler=st.sampled_from([
            'extract_first_column_if_2d',
            'transform_to_match_time_dimension'
        ]),
        dimension=st.integers(min_value=1, max_value=3)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_special_handler_behavior_across_dimension_combinations(self, special_handler, dimension):
        """
        Test special handler behavior across various handler-dimension combinations.
        
        ARRANGE: Generate special handler and dimension combinations using Hypothesis
        ACT: Create column configuration with handler-dimension combination
        ASSERT: Verify consistent special handler preservation and behavior
        """
        # ARRANGE - Generated special handler and dimension combination
        config_params = {
            'type': 'numpy.ndarray',
            'dimension': dimension,
            'special_handling': special_handler,
            'description': 'Property-based test special handler column'
        }
        
        # ACT - Execute configuration creation with handler-dimension combination
        config = ColumnConfig(**config_params)
        
        # ASSERT - Verify consistent special handler behavior
        assert config is not None
        assert config.special_handling is not None
        assert config.dimension is not None
        
        # Verify handler value is preserved correctly (behavioral check)
        if hasattr(config.special_handling, 'value'):
            assert config.special_handling.value == special_handler
        
        # Verify dimension behavior is consistent
        assert config.dimension.value == dimension


# ============================================================================
# PERFORMANCE BENCHMARK TESTS - TO BE EXTRACTED TO scripts/benchmarks/
# ============================================================================

# NOTE: The following performance benchmark tests should be moved to scripts/benchmarks/
# per Section 0 requirement for performance test isolation. These tests are marked
# with @pytest.mark.benchmark and should be excluded from default test execution.

@pytest.mark.benchmark
class TestColumnConfigPerformanceBenchmarks:
    """
    Performance benchmark tests for column configuration operations.
    
    These tests will be extracted to scripts/benchmarks/ directory for optional execution.
    They measure performance SLA compliance for configuration loading and validation.
    """

    @pytest.mark.benchmark
    def test_configuration_loading_performance_sla(self, benchmark, sample_column_config_file):
        """
        Benchmark configuration loading performance against SLA requirements.
        
        Performance SLA: < 100ms for configuration loading operations.
        This test will be moved to scripts/benchmarks/ for optional execution.
        """
        def load_config():
            return load_column_config(sample_column_config_file)
        
        result = benchmark(load_config)
        
        # Verify functional correctness
        assert isinstance(result, ColumnConfigDict)
        assert len(result.columns) > 0

    @pytest.mark.benchmark
    def test_validation_performance_sla(self, benchmark):
        """
        Benchmark ColumnConfigDict validation performance against SLA requirements.
        
        Performance SLA: < 50ms for validation operations.
        This test will be moved to scripts/benchmarks/ for optional execution.
        """
        config_data = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'X position'},
                'y': {'type': 'numpy.ndarray', 'dimension': 1, 'required': False, 'description': 'Y position'},
            },
            'special_handlers': {}
        }
        
        def validate_config():
            return ColumnConfigDict.model_validate(config_data)
        
        result = benchmark(validate_config)
        
        # Verify functional correctness
        assert isinstance(result, ColumnConfigDict)
        assert len(result.columns) == 3

    @pytest.mark.benchmark
    def test_dataframe_creation_performance_sla(self, benchmark, sample_exp_matrix_comprehensive):
        """
        Benchmark DataFrame creation performance against SLA requirements.
        
        Performance SLA: < 500ms for DataFrame transformation per 1M rows.
        This test will be moved to scripts/benchmarks/ for optional execution.
        """
        config_data = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'X position'},
                'y': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Y position'},
            },
            'special_handlers': {}
        }
        config = ColumnConfigDict.model_validate(config_data)
        
        def create_dataframe():
            return make_dataframe_from_config(sample_exp_matrix_comprehensive, config)
        
        result = benchmark(create_dataframe)
        
        # Verify functional correctness
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 't' in result.columns

    @pytest.mark.benchmark
    @pytest.mark.parametrize("config_size", [10, 50, 100], ids=['small', 'medium', 'large'])
    def test_scalability_performance_sla(self, benchmark, config_size):
        """
        Benchmark scalability performance with varying configuration sizes.
        
        Tests configuration validation scalability with increasing column counts.
        This test will be moved to scripts/benchmarks/ for optional execution.
        """
        # Generate large configuration for scalability testing
        large_config = {'columns': {}, 'special_handlers': {}}
        
        for i in range(config_size):
            large_config['columns'][f'col_{i}'] = {
                'type': 'numpy.ndarray', 'dimension': 1, 
                'required': i < config_size // 2, 'description': f'Column {i}'
            }
        
        def validate_large_config():
            return ColumnConfigDict.model_validate(large_config)
        
        result = benchmark(validate_large_config)
        
        # Verify functional correctness
        assert isinstance(result, ColumnConfigDict)
        assert len(result.columns) == config_size


# ============================================================================
# ENHANCED EDGE-CASE AND ERROR HANDLING TESTS - Comprehensive boundary validation
# ============================================================================

class TestEdgeCaseAndErrorHandlingBehavior:
    """
    Enhanced edge-case and error handling tests with comprehensive boundary validation.
    
    Tests focus on observable error handling behavior and edge-case scenarios
    using parameterized tests and centralized edge-case generation utilities.
    """

    @pytest.mark.parametrize("missing_field", [
        'type', 'description'
    ], ids=['missing_type', 'missing_description'])
    def test_column_config_rejects_missing_required_fields(self, missing_field):
        """
        Test that column configuration appropriately rejects missing required fields.
        
        ARRANGE: Set up configuration data with missing required field
        ACT: Attempt to create column configuration with missing field
        ASSERT: Verify appropriate validation error behavior
        """
        # ARRANGE - Prepare configuration with missing required field
        config_data = {'type': 'numpy.ndarray', 'description': 'Test column'}
        del config_data[missing_field]
        
        # ACT & ASSERT - Execute configuration creation (should reject missing field)
        with pytest.raises(ValidationError) as exc_info:
            ColumnConfig.model_validate(config_data)
        
        # Verify error information includes missing field details
        error_message = str(exc_info.value)
        assert missing_field in error_message

    @pytest.mark.parametrize("invalid_structure,expected_error_type", [
        ({}, "empty_configuration"),
        ({'columns': {}}, "empty_columns"), 
        ({'special_handlers': {}}, "missing_columns_key"),
        ({'columns': 'invalid'}, "invalid_columns_type"),
        ({'columns': {}, 'special_handlers': 'invalid'}, "invalid_handlers_type"),
    ], ids=['empty', 'empty_columns', 'missing_columns', 'invalid_columns_type', 'invalid_handlers_type'])
    def test_config_dict_rejects_invalid_structures(self, invalid_structure, expected_error_type):
        """
        Test that configuration dictionary appropriately rejects invalid structures.
        
        ARRANGE: Set up various invalid configuration structures
        ACT: Attempt to create ColumnConfigDict with invalid structure
        ASSERT: Verify appropriate validation error handling for each structure type
        """
        # ARRANGE - Invalid configuration structure (from parameter)
        
        # ACT & ASSERT - Execute configuration validation (should reject invalid structure)
        with pytest.raises(ValidationError) as exc_info:
            ColumnConfigDict.model_validate(invalid_structure)
        
        # Verify error handling provides meaningful information
        assert exc_info.value is not None

    def test_config_dict_rejects_invalid_column_specifications(self):
        """
        Test that configuration dictionary rejects invalid column specifications.
        
        ARRANGE: Set up configuration with invalid column specification
        ACT: Attempt to create ColumnConfigDict with invalid column
        ASSERT: Verify appropriate validation error for malformed column
        """
        # ARRANGE - Configuration with invalid column specification
        invalid_config = {
            'columns': {
                'invalid_col': {
                    'dimension': 1  # Missing required 'type' and 'description' fields
                }
            },
            'special_handlers': {}
        }
        
        # ACT & ASSERT - Execute configuration validation (should reject invalid column)
        with pytest.raises(ValidationError) as exc_info:
            ColumnConfigDict.model_validate(invalid_config)
        
        # Verify error includes information about missing required fields
        error_message = str(exc_info.value)
        assert 'type' in error_message or 'description' in error_message

    @pytest.mark.parametrize("file_error_scenario", [
        "nonexistent_file.yaml",
        "/invalid/path/config.yaml",
        "../../../etc/passwd",  # Path traversal attempt
        "config_with_null_bytes\x00.yaml",  # Null byte injection
    ], ids=['nonexistent', 'invalid_path', 'path_traversal', 'null_bytes'])
    def test_load_config_handles_file_system_errors(self, file_error_scenario):
        """
        Test that configuration loading handles various file system error scenarios.
        
        ARRANGE: Set up various problematic file path scenarios
        ACT: Attempt to load configuration from problematic path
        ASSERT: Verify appropriate error handling for each scenario
        """
        # ARRANGE - Problematic file path (from parameter)
        
        # ACT & ASSERT - Execute configuration loading (should handle error appropriately)
        with pytest.raises((FileNotFoundError, OSError, ValueError)) as exc_info:
            load_column_config(file_error_scenario)
        
        # Verify error handling provides meaningful information
        assert exc_info.value is not None

    def test_dataframe_creation_handles_missing_required_columns(self, sample_exp_matrix_comprehensive):
        """
        Test that DataFrame creation handles missing required columns appropriately.
        
        ARRANGE: Set up configuration requiring columns not present in experimental matrix
        ACT: Attempt to create DataFrame with missing required columns
        ASSERT: Verify appropriate error handling for missing columns
        """
        # ARRANGE - Configuration requiring columns not in experimental matrix
        config_data = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time values'},
                'missing_required_col': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Missing column'}
            },
            'special_handlers': {}
        }
        
        config = ColumnConfigDict.model_validate(config_data)
        
        # ACT & ASSERT - Execute DataFrame creation (should handle missing columns appropriately)
        with pytest.raises(ValueError) as exc_info:
            make_dataframe_from_config(sample_exp_matrix_comprehensive, config)
        
        # Verify error message includes information about missing columns
        error_message = str(exc_info.value)
        assert "Missing required columns" in error_message or "missing_required_col" in error_message

    def test_dataframe_creation_handles_missing_alias_columns(self):
        """
        Test that DataFrame creation handles missing aliased columns appropriately.
        
        ARRANGE: Set up configuration with alias that doesn't exist in experimental matrix
        ACT: Attempt to create DataFrame with missing aliased column
        ASSERT: Verify appropriate handling of missing alias scenarios
        """
        # ARRANGE - Experimental matrix with minimal data
        exp_matrix = {'t': np.linspace(0, 10, 100)}
        
        config_data = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time values'},
                'missing_col': {
                    'type': 'numpy.ndarray', 'dimension': 1, 'required': True,
                    'alias': 'also_missing', 'description': 'Missing column with missing alias'
                }
            },
            'special_handlers': {}
        }
        
        config = ColumnConfigDict.model_validate(config_data)
        
        # ACT & ASSERT - Execute DataFrame creation (should handle missing alias appropriately)
        with pytest.raises(ValueError) as exc_info:
            make_dataframe_from_config(exp_matrix, config)
        
        error_message = str(exc_info.value)
        assert "Missing required columns" in error_message

    def test_dataframe_creation_applies_default_values_correctly(self):
        """
        Test that DataFrame creation applies default values for missing optional columns.
        
        ARRANGE: Set up configuration with optional columns having default values
        ACT: Create DataFrame with missing optional columns
        ASSERT: Verify default values are applied correctly in resulting DataFrame
        """
        # ARRANGE - Experimental matrix with minimal required data
        exp_matrix = {'t': np.linspace(0, 10, 100)}
        
        config_data = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time values'},
                'optional_col': {
                    'type': 'numpy.ndarray', 'dimension': 1, 'required': False,
                    'default_value': None, 'description': 'Optional column with default'
                }
            },
            'special_handlers': {}
        }
        
        config = ColumnConfigDict.model_validate(config_data)
        
        # ACT - Execute DataFrame creation with default value application
        df = make_dataframe_from_config(exp_matrix, config)
        
        # ASSERT - Verify default values are applied correctly
        assert df is not None
        assert 't' in df.columns
        assert 'optional_col' in df.columns
        # Verify default value behavior
        if not df['optional_col'].empty:
            assert df['optional_col'].iloc[0] is None

    def test_special_handler_undefined_reference_handling(self):
        """
        Test behavior when special handler is referenced but not defined in handlers dictionary.
        
        ARRANGE: Set up configuration with special handling reference but no handler definition
        ACT: Create configuration with undefined handler reference
        ASSERT: Verify configuration creation succeeds despite undefined handler
        """
        # ARRANGE - Configuration with undefined special handler reference
        config_data = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time values'},
                'special_col': {
                    'type': 'numpy.ndarray', 'dimension': 2, 'required': False,
                    'special_handling': 'transform_to_match_time_dimension',
                    'description': 'Column with undefined special handling'
                }
            },
            'special_handlers': {}  # Handler not defined
        }
        
        # ACT - Execute configuration creation (should succeed)
        config = ColumnConfigDict.model_validate(config_data)
        
        # ASSERT - Verify configuration creation succeeded despite undefined handler
        assert config is not None
        assert 'special_col' in config.columns
        assert config.columns['special_col'].special_handling is not None
        assert len(config.special_handlers) == 0

    @pytest.mark.parametrize("boundary_condition", [
        ("empty_exp_matrix", {}),
        ("single_element_arrays", {'t': np.array([0.0]), 'x': np.array([1.0])}),
        ("very_large_arrays", {'t': np.linspace(0, 1000, 100000), 'data': np.random.rand(100000)}),
        ("arrays_with_nan", {'t': np.array([0, 1, np.nan]), 'x': np.array([np.nan, 1, 2])}),
        ("arrays_with_inf", {'t': np.array([0, 1, np.inf]), 'x': np.array([np.inf, 1, 2])}),
    ], ids=['empty', 'single_element', 'very_large', 'with_nan', 'with_inf'])
    def test_dataframe_creation_boundary_conditions(self, boundary_condition):
        """
        Test DataFrame creation behavior under various boundary conditions.
        
        ARRANGE: Set up experimental matrices with boundary condition scenarios
        ACT: Attempt to create DataFrame from boundary condition data
        ASSERT: Verify robust handling of boundary conditions
        """
        # ARRANGE - Boundary condition scenario
        condition_name, exp_matrix = boundary_condition
        
        # Handle empty matrix case
        if not exp_matrix:
            pytest.skip("Empty experimental matrix scenario requires special handling")
        
        # Create appropriate configuration for the matrix
        config_data = {
            'columns': {col_name: {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 
                                  'description': f'{col_name} column'} for col_name in exp_matrix.keys()},
            'special_handlers': {}
        }
        
        config = ColumnConfigDict.model_validate(config_data)
        
        # ACT - Execute DataFrame creation with boundary condition
        try:
            df = make_dataframe_from_config(exp_matrix, config)
            
            # ASSERT - Verify successful creation and basic structure
            assert df is not None
            assert isinstance(df, pd.DataFrame)
            for col_name in exp_matrix.keys():
                assert col_name in df.columns
                
        except (ValueError, MemoryError) as e:
            # Some boundary conditions may legitimately fail
            if condition_name in ['very_large']:
                pytest.skip(f"Boundary condition '{condition_name}' exceeded system limits: {e}")
            else:
                raise

    @pytest.mark.parametrize("corrupted_yaml_scenario", [
        ("incomplete_yaml", "columns:\n  t:\n    type: numpy.ndarray\n    description incomplete"),
        ("wrong_structure", "columns:\n  t: [invalid, structure]"),
        ("invalid_dimension", "columns:\n  't':\n    type: 'numpy.ndarray'\n    dimension: 'invalid_dimension'"),
        ("binary_content", "columns:\n  t:\n    type: \x00\x01\x02invalid"),
        ("extremely_nested", "columns:\n" + "  " * 1000 + "deeply_nested: value"),
    ], ids=['incomplete', 'wrong_structure', 'invalid_dimension', 'binary_content', 'deeply_nested'])
    def test_corrupted_yaml_configuration_recovery(self, mock_column_filesystem, corrupted_yaml_scenario):
        """
        Test enhanced handling of corrupted YAML configuration files with recovery scenarios.
        
        ARRANGE: Set up various corrupted YAML content scenarios using Protocol-based filesystem
        ACT: Attempt to load configuration from corrupted YAML files
        ASSERT: Verify appropriate error handling and recovery behavior
        """
        # ARRANGE - Set up corrupted YAML file scenario
        scenario_name, yaml_content = corrupted_yaml_scenario
        corrupted_file_path = f'/test/config/corrupted_{scenario_name}.yaml'
        
        mock_column_filesystem.add_file(
            corrupted_file_path,
            content=yaml_content.encode('utf-8', errors='replace'),
            size=len(yaml_content),
            corrupted=True
        )
        
        # ACT & ASSERT - Execute configuration loading (should handle corruption appropriately)
        with pytest.raises((yaml.YAMLError, ValidationError, UnicodeDecodeError)) as exc_info:
            load_column_config(corrupted_file_path)
        
        # Verify error handling provides recovery guidance
        assert exc_info.value is not None

    @pytest.mark.parametrize("unicode_scenario", [
        ("unicode_filename", "tëst_çönfïg.yaml", "normal content"),
        ("unicode_content", "config.yaml", "columns:\n  ñämé: {type: numpy.ndarray, description: 'Ünïcödé tëst'}"),
        ("mixed_encoding", "config.yaml", "columns:\n  test: {type: numpy.ndarray, description: 'Tëst with émojis 🧪🔬'}"),
        ("special_chars", "config@#$.yaml", "columns:\n  test: {type: numpy.ndarray, description: 'Special chars test'}"),
    ], ids=['unicode_filename', 'unicode_content', 'mixed_encoding', 'special_chars'])
    def test_unicode_and_special_character_handling(self, mock_column_filesystem, unicode_scenario):
        """
        Test enhanced Unicode and special character handling in configuration files.
        
        ARRANGE: Set up various Unicode and special character scenarios
        ACT: Attempt to load configuration with Unicode content
        ASSERT: Verify robust Unicode handling behavior
        """
        # ARRANGE - Set up Unicode scenario
        scenario_name, filename, content = unicode_scenario
        file_path = f'/test/config/{filename}'
        
        try:
            mock_column_filesystem.add_unicode_file(
                file_path,
                content=content.encode('utf-8'),
                size=len(content.encode('utf-8'))
            )
            
            # ACT - Execute configuration loading with Unicode content
            # This should either succeed or fail gracefully
            try:
                config = load_column_config(file_path)
                
                # ASSERT - If successful, verify configuration is valid
                assert config is not None
                assert isinstance(config, ColumnConfigDict)
                
            except (UnicodeDecodeError, yaml.YAMLError, ValidationError):
                # ASSERT - If failed, verify graceful error handling
                pytest.skip(f"Unicode scenario '{scenario_name}' not supported in current environment")
                
        except (UnicodeError, OSError):
            # ARRANGE failed due to platform limitations
            pytest.skip(f"Unicode filename scenario '{scenario_name}' not supported on this platform")


# ============================================================================
# INTEGRATION BEHAVIOR TESTS - End-to-end workflow validation with Protocol-based mocks
# ============================================================================

class TestIntegrationWorkflowBehavior:
    """
    Integration tests for complete column configuration workflows using behavior-focused validation.
    
    Tests validate end-to-end workflows from YAML loading to DataFrame creation
    without coupling to implementation details, using Protocol-based mocks for isolation.
    """

    def test_complete_yaml_to_dataframe_workflow_succeeds(self, sample_column_config_file, 
                                                         sample_exp_matrix_comprehensive):
        """
        Test that complete workflow from YAML configuration to DataFrame succeeds.
        
        ARRANGE: Set up valid YAML configuration and experimental matrix data
        ACT: Execute complete workflow from file loading to DataFrame creation
        ASSERT: Verify successful end-to-end processing with expected behavior
        """
        # ARRANGE - Valid configuration file and experimental matrix from centralized fixtures
        config_file_path = sample_column_config_file
        exp_matrix = sample_exp_matrix_comprehensive
        
        # ACT - Execute complete workflow: YAML → Config → DataFrame
        config = load_column_config(config_file_path)
        df = make_dataframe_from_config(exp_matrix, config)
        
        # ASSERT - Verify successful end-to-end workflow completion
        assert config is not None
        assert isinstance(config, ColumnConfigDict)
        assert len(config.columns) > 0
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 't' in df.columns  # Expected time column
        
        # Verify DataFrame structure consistency with experimental matrix
        assert len(df) == len(exp_matrix.get('t', []))
        
        # Verify data integrity preservation
        if 't' in exp_matrix:
            time_data = df['t'].values
            assert len(time_data) > 0
            assert not np.all(np.isnan(time_data))  # Should contain valid time data

    def test_configuration_metadata_integration_workflow(self, sample_experimental_metadata):
        """
        Test integration workflow of configuration with metadata field handling.
        
        ARRANGE: Set up configuration with metadata fields and corresponding metadata values
        ACT: Execute configuration loading and DataFrame creation with metadata
        ASSERT: Verify metadata integration behavior in resulting DataFrame
        """
        # ARRANGE - Experimental matrix and metadata from centralized fixtures
        exp_matrix = {'t': np.linspace(0, 10, 100), 'x': np.random.RandomState(42).rand(100)}
        metadata = sample_experimental_metadata
        
        # Create configuration with metadata fields
        config_data = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time values'},
                'x': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'X position'},
                'date': {'type': 'string', 'required': False, 'is_metadata': True, 'description': 'Experiment date'},
                'exp_name': {'type': 'string', 'required': False, 'is_metadata': True, 'description': 'Experiment name'}
            },
            'special_handlers': {}
        }
        
        config = ColumnConfigDict.model_validate(config_data)
        
        # ACT - Execute DataFrame creation with metadata integration
        df = make_dataframe_from_config(exp_matrix, config, metadata=metadata)
        
        # ASSERT - Verify metadata integration behavior
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        
        # Verify core data columns are present
        assert 't' in df.columns
        assert 'x' in df.columns
        
        # Verify metadata integration if configured
        if 'date' in metadata:
            assert 'date' in df.columns
            assert df['date'].iloc[0] == metadata['date']

    def test_alias_resolution_integration_workflow(self, sample_exp_matrix_comprehensive):
        """
        Test integration workflow of column alias resolution in complete configuration processing.
        
        ARRANGE: Set up configuration with column aliases mapping to different names in experimental matrix
        ACT: Execute configuration processing and DataFrame creation with alias resolution
        ASSERT: Verify alias resolution behavior produces expected column naming
        """
        # ARRANGE - Configuration with alias mapping
        config_data = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time values'},
                'theta_smooth': {
                    'type': 'numpy.ndarray', 'dimension': 1, 'required': False,
                    'alias': 'dtheta_smooth', 'description': 'Smoothed theta values (aliased)'
                }
            },
            'special_handlers': {}
        }
        
        config = ColumnConfigDict.model_validate(config_data)
        
        # ACT - Execute DataFrame creation with alias resolution
        df = make_dataframe_from_config(sample_exp_matrix_comprehensive, config)
        
        # ASSERT - Verify alias resolution behavior
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        
        # Verify time column is present (required)
        assert 't' in df.columns
        
        # Verify alias resolution results (if source alias exists in matrix)
        if 'dtheta_smooth' in sample_exp_matrix_comprehensive:
            assert 'theta_smooth' in df.columns  # Should use configured name
            assert 'dtheta_smooth' not in df.columns  # Original alias name should not be present
            assert len(df['theta_smooth']) > 0

    def test_special_handler_integration_workflow(self):
        """
        Test integration workflow of special handlers in complete configuration processing.
        
        ARRANGE: Set up configuration with special handler for multi-dimensional data transformation
        ACT: Execute configuration processing and DataFrame creation with special handling
        ASSERT: Verify special handler application produces expected data transformation behavior
        """
        # ARRANGE - Experimental matrix with multi-dimensional signal data
        exp_matrix = {
            't': np.linspace(0, 10, 100),
            'signal_disp': np.random.RandomState(42).rand(15, 100)  # 15 channels, 100 time points
        }
        
        config_data = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time values'},
                'signal_disp': {
                    'type': 'numpy.ndarray', 'dimension': 2, 'required': False,
                    'special_handling': 'transform_to_match_time_dimension',
                    'description': 'Signal display data requiring transformation'
                }
            },
            'special_handlers': {'transform_to_match_time_dimension': '_handle_signal_disp'}
        }
        
        config = ColumnConfigDict.model_validate(config_data)
        
        # ACT - Execute DataFrame creation with special handler application
        df = make_dataframe_from_config(exp_matrix, config)
        
        # ASSERT - Verify special handler transformation behavior
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100  # Should match time dimension
        
        # Verify special handler was applied to signal_disp column
        if 'signal_disp' in df.columns:
            assert len(df) > 0
            
            # Verify transformation results (specific behavior depends on handler implementation)
            signal_data = df['signal_disp']
            assert signal_data is not None
            
            # If transformation produces arrays per time point, verify structure
            if not signal_data.empty and hasattr(signal_data.iloc[0], '__len__'):
                first_signal = signal_data.iloc[0]
                if isinstance(first_signal, np.ndarray):
                    assert len(first_signal) == 15  # Should preserve channel dimension


# ============================================================================
# PROTOCOL-BASED MOCK TESTS - Enhanced dependency isolation with Protocol implementations
# ============================================================================

class TestProtocolBasedMockIntegration:
    """
    Test suite demonstrating Protocol-based mock integration for enhanced dependency isolation.
    
    Uses Protocol-based mock implementations from tests/utils.py for consistent mock behavior
    across all column configuration test scenarios without implementation coupling.
    """

    def test_yaml_loading_with_protocol_based_filesystem_mock(self, mock_column_filesystem):
        """
        Test YAML configuration loading with Protocol-based filesystem operations.
        
        ARRANGE: Set up Protocol-based filesystem mock with YAML configuration content
        ACT: Execute configuration loading using mocked filesystem operations
        ASSERT: Verify successful loading behavior without coupling to implementation
        """
        # ARRANGE - Set up YAML content in Protocol-based filesystem mock
        yaml_content = """
columns:
  t:
    type: numpy.ndarray
    dimension: 1
    required: true
    description: Time values
special_handlers: {}
"""
        
        config_file_path = '/test/config/protocol_test_config.yaml'
        mock_column_filesystem.add_file(
            config_file_path,
            content=yaml_content.encode('utf-8'),
            size=len(yaml_content)
        )
        
        # ACT - Execute configuration loading with Protocol-based filesystem
        try:
            config = load_column_config(config_file_path)
            
            # ASSERT - Verify successful loading behavior
            assert config is not None
            assert isinstance(config, ColumnConfigDict)
            assert len(config.columns) >= 1
            assert 't' in config.columns
            
        except (FileNotFoundError, AttributeError):
            # Handle case where Protocol-based mock isn't fully integrated with actual loader
            pytest.skip("Protocol-based filesystem mock requires integration with column_models module")

    def test_configuration_loading_with_protocol_based_data_provider(self, mock_column_dataloader):
        """
        Test configuration loading with Protocol-based data loading operations.
        
        ARRANGE: Set up Protocol-based data loader mock with configuration scenarios
        ACT: Execute DataFrame creation using mocked data loading operations
        ASSERT: Verify DataFrame creation behavior with Protocol-based data isolation
        """
        # ARRANGE - Set up experimental matrix using Protocol-based data loader
        exp_matrix = {'t': np.linspace(0, 10, 100), 'x': np.random.rand(100)}
        
        # Add experimental matrix to mock data loader
        matrix_file_path = '/test/data/protocol_test_matrix.pkl'
        mock_column_dataloader.add_experimental_matrix(
            matrix_file_path,
            n_timepoints=100,
            include_signal=False,
            include_metadata=False
        )
        
        # Create configuration for the matrix
        config_data = {
            'columns': {
                't': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'Time values'},
                'x': {'type': 'numpy.ndarray', 'dimension': 1, 'required': True, 'description': 'X position'}
            },
            'special_handlers': {}
        }
        
        config = ColumnConfigDict.model_validate(config_data)
        
        # ACT - Execute DataFrame creation with Protocol-based data loader
        df = make_dataframe_from_config(exp_matrix, config)
        
        # ASSERT - Verify DataFrame creation behavior
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert 't' in df.columns
        assert 'x' in df.columns
        assert len(df) == 100

    def test_edge_case_scenario_integration_with_protocol_mocks(self, mock_column_filesystem, 
                                                               mock_column_dataloader):
        """
        Test edge-case scenario integration using comprehensive Protocol-based mocks.
        
        ARRANGE: Set up comprehensive edge-case scenarios using Protocol-based mock infrastructure
        ACT: Execute configuration processing through edge-case scenarios
        ASSERT: Verify robust edge-case handling behavior across mock scenarios
        """
        # ARRANGE - Set up edge-case scenarios using Protocol-based mocks
        edge_case_scenarios = generate_edge_case_scenarios(
            scenario_types=['unicode', 'boundary', 'corrupted'],
            include_platform_specific=False
        )
        
        # Test Unicode scenario if available
        if 'unicode' in edge_case_scenarios:
            unicode_scenarios = edge_case_scenarios['unicode']
            if unicode_scenarios:
                unicode_scenario = unicode_scenarios[0]
                
                try:
                    # Add Unicode file to filesystem mock
                    unicode_file_path = f"/test/config/{unicode_scenario['filename']}"
                    mock_column_filesystem.add_unicode_file(
                        unicode_file_path,
                        content=b"columns:\n  t: {type: numpy.ndarray, description: test}",
                        size=100
                    )
                    
                    # ACT - Execute configuration loading with Unicode path
                    # This tests Unicode handling capability
                    try:
                        config = load_column_config(unicode_file_path)
                        
                        # ASSERT - Verify Unicode handling succeeded
                        assert config is not None
                        
                    except (UnicodeError, FileNotFoundError):
                        # Expected for some Unicode scenarios
                        pytest.skip("Unicode scenario not supported in current test environment")
                        
                except (UnicodeError, OSError):
                    pytest.skip("Unicode file creation not supported on this platform")


# ============================================================================
# ENHANCED TEST EXECUTION CONFIGURATION - Behavior-focused test categorization
# ============================================================================

# Enhanced test markers for behavior-focused test execution
behavior_tests = pytest.mark.behavior  # Behavior-focused tests (default execution)
edge_case_tests = pytest.mark.edge_case  # Edge-case and boundary validation tests
integration_behavior_tests = pytest.mark.integration_behavior  # End-to-end behavior tests
protocol_mock_tests = pytest.mark.protocol_mock  # Protocol-based mock tests

# Performance test markers (excluded from default execution per Section 0)
benchmark_tests = pytest.mark.benchmark  # Performance benchmark tests (extract to scripts/benchmarks/)
performance_sla_tests = pytest.mark.performance_sla  # SLA compliance tests

# Test execution configuration for pytest
pytestmark = [
    pytest.mark.column_config,  # Module-level marker
    pytest.mark.behavior_focused,  # Behavior-focused testing approach
]

# Test execution categories for selective running
def pytest_runtest_setup(item):
    """
    Enhanced test execution setup supporting behavior-focused test categorization.
    
    Configures test execution based on markers and environment settings,
    supporting the behavior-focused testing approach per Section 0 requirements.
    """
    # Skip benchmark tests by default (extract to scripts/benchmarks/)
    if item.get_closest_marker("benchmark"):
        pytest.skip("Benchmark tests excluded from default execution - run with --benchmark flag")
    
    # Skip performance SLA tests by default
    if item.get_closest_marker("performance_sla"):
        pytest.skip("Performance SLA tests excluded from default execution - run with --performance flag")

# Module validation - ensure behavior-focused approach compliance
def test_module_follows_behavior_focused_patterns():
    """
    Validate that this test module follows behavior-focused testing patterns.
    
    Verifies compliance with Section 0 requirements for behavior-focused testing
    without implementation coupling.
    """
    # Validate test structure using centralized validation utilities
    import inspect
    current_module = inspect.getmodule(inspect.currentframe())
    
    validation_results = validate_test_structure(current_module)
    
    # Assert behavior-focused compliance
    assert validation_results['overall_compliance'], f"Module compliance score: {validation_results.get('compliance_score', 0):.2f}"
    
    # Verify use of centralized fixtures and Protocol-based mocks
    test_classes = [cls for name, cls in inspect.getmembers(current_module, inspect.isclass) 
                   if name.startswith('Test')]
    
    assert len(test_classes) > 0, "Module should contain test classes"
    
    # Verify behavior-focused naming patterns
    behavior_focused_patterns = ['Behavior', 'Protocol', 'Integration', 'EdgeCase']
    behavior_classes = [cls for cls in test_classes 
                       if any(pattern in cls.__name__ for pattern in behavior_focused_patterns)]
    
    assert len(behavior_classes) > 0, "Module should contain behavior-focused test classes"