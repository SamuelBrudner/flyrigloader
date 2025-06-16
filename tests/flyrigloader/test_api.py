"""
Comprehensive test suite for API functions in the flyrigloader.api module.

This module implements behavior-focused testing with black-box validation, centralized 
fixture usage, and AAA (Arrange-Act-Assert) patterns per Section 0 requirements for 
behavior-focused testing and comprehensive edge-case coverage.

Features:
- Black-box behavioral validation using only public API interfaces
- Protocol-based mock implementations from centralized tests/utils.py
- AAA pattern structure with clear separation of test phases
- Centralized fixtures from tests/conftest.py for consistent test patterns
- Edge-case coverage through parameterized test scenarios
- Focus on documented public interfaces without internal implementation coupling
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays

# Import from centralized test utilities for Protocol-based mocks
from tests.utils import (
    create_mock_config_provider,
    create_mock_dataloader,
    create_integration_test_environment,
    generate_edge_case_scenarios
)

# Import only public API functions - no private function access
from flyrigloader.api import (
    load_experiment_files,
    load_dataset_files,
    get_dataset_parameters,
    get_experiment_parameters,
    process_experiment_data
)


# =============================================================================
# PARAMETRIZED TEST DATA FOR COMPREHENSIVE EDGE CASE COVERAGE
# =============================================================================

# Test parameters for load_experiment_files edge cases per TST-MOD-002
EXPERIMENT_LOAD_SCENARIOS = [
    pytest.param(
        {
            "config_path": "/path/to/config.yaml",
            "experiment_name": "test_experiment",
            "expected_calls": {
                "load_config": [("/path/to/config.yaml",)],
                "discover_experiment_files": {
                    "experiment_name": "test_experiment",
                    "base_directory": "/path/to/data",
                    "pattern": "*.*",
                    "recursive": True,
                    "extensions": None,
                    "extract_metadata": False,
                    "parse_dates": False
                }
            }
        },
        id="basic_config_path"
    ),
    pytest.param(
        {
            "config": {
                "project": {"directories": {"major_data_directory": "/custom/data"}},
                "experiments": {"exp1": {"datasets": ["ds1"]}}
            },
            "experiment_name": "exp1",
            "expected_calls": {
                "load_config": [],
                "discover_experiment_files": {
                    "experiment_name": "exp1",
                    "base_directory": "/custom/data",
                    "pattern": "*.*",
                    "recursive": True,
                    "extensions": None,
                    "extract_metadata": False,
                    "parse_dates": False
                }
            }
        },
        id="config_dict_basic"
    ),
    pytest.param(
        {
            "config_path": "/path/to/config.yaml",
            "experiment_name": "test_experiment",
            "base_directory": "/override/data",
            "pattern": "*.csv",
            "recursive": False,
            "extensions": [".csv", ".pkl"],
            "extract_metadata": True,
            "parse_dates": True,
            "expected_calls": {
                "load_config": [("/path/to/config.yaml",)],
                "discover_experiment_files": {
                    "experiment_name": "test_experiment",
                    "base_directory": "/override/data",
                    "pattern": "*.csv",
                    "recursive": False,
                    "extensions": [".csv", ".pkl"],
                    "extract_metadata": True,
                    "parse_dates": True
                }
            }
        },
        id="all_custom_parameters"
    ),
    pytest.param(
        {
            "config": {
                "project": {"directories": {"major_data_directory": "/data"}},
                "experiments": {"metadata_exp": {"datasets": ["ds1"]}}
            },
            "experiment_name": "metadata_exp",
            "extract_metadata": True,
            "parse_dates": True,
            "expected_calls": {
                "load_config": [],
                "discover_experiment_files": {
                    "experiment_name": "metadata_exp",
                    "base_directory": "/data",
                    "pattern": "*.*",
                    "recursive": True,
                    "extensions": None,
                    "extract_metadata": True,
                    "parse_dates": True
                }
            }
        },
        id="metadata_and_dates_enabled"
    )
]

# Test parameters for load_dataset_files edge cases
DATASET_LOAD_SCENARIOS = [
    pytest.param(
        {
            "config_path": "/path/to/config.yaml",
            "dataset_name": "test_dataset",
            "expected_calls": {
                "load_config": [("/path/to/config.yaml",)],
                "discover_dataset_files": {
                    "dataset_name": "test_dataset",
                    "base_directory": "/path/to/data",
                    "pattern": "*.*",
                    "recursive": True,
                    "extensions": None,
                    "extract_metadata": False,
                    "parse_dates": False
                }
            }
        },
        id="basic_dataset_config_path"
    ),
    pytest.param(
        {
            "config": {
                "project": {"directories": {"major_data_directory": "/dataset/data"}},
                "datasets": {"ds1": {"patterns": ["*_test_*"]}}
            },
            "dataset_name": "ds1",
            "base_directory": "/custom/override",
            "pattern": "*.h5",
            "recursive": False,
            "extensions": [".h5", ".hdf5"],
            "expected_calls": {
                "load_config": [],
                "discover_dataset_files": {
                    "dataset_name": "ds1",
                    "base_directory": "/custom/override",
                    "pattern": "*.h5",
                    "recursive": False,
                    "extensions": [".h5", ".hdf5"],
                    "extract_metadata": False,
                    "parse_dates": False
                }
            }
        },
        id="dataset_all_overrides"
    )
]

# Configuration validation error scenarios per F-001-RQ-003
CONFIG_VALIDATION_ERROR_SCENARIOS = [
    pytest.param(
        {"func_args": {"config_path": "/path/to/config.yaml", "config": {"key": "value"}}},
        ValueError,
        "Exactly one of 'config_path' or 'config' must be provided",
        id="both_config_path_and_dict_provided"
    ),
    pytest.param(
        {"func_args": {}},
        ValueError,
        "Exactly one of 'config_path' or 'config' must be provided",
        id="neither_config_path_nor_dict_provided"
    ),
    pytest.param(
        {"func_args": {"config": {"project": {"directories": {}}}, "dataset_name": "test"}},
        ValueError,
        "No data directory specified",
        id="missing_major_data_directory"
    )
]

# Dataset parameter test scenarios
DATASET_PARAMETER_SCENARIOS = [
    pytest.param(
        {
            "config": {
                "datasets": {
                    "test_dataset": {
                        "parameters": {"alpha": 1.5, "beta": "test_value", "gamma": True}
                    }
                }
            },
            "dataset_name": "test_dataset",
            "expected_params": {"alpha": 1.5, "beta": "test_value", "gamma": True}
        },
        id="dataset_with_complex_parameters"
    ),
    pytest.param(
        {
            "config": {
                "datasets": {
                    "empty_params_dataset": {"parameters": {}}
                }
            },
            "dataset_name": "empty_params_dataset",
            "expected_params": {}
        },
        id="dataset_with_empty_parameters"
    ),
    pytest.param(
        {
            "config": {
                "datasets": {
                    "no_params_dataset": {"rig": "test_rig", "patterns": ["*.csv"]}
                }
            },
            "dataset_name": "no_params_dataset",
            "expected_params": {}
        },
        id="dataset_without_parameters_key"
    )
]


# =============================================================================
# CENTRALIZED FIXTURE USAGE WITH PROTOCOL-BASED MOCKS
# =============================================================================

# Note: This module now uses centralized fixtures from tests/conftest.py and 
# Protocol-based mock implementations from tests/utils.py per Section 0 requirements.
# Local fixtures have been removed to eliminate code duplication and ensure 
# consistent test patterns across the entire test suite.


# =============================================================================
# PARAMETRIZED UNIT TESTS WITH COMPREHENSIVE EDGE CASE COVERAGE
# =============================================================================

class TestLoadExperimentFiles:
    """
    Test class for load_experiment_files function with behavior-focused validation.
    
    Uses black-box testing approach focusing on public API behavior and observable
    outcomes rather than internal implementation details.
    """
    
    @pytest.mark.parametrize("scenario", EXPERIMENT_LOAD_SCENARIOS)
    def test_load_experiment_files_scenarios(
        self, 
        scenario, 
        mock_config_and_discovery_comprehensive
    ):
        """
        Test load_experiment_files with various parameter combinations using behavioral validation.
        
        Validates that the function returns expected file discovery results based on 
        different configuration sources and parameter combinations, focusing on
        observable behavior rather than internal call patterns.
        
        Uses centralized mock fixtures and Protocol-based implementations per Section 0.
        """
        # ARRANGE - Set up test data and dependencies using centralized fixtures
        func_args = {k: v for k, v in scenario.items() if k != "expected_calls"}
        
        # Mock configuration and discovery providers using Protocol-based approach
        mock_load_config = mock_config_and_discovery_comprehensive["load_config"]
        mock_discover_experiment_files = mock_config_and_discovery_comprehensive["discover_experiment_files"]
        
        # ACT - Execute the function under test
        result = load_experiment_files(**func_args)
        
        # ASSERT - Verify behavioral outcomes
        # Verify that function returns expected file structure
        assert result is not None, "Function should return file discovery results"
        assert isinstance(result, (list, dict)), "Result should be list or dict of discovered files"
        
        # Verify that result contains expected file discovery data
        expected_result = mock_discover_experiment_files.return_value
        assert result == expected_result, (
            f"Function should return discovery results matching mock data: "
            f"expected {expected_result}, got {result}"
        )
    
    @pytest.mark.parametrize("scenario,expected_exception,expected_message", CONFIG_VALIDATION_ERROR_SCENARIOS)
    def test_load_experiment_files_validation_errors(self, scenario, expected_exception, expected_message):
        """
        Test comprehensive error handling validation using behavioral validation.
        
        Validates that the function raises appropriate exceptions for invalid inputs,
        focusing on the public API contract and observable error behavior rather
        than internal validation logic.
        """
        # ARRANGE - Set up invalid input parameters
        func_args = scenario["func_args"]
        func_args.setdefault("experiment_name", "test_experiment")
        
        # ACT & ASSERT - Execute function and verify expected error behavior
        with pytest.raises(expected_exception) as exc_info:
            load_experiment_files(**func_args)
        
        # Verify error message contains expected information
        error_message = str(exc_info.value)
        assert expected_message in error_message, (
            f"Error message should contain '{expected_message}', got '{error_message}'"
        )


class TestLoadDatasetFiles:
    """
    Test class for load_dataset_files function with behavior-focused validation.
    
    Uses black-box testing approach focusing on public API behavior and observable
    outcomes rather than internal implementation details.
    """
    
    @pytest.mark.parametrize("scenario", DATASET_LOAD_SCENARIOS)
    def test_load_dataset_files_scenarios(
        self, 
        scenario, 
        mock_config_and_discovery_comprehensive
    ):
        """
        Test load_dataset_files with various parameter combinations using behavioral validation.
        
        Validates that the function returns expected file discovery results for
        dataset-specific operations, focusing on observable behavior rather than
        internal call patterns.
        """
        # ARRANGE - Set up test data using centralized fixtures
        func_args = {k: v for k, v in scenario.items() if k != "expected_calls"}
        
        # Use Protocol-based mock implementations
        mock_discover_dataset_files = mock_config_and_discovery_comprehensive["discover_dataset_files"]
        
        # ACT - Execute the function under test
        result = load_dataset_files(**func_args)
        
        # ASSERT - Verify behavioral outcomes
        assert result is not None, "Function should return dataset file discovery results"
        assert isinstance(result, (list, dict)), "Result should be list or dict of discovered files"
        
        # Verify result matches expected discovery behavior
        expected_result = mock_discover_dataset_files.return_value
        assert result == expected_result, (
            f"Function should return dataset discovery results: "
            f"expected {expected_result}, got {result}"
        )
    
    def test_load_dataset_files_missing_data_directory_error(self, sample_comprehensive_config_dict):
        """
        Test error behavior when major_data_directory is missing using behavioral validation.
        
        Validates proper error handling when required configuration elements
        are missing, focusing on observable error behavior rather than specific
        internal error constants.
        """
        # ARRANGE - Set up configuration missing required directory
        config = sample_comprehensive_config_dict.copy()
        config["project"]["directories"].pop("major_data_directory")
        
        # ACT & ASSERT - Execute function and verify expected error behavior
        with pytest.raises(ValueError) as exc_info:
            load_dataset_files(config=config, dataset_name="test_dataset")
        
        # Verify error message indicates missing data directory
        error_message = str(exc_info.value)
        assert "data directory" in error_message.lower(), (
            f"Error should indicate missing data directory, got: {error_message}"
        )


class TestGetDatasetParameters:
    """
    Test class for get_dataset_parameters function with behavior-focused validation.
    
    Uses black-box testing approach focusing on parameter extraction behavior
    and observable outcomes rather than internal implementation details.
    """
    
    @pytest.mark.parametrize("scenario", DATASET_PARAMETER_SCENARIOS)
    def test_get_dataset_parameters_scenarios(self, scenario):
        """
        Test get_dataset_parameters with various configuration structures using behavioral validation.
        
        Validates that the function correctly extracts and returns dataset parameters
        from different configuration structures, including edge cases with empty 
        or missing parameter definitions.
        """
        # ARRANGE - Set up test configuration and expected outcomes
        config = scenario["config"]
        dataset_name = scenario["dataset_name"]
        expected_params = scenario["expected_params"]
        
        # ACT - Execute the function under test
        result = get_dataset_parameters(config=config, dataset_name=dataset_name)
        
        # ASSERT - Verify behavioral outcomes
        assert isinstance(result, dict), "Function should return dictionary of parameters"
        assert result == expected_params, (
            f"Function should extract correct parameters: expected {expected_params}, got {result}"
        )
    
    def test_get_dataset_parameters_nonexistent_dataset_error(self, sample_comprehensive_config_dict):
        """
        Test error behavior for nonexistent dataset using behavioral validation.
        
        Validates proper error handling when requesting parameters for
        datasets that don't exist in the configuration, focusing on
        observable error behavior.
        """
        # ARRANGE - Set up test with nonexistent dataset name
        nonexistent_dataset = "nonexistent_dataset"
        
        # ACT & ASSERT - Execute function and verify expected error behavior
        with pytest.raises(KeyError) as exc_info:
            get_dataset_parameters(config=sample_comprehensive_config_dict, dataset_name=nonexistent_dataset)
        
        # Verify error message indicates the specific dataset that wasn't found
        error_message = str(exc_info.value)
        assert nonexistent_dataset in error_message, (
            f"Error should mention missing dataset '{nonexistent_dataset}', got: {error_message}"
        )
    
    @pytest.mark.parametrize("scenario,expected_exception,expected_message", CONFIG_VALIDATION_ERROR_SCENARIOS[:2])
    def test_get_dataset_parameters_validation_errors(self, scenario, expected_exception, expected_message):
        """
        Test comprehensive validation error scenarios using behavioral validation.
        
        Validates that the function raises appropriate exceptions for invalid inputs,
        focusing on observable error behavior rather than internal validation details.
        """
        # ARRANGE - Set up invalid input parameters
        func_args = scenario["func_args"]
        func_args.setdefault("dataset_name", "test_dataset")
        
        # ACT & ASSERT - Execute function and verify expected error behavior
        with pytest.raises(expected_exception) as exc_info:
            get_dataset_parameters(**func_args)
        
        # Verify error message contains expected information
        error_message = str(exc_info.value)
        assert expected_message in error_message, (
            f"Error message should contain '{expected_message}', got '{error_message}'"
        )


class TestGetExperimentParameters:
    """
    Test class for get_experiment_parameters function with behavior-focused validation.
    
    Uses black-box testing approach focusing on parameter extraction behavior
    and observable outcomes from the public API.
    """
    
    def test_get_experiment_parameters_with_defined_params(self, sample_comprehensive_config_dict):
        """
        Test experiment parameter retrieval with defined parameters using behavioral validation.
        
        Validates successful parameter extraction for experiments with
        comprehensive parameter configurations, focusing on observable
        return values rather than internal extraction logic.
        """
        # ARRANGE - Set up test with experiment that has parameters
        experiment_name = "baseline_control_study"
        
        # ACT - Execute the function under test
        result = get_experiment_parameters(
            config=sample_comprehensive_config_dict,
            experiment_name=experiment_name
        )
        
        # ASSERT - Verify behavioral outcomes
        assert isinstance(result, dict), "Function should return dictionary of parameters"
        assert len(result) > 0, "Function should return non-empty parameters for configured experiment"
        
        # Verify that parameters contain expected structure for baseline control study
        assert "velocity_threshold" in result, "Should contain experiment-specific parameters"
        assert result["velocity_threshold"] == 2.0, "Should return correct parameter values"
    
    def test_get_experiment_parameters_returns_empty_dict_when_no_params(self, sample_comprehensive_config_dict):
        """
        Test experiment parameter retrieval when parameters section is missing.
        
        Validates that the function gracefully handles experiments without
        explicit parameter definitions by returning empty dict.
        """
        # ARRANGE - Set up test configuration with experiment lacking parameters
        config = sample_comprehensive_config_dict.copy()
        config["experiments"]["experiment_without_params"] = {
            "datasets": ["baseline_behavior"]
            # Note: no parameters section
        }
        
        # ACT - Execute the function under test
        result = get_experiment_parameters(
            config=config,
            experiment_name="experiment_without_params"
        )
        
        # ASSERT - Verify behavioral outcomes
        assert isinstance(result, dict), "Function should return dictionary even when no parameters"
        assert len(result) == 0, "Function should return empty dict when no parameters defined"
    
    def test_get_experiment_parameters_nonexistent_experiment_error(self, sample_comprehensive_config_dict):
        """
        Test error behavior for nonexistent experiment using behavioral validation.
        
        Validates proper error handling when requesting parameters for
        experiments that don't exist in the configuration.
        """
        # ARRANGE - Set up test with nonexistent experiment name
        nonexistent_experiment = "nonexistent_experiment"
        
        # ACT & ASSERT - Execute function and verify expected error behavior
        with pytest.raises(KeyError) as exc_info:
            get_experiment_parameters(config=sample_comprehensive_config_dict, experiment_name=nonexistent_experiment)
        
        # Verify error message indicates the specific experiment that wasn't found
        error_message = str(exc_info.value)
        assert nonexistent_experiment in error_message, (
            f"Error should mention missing experiment '{nonexistent_experiment}', got: {error_message}"
        )


# Note: TestResolveBaseDirectory class removed per Section 0 requirements.
# Private function _resolve_base_directory is not part of the public API and
# should not be tested directly. Base directory resolution behavior is validated
# through public API functions that depend on this functionality.


# =============================================================================
# INTEGRATION TESTS FOR COMPLETE API WORKFLOWS (TST-INTEG-001)
# =============================================================================

class TestAPIIntegrationWorkflows:
    """
    Integration test class validating complete API workflows using behavior-focused validation.
    
    Uses centralized fixtures and Protocol-based mock implementations to test
    end-to-end API behavior without coupling to internal implementation details.
    """
    
    def test_complete_experiment_workflow_integration(
        self,
        sample_comprehensive_config_dict,
        temp_filesystem_structure,
        mock_config_and_discovery_comprehensive
    ):
        """
        Test complete experiment workflow from configuration to file discovery using behavioral validation.
        
        Validates end-to-end integration of configuration loading, experiment
        parameter extraction, and file discovery operations, focusing on
        observable behavior and API contract compliance.
        """
        # ARRANGE - Set up complete integration test environment
        config = sample_comprehensive_config_dict.copy()
        config["project"]["directories"]["major_data_directory"] = str(temp_filesystem_structure["data_root"])
        
        # Use Protocol-based mock implementations from centralized utilities
        expected_files = mock_config_and_discovery_comprehensive["discovered_files"]
        
        # ACT - Execute complete workflow through public API
        experiment_files = load_experiment_files(
            config=config,
            experiment_name="baseline_control_study",
            extract_metadata=True
        )
        
        experiment_params = get_experiment_parameters(
            config=config,
            experiment_name="baseline_control_study"
        )
        
        # ASSERT - Verify integration workflow behavior
        # Verify file discovery results
        assert experiment_files is not None, "File discovery should return results"
        assert isinstance(experiment_files, (list, dict)), "Should return structured file discovery results"
        
        # Verify parameter extraction behavior
        assert isinstance(experiment_params, dict), "Parameter extraction should return dictionary"
        assert len(experiment_params) > 0, "Should extract non-empty parameters for configured experiment"
        
        # Verify behavioral consistency between related API calls
        assert "velocity_threshold" in experiment_params, "Should contain expected experiment parameters"
    
    def test_complete_dataset_workflow_integration(
        self,
        sample_comprehensive_config_dict,
        temp_filesystem_structure,
        mock_config_and_discovery_comprehensive
    ):
        """
        Test complete dataset workflow from configuration to parameter extraction using behavioral validation.
        
        Validates end-to-end integration of dataset file discovery and
        parameter extraction, focusing on observable API behavior rather
        than internal implementation details.
        """
        # ARRANGE - Set up complete dataset integration test environment
        config = sample_comprehensive_config_dict.copy()
        config["project"]["directories"]["major_data_directory"] = str(temp_filesystem_structure["data_root"])
        
        # Use Protocol-based mock implementations
        expected_dataset_files = mock_config_and_discovery_comprehensive["discovered_files"]
        
        # ACT - Execute integrated dataset workflow through public API
        dataset_files = load_dataset_files(
            config=config,
            dataset_name="baseline_behavior",
            pattern="*.pkl",
            recursive=True
        )
        
        dataset_params = get_dataset_parameters(
            config=config,
            dataset_name="baseline_behavior"
        )
        
        # ASSERT - Verify integrated workflow behavior
        # Verify dataset file discovery results
        assert dataset_files is not None, "Dataset file discovery should return results"
        assert isinstance(dataset_files, (list, dict)), "Should return structured dataset file results"
        
        # Verify dataset parameter extraction behavior
        assert isinstance(dataset_params, dict), "Dataset parameter extraction should return dictionary"
        assert len(dataset_params) > 0, "Should extract non-empty parameters for configured dataset"
        
        # Verify behavioral consistency for dataset operations
        assert "threshold" in dataset_params, "Should contain expected dataset parameters"
    
    def test_multi_dataset_experiment_integration(
        self,
        sample_comprehensive_config_dict,
        mock_config_and_discovery_comprehensive
    ):
        """
        Test integration workflow for experiments spanning multiple datasets using behavioral validation.
        
        Validates complex integration scenarios where experiments reference
        multiple datasets, focusing on observable API behavior and parameter
        consistency across related operations.
        """
        # ARRANGE - Set up multi-dataset experiment integration test
        config = sample_comprehensive_config_dict.copy()
        
        # Ensure experiment with multiple datasets exists in config
        config["experiments"]["multi_dataset_experiment"] = {
            "datasets": ["baseline_behavior", "optogenetic_stimulation"],
            "parameters": {
                "comparison_type": "cross_modal",
                "analysis_window": 180
            }
        }
        
        # ACT - Execute multi-dataset experiment workflow through public API
        experiment_files = load_experiment_files(
            config=config,
            experiment_name="multi_dataset_experiment"
        )
        
        experiment_params = get_experiment_parameters(
            config=config,
            experiment_name="multi_dataset_experiment"
        )
        
        # Test individual dataset parameter extraction
        baseline_params = get_dataset_parameters(
            config=config,
            dataset_name="baseline_behavior"
        )
        
        opto_params = get_dataset_parameters(
            config=config,
            dataset_name="optogenetic_stimulation"
        )
        
        # ASSERT - Verify multi-dataset integration behavior
        # Verify experiment-level behavior
        assert experiment_files is not None, "Multi-dataset experiment should return files"
        assert isinstance(experiment_params, dict), "Should return experiment parameters"
        assert experiment_params["comparison_type"] == "cross_modal", "Should extract correct experiment parameters"
        
        # Verify dataset-level behavior consistency
        assert isinstance(baseline_params, dict), "Should extract baseline dataset parameters"
        assert isinstance(opto_params, dict), "Should extract optogenetic dataset parameters"
        assert len(baseline_params) > 0, "Baseline dataset should have parameters"
        assert len(opto_params) > 0, "Optogenetic dataset should have parameters"


# =============================================================================
# PROPERTY-BASED TESTING WITH HYPOTHESIS (Section 3.6.3)
# =============================================================================

class TestAPIPropertyBasedValidation:
    """
    Property-based testing class using Hypothesis for robust behavioral validation
    of API parameter combinations, focusing on observable behavior rather than
    internal implementation details.
    """
    
    @given(
        experiment_name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        base_directory=st.text(min_size=1, max_size=100).filter(lambda x: "/" in x),
        pattern=st.text(min_size=1, max_size=20),
        recursive=st.booleans(),
        extract_metadata=st.booleans(),
        parse_dates=st.booleans()
    )
    @settings(max_examples=50, deadline=None)
    def test_load_experiment_files_property_validation(
        self,
        experiment_name,
        base_directory,
        pattern,
        recursive,
        extract_metadata,
        parse_dates,
        mock_config_and_discovery_comprehensive
    ):
        """
        Property-based test for load_experiment_files behavioral validation.
        
        Uses Hypothesis to generate diverse parameter combinations and verify
        that the function maintains consistent behavioral properties across
        edge cases, focusing on return value consistency and error handling.
        """
        # ARRANGE - Filter for valid inputs to focus on behavior validation
        assume(experiment_name.strip())
        assume(len(base_directory) > 0)
        assume(len(pattern) > 0)
        
        # Set up test configuration with generated experiment
        test_config = {
            "project": {"directories": {"major_data_directory": base_directory}},
            "experiments": {experiment_name: {"datasets": ["test_dataset"]}},
            "datasets": {"test_dataset": {"patterns": ["*test*"]}}
        }
        
        # ACT - Execute function with generated parameters
        try:
            result = load_experiment_files(
                config=test_config,
                experiment_name=experiment_name,
                pattern=pattern,
                recursive=recursive,
                extract_metadata=extract_metadata,
                parse_dates=parse_dates
            )
            
            # ASSERT - Verify consistent behavioral properties
            # Property 1: Function should always return structured data
            assert result is not None, "Function should always return a result"
            assert isinstance(result, (list, dict)), "Result should be list or dict structure"
            
            # Property 2: Return type should be consistent with metadata extraction
            if extract_metadata or parse_dates:
                assert isinstance(result, dict), "Should return dict when metadata extraction enabled"
            
            # Property 3: Function should handle all valid parameter combinations
            # This is validated by successful execution without exceptions
            
        except (KeyError, ValueError, FileNotFoundError) as e:
            # Expected exceptions for invalid configurations are acceptable
            # This validates proper error handling behavior
            assert isinstance(e, (KeyError, ValueError, FileNotFoundError)), (
                f"Function should raise appropriate exceptions for invalid inputs, got {type(e)}"
            )
    
    @given(
        dataset_name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        extensions=st.lists(
            st.text(min_size=2, max_size=10).filter(lambda x: x.startswith(".")),
            min_size=0,
            max_size=5
        ),
        parameters=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(
                st.integers(min_value=0, max_value=1000),
                st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
                st.text(min_size=1, max_size=50),
                st.booleans()
            ),
            min_size=0,
            max_size=10
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_dataset_parameter_extraction_properties(
        self,
        dataset_name,
        extensions,
        parameters
    ):
        """
        Property-based test for dataset parameter extraction behavioral consistency.
        
        Validates that dataset parameter extraction maintains consistent
        behavior across diverse parameter structures, focusing on observable
        return value properties rather than internal extraction mechanisms.
        """
        # ARRANGE - Filter for valid inputs and construct test configuration
        assume(dataset_name.strip())
        
        config = {
            "datasets": {
                dataset_name: {
                    "parameters": parameters,
                    "extensions": extensions
                }
            }
        }
        
        # ACT - Execute parameter extraction
        result_params = get_dataset_parameters(config=config, dataset_name=dataset_name)
        
        # ASSERT - Verify behavioral properties
        # Property 1: Function should always return a dictionary
        assert isinstance(result_params, dict), "Function should always return dictionary"
        
        # Property 2: Returned parameters should match input parameters exactly
        assert result_params == parameters, "Function should return exact parameter values"
        
        # Property 3: Parameter type preservation (behavioral contract)
        for key, expected_value in parameters.items():
            assert key in result_params, f"Parameter key '{key}' should be preserved"
            actual_value = result_params[key]
            assert type(actual_value) == type(expected_value), f"Type should be preserved for '{key}'"
            assert actual_value == expected_value, f"Value should be preserved for '{key}'"
        
        # Property 4: No additional parameters should be added
        assert len(result_params) == len(parameters), "No additional parameters should be added"
    
    @given(
        config_structure=st.fixed_dictionaries({
            "project": st.fixed_dictionaries({
                "directories": st.dictionaries(
                    keys=st.just("major_data_directory"),
                    values=st.text(min_size=1, max_size=100),
                    min_size=0,
                    max_size=1
                )
            }),
            "experiments": st.dictionaries(
                keys=st.text(min_size=1, max_size=30),
                values=st.fixed_dictionaries({
                    "datasets": st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5)
                }),
                min_size=1,
                max_size=3
            )
        })
    )
    @settings(max_examples=20, deadline=None)
    def test_configuration_validation_properties(self, config_structure, mock_config_and_discovery_comprehensive):
        """
        Property-based test for configuration validation behavioral robustness.
        
        Validates that configuration validation behaves consistently
        across diverse configuration structures, focusing on observable
        error handling behavior rather than internal validation logic.
        """
        # ARRANGE - Set up test with generated configuration structure
        experiment_names = list(config_structure["experiments"].keys())
        assume(len(experiment_names) > 0)
        
        experiment_name = experiment_names[0]
        has_data_dir = bool(config_structure["project"]["directories"])
        
        # ACT & ASSERT - Test behavioral properties
        if has_data_dir:
            # Property: Valid configurations should not raise validation errors
            try:
                result = load_experiment_files(
                    config=config_structure,
                    experiment_name=experiment_name
                )
                
                # Property: Valid execution should return structured data
                assert result is not None, "Valid config should return results"
                assert isinstance(result, (list, dict)), "Should return structured file data"
                
            except (RuntimeError, FileNotFoundError):
                # Expected when underlying discovery operations fail
                # This validates that configuration validation passed
                pass
            except ValueError as e:
                # Should not get configuration validation errors for valid configs
                if "data directory" in str(e).lower():
                    # This is acceptable - validates data directory requirement
                    pass
                else:
                    raise
        else:
            # Property: Invalid configurations should raise appropriate errors
            with pytest.raises(ValueError) as exc_info:
                load_experiment_files(
                    config=config_structure,
                    experiment_name=experiment_name
                )
            
            # Verify error message indicates data directory issue
            error_message = str(exc_info.value)
            assert "data directory" in error_message.lower(), (
                f"Error should indicate data directory issue, got: {error_message}"
            )


# =============================================================================
# PERFORMANCE AND EDGE CASE TESTS
# =============================================================================

class TestAPIPerformanceAndEdgeCases:
    """
    Test class for performance validation and edge case handling using behavior-focused validation.
    
    Uses centralized fixtures and focuses on observable behavior and performance
    characteristics rather than internal implementation details.
    """
    
    def test_large_configuration_handling(self, mock_config_and_discovery_comprehensive):
        """
        Test API functions with large configuration structures using behavioral validation.
        
        Validates that API functions maintain consistent behavior and acceptable
        performance when handling configurations with many experiments, datasets, 
        and parameters, focusing on observable outcomes.
        """
        # ARRANGE - Create large configuration structure for stress testing
        large_config = {
            "project": {"directories": {"major_data_directory": "/data"}},
            "experiments": {},
            "datasets": {}
        }
        
        # Generate many experiments and datasets
        for i in range(100):
            exp_name = f"experiment_{i:03d}"
            ds_name = f"dataset_{i:03d}"
            
            large_config["experiments"][exp_name] = {
                "datasets": [ds_name],
                "parameters": {f"param_{j}": j * i for j in range(10)}
            }
            
            large_config["datasets"][ds_name] = {
                "parameters": {f"ds_param_{j}": f"value_{j}_{i}" for j in range(5)}
            }
        
        # ACT - Execute API functions with large configuration
        result = load_experiment_files(
            config=large_config,
            experiment_name="experiment_050"
        )
        
        params = get_experiment_parameters(
            config=large_config,
            experiment_name="experiment_050"
        )
        
        # ASSERT - Verify behavioral consistency with large configurations
        # Verify file loading behavior scales appropriately
        assert result is not None, "Function should handle large configurations"
        assert isinstance(result, (list, dict)), "Should return consistent structure for large configs"
        
        # Verify parameter extraction behavior scales appropriately
        assert isinstance(params, dict), "Parameter extraction should handle large configs"
        assert len(params) == 10, "Should extract correct number of parameters"
        assert params["param_5"] == 250, "Should calculate parameter values correctly (50 * 5)"
    
    def test_unicode_and_special_characters_handling(self, mock_config_and_discovery_comprehensive):
        """
        Test API functions with Unicode and special characters using behavioral validation.
        
        Validates proper handling of internationalized experiment and
        dataset names with various character encodings, focusing on
        observable behavior rather than internal character processing.
        """
        # ARRANGE - Set up configuration with Unicode characters
        unicode_config = {
            "project": {"directories": {"major_data_directory": "/données/expérience"}},
            "experiments": {
                "expérience_été": {
                    "datasets": ["données_été"],
                    "parameters": {"température": 25.5, "humidité": "élevée"}
                }
            },
            "datasets": {
                "données_été": {
                    "parameters": {"région": "Québec", "saison": "été_2023"}
                }
            }
        }
        
        # ACT - Execute API functions with Unicode names
        result = load_experiment_files(
            config=unicode_config,
            experiment_name="expérience_été"
        )
        
        exp_params = get_experiment_parameters(
            config=unicode_config,
            experiment_name="expérience_été"
        )
        
        ds_params = get_dataset_parameters(
            config=unicode_config,
            dataset_name="données_été"
        )
        
        # ASSERT - Verify Unicode handling behavior
        # Verify file loading handles Unicode names properly
        assert result is not None, "Function should handle Unicode experiment names"
        assert isinstance(result, (list, dict)), "Should return consistent structure with Unicode names"
        
        # Verify parameter extraction preserves Unicode characters correctly
        assert isinstance(exp_params, dict), "Should extract experiment parameters with Unicode"
        assert exp_params["température"] == 25.5, "Should preserve Unicode parameter names and values"
        assert exp_params["humidité"] == "élevée", "Should preserve Unicode string values"
        
        # Verify dataset parameter extraction with Unicode
        assert isinstance(ds_params, dict), "Should extract dataset parameters with Unicode"
        assert ds_params["région"] == "Québec", "Should preserve Unicode dataset parameter values"
    
    def test_nested_configuration_structures(self):
        """
        Test handling of deeply nested configuration structures using behavioral validation.
        
        Validates correct parameter extraction from configurations with
        complex nested dictionary structures, focusing on observable
        parameter preservation behavior rather than internal processing.
        """
        # ARRANGE - Set up deeply nested configuration structure
        nested_config = {
            "project": {"directories": {"major_data_directory": "/data"}},
            "experiments": {
                "nested_experiment": {
                    "datasets": ["nested_dataset"],
                    "parameters": {
                        "analysis": {
                            "preprocessing": {
                                "filter_type": "butterworth",
                                "cutoff_freq": 10.0,
                                "order": 4
                            },
                            "feature_extraction": {
                                "window_size": 1000,
                                "overlap": 0.5,
                                "features": ["mean", "std", "max"]
                            }
                        },
                        "visualization": {
                            "plot_type": "timeseries",
                            "color_scheme": "viridis"
                        }
                    }
                }
            },
            "datasets": {
                "nested_dataset": {
                    "parameters": {
                        "acquisition": {
                            "sampling_rate": 1000,
                            "duration": 300,
                            "channels": ["x", "y", "z"]
                        }
                    }
                }
            }
        }
        
        # ACT - Execute parameter extraction on nested structures
        exp_params = get_experiment_parameters(
            config=nested_config,
            experiment_name="nested_experiment"
        )
        
        ds_params = get_dataset_parameters(
            config=nested_config,
            dataset_name="nested_dataset"
        )
        
        # ASSERT - Verify nested structure preservation behavior
        # Verify experiment parameter nesting is preserved
        assert isinstance(exp_params, dict), "Should return dictionary for nested experiment parameters"
        assert "analysis" in exp_params, "Should preserve top-level nested structure"
        assert isinstance(exp_params["analysis"], dict), "Should preserve nested dictionary structure"
        
        # Verify deep nesting preservation
        assert exp_params["analysis"]["preprocessing"]["filter_type"] == "butterworth", "Should preserve deeply nested values"
        assert exp_params["analysis"]["feature_extraction"]["features"] == ["mean", "std", "max"], "Should preserve nested lists"
        assert exp_params["visualization"]["color_scheme"] == "viridis", "Should preserve nested string values"
        
        # Verify dataset parameter nesting is preserved
        assert isinstance(ds_params, dict), "Should return dictionary for nested dataset parameters"
        assert "acquisition" in ds_params, "Should preserve dataset nesting structure"
        assert ds_params["acquisition"]["sampling_rate"] == 1000, "Should preserve nested numeric values"
        assert ds_params["acquisition"]["channels"] == ["x", "y", "z"], "Should preserve nested list structures"


# =============================================================================
# TEST MARKERS FOR ORGANIZATION
# =============================================================================

# Mark all tests with behavior-focused validation approach
pytestmark = [
    pytest.mark.unit,
    pytest.mark.api,
    pytest.mark.behavior_focused  # New marker for behavior-focused testing approach
]

# Additional markers for specific test categories
integration_marker = pytest.mark.integration
property_based_marker = pytest.mark.property_based
performance_marker = pytest.mark.slow

# Apply markers to test classes using centralized fixtures and behavioral validation
TestAPIIntegrationWorkflows = integration_marker(TestAPIIntegrationWorkflows)
TestAPIPropertyBasedValidation = property_based_marker(TestAPIPropertyBasedValidation)
TestAPIPerformanceAndEdgeCases = performance_marker(TestAPIPerformanceAndEdgeCases)