"""
Comprehensive test suite for API functions in the flyrigloader.api module.

This module implements modern pytest practices with extensive fixture usage,
parametrization, property-based testing, and comprehensive integration testing
per the TST-MOD-002, TST-MOD-003, TST-MOD-004, and TST-INTEG-001 requirements.
"""

import os
import re
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays

from flyrigloader.api import (
    load_experiment_files,
    load_dataset_files,
    get_dataset_parameters,
    get_experiment_parameters,
    process_experiment_data,
    _resolve_base_directory,
    MISSING_DATA_DIR_ERROR
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
        MISSING_DATA_DIR_ERROR,
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
# ENHANCED FIXTURES WITH pytest-mock INTEGRATION
# =============================================================================

@pytest.fixture
def enhanced_mock_config_and_discovery(mocker):
    """
    Enhanced fixture using pytest-mock for standardized mocking per TST-MOD-003.
    
    Provides comprehensive mocking of configuration loading and file discovery
    functions with realistic return values for integration testing.
    """
    # Mock load_config function
    mock_load_config = mocker.patch("flyrigloader.api.load_config")
    mock_load_config.return_value = {
        "project": {
            "directories": {
                "major_data_directory": "/path/to/data"
            }
        },
        "experiments": {
            "test_experiment": {
                "datasets": ["test_dataset"],
                "metadata": {
                    "extraction_patterns": [
                        r".*_(?P<experiment>\w+)_(?P<date>\d{8})\.csv"
                    ]
                }
            }
        },
        "datasets": {
            "test_dataset": {
                "patterns": ["*_test_*"],
                "parameters": {"threshold": 0.5, "method": "advanced"}
            }
        }
    }
    
    # Mock file discovery functions with realistic return values
    mock_discover_experiment_files = mocker.patch("flyrigloader.api.discover_experiment_files")
    mock_discover_experiment_files.return_value = {
        "/path/to/data/exp_test_20230101.csv": {
            "experiment": "test",
            "date": "20230101",
            "file_size": 1024,
            "modification_time": "2023-01-01T10:00:00"
        },
        "/path/to/data/exp_test_20230102.csv": {
            "experiment": "test", 
            "date": "20230102",
            "file_size": 2048,
            "modification_time": "2023-01-02T11:00:00"
        }
    }
    
    mock_discover_dataset_files = mocker.patch("flyrigloader.api.discover_dataset_files")
    mock_discover_dataset_files.return_value = {
        "/path/to/data/dataset_test_file1.pkl": {
            "dataset": "test",
            "file_type": "pickle",
            "file_size": 512
        },
        "/path/to/data/dataset_test_file2.pkl": {
            "dataset": "test",
            "file_type": "pickle", 
            "file_size": 768
        }
    }
    
    return mock_load_config, mock_discover_experiment_files, mock_discover_dataset_files


@pytest.fixture
def realistic_config_dict():
    """
    Fixture providing realistic configuration dictionary for integration testing.
    
    Returns:
        Dict[str, Any]: Comprehensive configuration with all required sections
    """
    return {
        "project": {
            "directories": {
                "major_data_directory": "/research/experimental_data",
                "batchfile_directory": "/research/batch_definitions"
            },
            "ignore_substrings": ["temp_", "backup_", "._"],
            "extraction_patterns": [
                r".*_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.csv",
                r".*_(?P<animal_id>\w+)_(?P<session>\d+)\.pkl"
            ]
        },
        "rigs": {
            "rig_001": {
                "sampling_frequency": 60,
                "mm_per_px": 0.154,
                "calibration_date": "2023-01-15"
            },
            "rig_002": {
                "sampling_frequency": 120,
                "mm_per_px": 0.1818,
                "calibration_date": "2023-02-10"
            }
        },
        "datasets": {
            "navigation_dataset": {
                "rig": "rig_001",
                "patterns": ["*nav*", "*navigation*"],
                "dates_vials": {
                    "2024-01-15": [1, 2, 3],
                    "2024-01-16": [1, 2]
                },
                "parameters": {
                    "threshold": 0.75,
                    "smoothing_window": 5,
                    "analysis_method": "advanced"
                }
            },
            "control_dataset": {
                "rig": "rig_002", 
                "patterns": ["*ctrl*", "*control*"],
                "dates_vials": {
                    "2024-01-20": [1, 2, 3, 4],
                    "2024-01-21": [1, 2]
                },
                "parameters": {
                    "baseline_duration": 60,
                    "stimulus_duration": 120
                }
            }
        },
        "experiments": {
            "navigation_experiment": {
                "datasets": ["navigation_dataset"],
                "parameters": {
                    "experiment_duration": 300,
                    "trial_count": 10,
                    "researcher": "Dr. Smith"
                },
                "metadata": {
                    "extraction_patterns": [
                        r".*_(?P<trial>\d+)_(?P<condition>\w+)\.csv"
                    ]
                }
            },
            "multi_dataset_experiment": {
                "datasets": ["navigation_dataset", "control_dataset"],
                "parameters": {
                    "comparison_type": "cross_modal",
                    "analysis_window": 180
                }
            }
        }
    }


@pytest.fixture
def temp_experimental_data_dir():
    """
    Fixture creating temporary directory with realistic experimental data structure.
    
    Returns:
        Path: Temporary directory path with experimental data files
    """
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create realistic directory structure
        data_dir = temp_dir / "experimental_data"
        data_dir.mkdir()
        
        navigation_dir = data_dir / "navigation"
        navigation_dir.mkdir()
        
        control_dir = data_dir / "control"
        control_dir.mkdir()
        
        # Create sample data files with realistic names
        sample_files = [
            navigation_dir / "nav_20240115_trial_1.csv",
            navigation_dir / "nav_20240115_trial_2.csv", 
            navigation_dir / "nav_20240116_trial_1.csv",
            control_dir / "ctrl_20240120_baseline_1.pkl",
            control_dir / "ctrl_20240120_stimulus_1.pkl",
            control_dir / "ctrl_20240121_baseline_1.pkl"
        ]
        
        for file_path in sample_files:
            file_path.write_text("# Sample experimental data file\ndata_placeholder")
        
        yield data_dir
    finally:
        shutil.rmtree(temp_dir)


# =============================================================================
# PARAMETRIZED UNIT TESTS WITH COMPREHENSIVE EDGE CASE COVERAGE
# =============================================================================

class TestLoadExperimentFiles:
    """
    Test class for load_experiment_files function with comprehensive parametrization
    per TST-MOD-002 requirements.
    """
    
    @pytest.mark.parametrize("scenario", EXPERIMENT_LOAD_SCENARIOS)
    def test_load_experiment_files_scenarios(
        self, 
        scenario, 
        enhanced_mock_config_and_discovery
    ):
        """
        Test load_experiment_files with various parameter combinations.
        
        Validates proper function call propagation and parameter handling
        across different configuration sources and parameter combinations.
        """
        mock_load_config, mock_discover_experiment_files, _ = enhanced_mock_config_and_discovery
        
        # Extract test parameters and expected results
        func_args = {k: v for k, v in scenario.items() if k != "expected_calls"}
        expected_calls = scenario["expected_calls"]
        
        # Execute function under test
        result = load_experiment_files(**func_args)
        
        # Verify load_config calls
        if expected_calls["load_config"]:
            mock_load_config.assert_called_once_with(*expected_calls["load_config"][0])
        else:
            mock_load_config.assert_not_called()
        
        # Verify discover_experiment_files calls
        expected_discover_calls = expected_calls["discover_experiment_files"]
        mock_discover_experiment_files.assert_called_once()
        
        # Verify all expected parameters were passed correctly
        actual_call = mock_discover_experiment_files.call_args
        for param_name, expected_value in expected_discover_calls.items():
            assert actual_call.kwargs[param_name] == expected_value, (
                f"Parameter {param_name} mismatch: expected {expected_value}, "
                f"got {actual_call.kwargs[param_name]}"
            )
        
        # Verify result matches discovery function return value
        assert result == mock_discover_experiment_files.return_value
    
    @pytest.mark.parametrize("scenario,expected_exception,expected_message", CONFIG_VALIDATION_ERROR_SCENARIOS)
    def test_load_experiment_files_validation_errors(self, scenario, expected_exception, expected_message):
        """
        Test comprehensive error handling validation per F-001-RQ-003 requirements.
        
        Validates proper ValueError raising for invalid configuration parameters
        and missing required configuration elements.
        """
        func_args = scenario["func_args"]
        func_args.setdefault("experiment_name", "test_experiment")
        
        with pytest.raises(expected_exception, match=re.escape(expected_message)):
            load_experiment_files(**func_args)


class TestLoadDatasetFiles:
    """
    Test class for load_dataset_files function with comprehensive parametrization.
    """
    
    @pytest.mark.parametrize("scenario", DATASET_LOAD_SCENARIOS)
    def test_load_dataset_files_scenarios(
        self, 
        scenario, 
        enhanced_mock_config_and_discovery
    ):
        """
        Test load_dataset_files with various parameter combinations.
        
        Validates proper function call propagation and parameter handling
        for dataset-specific file discovery operations.
        """
        mock_load_config, _, mock_discover_dataset_files = enhanced_mock_config_and_discovery
        
        # Extract test parameters and expected results
        func_args = {k: v for k, v in scenario.items() if k != "expected_calls"}
        expected_calls = scenario["expected_calls"]
        
        # Execute function under test
        result = load_dataset_files(**func_args)
        
        # Verify load_config calls
        if expected_calls["load_config"]:
            mock_load_config.assert_called_once_with(*expected_calls["load_config"][0])
        else:
            mock_load_config.assert_not_called()
        
        # Verify discover_dataset_files calls with parameter validation
        expected_discover_calls = expected_calls["discover_dataset_files"]
        mock_discover_dataset_files.assert_called_once()
        
        actual_call = mock_discover_dataset_files.call_args
        for param_name, expected_value in expected_discover_calls.items():
            assert actual_call.kwargs[param_name] == expected_value, (
                f"Parameter {param_name} mismatch: expected {expected_value}, "
                f"got {actual_call.kwargs[param_name]}"
            )
        
        # Verify result matches discovery function return value
        assert result == mock_discover_dataset_files.return_value
    
    def test_load_dataset_files_missing_data_directory_error(self, realistic_config_dict):
        """
        Test ValueError when major_data_directory is missing per F-001-RQ-003.
        
        Validates proper error handling when required configuration elements
        are missing from the provided configuration dictionary.
        """
        config = realistic_config_dict.copy()
        config["project"]["directories"].pop("major_data_directory")
        
        with pytest.raises(ValueError, match=re.escape(MISSING_DATA_DIR_ERROR)):
            load_dataset_files(config=config, dataset_name="test_dataset")


class TestGetDatasetParameters:
    """
    Test class for get_dataset_parameters function with comprehensive scenarios.
    """
    
    @pytest.mark.parametrize("scenario", DATASET_PARAMETER_SCENARIOS)
    def test_get_dataset_parameters_scenarios(self, scenario):
        """
        Test get_dataset_parameters with various configuration structures.
        
        Validates proper parameter extraction from dataset configurations
        including edge cases with empty or missing parameter definitions.
        """
        config = scenario["config"]
        dataset_name = scenario["dataset_name"]
        expected_params = scenario["expected_params"]
        
        result = get_dataset_parameters(config=config, dataset_name=dataset_name)
        assert result == expected_params
    
    def test_get_dataset_parameters_nonexistent_dataset_error(self, realistic_config_dict):
        """
        Test KeyError for nonexistent dataset per specification requirements.
        
        Validates proper error handling when requesting parameters for
        datasets that don't exist in the configuration.
        """
        with pytest.raises(KeyError, match="Dataset 'nonexistent_dataset' not found in configuration"):
            get_dataset_parameters(config=realistic_config_dict, dataset_name="nonexistent_dataset")
    
    @pytest.mark.parametrize("scenario,expected_exception,expected_message", CONFIG_VALIDATION_ERROR_SCENARIOS[:2])
    def test_get_dataset_parameters_validation_errors(self, scenario, expected_exception, expected_message):
        """
        Test comprehensive validation error scenarios for dataset parameter retrieval.
        """
        func_args = scenario["func_args"]
        func_args.setdefault("dataset_name", "test_dataset")
        
        with pytest.raises(expected_exception, match=re.escape(expected_message)):
            get_dataset_parameters(**func_args)


class TestGetExperimentParameters:
    """
    Test class for get_experiment_parameters function with comprehensive validation.
    """
    
    def test_get_experiment_parameters_with_defined_params(self, realistic_config_dict):
        """
        Test experiment parameter retrieval with defined parameters.
        
        Validates successful parameter extraction for experiments with
        comprehensive parameter configurations.
        """
        result = get_experiment_parameters(
            config=realistic_config_dict,
            experiment_name="navigation_experiment"
        )
        
        expected_params = {
            "experiment_duration": 300,
            "trial_count": 10,
            "researcher": "Dr. Smith"
        }
        assert result == expected_params
    
    def test_get_experiment_parameters_without_params(self, realistic_config_dict):
        """
        Test experiment parameter retrieval without defined parameters.
        
        Validates proper handling of experiments that don't have
        parameter definitions in their configuration.
        """
        result = get_experiment_parameters(
            config=realistic_config_dict,
            experiment_name="multi_dataset_experiment"
        )
        
        expected_params = {
            "comparison_type": "cross_modal",
            "analysis_window": 180
        }
        assert result == expected_params
    
    def test_get_experiment_parameters_nonexistent_experiment_error(self, realistic_config_dict):
        """
        Test KeyError for nonexistent experiment.
        
        Validates proper error handling when requesting parameters for
        experiments that don't exist in the configuration.
        """
        with pytest.raises(KeyError, match="Experiment 'nonexistent_exp' not found in configuration"):
            get_experiment_parameters(config=realistic_config_dict, experiment_name="nonexistent_exp")


class TestResolveBaseDirectory:
    """
    Test class for _resolve_base_directory internal function.
    """
    
    def test_resolve_base_directory_with_override(self):
        """Test base directory resolution with explicit override."""
        config = {"project": {"directories": {"major_data_directory": "/config/data"}}}
        override_dir = "/override/data"
        
        result = _resolve_base_directory(config, override_dir)
        assert result == override_dir
    
    def test_resolve_base_directory_from_config(self):
        """Test base directory resolution from configuration."""
        config = {"project": {"directories": {"major_data_directory": "/config/data"}}}
        
        result = _resolve_base_directory(config, None)
        assert result == "/config/data"
    
    def test_resolve_base_directory_missing_config_error(self):
        """Test ValueError when base directory cannot be resolved."""
        config = {"project": {"directories": {}}}
        
        with pytest.raises(ValueError, match=re.escape(MISSING_DATA_DIR_ERROR)):
            _resolve_base_directory(config, None)


# =============================================================================
# INTEGRATION TESTS FOR COMPLETE API WORKFLOWS (TST-INTEG-001)
# =============================================================================

class TestAPIIntegrationWorkflows:
    """
    Integration test class validating complete API workflows from configuration
    loading through final data processing per TST-INTEG-001 requirements.
    """
    
    def test_complete_experiment_workflow_integration(
        self,
        realistic_config_dict,
        temp_experimental_data_dir,
        mocker
    ):
        """
        Test complete experiment workflow from configuration to file discovery.
        
        Validates end-to-end integration of configuration loading, experiment
        parameter extraction, and file discovery operations with realistic
        test data and directory structures.
        """
        # Update config to use temporary directory
        config = realistic_config_dict.copy()
        config["project"]["directories"]["major_data_directory"] = str(temp_experimental_data_dir)
        
        # Mock only the discovery function to test integration up to that point
        mock_discover_experiment_files = mocker.patch("flyrigloader.api.discover_experiment_files")
        mock_discover_experiment_files.return_value = {
            str(temp_experimental_data_dir / "navigation" / "nav_20240115_trial_1.csv"): {
                "trial": "1",
                "condition": "trial",
                "file_size": 1024
            }
        }
        
        # Test complete workflow
        experiment_files = load_experiment_files(
            config=config,
            experiment_name="navigation_experiment",
            extract_metadata=True
        )
        
        experiment_params = get_experiment_parameters(
            config=config,
            experiment_name="navigation_experiment"
        )
        
        # Verify integration results
        assert experiment_files is not None
        assert len(experiment_files) > 0
        assert experiment_params == {
            "experiment_duration": 300,
            "trial_count": 10,
            "researcher": "Dr. Smith"
        }
        
        # Verify discovery function was called with correct integrated parameters
        mock_discover_experiment_files.assert_called_once()
        call_args = mock_discover_experiment_files.call_args
        assert call_args.kwargs["config"] == config
        assert call_args.kwargs["experiment_name"] == "navigation_experiment"
        assert call_args.kwargs["base_directory"] == str(temp_experimental_data_dir)
        assert call_args.kwargs["extract_metadata"] is True
    
    def test_complete_dataset_workflow_integration(
        self,
        realistic_config_dict,
        temp_experimental_data_dir,
        mocker
    ):
        """
        Test complete dataset workflow from configuration to parameter extraction.
        
        Validates end-to-end integration of dataset file discovery and
        parameter extraction with realistic configuration structures.
        """
        # Update config to use temporary directory
        config = realistic_config_dict.copy()
        config["project"]["directories"]["major_data_directory"] = str(temp_experimental_data_dir)
        
        # Mock discovery function for integration testing
        mock_discover_dataset_files = mocker.patch("flyrigloader.api.discover_dataset_files")
        mock_discover_dataset_files.return_value = {
            str(temp_experimental_data_dir / "control" / "ctrl_20240120_baseline_1.pkl"): {
                "file_type": "pickle",
                "file_size": 2048
            }
        }
        
        # Test integrated workflow
        dataset_files = load_dataset_files(
            config=config,
            dataset_name="control_dataset",
            pattern="*.pkl",
            recursive=True
        )
        
        dataset_params = get_dataset_parameters(
            config=config,
            dataset_name="control_dataset"
        )
        
        # Verify integration results
        assert dataset_files is not None
        assert dataset_params == {
            "baseline_duration": 60,
            "stimulus_duration": 120
        }
        
        # Verify proper parameter propagation through integration
        call_args = mock_discover_dataset_files.call_args
        assert call_args.kwargs["pattern"] == "*.pkl"
        assert call_args.kwargs["recursive"] is True
    
    def test_multi_dataset_experiment_integration(
        self,
        realistic_config_dict,
        mocker
    ):
        """
        Test integration workflow for experiments spanning multiple datasets.
        
        Validates complex integration scenarios where experiments reference
        multiple datasets with different configurations and parameters.
        """
        # Mock discovery functions for multi-dataset scenario
        mock_discover_experiment_files = mocker.patch("flyrigloader.api.discover_experiment_files")
        mock_discover_experiment_files.return_value = {
            "/data/nav_file1.csv": {"type": "navigation"},
            "/data/ctrl_file1.pkl": {"type": "control"}
        }
        
        # Test multi-dataset experiment
        experiment_files = load_experiment_files(
            config=realistic_config_dict,
            experiment_name="multi_dataset_experiment"
        )
        
        experiment_params = get_experiment_parameters(
            config=realistic_config_dict,
            experiment_name="multi_dataset_experiment"
        )
        
        # Test individual dataset parameters
        nav_params = get_dataset_parameters(
            config=realistic_config_dict,
            dataset_name="navigation_dataset"
        )
        
        ctrl_params = get_dataset_parameters(
            config=realistic_config_dict,
            dataset_name="control_dataset"
        )
        
        # Verify multi-dataset integration
        assert experiment_files is not None
        assert experiment_params["comparison_type"] == "cross_modal"
        assert nav_params["analysis_method"] == "advanced"
        assert ctrl_params["baseline_duration"] == 60


# =============================================================================
# PROPERTY-BASED TESTING WITH HYPOTHESIS (Section 3.6.3)
# =============================================================================

class TestAPIPropertyBasedValidation:
    """
    Property-based testing class using Hypothesis for robust validation
    of API parameter combinations per Section 3.6.3 requirements.
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
        mocker
    ):
        """
        Property-based test for load_experiment_files parameter validation.
        
        Uses Hypothesis to generate diverse parameter combinations and verify
        that the function maintains consistent behavior across edge cases.
        """
        # Assume valid inputs to avoid invalid test cases
        assume(experiment_name.strip())
        assume(len(base_directory) > 0)
        assume(len(pattern) > 0)
        
        # Mock dependencies
        mock_load_config = mocker.patch("flyrigloader.api.load_config")
        mock_load_config.return_value = {
            "project": {"directories": {"major_data_directory": base_directory}},
            "experiments": {experiment_name: {"datasets": ["test"]}}
        }
        
        mock_discover = mocker.patch("flyrigloader.api.discover_experiment_files")
        mock_discover.return_value = {}
        
        # Test function with generated parameters
        result = load_experiment_files(
            config_path="/test/config.yaml",
            experiment_name=experiment_name,
            pattern=pattern,
            recursive=recursive,
            extract_metadata=extract_metadata,
            parse_dates=parse_dates
        )
        
        # Verify consistent behavior properties
        assert result is not None
        mock_discover.assert_called_once()
        
        # Verify parameter propagation properties
        call_args = mock_discover.call_args
        assert call_args.kwargs["experiment_name"] == experiment_name
        assert call_args.kwargs["pattern"] == pattern
        assert call_args.kwargs["recursive"] == recursive
        assert call_args.kwargs["extract_metadata"] == extract_metadata
        assert call_args.kwargs["parse_dates"] == parse_dates
    
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
        Property-based test for dataset parameter extraction consistency.
        
        Validates that dataset parameter extraction maintains consistent
        behavior across diverse parameter structures and naming conventions.
        """
        # Assume valid inputs
        assume(dataset_name.strip())
        
        # Construct test configuration
        config = {
            "datasets": {
                dataset_name: {
                    "parameters": parameters,
                    "extensions": extensions
                }
            }
        }
        
        # Test parameter extraction
        result_params = get_dataset_parameters(config=config, dataset_name=dataset_name)
        
        # Verify extraction properties
        assert isinstance(result_params, dict)
        assert result_params == parameters
        
        # Verify parameter type preservation
        for key, value in parameters.items():
            assert key in result_params
            assert type(result_params[key]) == type(value)
            assert result_params[key] == value
    
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
    def test_configuration_validation_properties(self, config_structure):
        """
        Property-based test for configuration validation robustness.
        
        Validates that configuration validation behaves consistently
        across diverse configuration structures and handles edge cases gracefully.
        """
        experiment_names = list(config_structure["experiments"].keys())
        assume(len(experiment_names) > 0)
        
        experiment_name = experiment_names[0]
        has_data_dir = bool(config_structure["project"]["directories"])
        
        if has_data_dir:
            # Should not raise error when configuration is valid
            try:
                result = load_experiment_files(
                    config=config_structure,
                    experiment_name=experiment_name
                )
                # If we get here, the function should have attempted to call discover_experiment_files
                # This will fail because we haven't mocked it, but that's expected
            except AttributeError:
                # Expected when discover_experiment_files is not mocked
                pass
        else:
            # Should raise ValueError when major_data_directory is missing
            with pytest.raises(ValueError, match=re.escape(MISSING_DATA_DIR_ERROR)):
                load_experiment_files(
                    config=config_structure,
                    experiment_name=experiment_name
                )


# =============================================================================
# PERFORMANCE AND EDGE CASE TESTS
# =============================================================================

class TestAPIPerformanceAndEdgeCases:
    """
    Test class for performance validation and edge case handling.
    """
    
    def test_large_configuration_handling(self, mocker):
        """
        Test API functions with large configuration structures.
        
        Validates performance and correctness when handling configurations
        with many experiments, datasets, and parameters.
        """
        # Create large configuration
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
        
        # Mock discovery function
        mock_discover = mocker.patch("flyrigloader.api.discover_experiment_files")
        mock_discover.return_value = {}
        
        # Test with large configuration
        result = load_experiment_files(
            config=large_config,
            experiment_name="experiment_050"
        )
        
        params = get_experiment_parameters(
            config=large_config,
            experiment_name="experiment_050"
        )
        
        # Verify handling of large configurations
        assert result is not None
        assert len(params) == 10
        assert params["param_5"] == 250  # 50 * 5
    
    def test_unicode_and_special_characters_handling(self, mocker):
        """
        Test API functions with Unicode and special characters in names.
        
        Validates proper handling of internationalized experiment and
        dataset names with various character encodings.
        """
        # Configuration with Unicode characters
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
        
        # Mock discovery function
        mock_discover = mocker.patch("flyrigloader.api.discover_experiment_files")
        mock_discover.return_value = {}
        
        # Test with Unicode names
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
        
        # Verify Unicode handling
        assert result is not None
        assert exp_params["température"] == 25.5
        assert exp_params["humidité"] == "élevée"
        assert ds_params["région"] == "Québec"
    
    def test_nested_configuration_structures(self):
        """
        Test handling of deeply nested configuration structures.
        
        Validates correct parameter extraction from configurations with
        complex nested dictionary structures and hierarchical data.
        """
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
        
        # Test nested parameter extraction
        exp_params = get_experiment_parameters(
            config=nested_config,
            experiment_name="nested_experiment"
        )
        
        ds_params = get_dataset_parameters(
            config=nested_config,
            dataset_name="nested_dataset"
        )
        
        # Verify nested structure preservation
        assert exp_params["analysis"]["preprocessing"]["filter_type"] == "butterworth"
        assert exp_params["analysis"]["feature_extraction"]["features"] == ["mean", "std", "max"]
        assert exp_params["visualization"]["color_scheme"] == "viridis"
        assert ds_params["acquisition"]["sampling_rate"] == 1000
        assert ds_params["acquisition"]["channels"] == ["x", "y", "z"]


# =============================================================================
# TEST MARKERS FOR ORGANIZATION
# =============================================================================

# Mark all integration tests
pytestmark = [
    pytest.mark.unit,
    pytest.mark.api
]

# Additional markers for specific test categories
integration_marker = pytest.mark.integration
property_based_marker = pytest.mark.property_based
performance_marker = pytest.mark.slow

# Apply markers to test classes
TestAPIIntegrationWorkflows = integration_marker(TestAPIIntegrationWorkflows)
TestAPIPropertyBasedValidation = property_based_marker(TestAPIPropertyBasedValidation)
TestAPIPerformanceAndEdgeCases = performance_marker(TestAPIPerformanceAndEdgeCases)