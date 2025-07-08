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
import time
import logging
from copy import deepcopy

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
    # New decoupled architecture functions
    discover_experiment_manifest,
    load_data_file,
    transform_to_dataframe,
    MISSING_DATA_DIR_ERROR
)
from flyrigloader.config.models import LegacyConfigAdapter


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
    functions with realistic return values for integration testing. Now supports
    both dictionary and LegacyConfigAdapter configurations.
    """
    # Mock load_config function
    mock_load_config = mocker.patch("flyrigloader.api.load_config")
    
    # Define base configuration data
    base_config_data = {
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
    
    # Return LegacyConfigAdapter for enhanced configuration support
    mock_load_config.return_value = LegacyConfigAdapter(base_config_data)
    
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


@pytest.fixture(params=['dict', 'legacy_adapter'])
def config_in_both_formats(request, realistic_config_dict):
    """
    Fixture that provides configuration in both dictionary and LegacyConfigAdapter formats.
    
    This enables parametrized testing to ensure all API functions work with both
    configuration formats for comprehensive backward compatibility validation.
    """
    if request.param == 'dict':
        return realistic_config_dict
    else:  # 'legacy_adapter'
        return LegacyConfigAdapter(realistic_config_dict)


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
    
    def test_load_experiment_files_with_both_config_formats(self, config_in_both_formats, mocker):
        """
        Test load_experiment_files with both dictionary and LegacyConfigAdapter formats.
        
        This parametrized test ensures backward compatibility by validating that
        the function works identically with both configuration formats.
        """
        # Mock dependencies
        mock_deps = mocker.patch('flyrigloader.api.get_dependency_provider')
        mock_provider = mocker.Mock()
        mock_deps.return_value = mock_provider
        
        mock_provider.config.get_experiment_info.return_value = {
            'datasets': ['navigation_dataset']
        }
        expected_files = ['/research/experimental_data/nav_file1.pkl', '/research/experimental_data/nav_file2.pkl']
        mock_provider.discovery.discover_experiment_files.return_value = expected_files
        
        # Test with both formats (parametrized fixture handles the variation)
        result = load_experiment_files(
            config=config_in_both_formats,
            experiment_name='navigation_experiment',
            pattern='*.pkl'
        )
        
        # Verify results are identical regardless of config format
        assert result == expected_files
        
        # Verify discovery was called with correct parameters
        mock_provider.discovery.discover_experiment_files.assert_called_once()
        call_kwargs = mock_provider.discovery.discover_experiment_files.call_args.kwargs
        assert call_kwargs['experiment_name'] == 'navigation_experiment'
        assert call_kwargs['pattern'] == '*.pkl'
        assert call_kwargs['base_directory'] == '/research/experimental_data'


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
    
    def test_complete_decoupled_pipeline_integration(
        self,
        config_in_both_formats,
        temp_experimental_data_dir,
        mocker
    ):
        """
        Test complete integration of the new decoupled pipeline architecture.
        
        This test validates the full manifest-based workflow from discovery through
        selective loading and DataFrame transformation, demonstrating the improved
        memory efficiency and selective processing capabilities.
        """
        # Mock dependencies for realistic pipeline test
        mock_deps = mocker.patch('flyrigloader.api.get_dependency_provider')
        mock_provider = mocker.Mock()
        mock_deps.return_value = mock_provider
        
        # Mock experiment info
        mock_provider.config.get_experiment_info.return_value = {
            'datasets': ['navigation_dataset'],
            'parameters': {'analysis_window': 300}
        }
        
        # Create realistic manifest
        manifest = {}
        sample_files = [
            'nav_20240115_trial_1.pkl',
            'nav_20240115_trial_2.pkl', 
            'nav_20240116_trial_1.pkl',
            'nav_20240116_trial_2.pkl'
        ]
        
        for i, filename in enumerate(sample_files):
            file_path = str(temp_experimental_data_dir / "navigation" / filename)
            manifest[file_path] = {
                'path': file_path,
                'size': 1024 + (i * 256),
                'modified': f'2024-01-15T{10 + i}:00:00',
                'metadata': {'trial': str(i + 1), 'condition': 'navigation'},
                'parsed_dates': {'date': f'2024-01-{15 + (i // 2)}'}
            }
        
        mock_provider.discovery.discover_experiment_files.return_value = manifest
        
        # Mock file loading with different data for each file
        def mock_file_loading(file_path):
            file_index = list(manifest.keys()).index(file_path)
            return {
                't': list(range(100 * (file_index + 1))),  # Different sizes per file
                'x': [i * 0.1 * (file_index + 1) for i in range(100 * (file_index + 1))],
                'y': [i * 0.2 * (file_index + 1) for i in range(100 * (file_index + 1))]
            }
        
        mock_provider.io.read_pickle_any_format.side_effect = mock_file_loading
        
        # Mock path operations
        mocker.patch('pathlib.Path.exists', return_value=True)
        mocker.patch('pathlib.Path.stat').return_value.st_size = 2048
        
        # Mock DataFrame transformation
        def mock_dataframe_transform(exp_matrix, config_source=None, metadata=None):
            df = pd.DataFrame(exp_matrix)
            if metadata:
                for key, value in metadata.items():
                    df[key] = value
            return df
        
        mocker.patch('flyrigloader.io.transformers.transform_to_dataframe', side_effect=mock_dataframe_transform)
        
        # Update config to use temporary directory
        if hasattr(config_in_both_formats, '__getitem__'):  # Dictionary or LegacyConfigAdapter
            # Create a copy and update the data directory
            if isinstance(config_in_both_formats, LegacyConfigAdapter):
                config_copy = LegacyConfigAdapter(dict(config_in_both_formats))
            else:
                config_copy = config_in_both_formats.copy()
            config_copy['project']['directories']['major_data_directory'] = str(temp_experimental_data_dir)
        
        # Step 1: Discover complete manifest
        manifest_result = discover_experiment_manifest(
            config=config_copy,
            experiment_name='navigation_experiment',
            extract_metadata=True,
            parse_dates=True
        )
        
        # Verify manifest discovery
        assert len(manifest_result) == 4
        assert all('metadata' in file_info for file_info in manifest_result.values())
        assert all('parsed_dates' in file_info for file_info in manifest_result.values())
        
        # Step 2: Selective data loading (only first 2 files for memory efficiency)
        selected_files = list(manifest_result.keys())[:2]
        loaded_datasets = []
        
        for file_path in selected_files:
            raw_data = load_data_file(file_path, validate_format=True)
            loaded_datasets.append((file_path, raw_data))
        
        # Verify selective loading
        assert len(loaded_datasets) == 2
        assert len(loaded_datasets[0][1]['t']) == 100  # First file has 100 data points
        assert len(loaded_datasets[1][1]['t']) == 200  # Second file has 200 data points
        
        # Step 3: DataFrame transformation with metadata integration
        processed_dataframes = []
        
        for file_path, raw_data in loaded_datasets:
            # Get metadata from manifest
            file_metadata = manifest_result[file_path]['metadata']
            
            # Transform to DataFrame with metadata
            df = transform_to_dataframe(
                raw_data=raw_data,
                metadata=file_metadata,
                add_file_path=True,
                file_path=file_path
            )
            processed_dataframes.append(df)
        
        # Verify DataFrame transformations
        assert len(processed_dataframes) == 2
        
        for i, df in enumerate(processed_dataframes):
            # Check basic structure
            assert 'file_path' in df.columns
            assert 'trial' in df.columns
            assert 'condition' in df.columns
            assert 't' in df.columns
            assert 'x' in df.columns
            assert 'y' in df.columns
            
            # Verify metadata integration
            assert all(df['condition'] == 'navigation')
            assert all(df['trial'] == str(i + 1))
            
            # Verify data integrity
            expected_length = 100 * (i + 1)
            assert len(df) == expected_length
        
        # Step 4: Combine for analysis (optional demonstration)
        combined_df = pd.concat(processed_dataframes, ignore_index=True)
        
        # Verify combined results
        assert len(combined_df) == 300  # 100 + 200 from the two files
        assert len(combined_df['file_path'].unique()) == 2  # Two different source files
        assert set(combined_df['trial'].unique()) == {'1', '2'}  # Two different trials
        
        # Verify performance characteristics (memory efficiency)
        # Only 2 out of 4 available files were loaded and processed
        assert len(loaded_datasets) < len(manifest_result)
        
        # Log successful integration test
        print(f"✓ Successfully completed decoupled pipeline integration test")
        print(f"  Manifest discovered: {len(manifest_result)} files")
        print(f"  Data loaded: {len(loaded_datasets)} files")
        print(f"  DataFrames created: {len(processed_dataframes)}")
        print(f"  Combined data shape: {combined_df.shape}")
        print(f"  Memory efficiency: {len(loaded_datasets)}/{len(manifest_result)} files loaded")


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
# NEW DECOUPLED ARCHITECTURE TESTS
# =============================================================================

class TestDecoupledPipelineFunctions:
    """
    Test class for the new decoupled data loading architecture.
    
    This class validates the three-step manifest-based pipeline:
    1. discover_experiment_manifest() - File discovery with metadata
    2. load_data_file() - Raw data loading from individual files
    3. transform_to_dataframe() - Optional DataFrame transformation
    """
    
    def test_discover_experiment_manifest_basic_functionality(self, mocker):
        """Test basic manifest discovery functionality."""
        # Mock dependencies
        mock_deps = mocker.patch('flyrigloader.api.get_dependency_provider')
        mock_provider = mocker.Mock()
        mock_deps.return_value = mock_provider
        
        # Mock config loading
        mock_provider.config.get_experiment_info.return_value = {
            'datasets': ['test_dataset'],
            'parameters': {'param1': 'value1'}
        }
        
        # Mock file discovery returning manifest format
        expected_manifest = {
            '/data/exp_file1.pkl': {
                'path': '/data/exp_file1.pkl',
                'size': 1024,
                'modified': '2023-01-01T10:00:00',
                'metadata': {'experiment': 'test', 'trial': '1'},
                'parsed_dates': {}
            },
            '/data/exp_file2.pkl': {
                'path': '/data/exp_file2.pkl',
                'size': 2048,
                'modified': '2023-01-01T11:00:00',
                'metadata': {'experiment': 'test', 'trial': '2'},
                'parsed_dates': {}
            }
        }
        mock_provider.discovery.discover_experiment_files.return_value = expected_manifest
        
        # Test manifest discovery
        config_dict = {
            'project': {'directories': {'major_data_directory': '/data'}},
            'experiments': {'test_exp': {'datasets': ['test_dataset']}}
        }
        
        result = discover_experiment_manifest(
            config=config_dict,
            experiment_name='test_exp',
            extract_metadata=True,
            parse_dates=True
        )
        
        # Verify results
        assert result == expected_manifest
        assert len(result) == 2
        assert '/data/exp_file1.pkl' in result
        assert result['/data/exp_file1.pkl']['metadata']['experiment'] == 'test'
        
        # Verify dependency calls
        mock_provider.discovery.discover_experiment_files.assert_called_once()
        call_kwargs = mock_provider.discovery.discover_experiment_files.call_args.kwargs
        assert call_kwargs['experiment_name'] == 'test_exp'
        assert call_kwargs['extract_metadata'] is True
        assert call_kwargs['parse_dates'] is True
    
    def test_discover_experiment_manifest_with_legacy_config_adapter(self, mocker):
        """Test manifest discovery with LegacyConfigAdapter."""
        # Mock dependencies
        mock_deps = mocker.patch('flyrigloader.api.get_dependency_provider')
        mock_provider = mocker.Mock()
        mock_deps.return_value = mock_provider
        
        mock_provider.config.get_experiment_info.return_value = {
            'datasets': ['test_dataset']
        }
        mock_provider.discovery.discover_experiment_files.return_value = {
            '/data/test.pkl': {'path': '/data/test.pkl', 'size': 512, 'metadata': {}}
        }
        
        # Create LegacyConfigAdapter
        config_data = {
            'project': {'directories': {'major_data_directory': '/data'}},
            'experiments': {'test_exp': {'datasets': ['test_dataset']}}
        }
        adapter = LegacyConfigAdapter(config_data)
        
        # Test with LegacyConfigAdapter
        result = discover_experiment_manifest(
            config=adapter,
            experiment_name='test_exp'
        )
        
        # Verify the adapter was converted to dict internally
        assert isinstance(result, dict)
        assert len(result) == 1
    
    def test_discover_experiment_manifest_file_list_conversion(self, mocker):
        """Test manifest discovery converts file lists to manifest format."""
        # Mock dependencies
        mock_deps = mocker.patch('flyrigloader.api.get_dependency_provider')
        mock_provider = mocker.Mock()
        mock_deps.return_value = mock_provider
        
        mock_provider.config.get_experiment_info.return_value = {'datasets': ['test']}
        
        # Mock discovery returning simple file list (legacy format)
        file_list = ['/data/file1.pkl', '/data/file2.pkl']
        mock_provider.discovery.discover_experiment_files.return_value = file_list
        
        # Mock Path.stat for file size
        mock_stat = mocker.patch('pathlib.Path.stat')
        mock_stat.return_value.st_size = 1024
        mock_exists = mocker.patch('pathlib.Path.exists')
        mock_exists.return_value = True
        
        config_dict = {
            'project': {'directories': {'major_data_directory': '/data'}},
            'experiments': {'test_exp': {'datasets': ['test']}}
        }
        
        result = discover_experiment_manifest(
            config=config_dict,
            experiment_name='test_exp'
        )
        
        # Verify conversion to manifest format
        assert isinstance(result, dict)
        assert len(result) == 2
        for file_path in file_list:
            assert file_path in result
            assert result[file_path]['path'] == str(Path(file_path).resolve())
            assert result[file_path]['size'] == 1024
            assert 'metadata' in result[file_path]
    
    def test_load_data_file_basic_functionality(self, mocker):
        """Test basic data file loading functionality."""
        # Mock dependencies
        mock_deps = mocker.patch('flyrigloader.api.get_dependency_provider')
        mock_provider = mocker.Mock()
        mock_deps.return_value = mock_provider
        
        # Mock file existence
        mock_path_exists = mocker.patch('pathlib.Path.exists')
        mock_path_exists.return_value = True
        
        # Mock file stat
        mock_path_stat = mocker.patch('pathlib.Path.stat')
        mock_stat_result = mocker.Mock()
        mock_stat_result.st_size = 2048
        mock_path_stat.return_value = mock_stat_result
        
        # Mock pickle loading
        expected_data = {
            't': [1, 2, 3, 4, 5],
            'x': [0.1, 0.2, 0.3, 0.4, 0.5],
            'y': [0.2, 0.3, 0.4, 0.5, 0.6]
        }
        mock_provider.io.read_pickle_any_format.return_value = expected_data
        
        # Test data loading
        result = load_data_file('/data/test_file.pkl')
        
        # Verify results
        assert result == expected_data
        assert 't' in result
        assert len(result['t']) == 5
        
        # Verify dependency calls
        mock_provider.io.read_pickle_any_format.assert_called_once_with('/data/test_file.pkl')
    
    def test_load_data_file_validation_errors(self, mocker):
        """Test data file loading validation and error handling."""
        # Test with non-existent file
        mock_path_exists = mocker.patch('pathlib.Path.exists')
        mock_path_exists.return_value = False
        
        with pytest.raises(Exception, match="Data file not found"):
            load_data_file('/nonexistent/file.pkl')
        
        # Test with invalid file_path
        with pytest.raises(Exception, match="Invalid file_path"):
            load_data_file("")
        
        with pytest.raises(Exception, match="Invalid file_path"):
            load_data_file(None)
    
    def test_load_data_file_format_validation(self, mocker):
        """Test data file format validation."""
        # Mock dependencies and file existence
        mock_deps = mocker.patch('flyrigloader.api.get_dependency_provider')
        mock_provider = mocker.Mock()
        mock_deps.return_value = mock_provider
        
        mock_path_exists = mocker.patch('pathlib.Path.exists')
        mock_path_exists.return_value = True
        mock_path_stat = mocker.patch('pathlib.Path.stat')
        mock_path_stat.return_value.st_size = 1024
        
        # Test with non-dict data (should trigger warning but not error by default)
        mock_provider.io.read_pickle_any_format.return_value = [1, 2, 3]  # Not a dict
        
        result = load_data_file('/data/test.pkl', validate_format=False)
        assert result == [1, 2, 3]
        
        # Test with validation enabled (should raise error)
        with pytest.raises(ValueError, match="Invalid data format"):
            load_data_file('/data/test.pkl', validate_format=True)
        
        # Test with valid dict format
        valid_data = {'t': [1, 2, 3], 'x': [0.1, 0.2, 0.3]}
        mock_provider.io.read_pickle_any_format.return_value = valid_data
        
        result = load_data_file('/data/test.pkl', validate_format=True)
        assert result == valid_data
    
    def test_transform_to_dataframe_basic_functionality(self, mocker):
        """Test basic DataFrame transformation functionality."""
        # Mock the transform_to_dataframe import
        mock_transform = mocker.patch('flyrigloader.io.transformers.transform_to_dataframe')
        expected_df = pd.DataFrame({
            't': [1, 2, 3],
            'x': [0.1, 0.2, 0.3],
            'y': [0.2, 0.3, 0.4]
        })
        mock_transform.return_value = expected_df
        
        # Test DataFrame transformation
        raw_data = {
            't': [1, 2, 3],
            'x': [0.1, 0.2, 0.3],
            'y': [0.2, 0.3, 0.4]
        }
        
        result = transform_to_dataframe(
            raw_data=raw_data,
            add_file_path=False
        )
        
        # Verify results
        pd.testing.assert_frame_equal(result, expected_df)
        mock_transform.assert_called_once()
        
        # Check call arguments
        call_kwargs = mock_transform.call_args.kwargs
        assert call_kwargs['exp_matrix'] == raw_data
        assert call_kwargs['metadata'] is None
    
    def test_transform_to_dataframe_with_file_path(self, mocker):
        """Test DataFrame transformation with file path addition."""
        # Mock the transform function
        mock_transform = mocker.patch('flyrigloader.io.transformers.transform_to_dataframe')
        base_df = pd.DataFrame({
            't': [1, 2, 3],
            'x': [0.1, 0.2, 0.3]
        })
        mock_transform.return_value = base_df
        
        raw_data = {'t': [1, 2, 3], 'x': [0.1, 0.2, 0.3]}
        file_path = '/data/test_file.pkl'
        
        result = transform_to_dataframe(
            raw_data=raw_data,
            add_file_path=True,
            file_path=file_path
        )
        
        # Verify file_path column was added
        assert 'file_path' in result.columns
        assert all(result['file_path'] == str(Path(file_path).resolve()))
    
    def test_transform_to_dataframe_with_metadata(self, mocker):
        """Test DataFrame transformation with metadata inclusion."""
        mock_transform = mocker.patch('flyrigloader.io.transformers.transform_to_dataframe')
        base_df = pd.DataFrame({'t': [1, 2], 'x': [0.1, 0.2]})
        mock_transform.return_value = base_df
        
        raw_data = {'t': [1, 2], 'x': [0.1, 0.2]}
        metadata = {'experiment': 'test_exp', 'trial': 1}
        
        result = transform_to_dataframe(
            raw_data=raw_data,
            metadata=metadata,
            add_file_path=False
        )
        
        # Verify metadata was passed
        call_kwargs = mock_transform.call_args.kwargs
        assert call_kwargs['metadata'] == metadata
    
    def test_transform_to_dataframe_strict_schema(self, mocker):
        """Test DataFrame transformation with strict schema enforcement."""
        # Mock dependencies
        mock_transform = mocker.patch('flyrigloader.io.transformers.transform_to_dataframe')
        mock_get_config = mocker.patch('flyrigloader.api._get_config_from_source')
        
        # Create DataFrame with extra columns
        df_with_extra = pd.DataFrame({
            't': [1, 2, 3],
            'x': [0.1, 0.2, 0.3],
            'y': [0.2, 0.3, 0.4],
            'extra_col': ['a', 'b', 'c']  # This should be dropped
        })
        mock_transform.return_value = df_with_extra
        
        # Mock column config with only t, x, y allowed
        mock_schema = mocker.Mock()
        mock_schema.columns.keys.return_value = ['t', 'x', 'y']
        mock_get_config.return_value = mock_schema
        
        raw_data = {'t': [1, 2, 3], 'x': [0.1, 0.2, 0.3], 'y': [0.2, 0.3, 0.4]}
        column_config = {'columns': {'t': {}, 'x': {}, 'y': {}}}
        
        result = transform_to_dataframe(
            raw_data=raw_data,
            column_config_path=column_config,
            strict_schema=True,
            add_file_path=False
        )
        
        # Verify extra column was dropped
        assert 'extra_col' not in result.columns
        assert set(result.columns) <= {'t', 'x', 'y'}
    
    def test_transform_to_dataframe_validation_errors(self):
        """Test DataFrame transformation validation errors."""
        # Test with invalid raw_data type
        with pytest.raises(Exception, match="Invalid raw_data"):
            transform_to_dataframe(raw_data="not_a_dict")
        
        with pytest.raises(Exception, match="Invalid raw_data"):
            transform_to_dataframe(raw_data=None)
        
        # Test with empty raw_data
        with pytest.raises(Exception, match="Empty raw_data"):
            transform_to_dataframe(raw_data={})
        
        # Test add_file_path=True without file_path
        raw_data = {'t': [1, 2], 'x': [0.1, 0.2]}
        with pytest.raises(Exception, match="file_path parameter required"):
            transform_to_dataframe(raw_data=raw_data, add_file_path=True, file_path=None)


class TestBackwardCompatibilityTests:
    """
    Test class ensuring backward compatibility between legacy and new pipeline approaches.
    
    This class validates that the existing process_experiment_data() function produces
    identical results to the new three-step manifest-based pipeline workflow.
    """
    
    def test_process_experiment_data_uses_new_architecture(self, mocker):
        """Test that process_experiment_data internally uses the new decoupled architecture."""
        # Mock the new functions
        mock_load_data_file = mocker.patch('flyrigloader.api.load_data_file')
        mock_transform_to_dataframe = mocker.patch('flyrigloader.api.transform_to_dataframe')
        
        # Mock return values
        raw_data = {'t': [1, 2, 3], 'x': [0.1, 0.2, 0.3]}
        expected_df = pd.DataFrame(raw_data)
        expected_df['file_path'] = '/data/test.pkl'
        
        mock_load_data_file.return_value = raw_data
        mock_transform_to_dataframe.return_value = expected_df
        
        # Test process_experiment_data
        result = process_experiment_data('/data/test.pkl')
        
        # Verify new functions were called
        mock_load_data_file.assert_called_once_with(
            file_path='/data/test.pkl',
            validate_format=True,
            _deps=mocker.ANY
        )
        mock_transform_to_dataframe.assert_called_once_with(
            raw_data=raw_data,
            column_config_path=None,
            metadata=None,
            add_file_path=True,
            file_path='/data/test.pkl',
            strict_schema=False,
            _deps=mocker.ANY
        )
        
        # Verify result
        pd.testing.assert_frame_equal(result, expected_df)
    
    def test_equivalent_results_legacy_vs_new_pipeline(self, mocker, temp_experimental_data_dir):
        """Test that legacy and new pipeline produce identical results."""
        # Create test data file
        test_file = temp_experimental_data_dir / "test_data.pkl"
        test_data = {
            't': [1.0, 2.0, 3.0, 4.0],
            'x': [0.1, 0.2, 0.3, 0.4],
            'y': [0.2, 0.3, 0.4, 0.5]
        }
        
        # Mock pickle loading
        mock_deps = mocker.patch('flyrigloader.api.get_dependency_provider')
        mock_provider = mocker.Mock()
        mock_deps.return_value = mock_provider
        mock_provider.io.read_pickle_any_format.return_value = test_data
        
        # Mock path operations
        mocker.patch('pathlib.Path.exists', return_value=True)
        mocker.patch('pathlib.Path.stat').return_value.st_size = 1024
        
        # Mock transform function to return DataFrame
        def mock_transform_func(exp_matrix, config_source=None, metadata=None):
            df = pd.DataFrame(exp_matrix)
            if metadata:
                for key, value in metadata.items():
                    df[key] = value
            return df
        
        mocker.patch('flyrigloader.io.transformers.transform_to_dataframe', side_effect=mock_transform_func)
        
        # Test legacy approach
        legacy_result = process_experiment_data(str(test_file))
        
        # Test new approach
        raw_data = load_data_file(str(test_file))
        new_result = transform_to_dataframe(
            raw_data=raw_data,
            add_file_path=True,
            file_path=str(test_file)
        )
        
        # Compare results (excluding file_path column which might have different formatting)
        data_columns = [col for col in legacy_result.columns if col != 'file_path']
        pd.testing.assert_frame_equal(
            legacy_result[data_columns].sort_index(axis=1),
            new_result[data_columns].sort_index(axis=1)
        )
        
        # Verify both have file_path column
        assert 'file_path' in legacy_result.columns
        assert 'file_path' in new_result.columns
    
    def test_side_by_side_result_comparison(self, mocker):
        """Test side-by-side comparison of legacy vs new pipeline results."""
        # Mock data
        test_data = {
            't': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            'x': np.array([1.1, 1.2, 1.3, 1.4, 1.5]),
            'y': np.array([2.1, 2.2, 2.3, 2.4, 2.5])
        }
        
        # Mock dependencies
        mock_deps = mocker.patch('flyrigloader.api.get_dependency_provider')
        mock_provider = mocker.Mock()
        mock_deps.return_value = mock_provider
        mock_provider.io.read_pickle_any_format.return_value = test_data
        
        mocker.patch('pathlib.Path.exists', return_value=True)
        mocker.patch('pathlib.Path.stat').return_value.st_size = 2048
        
        # Mock transform to return consistent DataFrame
        def consistent_transform(exp_matrix, config_source=None, metadata=None):
            df = pd.DataFrame(exp_matrix)
            df = df.round(6)  # Ensure consistent precision
            return df
        
        mocker.patch('flyrigloader.io.transformers.transform_to_dataframe', side_effect=consistent_transform)
        
        file_path = '/data/comparison_test.pkl'
        
        # Run both approaches multiple times to test consistency
        results_legacy = []
        results_new = []
        
        for i in range(3):
            # Legacy approach
            legacy_df = process_experiment_data(file_path)
            results_legacy.append(deepcopy(legacy_df))
            
            # New approach
            raw_data = load_data_file(file_path)
            new_df = transform_to_dataframe(
                raw_data=raw_data,
                add_file_path=True,
                file_path=file_path
            )
            results_new.append(deepcopy(new_df))
        
        # Verify consistency within each approach
        for i in range(1, len(results_legacy)):
            data_cols = [col for col in results_legacy[0].columns if col != 'file_path']
            pd.testing.assert_frame_equal(
                results_legacy[0][data_cols],
                results_legacy[i][data_cols],
                check_exact=False,
                rtol=1e-10
            )
        
        for i in range(1, len(results_new)):
            data_cols = [col for col in results_new[0].columns if col != 'file_path']
            pd.testing.assert_frame_equal(
                results_new[0][data_cols],
                results_new[i][data_cols],
                check_exact=False,
                rtol=1e-10
            )
        
        # Verify equivalence between approaches
        data_cols = [col for col in results_legacy[0].columns if col != 'file_path']
        pd.testing.assert_frame_equal(
            results_legacy[0][data_cols].sort_index(axis=1),
            results_new[0][data_cols].sort_index(axis=1),
            check_exact=False,
            rtol=1e-10
        )


class TestEnhancedBaseDirectoryResolution:
    """
    Test class for enhanced _resolve_base_directory function with logging and audit trails.
    
    This class validates the precedence order, logging functionality, and comprehensive
    error handling of the enhanced base directory resolution system.
    """
    
    def test_base_directory_precedence_explicit_parameter(self, caplog):
        """Test precedence 1: explicit base_directory parameter."""
        config = {
            'project': {
                'directories': {
                    'major_data_directory': '/config/data'
                }
            }
        }
        explicit_dir = '/explicit/override/data'
        
        with caplog.at_level(logging.INFO):
            result = _resolve_base_directory(config, explicit_dir, 'test_operation')
        
        # Verify precedence
        assert result == explicit_dir
        
        # Verify logging
        info_messages = [record.message for record in caplog.records if record.levelname == 'INFO']
        assert any('Using explicit base_directory parameter' in msg for msg in info_messages)
        assert any('✓ Base directory resolution complete' in msg for msg in info_messages)
        assert any('explicit function parameter' in msg for msg in info_messages)
    
    def test_base_directory_precedence_config_directory(self, caplog):
        """Test precedence 2: configuration major_data_directory."""
        config = {
            'project': {
                'directories': {
                    'major_data_directory': '/config/data/path'
                }
            }
        }
        
        with caplog.at_level(logging.INFO):
            result = _resolve_base_directory(config, None, 'test_operation')
        
        # Verify precedence
        assert result == '/config/data/path'
        
        # Verify logging
        info_messages = [record.message for record in caplog.records if record.levelname == 'INFO']
        assert any('Using major_data_directory from config' in msg for msg in info_messages)
        assert any('configuration major_data_directory' in msg for msg in info_messages)
    
    def test_base_directory_precedence_environment_variable(self, caplog, monkeypatch):
        """Test precedence 3: FLYRIGLOADER_DATA_DIR environment variable."""
        # Set environment variable
        env_dir = '/env/variable/data'
        monkeypatch.setenv('FLYRIGLOADER_DATA_DIR', env_dir)
        
        # Config without major_data_directory
        config = {
            'project': {
                'directories': {}
            }
        }
        
        with caplog.at_level(logging.INFO):
            result = _resolve_base_directory(config, None, 'test_operation')
        
        # Verify precedence
        assert result == env_dir
        
        # Verify logging
        info_messages = [record.message for record in caplog.records if record.levelname == 'INFO']
        assert any('Using FLYRIGLOADER_DATA_DIR environment variable' in msg for msg in info_messages)
        assert any('FLYRIGLOADER_DATA_DIR environment variable' in msg for msg in info_messages)
    
    def test_base_directory_resolution_failure(self, caplog, monkeypatch):
        """Test comprehensive error when no base directory can be resolved."""
        # Clear environment variable
        monkeypatch.delenv('FLYRIGLOADER_DATA_DIR', raising=False)
        
        # Config without major_data_directory
        config = {
            'project': {
                'directories': {}
            }
        }
        
        with pytest.raises(Exception) as exc_info:
            _resolve_base_directory(config, None, 'test_operation')
        
        # Verify comprehensive error message
        error_msg = str(exc_info.value)
        assert 'No data directory specified for test_operation' in error_msg
        assert 'Tried all resolution methods in precedence order' in error_msg
        assert '1. Explicit \'base_directory\' parameter (not provided)' in error_msg
        assert '2. Configuration \'project.directories.major_data_directory\' (not found)' in error_msg
        assert '3. Environment variable \'FLYRIGLOADER_DATA_DIR\' (not set)' in error_msg
        assert 'Please provide a data directory using one of these methods' in error_msg
    
    def test_base_directory_audit_logging(self, caplog, mocker):
        """Test comprehensive audit logging for base directory resolution."""
        # Mock Path.exists to avoid filesystem checks
        mock_path_exists = mocker.patch('pathlib.Path.exists')
        mock_path_exists.return_value = True
        
        config = {
            'project': {
                'directories': {
                    'major_data_directory': '/audit/test/data'
                }
            }
        }
        
        with caplog.at_level(logging.DEBUG):
            result = _resolve_base_directory(config, None, 'audit_test_operation')
        
        # Verify detailed logging
        debug_messages = [record.message for record in caplog.records if record.levelname == 'DEBUG']
        info_messages = [record.message for record in caplog.records if record.levelname == 'INFO']
        
        # Check debug logging
        assert any('Starting base directory resolution for audit_test_operation' in msg for msg in debug_messages)
        assert any('Resolution precedence: 1) explicit parameter, 2) config major_data_directory, 3) FLYRIGLOADER_DATA_DIR env var' in msg for msg in debug_messages)
        assert any('No explicit base_directory provided' in msg for msg in debug_messages)
        assert any('Resolution source: configuration major_data_directory (precedence 2)' in msg for msg in debug_messages)
        assert any('Resolved directory exists' in msg for msg in debug_messages)
        
        # Check info logging
        assert any('Using major_data_directory from config: /audit/test/data' in msg for msg in info_messages)
        assert any('✓ Base directory resolution complete for audit_test_operation' in msg for msg in info_messages)
        assert any('Resolved path: /audit/test/data' in msg for msg in info_messages)
        assert any('Resolution source: configuration major_data_directory' in msg for msg in info_messages)
    
    def test_base_directory_with_legacy_config_adapter(self, caplog):
        """Test base directory resolution with LegacyConfigAdapter."""
        config_data = {
            'project': {
                'directories': {
                    'major_data_directory': '/adapter/test/data'
                }
            }
        }
        adapter = LegacyConfigAdapter(config_data)
        
        with caplog.at_level(logging.INFO):
            result = _resolve_base_directory(adapter, None, 'adapter_test')
        
        # Verify it works with adapter
        assert result == '/adapter/test/data'
        
        # Verify logging
        info_messages = [record.message for record in caplog.records if record.levelname == 'INFO']
        assert any('Using major_data_directory from config' in msg for msg in info_messages)
    
    def test_base_directory_nonexistent_path_warning(self, caplog, mocker):
        """Test warning when resolved directory doesn't exist."""
        # Mock Path.exists to return False
        mock_path_exists = mocker.patch('pathlib.Path.exists')
        mock_path_exists.return_value = False
        
        config = {
            'project': {
                'directories': {
                    'major_data_directory': '/nonexistent/path'
                }
            }
        }
        
        with caplog.at_level(logging.WARNING):
            result = _resolve_base_directory(config, None, 'nonexistent_test')
        
        # Verify result still returned
        assert result == '/nonexistent/path'
        
        # Verify warning logged
        warning_messages = [record.message for record in caplog.records if record.levelname == 'WARNING']
        assert any('Resolved directory does not exist: /nonexistent/path' in msg for msg in warning_messages)
        assert any('This may cause file discovery failures in nonexistent_test' in msg for msg in warning_messages)


class TestLegacyConfigAdapterIntegration:
    """
    Test class for LegacyConfigAdapter integration with existing API functions.
    
    This class validates that API functions work correctly with both dictionary
    configurations and LegacyConfigAdapter objects, ensuring backward compatibility.
    """
    
    def test_load_experiment_files_with_legacy_adapter(self, mocker):
        """Test load_experiment_files with LegacyConfigAdapter."""
        # Mock dependencies
        mock_deps = mocker.patch('flyrigloader.api.get_dependency_provider')
        mock_provider = mocker.Mock()
        mock_deps.return_value = mock_provider
        
        mock_provider.config.get_experiment_info.return_value = {
            'datasets': ['test_dataset']
        }
        mock_provider.discovery.discover_experiment_files.return_value = ['/data/file1.pkl']
        
        # Create LegacyConfigAdapter
        config_data = {
            'project': {'directories': {'major_data_directory': '/test/data'}},
            'experiments': {'test_exp': {'datasets': ['test_dataset']}}
        }
        adapter = LegacyConfigAdapter(config_data)
        
        # Test with adapter
        result = load_experiment_files(
            config=adapter,
            experiment_name='test_exp'
        )
        
        # Verify it works
        assert result == ['/data/file1.pkl']
        
        # Verify calls were made with proper config
        mock_provider.discovery.discover_experiment_files.assert_called_once()
        call_kwargs = mock_provider.discovery.discover_experiment_files.call_args.kwargs
        assert call_kwargs['experiment_name'] == 'test_exp'
        assert call_kwargs['base_directory'] == '/test/data'
    
    def test_get_experiment_parameters_with_legacy_adapter(self, mocker):
        """Test get_experiment_parameters with LegacyConfigAdapter."""
        # Mock dependencies
        mock_deps = mocker.patch('flyrigloader.api.get_dependency_provider')
        mock_provider = mocker.Mock()
        mock_deps.return_value = mock_provider
        
        expected_params = {
            'threshold': 0.5,
            'analysis_method': 'correlation',
            'window_size': 10
        }
        mock_provider.config.get_experiment_info.return_value = {
            'parameters': expected_params
        }
        
        # Create LegacyConfigAdapter with experiment parameters
        config_data = {
            'experiments': {
                'param_test_exp': {
                    'datasets': ['test_dataset'],
                    'parameters': expected_params
                }
            }
        }
        adapter = LegacyConfigAdapter(config_data)
        
        # Test parameter retrieval
        result = get_experiment_parameters(
            config=adapter,
            experiment_name='param_test_exp'
        )
        
        # Verify results
        assert result == expected_params
        mock_provider.config.get_experiment_info.assert_called_once()
    
    def test_legacy_adapter_pydantic_model_access(self):
        """Test accessing underlying Pydantic models from LegacyConfigAdapter."""
        config_data = {
            'project': {
                'directories': {'major_data_directory': '/test/data'},
                'ignore_substrings': ['temp', 'backup']
            },
            'datasets': {
                'test_dataset': {
                    'rig': 'rig1',
                    'dates_vials': {'2023-01-01': [1, 2, 3]}
                }
            },
            'experiments': {
                'test_experiment': {
                    'datasets': ['test_dataset'],
                    'parameters': {'param1': 'value1'}
                }
            }
        }
        
        adapter = LegacyConfigAdapter(config_data)
        
        # Test accessing Pydantic models
        project_model = adapter.get_model('project')
        assert project_model is not None
        assert project_model.directories['major_data_directory'] == '/test/data'
        assert project_model.ignore_substrings == ['temp', 'backup']
        
        dataset_model = adapter.get_model('dataset', 'test_dataset')
        assert dataset_model is not None
        assert dataset_model.rig == 'rig1'
        assert dataset_model.dates_vials == {'2023-01-01': [1, 2, 3]}
        
        experiment_model = adapter.get_model('experiment', 'test_experiment')
        assert experiment_model is not None
        assert experiment_model.datasets == ['test_dataset']
        assert experiment_model.parameters == {'param1': 'value1'}
    
    def test_legacy_adapter_dictionary_interface(self):
        """Test that LegacyConfigAdapter maintains dictionary interface."""
        config_data = {
            'project': {'directories': {'major_data_directory': '/test'}},
            'datasets': {'ds1': {'rig': 'rig1'}},
            'experiments': {'exp1': {'datasets': ['ds1']}}
        }
        
        adapter = LegacyConfigAdapter(config_data)
        
        # Test dictionary-style access
        assert adapter['project']['directories']['major_data_directory'] == '/test'
        assert adapter['datasets']['ds1']['rig'] == 'rig1'
        assert adapter['experiments']['exp1']['datasets'] == ['ds1']
        
        # Test dictionary methods
        assert 'project' in adapter
        assert 'nonexistent' not in adapter
        assert adapter.get('project') is not None
        assert adapter.get('nonexistent') is None
        assert adapter.get('nonexistent', 'default') == 'default'
        
        # Test iteration
        keys = list(adapter.keys())
        assert 'project' in keys
        assert 'datasets' in keys
        assert 'experiments' in keys
        
        values = list(adapter.values())
        assert len(values) == 3
        
        items = list(adapter.items())
        assert len(items) == 3
        assert any(key == 'project' for key, value in items)
    
    def test_legacy_adapter_validation(self):
        """Test LegacyConfigAdapter validation functionality."""
        # Valid configuration
        valid_config = {
            'project': {
                'directories': {'major_data_directory': '/valid/path'},
                'extraction_patterns': [r'(?P<date>\d{4}-\d{2}-\d{2})']
            },
            'datasets': {
                'valid_dataset': {
                    'rig': 'rig1',
                    'dates_vials': {'2023-01-01': [1, 2, 3]}
                }
            },
            'experiments': {
                'valid_experiment': {
                    'datasets': ['valid_dataset'],
                    'parameters': {'threshold': 0.5}
                }
            }
        }
        
        adapter = LegacyConfigAdapter(valid_config)
        assert adapter.validate_all() is True
        
        # Invalid configuration
        invalid_config = {
            'project': {
                'directories': {'major_data_directory': '/valid/path'},
                'extraction_patterns': ['[invalid regex']  # Invalid regex
            }
        }
        
        adapter_invalid = LegacyConfigAdapter(invalid_config)
        assert adapter_invalid.validate_all() is False


class TestPerformanceValidation:
    """
    Test class for performance validation of the new decoupled architecture.
    
    This class ensures SLA compliance for manifest generation, data loading,
    and transformation operations with performance benchmarks and timing tests.
    """
    
    def test_manifest_discovery_performance(self, mocker):
        """Test performance of manifest discovery for large datasets."""
        # Mock dependencies
        mock_deps = mocker.patch('flyrigloader.api.get_dependency_provider')
        mock_provider = mocker.Mock()
        mock_deps.return_value = mock_provider
        
        mock_provider.config.get_experiment_info.return_value = {'datasets': ['large_dataset']}
        
        # Simulate large manifest (1000 files)
        large_manifest = {}
        for i in range(1000):
            file_path = f'/data/large_exp_file_{i:04d}.pkl'
            large_manifest[file_path] = {
                'path': file_path,
                'size': 1024 + i,
                'modified': f'2023-01-01T{i % 24:02d}:00:00',
                'metadata': {'trial': str(i), 'condition': f'cond_{i % 10}'},
                'parsed_dates': {}
            }
        
        mock_provider.discovery.discover_experiment_files.return_value = large_manifest
        
        # Measure performance
        config_dict = {
            'project': {'directories': {'major_data_directory': '/data'}},
            'experiments': {'large_exp': {'datasets': ['large_dataset']}}
        }
        
        start_time = time.perf_counter()
        result = discover_experiment_manifest(
            config=config_dict,
            experiment_name='large_exp',
            extract_metadata=True
        )
        end_time = time.perf_counter()
        
        # Performance assertions
        discovery_time = end_time - start_time
        assert discovery_time < 5.0, f"Manifest discovery took {discovery_time:.2f}s, should be < 5.0s"
        assert len(result) == 1000
        
        # Verify structure is maintained
        sample_file = list(result.keys())[0]
        assert 'metadata' in result[sample_file]
        assert 'size' in result[sample_file]
    
    def test_data_loading_performance(self, mocker):
        """Test performance of individual file loading operations."""
        # Mock dependencies
        mock_deps = mocker.patch('flyrigloader.api.get_dependency_provider')
        mock_provider = mocker.Mock()
        mock_deps.return_value = mock_provider
        
        mocker.patch('pathlib.Path.exists', return_value=True)
        mocker.patch('pathlib.Path.stat').return_value.st_size = 1024 * 1024  # 1MB
        
        # Simulate medium-sized data
        large_data = {
            't': list(range(10000)),  # 10k time points
            'x': [i * 0.01 for i in range(10000)],
            'y': [i * 0.02 for i in range(10000)],
            'z': [i * 0.03 for i in range(10000)]
        }
        mock_provider.io.read_pickle_any_format.return_value = large_data
        
        # Measure loading performance
        file_path = '/data/large_file.pkl'
        
        start_time = time.perf_counter()
        result = load_data_file(file_path)
        end_time = time.perf_counter()
        
        # Performance assertions
        loading_time = end_time - start_time
        assert loading_time < 2.0, f"Data loading took {loading_time:.2f}s, should be < 2.0s"
        assert result == large_data
        assert len(result['t']) == 10000
    
    def test_dataframe_transformation_performance(self, mocker):
        """Test performance of DataFrame transformation operations."""
        # Mock transform function to simulate realistic processing time
        def mock_transform_with_delay(exp_matrix, config_source=None, metadata=None):
            # Simulate some processing time for large datasets
            data_size = len(next(iter(exp_matrix.values())))
            if data_size > 5000:
                time.sleep(0.01)  # Simulate processing delay for large data
            
            df = pd.DataFrame(exp_matrix)
            if metadata:
                for key, value in metadata.items():
                    df[key] = value
            return df
        
        mocker.patch('flyrigloader.io.transformers.transform_to_dataframe', side_effect=mock_transform_with_delay)
        
        # Large dataset for transformation
        large_raw_data = {
            't': list(range(8000)),
            'x': [i * 0.1 for i in range(8000)],
            'y': [i * 0.2 for i in range(8000)]
        }
        
        metadata = {'experiment': 'performance_test', 'trial': 1}
        
        # Measure transformation performance
        start_time = time.perf_counter()
        result = transform_to_dataframe(
            raw_data=large_raw_data,
            metadata=metadata,
            add_file_path=True,
            file_path='/data/perf_test.pkl'
        )
        end_time = time.perf_counter()
        
        # Performance assertions
        transform_time = end_time - start_time
        assert transform_time < 3.0, f"DataFrame transformation took {transform_time:.2f}s, should be < 3.0s"
        assert len(result) == 8000
        assert 'file_path' in result.columns
    
    def test_end_to_end_pipeline_performance(self, mocker):
        """Test performance of complete end-to-end pipeline."""
        # Mock all dependencies for realistic performance test
        mock_deps = mocker.patch('flyrigloader.api.get_dependency_provider')
        mock_provider = mocker.Mock()
        mock_deps.return_value = mock_provider
        
        # Mock manifest discovery
        manifest = {}
        for i in range(50):  # 50 files
            file_path = f'/data/pipeline_file_{i:02d}.pkl'
            manifest[file_path] = {
                'path': file_path,
                'size': 2048,
                'metadata': {'trial': str(i)},
                'parsed_dates': {}
            }
        
        mock_provider.config.get_experiment_info.return_value = {'datasets': ['pipeline_dataset']}
        mock_provider.discovery.discover_experiment_files.return_value = manifest
        
        # Mock file loading and transformation
        sample_data = {
            't': list(range(1000)),
            'x': [i * 0.1 for i in range(1000)],
            'y': [i * 0.2 for i in range(1000)]
        }
        mock_provider.io.read_pickle_any_format.return_value = sample_data
        
        mocker.patch('pathlib.Path.exists', return_value=True)
        mocker.patch('pathlib.Path.stat').return_value.st_size = 2048
        
        def mock_transform(exp_matrix, config_source=None, metadata=None):
            df = pd.DataFrame(exp_matrix)
            if metadata:
                for key, value in metadata.items():
                    df[key] = value
            return df
        
        mocker.patch('flyrigloader.io.transformers.transform_to_dataframe', side_effect=mock_transform)
        
        config_dict = {
            'project': {'directories': {'major_data_directory': '/data'}},
            'experiments': {'pipeline_exp': {'datasets': ['pipeline_dataset']}}
        }
        
        # Measure end-to-end performance
        start_time = time.perf_counter()
        
        # Step 1: Discover manifest
        manifest_result = discover_experiment_manifest(
            config=config_dict,
            experiment_name='pipeline_exp'
        )
        
        # Step 2: Load first 10 files
        loaded_data = []
        for file_path in list(manifest_result.keys())[:10]:
            raw_data = load_data_file(file_path)
            loaded_data.append(raw_data)
        
        # Step 3: Transform first 5 files
        dataframes = []
        for i, raw_data in enumerate(loaded_data[:5]):
            df = transform_to_dataframe(
                raw_data=raw_data,
                add_file_path=True,
                file_path=list(manifest_result.keys())[i]
            )
            dataframes.append(df)
        
        end_time = time.perf_counter()
        
        # Performance assertions
        total_time = end_time - start_time
        assert total_time < 10.0, f"End-to-end pipeline took {total_time:.2f}s, should be < 10.0s"
        assert len(manifest_result) == 50
        assert len(loaded_data) == 10
        assert len(dataframes) == 5
        
        # Verify data integrity
        for df in dataframes:
            assert 'file_path' in df.columns
            assert len(df) == 1000  # Sample data size
    
    def test_memory_efficiency_validation(self, mocker):
        """Test memory efficiency of the decoupled architecture."""
        # This test validates that the new architecture doesn't load unnecessary data
        mock_deps = mocker.patch('flyrigloader.api.get_dependency_provider')
        mock_provider = mocker.Mock()
        mock_deps.return_value = mock_provider
        
        # Track how many times data loading is called
        load_call_count = 0
        
        def track_load_calls(file_path):
            nonlocal load_call_count
            load_call_count += 1
            return {'t': [1, 2, 3], 'x': [0.1, 0.2, 0.3]}
        
        mock_provider.io.read_pickle_any_format.side_effect = track_load_calls
        mocker.patch('pathlib.Path.exists', return_value=True)
        mocker.patch('pathlib.Path.stat').return_value.st_size = 1024
        
        # Test selective loading - only load 3 out of 10 available files
        manifest = {}
        for i in range(10):
            file_path = f'/data/memory_test_{i}.pkl'
            manifest[file_path] = {'path': file_path, 'size': 1024}
        
        mock_provider.config.get_experiment_info.return_value = {'datasets': ['memory_dataset']}
        mock_provider.discovery.discover_experiment_files.return_value = manifest
        
        config_dict = {
            'project': {'directories': {'major_data_directory': '/data'}},
            'experiments': {'memory_exp': {'datasets': ['memory_dataset']}}
        }
        
        # Discover manifest (should not load any data)
        manifest_result = discover_experiment_manifest(
            config=config_dict,
            experiment_name='memory_exp'
        )
        
        # Verify no data loading occurred during manifest discovery
        assert load_call_count == 0
        assert len(manifest_result) == 10
        
        # Load only 3 specific files
        selected_files = list(manifest_result.keys())[:3]
        for file_path in selected_files:
            load_data_file(file_path)
        
        # Verify only 3 load calls were made
        assert load_call_count == 3, f"Expected 3 load calls, got {load_call_count}"


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

# Apply markers to new test classes
TestDecoupledPipelineFunctions = pytest.mark.unit(TestDecoupledPipelineFunctions)
TestBackwardCompatibilityTests = integration_marker(TestBackwardCompatibilityTests)
TestEnhancedBaseDirectoryResolution = pytest.mark.unit(TestEnhancedBaseDirectoryResolution)
TestLegacyConfigAdapterIntegration = integration_marker(TestLegacyConfigAdapterIntegration)
TestPerformanceValidation = performance_marker(TestPerformanceValidation)