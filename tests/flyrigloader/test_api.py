"""
Tests for API functions in the flyrigloader.api module.

This module provides comprehensive test coverage for the high-level API functions,
implementing modern pytest practices with parametrization, fixtures, and integration testing.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import pytest
from hypothesis import given, strategies as st, assume
from unittest.mock import MagicMock

from flyrigloader.api import (
    load_experiment_files,
    load_dataset_files,
    get_dataset_parameters,
    get_experiment_parameters,
    MISSING_DATA_DIR_ERROR
)


# =============================================================================
# Test Data Strategies for Hypothesis Property-Based Testing
# =============================================================================

@st.composite
def valid_config_strategy(draw):
    """Generate valid configuration dictionaries for property-based testing."""
    return {
        "project": {
            "directories": {
                "major_data_directory": draw(st.text(min_size=1, max_size=100))
            }
        },
        "experiments": {
            draw(st.text(min_size=1, max_size=50)): {
                "datasets": [draw(st.text(min_size=1, max_size=50))]
            }
        },
        "datasets": {
            draw(st.text(min_size=1, max_size=50)): {
                "patterns": [draw(st.text(min_size=1, max_size=50))]
            }
        }
    }


@st.composite
def experiment_params_strategy(draw):
    """Generate valid experiment parameters for testing."""
    return {
        "experiment_name": draw(st.text(min_size=1, max_size=50)),
        "base_directory": draw(st.one_of(st.none(), st.text(min_size=1, max_size=100))),
        "pattern": draw(st.text(min_size=1, max_size=20)),
        "recursive": draw(st.booleans()),
        "extensions": draw(st.one_of(st.none(), st.lists(st.text(min_size=1, max_size=10)))),
        "extract_metadata": draw(st.booleans()),
        "parse_dates": draw(st.booleans())
    }


# =============================================================================
# Test Cases for load_experiment_files
# =============================================================================

class TestLoadExperimentFiles:
    """Test suite for load_experiment_files function."""

    @pytest.mark.parametrize("config_source,experiment_name,expected_calls", [
        # Test with config_path parameter
        (
            {"config_path": "/path/to/config.yaml"},
            "test_experiment",
            {
                "load_config_calls": 1,
                "discover_experiment_calls": 1,
                "config_arg": "mock_config"
            }
        ),
        # Test with config dictionary parameter
        (
            {"config": {"project": {"directories": {"major_data_directory": "/data"}}}},
            "test_experiment",
            {
                "load_config_calls": 0,
                "discover_experiment_calls": 1,
                "config_arg": "config_dict"
            }
        ),
    ])
    def test_load_experiment_files_basic_scenarios(
        self, mock_config_and_discovery, config_source, experiment_name, expected_calls
    ):
        """Test basic load_experiment_files scenarios with different config sources."""
        mock_load_config, mock_discover_experiment_files, _ = mock_config_and_discovery
        
        # Call the function under test
        result = load_experiment_files(
            experiment_name=experiment_name,
            **config_source
        )
        
        # Verify mock calls based on expected behavior
        assert mock_load_config.call_count == expected_calls["load_config_calls"]
        assert mock_discover_experiment_files.call_count == expected_calls["discover_experiment_calls"]
        
        # Verify result is what discovery function returns
        assert result == mock_discover_experiment_files.return_value


    @pytest.mark.parametrize("custom_params,expected_discovery_args", [
        # Test with custom base_directory
        (
            {"base_directory": "/custom/data/dir"},
            {"base_directory": "/custom/data/dir"}
        ),
        # Test with custom pattern
        (
            {"pattern": "*.csv"},
            {"pattern": "*.csv"}
        ),
        # Test with non-recursive search
        (
            {"recursive": False},
            {"recursive": False}
        ),
        # Test with custom extensions
        (
            {"extensions": [".csv", ".pkl"]},
            {"extensions": [".csv", ".pkl"]}
        ),
        # Test with metadata extraction enabled
        (
            {"extract_metadata": True},
            {"extract_metadata": True}
        ),
        # Test with date parsing enabled
        (
            {"parse_dates": True, "extract_metadata": True},
            {"parse_dates": True, "extract_metadata": True}
        ),
        # Test with multiple custom parameters
        (
            {"base_directory": "/custom", "pattern": "*.pkl", "recursive": False},
            {"base_directory": "/custom", "pattern": "*.pkl", "recursive": False}
        ),
    ])
    def test_load_experiment_files_parameter_forwarding(
        self, mock_config_and_discovery, sample_config_dict, custom_params, expected_discovery_args
    ):
        """Test that custom parameters are properly forwarded to discovery function."""
        _, mock_discover_experiment_files, _ = mock_config_and_discovery
        
        # Call function with custom parameters
        load_experiment_files(
            config=sample_config_dict,
            experiment_name="test_experiment",
            **custom_params
        )
        
        # Verify discovery function was called with expected parameters
        call_args = mock_discover_experiment_files.call_args
        for param_name, expected_value in expected_discovery_args.items():
            assert call_args.kwargs[param_name] == expected_value


    @pytest.mark.parametrize("invalid_config_params,expected_error", [
        # Test with neither config_path nor config provided
        ({}, "Exactly one of 'config_path' or 'config' must be provided"),
        # Test with both config_path and config provided
        (
            {"config_path": "/path/config.yaml", "config": {}},
            "Exactly one of 'config_path' or 'config' must be provided"
        ),
    ])
    def test_load_experiment_files_config_validation_errors(
        self, invalid_config_params, expected_error
    ):
        """Test that proper ValueError is raised for invalid config parameters."""
        with pytest.raises(ValueError, match=re.escape(expected_error)):
            load_experiment_files(
                experiment_name="test_experiment",
                **invalid_config_params
            )


    def test_load_experiment_files_missing_data_directory_error(self, sample_config_dict):
        """Test ValueError when major_data_directory is missing and no base_directory provided."""
        config = sample_config_dict.copy()
        config["project"]["directories"].pop("major_data_directory")
        
        with pytest.raises(ValueError, match=re.escape(MISSING_DATA_DIR_ERROR)):
            load_experiment_files(config=config, experiment_name="test_experiment")


    @given(
        config=valid_config_strategy(),
        params=experiment_params_strategy()
    )
    def test_load_experiment_files_property_based(self, mocker, config, params):
        """Property-based test for load_experiment_files with various valid inputs."""
        # Assume we have a valid experiment in the config
        experiment_name = list(config["experiments"].keys())[0]
        assume(experiment_name != "")
        
        # Mock the dependencies
        mock_discover = mocker.patch("flyrigloader.api.discover_experiment_files")
        mock_discover.return_value = ["/mock/file1.pkl", "/mock/file2.pkl"]
        
        # Test the function
        result = load_experiment_files(
            config=config,
            experiment_name=experiment_name,
            base_directory=params["base_directory"],
            pattern=params["pattern"],
            recursive=params["recursive"],
            extensions=params["extensions"],
            extract_metadata=params["extract_metadata"],
            parse_dates=params["parse_dates"]
        )
        
        # Verify function was called and returned expected result
        assert mock_discover.called
        assert result == mock_discover.return_value


# =============================================================================
# Test Cases for load_dataset_files
# =============================================================================

class TestLoadDatasetFiles:
    """Test suite for load_dataset_files function."""

    @pytest.mark.parametrize("config_source,dataset_name,expected_calls", [
        # Test with config_path parameter
        (
            {"config_path": "/path/to/config.yaml"},
            "test_dataset",
            {
                "load_config_calls": 1,
                "discover_dataset_calls": 1
            }
        ),
        # Test with config dictionary parameter
        (
            {"config": {"project": {"directories": {"major_data_directory": "/data"}}}},
            "test_dataset",
            {
                "load_config_calls": 0,
                "discover_dataset_calls": 1
            }
        ),
    ])
    def test_load_dataset_files_basic_scenarios(
        self, mock_config_and_discovery, config_source, dataset_name, expected_calls
    ):
        """Test basic load_dataset_files scenarios with different config sources."""
        mock_load_config, _, mock_discover_dataset_files = mock_config_and_discovery
        
        # Call the function under test
        result = load_dataset_files(
            dataset_name=dataset_name,
            **config_source
        )
        
        # Verify mock calls
        assert mock_load_config.call_count == expected_calls["load_config_calls"]
        assert mock_discover_dataset_files.call_count == expected_calls["discover_dataset_calls"]
        
        # Verify result
        assert result == mock_discover_dataset_files.return_value


    @pytest.mark.parametrize("custom_params,expected_discovery_args", [
        # Test with custom base_directory
        (
            {"base_directory": "/custom/data/dir"},
            {"base_directory": "/custom/data/dir"}
        ),
        # Test with custom pattern and non-recursive search
        (
            {"pattern": "*.csv", "recursive": False},
            {"pattern": "*.csv", "recursive": False}
        ),
        # Test with extensions and metadata extraction
        (
            {"extensions": [".pkl"], "extract_metadata": True},
            {"extensions": [".pkl"], "extract_metadata": True}
        ),
    ])
    def test_load_dataset_files_parameter_forwarding(
        self, mock_config_and_discovery, sample_config_dict, custom_params, expected_discovery_args
    ):
        """Test that custom parameters are properly forwarded to discovery function."""
        _, _, mock_discover_dataset_files = mock_config_and_discovery
        
        # Call function with custom parameters
        load_dataset_files(
            config=sample_config_dict,
            dataset_name="test_dataset",
            **custom_params
        )
        
        # Verify discovery function was called with expected parameters
        call_args = mock_discover_dataset_files.call_args
        for param_name, expected_value in expected_discovery_args.items():
            assert call_args.kwargs[param_name] == expected_value


    def test_load_dataset_files_missing_major_data_directory(self, sample_config_dict):
        """Test ValueError when major_data_directory is missing for datasets."""
        config = sample_config_dict.copy()
        config["project"]["directories"].pop("major_data_directory")
        
        with pytest.raises(ValueError, match=re.escape(MISSING_DATA_DIR_ERROR)):
            load_dataset_files(config=config, dataset_name="test_dataset")


    @pytest.mark.parametrize("invalid_config_params,expected_error", [
        # Test with neither config_path nor config provided
        ({}, "Exactly one of 'config_path' or 'config' must be provided"),
        # Test with both config_path and config provided
        (
            {"config_path": "/path/config.yaml", "config": {}},
            "Exactly one of 'config_path' or 'config' must be provided"
        ),
    ])
    def test_load_dataset_files_config_validation_errors(
        self, invalid_config_params, expected_error
    ):
        """Test that proper ValueError is raised for invalid config parameters."""
        with pytest.raises(ValueError, match=re.escape(expected_error)):
            load_dataset_files(
                dataset_name="test_dataset",
                **invalid_config_params
            )


# =============================================================================
# Test Cases for get_dataset_parameters
# =============================================================================

class TestGetDatasetParameters:
    """Test suite for get_dataset_parameters function."""

    @pytest.mark.parametrize("config_data,dataset_name,expected_result", [
        # Test with dataset that has parameters
        (
            {
                "datasets": {
                    "my_dataset": {
                        "parameters": {"alpha": 1, "beta": "b"}
                    }
                }
            },
            "my_dataset",
            {"alpha": 1, "beta": "b"}
        ),
        # Test with dataset that has no parameters
        (
            {
                "datasets": {
                    "my_dataset": {"rig": "rig1"}
                }
            },
            "my_dataset",
            {}
        ),
        # Test with dataset that has empty parameters
        (
            {
                "datasets": {
                    "my_dataset": {
                        "parameters": {}
                    }
                }
            },
            "my_dataset",
            {}
        ),
    ])
    def test_get_dataset_parameters_valid_scenarios(
        self, mocker, config_data, dataset_name, expected_result
    ):
        """Test get_dataset_parameters with valid scenarios."""
        # Mock get_dataset_info to return expected dataset info
        mock_get_dataset_info = mocker.patch("flyrigloader.api.get_dataset_info")
        mock_get_dataset_info.return_value = config_data["datasets"][dataset_name]
        
        # Test with config dictionary
        result = get_dataset_parameters(config=config_data, dataset_name=dataset_name)
        assert result == expected_result
        
        # Verify get_dataset_info was called correctly
        mock_get_dataset_info.assert_called_once_with(config_data, dataset_name)


    def test_get_dataset_parameters_nonexistent_dataset(self, mocker):
        """Test KeyError when dataset is not present in config."""
        # Mock get_dataset_info to raise KeyError
        mock_get_dataset_info = mocker.patch("flyrigloader.api.get_dataset_info")
        mock_get_dataset_info.side_effect = KeyError("Dataset 'missing' not found")
        
        config = {"datasets": {"other_dataset": {}}}
        with pytest.raises(KeyError):
            get_dataset_parameters(config=config, dataset_name="missing")


    @pytest.mark.parametrize("invalid_config_params,expected_error", [
        # Test with neither config_path nor config provided
        ({}, "Exactly one of 'config_path' or 'config' must be provided"),
        # Test with both config_path and config provided
        (
            {"config_path": "/path/config.yaml", "config": {}},
            "Exactly one of 'config_path' or 'config' must be provided"
        ),
    ])
    def test_get_dataset_parameters_config_validation_errors(
        self, invalid_config_params, expected_error
    ):
        """Test that proper ValueError is raised for invalid config parameters."""
        with pytest.raises(ValueError, match=re.escape(expected_error)):
            get_dataset_parameters(
                dataset_name="test_dataset",
                **invalid_config_params
            )


    def test_get_dataset_parameters_with_config_path(self, mocker):
        """Test get_dataset_parameters with config_path parameter."""
        # Mock load_config and get_dataset_info
        mock_load_config = mocker.patch("flyrigloader.api.load_config")
        mock_get_dataset_info = mocker.patch("flyrigloader.api.get_dataset_info")
        
        mock_config = {"datasets": {"test_dataset": {"parameters": {"key": "value"}}}}
        mock_load_config.return_value = mock_config
        mock_get_dataset_info.return_value = {"parameters": {"key": "value"}}
        
        # Call function with config_path
        result = get_dataset_parameters(
            config_path="/path/to/config.yaml",
            dataset_name="test_dataset"
        )
        
        # Verify mocks were called correctly
        mock_load_config.assert_called_once_with("/path/to/config.yaml")
        mock_get_dataset_info.assert_called_once_with(mock_config, "test_dataset")
        assert result == {"key": "value"}


# =============================================================================
# Test Cases for get_experiment_parameters
# =============================================================================

class TestGetExperimentParameters:
    """Test suite for get_experiment_parameters function."""

    @pytest.mark.parametrize("config_data,experiment_name,expected_result", [
        # Test with experiment that has parameters
        (
            {
                "experiments": {
                    "my_experiment": {
                        "parameters": {"learning_rate": 0.01, "epochs": 100}
                    }
                }
            },
            "my_experiment",
            {"learning_rate": 0.01, "epochs": 100}
        ),
        # Test with experiment that has no parameters
        (
            {
                "experiments": {
                    "my_experiment": {"datasets": ["dataset1"]}
                }
            },
            "my_experiment",
            {}
        ),
    ])
    def test_get_experiment_parameters_valid_scenarios(
        self, mocker, config_data, experiment_name, expected_result
    ):
        """Test get_experiment_parameters with valid scenarios."""
        # Mock get_experiment_info to return expected experiment info
        mock_get_experiment_info = mocker.patch("flyrigloader.api.get_experiment_info")
        mock_get_experiment_info.return_value = config_data["experiments"][experiment_name]
        
        # Test with config dictionary
        result = get_experiment_parameters(config=config_data, experiment_name=experiment_name)
        assert result == expected_result
        
        # Verify get_experiment_info was called correctly
        mock_get_experiment_info.assert_called_once_with(config_data, experiment_name)


    def test_get_experiment_parameters_with_config_path(self, mocker):
        """Test get_experiment_parameters with config_path parameter."""
        # Mock load_config and get_experiment_info
        mock_load_config = mocker.patch("flyrigloader.api.load_config")
        mock_get_experiment_info = mocker.patch("flyrigloader.api.get_experiment_info")
        
        mock_config = {"experiments": {"test_exp": {"parameters": {"param1": "value1"}}}}
        mock_load_config.return_value = mock_config
        mock_get_experiment_info.return_value = {"parameters": {"param1": "value1"}}
        
        # Call function with config_path
        result = get_experiment_parameters(
            config_path="/path/to/config.yaml",
            experiment_name="test_exp"
        )
        
        # Verify mocks were called correctly
        mock_load_config.assert_called_once_with("/path/to/config.yaml")
        mock_get_experiment_info.assert_called_once_with(mock_config, "test_exp")
        assert result == {"param1": "value1"}


# =============================================================================
# Integration Test Cases
# =============================================================================

class TestApiIntegration:
    """Integration tests for complete API workflows."""

    def test_complete_experiment_workflow_with_file_config(
        self, sample_config_file, mocker
    ):
        """Test complete workflow from file config to experiment files discovery."""
        # Mock the discovery function to return realistic data
        mock_discover = mocker.patch("flyrigloader.api.discover_experiment_files")
        mock_discover.return_value = {
            "/path/to/data/experiment_20240101_test_1.pkl": {
                "date": "20240101",
                "experiment": "test",
                "replicate": "1"
            },
            "/path/to/data/experiment_20240102_test_2.pkl": {
                "date": "20240102",
                "experiment": "test",
                "replicate": "2"
            }
        }
        
        # Execute complete workflow
        result = load_experiment_files(
            config_path=sample_config_file,
            experiment_name="test_experiment",
            extract_metadata=True
        )
        
        # Verify workflow completed successfully
        assert isinstance(result, dict)
        assert len(result) == 2
        assert all("date" in metadata for metadata in result.values())
        
        # Verify discovery was called with correct parameters
        mock_discover.assert_called_once()
        call_args = mock_discover.call_args.kwargs
        assert call_args["experiment_name"] == "test_experiment"
        assert call_args["extract_metadata"] is True


    def test_complete_dataset_workflow_with_dict_config(
        self, sample_config_dict, mocker
    ):
        """Test complete workflow from dict config to dataset files discovery."""
        # Mock the discovery function
        mock_discover = mocker.patch("flyrigloader.api.discover_dataset_files")
        mock_discover.return_value = [
            "/path/to/data/dataset_file1.csv",
            "/path/to/data/dataset_file2.csv"
        ]
        
        # Execute complete workflow
        result = load_dataset_files(
            config=sample_config_dict,
            dataset_name="test_dataset",
            pattern="*.csv",
            recursive=True
        )
        
        # Verify workflow completed successfully
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(path.endswith(".csv") for path in result)
        
        # Verify discovery was called with correct parameters
        mock_discover.assert_called_once()
        call_args = mock_discover.call_args.kwargs
        assert call_args["dataset_name"] == "test_dataset"
        assert call_args["pattern"] == "*.csv"
        assert call_args["recursive"] is True


    def test_parameter_retrieval_integration(self, sample_config_dict, mocker):
        """Test integration of parameter retrieval with realistic config."""
        # Add parameters to test config
        config_with_params = sample_config_dict.copy()
        config_with_params["datasets"]["test_dataset"]["parameters"] = {
            "sampling_rate": 60,
            "threshold": 0.5
        }
        config_with_params["experiments"]["test_experiment"]["parameters"] = {
            "trial_duration": 300,
            "conditions": ["A", "B"]
        }
        
        # Mock the info functions
        mocker.patch("flyrigloader.api.get_dataset_info").return_value = {
            "parameters": {"sampling_rate": 60, "threshold": 0.5}
        }
        mocker.patch("flyrigloader.api.get_experiment_info").return_value = {
            "parameters": {"trial_duration": 300, "conditions": ["A", "B"]}
        }
        
        # Test dataset parameter retrieval
        dataset_params = get_dataset_parameters(
            config=config_with_params,
            dataset_name="test_dataset"
        )
        assert dataset_params == {"sampling_rate": 60, "threshold": 0.5}
        
        # Test experiment parameter retrieval
        experiment_params = get_experiment_parameters(
            config=config_with_params,
            experiment_name="test_experiment"
        )
        assert experiment_params == {"trial_duration": 300, "conditions": ["A", "B"]}


    @pytest.mark.parametrize("workflow_type,api_function,name_param", [
        ("experiment", load_experiment_files, "experiment_name"),
        ("dataset", load_dataset_files, "dataset_name"),
    ])
    def test_api_error_propagation_integration(
        self, sample_config_dict, mocker, workflow_type, api_function, name_param
    ):
        """Test that errors from discovery functions are properly propagated."""
        # Mock discovery to raise an error
        if workflow_type == "experiment":
            mock_discover = mocker.patch("flyrigloader.api.discover_experiment_files")
        else:
            mock_discover = mocker.patch("flyrigloader.api.discover_dataset_files")
        
        mock_discover.side_effect = KeyError(f"{workflow_type} 'nonexistent' not found")
        
        # Verify error is propagated
        with pytest.raises(KeyError, match=f"{workflow_type} 'nonexistent' not found"):
            api_function(
                config=sample_config_dict,
                **{name_param: "nonexistent"}
            )


# =============================================================================
# Edge Case and Error Handling Tests
# =============================================================================

class TestApiEdgeCases:
    """Test edge cases and error scenarios for API functions."""

    @pytest.mark.parametrize("config_mutation", [
        # Missing project section
        lambda config: config.pop("project"),
        # Missing directories section
        lambda config: config["project"].pop("directories"),
        # Empty directories section
        lambda config: config["project"]["directories"].clear(),
    ])
    def test_missing_data_directory_scenarios(self, sample_config_dict, config_mutation):
        """Test various scenarios where data directory is missing."""
        config = sample_config_dict.copy()
        config_mutation(config)
        
        with pytest.raises(ValueError, match=re.escape(MISSING_DATA_DIR_ERROR)):
            load_experiment_files(config=config, experiment_name="test_experiment")
            
        with pytest.raises(ValueError, match=re.escape(MISSING_DATA_DIR_ERROR)):
            load_dataset_files(config=config, dataset_name="test_dataset")


    def test_deep_copy_behavior(self, sample_config_dict, mocker):
        """Test that config dictionaries are deep copied to prevent mutation."""
        original_config = sample_config_dict.copy()
        
        # Mock discovery to modify the config (this should not affect original)
        def modify_config(*args, **kwargs):
            config = args[0] if args else kwargs["config"]
            config["project"]["directories"]["major_data_directory"] = "/modified/path"
            return []
        
        mock_discover = mocker.patch("flyrigloader.api.discover_experiment_files")
        mock_discover.side_effect = modify_config
        
        # Call API function
        load_experiment_files(
            config=sample_config_dict,
            experiment_name="test_experiment"
        )
        
        # Verify original config was not modified
        assert sample_config_dict == original_config


    @given(st.text())
    def test_experiment_name_property_based(self, mocker, experiment_name):
        """Property-based test for experiment name handling."""
        assume(experiment_name.strip() != "")  # Assume non-empty experiment names
        
        # Mock dependencies
        mock_discover = mocker.patch("flyrigloader.api.discover_experiment_files")
        mock_discover.return_value = []
        
        config = {
            "project": {"directories": {"major_data_directory": "/data"}},
            "experiments": {experiment_name: {"datasets": ["test"]}}
        }
        
        # Test should not raise an exception with valid experiment name
        result = load_experiment_files(config=config, experiment_name=experiment_name)
        assert result == []
        
        # Verify experiment name was passed correctly
        call_args = mock_discover.call_args.kwargs
        assert call_args["experiment_name"] == experiment_name


    def test_config_validation_with_none_values(self):
        """Test behavior when config contains None values."""
        config_with_nones = {
            "project": {
                "directories": {
                    "major_data_directory": None
                }
            }
        }
        
        with pytest.raises(ValueError, match=re.escape(MISSING_DATA_DIR_ERROR)):
            load_experiment_files(config=config_with_nones, experiment_name="test")


    def test_concurrent_api_calls(self, sample_config_dict, mocker):
        """Test that API functions are safe for concurrent usage."""
        import threading
        import queue
        
        # Mock discovery functions
        mock_exp_discover = mocker.patch("flyrigloader.api.discover_experiment_files")
        mock_ds_discover = mocker.patch("flyrigloader.api.discover_dataset_files")
        
        mock_exp_discover.return_value = ["exp_file.pkl"]
        mock_ds_discover.return_value = ["ds_file.csv"]
        
        results = queue.Queue()
        
        def worker():
            """Worker function for concurrent testing."""
            try:
                exp_result = load_experiment_files(
                    config=sample_config_dict,
                    experiment_name="test_experiment"
                )
                ds_result = load_dataset_files(
                    config=sample_config_dict,
                    dataset_name="test_dataset"
                )
                results.put((exp_result, ds_result))
            except Exception as e:
                results.put(e)
        
        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        
        # Wait for completion and collect results
        for thread in threads:
            thread.join()
        
        # Verify all threads completed successfully
        thread_results = []
        while not results.empty():
            result = results.get()
            assert not isinstance(result, Exception), f"Thread failed with: {result}"
            thread_results.append(result)
        
        assert len(thread_results) == 3
        # All results should be identical
        expected_result = (["exp_file.pkl"], ["ds_file.csv"])
        assert all(result == expected_result for result in thread_results)