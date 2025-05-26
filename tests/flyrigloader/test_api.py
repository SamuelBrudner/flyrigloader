"""
Tests for API functions in the flyrigloader.api module.
"""

import os
from unittest.mock import patch, MagicMock, call

import pytest

from flyrigloader.api import (
    load_experiment_files,
    load_dataset_files,
    get_dataset_parameters
)


def test_load_experiment_files_with_config_path(mock_config_and_discovery):
    """Test loading experiment files with a config path."""
    mock_load_config, mock_discover_experiment_files, _ = mock_config_and_discovery
    
    # Call the function under test
    result = load_experiment_files(
        config_path="/path/to/config.yaml",
        experiment_name="test_experiment"
    )
    
    # Verify mock calls
    mock_load_config.assert_called_once_with("/path/to/config.yaml")
    mock_discover_experiment_files.assert_called_once_with(
        config=mock_load_config.return_value,
        experiment_name="test_experiment",
        base_directory="/path/to/data",
        pattern="*.*",
        recursive=True,
        extensions=None,
        extract_metadata=False,
        parse_dates=False
    )
    
    # Verify result is the list of files returned by discover_experiment_files
    assert result == mock_discover_experiment_files.return_value


def test_load_experiment_files_with_config_dict(mock_config_and_discovery):
    """Test loading experiment files with a config dictionary."""
    _, mock_discover_experiment_files, _ = mock_config_and_discovery
    
    # Sample config dict
    config = {
        "project": {
            "directories": {
                "major_data_directory": "/path/to/data"
            }
        },
        "experiments": {
            "test_experiment": {
                "datasets": ["test_dataset"]
            }
        }
    }
    
    # Call the function under test
    result = load_experiment_files(
        config=config,
        experiment_name="test_experiment"
    )
    
    # Verify mock calls (load_config should not be called)
    mock_discover_experiment_files.assert_called_once_with(
        config=config,
        experiment_name="test_experiment",
        base_directory="/path/to/data",
        pattern="*.*",
        recursive=True,
        extensions=None,
        extract_metadata=False,
        parse_dates=False
    )
    
    # Verify result is the list of files returned by discover_experiment_files
    assert result == mock_discover_experiment_files.return_value


def test_load_experiment_files_with_custom_params(mock_config_and_discovery):
    """Test loading experiment files with custom parameters."""
    _, mock_discover_experiment_files, _ = mock_config_and_discovery
    
    # Sample config dict
    config = {
        "project": {
            "directories": {
                "major_data_directory": "/path/to/data"
            }
        },
        "experiments": {
            "test_experiment": {
                "datasets": ["test_dataset"]
            }
        }
    }
    
    # Call the function under test with custom params
    result = load_experiment_files(
        config=config,
        experiment_name="test_experiment",
        base_directory="/custom/data/dir",
        pattern="*.csv",
        recursive=False,
        extensions=[".csv"]
    )
    
    # Verify mock calls with custom params
    mock_discover_experiment_files.assert_called_once_with(
        config=config,
        experiment_name="test_experiment",
        base_directory="/custom/data/dir",
        pattern="*.csv",
        recursive=False,
        extensions=[".csv"],
        extract_metadata=False,
        parse_dates=False
    )
    
    # Verify result
    assert result == mock_discover_experiment_files.return_value


def test_load_experiment_files_with_metadata_extraction(mock_config_and_discovery):
    """Test loading experiment files with metadata extraction."""
    _, mock_discover_experiment_files, _ = mock_config_and_discovery
    
    # Call the function with metadata extraction enabled
    result = load_experiment_files(
        config_path="/path/to/config.yaml",
        experiment_name="test_experiment",
        extract_metadata=True
    )
    
    # Verify extract_metadata parameter is passed correctly
    mock_discover_experiment_files.assert_called_once_with(
        config=mock_config_and_discovery[0].return_value,
        experiment_name="test_experiment",
        base_directory="/path/to/data",
        pattern="*.*",
        recursive=True,
        extensions=None,
        extract_metadata=True,
        parse_dates=False
    )
    
    # Verify result
    assert result == mock_discover_experiment_files.return_value


def test_load_experiment_files_with_date_parsing(mock_config_and_discovery):
    """Test loading experiment files with date parsing."""
    _, mock_discover_experiment_files, _ = mock_config_and_discovery
    
    # Call the function with date parsing enabled
    result = load_experiment_files(
        config_path="/path/to/config.yaml",
        experiment_name="test_experiment",
        extract_metadata=True,
        parse_dates=True
    )
    
    # Verify both extract_metadata and parse_dates parameters are passed correctly
    mock_discover_experiment_files.assert_called_once_with(
        config=mock_config_and_discovery[0].return_value,
        experiment_name="test_experiment",
        base_directory="/path/to/data",
        pattern="*.*",
        recursive=True,
        extensions=None,
        extract_metadata=True,
        parse_dates=True
    )
    
    # Verify result
    assert result == mock_discover_experiment_files.return_value


def test_load_dataset_files_with_config_path(mock_config_and_discovery):
    """Test loading dataset files with a config path."""
    mock_load_config, _, mock_discover_dataset_files = mock_config_and_discovery
    
    # Call the function under test
    result = load_dataset_files(
        config_path="/path/to/config.yaml",
        dataset_name="test_dataset"
    )
    
    # Verify mock calls
    mock_load_config.assert_called_once_with("/path/to/config.yaml")
    mock_discover_dataset_files.assert_called_once_with(
        config=mock_load_config.return_value,
        dataset_name="test_dataset",
        base_directory="/path/to/data",
        pattern="*.*",
        recursive=True,
        extensions=None,
        extract_metadata=False,
        parse_dates=False
    )
    
    # Verify result is the list of files returned by discover_dataset_files
    assert result == mock_discover_dataset_files.return_value


def test_load_dataset_files_with_config_dict(mock_config_and_discovery):
    """Test loading dataset files with a config dictionary."""
    _, _, mock_discover_dataset_files = mock_config_and_discovery
    
    # Sample config dict
    config = {
        "project": {
            "directories": {
                "major_data_directory": "/path/to/data"
            }
        },
        "datasets": {
            "test_dataset": {
                "patterns": ["*_test_*"]
            }
        }
    }
    
    # Call the function under test
    result = load_dataset_files(
        config=config,
        dataset_name="test_dataset"
    )
    
    # Verify mock calls (load_config should not be called)
    mock_discover_dataset_files.assert_called_once_with(
        config=config,
        dataset_name="test_dataset",
        base_directory="/path/to/data",
        pattern="*.*",
        recursive=True,
        extensions=None,
        extract_metadata=False,
        parse_dates=False
    )
    
    # Verify result is the list of files returned by discover_dataset_files
    assert result == mock_discover_dataset_files.return_value


def test_load_dataset_files_with_custom_params(mock_config_and_discovery):
    """Test loading dataset files with custom parameters."""
    _, _, mock_discover_dataset_files = mock_config_and_discovery
    
    # Sample config dict
    config = {
        "project": {
            "directories": {
                "major_data_directory": "/path/to/data"
            }
        },
        "datasets": {
            "test_dataset": {
                "patterns": ["*_test_*"]
            }
        }
    }
    
    # Call the function under test with custom params
    result = load_dataset_files(
        config=config,
        dataset_name="test_dataset",
        base_directory="/custom/data/dir",
        pattern="*.csv",
        recursive=False,
        extensions=[".csv"]
    )
    
    # Verify mock calls with custom params
    mock_discover_dataset_files.assert_called_once_with(
        config=config,
        dataset_name="test_dataset",
        base_directory="/custom/data/dir",
        pattern="*.csv",
        recursive=False,
        extensions=[".csv"],
        extract_metadata=False,
        parse_dates=False
    )
    
    # Verify result
    assert result == mock_discover_dataset_files.return_value


def test_load_dataset_files_with_metadata_extraction(mock_config_and_discovery):
    """Test loading dataset files with metadata extraction."""
    _, _, mock_discover_dataset_files = mock_config_and_discovery
    
    # Call the function with metadata extraction enabled
    result = load_dataset_files(
        config_path="/path/to/config.yaml",
        dataset_name="test_dataset",
        extract_metadata=True
    )
    
    # Verify extract_metadata parameter is passed correctly
    mock_discover_dataset_files.assert_called_once_with(
        config=mock_config_and_discovery[0].return_value,
        dataset_name="test_dataset",
        base_directory="/path/to/data",
        pattern="*.*",
        recursive=True,
        extensions=None,
        extract_metadata=True,
        parse_dates=False
    )
    
    # Verify result
    assert result == mock_discover_dataset_files.return_value


def test_load_dataset_files_with_date_parsing(mock_config_and_discovery):
    """Test loading dataset files with date parsing."""
    _, _, mock_discover_dataset_files = mock_config_and_discovery
    
    # Call the function with date parsing enabled
    result = load_dataset_files(
        config_path="/path/to/config.yaml",
        dataset_name="test_dataset",
        extract_metadata=True,
        parse_dates=True
    )
    
    # Verify both extract_metadata and parse_dates parameters are passed correctly
    mock_discover_dataset_files.assert_called_once_with(
        config=mock_config_and_discovery[0].return_value,
        dataset_name="test_dataset",
        base_directory="/path/to/data",
        pattern="*.*",
        recursive=True,
        extensions=None,
        extract_metadata=True,
        parse_dates=True
    )
    
    # Verify result
    assert result == mock_discover_dataset_files.return_value

def test_get_dataset_parameters_with_defined_params():
    """Return dataset-specific parameters when defined."""
    config = {
        "datasets": {
            "my_dataset": {
                "parameters": {"alpha": 1, "beta": "b"}
            }
        }
    }
    result = api.get_dataset_parameters(config=config, dataset_name="my_dataset")
    assert result == {"alpha": 1, "beta": "b"}


def test_get_dataset_parameters_without_params():
    """Return empty dict when dataset has no parameters."""
    config = {"datasets": {"my_dataset": {"rig": "rig1"}}}
    result = api.get_dataset_parameters(config=config, dataset_name="my_dataset")
    assert result == {}


def test_get_dataset_parameters_nonexistent_dataset():
    """Raise KeyError when dataset is not present in config."""
    config = {"datasets": {"other_dataset": {}}}
    with pytest.raises(KeyError):
        api.get_dataset_parameters(config=config, dataset_name="missing")

