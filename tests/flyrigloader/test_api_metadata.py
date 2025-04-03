"""
Tests for metadata extraction in the high-level API functions.
"""
import os
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from flyrigloader.api import (
    load_experiment_files,
    load_dataset_files
)


@patch('flyrigloader.api.discover_experiment_files')
@patch('flyrigloader.api.load_config')
def test_metadata_extraction_with_config_path(mock_load_config, mock_discover_experiment_files):
    """Test metadata extraction using a path to a config file."""
    # Set up mocks
    mock_config = {
        "project": {
            "directories": {
                "major_data_directory": "/path/to/data"
            },
            "extraction_patterns": [
                r".*_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.pkl"
            ]
        },
        "experiments": {
            "test_experiment": {
                "datasets": ["test_dataset"],
                "metadata": {
                    "extraction_patterns": [
                        r".*_(?P<experiment>\w+)_(?P<date>\d{8})\.pkl"
                    ]
                }
            }
        },
        "datasets": {
            "test_dataset": {
                "patterns": ["*_test_*"]
            }
        }
    }
    mock_load_config.return_value = mock_config
    
    # Mock experiment file discovery with metadata
    mock_discover_experiment_files.return_value = {
        "/path/to/data/file_20230101_test_1.pkl": {
            "date": "20230101",
            "condition": "test",
            "replicate": "1"
        },
        "/path/to/data/file_20230102_control_2.pkl": {
            "date": "20230102",
            "condition": "control",
            "replicate": "2"
        }
    }
    
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as config_file:
        config_file.write(b"""
project:
  name: test_project
  directories:
    major_data_directory: /path/to/data
  extraction_patterns:
    - .*_(?P<date>\\d{8})_(?P<condition>\\w+)_(?P<replicate>\\d+)\\.pkl

experiments:
  test_experiment:
    datasets: ["test_dataset"]
    metadata:
      extraction_patterns:
        - .*_(?P<experiment>\\w+)_(?P<date>\\d{8})\\.pkl

datasets:
  test_dataset:
    patterns:
      - "*_test_*"
""")
        config_path = config_file.name
    
    try:
        # Call the function with config_path
        result = load_experiment_files(
            config_path=config_path,
            experiment_name="test_experiment",
            extract_metadata=True
        )
        
        # Verify mock calls
        mock_load_config.assert_called_once_with(config_path)
        mock_discover_experiment_files.assert_called_once_with(
            config=mock_config,
            experiment_name="test_experiment",
            base_directory="/path/to/data",
            pattern="*.*",
            recursive=True,
            extensions=None,
            extract_metadata=True,
            parse_dates=False
        )
        
        # Verify results structure
        assert isinstance(result, dict)
        assert "/path/to/data/file_20230101_test_1.pkl" in result
        assert result["/path/to/data/file_20230101_test_1.pkl"]["date"] == "20230101"
        assert result["/path/to/data/file_20230101_test_1.pkl"]["condition"] == "test"
    
    finally:
        # Clean up
        os.unlink(config_path)


@patch('flyrigloader.api.discover_dataset_files')
def test_metadata_extraction_with_preloaded_config(mock_discover_dataset_files):
    """Test metadata extraction using a pre-loaded config dictionary."""
    # Mock dataset file discovery with metadata
    mock_discover_dataset_files.return_value = {
        "/path/to/data/file_20230101_test_1.pkl": {
            "date": "20230101",
            "condition": "test",
            "replicate": "1"
        },
        "/path/to/data/file_20230102_control_2.pkl": {
            "date": "20230102",
            "condition": "control",
            "replicate": "2"
        }
    }
    
    # Pre-loaded config
    config = {
        "project": {
            "directories": {
                "major_data_directory": "/path/to/data"
            },
            "extraction_patterns": [
                r".*_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.pkl"
            ]
        },
        "experiments": {
            "test_experiment": {
                "datasets": ["test_dataset"]
            }
        },
        "datasets": {
            "test_dataset": {
                "patterns": ["*_test_*"],
                "metadata": {
                    "extraction_patterns": [
                        r".*_(?P<dataset>\w+)_(?P<date>\d{8})\.pkl"
                    ]
                }
            }
        }
    }
    
    # Call the function with pre-loaded config
    result = load_dataset_files(
        config=config,
        dataset_name="test_dataset",
        extract_metadata=True
    )
    
    # Verify mock calls
    mock_discover_dataset_files.assert_called_once_with(
        config=config,
        dataset_name="test_dataset",
        base_directory="/path/to/data",
        pattern="*.*",
        recursive=True,
        extensions=None,
        extract_metadata=True,
        parse_dates=False
    )
    
    # Verify results structure
    assert isinstance(result, dict)
    assert "/path/to/data/file_20230101_test_1.pkl" in result
    assert result["/path/to/data/file_20230101_test_1.pkl"]["date"] == "20230101"
    assert result["/path/to/data/file_20230101_test_1.pkl"]["condition"] == "test"


@patch('flyrigloader.api.discover_experiment_files')
def test_date_parsing_functionality(mock_discover_experiment_files):
    """Test date parsing functionality."""
    # Mock file discovery with parsed dates
    mock_discover_experiment_files.return_value = {
        "/path/to/data/file_20230101_test_1.pkl": {
            "date": "20230101",
            "condition": "test",
            "replicate": "1",
            "parsed_date": "2023-01-01"  # This would be a datetime object in reality
        },
        "/path/to/data/file_20230102_control_2.pkl": {
            "date": "20230102",
            "condition": "control",
            "replicate": "2",
            "parsed_date": "2023-01-02"  # This would be a datetime object in reality
        }
    }
    
    # Pre-loaded config
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
        },
        "datasets": {
            "test_dataset": {
                "patterns": ["*_test_*"]
            }
        }
    }
    
    # Call the function with parse_dates=True
    result = load_experiment_files(
        config=config,
        experiment_name="test_experiment",
        parse_dates=True
    )
    
    # Verify that proper arguments were passed
    mock_discover_experiment_files.assert_called_once_with(
        config=config,
        experiment_name="test_experiment",
        base_directory="/path/to/data",
        pattern="*.*",
        recursive=True,
        extensions=None,
        extract_metadata=False,
        parse_dates=True
    )
    
    # Verify results structure
    assert isinstance(result, dict)
    assert "/path/to/data/file_20230101_test_1.pkl" in result
    assert "parsed_date" in result["/path/to/data/file_20230101_test_1.pkl"]
    assert result["/path/to/data/file_20230101_test_1.pkl"]["parsed_date"] == "2023-01-01"


def test_config_validation_with_metadata():
    """Test that API functions validate config parameters with metadata extraction."""
    # Test with neither config_path nor config provided
    with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
        load_experiment_files(
            experiment_name="test_experiment",
            extract_metadata=True
        )
    
    # Test with both config_path and config provided
    with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
        load_dataset_files(
            config_path="config.yaml",
            config={"test": "config"},
            dataset_name="test_dataset",
            extract_metadata=True,
            parse_dates=True
        )
