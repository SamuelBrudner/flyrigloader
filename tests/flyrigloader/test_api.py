"""
Tests for the high-level API functions.
"""
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from flyrigloader.api import (
    load_experiment_files,
    load_dataset_files, 
    get_experiment_parameters,
    process_experiment_data,
    get_default_column_config
)
from flyrigloader.io.column_models import ColumnConfigDict


@patch('flyrigloader.api.load_config')
@patch('flyrigloader.api.discover_experiment_files')
def test_load_experiment_files(mock_discover_files, mock_load_config):
    """Test that load_experiment_files correctly passes arguments to the discovery function."""
    # Set up mocks
    mock_config = {
        "project": {
            "directories": {
                "major_data_directory": "/path/to/data"
            }
        }
    }
    mock_load_config.return_value = mock_config
    mock_discover_files.return_value = ["file1.txt", "file2.txt"]
    
    # Call the function with config_path
    result = load_experiment_files(
        config_path="config.yaml",
        experiment_name="test_experiment",
        file_pattern="*.csv",
        recursive=True,
        extensions=["csv"]
    )
    
    # Verify correct calls were made
    mock_load_config.assert_called_once_with("config.yaml")
    mock_discover_files.assert_called_once_with(
        config=mock_config,
        experiment_name="test_experiment",
        base_directory="/path/to/data",
        pattern="*.csv",
        recursive=True,
        extensions=["csv"]
    )
    
    # Verify result
    assert result == ["file1.txt", "file2.txt"]


@patch('flyrigloader.api.discover_experiment_files')
def test_load_experiment_files_with_config(mock_discover_files):
    """Test that load_experiment_files works with a pre-loaded config dictionary."""
    # Set up mocks
    mock_config = {
        "project": {
            "directories": {
                "major_data_directory": "/path/to/data"
            }
        }
    }
    mock_discover_files.return_value = ["file1.txt", "file2.txt"]
    
    # Call the function with pre-loaded config
    result = load_experiment_files(
        config=mock_config,
        experiment_name="test_experiment",
        file_pattern="*.csv",
        recursive=True,
        extensions=["csv"]
    )
    
    # Verify correct calls were made (no load_config call)
    mock_discover_files.assert_called_once_with(
        config=mock_config,
        experiment_name="test_experiment",
        base_directory="/path/to/data",
        pattern="*.csv",
        recursive=True,
        extensions=["csv"]
    )
    
    # Verify result
    assert result == ["file1.txt", "file2.txt"]


def test_load_experiment_files_validation():
    """Test that load_experiment_files validates config parameters correctly."""
    # Test with neither config_path nor config provided
    with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
        load_experiment_files(experiment_name="test_experiment")
    
    # Test with both config_path and config provided
    with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
        load_experiment_files(
            config_path="config.yaml",
            config={"test": "config"},
            experiment_name="test_experiment"
        )


@patch('flyrigloader.api.load_config')
@patch('flyrigloader.api.discover_dataset_files')
def test_load_dataset_files(mock_discover_files, mock_load_config):
    """Test that load_dataset_files correctly passes arguments to the discovery function."""
    # Set up mocks
    mock_config = {
        "project": {
            "directories": {
                "major_data_directory": "/path/to/data"
            }
        }
    }
    mock_load_config.return_value = mock_config
    mock_discover_files.return_value = ["file1.txt", "file2.txt"]
    
    # Call the function with config_path
    result = load_dataset_files(
        config_path="config.yaml",
        dataset_name="test_dataset",
        file_pattern="*.csv",
        recursive=True,
        extensions=["csv"]
    )
    
    # Verify correct calls were made
    mock_load_config.assert_called_once_with("config.yaml")
    mock_discover_files.assert_called_once_with(
        config=mock_config,
        dataset_name="test_dataset",
        base_directory="/path/to/data",
        pattern="*.csv",
        recursive=True,
        extensions=["csv"]
    )
    
    # Verify result
    assert result == ["file1.txt", "file2.txt"]


@patch('flyrigloader.api.discover_dataset_files')
def test_load_dataset_files_with_config(mock_discover_files):
    """Test that load_dataset_files works with a pre-loaded config dictionary."""
    # Set up mocks
    mock_config = {
        "project": {
            "directories": {
                "major_data_directory": "/path/to/data"
            }
        }
    }
    mock_discover_files.return_value = ["file1.txt", "file2.txt"]
    
    # Call the function with pre-loaded config
    result = load_dataset_files(
        config=mock_config,
        dataset_name="test_dataset",
        file_pattern="*.csv",
        recursive=True,
        extensions=["csv"]
    )
    
    # Verify correct calls were made (no load_config call)
    mock_discover_files.assert_called_once_with(
        config=mock_config,
        dataset_name="test_dataset",
        base_directory="/path/to/data",
        pattern="*.csv",
        recursive=True,
        extensions=["csv"]
    )
    
    # Verify result
    assert result == ["file1.txt", "file2.txt"]


def test_load_dataset_files_validation():
    """Test that load_dataset_files validates config parameters correctly."""
    # Test with neither config_path nor config provided
    with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
        load_dataset_files(dataset_name="test_dataset")
    
    # Test with both config_path and config provided
    with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
        load_dataset_files(
            config_path="config.yaml",
            config={"test": "config"},
            dataset_name="test_dataset"
        )


@patch('flyrigloader.api.load_config')
@patch('flyrigloader.api.get_experiment_info')
def test_get_experiment_parameters(mock_get_experiment_info, mock_load_config):
    """Test that get_experiment_parameters correctly extracts parameters."""
    # Set up mocks
    mock_load_config.return_value = {"experiments": {"test_experiment": {}}}
    mock_get_experiment_info.return_value = {
        "parameters": {
            "param1": "value1",
            "param2": 42
        }
    }
    
    # Call the function with config_path
    result = get_experiment_parameters(
        config_path="config.yaml",
        experiment_name="test_experiment"
    )
    
    # Verify correct calls were made
    mock_load_config.assert_called_once_with("config.yaml")
    mock_get_experiment_info.assert_called_once_with(
        mock_load_config.return_value, 
        "test_experiment"
    )
    
    # Verify result
    assert result == {"param1": "value1", "param2": 42}


@patch('flyrigloader.api.get_experiment_info')
def test_get_experiment_parameters_with_config(mock_get_experiment_info):
    """Test that get_experiment_parameters works with a pre-loaded config dictionary."""
    # Set up mocks
    mock_config = {"experiments": {"test_experiment": {}}}
    mock_get_experiment_info.return_value = {
        "parameters": {
            "param1": "value1",
            "param2": 42
        }
    }
    
    # Call the function with pre-loaded config
    result = get_experiment_parameters(
        config=mock_config,
        experiment_name="test_experiment"
    )
    
    # Verify correct calls were made
    mock_get_experiment_info.assert_called_once_with(
        mock_config, 
        "test_experiment"
    )
    
    # Verify result
    assert result == {"param1": "value1", "param2": 42}


def test_get_experiment_parameters_validation():
    """Test that get_experiment_parameters validates config parameters correctly."""
    # Test with neither config_path nor config provided
    with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
        get_experiment_parameters(experiment_name="test_experiment")
    
    # Test with both config_path and config provided
    with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
        get_experiment_parameters(
            config_path="config.yaml",
            config={"test": "config"},
            experiment_name="test_experiment"
        )


def test_process_experiment_data():
    """Test that process_experiment_data correctly processes data."""
    # Create temporary experiment data
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_file:
        temp_path = temp_file.name
        
    try:
        # Create test data
        exp_matrix = create_test_exp_matrix()
        
        # Mock the read_pickle_any_format function
        with patch('flyrigloader.api.read_pickle_any_format') as mock_read_pickle:
            mock_read_pickle.return_value = exp_matrix
            
            _check_default_config(temp_path, exp_matrix, mock_read_pickle)
            _check_custom_config(temp_path, exp_matrix, mock_read_pickle)
                
    finally:
        # Clean up
        os.unlink(temp_path)


def create_test_exp_matrix():
    """Create test experimental data matrix with required columns."""
    return {
        "t": np.array([0, 1, 2, 3, 4]),
        "x": np.array([10, 11, 12, 13, 14]),
        "y": np.array([20, 21, 22, 23, 24]),
        "trjn": np.array([1, 1, 1, 1, 1]),
        "theta": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        "theta_smooth": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        "dtheta": np.array([0.01, 0.01, 0.01, 0.01, 0.01]),
        "vx": np.array([1.0, 1.1, 1.2, 1.3, 1.4]),
        "vy": np.array([2.0, 2.1, 2.2, 2.3, 2.4]),
        "spd": np.array([2.2, 2.4, 2.5, 2.6, 2.8]),
        "jump": np.array([0, 0, 1, 0, 0])
    }


def _check_default_config(temp_path, exp_matrix, mock_read_pickle):
    """Test process_experiment_data with default configuration."""
    with patch('flyrigloader.api.make_dataframe_from_config') as mock_make_df:
        # Set up mock return value
        expected_df = pd.DataFrame({
            "t": exp_matrix["t"],
            "x": exp_matrix["x"],
            "y": exp_matrix["y"]
        })
        mock_make_df.return_value = expected_df
        
        # Call the function
        result = process_experiment_data(
            data_path=temp_path,
            metadata={"date": "2023-01-01"}
        )
        
        # Verify correct calls were made
        mock_read_pickle.assert_called_once_with(temp_path)
        mock_make_df.assert_called_once_with(
            exp_matrix=exp_matrix,
            config_source=None,
            metadata={"date": "2023-01-01"}
        )
        
        # Verify result
        assert result is expected_df
    
    # Reset mock for next test
    mock_read_pickle.reset_mock()


def _check_custom_config(temp_path, exp_matrix, mock_read_pickle):
    """Test process_experiment_data with custom configuration."""
    with patch('flyrigloader.api.make_dataframe_from_config') as mock_make_df:
        # Set up mock return value
        expected_df = pd.DataFrame({
            "t": exp_matrix["t"],
            "x": exp_matrix["x"]
        })
        mock_make_df.return_value = expected_df
        
        # Custom config
        config = {"columns": {"t": {"type": "numpy.ndarray", "dimension": 1}}}
        
        # Call the function
        result = process_experiment_data(
            data_path=temp_path,
            column_config_path=config,
            metadata={"date": "2023-01-01"}
        )
        
        # Verify correct calls were made
        mock_make_df.assert_called_once_with(
            exp_matrix=exp_matrix,
            config_source=config,
            metadata={"date": "2023-01-01"}
        )
        
        # Verify result
        assert result is expected_df


def test_get_default_column_config():
    """Test that get_default_column_config returns a valid configuration."""
    # Call the function
    with patch('flyrigloader.api.get_config_from_source') as mock_get_config:
        # Set up mock
        mock_config = MagicMock(spec=ColumnConfigDict)
        mock_get_config.return_value = mock_config
        
        # Call the function
        result = get_default_column_config()
        
        # Verify correct calls were made
        mock_get_config.assert_called_once_with(None)
        
        # Verify result
        assert result is mock_config


def test_api_integration():
    """Test the complete API flow for processing experimental data."""
    # This test follows the complete workflow that a calling project would use
    
    # Create a temporary experiment configuration
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as config_file:
        config_file.write(b"""
project:
  name: test_project
  directories:
    major_data_directory: /tmp

datasets:
  test_dataset:
    patterns:
      - "*_test_*"

experiments:
  test_experiment:
    dataset: test_dataset
    parameters:
      param1: value1
      param2: 42
""")
        config_path = config_file.name
    
    # Create a temporary data file
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as data_file:
        data_path = data_file.name
        
    try:
        # Create test data with all required columns from default config
        exp_matrix = create_test_exp_matrix()
        # Test with config path
        _run_integration_test(config_path, data_path, exp_matrix, use_config_path=True)
        # Test with pre-loaded config
        _run_integration_test(config_path, data_path, exp_matrix, use_config_path=False)
    
    finally:
        # Clean up
        os.unlink(config_path)
        os.unlink(data_path)


def _run_integration_test(config_path, data_path, exp_matrix, use_config_path=True):
    """Run the integration test workflow."""
    # Load the config if testing with pre-loaded config
    config = None
    if not use_config_path:
        with patch('flyrigloader.api.load_config') as mock_load_config:
            mock_config = {
                "project": {
                    "directories": {
                        "major_data_directory": "/tmp"
                    }
                },
                "datasets": {
                    "test_dataset": {
                        "patterns": ["*_test_*"]
                    }
                },
                "experiments": {
                    "test_experiment": {
                        "dataset": "test_dataset",
                        "parameters": {
                            "param1": "value1",
                            "param2": 42
                        }
                    }
                }
            }
            mock_load_config.return_value = mock_config
            config = mock_config
    
    # Set up patches
    with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
        mock_discover.return_value = [data_path]
        
        with patch('flyrigloader.api.read_pickle_any_format') as mock_read_pickle:
            mock_read_pickle.return_value = exp_matrix
            
            # Run the complete workflow a calling project would use
            
            # 1. Get experiment files
            args = {"experiment_name": "test_experiment"}
            if use_config_path:
                args["config_path"] = config_path
            else:
                args["config"] = config
                
            files = load_experiment_files(**args)
            assert data_path in files
            
            # 2. Get experiment parameters
            with patch('flyrigloader.api.get_experiment_info') as mock_get_info:
                mock_get_info.return_value = {
                    "parameters": {"param1": "value1", "param2": 42}
                }
                
                params = get_experiment_parameters(**args)
                assert params["param1"] == "value1"
                assert params["param2"] == 42
            
            # 3. Process the data
            with patch('flyrigloader.api.make_dataframe_from_config') as mock_make_df:
                expected_df = pd.DataFrame({
                    "t": exp_matrix["t"],
                    "x": exp_matrix["x"],
                    "y": exp_matrix["y"]
                })
                mock_make_df.return_value = expected_df
                
                # Process with default config
                df = process_experiment_data(
                    data_path=data_path,
                    metadata={"date": "2023-01-01"}
                )
                assert df is expected_df
                
                # 4. Verify the default config is accessible
                with patch('flyrigloader.api.get_config_from_source') as mock_get_config:
                    mock_config = MagicMock(spec=ColumnConfigDict)
                    mock_get_config.return_value = mock_config
                    
                    config = get_default_column_config()
                    assert config is mock_config
