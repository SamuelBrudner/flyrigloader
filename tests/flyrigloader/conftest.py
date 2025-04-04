"""
Shared fixtures for flyrigloader tests.

This module contains pytest fixtures that are shared across multiple test files
to reduce code duplication and ensure consistency in test data.
"""
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import numpy as np
import pytest
import yaml
from unittest.mock import MagicMock


# --- Configuration Fixtures ---

@pytest.fixture
def sample_config_file():
    """
    Create a temporary config file with sample configuration.
    
    Returns:
        str: Path to the temporary config file
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a sample configuration file
        config_path = os.path.join(temp_dir, "config.yaml")
        
        # Sample config structure that mimics base_config.yaml with metadata extraction patterns
        config_data = {
            "project": {
                "directories": {
                    "major_data_directory": "/path/to/data",
                    "batchfile_directory": "/path/to/batch_defs"
                },
                "ignore_substrings": [
                    "static_horiz_ribbon",
                    "._"
                ],
                "extraction_patterns": [
                    r".*_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.csv"
                ]
            },
            "rigs": {
                "old_opto": {
                    "sampling_frequency": 60,
                    "mm_per_px": 0.154
                },
                "new_opto": {
                    "sampling_frequency": 60,
                    "mm_per_px": 0.1818
                }
            },
            "datasets": {
                "test_dataset": {
                    "rig": "old_opto",
                    "patterns": ["*_test_*"],
                    "dates_vials": {
                        "2024-12-20": [2],
                        "2024-12-22": [1, 2]
                    },
                    "metadata": {
                        "extraction_patterns": [
                            r".*_(?P<dataset>\w+)_(?P<date>\d{8})\.csv"
                        ]
                    }
                },
                "plume_movie_navigation": {
                    "rig": "old_opto",
                    "dates_vials": {
                        "2024-10-18": [1, 3, 4, 5],
                        "2024-10-24": [1, 2]
                    }
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
                },
                "multi_experiment": {
                    "datasets": [
                        "test_dataset",
                        "plume_movie_navigation"
                    ],
                    "filters": {
                        "ignore_substrings": ["smoke_2a"]
                    }
                }
            }
        }
        
        # Write the config to the file
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        yield config_path
    finally:
        # Clean up after the test
        import shutil
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config_dict():
    """
    Return a sample configuration dictionary without writing to file.
    
    Returns:
        Dict[str, Any]: Sample configuration dictionary
    """
    return {
        "project": {
            "directories": {
                "major_data_directory": "/path/to/data"
            },
            "ignore_substrings": ["static_horiz_ribbon", "._"],
            "extraction_patterns": [
                r".*_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.csv"
            ]
        },
        "datasets": {
            "test_dataset": {
                "rig": "old_opto",
                "patterns": ["*_test_*"],
                "metadata": {
                    "extraction_patterns": [
                        r".*_(?P<dataset>\w+)_(?P<date>\d{8})\.csv"
                    ]
                }
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
        }
    }


# --- Column Configuration Fixtures ---

@pytest.fixture
def sample_column_config_file():
    """
    Create a temporary column config file.
    
    Returns:
        str: Path to the temporary column config file
    """
    # Create a temporary configuration file
    temp_file = tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False)
    config_path = temp_file.name
    
    # Define test configuration
    test_config = {
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
            'dtheta': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': False,
                'description': 'Change in heading',
                'alias': 'dtheta_smooth'
            },
            'signal': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': False,
                'description': 'Signal values',
                'default_value': None
            },
            'signal_disp': {
                'type': 'numpy.ndarray',
                'dimension': 2,
                'required': False,
                'description': 'Signal display data',
                'special_handling': 'transform_to_match_time_dimension'
            },
            'date': {
                'type': 'string',
                'required': False,
                'is_metadata': True,
                'description': 'Experiment date'
            },
            'exp_name': {
                'type': 'string',
                'required': False,
                'is_metadata': True,
                'description': 'Experiment name'
            },
            'rig': {
                'type': 'string',
                'required': False,
                'is_metadata': True,
                'description': 'Rig identifier'
            },
            'fly_id': {
                'type': 'string',
                'required': False,
                'is_metadata': True,
                'description': 'Fly ID'
            }
        },
        'special_handlers': {
            'transform_to_match_time_dimension': '_handle_signal_disp'
        }
    }
    
    yaml.dump(test_config, temp_file)
    temp_file.close()
    
    yield config_path
    
    # Clean up
    os.unlink(config_path)


# --- Test Data Fixtures ---

@pytest.fixture
def sample_exp_matrix():
    """
    Create sample experimental data matrix for tests.
    
    Returns:
        Dict[str, np.ndarray]: Sample experimental data
    """
    return {
        't': np.linspace(0, 10, 100),
        'x': np.random.rand(100),
        'y': np.random.rand(100)
    }


@pytest.fixture
def sample_exp_matrix_with_signal_disp():
    """
    Create sample experimental data matrix with signal_disp for tests.
    
    Returns:
        Dict[str, np.ndarray]: Sample experimental data with signal_disp
    """
    return {
        't': np.linspace(0, 10, 100),
        'x': np.random.rand(100),
        'y': np.random.rand(100),
        'signal_disp': np.random.rand(15, 100)  # 15 channels, 100 time points
    }


@pytest.fixture
def sample_exp_matrix_with_aliases():
    """
    Create sample experimental data matrix with aliased column names.
    
    Returns:
        Dict[str, np.ndarray]: Sample experimental data with aliased columns
    """
    return {
        't': np.linspace(0, 10, 100),
        'x': np.random.rand(100),
        'y': np.random.rand(100),  # Added y column which is required
        'dtheta_smooth': np.random.rand(100)  # Instead of 'dtheta'
    }


@pytest.fixture
def sample_metadata():
    """
    Create sample metadata dictionary for tests.
    
    Returns:
        Dict[str, str]: Sample metadata
    """
    return {
        'date': '2025-04-01',
        'exp_name': 'test_experiment',
        'rig': 'test_rig'
    }


# --- Mock Fixtures ---

@pytest.fixture
def mock_config_and_discovery(monkeypatch):
    """
    Setup mocks for config loading and file discovery.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture
        
    Returns:
        tuple: (mock_load_config, mock_discover_experiment_files, mock_discover_dataset_files)
    """
    from unittest.mock import MagicMock
    
    # Create mock functions
    mock_load_config = MagicMock()
    mock_discover_experiment_files = MagicMock()
    mock_discover_dataset_files = MagicMock()
    
    # Configure mock_load_config to return a standard test config
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
                "patterns": ["*_test_*"]
            }
        }
    }
    
    # Configure discovery mocks to return standard test files
    mock_discover_experiment_files.return_value = {
        "/path/to/data/file_20230101_test_1.csv": {
            "date": "20230101",
            "condition": "test",
            "replicate": "1"
        }
    }
    
    mock_discover_dataset_files.return_value = {
        "/path/to/data/file_20230101_test_1.csv": {
            "date": "20230101",
            "condition": "test",
            "replicate": "1"
        }
    }
    
    # Apply the patches
    monkeypatch.setattr("flyrigloader.api.load_config", mock_load_config)
    monkeypatch.setattr("flyrigloader.api.discover_experiment_files", mock_discover_experiment_files)
    monkeypatch.setattr("flyrigloader.api.discover_dataset_files", mock_discover_dataset_files)
    
    return mock_load_config, mock_discover_experiment_files, mock_discover_dataset_files
