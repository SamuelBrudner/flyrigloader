"""
Tests for Pydantic-based data processing functions.

These tests verify the functionality of the Pydantic-based column configuration and
data processing implementation in the consolidated pickle.py module.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

# Import from the consolidated implementation
from flyrigloader.io.pickle import make_dataframe_from_config


def create_test_config(temp_file):
    """Helper to create a test config file."""
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
            }
        }
    }
    yaml.dump(test_config, temp_file)
    temp_file.close()
    return test_config


def test_make_dataframe_from_config_basic():
    """Test basic functionality of make_dataframe_from_config."""
    # Create test data
    exp_matrix = {
        't': np.linspace(0, 10, 100),
        'x': np.random.rand(100),
        'y': np.random.rand(100)
    }
    
    # Create a temporary configuration file
    temp_file = tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False)
    config_path = temp_file.name
    create_test_config(temp_file)
    
    # Test function with basic config
    df = make_dataframe_from_config(exp_matrix, config_path)
    
    # Verify result
    assert isinstance(df, pd.DataFrame)
    assert 't' in df.columns
    assert 'x' in df.columns
    assert 'y' in df.columns
    assert len(df) == 100
    
    # Clean up
    os.unlink(config_path)


def test_make_dataframe_from_config_missing_required():
    """Test validation of required columns."""
    # Create test data missing a required column
    exp_matrix = {
        't': np.linspace(0, 10, 100),
        # Missing 'x' column
        'y': np.random.rand(100)
    }
    
    # Create a temporary configuration file
    temp_file = tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False)
    config_path = temp_file.name
    create_test_config(temp_file)
    
    # Function should raise ValueError due to missing required column
    with pytest.raises(ValueError) as excinfo:
        make_dataframe_from_config(exp_matrix, config_path)
    
    # Verify error message
    assert "Missing required columns: x" in str(excinfo.value)
    
    # Clean up
    os.unlink(config_path)


def test_make_dataframe_from_config_with_aliases():
    """Test handling of column aliases."""
    # Create test data with aliased column names
    exp_matrix = {
        't': np.linspace(0, 10, 100),
        'x': np.random.rand(100),
        'dtheta_smooth': np.random.rand(100)  # Instead of 'dtheta'
    }
    
    # Create a temporary configuration file
    temp_file = tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False)
    config_path = temp_file.name
    
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
            'dtheta': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': True,
                'description': 'Angular velocity',
                'alias': 'dtheta_smooth'
            }
        }
    }
    
    yaml.dump(test_config, temp_file)
    temp_file.close()
    
    # Test function with aliased columns
    df = make_dataframe_from_config(exp_matrix, config_path)
    
    # Verify result
    assert isinstance(df, pd.DataFrame)
    assert 't' in df.columns
    assert 'x' in df.columns
    assert 'dtheta' in df.columns  # Renamed from dtheta_smooth
    assert len(df) == 100
    
    # Clean up
    os.unlink(config_path)


def test_make_dataframe_from_config_with_metadata():
    """Test adding metadata to DataFrame."""
    # Create test data
    exp_matrix = {
        't': np.linspace(0, 10, 100),
        'x': np.random.rand(100),
        'y': np.random.rand(100)
    }
    
    # Metadata to add
    metadata = {
        'date': '2025-04-01',
        'exp_name': 'test_experiment',
        'rig': 'test_rig'
    }
    
    # Create a temporary configuration file
    temp_file = tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False)
    config_path = temp_file.name
    
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
            'date': {
                'type': 'str',
                'required': False,
                'description': 'Experiment date',
                'is_metadata': True
            },
            'exp_name': {
                'type': 'str',
                'required': False,
                'description': 'Experiment name',
                'is_metadata': True
            },
            'rig': {
                'type': 'str',
                'required': False,
                'description': 'Rig name',
                'is_metadata': True
            }
        }
    }
    
    yaml.dump(test_config, temp_file)
    temp_file.close()
    
    # Test function with metadata
    df = make_dataframe_from_config(exp_matrix, config_path, metadata=metadata)
    
    # Verify result
    assert isinstance(df, pd.DataFrame)
    assert 'date' in df.columns
    assert 'exp_name' in df.columns
    assert 'rig' in df.columns
    assert df['date'].iloc[0] == '2025-04-01'
    assert df['exp_name'].iloc[0] == 'test_experiment'
    assert df['rig'].iloc[0] == 'test_rig'
    
    # Clean up
    os.unlink(config_path)


def test_make_dataframe_from_config_with_signal_disp():
    """Test handling of signal_disp special column."""
    # Create test data with signal_disp
    t = np.linspace(0, 10, 100)
    exp_matrix = {
        't': t,
        'x': np.random.rand(100),
        'y': np.random.rand(100),
        'signal_disp': np.random.rand(15, 100)  # 15 channels, 100 time points
    }
    
    # Create a temporary configuration file
    temp_file = tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False)
    config_path = temp_file.name
    
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
            'signal_disp': {
                'type': 'numpy.ndarray',
                'dimension': 2,
                'required': False,
                'description': 'Signal display data',
                'special_handling': 'transform_to_match_time_dimension'
            }
        },
        'special_handlers': {
            'transform_to_match_time_dimension': '_handle_signal_disp'
        }
    }
    
    yaml.dump(test_config, temp_file)
    temp_file.close()
    
    # Test function with signal_disp
    df = make_dataframe_from_config(exp_matrix, config_path)
    
    # Verify result
    assert isinstance(df, pd.DataFrame)
    assert 'signal_disp' in df.columns
    assert len(df) == 100
    
    # Each item should be a 15-element array
    assert isinstance(df['signal_disp'].iloc[0], np.ndarray)
    assert df['signal_disp'].iloc[0].shape == (15,)
    
    # Clean up
    os.unlink(config_path)


def test_make_dataframe_from_config_with_default_values():
    """Test using default values for missing optional columns."""
    # Create test data missing an optional column
    exp_matrix = {
        't': np.linspace(0, 10, 100),
        'x': np.random.rand(100),
        'y': np.random.rand(100)
        # Missing 'signal' column
    }
    
    # Create a temporary configuration file
    temp_file = tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False)
    config_path = temp_file.name
    
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
            'signal': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': False,
                'description': 'Primary signal',
                'default_value': None
            }
        }
    }
    
    yaml.dump(test_config, temp_file)
    temp_file.close()
    
    # Test function with missing optional column
    df = make_dataframe_from_config(exp_matrix, config_path)
    
    # Verify result
    assert isinstance(df, pd.DataFrame)
    assert 'signal' in df.columns
    assert df['signal'].iloc[0] is None
    
    # Clean up
    os.unlink(config_path)


def test_make_dataframe_from_config_with_invalid_config():
    """Test validation of invalid configuration."""
    # Create test data
    exp_matrix = {
        't': np.linspace(0, 10, 100),
        'x': np.random.rand(100),
        'y': np.random.rand(100)
    }
    
    # Create a temporary configuration file with invalid dimension
    temp_file = tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False)
    config_path = temp_file.name
    
    test_config = {
        'columns': {
            't': {
                'type': 'numpy.ndarray',
                'dimension': 4,  # Invalid dimension (must be 1, 2, or 3)
                'required': True,
                'description': 'Time values'
            }
        }
    }
    
    yaml.dump(test_config, temp_file)
    temp_file.close()
    
    # Function should raise ValidationError due to invalid dimension
    with pytest.raises(ValidationError) as excinfo:
        make_dataframe_from_config(exp_matrix, config_path)
    
    # Verify error contains information about invalid dimension
    assert "Dimension must be 1, 2, or 3" in str(excinfo.value)
    
    # Clean up
    os.unlink(config_path)
