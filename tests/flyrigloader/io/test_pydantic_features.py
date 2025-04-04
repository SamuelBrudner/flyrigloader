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


# Note: This function can be removed once all tests are updated to use fixtures
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


def test_make_dataframe_from_config_basic(sample_column_config_file, sample_exp_matrix):
    """Test basic functionality of make_dataframe_from_config."""
    # Test function with basic config
    df = make_dataframe_from_config(sample_exp_matrix, sample_column_config_file)
    
    # Verify result
    assert isinstance(df, pd.DataFrame)
    assert 't' in df.columns
    assert 'x' in df.columns
    assert 'y' in df.columns
    assert len(df) == 100


def test_make_dataframe_from_config_missing_required(sample_column_config_file):
    """Test validation of required columns."""
    # Create test data missing a required column
    exp_matrix = {
        't': np.linspace(0, 10, 100),
        # Missing 'x' column
        'y': np.random.rand(100)
    }
    
    # Function should raise ValueError due to missing required column
    with pytest.raises(ValueError) as excinfo:
        make_dataframe_from_config(exp_matrix, sample_column_config_file)
    
    # Verify error message
    assert "Missing required columns: x" in str(excinfo.value)


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
    assert 'dtheta' in df.columns  # Should be renamed from dtheta_smooth
    assert len(df) == 100
    
    # Clean up
    os.unlink(config_path)


def test_make_dataframe_from_config_with_metadata(sample_column_config_file, sample_exp_matrix):
    """Test adding metadata to DataFrame."""
    # Create metadata
    metadata = {
        'date': '2025-04-01',
        'fly_id': 'fly-123',
        'exp_name': 'test_experiment',
        'rig': 'test_rig'
    }
    
    # Test function with metadata
    df = make_dataframe_from_config(
        sample_exp_matrix,
        sample_column_config_file,
        metadata=metadata
    )
    
    # Verify result
    assert isinstance(df, pd.DataFrame)
    assert 'date' in df.columns
    assert 'fly_id' in df.columns
    assert 'exp_name' in df.columns
    assert 'rig' in df.columns
    
    # All rows should have the same metadata values
    assert df['date'].iloc[0] == '2025-04-01'
    assert df['fly_id'].iloc[0] == 'fly-123'
    assert df['exp_name'].iloc[0] == 'test_experiment'
    assert df['rig'].iloc[0] == 'test_rig'


def test_make_dataframe_from_config_with_signal_disp(sample_exp_matrix_with_signal_disp, sample_column_config_file):
    """Test handling of signal_disp special column."""
    # Test function with signal_disp
    df = make_dataframe_from_config(sample_exp_matrix_with_signal_disp, sample_column_config_file)
    
    # Verify result
    assert isinstance(df, pd.DataFrame)
    assert 'signal_disp' in df.columns
    assert len(df) == 100
    
    # Each item should be a 15-element array
    assert isinstance(df['signal_disp'].iloc[0], np.ndarray)
    assert df['signal_disp'].iloc[0].shape == (15,)


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
