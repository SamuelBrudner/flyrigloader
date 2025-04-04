"""
Tests for column configuration functionality.

These tests verify that the column configuration system works as expected.
"""

import os
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import yaml

from flyrigloader.io.pickle import (
    make_dataframe_from_config
)
from flyrigloader.io.column_models import load_column_config

def test_column_config_file_creation():
    """Test that we can create a valid column configuration file."""
    # Create a temporary configuration file
    temp_file = tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False)
    config_path = temp_file.name
    
    # Define a minimal test configuration
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
    
    # Write configuration to file
    yaml.dump(test_config, temp_file)
    temp_file.close()
    
    # Verify file exists and has content
    assert os.path.exists(config_path)
    
    # Load the configuration and validate structure
    with open(config_path, 'r') as f:
        loaded_config = yaml.safe_load(f)
    
    assert 'columns' in loaded_config
    assert 't' in loaded_config['columns']
    assert loaded_config['columns']['t']['required'] is True
    
    # Clean up
    os.unlink(config_path)


def test_validate_required_columns_basic(sample_column_config_file, sample_exp_matrix):
    """Test basic functionality of validate_required_columns."""
    # This test now indirectly validates required columns by ensuring
    # make_dataframe_from_config doesn't raise an error when all required columns are present
    df = make_dataframe_from_config(sample_exp_matrix, sample_column_config_file)
    assert isinstance(df, pd.DataFrame)
    assert 't' in df.columns
    assert 'x' in df.columns
    assert 'y' in df.columns
    assert len(df) == 100

def test_validate_required_columns_missing(sample_column_config_file):
    """Test behavior with missing required columns."""
    # Create test data missing a required column
    test_exp_matrix = {
        't': np.linspace(0, 10, 100),
        # Missing 'x' column
        'y': np.random.rand(100)
    }
    
    # Function should raise ValueError
    with pytest.raises(ValueError) as excinfo:
        make_dataframe_from_config(test_exp_matrix, sample_column_config_file)
    
    # Verify error message
    assert "Missing required columns: x" in str(excinfo.value)

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

def test_make_dataframe_from_config_with_aliases(sample_column_config_file, sample_exp_matrix_with_aliases):
    """Test handling of column aliases."""
    # Test function with aliased columns
    df = make_dataframe_from_config(sample_exp_matrix_with_aliases, sample_column_config_file)
    
    # Verify result
    assert isinstance(df, pd.DataFrame)
    assert 't' in df.columns
    assert 'x' in df.columns
    assert 'dtheta' in df.columns  # Renamed from dtheta_smooth
    assert len(df) == 100

def test_make_dataframe_from_config_with_metadata(sample_column_config_file, sample_exp_matrix, sample_metadata):
    """Test adding metadata to DataFrame."""
    # Test function with metadata
    df = make_dataframe_from_config(sample_exp_matrix, sample_column_config_file, metadata=sample_metadata)
    
    # Verify result
    assert isinstance(df, pd.DataFrame)
    assert 'date' in df.columns
    assert 'exp_name' in df.columns
    assert 'rig' in df.columns
    assert df['date'].iloc[0] == '2025-04-01'
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

def test_make_dataframe_from_config_with_default_values(sample_column_config_file, sample_exp_matrix):
    """Test using default values for missing optional columns."""
    # Test function with missing optional column
    df = make_dataframe_from_config(sample_exp_matrix, sample_column_config_file)
    
    # Verify result
    assert isinstance(df, pd.DataFrame)
    assert 'signal' in df.columns
    assert df['signal'].iloc[0] is None
