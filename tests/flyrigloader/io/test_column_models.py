"""
Tests for the Pydantic-based column configuration models.
"""

import os
import tempfile
import yaml
import pytest
import numpy as np
from pydantic import ValidationError
import pandas as pd

from flyrigloader.io.column_models import (
    ColumnConfig, 
    ColumnConfigDict, 
    ColumnDimension, 
    SpecialHandlerType,
    load_column_config,
    get_config_from_source
)
from flyrigloader.io.pickle import make_dataframe_from_config


def test_column_config_validation():
    """Test basic validation of individual column configurations."""
    # Valid configuration
    config = ColumnConfig(
        type="numpy.ndarray",
        dimension=1,
        required=True,
        description="Time values"
    )
    assert config.type == "numpy.ndarray"
    assert config.dimension == ColumnDimension.ONE_D
    assert config.required is True
    
    # Invalid dimension
    with pytest.raises(ValidationError):
        ColumnConfig(
            type="numpy.ndarray",
            dimension=4,  # Only 1, 2, or 3 are valid
            required=True,
            description="Invalid dimension"
        )
    
    # Invalid special handler
    with pytest.raises(ValidationError):
        ColumnConfig(
            type="numpy.ndarray",
            dimension=1,
            required=True,
            description="Test column",
            special_handling="invalid_handler"  # Not a valid handler type
        )


def test_column_config_warnings():
    """Test warning conditions in column configurations."""
    # Should warn about dimension for non-numpy type
    config = ColumnConfig(
        type="str",
        dimension=1,
        required=False,
        description="String with dimension"
    )
    assert config.type == "str"
    assert config.dimension == ColumnDimension.ONE_D
    
    # Should warn about transform_time_dimension with 1D array
    config = ColumnConfig(
        type="numpy.ndarray",
        dimension=1,
        required=False,
        description="Wrong dimension for handler",
        special_handling=SpecialHandlerType.TRANSFORM_TIME_DIMENSION
    )
    assert config.special_handling == SpecialHandlerType.TRANSFORM_TIME_DIMENSION


def test_column_config_dict_validation():
    """Test validation of the complete column configuration dictionary."""
    # Valid configuration
    config_dict = ColumnConfigDict(
        columns={
            "t": ColumnConfig(
                type="numpy.ndarray",
                dimension=1,
                required=True,
                description="Time values"
            ),
            "x": ColumnConfig(
                type="numpy.ndarray",
                dimension=1,
                required=True,
                description="X position"
            ),
            "signal_disp": ColumnConfig(
                type="numpy.ndarray",
                dimension=2,
                required=False,
                description="Signal display data",
                special_handling=SpecialHandlerType.TRANSFORM_TIME_DIMENSION
            )
        },
        special_handlers={
            "transform_to_match_time_dimension": "_handle_signal_disp"
        }
    )
    
    assert "t" in config_dict.columns
    assert config_dict.columns["signal_disp"].special_handling == SpecialHandlerType.TRANSFORM_TIME_DIMENSION
    assert "transform_to_match_time_dimension" in config_dict.special_handlers


def test_column_config_dict_missing_handler():
    """Test warning when a special handler is used but not defined."""
    # Missing handler in special_handlers
    config_dict = ColumnConfigDict(
        columns={
            "t": ColumnConfig(
                type="numpy.ndarray",
                dimension=1,
                required=True,
                description="Time values"
            ),
            "signal_disp": ColumnConfig(
                type="numpy.ndarray",
                dimension=2,
                required=False,
                description="Signal display data",
                special_handling=SpecialHandlerType.TRANSFORM_TIME_DIMENSION
            )
        },
        # No special_handlers defined
    )
    
    assert "signal_disp" in config_dict.columns
    assert config_dict.special_handlers == {}  # Empty but valid


def test_load_column_config_from_yaml():
    """Test loading column configuration from a YAML file."""
    # Create a test configuration file
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as temp_file:
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

    try:
        _extracted_from_test_load_column_config_from_yaml_38(config_path)
    finally:
        # Clean up
        os.unlink(config_path)


# TODO Rename this here and in `test_load_column_config_from_yaml`
def _extracted_from_test_load_column_config_from_yaml_38(config_path):
    # Load and validate the configuration
    config = load_column_config(config_path)

    # Verify the loaded configuration
    assert isinstance(config, ColumnConfigDict)
    assert "t" in config.columns
    assert config.columns["t"].dimension == ColumnDimension.ONE_D
    assert config.columns["signal_disp"].special_handling == SpecialHandlerType.TRANSFORM_TIME_DIMENSION
    assert config.special_handlers["transform_to_match_time_dimension"] == "_handle_signal_disp"


def test_load_column_config_with_invalid_yaml():
    """Test error handling when loading invalid YAML configuration."""
    # Create an invalid test configuration file
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as temp_file:
        config_path = temp_file.name
        
        test_config = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 5,  # Invalid dimension
                    'required': True,
                    'description': 'Time values'
                }
            }
        }
        
        yaml.dump(test_config, temp_file)
    
    try:
        # Should raise ValidationError for invalid dimension
        with pytest.raises(ValidationError):
            load_column_config(config_path)
    finally:
        # Clean up
        os.unlink(config_path)


def test_get_config_from_source_with_path():
    """Test loading configuration from a file path."""
    # Create a test configuration file
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as temp_file:
        config_path = temp_file.name
        
        test_config = {
            'columns': {
                't': {
                    'type': 'numpy.ndarray',
                    'dimension': 1,
                    'required': True,
                    'description': 'Time values'
                }
            }
        }
        
        yaml.dump(test_config, temp_file)
    
    try:
        # Load configuration from path
        config = get_config_from_source(config_path)
        
        # Verify the configuration
        assert isinstance(config, ColumnConfigDict)
        assert "t" in config.columns
        assert config.columns["t"].dimension == ColumnDimension.ONE_D
    finally:
        # Clean up
        os.unlink(config_path)


def test_get_config_from_source_with_dict():
    """Test loading configuration from a dictionary."""
    # Create a test configuration dictionary
    test_config = {
        'columns': {
            't': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': True,
                'description': 'Time values'
            }
        }
    }
    
    # Load configuration from dictionary
    config = get_config_from_source(test_config)
    
    # Verify the configuration
    assert isinstance(config, ColumnConfigDict)
    assert "t" in config.columns
    assert config.columns["t"].dimension == ColumnDimension.ONE_D


def test_get_config_from_source_with_model():
    """Test loading configuration from a ColumnConfigDict instance."""
    # Create a test ColumnConfigDict instance
    original_config = ColumnConfigDict(
        columns={
            't': ColumnConfig(
                type="numpy.ndarray",
                dimension=1,
                required=True,
                description="Time values"
            )
        }
    )
    
    # Pass the instance through get_config_from_source
    config = get_config_from_source(original_config)
    
    # Verify it's the same instance
    assert config is original_config
    assert "t" in config.columns
    assert config.columns["t"].dimension == ColumnDimension.ONE_D


def test_get_config_from_source_with_none():
    """Test that get_config_from_source uses default config when None is provided."""
    from flyrigloader.io.column_models import get_config_from_source, get_default_config_path
    
    # Get the default configuration
    default_config_path = get_default_config_path()
    
    # Load config with None (should use default)
    config = get_config_from_source(None)
    
    # Manually load default config for comparison
    expected_config = get_config_from_source(default_config_path)
    
    # Verify that the configs are equivalent (same columns)
    assert set(config.columns.keys()) == set(expected_config.columns.keys())
    assert config.special_handlers == expected_config.special_handlers


def test_get_config_from_source_with_invalid_type():
    """Test error handling with invalid configuration source type."""
    # Try to load configuration from an invalid type (list)
    with pytest.raises(TypeError):
        get_config_from_source([1, 2, 3])


def test_make_dataframe_with_dict_config():
    """Test make_dataframe_from_config with a dictionary configuration."""
    # Create test data
    exp_matrix = {
        't': np.linspace(0, 10, 100),
        'x': np.random.rand(100),
        'y': np.random.rand(100)
    }
    
    # Test configuration as a dictionary
    config_dict = {
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
    
    # Test function with dictionary config
    df = make_dataframe_from_config(exp_matrix, config_dict)
    
    # Verify result
    assert isinstance(df, pd.DataFrame)
    assert 't' in df.columns
    assert 'x' in df.columns
    assert 'y' in df.columns
    assert len(df) == 100


def test_make_dataframe_with_model_config():
    """Test make_dataframe_from_config with a ColumnConfigDict instance."""
    # Create test data
    exp_matrix = {
        't': np.linspace(0, 10, 100),
        'x': np.random.rand(100),
        'y': np.random.rand(100)
    }
    
    # Create a test ColumnConfigDict instance
    config_model = ColumnConfigDict(
        columns={
            't': ColumnConfig(
                type="numpy.ndarray",
                dimension=1,
                required=True,
                description="Time values"
            ),
            'x': ColumnConfig(
                type="numpy.ndarray",
                dimension=1,
                required=True,
                description="X position"
            ),
            'y': ColumnConfig(
                type="numpy.ndarray",
                dimension=1,
                required=True,
                description="Y position"
            )
        }
    )
    
    # Test function with model config
    df = make_dataframe_from_config(exp_matrix, config_model)
    
    # Verify result
    assert isinstance(df, pd.DataFrame)
    assert 't' in df.columns
    assert 'x' in df.columns
    assert 'y' in df.columns
    assert len(df) == 100


def test_make_dataframe_with_default_config():
    """Test that make_dataframe_from_config works with default configuration."""
    from flyrigloader.io.pickle import make_dataframe_from_config
    
    # Create a simple test matrix with all required columns from default config
    time_values = np.array([0, 1, 2, 3, 4])
    exp_matrix = {
        # Time dimension
        "t": time_values,
        
        # Position and tracking
        "trjn": np.array([1, 1, 1, 1, 1]),
        "x": np.array([10, 11, 12, 13, 14]),
        "y": np.array([20, 21, 22, 23, 24]),
        
        # Orientation and movement
        "theta": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        "theta_smooth": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        "dtheta": np.array([0.01, 0.01, 0.01, 0.01, 0.01]),
        "vx": np.array([1.0, 1.1, 1.2, 1.3, 1.4]),
        "vy": np.array([2.0, 2.1, 2.2, 2.3, 2.4]),
        "spd": np.array([2.2, 2.4, 2.5, 2.6, 2.8]),
        
        # Event markers
        "jump": np.array([0, 0, 1, 0, 0])
    }
    
    # Create DataFrame without specifying config (should use default)
    df = make_dataframe_from_config(exp_matrix)
    
    # Verify DataFrame has expected columns
    assert "t" in df.columns
    assert "x" in df.columns
    assert "y" in df.columns
    assert "trjn" in df.columns
    assert "theta" in df.columns
    assert "theta_smooth" in df.columns
    assert "dtheta" in df.columns
    assert "vx" in df.columns
    assert "vy" in df.columns
    assert "spd" in df.columns
    assert "jump" in df.columns
    
    # Check data values
    np.testing.assert_array_equal(df["t"].values, exp_matrix["t"])
    np.testing.assert_array_equal(df["x"].values, exp_matrix["x"])
    
    # Verify row count matches the time dimension
    assert len(df) == len(time_values)
