"""
Tests for the Pydantic-based column configuration models.
"""

import os
import tempfile
import yaml
import pytest
import numpy as np
from pydantic import ValidationError

from flyrigloader.io.column_models import (
    ColumnConfig, 
    ColumnConfigDict, 
    ColumnDimension, 
    SpecialHandlerType,
    load_column_config
)


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
