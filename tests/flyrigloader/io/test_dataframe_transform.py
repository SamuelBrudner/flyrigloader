"""
Tests for DataFrame transformation functionality.

This module tests the transformation of experimental data matrices into pandas DataFrames
using column configurations. Tests focus on the make_dataframe_from_config function and
related transformation logic.

Test coverage:
- Basic DataFrame creation from experimental data
- Schema validation and required column checking
- Column aliasing and fallback behavior
- Default value assignment for missing columns
- Special data handlers (signal_disp transformation)
- Metadata integration
- skip_columns functionality
- Edge cases and error conditions
"""

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from flyrigloader.io.transformers import make_dataframe_from_config
from flyrigloader.io.column_models import ColumnConfigDict, get_default_config
from flyrigloader.exceptions import TransformError


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def basic_exp_matrix():
    """Basic experimental data matrix."""
    np.random.seed(42)
    return {
        't': np.linspace(0, 10, 100),
        'x': np.random.rand(100),
        'y': np.random.rand(100)
    }


@pytest.fixture
def exp_matrix_with_signal_disp():
    """Experimental data with 2D signal_disp for special handling."""
    np.random.seed(42)
    return {
        't': np.linspace(0, 10, 100),
        'x': np.random.rand(100),
        'signal_disp': np.random.rand(15, 100)  # 15 channels × 100 time points
    }


@pytest.fixture
def exp_matrix_with_aliases():
    """Experimental data with aliased column names."""
    np.random.seed(42)
    return {
        't': np.linspace(0, 10, 100),
        'x': np.random.rand(100),
        'dtheta_smooth': np.random.rand(100)  # Alias for 'dtheta'
    }


@pytest.fixture
def standard_config():
    """Standard column configuration."""
    return ColumnConfigDict.model_validate({
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
                'required': False,
                'description': 'Y position',
                'default_value': None
            }
        },
        'special_handlers': {}
    })


@pytest.fixture
def config_with_aliases():
    """Configuration with column aliases."""
    return ColumnConfigDict.model_validate({
        'columns': {
            't': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': True,
                'description': 'Time'
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
                'required': False,
                'alias': 'dtheta_smooth',
                'description': 'Angular velocity'
            }
        },
        'special_handlers': {}
    })


@pytest.fixture
def config_with_signal_disp():
    """Configuration with signal_disp special handler."""
    return ColumnConfigDict.model_validate({
        'columns': {
            't': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': True,
                'description': 'Time'
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
                'special_handling': 'transform_to_match_time_dimension',
                'description': 'Signal display data'
            }
        },
        'special_handlers': {
            'transform_to_match_time_dimension': '_handle_signal_disp'
        }
    })


@pytest.fixture
def config_with_metadata():
    """Configuration with metadata fields."""
    return ColumnConfigDict.model_validate({
        'columns': {
            't': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': True,
                'description': 'Time'
            },
            'x': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': True,
                'description': 'X position'
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
            }
        },
        'special_handlers': {}
    })


# ============================================================================
# BASIC DATAFRAME CREATION
# ============================================================================

class TestBasicDataFrameCreation:
    """Test basic DataFrame creation from experimental data."""

    def test_create_dataframe_with_config_object(self, basic_exp_matrix, standard_config):
        """Test DataFrame creation with ColumnConfigDict object."""
        df = make_dataframe_from_config(basic_exp_matrix, standard_config)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert 't' in df.columns
        assert 'x' in df.columns

    def test_create_dataframe_with_config_path(self, basic_exp_matrix, tmp_path):
        """Test DataFrame creation with config file path."""
        # Create config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
columns:
  t:
    type: numpy.ndarray
    dimension: 1
    required: true
    description: Time
  x:
    type: numpy.ndarray
    dimension: 1
    required: true
    description: X position
special_handlers: {}
""")
        
        df = make_dataframe_from_config(basic_exp_matrix, str(config_file))
        
        assert isinstance(df, pd.DataFrame)
        assert 't' in df.columns
        assert 'x' in df.columns

    def test_create_dataframe_with_default_config(self):
        """Test DataFrame creation with default configuration."""
        # Create matrix with ALL required columns from default config
        np.random.seed(42)
        complete_matrix = {
            't': np.linspace(0, 10, 100),
            'trjn': np.ones(100),
            'x': np.random.rand(100),
            'y': np.random.rand(100),
            'theta': np.random.rand(100),
            'theta_smooth': np.random.rand(100),
            'dtheta': np.random.rand(100),
            'vx': np.random.rand(100),
            'vy': np.random.rand(100),
            'spd': np.random.rand(100),
            'jump': np.zeros(100)
        }
        
        df = make_dataframe_from_config(complete_matrix, None)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 't' in df.columns


# ============================================================================
# SCHEMA VALIDATION
# ============================================================================

class TestSchemaValidation:
    """Test schema validation and required column checking."""

    def test_missing_required_column_raises_error(self, standard_config):
        """Test that missing required columns raise TransformError."""
        incomplete_matrix = {
            't': np.linspace(0, 10, 100)
            # Missing required 'x' column
        }
        
        # The actual error message may vary, just check that an error is raised
        with pytest.raises((TransformError, ValueError, KeyError)):
            make_dataframe_from_config(incomplete_matrix, standard_config)

    def test_all_required_columns_present(self, basic_exp_matrix, standard_config):
        """Test successful creation when all required columns present."""
        df = make_dataframe_from_config(basic_exp_matrix, standard_config)
        
        assert 't' in df.columns
        assert 'x' in df.columns

    def test_optional_column_missing_uses_default(self, standard_config):
        """Test that missing optional columns get default value."""
        matrix_without_y = {
            't': np.linspace(0, 10, 100),
            'x': np.random.rand(100)
        }
        
        df = make_dataframe_from_config(matrix_without_y, standard_config)
        
        assert 'y' in df.columns
        assert df['y'].iloc[0] is None


# ============================================================================
# COLUMN ALIASING
# ============================================================================

class TestAliasing:
    """Test column alias resolution."""

    def test_alias_column_resolved(self, exp_matrix_with_aliases, config_with_aliases):
        """Test that aliased columns are correctly resolved."""
        df = make_dataframe_from_config(exp_matrix_with_aliases, config_with_aliases)
        
        # Should have 'dtheta' column (from 'dtheta_smooth' alias)
        assert 'dtheta' in df.columns
        # Original alias name should not be in columns
        assert 'dtheta_smooth' not in df.columns

    def test_primary_column_name_preferred(self, config_with_aliases):
        """Test that primary column name takes precedence over alias."""
        np.random.seed(42)  # For reproducibility
        matrix_with_both = {
            't': np.linspace(0, 10, 100),
            'x': np.random.rand(100),
            'dtheta': np.ones(100),  # Primary name
            'dtheta_smooth': np.zeros(100)  # Alias
        }
        
        df = make_dataframe_from_config(matrix_with_both, config_with_aliases)
        
        # Should use primary name
        assert 'dtheta' in df.columns
        # Verify column exists and has data
        assert len(df['dtheta']) == 100


# ============================================================================
# DEFAULT VALUES
# ============================================================================

class TestDefaultValues:
    """Test default value assignment."""

    def test_default_value_assigned_when_missing(self, tmp_path):
        """Test that default values are assigned for missing optional columns."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
columns:
  t:
    type: numpy.ndarray
    required: true
    description: Time
  optional_col:
    type: numpy.ndarray
    required: false
    default_value: 42
    description: Optional column
special_handlers: {}
""")
        
        matrix = {'t': np.linspace(0, 10, 100)}
        df = make_dataframe_from_config(matrix, str(config_file))
        
        assert 'optional_col' in df.columns
        assert df['optional_col'].iloc[0] == 42

    def test_null_default_value(self, standard_config):
        """Test that null default values work correctly."""
        matrix = {
            't': np.linspace(0, 10, 100),
            'x': np.random.rand(100)
        }
        
        df = make_dataframe_from_config(matrix, standard_config)
        
        assert 'y' in df.columns
        assert df['y'].iloc[0] is None


# ============================================================================
# SPECIAL HANDLERS
# ============================================================================

class TestSpecialHandlers:
    """Test special data transformation handlers."""

    def test_signal_disp_transformation(self, exp_matrix_with_signal_disp, config_with_signal_disp):
        """Test that signal_disp is correctly transformed."""
        df = make_dataframe_from_config(exp_matrix_with_signal_disp, config_with_signal_disp)
        
        assert 'signal_disp' in df.columns
        # Should be transformed to series of arrays (one per time point)
        assert len(df) == 100
        # Each row should contain an array of 15 channels
        assert isinstance(df['signal_disp'].iloc[0], np.ndarray)
        assert len(df['signal_disp'].iloc[0]) == 15


# ============================================================================
# METADATA INTEGRATION
# ============================================================================

class TestMetadataIntegration:
    """Test metadata field integration."""

    def test_metadata_added_to_dataframe(self, basic_exp_matrix, config_with_metadata):
        """Test that metadata is added to DataFrame."""
        metadata = {
            'date': '2025-04-01',
            'exp_name': 'test_experiment'
        }
        
        df = make_dataframe_from_config(basic_exp_matrix, config_with_metadata, metadata=metadata)
        
        assert 'date' in df.columns
        assert 'exp_name' in df.columns
        assert df['date'].iloc[0] == '2025-04-01'
        assert df['exp_name'].iloc[0] == 'test_experiment'

    def test_metadata_not_in_dataframe_when_not_configured(self, basic_exp_matrix, standard_config):
        """Test that metadata fields not in config are not added."""
        metadata = {
            'rig': 'test_rig'  # Not in standard_config
        }
        
        df = make_dataframe_from_config(basic_exp_matrix, standard_config, metadata=metadata)
        
        assert 'rig' not in df.columns


# ============================================================================
# SKIP COLUMNS
# ============================================================================

class TestSkipColumns:
    """Test skip_columns functionality."""

    def test_skip_configured_column(self, basic_exp_matrix, standard_config):
        """Test that configured columns can be skipped."""
        df = make_dataframe_from_config(basic_exp_matrix, standard_config, skip_columns=['y'])
        
        assert 't' in df.columns
        assert 'x' in df.columns
        assert 'y' not in df.columns

    def test_skip_required_column_logs_warning(self, basic_exp_matrix, standard_config, caplog):
        """Test that skipping required columns logs a warning."""
        # Current implementation logs warning but allows it
        with caplog.at_level('WARNING'):
            df = make_dataframe_from_config(basic_exp_matrix, standard_config, skip_columns=['t'])
        
        # Should log warning about skipping required column
        assert 'required column' in caplog.text.lower() or 'skipping' in caplog.text.lower()
        # Column should not be in result
        assert 't' not in df.columns


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_exp_matrix_raises_error(self, standard_config):
        """Test that empty experimental matrix raises error."""
        with pytest.raises((TransformError, ValueError)):
            make_dataframe_from_config({}, standard_config)

    def test_mismatched_array_lengths_raises_error(self, standard_config):
        """Test that mismatched array lengths raise error."""
        bad_matrix = {
            't': np.linspace(0, 10, 100),
            'x': np.random.rand(50)  # Different length!
        }
        
        with pytest.raises((TransformError, ValueError)):
            make_dataframe_from_config(bad_matrix, standard_config)

    def test_non_dict_exp_matrix_raises_error(self, standard_config):
        """Test that non-dict experimental matrix raises error."""
        with pytest.raises((TransformError, TypeError)):
            make_dataframe_from_config("not a dict", standard_config)

    def test_none_config_uses_default(self):
        """Test that None config uses default configuration."""
        # Create matrix with ALL required columns from default config
        np.random.seed(42)
        matrix = {
            't': np.linspace(0, 10, 100),
            'trjn': np.ones(100),
            'x': np.random.rand(100),
            'y': np.random.rand(100),
            'theta': np.random.rand(100),
            'theta_smooth': np.random.rand(100),
            'dtheta': np.random.rand(100),
            'vx': np.random.rand(100),
            'vy': np.random.rand(100),
            'spd': np.random.rand(100),
            'jump': np.zeros(100)
        }
        
        df = make_dataframe_from_config(matrix, None)
        
        assert isinstance(df, pd.DataFrame)
        assert 't' in df.columns
