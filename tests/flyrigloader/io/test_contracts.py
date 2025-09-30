"""
Contract guard tests for column configuration system.

These tests verify that functions honor their contracts:
- Preconditions are enforced (invalid inputs rejected)
- Postconditions are guaranteed (outputs meet specifications)
- Invariants are preserved (system properties always hold)

See docs/SEMANTIC_MODEL.md for the formal semantic model and contracts.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings
from pydantic import ValidationError

from flyrigloader.io.column_models import (
    ColumnConfig,
    ColumnConfigDict,
    load_column_config,
    get_default_config,
)
from flyrigloader.io.transformers import make_dataframe_from_config
from flyrigloader.exceptions import TransformError


# ============================================================================
# QUICK WIN #1: INV-5 - Output Row Count Invariant
# ============================================================================

class TestOutputConsistencyInvariant:
    """
    INV-5: DataFrame row count ALWAYS equals input array length.
    
    This is a fundamental invariant that must hold for ANY valid input.
    """
    
    def test_output_row_count_matches_input_basic(self):
        """Basic test: output row count matches input."""
        config = ColumnConfigDict.model_validate({
            'columns': {
                't': {'type': 'numpy.ndarray', 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'required': True, 'description': 'X'}
            },
            'special_handlers': {}
        })
        
        exp_matrix = {
            't': np.linspace(0, 10, 100),
            'x': np.random.rand(100)
        }
        
        df = make_dataframe_from_config(exp_matrix, config)
        
        # INV-5: MUST hold
        assert len(df) == 100
    
    @given(row_count=st.integers(min_value=1, max_value=1000))
    @settings(max_examples=50, deadline=None)
    def test_output_row_count_invariant_property(self, row_count):
        """Property test: row count invariant holds for ANY valid row count."""
        config = ColumnConfigDict.model_validate({
            'columns': {
                't': {'type': 'numpy.ndarray', 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'required': True, 'description': 'X'}
            },
            'special_handlers': {}
        })
        
        exp_matrix = {
            't': np.arange(row_count, dtype=float),
            'x': np.arange(row_count, dtype=float)
        }
        
        df = make_dataframe_from_config(exp_matrix, config)
        
        # INV-5: MUST hold for ANY row_count
        assert len(df) == row_count
    
    def test_output_row_count_with_optional_columns(self):
        """Row count invariant holds even with optional columns."""
        config = ColumnConfigDict.model_validate({
            'columns': {
                't': {'type': 'numpy.ndarray', 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'required': False, 'description': 'X'}
            },
            'special_handlers': {}
        })
        
        # Only provide required column
        exp_matrix = {'t': np.linspace(0, 10, 50)}
        
        df = make_dataframe_from_config(exp_matrix, config)
        
        # INV-5: Still holds
        assert len(df) == 50


# ============================================================================
# QUICK WIN #2: Idempotency Tests
# ============================================================================

class TestConfigLoadingIdempotency:
    """
    Property: Loading the same config file twice produces equivalent results.
    
    This is a fundamental property - config loading should be idempotent.
    """
    
    def test_config_loading_idempotent(self, tmp_path):
        """Loading same config twice gives equivalent results."""
        # Create test config file
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
columns:
  t:
    type: numpy.ndarray
    dimension: 1
    required: true
    description: Time values
  x:
    type: numpy.ndarray
    dimension: 1
    required: true
    description: X position
special_handlers: {}
""")
        
        # Load twice
        config1 = load_column_config(str(config_file))
        config2 = load_column_config(str(config_file))
        
        # Must be equivalent
        assert config1.columns.keys() == config2.columns.keys()
        assert all(
            config1.columns[k].type == config2.columns[k].type 
            for k in config1.columns
        )
        assert all(
            config1.columns[k].required == config2.columns[k].required
            for k in config1.columns
        )
        
        # Should produce identical dict representation
        assert config1.model_dump() == config2.model_dump()
    
    def test_get_default_config_idempotent(self):
        """get_default_config() is idempotent."""
        config1 = get_default_config()
        config2 = get_default_config()
        
        # Must be equivalent
        assert config1.columns.keys() == config2.columns.keys()
        assert config1.model_dump() == config2.model_dump()


# ============================================================================
# QUICK WIN #3: Precondition Enforcement
# ============================================================================

class TestPreconditionEnforcement:
    """
    Contract: Functions must enforce their preconditions.
    
    Invalid inputs should be rejected with clear error messages.
    """
    
    def test_make_dataframe_rejects_mismatched_lengths(self):
        """
        CONTRACT VIOLATION: Arrays in exp_matrix must have same length (INV-2).
        
        This must be caught and rejected.
        """
        config = ColumnConfigDict.model_validate({
            'columns': {
                't': {'type': 'numpy.ndarray', 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'required': True, 'description': 'X'}
            },
            'special_handlers': {}
        })
        
        # PRECONDITION VIOLATION: mismatched lengths
        bad_matrix = {
            't': np.zeros(100),
            'x': np.zeros(50)  # Different length!
        }
        
        # Must reject
        with pytest.raises((TransformError, ValueError)):
            make_dataframe_from_config(bad_matrix, config)
    
    def test_make_dataframe_rejects_missing_required_column(self):
        """
        CONTRACT VIOLATION: Required columns must be present (INV-3).
        
        This must be caught and rejected.
        """
        config = ColumnConfigDict.model_validate({
            'columns': {
                't': {'type': 'numpy.ndarray', 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'required': True, 'description': 'X'}
            },
            'special_handlers': {}
        })
        
        # PRECONDITION VIOLATION: missing required 'x'
        incomplete_matrix = {
            't': np.zeros(100)
            # Missing 'x'!
        }
        
        # Must reject
        with pytest.raises((TransformError, ValueError, KeyError)):
            make_dataframe_from_config(incomplete_matrix, config)
    
    def test_load_column_config_rejects_nonexistent_file(self):
        """
        CONTRACT VIOLATION: File must exist.
        
        This must be caught and rejected.
        """
        # PRECONDITION VIOLATION: file doesn't exist
        with pytest.raises((FileNotFoundError, OSError, RuntimeError, TypeError)):
            load_column_config("/nonexistent/path/to/config.yaml")
    
    def test_column_config_dict_rejects_invalid_structure(self):
        """
        CONTRACT VIOLATION: Must have valid Pydantic structure (INV-1).
        
        This must be caught and rejected.
        """
        # PRECONDITION VIOLATION: missing required 'type' field
        invalid_config = {
            'columns': {
                't': {
                    # Missing 'type' field!
                    'description': 'Time'
                }
            },
            'special_handlers': {}
        }
        
        # Must reject
        with pytest.raises(ValidationError):
            ColumnConfigDict.model_validate(invalid_config)


# ============================================================================
# Postcondition Verification Tests
# ============================================================================

class TestPostconditionVerification:
    """
    Contract: Functions must guarantee their postconditions.
    
    Outputs must meet the specifications in the contract.
    """
    
    def test_load_column_config_postconditions(self, tmp_path):
        """
        Verify load_column_config() postconditions:
        - Returns valid ColumnConfigDict
        - result.columns is non-empty
        - All columns are valid ColumnConfig instances
        """
        # Create valid config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
columns:
  t:
    type: numpy.ndarray
    required: true
    description: Time
special_handlers: {}
""")
        
        # OPERATION
        result = load_column_config(str(config_file))
        
        # POSTCONDITION: returns ColumnConfigDict
        assert isinstance(result, ColumnConfigDict)
        
        # POSTCONDITION: columns is non-empty
        assert len(result.columns) > 0
        
        # POSTCONDITION: all columns are valid ColumnConfig
        for col in result.columns.values():
            assert isinstance(col, ColumnConfig)
            assert hasattr(col, 'type')
            assert hasattr(col, 'description')
    
    def test_make_dataframe_postconditions(self):
        """
        Verify make_dataframe_from_config() postconditions:
        - Returns pandas DataFrame
        - All required columns present
        - Row count matches input
        """
        config = ColumnConfigDict.model_validate({
            'columns': {
                't': {'type': 'numpy.ndarray', 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'required': True, 'description': 'X'}
            },
            'special_handlers': {}
        })
        
        exp_matrix = {
            't': np.linspace(0, 10, 100),
            'x': np.random.rand(100)
        }
        
        # OPERATION
        result = make_dataframe_from_config(exp_matrix, config)
        
        # POSTCONDITION: returns DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # POSTCONDITION: all required columns present
        required_cols = [name for name, col in config.columns.items() if col.required]
        assert all(col in result.columns for col in required_cols)
        
        # POSTCONDITION: row count matches input (INV-5)
        assert len(result) == 100


# ============================================================================
# Determinism Tests
# ============================================================================

class TestDeterminism:
    """
    Property: Same inputs always produce same outputs.
    
    Operations should be deterministic.
    """
    
    def test_transformation_deterministic(self):
        """Same config + data always produces same DataFrame."""
        config = ColumnConfigDict.model_validate({
            'columns': {
                't': {'type': 'numpy.ndarray', 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'required': True, 'description': 'X'}
            },
            'special_handlers': {}
        })
        
        # Use fixed seed for reproducibility
        np.random.seed(42)
        exp_matrix = {
            't': np.linspace(0, 10, 100),
            'x': np.random.rand(100)
        }
        
        # Transform twice
        df1 = make_dataframe_from_config(exp_matrix, config)
        df2 = make_dataframe_from_config(exp_matrix, config)
        
        # Must be identical
        assert df1.equals(df2)
        assert list(df1.columns) == list(df2.columns)
        assert len(df1) == len(df2)


# ============================================================================
# Order Independence Tests
# ============================================================================

class TestOrderIndependence:
    """
    Property: Column order in exp_matrix shouldn't affect transformation.
    
    The transformation should be order-independent.
    """
    
    def test_column_order_independence(self):
        """Column order in exp_matrix doesn't affect result."""
        config = ColumnConfigDict.model_validate({
            'columns': {
                't': {'type': 'numpy.ndarray', 'required': True, 'description': 'Time'},
                'x': {'type': 'numpy.ndarray', 'required': True, 'description': 'X'},
                'y': {'type': 'numpy.ndarray', 'required': True, 'description': 'Y'}
            },
            'special_handlers': {}
        })
        
        # Use deterministic data (not random) for order independence test
        t_data = np.linspace(0, 10, 100)
        x_data = np.arange(100, dtype=float)
        y_data = np.arange(100, 200, dtype=float)
        
        # Create matrices with different column orders but same data
        matrix1 = {
            't': t_data.copy(),
            'x': x_data.copy(),
            'y': y_data.copy()
        }
        
        matrix2 = {
            'y': y_data.copy(),  # Different order
            't': t_data.copy(),
            'x': x_data.copy()
        }
        
        df1 = make_dataframe_from_config(matrix1, config)
        df2 = make_dataframe_from_config(matrix2, config)
        
        # Same columns present
        assert set(df1.columns) == set(df2.columns)
        
        # Same data (order-independent)
        np.testing.assert_array_equal(df1['t'].values, df2['t'].values)
        np.testing.assert_array_equal(df1['x'].values, df2['x'].values)
        np.testing.assert_array_equal(df1['y'].values, df2['y'].values)

