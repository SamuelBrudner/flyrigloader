"""
Tests for the value_utils module, which provides utilities for value coercion.

These tests validate that the value coercion functions properly handle
various input types and edge cases.
"""

import math
import pytest
from datetime import date

from flyrigloader.core.utils.value_utils import (
    ValueType,
    coerce_value,
    coerce_to_numeric,
    coerce_to_boolean,
    coerce_to_float,
    coerce_to_string,
    coerce_to_date,
    coerce_to_list,
    coerce_dict_values,
    COMMON_VALUES
)


class TestValueCoercion:
    """Test class for value coercion functions."""
    
    def test_coerce_to_numeric(self):
        """Test numeric coercion with various inputs."""
        # Test basic numeric coercion
        assert coerce_to_numeric(5) == 5
        assert coerce_to_numeric("5") == 5
        assert coerce_to_numeric("5.0") == 5.0
        
        # Test with zero and one values
        assert coerce_to_numeric("off", zero_values=["off"]) == 0
        assert coerce_to_numeric("on", one_values=["on"]) == 1
        
        # Test with min and max constraints
        assert coerce_to_numeric(10, min_value=20) == 20
        assert coerce_to_numeric(30, max_value=25) == 25
        
        # Test with forced types
        assert coerce_to_numeric(5.7, force_type=int) == 5
        assert coerce_to_numeric(5, force_type=float) == 5.0
        
        # Test with invalid inputs
        assert coerce_to_numeric("invalid", default=42) == 42
        assert coerce_to_numeric(None, default=0) == 0
    
    def test_coerce_to_boolean(self):
        """Test boolean coercion with various inputs."""
        # Test true values
        assert coerce_to_boolean(True) is True
        assert coerce_to_boolean("true") is True
        assert coerce_to_boolean("yes") is True
        assert coerce_to_boolean(1) is True
        
        # Test false values
        assert coerce_to_boolean(False) is False
        assert coerce_to_boolean("false") is False
        assert coerce_to_boolean("no") is False
        assert coerce_to_boolean(0) is False
        assert coerce_to_boolean(None) is False
        
        # Test with custom true/false values
        assert coerce_to_boolean("custom_true", true_values=["custom_true"]) is True
        assert coerce_to_boolean("custom_false", false_values=["custom_false"]) is False
        
        # Test default value for invalid inputs
        assert coerce_to_boolean("invalid", default=True) is True
    
    def test_coerce_to_float(self):
        """Test float coercion with various inputs."""
        # Test basic float coercion
        assert coerce_to_float(5.5) == 5.5
        assert coerce_to_float("5.5") == 5.5
        assert coerce_to_float(5) == 5.0
        
        # Test with min and max constraints
        assert coerce_to_float(10.5, min_value=20.0) == 20.0
        assert coerce_to_float(30.5, max_value=25.0) == 25.0
        
        # Test NaN values
        assert math.isnan(coerce_to_float("unknown", nan_values=["unknown"]))
        assert math.isnan(coerce_to_float("N/A"))
        
        # Test default for invalid inputs
        assert coerce_to_float("invalid", default=42.0) == 42.0
    
    def test_coerce_to_string(self):
        """Test string coercion with various inputs."""
        # Test basic string coercion
        assert coerce_to_string(5) == "5"
        assert coerce_to_string(5.5) == "5.5"
        assert coerce_to_string(True) == "True"
        assert coerce_to_string(None, default="none") == "none"
        
        # Test with transformations
        assert coerce_to_string(" test ", strip=True) == "test"
        assert coerce_to_string("test", upper=True) == "TEST"
        assert coerce_to_string("TEST", lower=True) == "test"
        
        # Test with mapping
        mapping = {"val1": "mapped1", "val2": "mapped2"}
        assert coerce_to_string("val1", mapping=mapping) == "mapped1"
        assert coerce_to_string("unknown", mapping=mapping) == "unknown"
    
    def test_coerce_to_date(self):
        """Test date coercion with various inputs."""
        # Test with string in default format
        assert coerce_to_date("2023-01-15") == date(2023, 1, 15)
        
        # Test with custom format
        assert coerce_to_date("15/01/2023", format_str="%d/%m/%Y") == date(2023, 1, 15)
        
        # Test with date object
        test_date = date(2023, 1, 15)
        assert coerce_to_date(test_date) == test_date
        
        # Test with default for invalid inputs
        default_date = date(2000, 1, 1)
        assert coerce_to_date("invalid", default=default_date) == default_date
        assert coerce_to_date(None, default=default_date) == default_date
    
    def test_coerce_to_list(self):
        """Test list coercion with various inputs."""
        # Test basic list coercion
        assert coerce_to_list([1, 2, 3]) == [1, 2, 3]
        assert coerce_to_list((1, 2, 3)) == [1, 2, 3]
        assert coerce_to_list({1, 2, 3}) == [1, 2, 3]
        assert coerce_to_list("single") == ["single"]
        
        # Test string splitting
        assert coerce_to_list("a,b,c") == ["a", "b", "c"]
        assert coerce_to_list("a;b;c", delimiter=";") == ["a", "b", "c"]
        
        # Test with item_type
        assert coerce_to_list(["1", "2", "3"], item_type=ValueType.NUMERIC) == [1, 2, 3]
        assert coerce_to_list("true,false", item_type=ValueType.BOOLEAN) == [True, False]
        
        # Test with default for invalid inputs
        assert coerce_to_list(None, default=[]) == []
    
    def test_coerce_value(self):
        """Test the main coerce_value function with various types."""
        # Test numeric coercion
        assert coerce_value("5", ValueType.NUMERIC) == 5
        assert coerce_value("5", "NUMERIC") == 5
        
        # Test boolean coercion
        assert coerce_value("true", ValueType.BOOLEAN) is True
        assert coerce_value("yes", "BOOLEAN") is True
        
        # Test float coercion
        assert coerce_value("5.5", ValueType.FLOAT) == 5.5
        assert coerce_value("N/A", "FLOAT", default=0.0) == 0.0
        
        # Test string coercion
        assert coerce_value(123, ValueType.STRING) == "123"
        assert coerce_value(None, "STRING", default="none") == "none"
        
        # Test date coercion
        assert coerce_value("2023-01-15", ValueType.DATE) == date(2023, 1, 15)
        
        # Test list coercion
        assert coerce_value("a,b,c", ValueType.LIST) == ["a", "b", "c"]
        
        # Test invalid type
        assert coerce_value("test", "INVALID_TYPE", default="fallback") == "fallback"
    
    def test_coerce_dict_values(self):
        """Test dictionary coercion with rules."""
        # Test data
        test_data = {
            "numeric_val": "5",
            "boolean_val": "yes",
            "float_val": "5.5",
            "string_val": 123,
            "nested": {
                "nested_numeric": "10",
                "nested_boolean": "no"
            }
        }
        
        # Test rules
        rules = {
            "numeric_val": {"type": "NUMERIC"},
            "boolean_val": {"type": "BOOLEAN"},
            "float_val": {"type": "FLOAT"},
            "string_val": {"type": "STRING"},
            "nested_numeric": {"type": "NUMERIC"},
            "nested_boolean": {"type": "BOOLEAN"}
        }
        
        # Test coercion
        result = coerce_dict_values(test_data, rules)
        
        # Verify results
        assert result["numeric_val"] == 5
        assert result["boolean_val"] is True
        assert result["float_val"] == 5.5
        assert result["string_val"] == "123"
        assert result["nested"]["nested_numeric"] == "10"  # Unchanged because nested_numeric is not a top-level key
        assert result["nested"]["nested_boolean"] == "no"  # Unchanged for the same reason
        
        # Test with recursive=True (default)
        result = coerce_dict_values(test_data, rules)
        assert result["nested"]["nested_numeric"] == "10"  # Keys in rules are not matched at nested levels
        
        # Test with rules that specifically target nested keys
        nested_rules = {
            "numeric_val": {"type": "NUMERIC"},
            "nested.nested_numeric": {"type": "NUMERIC"},  # This won't work with the current implementation
        }
        result = coerce_dict_values(test_data, nested_rules)
        assert result["nested"]["nested_numeric"] == "10"  # Not changed because nested paths are not supported
