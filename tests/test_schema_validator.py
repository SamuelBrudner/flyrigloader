"""
Tests for the enhanced schema validation functionality.
"""

import pandas as pd
import pytest
import numpy as np
from pathlib import Path
import tempfile

from flyrigloader.schema.validator import (
    create_schema_from_dict,
    validate_dataframe,
    apply_schema,
    register_custom_validator,
    quick_validate
)


def test_basic_validation():
    """Test basic schema validation functionality."""
    # Create a sample DataFrame
    df = pd.DataFrame({
        'temperature': [25.5, 26.2, 24.8, 25.0],
        'humidity': [60, 65, 62, 58],
        'status': ['active', 'active', 'inactive', 'active']
    })
    
    # Define a schema
    schema_dict = {
        "column_mappings": {
            "data_columns": {
                "temperature": {
                    "dtype": "float64",
                    "description": "Temperature in Celsius",
                    "checks": [
                        {
                            "check_type": "in_range",
                            "min_value": 20.0,
                            "max_value": 30.0,
                            "error_message": "Temperature must be between 20°C and 30°C"
                        }
                    ]
                },
                "humidity": {
                    "dtype": "int64",
                    "description": "Relative humidity percentage",
                    "checks": [
                        {
                            "check_type": "in_range",
                            "min_value": 0,
                            "max_value": 100
                        }
                    ],
                    "error_templates": {
                        "in_range": "Humidity must be between {min_value}% and {max_value}%"
                    }
                },
                "status": {
                    "dtype": "string",
                    "description": "Current status",
                    "checks": [
                        {
                            "check_type": "isin",
                            "values": ["active", "inactive", "pending"]
                        }
                    ]
                }
            }
        },
        "strict": False
    }
    
    # Validate the DataFrame
    valid, errors = validate_dataframe(df, schema_dict)
    assert valid, f"Validation should pass but got errors: {errors}"


def test_failed_validation_with_enhanced_errors():
    """Test validation failures with enhanced error messages."""
    # Create a DataFrame with validation issues
    df = pd.DataFrame({
        'temperature': [15.0, 35.0, 25.0, None],  # Outside valid range and has null
        'humidity': [60, 120, 62, 58],            # One value exceeds 100%
        'status': ['active', 'unknown', 'inactive', 'pending']  # One invalid status
    })
    
    # Define a schema with detailed error messages
    schema_dict = {
        "column_mappings": {
            "data_columns": {
                "temperature": {
                    "dtype": "float64",
                    "nullable": False,
                    "description": "Temperature in Celsius",
                    "checks": [
                        {
                            "check_type": "in_range",
                            "min_value": 20.0,
                            "max_value": 30.0,
                            "error_message": "Temperature must be between 20°C and 30°C"
                        }
                    ]
                },
                "humidity": {
                    "dtype": "int64",
                    "description": "Relative humidity percentage",
                    "checks": [
                        {
                            "check_type": "in_range",
                            "min_value": 0,
                            "max_value": 100
                        }
                    ],
                    "error_templates": {
                        "in_range": "Humidity must be between {min_value}% and {max_value}%"
                    }
                },
                "status": {
                    "dtype": "string",
                    "description": "Current status",
                    "checks": [
                        {
                            "check_type": "isin",
                            "values": ["active", "inactive", "pending"]
                        }
                    ]
                }
            }
        },
        "strict": False
    }
    
    # Validate the DataFrame
    valid, errors = validate_dataframe(df, schema_dict)
    
    # Should fail with multiple errors
    assert not valid, "Validation should fail"
    assert len(errors) > 0, "Should have validation errors"
    
    # Print errors for debugging
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
    
    # Check for specific error messages
    temperature_error = any("Temperature must be between" in error for error in errors)
    humidity_error = any("Humidity must be between" in error for error in errors)
    status_error = any("status" in error and "unknown" in error for error in errors)
    nullable_error = any("nullable" in error for error in errors)
    
    assert temperature_error, "Should have temperature range error"
    assert humidity_error, "Should have humidity range error"
    assert status_error, "Should have status value error"
    assert nullable_error, "Should have nullable validation error"


def test_custom_validation_functions():
    """Test custom validation functions."""
    # Create a sample DataFrame
    df = pd.DataFrame({
        'start_date': pd.to_datetime(['2023-01-01', '2023-02-15', '2023-03-10']),
        'end_date': pd.to_datetime(['2023-01-15', '2023-02-10', '2023-04-01'])
    })
    
    # Define basic schema
    schema_dict = {
        "column_mappings": {
            "data_columns": {
                "start_date": "datetime64[ns]",
                "end_date": "datetime64[ns]"
            }
        }
    }
    
    # Create schema
    schema = create_schema_from_dict(schema_dict)
    
    # Define custom validation function that checks if end_date >= start_date
    def validate_date_range(df):
        return (df['end_date'] >= df['start_date']).all()
    
    # Add custom validator to the schema
    register_custom_validator(
        schema,
        name="date_range_check",
        validator_func=validate_date_range,
        columns=["start_date", "end_date"],
        error_message="End date must be on or after start date",
        description="Validates that date ranges are valid"
    )
    
    # Test with valid data
    valid, errors = validate_dataframe(df, schema)
    assert valid, f"Validation should pass but got errors: {errors}"
    
    # Test with invalid data - create a row where end_date < start_date
    invalid_df = df.copy()
    invalid_df.loc[1, 'end_date'] = pd.to_datetime('2023-02-01')  # Before start_date
    
    valid, errors = validate_dataframe(invalid_df, schema)
    assert not valid, "Validation should fail"
    assert any("End date must be on or after start date" in error for error in errors), \
        "Should contain custom validation error message"


def test_apply_schema_with_conversions():
    """Test applying a schema with type conversions."""
    # Create a DataFrame with types that need conversion
    df = pd.DataFrame({
        'id': ['1', '2', '3'],  # Strings that should be integers
        'value': ['10.5', '20.7', '15.2'],  # Strings that should be floats
        'flag': [1, 0, 1]  # Integers that should be booleans
    })
    
    # Define schema with target types
    schema_dict = {
        "column_mappings": {
            "data_columns": {
                "id": "int64",
                "value": "float64",
                "flag": "bool"
            }
        }
    }
    
    # Apply the schema
    converted_df = apply_schema(df, schema_dict)
    
    # Check that types were converted correctly
    assert pd.api.types.is_integer_dtype(converted_df['id'].dtype)
    assert pd.api.types.is_float_dtype(converted_df['value'].dtype)
    assert pd.api.types.is_bool_dtype(converted_df['flag'].dtype)
    
    # Check values
    assert converted_df['id'].tolist() == [1, 2, 3]
    assert converted_df['value'].tolist() == [10.5, 20.7, 15.2]
    assert converted_df['flag'].tolist() == [True, False, True]


def test_complex_custom_validation():
    """Test more complex custom validations across multiple columns."""
    # Create a sample DataFrame
    df = pd.DataFrame({
        'temperature': [25.5, 26.2, 24.8, 25.0],
        'humidity': [60, 65, 62, 58],
        'heat_index': [28.1, 29.3, 26.7, 27.2]
    })
    
    # Define basic schema
    schema_dict = {
        "column_mappings": {
            "data_columns": {
                "temperature": "float64",
                "humidity": "int64",
                "heat_index": "float64"
            }
        }
    }
    
    # Create schema
    schema = create_schema_from_dict(schema_dict)
    
    # Define custom validation function that checks if heat index is consistent
    # with temperature and humidity (simplified calculation)
    def validate_heat_index(df):
        # Simplified heat index calculation for validation
        # In reality, heat index is a complex formula
        for idx, row in df.iterrows():
            # Very basic approximation - real formula is more complex
            approx_heat_index = row['temperature'] + (row['humidity'] / 100) * 3
            actual_heat_index = row['heat_index']
            
            # Allow some tolerance
            if abs(approx_heat_index - actual_heat_index) > 5:
                return False
        return True
    
    # Add custom validator to the schema
    register_custom_validator(
        schema,
        name="heat_index_check",
        validator_func=validate_heat_index,
        columns=["temperature", "humidity", "heat_index"],
        error_message="Heat index is inconsistent with temperature and humidity",
        description="Validates heat index calculation"
    )
    
    # Test with valid data
    valid, errors = validate_dataframe(df, schema)
    assert valid, f"Validation should pass but got errors: {errors}"
    
    # Test with invalid data - create a row with inconsistent heat index
    invalid_df = df.copy()
    invalid_df.loc[1, 'heat_index'] = 45.0  # Unrealistically high for given temp/humidity
    
    valid, errors = validate_dataframe(invalid_df, schema)
    assert not valid, "Validation should fail"
    assert any("Heat index is inconsistent" in error for error in errors), \
        "Should contain custom validation error message"


def test_schema_with_multiple_custom_validators():
    """Test schema with multiple custom validators."""
    # Create a sample DataFrame with financial data
    df = pd.DataFrame({
        'income': [5000, 6000, 4500],
        'expenses': [3000, 4000, 2800],
        'savings': [2000, 2000, 1700],
        'month': ['January', 'February', 'March']
    })
    
    # Define basic schema
    schema_dict = {
        "column_mappings": {
            "data_columns": {
                "income": "float64",
                "expenses": "float64",
                "savings": "float64",
                "month": {
                    "dtype": "string",
                    "checks": [
                        {
                            "check_type": "isin",
                            "values": ["January", "February", "March", "April", 
                                      "May", "June", "July", "August", 
                                      "September", "October", "November", "December"]
                        }
                    ]
                }
            }
        }
    }
    
    # Create schema
    schema = create_schema_from_dict(schema_dict)
    
    # Validator 1: Check if income >= expenses
    def validate_income_expenses(df):
        return (df['income'] >= df['expenses']).all()
    
    # Validator 2: Check if savings = income - expenses
    def validate_savings_balance(df):
        # Allow small rounding errors
        epsilon = 0.01
        return ((df['income'] - df['expenses'] - df['savings']).abs() < epsilon).all()
    
    # Add validators to the schema
    register_custom_validator(
        schema,
        name="income_expense_check",
        validator_func=validate_income_expenses,
        columns=["income", "expenses"],
        error_message="Income must be greater than or equal to expenses",
        description="Validates income-expense relationship"
    )
    
    register_custom_validator(
        schema,
        name="savings_balance_check",
        validator_func=validate_savings_balance,
        columns=["income", "expenses", "savings"],
        error_message="Savings must equal income minus expenses",
        description="Validates financial balance"
    )
    
    # Test with valid data
    valid, errors = validate_dataframe(df, schema)
    assert valid, f"Validation should pass but got errors: {errors}"
    
    # Test with invalid data 1: expenses > income
    invalid_df1 = df.copy()
    invalid_df1.loc[1, 'expenses'] = 7000  # Higher than income
    
    valid1, errors1 = validate_dataframe(invalid_df1, schema)
    assert not valid1, "Validation should fail for invalid_df1"
    assert any("Income must be greater than or equal to expenses" in error for error in errors1), \
        "Should contain income-expense validation error"
    
    # Test with invalid data 2: savings != income - expenses
    invalid_df2 = df.copy()
    invalid_df2.loc[0, 'savings'] = 1000  # Incorrect savings amount
    
    valid2, errors2 = validate_dataframe(invalid_df2, schema)
    assert not valid2, "Validation should fail for invalid_df2"
    assert any("Savings must equal income minus expenses" in error for error in errors2), \
        "Should contain savings balance validation error"


if __name__ == "__main__":
    # Run the tests manually if executing this file directly
    test_basic_validation()
    test_failed_validation_with_enhanced_errors()
    test_custom_validation_functions()
    test_apply_schema_with_conversions()
    test_complex_custom_validation()
    test_schema_with_multiple_custom_validators()
    print("All tests passed!")
