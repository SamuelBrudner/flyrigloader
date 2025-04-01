"""
Example script demonstrating the enhanced schema validation system.

This shows practical usage of custom validation functions and improved error messages
for validating experimental data in the flyrigloader project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from loguru import logger

from flyrigloader.schema.validator import (
    create_schema_from_dict,
    validate_dataframe,
    apply_schema,
    register_custom_validator
)


def main():
    # ===== Example 1: Basic schema validation with custom error messages =====
    logger.info("=== Example 1: Basic validation with custom error messages ===")
    
    # Sample experimental data
    exp_data = pd.DataFrame({
        'fly_id': ['fly001', 'fly002', 'fly003', 'fly004'],
        'temperature': [25.5, 26.2, 24.8, 25.0],
        'stimulus_type': ['light', 'odor', 'sound', 'heat'],
        'response_amplitude': [0.82, 0.67, 0.91, 0.75],
        'trial_number': [1, 2, 3, 4]
    })
    
    # Define a schema with custom error messages
    experiment_schema = {
        "column_mappings": {
            "data_columns": {
                "fly_id": {
                    "dtype": "string",
                    "description": "Unique identifier for the fly"
                },
                "temperature": {
                    "dtype": "float64",
                    "description": "Ambient temperature during experiment",
                    "checks": [
                        {
                            "check_type": "in_range",
                            "min_value": 20.0,
                            "max_value": 30.0,
                            "error_message": "Temperature outside acceptable range (20-30°C) for fly experiments"
                        }
                    ]
                },
                "stimulus_type": {
                    "dtype": "string",
                    "description": "Type of stimulus applied",
                    "checks": [
                        {
                            "check_type": "isin",
                            "values": ["light", "odor", "sound", "heat", "mechanical"],
                            "error_message": "Unknown stimulus type - must be one of: light, odor, sound, heat, mechanical"
                        }
                    ]
                },
                "response_amplitude": {
                    "dtype": "float64",
                    "description": "Normalized response amplitude",
                    "checks": [
                        {
                            "check_type": "in_range",
                            "min_value": 0.0,
                            "max_value": 1.0,
                            "error_message": "Response amplitude must be normalized between 0-1"
                        }
                    ]
                },
                "trial_number": {
                    "dtype": "int64",
                    "description": "Sequential trial number",
                    "checks": [
                        {
                            "check_type": "greater_than",
                            "min_value": 0
                        }
                    ]
                }
            }
        },
        "strict": True  # Enforce presence of all columns
    }
    
    # Validate the DataFrame
    valid, errors = validate_dataframe(exp_data, experiment_schema, "experiment_data")
    
    if valid:
        logger.success("Experiment data is valid!")
    else:
        logger.error("Validation failed with the following errors:")
        for error in errors:
            logger.error(f"  - {error}")
    
    # ===== Example 2: Creating an invalid DataFrame to show error handling =====
    logger.info("\n=== Example 2: Handling validation errors ===")
    
    # Create a DataFrame with validation issues
    invalid_data = pd.DataFrame({
        'fly_id': ['fly001', 'fly002', 'fly003', 'fly004'],
        'temperature': [25.5, 36.2, 24.8, 18.0],  # Two temperatures outside range
        'stimulus_type': ['light', 'unknown', 'sound', 'heat'],  # Invalid stimulus
        'response_amplitude': [0.82, 0.67, 1.5, 0.75],  # One value > 1
        'trial_number': [1, 2, 3, 4]
    })
    
    # Validate the DataFrame
    valid, errors = validate_dataframe(invalid_data, experiment_schema, "invalid_experiment_data")
    
    if valid:
        logger.success("Data is valid!")
    else:
        logger.error("Validation failed with the following errors:")
        for error in errors:
            logger.error(f"  - {error}")
        
        # Attempt to fix some errors automatically
        logger.info("Attempting to fix validation errors...")
        
        # Clamp temperature values to valid range
        invalid_data['temperature'] = invalid_data['temperature'].clip(20.0, 30.0)
        
        # Clamp response amplitude to valid range
        invalid_data['response_amplitude'] = invalid_data['response_amplitude'].clip(0.0, 1.0)
        
        # Replace invalid stimulus types
        invalid_data['stimulus_type'] = invalid_data['stimulus_type'].replace('unknown', 'light')
        
        # Re-validate
        valid, errors = validate_dataframe(invalid_data, experiment_schema, "fixed_experiment_data")
        
        if valid:
            logger.success("Fixed data is now valid!")
        else:
            logger.error("Could not fix all validation errors:")
            for error in errors:
                logger.error(f"  - {error}")
    
    # ===== Example 3: Custom validation functions =====
    logger.info("\n=== Example 3: Custom validation functions ===")
    
    # Sample time series data
    timeseries_data = pd.DataFrame({
        'time': pd.date_range(start='2023-01-01', periods=10, freq='1h'),
        'temperature': [25.0, 25.2, 25.3, 25.5, 25.7, 25.9, 26.1, 26.2, 26.0, 25.8],
        'stimulus_on': [False, False, True, True, True, True, False, False, False, False]
    })
    
    # Define basic schema
    timeseries_schema_dict = {
        "column_mappings": {
            "data_columns": {
                "time": "datetime64[ns]",
                "temperature": "float64",
                "stimulus_on": "bool"
            }
        }
    }
    
    # Create schema
    timeseries_schema = create_schema_from_dict(timeseries_schema_dict)
    
    # Define custom validation function 1: Check for monotonically increasing timestamps
    def validate_monotonic_time(df):
        return df['time'].is_monotonic_increasing
    
    # Define custom validation function 2: Check for realistic temperature changes
    def validate_temperature_gradient(df):
        # Calculate max temperature change between consecutive time points
        max_temp_change = np.abs(df['temperature'].diff().dropna()).max()
        # Ensure temperature doesn't change more than 0.5°C per hour
        return max_temp_change <= 0.5
    
    # Define custom validation function 3: Check stimulus on/off transitions
    def validate_stimulus_transitions(df):
        # Count number of transitions (changes from True to False or vice versa)
        transitions = (df['stimulus_on'] != df['stimulus_on'].shift()).sum()
        # In a valid experiment, expect at least 2 transitions (on and off)
        return transitions >= 2
    
    # Register custom validators
    register_custom_validator(
        timeseries_schema,
        name="monotonic_time_check",
        validator_func=validate_monotonic_time,
        columns=["time"],
        error_message="Time values must be strictly increasing",
        description="Validates that timestamps are in chronological order"
    )
    
    register_custom_validator(
        timeseries_schema,
        name="temperature_gradient_check",
        validator_func=validate_temperature_gradient,
        columns=["temperature"],
        error_message="Temperature changes too rapidly (> 0.5°C per hour)",
        description="Validates realistic temperature changes"
    )
    
    register_custom_validator(
        timeseries_schema,
        name="stimulus_transitions_check",
        validator_func=validate_stimulus_transitions,
        columns=["stimulus_on"],
        error_message="Experiment must include at least one complete stimulus on/off cycle",
        description="Validates stimulus protocol"
    )
    
    # Validate the DataFrame
    valid, errors = validate_dataframe(timeseries_data, timeseries_schema, "timeseries_data")
    
    if valid:
        logger.success("Time series data is valid!")
    else:
        logger.error("Validation failed with the following errors:")
        for error in errors:
            logger.error(f"  - {error}")
    
    # ===== Example 4: Invalid time series data =====
    logger.info("\n=== Example 4: Invalid time series with custom validation errors ===")
    
    # Create invalid time series
    invalid_timeseries = timeseries_data.copy()
    
    # Introduce errors:
    # 1. Add a non-monotonic timestamp
    invalid_timeseries.loc[5, 'time'] = pd.Timestamp('2023-01-01 03:30:00')  # Out of order
    
    # 2. Add unrealistic temperature jump
    invalid_timeseries.loc[7, 'temperature'] = 28.0  # Too large a jump
    
    # 3. Remove stimulus transitions (make all False)
    invalid_timeseries['stimulus_on'] = False
    
    # Validate
    valid, errors = validate_dataframe(invalid_timeseries, timeseries_schema, "invalid_timeseries")
    
    if valid:
        logger.success("Time series data is valid!")
    else:
        logger.error("Validation failed with the following errors:")
        for error in errors:
            logger.error(f"  - {error}")
        
        # Visualize the problematic data
        logger.info("Generating visualization of problematic time series data...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot original data
        ax1.plot(timeseries_data['time'], timeseries_data['temperature'], 'b-', label='Valid Data')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Temperature Data')
        ax1.grid(True)
        ax1.legend()
        
        # Plot invalid data
        ax2.plot(invalid_timeseries['time'], invalid_timeseries['temperature'], 'r-', label='Invalid Data')
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_xlabel('Time')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = Path("validation_examples")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "temperature_validation_issues.png")
        logger.info(f"Plot saved to {output_dir / 'temperature_validation_issues.png'}")


if __name__ == "__main__":
    main()
