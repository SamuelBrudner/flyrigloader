"""
Tests for the data_lineage_integration module.
"""

import pandas as pd
import pytest
from flyrigloader.lineage import (
    track_lineage_step,
    with_lineage_tracking,
    complete_lineage_step,
    get_lineage_from_dataframe
)


def test_track_lineage_step_decorator():
    """Test the track_lineage_step decorator."""
    # Create a sample DataFrame
    df = pd.DataFrame({
        'temperature': [25.5, 26.2, 24.8, 25.0],
        'humidity': [60, 65, 62, 58]
    })
    
    # Define a function with the decorator
    @track_lineage_step("filter_data", "Filter rows by temperature", {"threshold": 25.0})
    def filter_by_temperature(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
        return data[data['temperature'] >= threshold]
    
    # Apply the function
    result_df = filter_by_temperature(df, 25.0)
    
    # Check that lineage is attached
    lineage = get_lineage_from_dataframe(result_df)
    assert lineage is not None
    assert len(lineage.steps) == 1
    assert lineage.steps[0]["name"] == "filter_data"
    assert lineage.steps[0]["metadata"]["threshold"] == 25.0
    assert lineage.steps[0]["metadata"]["status"] == "success"
    
    # Check that the filtering worked correctly
    assert len(result_df) == 3  # Only keeping temps >= 25.0


def test_track_lineage_step_error_handling():
    """Test error handling in track_lineage_step decorator."""
    # Create a sample DataFrame
    df = pd.DataFrame({
        'temperature': [25.5, 26.2, 24.8, 25.0],
        'humidity': [60, 65, 62, 58]
    })
    
    # Define a function with the decorator that will raise an error
    @track_lineage_step("problematic_operation", "This will fail")
    def operation_with_error(data: pd.DataFrame) -> pd.DataFrame:
        # Intentionally cause an error
        return data['non_existent_column'].to_frame()
    
    # Call the function and check that the error is recorded in lineage
    with pytest.raises(KeyError):
        operation_with_error(df)
    
    # The original DataFrame should still have lineage with error info
    lineage = get_lineage_from_dataframe(df)
    assert lineage is not None
    assert len(lineage.steps) == 1
    assert lineage.steps[0]["name"] == "problematic_operation"
    assert lineage.steps[0]["metadata"]["status"] == "error"
    assert "error_type" in lineage.steps[0]["metadata"]
    assert "error_message" in lineage.steps[0]["metadata"]


def test_manual_lineage_tracking():
    """Test the manual lineage tracking functions."""
    # Create a sample DataFrame
    df = pd.DataFrame({
        'value': [1, 2, 3, 4, 5]
    })
    
    # Start tracking a step
    df = with_lineage_tracking(df, "manual_step", "Manual operation")
    
    # Perform an operation
    df = df[df['value'] > 2]
    
    # Complete the step with metadata
    df = complete_lineage_step(df, "success", {"rows_removed": 2})
    
    # Verify the lineage
    lineage = get_lineage_from_dataframe(df)
    assert lineage is not None
    assert len(lineage.steps) == 1
    assert lineage.steps[0]["name"] == "manual_step"
    assert lineage.steps[0]["metadata"]["status"] == "success"
    assert lineage.steps[0]["metadata"]["rows_removed"] == 2
    
    # Verify the operation was applied correctly
    assert len(df) == 3  # Only values > 2 remain (3, 4, 5)


if __name__ == "__main__":
    # Run the tests
    test_track_lineage_step_decorator()
    test_track_lineage_step_error_handling()
    test_manual_lineage_tracking()
    print("All tests passed!")
