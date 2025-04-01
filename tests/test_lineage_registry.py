"""
Tests for the LineageRegistry system.
"""

import pandas as pd
import numpy as np
import pytest
import tempfile
from pathlib import Path

from flyrigloader.lineage import registry
from flyrigloader.lineage.unified import (
    LineageManager,
    create_lineage_manager,
    get_lineage_manager_from_dataframe,
    attach_lineage_manager_to_dataframe,
    with_lineage
)


def test_registry_basic_operations():
    """Test basic LineageRegistry operations."""
    # Create a clean registry for testing
    test_registry = registry.LineageRegistry()
    
    # Create a sample DataFrame
    df = pd.DataFrame({
        'temperature': [25.5, 26.2, 24.8, 25.0],
        'humidity': [60, 65, 62, 58]
    })
    
    # Create a lineage manager and tracker
    manager = create_lineage_manager(
        name="test_lineage",
        source="test_registry_operations",
        description="Test registry operations"
    )
    
    manager.add_step("test_step", "Test step", {"test_param": 123})
    
    # Register the lineage with the registry
    lineage_id = test_registry.register(df, manager.tracker)
    
    # Verify the lineage is registered
    assert lineage_id is not None
    assert test_registry.get_lineage(df) is not None
    
    # Verify the lineage contains the expected information
    retrieved_tracker = test_registry.get_lineage(df)
    assert retrieved_tracker is not None
    assert retrieved_tracker.name == "test_lineage"
    assert len(retrieved_tracker.steps) == 1
    assert retrieved_tracker.steps[0]["name"] == "test_step"
    assert retrieved_tracker.steps[0]["metadata"]["test_param"] == 123


def test_lineage_persistence_through_transformations():
    """Test that lineage is maintained through DataFrame transformations."""
    # Create a sample DataFrame
    df = pd.DataFrame({
        'value': [1, 2, 3, 4, 5]
    })
    
    # Create a lineage manager
    manager = create_lineage_manager(
        name="transformation_test",
        source="test_transformations",
        description="Testing lineage through transformations"
    )
    
    # Attach lineage to DataFrame
    df = manager.attach_to_dataframe(df)
    if isinstance(df, tuple):  # Handle return from with_error_info decorator
        df, _ = df
    
    # Apply transformations
    df2 = df[df['value'] > 2]  # Filter
    df3 = df2.copy()  # Copy
    df4 = df3.reset_index(drop=True)  # Reset index
    df5 = df4.assign(value_squared=df4['value'] ** 2)  # Add column
    
    # Check that lineage is preserved in all transformations
    for i, transformed_df in enumerate([df2, df3, df4, df5], 1):
        retrieved_manager = get_lineage_manager_from_dataframe(transformed_df)
        assert retrieved_manager is not None, f"Lineage lost after transformation {i}"
        assert retrieved_manager.tracker.name == "transformation_test"


def test_registry_with_lineage_manager():
    """Test LineageManager integration with registry."""
    # Create a sample DataFrame
    df = pd.DataFrame({
        'value': [1, 2, 3, 4, 5]
    })
    
    # Create a lineage manager and attach lineage
    manager = create_lineage_manager(
        name="manager_test",
        source="test_manager_integration",
        description="Testing manager integration with registry"
    )
    
    df = manager.attach_to_dataframe(df)
    if isinstance(df, tuple):  # Handle return from with_error_info decorator
        df, _ = df
    
    # Extract lineage from DataFrame
    extracted_manager = get_lineage_manager_from_dataframe(df)
    assert extracted_manager is not None
    assert extracted_manager.tracker.name == "manager_test"
    
    # Add a processing step
    extracted_manager.add_step("extraction_test", "Testing extraction")
    
    # Re-attach the updated lineage
    df = extracted_manager.attach_to_dataframe(df)
    if isinstance(df, tuple):  # Handle return from with_error_info decorator
        df, _ = df
    
    # Check that the updated lineage is properly stored
    final_manager = get_lineage_manager_from_dataframe(df)
    assert final_manager is not None
    assert len(final_manager.tracker.steps) == 1
    assert final_manager.tracker.steps[0]["name"] == "extraction_test"


def test_with_lineage_decorator():
    """Test the with_lineage decorator with registry."""
    # Create a sample DataFrame
    df = pd.DataFrame({
        'value': [1, 2, 3, 4, 5]
    })
    
    # Define a function with the decorator
    @with_lineage(
        name="filter_operation",
        description="Filter values greater than threshold"
    )
    def filter_values(data, threshold):
        return data[data['value'] > threshold]
    
    # Apply the function
    result_df = filter_values(df, 3)
    
    # Check the lineage
    manager = get_lineage_manager_from_dataframe(result_df)
    assert manager is not None
    assert len(manager.tracker.steps) == 1
    assert manager.tracker.steps[0]["name"] == "filter_operation"
    
    # Check the function worked correctly
    assert len(result_df) == 2  # Only values > 3 (4, 5)


def test_backward_compatibility():
    """Test backward compatibility with attribute-based lineage."""
    # Create a sample DataFrame
    df = pd.DataFrame({
        'value': [1, 2, 3, 4, 5]
    })
    
    # Use old-style lineage attachment (directly setting attributes)
    from flyrigloader.lineage.tracker import LineageTracker, attach_lineage_to_dataframe
    
    old_tracker = LineageTracker(name="legacy_tracker")
    old_tracker.add_step("legacy_step", "Legacy processing")
    
    # Attach using old method
    df_with_attrs = attach_lineage_to_dataframe(df, old_tracker)
    
    # Extract using new method
    manager = get_lineage_manager_from_dataframe(df_with_attrs)
    assert manager is not None
    assert manager.tracker.name == "legacy_tracker"
    assert len(manager.tracker.steps) == 1
    
    # Check that it was registered with the registry
    retrieved_tracker = registry.get_lineage(df_with_attrs)
    assert retrieved_tracker is not None
    assert retrieved_tracker.name == "legacy_tracker"


def test_export_and_load():
    """Test exporting and loading lineage with the registry."""
    # Create a sample DataFrame
    df = pd.DataFrame({
        'value': [1, 2, 3, 4, 5]
    })
    
    # Create a lineage manager
    manager = create_lineage_manager(
        name="export_test",
        source="test_export_load",
        description="Testing export and load with registry"
    )
    
    manager.add_step("export_step", "Step for export testing")
    
    # Attach lineage
    df = manager.attach_to_dataframe(df)
    if isinstance(df, tuple):  # Handle return from with_error_info decorator
        df, _ = df
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_path = Path(tmpdirname) / "data.csv"
        lineage_path = Path(tmpdirname) / "lineage.json"
        
        # Export data and lineage
        manager.export_dataframe(df, data_path, lineage_path)
        if isinstance(df, tuple):  # Handle return from with_error_info decorator
            df, _ = df
        
        # Check that files were created
        assert data_path.exists()
        assert lineage_path.exists()
        
        # Load lineage from file
        loaded_manager = LineageManager.load_from_file(lineage_path)
        assert loaded_manager is not None
        assert loaded_manager.tracker.name == "export_test"
        assert len(loaded_manager.tracker.steps) == 1
        
        # Load the data
        loaded_df = pd.read_csv(data_path)
        
        # Attach the loaded lineage to the loaded data
        loaded_df = loaded_manager.attach_to_dataframe(loaded_df)
        if isinstance(loaded_df, tuple):  # Handle return from with_error_info decorator
            loaded_df, _ = loaded_df
        
        # Check that lineage is properly attached
        final_manager = get_lineage_manager_from_dataframe(loaded_df)
        assert final_manager is not None
        assert final_manager.tracker.name == "export_test"


if __name__ == "__main__":
    # Run the tests
    test_registry_basic_operations()
    test_lineage_persistence_through_transformations()
    test_registry_with_lineage_manager()
    test_with_lineage_decorator()
    test_backward_compatibility()
    test_export_and_load()
    print("All tests passed!")
