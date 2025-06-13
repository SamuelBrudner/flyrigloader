"""
I/O module for flyrigloader.

This package contains modules for reading and writing experimental data in various formats,
including pickle files and other data formats used in fly behavior experiments.
"""

from flyrigloader.io.pickle import (
    read_pickle_any_format,
    load_experimental_data,
    PickleLoader,
    DataFrameTransformer,
    DependencyContainer,
    set_global_dependencies,
    reset_global_dependencies,
    create_test_pickle_loader,
    create_test_dataframe_transformer
)

__all__ = [
    # Core functions
    "read_pickle_any_format",
    "load_experimental_data",
    
    # Main classes
    "PickleLoader",
    "DataFrameTransformer",
    "DependencyContainer",
    
    # Test utilities
    "set_global_dependencies",
    "reset_global_dependencies",
    "create_test_pickle_loader",
    "create_test_dataframe_transformer"
]
