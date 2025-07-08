"""
I/O module for flyrigloader.

This package contains modules for reading and writing experimental data in various formats,
including pickle files and other data formats used in fly behavior experiments.

The module has been refactored to provide better separation of concerns:
- pickle.py: Pure data loading operations
- transformers.py: Optional DataFrame utilities and transformations
"""

from flyrigloader.io.pickle import (
    read_pickle_any_format,
    load_experimental_data,
    PickleLoader,
    DependencyContainer,
    set_global_dependencies,
    reset_global_dependencies,
    create_test_pickle_loader
)

from flyrigloader.io.transformers import (
    DataFrameTransformer,
    make_dataframe_from_config,
    handle_signal_disp,
    extract_columns_from_matrix,
    ensure_1d_array,
    transform_to_dataframe,
    create_test_dataframe_transformer
)

__all__ = [
    # Core loading functions
    "read_pickle_any_format",
    "load_experimental_data",
    
    # Main classes
    "PickleLoader",
    "DataFrameTransformer",
    "DependencyContainer",
    
    # Transformation functions
    "make_dataframe_from_config",
    "handle_signal_disp", 
    "extract_columns_from_matrix",
    "ensure_1d_array",
    "transform_to_dataframe",
    
    # Test utilities
    "set_global_dependencies",
    "reset_global_dependencies",
    "create_test_pickle_loader",
    "create_test_dataframe_transformer"
]
