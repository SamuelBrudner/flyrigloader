"""
I/O module for flyrigloader with decoupled loading and transformation architecture.

This package contains modules for reading and writing experimental data in various formats,
including pickle files and other data formats used in fly behavior experiments.

The module provides a decoupled architecture separating data loading from transformation:
- pickle.py: Pure data loading operations (file discovery and raw data retrieval)
- transformers.py: Optional transformation utilities (DataFrame conversion, column filtering)

This separation enables the new manifest-based workflow where data discovery, loading, and
transformation are decoupled for better control over memory usage and processing pipelines.
Users can now choose to load raw data and apply transformations selectively, or use the
traditional integrated approach for backward compatibility.

Key transformation functions exported:
- make_dataframe_from_config: Enhanced DataFrame creation with configuration support
- handle_signal_disp: Signal display data processing with proper dimensionality
- extract_columns_from_matrix: Column extraction and processing utilities  
- ensure_1d_array: Array dimensionality transformation utilities
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
