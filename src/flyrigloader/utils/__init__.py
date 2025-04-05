"""
Utility functions for flyrigloader.

This package contains various utilities for working with file paths,
discovery results, and other common operations.
"""

from flyrigloader.utils.paths import (
    get_relative_path,
    get_absolute_path,
    find_common_base_directory,
    ensure_directory_exists
)

from flyrigloader.utils.dataframe import (
    build_manifest_df,
    filter_manifest_df,
    extract_unique_values
)

__all__ = [
    # Path utilities
    'get_relative_path',
    'get_absolute_path',
    'find_common_base_directory',
    'ensure_directory_exists',
    
    # DataFrame utilities
    'build_manifest_df',
    'filter_manifest_df',
    'extract_unique_values'
]
