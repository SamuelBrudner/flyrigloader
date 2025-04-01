"""
Utilities for lineage tracking.

This package provides utility functions for working with lineage tracking,
including DataFrame handling, export/serialization, and compatibility with
legacy code.
"""

from .dataframe import (
    get_lineage_attribute,
    set_lineage_attribute,
    has_lineage,
    is_minimal_lineage,
    extract_lineage_dict,
    initialize_lineage_attributes,
    clean_lineage_attributes
)

from .export import (
    export_lineage_to_json,
    export_lineage_to_yaml,
    export_lineage_to_html,
    export_lineage_summary
)

__all__ = [
    # DataFrame utilities
    'get_lineage_attribute',
    'set_lineage_attribute',
    'has_lineage',
    'is_minimal_lineage',
    'extract_lineage_dict',
    'initialize_lineage_attributes',
    'clean_lineage_attributes',
    
    # Export utilities
    'export_lineage_to_json',
    'export_lineage_to_yaml',
    'export_lineage_to_html',
    'export_lineage_summary'
]
