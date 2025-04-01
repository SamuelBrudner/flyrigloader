"""
Core lineage tracking infrastructure.

This package provides the fundamental building blocks for lineage tracking,
including data structures, storage backends, and validation utilities.
These components are used by the higher-level API but can also be used
directly for advanced use cases.
"""

from .storage import (
    LineageStorage,
    AttributeStorage,
    RegistryStorage,
    NullStorage,
    create_storage,
    # Constants
    ATTR_MAIN_LINEAGE,
    ATTR_LINEAGE_IDS,
    ATTR_LINEAGES_DICT,
    DF_LINEAGE_ID_ATTR
)

from .tracker import (
    LineageRecord,
    create_lineage_record
)

from .validation import (
    validate_dataframe,
    ensure_dataframe,
    validate_lineage_storage_params,
    validate_lineage_complexity,
    verify_callable,
    verify_instance,
    verify_callable_signature
)

__all__ = [
    # Storage classes and utilities
    'LineageStorage',
    'AttributeStorage',
    'RegistryStorage',
    'NullStorage',
    'create_storage',
    
    # Storage constants
    'ATTR_MAIN_LINEAGE',
    'ATTR_LINEAGE_IDS',
    'ATTR_LINEAGES_DICT',
    'DF_LINEAGE_ID_ATTR',
    
    # Core data structures
    'LineageRecord',
    'create_lineage_record',
    
    # Validation utilities
    'validate_dataframe',
    'ensure_dataframe',
    'validate_lineage_storage_params',
    'validate_lineage_complexity',
    'verify_callable',
    'verify_instance',
    'verify_callable_signature'
]
