"""
lineage package - Tools for tracking data lineage and provenance.

This package provides functionality for creating, tracking, and exporting
lineage information throughout the data processing pipeline, enabling data
provenance and reproducibility.

The package is organized into several layers:
- api: User-facing interfaces (standard and minimal)
- core: Fundamental infrastructure (storage, data structures)
- utils: Helper utilities for working with lineage data

For most use cases, use the classes and functions directly from this module.
Advanced users may import from the submodules for more control.
"""

import warnings
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar

import pandas as pd

# ===== Import new architecture components =====

# Import from api layer (standard interface)
from .api.standard import (
    LineageTracker,
    create_tracker
)

# Import from api layer (minimal interface)
from .api.minimal import (
    MinimalLineageTracker,
    create_minimal_tracker,
    NullLineageTracker
)

# Import from api layer (decorators)
from .api.decorators import (
    track_lineage,
    track_minimal_lineage,
    propagate_lineage,
    start_tracking,
    complete_tracking
)

# Import from api layer (registry)
from .api.registry import (
    LineageRegistry,
    register_lineage,
    get_lineage,
    update_lineage,
    merge_lineages,
    remove_lineage,
    save_registry,
    load_registry
)

# Import from core layer
from .core.tracker import (
    LineageRecord,
    create_lineage_record
)

# Import from utils layer
from .utils.dataframe import (
    has_lineage,
    is_minimal_lineage,
    extract_lineage_dict
)

from .utils.export import (
    export_lineage_to_json,
    export_lineage_to_yaml,
    export_lineage_to_html,
    export_lineage_summary
)

# Initialize the global registry instance
registry = LineageRegistry.get_instance()

# ===== Define public API =====

# Define what's available when doing 'from flyrigloader.lineage import *'
__all__ = [
    # Core components
    'LineageRecord',  
    'create_lineage_record',
    
    # Standard API
    'LineageTracker',
    'create_tracker',
    
    # Minimal API
    'MinimalLineageTracker',
    'create_minimal_tracker',
    'NullLineageTracker',
    
    # Decorators
    'track_lineage',
    'track_minimal_lineage',
    'propagate_lineage',
    'start_tracking',
    'complete_tracking',
    
    # Registry
    'LineageRegistry',
    'registry',
    'register_lineage',
    'get_lineage',
    'update_lineage', 
    'merge_lineages',
    'remove_lineage',
    'save_registry',
    'load_registry',
    
    # Utility functions
    'has_lineage',
    'is_minimal_lineage',
    'extract_lineage_dict',
    'export_lineage_to_json',
    'export_lineage_to_yaml',
    'export_lineage_to_html',
    'export_lineage_summary'
]
