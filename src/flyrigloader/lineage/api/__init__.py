"""
User-facing API for lineage tracking.

This package provides the main interfaces for lineage tracking in the
flyrigloader package, including both standard and minimal APIs, as well
as decorators for automatic tracking.
"""

from .standard import (
    LineageTracker,
    create_tracker
)

from .minimal import (
    MinimalLineageTracker,
    create_minimal_tracker
)

from .decorators import (
    track_lineage,
    track_minimal_lineage,
    propagate_lineage
)

__all__ = [
    # Standard API
    'LineageTracker',
    'create_tracker',
    
    # Minimal API
    'MinimalLineageTracker',
    'create_minimal_tracker',
    
    # Decorators
    'track_lineage',
    'track_minimal_lineage',
    'propagate_lineage'
]
