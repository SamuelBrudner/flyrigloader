"""Modular configuration validator exports."""
from .base import ConfigValidationError, raise_validation_error
from .discovery import date_format_validator, pattern_validation
from .storage import (
    DEFAULT_SENSITIVE_ROOTS,
    PathSecurityPolicy,
    path_existence_validator,
    path_traversal_protection,
)
from .versioning import (
    validate_config_version,
    validate_config_with_version,
    validate_version_compatibility,
    validate_version_format,
)

__all__ = [
    "ConfigValidationError",
    "DEFAULT_SENSITIVE_ROOTS",
    "PathSecurityPolicy",
    "date_format_validator",
    "pattern_validation",
    "path_existence_validator",
    "path_traversal_protection",
    "raise_validation_error",
    "validate_config_version",
    "validate_config_with_version",
    "validate_version_compatibility",
    "validate_version_format",
]
