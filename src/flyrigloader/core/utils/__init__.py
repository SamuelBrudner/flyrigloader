"""
Core utilities module for flyrigloader.

This module contains foundational utility functions with minimal dependencies
that are used throughout the rest of the codebase.

Error Handling Conventions:
---------------------------
The flyrigloader package uses two complementary error handling patterns:

1. Tuple Returns with Metadata - For data processing functions:
   - Returns (result, metadata) tuple 
   - metadata always includes a "success" key
   - Used in pipeline and data assembly functions

2. Exception Raising - For utilities and low-level functions:
   - Raises appropriate exceptions with descriptive messages
   - Logs errors before raising
   - Used in utility functions and core operations

See the error_handling.md documentation for details.
"""

# Import commonly used utilities for convenience
from .path_utils import (
    PathLike, ensure_path, ensure_path_exists, get_absolute_path,
    normalize_path, get_relative_path, split_path, join_paths,
    change_extension, get_related_path
)

from .file_utils import (
    safe_load_yaml, ensure_file_exists, ensure_directory,
    check_and_log, list_directory_with_pattern
)

# Import dependency management system
# This provides thread-safe implementations and explicit dependency injection
from .dependency import (
    DependencyContainer,
    ThreadSafeDependencyManager,
    dependencies,
    container
)

# Import dictionary utilities for consistent dictionary operations
from .dict_utils import (
    deep_update, deep_merge, get_nested_value, set_nested_value, filter_dict
)

# Import environment variable utilities
from .env_utils import (
    get_env_variable, get_env_bool, get_env_int, get_env_float,
    get_env_list, get_env_dict, get_env_or_config
)

# Import value coercion utilities
from .value_utils import (
    ValueType, coerce_value, coerce_to_numeric, coerce_to_boolean,
    coerce_to_float, coerce_to_string, coerce_to_date, coerce_to_list,
    coerce_dict_values, COMMON_VALUES
)

# Import time utilities
from .time_utils import (
    get_current_timestamp, format_timestamp, parse_timestamp,
    get_timestamp_difference, validate_frequency
)

# Import schema utilities
from .schema_utils import (
    map_type_string_to_pandas, apply_column_operations, 
    INT_TYPES, FLOAT_TYPES, STR_TYPES, BOOL_TYPES, LIST_TYPES, DATE_TYPES, CATEGORY_TYPES,
    is_type_in_category, extract_column_info, create_empty_column, convert_column,
    coerce_to_list, ensure_1d, get_column_definitions
)

# Import type checking utilities
from .typing import (
    is_numeric, is_string, is_bool, is_dict, is_list, is_set,
    is_tuple, is_none, is_path, is_date, is_datetime
)

# Import error handling utilities
from .error_utils import (
    create_metadata,
    create_error_metadata,
    update_metadata
)

__all__ = [
    # Path utilities
    'PathLike', 'ensure_path', 'ensure_path_exists', 'get_absolute_path',
    'normalize_path', 'get_relative_path', 'split_path', 'join_paths',
    'change_extension', 'get_related_path',
    
    # File utilities
    'safe_load_yaml', 'ensure_file_exists', 'ensure_directory',
    'check_and_log', 'list_directory_with_pattern',
    
    # Dependency management
    'DependencyContainer', 'ThreadSafeDependencyManager', 'dependencies', 'container',
    
    # Dictionary utilities
    'deep_update', 'deep_merge', 'get_nested_value', 'set_nested_value',
    'filter_dict',
    
    # Environment variable utilities
    'get_env_variable', 'get_env_bool', 'get_env_int', 'get_env_float',
    'get_env_list', 'get_env_dict', 'get_env_or_config',
    
    # Value coercion utilities
    'ValueType', 'coerce_value', 'coerce_to_numeric', 'coerce_to_boolean',
    'coerce_to_float', 'coerce_to_string', 'coerce_to_date', 'coerce_to_list',
    'coerce_dict_values', 'COMMON_VALUES',
    
    # Time utilities
    'get_current_timestamp', 'format_timestamp', 'parse_timestamp',
    'get_timestamp_difference', 'validate_frequency',
    
    # Schema utilities
    'map_type_string_to_pandas', 'apply_column_operations', 
    'INT_TYPES', 'FLOAT_TYPES', 'STR_TYPES', 'BOOL_TYPES', 'LIST_TYPES', 'DATE_TYPES', 'CATEGORY_TYPES',
    'is_type_in_category', 'extract_column_info', 'create_empty_column', 'convert_column',
    'coerce_to_list', 'ensure_1d', 'get_column_definitions',
    
    # Type checking utilities
    'is_numeric', 'is_string', 'is_bool', 'is_dict', 'is_list', 'is_set',
    'is_tuple', 'is_none', 'is_path', 'is_date', 'is_datetime',
    
    # Error handling utilities
    'create_metadata',
    'create_error_metadata',
    'update_metadata'
]
