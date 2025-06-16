"""
Utility functions for flyrigloader.

This package contains various utilities for working with file paths,
discovery results, and other common operations with enhanced testability
support through dependency injection patterns and test hooks.

Enhanced Features for Testability (TST-REF-003):
- Test-specific entry points for comprehensive mocking scenarios
- Optional dependency provider configuration for behavior modification
- Pytest.monkeypatch-aware import patterns for isolation testing
- Enhanced error handling with improved logging for test observability
"""

import os
import sys
from typing import Any, Dict, Optional, Union, Callable, TypeVar

# Enhanced error handling with logging support for test observability
try:
    from flyrigloader import logger
except ImportError:
    # Fallback to standard logging for test environments without Loguru
    import logging
    
    class LoguruCompat:
        """Compatibility layer for environments without Loguru."""
        
        def __init__(self):
            self._logger = logging.getLogger(__name__)
            
        def debug(self, message: str) -> None:
            self._logger.debug(message)
            
        def info(self, message: str) -> None:
            self._logger.info(message)
            
        def warning(self, message: str) -> None:
            self._logger.warning(message)
            
        def error(self, message: str) -> None:
            self._logger.error(message)
    
    logger = LoguruCompat()

# Type variable for generic function types
F = TypeVar('F', bound=Callable[..., Any])

# Global dependency provider for test hook support (TST-REF-003)
_dependency_providers: Dict[str, Callable[..., Any]] = {}

# Test mode detection (disabled in production per TST-REF-003 validation rules)
def _is_test_environment() -> bool:
    """
    Detect if running in test environment for security isolation.
    
    Returns:
        True if in test environment, False in production
        
    Note:
        Test hooks are disabled in production per TST-REF-003 requirements
    """
    return (
        "pytest" in sys.modules or 
        "unittest" in sys.modules or
        os.environ.get("PYTEST_CURRENT_TEST") is not None or
        os.environ.get("TESTING") == "1"
    )

def register_test_provider(name: str, provider: Callable[..., Any]) -> None:
    """
    Register a test-specific dependency provider (TST-REF-003).
    
    This function allows test hooks to modify utility function behavior
    during test execution while maintaining production security isolation.
    
    Args:
        name: Provider identifier (e.g., 'paths.get_relative_path')
        provider: Callable that replaces the original function
        
    Raises:
        RuntimeError: If called outside test environment (production security)
        
    Example:
        # In test scenarios
        def mock_get_relative_path(path, base_dir):
            return Path("mocked/relative/path")
        register_test_provider('get_relative_path', mock_get_relative_path)
    """
    if not _is_test_environment():
        raise RuntimeError(
            "Test providers can only be registered in test environments. "
            "This is a security measure per TST-REF-003 validation rules."
        )
    
    logger.debug(f"Registering test provider for '{name}' in test environment")
    _dependency_providers[name] = provider

def clear_test_providers() -> None:
    """
    Clear all registered test providers.
    
    Useful for test teardown to ensure clean state between tests.
    Only available in test environments per TST-REF-003 security requirements.
    
    Raises:
        RuntimeError: If called outside test environment
    """
    if not _is_test_environment():
        raise RuntimeError(
            "Test providers can only be cleared in test environments. "
            "This is a security measure per TST-REF-003 validation rules."
        )
    
    logger.debug("Clearing all test providers")
    _dependency_providers.clear()

def get_test_provider(name: str) -> Optional[Callable[..., Any]]:
    """
    Retrieve a registered test provider if available.
    
    Args:
        name: Provider identifier
        
    Returns:
        Test provider function or None if not registered
        
    Note:
        Returns None in production environments for security
    """
    if not _is_test_environment():
        return None
    
    return _dependency_providers.get(name)

# Enhanced import patterns with testability-aware behavior
def _import_with_test_hooks(module_name: str, function_name: str, fallback_func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Import function with test hook support for pytest.monkeypatch scenarios.
    
    This enables controlled re-export behavior during test execution while
    maintaining production functionality and security isolation.
    
    Args:
        module_name: Source module name for logging
        function_name: Function identifier for test provider lookup
        fallback_func: Original function to use in production
        
    Returns:
        Test provider function if in test environment and registered,
        otherwise the original function
    """
    # Check for test provider override (only in test environments)
    test_provider = get_test_provider(function_name)
    if test_provider is not None:
        logger.debug(f"Using test provider for {module_name}.{function_name}")
        return test_provider
    
    return fallback_func

# Enhanced import handling with improved logging and error recovery
try:
    from flyrigloader.utils.dataframe import combine_metadata_and_data as _combine_metadata_and_data
    from flyrigloader.utils.paths import (
        get_relative_path as _get_relative_path,
        get_absolute_path as _get_absolute_path,
        find_common_base_directory as _find_common_base_directory,
        ensure_directory_exists as _ensure_directory_exists,
        ensure_directory as _ensure_directory
    )
    logger.debug("Successfully imported path utilities")
except ImportError as e:
    logger.error(f"Failed to import path utilities: {e}")
    raise ImportError(
        f"Required path utilities could not be imported: {e}. "
        "This may indicate missing dependencies or module structure issues."
    ) from e

try:
    from flyrigloader.utils.dataframe import (
        build_manifest_df as _build_manifest_df,
        filter_manifest_df as _filter_manifest_df,
        extract_unique_values as _extract_unique_values
    )
    logger.debug("Successfully imported dataframe utilities")
except ImportError as e:
    logger.error(f"Failed to import dataframe utilities: {e}")
    raise ImportError(
        f"Required dataframe utilities could not be imported: {e}. "
        "This may indicate missing dependencies (pandas, numpy) or module structure issues."
    ) from e

# Testability-aware function exports with dependency injection support
# These functions support pytest.monkeypatch scenarios through test providers

def get_relative_path(*args, **kwargs):
    """
    Get a path relative to a base directory with test hook support.
    
    Supports pytest.monkeypatch scenarios through registered test providers
    while maintaining production functionality and security isolation.
    """
    func = _import_with_test_hooks('paths', 'get_relative_path', _get_relative_path)
    return func(*args, **kwargs)

def get_absolute_path(*args, **kwargs):
    """
    Convert a relative path to an absolute path with test hook support.
    
    Supports pytest.monkeypatch scenarios through registered test providers
    while maintaining production functionality and security isolation.
    """
    func = _import_with_test_hooks('paths', 'get_absolute_path', _get_absolute_path)
    return func(*args, **kwargs)

def find_common_base_directory(*args, **kwargs):
    """
    Find the common base directory for a list of paths with test hook support.
    
    Supports pytest.monkeypatch scenarios through registered test providers
    while maintaining production functionality and security isolation.
    """
    func = _import_with_test_hooks('paths', 'find_common_base_directory', _find_common_base_directory)
    return func(*args, **kwargs)

def ensure_directory_exists(*args, **kwargs):
    """
    Ensure that a directory exists, creating it if necessary, with test hook support.
    
    Supports pytest.monkeypatch scenarios through registered test providers
    while maintaining production functionality and security isolation.
    """
    func = _import_with_test_hooks('paths', 'ensure_directory_exists', _ensure_directory_exists)
    return func(*args, **kwargs)


def ensure_directory(*args, **kwargs):
    """Ensure a directory exists using the public utility with test hook support."""
    func = _import_with_test_hooks('paths', 'ensure_directory', _ensure_directory)
    return func(*args, **kwargs)

def build_manifest_df(*args, **kwargs):
    """
    Convert discovery results to a pandas DataFrame with test hook support.
    
    Supports pytest.monkeypatch scenarios through registered test providers
    while maintaining production functionality and security isolation.
    """
    func = _import_with_test_hooks('dataframe', 'build_manifest_df', _build_manifest_df)
    return func(*args, **kwargs)

def filter_manifest_df(*args, **kwargs):
    """
    Filter a manifest DataFrame based on column values with test hook support.
    
    Supports pytest.monkeypatch scenarios through registered test providers
    while maintaining production functionality and security isolation.
    """
    func = _import_with_test_hooks('dataframe', 'filter_manifest_df', _filter_manifest_df)
    return func(*args, **kwargs)

def extract_unique_values(*args, **kwargs):
    """
    Extract unique values from a column in a manifest DataFrame with test hook support.
    
    Supports pytest.monkeypatch scenarios through registered test providers
    while maintaining production functionality and security isolation.
    """
    func = _import_with_test_hooks('dataframe', 'extract_unique_values', _extract_unique_values)
    return func(*args, **kwargs)

def combine_metadata_and_data(*args, **kwargs):
    """
    Combine metadata with data into a single dictionary with test hook support.
    
    This function merges metadata into the data dictionary with an optional prefix
    to avoid key collisions. It handles nested dictionaries and ensures that 
    metadata doesn't overwrite existing data keys.
    
    Supports pytest.monkeypatch scenarios through registered test providers
    while maintaining production functionality and security isolation.
    """
    func = _import_with_test_hooks('dataframe', 'combine_metadata_and_data', _combine_metadata_and_data)
    return func(*args, **kwargs)

# Enhanced __all__ with test utilities (available only in test environments)
_base_exports = [
    # Path utilities
    'get_relative_path',
    'get_absolute_path',
    'find_common_base_directory',
    'ensure_directory_exists',
    'ensure_directory',
    
    # DataFrame utilities
    'build_manifest_df',
    'filter_manifest_df',
    'extract_unique_values',
    'combine_metadata_and_data'
]

# Test-specific exports (TST-REF-003: only available in test environments)
_test_exports = [
    'register_test_provider',
    'clear_test_providers',
    'get_test_provider'
]

# Conditional export based on environment (production security isolation)
if _is_test_environment():
    __all__ = _base_exports + _test_exports
    logger.debug("Test environment detected: exposing test utilities")
else:
    __all__ = _base_exports
    logger.debug("Production environment detected: test utilities hidden")

# Module initialization logging
logger.info(f"flyrigloader.utils initialized with {len(__all__)} exported functions")
