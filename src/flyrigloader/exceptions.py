"""
FlyRigLoader Exception Hierarchy

This module provides a comprehensive domain-specific exception hierarchy for FlyRigLoader
that enables granular error handling, context preservation, and consistent logging across
the entire data loading pipeline.

The exception hierarchy follows a clear domain-based structure:
- FlyRigLoaderError: Base exception for all FlyRigLoader-specific errors
- ConfigError: Configuration validation and loading failures
- DiscoveryError: File discovery and manifest creation failures
- LoadError: Data loading and pickle format failures
- TransformError: Data transformation and schema validation failures

Each exception class provides:
- Context preservation utilities for exception chaining
- Error codes for programmatic error handling
- Structured logging integration
- Detailed error information for debugging

Usage Examples:
    Basic error handling:
    >>> try:
    ...     load_experiment_files(config)
    ... except ConfigError as e:
    ...     logger.error(f"Configuration error: {e}")
    ...     if e.error_code == "CONFIG_001":
    ...         # Handle missing configuration file
    
    Context preservation:
    >>> try:
    ...     process_data(raw_data)
    ... except ValueError as e:
    ...     raise TransformError("Data transformation failed").with_context({
    ...         "original_error": str(e),
    ...         "data_shape": raw_data.shape
    ...     })
    
    Error code checking:
    >>> try:
    ...     discover_files(pattern)
    ... except DiscoveryError as e:
    ...     if e.error_code == "DISCOVERY_003":
    ...         # Handle pattern compilation error
    ...         logger.warning("Invalid pattern, using default")
    ...         pattern = DEFAULT_PATTERN
"""

import sys
from typing import Any, Dict, Optional, Union
from pathlib import Path


class FlyRigLoaderError(Exception):
    """
    Base exception class for all FlyRigLoader-specific errors.
    
    This exception provides the foundation for the domain-specific error hierarchy,
    including context preservation utilities, error codes for programmatic handling,
    and integration with the logging system.
    
    Attributes:
        error_code (str): Unique identifier for programmatic error handling
        context (Dict[str, Any]): Additional context information for debugging
        
    Methods:
        with_context(context): Add additional context to the exception
        
    Error Codes:
        FLYRIG_001: Generic FlyRigLoader error
        FLYRIG_002: Unexpected internal error
        FLYRIG_003: Operation cancelled by user
        FLYRIG_004: System resource limitation
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "FLYRIG_001",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the FlyRigLoaderError with message, error code, and context.
        
        Args:
            message: Human-readable error description
            error_code: Unique identifier for programmatic error handling
            context: Additional context information for debugging
        """
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        
        # Preserve original exception information if this is a re-raise
        if hasattr(sys, '_getframe'):
            frame = sys._getframe(1)
            if frame:
                self.context.setdefault('source_file', frame.f_code.co_filename)
                self.context.setdefault('source_line', frame.f_lineno)
                self.context.setdefault('source_function', frame.f_code.co_name)
    
    def with_context(self, context: Dict[str, Any]) -> 'FlyRigLoaderError':
        """
        Add additional context to the exception and return self for chaining.
        
        This method enables context preservation when re-raising exceptions
        or adding debugging information at different levels of the call stack.
        
        Args:
            context: Dictionary of context information to add
            
        Returns:
            Self for method chaining
            
        Example:
            >>> raise FlyRigLoaderError("Operation failed").with_context({
            ...     "operation": "data_loading",
            ...     "file_path": "/path/to/file.pkl",
            ...     "attempt": 2
            ... })
        """
        self.context.update(context)
        return self
    
    def __str__(self) -> str:
        """
        Return a detailed string representation including context information.
        
        Returns:
            Formatted error message with context details
        """
        message = super().__str__()
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{message} [Error Code: {self.error_code}, Context: {context_str}]"
        return f"{message} [Error Code: {self.error_code}]"
    
    def __repr__(self) -> str:
        """Return a detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message={super().__str__()!r}, "
            f"error_code={self.error_code!r}, "
            f"context={self.context!r})"
        )


class ConfigError(FlyRigLoaderError):
    """
    Configuration validation and loading errors.
    
    This exception is raised when configuration files cannot be loaded,
    parsed, or validated. It covers YAML parsing errors, Pydantic validation
    failures, missing configuration files, and security validation issues.
    
    Error Codes:
        CONFIG_001: Configuration file not found
        CONFIG_002: YAML parsing error
        CONFIG_003: Pydantic validation failure
        CONFIG_004: Security validation failed (path traversal, etc.)
        CONFIG_005: Unsupported configuration schema version
        CONFIG_006: Environment variable validation failed
        CONFIG_007: Configuration builder validation failed
        CONFIG_008: Legacy configuration adapter error
        CONFIG_009: Configuration inheritance resolution failed
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "CONFIG_001",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize ConfigError with configuration-specific context.
        
        Args:
            message: Human-readable error description
            error_code: Configuration-specific error code
            context: Additional context including config paths, validation errors
        """
        super().__init__(message, error_code, context)
        
        # Add configuration-specific context defaults
        if context:
            if 'config_path' in context and isinstance(context['config_path'], (str, Path)):
                self.context['config_path'] = str(context['config_path'])
            if 'config_section' in context:
                self.context['config_section'] = context['config_section']
            if 'validation_errors' in context:
                self.context['validation_errors'] = context['validation_errors']


class DiscoveryError(FlyRigLoaderError):
    """
    File discovery and manifest creation errors.
    
    This exception is raised when file discovery operations fail, including
    pattern compilation errors, filesystem access issues, metadata extraction
    failures, and manifest generation problems.
    
    Error Codes:
        DISCOVERY_001: File system access denied
        DISCOVERY_002: Directory not found
        DISCOVERY_003: Pattern compilation error
        DISCOVERY_004: Metadata extraction failed
        DISCOVERY_005: File statistics collection failed
        DISCOVERY_006: Manifest generation failed
        DISCOVERY_007: Ignore pattern processing failed
        DISCOVERY_008: Mandatory substring validation failed
        DISCOVERY_009: File extension filtering failed
        DISCOVERY_010: Recursive discovery limit exceeded
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "DISCOVERY_001",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize DiscoveryError with discovery-specific context.
        
        Args:
            message: Human-readable error description
            error_code: Discovery-specific error code
            context: Additional context including paths, patterns, file counts
        """
        super().__init__(message, error_code, context)
        
        # Add discovery-specific context defaults
        if context:
            if 'search_path' in context and isinstance(context['search_path'], (str, Path)):
                self.context['search_path'] = str(context['search_path'])
            if 'pattern' in context:
                self.context['pattern'] = context['pattern']
            if 'files_processed' in context:
                self.context['files_processed'] = context['files_processed']
            if 'discovery_time' in context:
                self.context['discovery_time'] = context['discovery_time']


class LoadError(FlyRigLoaderError):
    """
    Data loading and pickle format errors.
    
    This exception is raised when data loading operations fail, including
    file format errors, I/O exceptions, registry lookup failures, and
    data corruption issues.
    
    Error Codes:
        LOAD_001: File not found or inaccessible
        LOAD_002: Unsupported file format
        LOAD_003: Pickle loading failed
        LOAD_004: Data corruption detected
        LOAD_005: Registry lookup failed
        LOAD_006: Compression format not supported
        LOAD_007: Memory allocation failed
        LOAD_008: File permissions denied
        LOAD_009: Loader registration failed
        LOAD_010: Format auto-detection failed
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "LOAD_001",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize LoadError with loading-specific context.
        
        Args:
            message: Human-readable error description
            error_code: Loading-specific error code
            context: Additional context including file paths, formats, sizes
        """
        super().__init__(message, error_code, context)
        
        # Add loading-specific context defaults
        if context:
            if 'file_path' in context and isinstance(context['file_path'], (str, Path)):
                self.context['file_path'] = str(context['file_path'])
            if 'file_format' in context:
                self.context['file_format'] = context['file_format']
            if 'file_size' in context:
                self.context['file_size'] = context['file_size']
            if 'loader_class' in context:
                self.context['loader_class'] = context['loader_class']
            if 'compression_type' in context:
                self.context['compression_type'] = context['compression_type']


class TransformError(FlyRigLoaderError):
    """
    Data transformation and schema validation errors.
    
    This exception is raised when data transformation operations fail,
    including schema validation errors, type conversion failures, DataFrame
    construction issues, and column mapping problems.
    
    Error Codes:
        TRANSFORM_001: Schema validation failed
        TRANSFORM_002: Type conversion error
        TRANSFORM_003: DataFrame construction failed
        TRANSFORM_004: Column mapping error
        TRANSFORM_005: Data dimension mismatch
        TRANSFORM_006: Missing required columns
        TRANSFORM_007: Invalid data format
        TRANSFORM_008: Transformation handler failed
        TRANSFORM_009: Metadata integration failed
        TRANSFORM_010: Memory limit exceeded during transformation
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "TRANSFORM_001",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize TransformError with transformation-specific context.
        
        Args:
            message: Human-readable error description
            error_code: Transformation-specific error code
            context: Additional context including data shapes, schemas, columns
        """
        super().__init__(message, error_code, context)
        
        # Add transformation-specific context defaults
        if context:
            if 'data_shape' in context:
                self.context['data_shape'] = context['data_shape']
            if 'schema_name' in context:
                self.context['schema_name'] = context['schema_name']
            if 'expected_columns' in context:
                self.context['expected_columns'] = context['expected_columns']
            if 'actual_columns' in context:
                self.context['actual_columns'] = context['actual_columns']
            if 'transformation_step' in context:
                self.context['transformation_step'] = context['transformation_step']
            if 'data_type' in context:
                self.context['data_type'] = context['data_type']


class RegistryError(FlyRigLoaderError):
    """
    Plugin registry and loader resolution errors.
    
    This exception is raised when plugin registration, discovery, or resolution
    operations fail, including plugin conflicts, entry-point discovery issues,
    thread-safety violations, and priority-based conflict detection problems.
    
    Error Codes:
        REGISTRY_001: Plugin registration failed
        REGISTRY_002: Loader not found for extension
        REGISTRY_003: Plugin conflict detected
        REGISTRY_004: Entry-point discovery failed
        REGISTRY_005: Plugin priority resolution failed
        REGISTRY_006: Thread-safety violation detected
        REGISTRY_007: Registry initialization failed
        REGISTRY_008: Plugin compatibility check failed
        REGISTRY_009: Auto-registration decorator failed
        REGISTRY_010: Registry lookup timeout
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "REGISTRY_001",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize RegistryError with registry-specific context.
        
        Args:
            message: Human-readable error description
            error_code: Registry-specific error code
            context: Additional context including plugin names, extensions, priorities
        """
        super().__init__(message, error_code, context)
        
        # Add registry-specific context defaults
        if context:
            if 'plugin_name' in context:
                self.context['plugin_name'] = context['plugin_name']
            if 'extension' in context:
                self.context['extension'] = context['extension']
            if 'loader_class' in context:
                self.context['loader_class'] = context['loader_class']
            if 'priority' in context:
                self.context['priority'] = context['priority']
            if 'conflicting_plugins' in context:
                self.context['conflicting_plugins'] = context['conflicting_plugins']
            if 'entry_point' in context:
                self.context['entry_point'] = context['entry_point']
            if 'registered_loaders' in context:
                self.context['registered_loaders'] = context['registered_loaders']


class VersionError(FlyRigLoaderError):
    """Configuration version errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "VERSION_001",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize VersionError with version-specific context.
        
        Args:
            message: Human-readable error description
            error_code: Version-specific error code
            context: Additional context including versions and compatibility details
        """
        super().__init__(message, error_code, context)
        
        # Add version-specific context defaults
        if context:
            if 'current_version' in context:
                self.context['current_version'] = context['current_version']
            if 'target_version' in context:
                self.context['target_version'] = context['target_version']
            if 'detected_version' in context:
                self.context['detected_version'] = context['detected_version']
            if 'supported_versions' in context:
                self.context['supported_versions'] = context['supported_versions']
            if 'config_schema' in context:
                self.context['config_schema'] = context['config_schema']


class KedroIntegrationError(FlyRigLoaderError):
    """
    Kedro integration and catalog configuration errors.
    
    This exception is raised when Kedro-specific operations fail, including
    AbstractDataset implementation issues, catalog configuration problems,
    data catalog integration failures, and Kedro pipeline compatibility issues.
    
    Error Codes:
        KEDRO_001: Kedro not installed or imported
        KEDRO_002: AbstractDataset implementation failed
        KEDRO_003: Catalog configuration invalid
        KEDRO_004: Dataset instantiation failed
        KEDRO_005: Data catalog integration failed
        KEDRO_006: Pipeline compatibility issue
        KEDRO_007: Kedro version compatibility failed
        KEDRO_008: Dataset serialization failed
        KEDRO_009: Catalog YAML parsing failed
        KEDRO_010: Kedro hook integration failed
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "KEDRO_001",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize KedroIntegrationError with Kedro-specific context.
        
        Args:
            message: Human-readable error description
            error_code: Kedro-specific error code
            context: Additional context including dataset info, catalog config, Kedro version
        """
        super().__init__(message, error_code, context)
        
        # Add Kedro-specific context defaults
        if context:
            if 'dataset_name' in context:
                self.context['dataset_name'] = context['dataset_name']
            if 'catalog_config' in context:
                self.context['catalog_config'] = context['catalog_config']
            if 'kedro_version' in context:
                self.context['kedro_version'] = context['kedro_version']
            if 'dataset_type' in context:
                self.context['dataset_type'] = context['dataset_type']
            if 'pipeline_name' in context:
                self.context['pipeline_name'] = context['pipeline_name']
            if 'catalog_filepath' in context and isinstance(context['catalog_filepath'], (str, Path)):
                self.context['catalog_filepath'] = str(context['catalog_filepath'])
            if 'serialization_format' in context:
                self.context['serialization_format'] = context['serialization_format']


# Legacy exception aliases for backward compatibility
# These are preserved during the transition period but will be deprecated
FileStatsError = DiscoveryError
PatternCompilationError = DiscoveryError

# Utility function for consistent error logging and raising
def log_and_raise(
    exception: FlyRigLoaderError,
    logger: Optional[Any] = None,
    level: str = "error"
) -> None:
    """
    Log an exception with context and then raise it.
    
    This utility function ensures consistent error logging across the system
    before exceptions are raised. It captures the exception context and logs
    it at the specified level.
    
    Args:
        exception: The exception to log and raise
        logger: Logger instance to use (optional)
        level: Log level ("error", "warning", "critical")
        
    Raises:
        The provided exception after logging
        
    Example:
        >>> from flyrigloader import logger
        >>> error = ConfigError("Invalid configuration")
        >>> log_and_raise(error, logger, "error")
    """
    if logger is not None:
        log_method = getattr(logger, level, logger.error)
        log_method(f"{exception.__class__.__name__}: {exception}")
        
        # Log context information if available
        if exception.context:
            for key, value in exception.context.items():
                log_method(f"  {key}: {value}")
    
    raise exception


# Export all exception classes for public use
__all__ = [
    'FlyRigLoaderError',
    'ConfigError',
    'DiscoveryError',
    'LoadError',
    'TransformError',
    'RegistryError',
    'VersionError',
    'KedroIntegrationError',
    'log_and_raise',
    # Legacy aliases (deprecated)
    'FileStatsError',
    'PatternCompilationError'
]