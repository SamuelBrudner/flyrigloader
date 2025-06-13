"""
Column configuration models using Pydantic for validation.

This module provides Pydantic models for defining and validating
experimental data column configurations with enhanced testability features.

Enhanced for comprehensive testing through:
- Dependency injection patterns for external libraries (PyYAML, Loguru)
- Configurable validation behavior supporting pytest.monkeypatch scenarios
- Modular function decomposition for granular unit testing
- Test-specific entry points and hooks for controlled test execution
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any, Literal, Callable, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Pydantic imports with v2 compatibility (F-020)
from pydantic import BaseModel, Field, field_validator, model_validator

import numpy as np
import os

# Dependency injection interfaces for testability (TST-REF-001)
class YamlLoaderProtocol(Protocol):
    """Protocol for YAML loading dependencies to enable comprehensive mocking."""
    
    def safe_load(self, stream) -> Any:
        """Load YAML content safely from a stream."""
        ...

class LoggerProtocol(Protocol):
    """Protocol for logging dependencies to enable test isolation."""
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        ...
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        ...
    
    def info(self, message: str) -> None:
        """Log info message."""
        ...
    
    def error(self, message: str) -> None:
        """Log error message."""
        ...

class FileSystemProtocol(Protocol):
    """Protocol for file system operations to enable testing isolation."""
    
    def open(self, path: str, mode: str = 'r'):
        """Open file with specified mode."""
        ...
    
    def exists(self, path: str) -> bool:
        """Check if file exists."""
        ...
    
    def dirname(self, path: str) -> str:
        """Get directory name of path."""
        ...
    
    def abspath(self, path: str) -> str:
        """Get absolute path."""
        ...
    
    def join(self, *paths: str) -> str:
        """Join path components."""
        ...

# Default implementations (TST-REF-001)
class DefaultYamlLoader:
    """Default YAML loader implementation using PyYAML."""
    
    def safe_load(self, stream) -> Any:
        import yaml
        return yaml.safe_load(stream)

class DefaultLogger:
    """Default logger implementation using Loguru."""
    
    def __init__(self):
        from flyrigloader import logger
        self._logger = logger
    
    def debug(self, message: str) -> None:
        self._logger.debug(message)
    
    def warning(self, message: str) -> None:
        self._logger.warning(message)
    
    def info(self, message: str) -> None:
        self._logger.info(message)
    
    def error(self, message: str) -> None:
        self._logger.error(message)

class DefaultFileSystem:
    """Default file system implementation."""
    
    def open(self, path: str, mode: str = 'r'):
        return open(path, mode)
    
    def exists(self, path: str) -> bool:
        return os.path.exists(path)
    
    def dirname(self, path: str) -> str:
        return os.path.dirname(path)
    
    def abspath(self, path: str) -> str:
        return os.path.abspath(path)
    
    def join(self, *paths: str) -> str:
        return os.path.join(*paths)

# Configurable dependency container (TST-REF-001, TST-REF-003)
@dataclass
class DependencyContainer:
    """
    Dependency injection container for configurable testing scenarios.
    
    Supports pytest.monkeypatch scenarios by allowing complete dependency
    replacement during test execution per TST-REF-001 requirements.
    """
    yaml_loader: YamlLoaderProtocol = field(default_factory=DefaultYamlLoader)
    logger: LoggerProtocol = field(default_factory=DefaultLogger)
    file_system: FileSystemProtocol = field(default_factory=DefaultFileSystem)
    
    # Test-specific hooks (TST-REF-003)
    validation_hooks: Dict[str, Callable] = field(default_factory=dict)
    test_mode: bool = False
    
    def register_validation_hook(self, hook_name: str, hook_func: Callable) -> None:
        """Register test-specific validation hook for controlled test execution."""
        self.validation_hooks[hook_name] = hook_func
        
    def execute_validation_hook(self, hook_name: str, *args, **kwargs) -> Any:
        """Execute registered validation hook if available."""
        if hook_name in self.validation_hooks:
            return self.validation_hooks[hook_name](*args, **kwargs)
        return None

# Global dependency container with test-specific entry points (TST-REF-003)
_dependency_container = DependencyContainer()

def get_dependency_container() -> DependencyContainer:
    """Get the current dependency container (test hook available)."""
    return _dependency_container

def set_dependency_container(container: DependencyContainer) -> None:
    """Set dependency container for test scenarios (TST-REF-003)."""
    global _dependency_container
    _dependency_container = container

def reset_dependency_container() -> None:
    """Reset to default dependencies (test cleanup hook)."""
    global _dependency_container
    _dependency_container = DependencyContainer()

# Enhanced validation configuration (F-016)
@dataclass 
class ValidationConfig:
    """
    Configuration for validation behavior supporting comprehensive mock scenarios.
    
    Enables configurable validation behavior for pytest.monkeypatch testing
    per Feature F-016 requirements.
    """
    strict_mode: bool = True
    log_validation_steps: bool = False
    enable_dimension_warnings: bool = True
    enable_handler_warnings: bool = True
    custom_validators: Dict[str, Callable] = field(default_factory=dict)
    
    def register_custom_validator(self, field_name: str, validator: Callable) -> None:
        """Register custom validator for specific field (test hook)."""
        self.custom_validators[field_name] = validator


class ColumnDimension(int, Enum):
    """Enumeration of supported array dimensions."""
    ONE_D = 1
    TWO_D = 2
    THREE_D = 3


class SpecialHandlerType(str, Enum):
    """Types of special handlers for column data processing."""
    EXTRACT_FIRST_COLUMN = "extract_first_column_if_2d"
    TRANSFORM_TIME_DIMENSION = "transform_to_match_time_dimension"


class ColumnConfig(BaseModel):
    """
    Configuration for a single column in experimental data.
    
    Enhanced with configurable validation behavior and dependency injection
    support for comprehensive pytest.monkeypatch scenarios.
    
    Attributes:
        type: The expected data type (e.g., "numpy.ndarray", "str", etc.)
        dimension: For arrays, the expected dimensionality (1D, 2D, etc.)
        required: Whether the column is required in the dataset
        description: Human-readable description of the column
        alias: Optional alternative name for the column in the source data
        is_metadata: Whether the column is metadata (not part of raw data)
        default_value: Optional default value for missing columns
        special_handling: Optional special processing needed for this column
    """
    type: str
    dimension: Optional[ColumnDimension] = None
    required: bool = False
    description: str
    alias: Optional[str] = None
    is_metadata: bool = False
    default_value: Optional[Any] = None
    special_handling: Optional[SpecialHandlerType] = None
    
    @field_validator('dimension', mode='before')
    @classmethod
    def validate_dimension(cls, v, info=None):
        """
        Convert integer dimension to enum with enhanced logging for test observability.
        
        Enhanced per Section 2.2.8 requirements with improved error handling
        and logging for better test assertion capabilities.
        """
        # Get dependency container for logging (TST-REF-001)
        deps = get_dependency_container()
        
        # Execute validation hook if registered (TST-REF-003)
        hook_result = deps.execute_validation_hook('dimension_validation', v, info)
        if hook_result is not None:
            return hook_result
            
        if v is None:
            deps.logger.debug("Dimension validation: None value passed, returning None")
            return None
            
        if isinstance(v, int):
            try:
                result = ColumnDimension(v)
                deps.logger.debug(f"Dimension validation: Successfully converted {v} to {result}")
                return result
            except ValueError as e:
                error_msg = f"Dimension must be 1, 2, or 3, got {v}"
                deps.logger.error(f"Dimension validation failed: {error_msg}")
                raise ValueError(error_msg) from e
        
        deps.logger.debug(f"Dimension validation: Returning unchanged value {v}")
        return v
    
    @field_validator('special_handling', mode='before')
    @classmethod
    def validate_special_handling(cls, v, info=None):
        """
        Convert string handler types to enum with enhanced error handling.
        
        Enhanced per Section 2.2.8 requirements with improved logging
        for test observability and assertion capabilities.
        """
        # Get dependency container for logging (TST-REF-001)
        deps = get_dependency_container()
        
        # Execute validation hook if registered (TST-REF-003)
        hook_result = deps.execute_validation_hook('special_handling_validation', v, info)
        if hook_result is not None:
            return hook_result
            
        if v is None:
            deps.logger.debug("Special handling validation: None value passed, returning None")
            return None
            
        if isinstance(v, str):
            try:
                result = SpecialHandlerType(v)
                deps.logger.debug(f"Special handling validation: Successfully converted '{v}' to {result}")
                return result
            except ValueError as e:
                valid_handlers = [h.value for h in SpecialHandlerType]
                error_msg = f"Special handler must be one of {valid_handlers}, got {v}"
                deps.logger.error(f"Special handling validation failed: {error_msg}")
                raise ValueError(error_msg) from e
        
        deps.logger.debug(f"Special handling validation: Returning unchanged value {v}")
        return v
    
    @model_validator(mode='after')
    def validate_configuration(self):
        """
        Validate the overall configuration for consistency with enhanced logging.
        
        Enhanced per Section 2.2.8 requirements with comprehensive logging
        for improved test observability and validation tracking.
        """
        # Get dependency container for logging (TST-REF-001)
        deps = get_dependency_container()
        
        # Execute validation hook if registered (TST-REF-003)
        hook_result = deps.execute_validation_hook('model_validation', self)
        if hook_result is not None:
            return hook_result
        
        values = self.model_dump()
        deps.logger.debug(f"Model validation starting for column configuration: {values.get('description', 'unnamed')}")
        
        # If we have a dimension, ensure type is compatible
        if values.get('dimension') and not values.get('type', '').startswith('numpy'):
            warning_msg = f"Dimension specified for non-numpy type {values.get('type')}"
            deps.logger.warning(warning_msg)
        
        # If special handling is for transforming time dimension, ensure we have a dimension
        if (values.get('special_handling') == SpecialHandlerType.TRANSFORM_TIME_DIMENSION and 
                values.get('dimension') != ColumnDimension.TWO_D):
            warning_msg = "transform_to_match_time_dimension should be used with 2D arrays"
            deps.logger.warning(warning_msg)
        
        deps.logger.debug(f"Model validation completed successfully for column: {values.get('description', 'unnamed')}")
        return self
        

class ColumnConfigDict(BaseModel):
    """
    Configuration for all columns in experimental data.
    
    Enhanced with dependency injection and configurable validation behavior
    for comprehensive pytest.monkeypatch testing scenarios.
    
    This model allows validation of the entire column configuration
    dictionary to ensure it's properly structured with enhanced observability.
    """
    columns: Dict[str, ColumnConfig]
    special_handlers: Dict[str, str] = Field(default_factory=dict)
    
    @field_validator('special_handlers')
    @classmethod
    def validate_special_handlers(cls, v, info):
        """
        Ensure all referenced handlers have an implementation with enhanced logging.
        
        Enhanced per Section 2.2.8 requirements with comprehensive error handling
        and logging for improved test observability.
        """
        # Get dependency container for logging (TST-REF-001)
        deps = get_dependency_container()
        
        # Execute validation hook if registered (TST-REF-003)
        hook_result = deps.execute_validation_hook('special_handlers_validation', v, info)
        if hook_result is not None:
            return hook_result
        
        # Get values from context
        values = info.data
        
        deps.logger.debug(f"Special handlers validation starting with {len(v)} handlers defined")
        
        # Get all special handler types being used as a set comprehension
        required_handlers = {
            col.special_handling.value 
            for col in values.get('columns', {}).values() 
            if col.special_handling
        }
        
        deps.logger.debug(f"Found {len(required_handlers)} required handlers: {list(required_handlers)}")
        
        # Check that all required handlers are defined
        missing_handlers = []
        for handler in required_handlers:
            if handler not in v:
                warning_msg = f"Special handler '{handler}' is used but not defined in special_handlers"
                deps.logger.warning(warning_msg)
                missing_handlers.append(handler)
        
        if missing_handlers:
            deps.logger.warning(f"Missing handler implementations: {missing_handlers}")
        else:
            deps.logger.debug("All required special handlers are properly defined")
        
        deps.logger.debug("Special handlers validation completed successfully")
        return v


# Constants for default configuration
DEFAULT_CONFIG_FILENAME = "column_config.yaml"

# Modular function decomposition for enhanced unit test granularity (TST-REF-002)

def get_default_config_path(dependencies: Optional[DependencyContainer] = None) -> str:
    """
    Get the path to the default column configuration file.
    
    Enhanced with dependency injection for comprehensive testing isolation
    per TST-REF-001 requirements.
    
    Args:
        dependencies: Optional dependency container for testing scenarios
        
    Returns:
        str: Absolute path to the default configuration file.
    """
    deps = dependencies or get_dependency_container()
    
    # Get the directory where this file is located
    current_dir = deps.file_system.dirname(deps.file_system.abspath(__file__))
    # Return the path to the default config file in the same directory
    config_path = deps.file_system.join(current_dir, DEFAULT_CONFIG_FILENAME)
    
    deps.logger.debug(f"Default config path resolved to: {config_path}")
    return config_path


def _load_yaml_content(file_path: str, dependencies: Optional[DependencyContainer] = None) -> Dict[str, Any]:
    """
    Load YAML content from file with dependency injection support.
    
    Modular function for YAML loading per TST-REF-002 requirements,
    enabling isolated unit testing of YAML loading functionality.
    
    Args:
        file_path: Path to YAML file
        dependencies: Optional dependency container for testing scenarios
        
    Returns:
        Dict containing parsed YAML content
        
    Raises:
        FileNotFoundError: If the file does not exist
        yaml.YAMLError: If YAML parsing fails
    """
    deps = dependencies or get_dependency_container()
    
    deps.logger.debug(f"Loading YAML content from {file_path}")
    
    # Check file existence first for better error handling (Section 2.2.8)
    if not deps.file_system.exists(file_path):
        error_msg = f"Configuration file not found: {file_path}"
        deps.logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        with deps.file_system.open(file_path, 'r') as f:
            config_data = deps.yaml_loader.safe_load(f)
        
        deps.logger.debug(f"Successfully loaded YAML data with {len(config_data) if config_data else 0} top-level keys")
        return config_data
    
    except Exception as e:
        error_msg = f"Failed to load YAML from {file_path}: {str(e)}"
        deps.logger.error(error_msg)
        raise


def _validate_config_data(config_data: Dict[str, Any], dependencies: Optional[DependencyContainer] = None) -> ColumnConfigDict:
    """
    Validate configuration data using Pydantic model.
    
    Modular function for configuration validation per TST-REF-002 requirements,
    enabling isolated unit testing of validation functionality.
    
    Args:
        config_data: Dictionary containing configuration data
        dependencies: Optional dependency container for testing scenarios
        
    Returns:
        ColumnConfigDict: Validated configuration model
        
    Raises:
        ValidationError: If configuration validation fails
    """
    deps = dependencies or get_dependency_container()
    
    deps.logger.debug("Starting Pydantic model validation")
    
    try:
        # Execute validation hook if registered (TST-REF-003)
        hook_result = deps.execute_validation_hook('config_validation', config_data)
        if hook_result is not None:
            deps.logger.debug("Using validation hook result")
            return hook_result
        
        validated_config = ColumnConfigDict.model_validate(config_data)
        deps.logger.debug(f"Successfully validated configuration with {len(validated_config.columns)} columns")
        return validated_config
    
    except Exception as e:
        error_msg = f"Configuration validation failed: {str(e)}"
        deps.logger.error(error_msg)
        raise


def load_column_config(config_path: str, dependencies: Optional[DependencyContainer] = None) -> ColumnConfigDict:
    """
    Load and validate column configuration from a YAML file.
    
    Enhanced with dependency injection and modular function decomposition
    per TST-REF-001 and TST-REF-002 requirements for comprehensive testing.
    
    Args:
        config_path: Path to the YAML configuration file
        dependencies: Optional dependency container for testing scenarios
        
    Returns:
        ColumnConfigDict: Validated column configuration model
        
    Raises:
        FileNotFoundError: If the configuration file is not found
        ValidationError: If the configuration is invalid
    """
    deps = dependencies or get_dependency_container()
    
    deps.logger.info(f"Loading column configuration from {config_path}")
    
    # Modular decomposition (TST-REF-002)
    config_data = _load_yaml_content(config_path, deps)
    validated_config = _validate_config_data(config_data, deps)
    
    deps.logger.info("Column configuration loaded and validated successfully")
    return validated_config


def _handle_string_source(config_source: str, dependencies: Optional[DependencyContainer] = None) -> ColumnConfigDict:
    """
    Handle string configuration source (file path).
    
    Modular function for string source handling per TST-REF-002 requirements.
    """
    deps = dependencies or get_dependency_container()
    deps.logger.debug(f"Handling string source as file path: {config_source}")
    return load_column_config(config_source, deps)


def _handle_dict_source(config_source: Dict[str, Any], dependencies: Optional[DependencyContainer] = None) -> ColumnConfigDict:
    """
    Handle dictionary configuration source.
    
    Modular function for dictionary source handling per TST-REF-002 requirements.
    """
    deps = dependencies or get_dependency_container()
    deps.logger.debug("Handling dictionary source for configuration")
    return _validate_config_data(config_source, deps)


def _handle_model_source(config_source: ColumnConfigDict, dependencies: Optional[DependencyContainer] = None) -> ColumnConfigDict:
    """
    Handle ColumnConfigDict configuration source.
    
    Modular function for model source handling per TST-REF-002 requirements.
    """
    deps = dependencies or get_dependency_container()
    deps.logger.debug("Using provided ColumnConfigDict instance")
    return config_source


def _handle_none_source(dependencies: Optional[DependencyContainer] = None) -> ColumnConfigDict:
    """
    Handle None configuration source (use default).
    
    Modular function for default source handling per TST-REF-002 requirements.
    """
    deps = dependencies or get_dependency_container()
    deps.logger.debug("Using default configuration")
    default_path = get_default_config_path(deps)
    return load_column_config(default_path, deps)


def get_config_from_source(
    config_source: Union[str, Dict[str, Any], ColumnConfigDict, None] = None,
    dependencies: Optional[DependencyContainer] = None
) -> ColumnConfigDict:
    """
    Get a validated ColumnConfigDict from different types of configuration sources.
    
    Enhanced with dependency injection and modular function decomposition
    per TST-REF-001, TST-REF-002, and TST-REF-003 requirements.
    
    Args:
        config_source: The configuration source, which can be:
            - A string path to a YAML configuration file
            - A dictionary containing configuration data
            - A ColumnConfigDict instance
            - None (uses default configuration)
        dependencies: Optional dependency container for testing scenarios
            
    Returns:
        ColumnConfigDict: Validated column configuration model
        
    Raises:
        TypeError: If the config_source type is invalid
        ValidationError: If the configuration is invalid
        FileNotFoundError: If a specified file is not found
    """
    deps = dependencies or get_dependency_container()
    
    deps.logger.debug(f"Processing configuration source of type: {type(config_source).__name__}")
    
    # Execute source handling hook if registered (TST-REF-003)
    hook_result = deps.execute_validation_hook('source_handling', config_source, dependencies)
    if hook_result is not None:
        deps.logger.debug("Using source handling hook result")
        return hook_result
    
    # Modular decomposition for different source types (TST-REF-002)
    try:
        if isinstance(config_source, str):
            return _handle_string_source(config_source, deps)
        
        elif isinstance(config_source, dict):
            return _handle_dict_source(config_source, deps)
        
        elif isinstance(config_source, ColumnConfigDict):
            return _handle_model_source(config_source, deps)
        
        elif config_source is None:
            return _handle_none_source(deps)
        
        else:
            error_msg = (
                "config_source must be a path to a YAML file, a configuration dictionary, "
                f"a ColumnConfigDict instance, or None, got {type(config_source)}"
            )
            deps.logger.error(error_msg)
            raise TypeError(error_msg)
    
    except Exception as e:
        deps.logger.error(f"Failed to process configuration source: {str(e)}")
        raise


# Test utilities and helper functions (TST-REF-003)

def create_test_dependency_container(
    yaml_loader: Optional[YamlLoaderProtocol] = None,
    logger: Optional[LoggerProtocol] = None,
    file_system: Optional[FileSystemProtocol] = None,
    test_mode: bool = True
) -> DependencyContainer:
    """
    Create a dependency container configured for testing scenarios.
    
    Test-specific entry point per TST-REF-003 requirements for controlled
    behavior during test execution.
    
    Args:
        yaml_loader: Custom YAML loader for testing
        logger: Custom logger for testing
        file_system: Custom file system for testing
        test_mode: Enable test mode features
        
    Returns:
        DependencyContainer: Configured container for testing
    """
    container = DependencyContainer(
        yaml_loader=yaml_loader or DefaultYamlLoader(),
        logger=logger or DefaultLogger(),
        file_system=file_system or DefaultFileSystem(),
        test_mode=test_mode
    )
    
    container.logger.debug("Created test dependency container")
    return container


def register_validation_behavior(
    validation_config: ValidationConfig,
    dependencies: Optional[DependencyContainer] = None
) -> None:
    """
    Register custom validation behavior for testing scenarios.
    
    Configurable validation behavior parameters supporting comprehensive
    mock-based testing scenarios per Feature F-016 requirements.
    
    Args:
        validation_config: Configuration for validation behavior
        dependencies: Optional dependency container
    """
    deps = dependencies or get_dependency_container()
    
    # Register custom validators as hooks
    for field_name, validator in validation_config.custom_validators.items():
        hook_name = f"{field_name}_validation"
        deps.register_validation_hook(hook_name, validator)
        deps.logger.debug(f"Registered custom validator for {field_name}")
    
    deps.logger.debug("Validation behavior configuration completed")


def validate_column_config_with_hooks(
    config_data: Dict[str, Any],
    validation_hooks: Dict[str, Callable],
    dependencies: Optional[DependencyContainer] = None
) -> ColumnConfigDict:
    """
    Validate configuration with custom validation hooks.
    
    Test-specific entry point per TST-REF-003 requirements allowing
    controlled behavior during test execution.
    
    Args:
        config_data: Configuration data to validate
        validation_hooks: Custom validation hooks for testing
        dependencies: Optional dependency container
        
    Returns:
        ColumnConfigDict: Validated configuration
    """
    deps = dependencies or get_dependency_container()
    
    # Register hooks temporarily
    original_hooks = deps.validation_hooks.copy()
    try:
        deps.validation_hooks.update(validation_hooks)
        deps.logger.debug(f"Registered {len(validation_hooks)} validation hooks for testing")
        
        result = _validate_config_data(config_data, deps)
        deps.logger.debug("Validation completed with custom hooks")
        return result
    
    finally:
        # Restore original hooks
        deps.validation_hooks = original_hooks


def get_validation_diagnostics(dependencies: Optional[DependencyContainer] = None) -> Dict[str, Any]:
    """
    Get diagnostic information about validation state for testing.
    
    Test utility function per TST-REF-003 requirements for improved
    test observability and assertion capabilities.
    
    Args:
        dependencies: Optional dependency container
        
    Returns:
        Dict: Diagnostic information including hook counts, test mode status, etc.
    """
    deps = dependencies or get_dependency_container()
    
    diagnostics = {
        'test_mode': deps.test_mode,
        'registered_hooks': list(deps.validation_hooks.keys()),
        'hook_count': len(deps.validation_hooks),
        'dependency_types': {
            'yaml_loader': type(deps.yaml_loader).__name__,
            'logger': type(deps.logger).__name__,
            'file_system': type(deps.file_system).__name__
        }
    }
    
    deps.logger.debug(f"Generated validation diagnostics: {diagnostics}")
    return diagnostics


def validate_experimental_data(experimental_data: Dict[str, Any], column_config: Union[Dict[str, Any], ColumnConfigDict], 
                           dependencies: Optional[DependencyContainer] = None) -> Dict[str, Any]:
    """
    Validate experimental data against a column configuration.
    
    This function validates that the provided experimental data dictionary conforms to the
    specified column configuration, checking for required fields, data types, and dimensions.
    
    Args:
        experimental_data: Dictionary containing experimental data to validate
        column_config: Column configuration dictionary or ColumnConfigDict instance
        dependencies: Optional dependency container for testing scenarios
        
    Returns:
        Dict[str, Any]: Validated and potentially transformed experimental data
        
    Raises:
        ValueError: If validation fails
        TypeError: If input types are incorrect
    """
    if not isinstance(experimental_data, dict):
        raise TypeError(f"experimental_data must be a dictionary, got {type(experimental_data).__name__}")
    
    # Get dependency container
    deps = dependencies or get_dependency_container()
    
    # Convert dict to ColumnConfigDict if needed
    if not isinstance(column_config, ColumnConfigDict):
        try:
            column_config = ColumnConfigDict.model_validate(column_config)
        except Exception as e:
            raise ValueError(f"Invalid column configuration: {str(e)}") from e
    
    validated_data = {}
    
    # Check required columns
    for col_name, col_config in column_config.columns.items():
        if col_config.required and col_name not in experimental_data:
            if col_config.default_value is not None:
                validated_data[col_name] = col_config.default_value
                continue
            raise ValueError(f"Missing required column: {col_name}")
        
        if col_name in experimental_data:
            value = experimental_data[col_name]
            
            # Type checking
            if col_config.type == "numpy.ndarray":
                if not isinstance(value, np.ndarray):
                    raise TypeError(f"Column {col_name} must be a numpy.ndarray, got {type(value).__name__}")
                
                # Dimension checking
                if col_config.dimension is not None:
                    if len(value.shape) != col_config.dimension.value:
                        raise ValueError(
                            f"Column {col_name} must be {col_config.dimension.value}D, "
                            f"got {len(value.shape)}D"
                        )
            
            validated_data[col_name] = value
    
    # Add any missing optional columns with default values
    for col_name, col_config in column_config.columns.items():
        if col_name not in validated_data and col_config.default_value is not None:
            validated_data[col_name] = col_config.default_value
    
    return validated_data


def transform_to_standardized_format(
    experimental_data: Dict[str, Any],
    column_config: Union[Dict[str, Any], ColumnConfigDict],
    dependencies: Optional[DependencyContainer] = None
) -> Dict[str, Any]:
    """
    Transform experimental data into a standardized format based on column configuration.
    
    This function applies any necessary transformations to the input data to ensure
    it matches the expected format defined in the column configuration. This includes:
    - Renaming columns based on aliases
    - Applying special handling for specific column types
    - Ensuring consistent data types and shapes
    
    Args:
        experimental_data: Dictionary containing the raw experimental data
        column_config: Column configuration dictionary or ColumnConfigDict instance
        dependencies: Optional dependency container for testing scenarios
        
    Returns:
        Dict[str, Any]: Transformed data in standardized format
        
    Raises:
        ValueError: If transformation fails or data cannot be standardized
        TypeError: If input types are incorrect
    """
    # Get dependencies
    deps = dependencies or _dependency_container
    logger = deps.logger
    
    # Convert dict config to ColumnConfigDict if needed
    if not isinstance(column_config, ColumnConfigDict):
        try:
            column_config = ColumnConfigDict.model_validate(column_config)
        except Exception as e:
            error_msg = f"Invalid column configuration: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    # Make a copy of the input data to avoid modifying the original
    transformed_data = experimental_data.copy()
    
    # Apply column aliases
    for col_name, col_config in column_config.columns.items():
        if col_config.alias and col_config.alias in transformed_data:
            transformed_data[col_name] = transformed_data.pop(col_config.alias)
    
    # Apply special handling for specific column types
    for col_name, col_config in column_config.columns.items():
        if col_name not in transformed_data:
            if col_config.default_value is not None:
                transformed_data[col_name] = col_config.default_value
            continue
            
        # Handle special column types
        if col_config.special_handling == SpecialHandlerType.EXTRACT_FIRST_COLUMN_IF_2D:
            data = transformed_data[col_name]
            if isinstance(data, np.ndarray) and data.ndim == 2:
                transformed_data[col_name] = data[:, 0]
                
        elif col_config.special_handling == SpecialHandlerType.TRANSFORM_TIME_DIMENSION:
            # Implementation for time dimension transformation would go here
            pass
    
    # Validate the transformed data
    try:
        return validate_experimental_data(transformed_data, column_config, dependencies=dependencies)
    except Exception as e:
        logger.error(f"Failed to validate transformed data: {str(e)}")
        raise


# Import compatibility layer for backward compatibility
try:
    from flyrigloader import logger
    _legacy_logger = logger
except ImportError:
    _legacy_logger = DefaultLogger()

# Maintain backward compatibility while enabling dependency injection
logger = _legacy_logger