"""
FlyRigLoader - Tools for managing and controlling fly rigs for neuroscience experiments.

This module provides test-configurable logging initialization supporting pytest.monkeypatch 
scenarios for comprehensive test isolation and mocking capabilities per TST-INF-002 and 
TST-REF-001 requirements.
"""

__version__ = "0.1.0"

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Callable, TextIO
from loguru import logger
import warnings


# --- Logger Configuration Classes and Types ---

class LoggingConfigError(Exception):
    """
    Exception raised when logging configuration fails validation or setup.
    
    This exception supports test scenario validation per Section 2.2.8 
    test modernization requirements.
    """
    pass


class LoggerState:
    """
    Manages logger state for test scenarios and dependency injection.
    
    This class enables controlled logger state management during test execution 
    per Feature F-016 testability refactoring requirements.
    """
    
    def __init__(self):
        self._initialized = False
        self._test_mode = False
        self._sink_ids = []
        self._original_handlers = []
    
    def is_initialized(self) -> bool:
        """Check if logger has been initialized."""
        return self._initialized
    
    def is_test_mode(self) -> bool:
        """Check if logger is in test mode."""
        return self._test_mode
    
    def mark_initialized(self, test_mode: bool = False):
        """Mark logger as initialized."""
        self._initialized = True
        self._test_mode = test_mode
    
    def add_sink_id(self, sink_id: int):
        """Track sink IDs for cleanup."""
        self._sink_ids.append(sink_id)
    
    def reset(self):
        """Reset logger state for test isolation."""
        self._initialized = False
        self._test_mode = False
        self._sink_ids.clear()
        self._original_handlers.clear()


# Global logger state for dependency injection
_logger_state = LoggerState()


# --- Configuration Validation Functions ---

def validate_log_level(level: str) -> str:
    """
    Validate log level parameter for test scenarios.
    
    Integration of dependency validation for logging configuration ensuring 
    test-time configuration validity per TST-REF-003 requirements.
    
    Args:
        level: Log level string to validate
        
    Returns:
        Validated log level string
        
    Raises:
        LoggingConfigError: If log level is invalid
    """
    valid_levels = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
    level_upper = level.upper()
    
    if level_upper not in valid_levels:
        raise LoggingConfigError(
            f"Invalid log level '{level}'. Must be one of: {', '.join(valid_levels)}"
        )
    
    return level_upper


def validate_output_destination(destination: Union[str, Path, TextIO, None]) -> Union[str, Path, TextIO, None]:
    """
    Validate output destination parameter for pytest fixture-based testing scenarios.
    
    Args:
        destination: Output destination to validate
        
    Returns:
        Validated destination
        
    Raises:
        LoggingConfigError: If destination is invalid
    """
    if destination is None:
        return None
    
    if hasattr(destination, 'write'):  # File-like object (including sys.stderr, sys.stdout)
        return destination
    
    try:
        # Convert to Path for validation
        path_dest = Path(destination)
        
        # Validate parent directory exists or can be created
        if not path_dest.parent.exists():
            try:
                path_dest.parent.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                raise LoggingConfigError(
                    f"Cannot create directory for log destination '{destination}': {e}"
                )
        
        return str(path_dest)
    except (TypeError, ValueError) as e:
        raise LoggingConfigError(
            f"Invalid output destination '{destination}': {e}"
        )


# --- Core Logging Configuration Functions ---

def configure_console_logging(
    level: str = "INFO",
    format_template: Optional[str] = None,
    colorize: bool = True,
    destination: TextIO = sys.stderr
) -> int:
    """
    Configure console logging sink with test-configurable parameters.
    
    Supports pytest.monkeypatch scenarios for Loguru logger configuration 
    per TST-INF-002 requirements.
    
    Args:
        level: Log level for console output
        format_template: Custom format template (uses default if None)
        colorize: Enable colored console output
        destination: Console destination (default: sys.stderr)
        
    Returns:
        Sink ID for tracking and cleanup
        
    Raises:
        LoggingConfigError: If configuration fails
    """
    try:
        validated_level = validate_log_level(level)
        
        if format_template is None:
            format_template = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
            )
        
        sink_id = logger.add(
            destination,
            level=validated_level,
            format=format_template,
            colorize=colorize
        )
        
        _logger_state.add_sink_id(sink_id)
        return sink_id
        
    except Exception as e:
        raise LoggingConfigError(f"Failed to configure console logging: {e}") from e


def configure_file_logging(
    log_file_path: Union[str, Path],
    level: str = "DEBUG",
    rotation: str = "10 MB",
    retention: str = "7 days",
    compression: str = "zip",
    format_template: Optional[str] = None,
    encoding: str = "utf-8",
    create_dirs: bool = True
) -> int:
    """
    Configure file logging sink with comprehensive parameters.
    
    Addition of configurable log level and output destination parameters 
    supporting pytest fixture-based testing scenarios per Section 2.1.16 requirements.
    
    Args:
        log_file_path: Path to log file
        level: Log level for file output
        rotation: Log rotation setting
        retention: Log retention setting
        compression: Compression method for rotated logs
        format_template: Custom format template (uses default if None)
        encoding: File encoding
        create_dirs: Create parent directories if they don't exist
        
    Returns:
        Sink ID for tracking and cleanup
        
    Raises:
        LoggingConfigError: If configuration fails
    """
    try:
        validated_level = validate_log_level(level)
        validated_path = validate_output_destination(log_file_path)
        
        if create_dirs:
            Path(validated_path).parent.mkdir(parents=True, exist_ok=True)
        
        if format_template is None:
            format_template = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} - {message}"
            )
        
        sink_id = logger.add(
            validated_path,
            rotation=rotation,
            retention=retention,
            compression=compression,
            level=validated_level,
            format=format_template,
            encoding=encoding
        )
        
        _logger_state.add_sink_id(sink_id)
        return sink_id
        
    except Exception as e:
        raise LoggingConfigError(f"Failed to configure file logging: {e}") from e


# --- Test-Specific Entry Points ---

def configure_test_logging(
    console_level: str = "DEBUG",
    console_destination: Optional[TextIO] = None,
    enable_file_logging: bool = False,
    file_destination: Optional[Union[str, Path]] = None,
    capture_warnings: bool = True,
    **kwargs
) -> Dict[str, int]:
    """
    Configure logging specifically for test scenarios.
    
    Implementation of test-specific entry points allowing controlled logger 
    state management during test execution per Feature F-016 testability 
    refactoring requirements.
    
    Args:
        console_level: Console log level for tests
        console_destination: Console destination (None uses sys.stderr)
        enable_file_logging: Whether to enable file logging in tests
        file_destination: File destination for test logs
        capture_warnings: Whether to capture Python warnings
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary mapping sink types to sink IDs
        
    Raises:
        LoggingConfigError: If test logging configuration fails
    """
    try:
        # Reset logger state for test isolation
        reset_logging()
        
        sink_ids = {}
        
        # Configure console logging
        console_dest = console_destination if console_destination is not None else sys.stderr
        sink_ids['console'] = configure_console_logging(
            level=console_level,
            destination=console_dest,
            colorize=False,  # Disable colors in tests
            **{k: v for k, v in kwargs.items() if k.startswith('console_')}
        )
        
        # Configure file logging if requested
        if enable_file_logging and file_destination:
            sink_ids['file'] = configure_file_logging(
                log_file_path=file_destination,
                **{k: v for k, v in kwargs.items() if k.startswith('file_')}
            )
        
        # Capture warnings if requested
        if capture_warnings:
            warnings.showwarning = lambda *args: logger.warning(
                f"Warning: {args[0]} ({args[1]}:{args[2]})"
            )
        
        _logger_state.mark_initialized(test_mode=True)
        return sink_ids
        
    except Exception as e:
        raise LoggingConfigError(f"Failed to configure test logging: {e}") from e


def reset_logging():
    """
    Reset logger configuration for test isolation.
    
    Enhanced error handling in logging setup with proper exception propagation 
    for test scenario validation per Section 2.2.8 test modernization requirements.
    
    Raises:
        LoggingConfigError: If reset fails
    """
    try:
        # Remove all existing handlers
        logger.remove()
        
        # Reset global state
        _logger_state.reset()
        
    except Exception as e:
        raise LoggingConfigError(f"Failed to reset logging configuration: {e}") from e


# --- Production Logging Initialization ---

def initialize_production_logging(
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    log_dir: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
    **kwargs
) -> Dict[str, int]:
    """
    Initialize production logging configuration with dependency injection support.
    
    Addition of dependency injection patterns for logger configuration enabling 
    comprehensive test isolation and mocking capabilities per TST-REF-001 requirements.
    
    Args:
        console_level: Console logging level
        file_level: File logging level
        log_dir: Directory for log files (computed if None)
        project_root: Project root directory (computed if None)
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary mapping sink types to sink IDs
        
    Raises:
        LoggingConfigError: If production logging initialization fails
    """
    try:
        # Remove default handlers
        logger.remove()
        
        sink_ids = {}
        
        # Configure console logging
        sink_ids['console'] = configure_console_logging(level=console_level, **kwargs)
        
        # Determine log directory
        if log_dir is None:
            if project_root is None:
                # Compute project root relative to this file
                project_root = Path(__file__).parent.parent.parent
            else:
                project_root = Path(project_root)
            
            log_dir = project_root / "logs"
        else:
            log_dir = Path(log_dir)
        
        # Create log file path
        log_file_path = log_dir / "flyrigloader_{time:YYYYMMDD}.log"
        
        # Configure file logging
        sink_ids['file'] = configure_file_logging(
            log_file_path=log_file_path,
            level=file_level,
            **kwargs
        )
        
        _logger_state.mark_initialized(test_mode=False)
        logger.info("--- FlyRigLoader Logger Initialized ---")
        
        return sink_ids
        
    except Exception as e:
        raise LoggingConfigError(f"Failed to initialize production logging: {e}") from e


# --- Module-Level Logger State Access ---

def get_logger_state() -> LoggerState:
    """
    Get current logger state for test inspection.
    
    Returns:
        Current logger state object
    """
    return _logger_state


def is_logging_initialized() -> bool:
    """
    Check if logging has been initialized.
    
    Returns:
        True if logging is initialized, False otherwise
    """
    return _logger_state.is_initialized()


def is_test_mode() -> bool:
    """
    Check if logging is in test mode.
    
    Returns:
        True if in test mode, False otherwise
    """
    return _logger_state.is_test_mode()


# --- Backward Compatibility and Auto-Initialization ---

def _auto_initialize_logging():
    """
    Auto-initialize logging for backward compatibility.
    
    This function maintains compatibility with existing usage while supporting 
    new test-configurable patterns.
    """
    # Only auto-initialize if not already configured and not in test environment
    if not _logger_state.is_initialized() and not _is_pytest_running():
        try:
            initialize_production_logging()
        except LoggingConfigError as e:
            # Fall back to basic stderr logging if production initialization fails
            warnings.warn(f"Failed to initialize production logging: {e}. Using basic stderr logging.")
            logger.add(sys.stderr, level="INFO")
            _logger_state.mark_initialized(test_mode=False)


def _is_pytest_running() -> bool:
    """
    Detect if code is running under pytest.
    
    Returns:
        True if pytest is detected, False otherwise
    """
    return "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ


# --- Module Initialization ---

# Auto-initialize logging for backward compatibility
_auto_initialize_logging()

# --- End Logger Configuration ---
