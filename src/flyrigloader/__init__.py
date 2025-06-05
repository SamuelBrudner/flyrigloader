"""
FlyRigLoader - Tools for managing and controlling fly rigs for neuroscience experiments.
"""

__version__ = "0.1.0"

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
from loguru import logger


# --- Logger Configuration Infrastructure ---

class LoggerConfig:
    """Configuration container for Loguru logger settings supporting test injection."""
    
    def __init__(
        self,
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        log_dir: Optional[Union[str, Path]] = None,
        console_format: Optional[str] = None,
        file_format: Optional[str] = None,
        rotation: str = "10 MB",
        retention: str = "7 days",
        compression: str = "zip",
        colorize: bool = True,
        enable_file_logging: bool = True,
        enable_console_logging: bool = True
    ):
        """
        Initialize logger configuration.
        
        Args:
            console_level: Log level for console output
            file_level: Log level for file output  
            log_dir: Directory for log files (defaults to project_root/logs)
            console_format: Custom console format string
            file_format: Custom file format string
            rotation: Log file rotation setting
            retention: Log file retention period
            compression: Log file compression format
            colorize: Whether to use colors in console output
            enable_file_logging: Whether to enable file logging
            enable_console_logging: Whether to enable console logging
        """
        self.console_level = console_level
        self.file_level = file_level
        self.log_dir = log_dir
        self.rotation = rotation
        self.retention = retention
        self.compression = compression
        self.colorize = colorize
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        
        # Set default formats if not provided
        self.console_format = console_format or (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        self.file_format = file_format or (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} - {message}"
        )


def _validate_logger_config(config: LoggerConfig) -> None:
    """
    Validate logger configuration parameters.
    
    Args:
        config: LoggerConfig instance to validate
        
    Raises:
        ValueError: If configuration parameters are invalid
        TypeError: If configuration types are incorrect
    """
    # Validate log levels
    valid_levels = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
    
    if config.console_level.upper() not in valid_levels:
        raise ValueError(f"Invalid console_level: {config.console_level}. Must be one of {valid_levels}")
    
    if config.file_level.upper() not in valid_levels:
        raise ValueError(f"Invalid file_level: {config.file_level}. Must be one of {valid_levels}")
    
    # Validate boolean flags
    if not isinstance(config.colorize, bool):
        raise TypeError(f"colorize must be boolean, got {type(config.colorize)}")
    
    if not isinstance(config.enable_file_logging, bool):
        raise TypeError(f"enable_file_logging must be boolean, got {type(config.enable_file_logging)}")
    
    if not isinstance(config.enable_console_logging, bool):
        raise TypeError(f"enable_console_logging must be boolean, got {type(config.enable_console_logging)}")
    
    # Validate log directory if provided
    if config.log_dir is not None:
        if not isinstance(config.log_dir, (str, Path)):
            raise TypeError(f"log_dir must be string or Path, got {type(config.log_dir)}")


def _get_default_log_directory() -> Path:
    """
    Determine the default log directory relative to project root.
    
    Returns:
        Path object pointing to the logs directory
        
    Raises:
        OSError: If unable to determine or create log directory
    """
    try:
        # Calculate project root from __init__.py location
        project_root = Path(__file__).parent.parent.parent
        log_dir = project_root / "logs"
        return log_dir
    except Exception as e:
        raise OSError(f"Unable to determine default log directory: {e}") from e


def _ensure_log_directory(log_dir: Path) -> None:
    """
    Ensure log directory exists, creating it if necessary.
    
    Args:
        log_dir: Path to log directory
        
    Raises:
        OSError: If unable to create log directory
        PermissionError: If insufficient permissions to create directory
    """
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Insufficient permissions to create log directory {log_dir}: {e}") from e
    except OSError as e:
        raise OSError(f"Unable to create log directory {log_dir}: {e}") from e


def initialize_logger(config: Optional[LoggerConfig] = None) -> None:
    """
    Initialize the Loguru logger with configurable settings.
    
    This function provides dependency injection support for testing scenarios
    and allows complete control over logger configuration.
    
    Args:
        config: Optional LoggerConfig instance. If None, uses default configuration.
        
    Raises:
        ValueError: If configuration validation fails
        TypeError: If configuration types are incorrect
        OSError: If log directory operations fail
        PermissionError: If insufficient permissions for file operations
        RuntimeError: If logger initialization fails
    """
    if config is None:
        config = LoggerConfig()
    
    try:
        # Validate configuration before proceeding
        _validate_logger_config(config)
        
        # Remove any existing logger sinks to start fresh
        logger.remove()
        
        # Configure console logging if enabled
        if config.enable_console_logging:
            logger.add(
                sys.stderr,
                level=config.console_level.upper(),
                format=config.console_format,
                colorize=config.colorize
            )
        
        # Configure file logging if enabled
        if config.enable_file_logging:
            # Determine log directory
            if config.log_dir is None:
                log_dir = _get_default_log_directory()
            else:
                log_dir = Path(config.log_dir)
            
            # Ensure log directory exists
            _ensure_log_directory(log_dir)
            
            # Construct log file path with timestamp
            log_file_path = log_dir / "flyrigloader_{time:YYYYMMDD}.log"
            
            logger.add(
                str(log_file_path),
                rotation=config.rotation,
                retention=config.retention,
                compression=config.compression,
                level=config.file_level.upper(),
                format=config.file_format,
                encoding="utf-8"
            )
        
        # Log successful initialization
        logger.info("--- FlyRigLoader Logger Initialized ---")
        
    except Exception as e:
        # Re-raise with additional context for debugging
        raise RuntimeError(f"Logger initialization failed: {e}") from e


def configure_test_logger(
    console_level: str = "DEBUG",
    disable_file_logging: bool = True,
    disable_console_colors: bool = True
) -> None:
    """
    Configure logger for testing scenarios with sensible defaults.
    
    This function provides a test-specific entry point for logger configuration
    that is optimized for pytest execution environments.
    
    Args:
        console_level: Log level for test console output
        disable_file_logging: Whether to disable file logging during tests
        disable_console_colors: Whether to disable console colors for CI
        
    Raises:
        RuntimeError: If test logger configuration fails
    """
    try:
        test_config = LoggerConfig(
            console_level=console_level,
            enable_file_logging=not disable_file_logging,
            colorize=not disable_console_colors,
            enable_console_logging=True
        )
        initialize_logger(test_config)
    except Exception as e:
        raise RuntimeError(f"Test logger configuration failed: {e}") from e


def reset_logger() -> None:
    """
    Reset logger to uninitialized state.
    
    This function removes all logger sinks and is primarily intended
    for testing scenarios where logger state needs to be controlled.
    
    Raises:
        RuntimeError: If logger reset fails
    """
    try:
        logger.remove()
    except Exception as e:
        raise RuntimeError(f"Logger reset failed: {e}") from e


# Initialize logger with default configuration on module import
# This maintains backward compatibility while allowing test overrides
try:
    initialize_logger()
except Exception as init_error:
    # If initialization fails, log to stderr and continue
    print(f"Warning: Logger initialization failed: {init_error}", file=sys.stderr)
    # Ensure logger has at least basic stderr output for error reporting
    logger.add(sys.stderr, level="ERROR")

# --- End Logger Configuration ---
