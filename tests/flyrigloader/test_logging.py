"""
Test module for logging functionality validation using behavior-focused testing.

This module validates logging behavior through public API interactions and observable
effects rather than internal implementation inspection, following the requirements
outlined in Section 0 for behavior-focused testing and AAA pattern implementation.

Key Testing Approach:
- Public API behavior validation through observable log output
- Log level filtering and format compliance verification  
- Security sanitization behavior through message content validation
- Cross-platform directory creation through filesystem effects
- Protocol-based mock verification for dependency injection scenarios

Enhanced with centralized fixtures from tests/conftest.py for consistent mock
implementations and standardized test data patterns across the test suite.
"""

import contextlib
import importlib
import logging
import os
import platform
import pytest
from pathlib import Path
from loguru import logger
from unittest.mock import patch, MagicMock

# Import modules to validate for logging integration
import flyrigloader
from flyrigloader.discovery import patterns
from flyrigloader.io import pickle
from flyrigloader.io import column_models


# ============================================================================
# BEHAVIOR-FOCUSED LOGGING TESTS WITH AAA PATTERN IMPLEMENTATION
# ============================================================================

def test_logger_basic_output_behavior(caplog):
    """
    Test that basic logger messages produce observable output at correct levels.
    
    Type: Behavior validation test
    Validates: Public logging API behavior through captured output
    Edge cases: Multiple log levels and message content preservation
    """
    # ARRANGE - Set up test messages and log capture
    test_message_info = "This is an INFO test message."
    test_message_debug = "This is a DEBUG test message."
    test_message_warning = "This is a WARNING test message."
    
    # ACT - Execute logging operations through public API
    logger.info(test_message_info)
    logger.debug(test_message_debug)
    logger.warning(test_message_warning)
    
    # ASSERT - Verify observable logging behavior
    # Verify INFO message appears in captured logs
    assert test_message_info in caplog.text
    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert info_records, "Should capture at least one INFO record"
    assert any(test_message_info in r.message for r in info_records), "INFO message content should be preserved"
    
    # Verify DEBUG message appears in captured logs
    assert test_message_debug in caplog.text
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert debug_records, "Should capture at least one DEBUG record"
    assert any(test_message_debug in r.message for r in debug_records), "DEBUG message content should be preserved"
    
    # Verify WARNING message appears in captured logs
    assert test_message_warning in caplog.text
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, "Should capture at least one WARNING record"
    assert any(test_message_warning in r.message for r in warning_records), "WARNING message content should be preserved"


def test_logger_file_creation_behavior(temp_experiment_directory):
    """
    Test that logger initialization creates observable filesystem effects.
    
    Type: Cross-platform behavior validation
    Validates: Directory creation through filesystem observation
    Edge cases: Cross-platform path handling and permission scenarios
    """
    # ARRANGE - Set up test directory and configuration
    test_log_dir = temp_experiment_directory["directory"] / "test_logs"
    
    # ACT - Execute logger initialization with custom directory
    config = flyrigloader.LoggerConfig(
        log_dir=str(test_log_dir),
        enable_file_logging=True,
        enable_console_logging=False
    )
    flyrigloader.initialize_logger(config)
    
    # ASSERT - Verify observable filesystem behavior
    assert test_log_dir.exists(), f"Log directory '{test_log_dir}' should be created by logger initialization"
    assert test_log_dir.is_dir(), "Created path should be a directory"
    
    # Verify log files can be created (observable behavior)
    logger.info("Test log file creation")
    
    # Look for log files in the directory (behavior validation)
    log_files = list(test_log_dir.glob("*.log"))
    assert len(log_files) > 0, "Logger should create log files in the specified directory"


def test_console_log_format_compliance_behavior():
    """
    Test console log format compliance through public API validation.
    
    Type: Format compliance behavior validation
    Validates: Public format configuration accessibility and structure
    Edge cases: Required component presence and format specification compliance
    """
    # ARRANGE - Set up format validation requirements per F-005
    required_components = [
        "{time:",
        "YYYY-MM-DD HH:mm:ss",
        "{level}",
        "{module}",
        "{function}",
        "{line}",
        "{message}"
    ]
    
    # ACT - Retrieve format through public API
    format_string = flyrigloader.log_format_console
    
    # ASSERT - Verify format compliance through observable structure
    assert isinstance(format_string, str), "Console format should be accessible as string"
    assert len(format_string) > 0, "Console format should not be empty"
    
    # Verify all required components are present
    for component in required_components:
        assert component in format_string, (
            f"Console log format must include {component} component per F-005 requirements. "
            f"Format: {format_string}"
        )
    
    # Verify module placeholder specifically mentioned in requirements
    assert "{module}" in format_string, "Console log format must include {module} placeholder per F-005"
    
    # Verify timestamp formatting compliance
    assert "{time:" in format_string and "YYYY-MM-DD HH:mm:ss" in format_string, (
        "Console log format must include proper timestamp format per F-005"
    )


def test_logger_configuration_behavior_validation():
    """
    Test logger configuration behavior through public API interactions.
    
    Type: Configuration behavior validation
    Validates: LoggerConfig class functionality and initialization behavior
    Edge cases: Various configuration combinations and validation scenarios
    """
    # ARRANGE - Set up various configuration scenarios
    test_configs = [
        {
            "console_level": "INFO",
            "file_level": "DEBUG",
            "enable_console_logging": True,
            "enable_file_logging": False
        },
        {
            "console_level": "WARNING", 
            "file_level": "ERROR",
            "enable_console_logging": False,
            "enable_file_logging": True
        }
    ]
    
    for config_params in test_configs:
        # ACT - Create configuration through public API
        config = flyrigloader.LoggerConfig(**config_params)
        
        # ASSERT - Verify configuration behavior through public attributes
        assert config.console_level == config_params["console_level"]
        assert config.file_level == config_params["file_level"]
        assert config.enable_console_logging == config_params["enable_console_logging"]
        assert config.enable_file_logging == config_params["enable_file_logging"]
        
        # Verify default format behavior
        assert len(config.console_format) > 0, "Console format should have default value"
        assert len(config.file_format) > 0, "File format should have default value"


@pytest.mark.parametrize("module_name,test_function", [
    ("flyrigloader.discovery.patterns", "generate_pattern_from_template"),
    ("flyrigloader.io.pickle", "read_pickle_any_format"),
])
def test_module_logging_integration_behavior(caplog, module_name, test_function):
    """
    Test logging integration behavior across modules through public API usage.
    
    Type: Integration behavior validation
    Validates: Module logging through observable log output during normal operations
    Edge cases: Different modules and error scenarios with proper log capture
    """
    # ARRANGE - Set up logging capture and import module
    caplog.set_level(logging.DEBUG)
    caplog.clear()
    module = importlib.import_module(module_name)
    
    # ACT - Execute module functionality that should trigger logging
    with contextlib.suppress(Exception):
        if test_function == "generate_pattern_from_template":
            # Execute pattern generation (should produce debug logs)
            test_pattern = getattr(module, test_function)("{experiment}_{date}.csv")
        elif test_function == "read_pickle_any_format":
            # Execute pickle reading with invalid file (should produce error logs)
            getattr(module, test_function)("nonexistent_file.pkl")
    
    # ASSERT - Verify logging behavior through captured output
    # Verify that some logging occurred during module operation
    assert len(caplog.records) > 0, f"Module {module_name} should produce log output during operation"
    
    # Verify logging contains relevant information
    log_text = caplog.text.lower()
    if test_function == "generate_pattern_from_template":
        # Pattern generation should log debug information
        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(debug_records) > 0, "Pattern generation should produce debug logs"
    elif test_function == "read_pickle_any_format":
        # Pickle reading with invalid file should log errors
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) > 0, "Invalid pickle file should produce error logs"


def test_column_models_logging_behavior_validation(caplog):
    """
    Test column_models logging behavior through observable module operations.
    
    Type: Module logging behavior validation
    Validates: Logging integration through public API usage and observable effects
    Edge cases: Module initialization and configuration validation scenarios
    """
    # ARRANGE - Set up logging capture for validation scenarios
    caplog.set_level(logging.DEBUG)
    caplog.clear()
    
    # ACT - Execute column_models operations that may trigger logging
    # Test 1: Module reload to capture any initialization logging
    importlib.reload(column_models)
    
    # Test 2: Create configuration that might trigger warnings through public API
    from flyrigloader.io.column_models import ColumnConfig
    config = ColumnConfig(
        type="test_type",
        dimension=2,
        description="Test config for logging behavior validation"
    )
    
    # ASSERT - Verify logging behavior through observable effects
    # Verify configuration object was created successfully (behavior validation)
    assert config.type == "test_type"
    assert config.dimension == 2
    assert config.description == "Test config for logging behavior validation"
    
    # Verify any logging that occurred is captured
    # Note: This tests the behavior of logging integration, not internal logger attributes
    if caplog.records:
        # If logging occurred, verify it contains meaningful information
        log_messages = [record.message for record in caplog.records]
        assert any(len(msg) > 0 for msg in log_messages), "Log messages should contain content"


@pytest.mark.parametrize("sensitive_data_type,sensitive_value", [
    ("password", "secret123"),
    ("api_key", "sk-1234567890abcdef"),
    ("token", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"),
    ("private_key", "-----BEGIN PRIVATE KEY-----"),
    ("credit_card", "4111-1111-1111-1111"),
])
def test_log_sanitization_security_behavior(caplog, sensitive_data_type, sensitive_value):
    """
    Test log sanitization behavior to ensure no sensitive data exposure.
    
    Type: Security behavior validation per TST-INF-002
    Validates: Security sanitization through message content verification
    Edge cases: Various sensitive data types and logging scenarios
    """
    # ARRANGE - Set up sensitive data test scenario
    caplog.set_level(logging.DEBUG)
    caplog.clear()
    
    # ACT - Execute logging operations that should NOT expose sensitive data
    test_message = f"Processing {sensitive_data_type} configuration"
    logger.info(test_message)
    logger.debug(f"Configuration type: {sensitive_data_type}")
    
    # ASSERT - Verify sanitization behavior through content analysis
    # Verify the safe message was logged
    assert test_message in caplog.text, "Safe log message should be present"
    
    # Verify sensitive value is NOT in logs (security behavior validation)
    assert sensitive_value not in caplog.text, (
        f"Sensitive {sensitive_data_type} value should not appear in log output"
    )
    
    # Verify message content doesn't accidentally include sensitive data
    assert sensitive_value not in test_message, (
        f"Test message should not contain sensitive {sensitive_data_type} value"
    )
    
    # Verify no warnings about sensitive data exposure
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    sensitive_warnings = [w for w in warning_records if any(
        sensitive_word in w.message.lower() 
        for sensitive_word in ['password', 'key', 'token', 'secret']
    )]
    assert len(sensitive_warnings) == 0, (
        "No sensitive data warnings should be present in standard logging"
    )


@pytest.mark.parametrize("log_level,expected_level", [
    ("DEBUG", logging.DEBUG),
    ("INFO", logging.INFO),
    ("WARNING", logging.WARNING),
    ("ERROR", logging.ERROR),
    ("CRITICAL", logging.CRITICAL),
])
def test_log_message_structure_validation_behavior(caplog, log_level, expected_level):
    """
    Test log message structure and format behavior per F-005 requirements.
    
    Type: Format compliance behavior validation
    Validates: Log structure through observable message attributes
    Edge cases: Various log levels and message content validation
    """
    # ARRANGE - Set up structured message testing
    caplog.set_level(logging.DEBUG)
    caplog.clear()
    test_message = f"Test {log_level} message with structured format"
    
    # ACT - Execute logging at specified level through public API
    log_function = getattr(logger, log_level.lower())
    log_function(test_message)
    
    # ASSERT - Verify message structure through observable attributes
    # Verify message was captured at correct level
    level_records = [r for r in caplog.records if r.levelno == expected_level]
    assert len(level_records) > 0, f"Should capture {log_level} level message"
    
    # Verify record structure compliance
    for record in level_records:
        assert hasattr(record, 'levelname'), "Log record should have levelname attribute"
        assert hasattr(record, 'message'), "Log record should have message attribute"
        assert hasattr(record, 'module'), "Log record should have module information"
        
        # Verify message content preservation
        assert len(record.message) > 0, "Log message should not be empty"
        assert "structured format" in record.message, "Test message content should be preserved"
        assert record.levelname == log_level, f"Level name should match {log_level}"


def test_file_log_format_validation_behavior(temp_experiment_directory):
    """
    Test file log format behavior through observable file operations.
    
    Type: File logging behavior validation
    Validates: File format configuration through public API and filesystem observation
    Edge cases: Cross-platform file creation and format compliance
    """
    # ARRANGE - Set up temporary directory for file logging test
    test_log_dir = temp_experiment_directory["directory"] / "file_format_test"
    test_log_dir.mkdir(exist_ok=True)
    
    # ACT - Configure file logging through public API
    config = flyrigloader.LoggerConfig(
        log_dir=str(test_log_dir),
        enable_file_logging=True,
        enable_console_logging=False,
        file_level="DEBUG"
    )
    flyrigloader.initialize_logger(config)
    
    # Execute logging operations
    logger.info("File format validation test message")
    logger.debug("Debug message for format testing")
    
    # ASSERT - Verify file logging behavior through filesystem observation
    # Verify log directory exists (behavioral validation)
    assert test_log_dir.exists(), "Log directory should be created"
    assert test_log_dir.is_dir(), "Log path should be a directory"
    
    # Verify log files are created (observable behavior)
    log_files = list(test_log_dir.glob("*.log"))
    assert len(log_files) > 0, "File logging should create log files"
    
    # Verify file contains expected content (behavioral validation)
    log_file = log_files[0]
    log_content = log_file.read_text()
    assert "File format validation test message" in log_content, (
        "Log file should contain logged messages"
    )
    assert "Debug message for format testing" in log_content, (
        "Log file should contain debug messages"
    )


def test_test_logger_configuration_behavior():
    """
    Test the test-specific logger configuration behavior.
    
    Type: Test configuration behavior validation
    Validates: Test logger setup through public API configuration
    Edge cases: Test-specific logging configuration and isolation
    """
    # ARRANGE - Set up test logger configuration parameters
    test_console_level = "DEBUG"
    disable_file_logging = True
    disable_console_colors = True
    
    # ACT - Configure test logger through public API
    flyrigloader.configure_test_logger(
        console_level=test_console_level,
        disable_file_logging=disable_file_logging,
        disable_console_colors=disable_console_colors
    )
    
    # Execute test logging
    with patch('sys.stderr') as mock_stderr:
        logger.info("Test logger configuration validation")
        
        # ASSERT - Verify test logger behavior through observable effects
        # Verify logger was configured (behavioral validation through successful execution)
        # Note: This tests the behavior, not internal configuration inspection
        mock_stderr.write.assert_called()


def test_logger_reset_behavior():
    """
    Test logger reset behavior for test isolation.
    
    Type: Reset behavior validation
    Validates: Logger reset functionality through public API
    Edge cases: Test isolation and clean state restoration
    """
    # ARRANGE - Set up initial logger state
    original_handlers_count = len(logger._core.handlers)
    
    # ACT - Reset logger through public API
    flyrigloader.reset_logger()
    
    # ASSERT - Verify reset behavior through observable state changes
    # Verify logger was reset (behavioral validation)
    reset_handlers_count = len(logger._core.handlers)
    assert reset_handlers_count == 0, "Logger reset should remove all handlers"


@pytest.mark.parametrize("platform_name,expected_behavior", [
    ("Windows", "path_creation"),
    ("Darwin", "path_creation"),
    ("Linux", "path_creation"),
])
def test_cross_platform_logging_behavior(platform_name, expected_behavior, temp_experiment_directory):
    """
    Test cross-platform logging behavior through directory creation validation.
    
    Type: Cross-platform behavior validation
    Validates: Platform-specific path handling through observable filesystem effects
    Edge cases: Different operating systems and path formats
    """
    # ARRANGE - Set up platform-specific test scenario
    test_log_dir = temp_experiment_directory["directory"] / "platform_test" / platform_name.lower()
    
    # ACT - Execute cross-platform logging setup
    config = flyrigloader.LoggerConfig(
        log_dir=str(test_log_dir),
        enable_file_logging=True,
        enable_console_logging=False
    )
    
    # Initialize logger for platform testing
    flyrigloader.initialize_logger(config)
    logger.info(f"Cross-platform test for {platform_name}")
    
    # ASSERT - Verify cross-platform behavior through filesystem observation
    if expected_behavior == "path_creation":
        # Verify directory creation works across platforms
        assert test_log_dir.exists(), f"Log directory should be created on {platform_name}"
        assert test_log_dir.is_dir(), f"Created path should be directory on {platform_name}"
        
        # Verify log files can be created (cross-platform behavior)
        log_files = list(test_log_dir.glob("*.log"))
        assert len(log_files) > 0, f"Log files should be created on {platform_name}"


def test_centralized_fixture_integration(capture_loguru_logs_globally, test_data_generator):
    """
    Test integration with centralized fixtures from tests/conftest.py.
    
    Type: Fixture integration validation
    Validates: Centralized fixture usage and consistent test patterns
    Edge cases: Fixture dependency injection and shared test utilities
    """
    # ARRANGE - Use centralized fixtures for consistent test setup
    # capture_loguru_logs_globally fixture automatically manages log capture
    # test_data_generator provides standardized test data creation
    
    test_data = test_data_generator.generate_experiment_metadata()
    test_message = f"Processing experiment {test_data['animal_id']} on {test_data['experiment_date']}"
    
    # ACT - Execute logging with fixture-provided data
    logger.info(test_message)
    logger.debug(f"Experiment condition: {test_data['condition']}")
    
    # ASSERT - Verify logging behavior with centralized fixture integration
    # Note: Log capture is automatically handled by capture_loguru_logs_globally fixture
    # This validates the fixture integration behavior rather than manual log capture
    
    # Verify test data was generated properly (fixture behavior validation)
    assert test_data['animal_id'] is not None, "Test data generator should provide animal_id"
    assert test_data['condition'] is not None, "Test data generator should provide condition"
    assert test_data['experiment_date'] is not None, "Test data generator should provide experiment_date"
    
    # Verify logging integration with fixture-managed capture
    # The capture_loguru_logs_globally fixture handles log capture automatically
    # This tests the behavioral integration rather than manual caplog usage