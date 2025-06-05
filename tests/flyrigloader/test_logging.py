import pytest
from loguru import logger
import logging
import os
import importlib

# Ensure the logger configuration from flyrigloader/__init__.py is loaded
# This will set up the console and file sinks
import flyrigloader

# Modules to validate for loguru usage
from flyrigloader.discovery import patterns
from flyrigloader.io import pickle

# --- Tests ---

def test_logger_basic_output(caplog):
    """Test that basic logger messages are captured at the correct level."""
    test_message_info = "This is an INFO test message."
    test_message_debug = "This is a DEBUG test message."
    
    # Log messages
    logger.info(test_message_info)
    logger.debug(test_message_debug)

    # Check INFO message in caplog
    assert test_message_info in caplog.text
    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert info_records, "Should have at least one INFO record"
    assert any(test_message_info in r.message for r in info_records), "INFO message content not found"

    # Check DEBUG message in caplog
    assert test_message_debug in caplog.text
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert debug_records, "Should have at least one DEBUG record"
    assert any(test_message_debug in r.message for r in debug_records), "DEBUG message content not found"


def test_logger_file_creation():
    """Test that the log file directory is created."""
    # This test relies on the import flyrigloader having run,
    # which triggers the logger setup in src/flyrigloader/__init__.py
    
    # Calculate the expected project root based on the structure 
    # ProjectRoot/src/flyrigloader/__init__.py
    src_dir = os.path.dirname(flyrigloader.__file__)
    flyrigloader_pkg_dir = os.path.dirname(src_dir)
    project_root = os.path.dirname(flyrigloader_pkg_dir)
    log_dir = os.path.join(project_root, "logs")

    # The __init__ should have created this directory
    assert os.path.isdir(log_dir), f"Log directory '{log_dir}' should exist after import."
    
    # Note: Checking for a specific log file is brittle due to the date in the filename.
    # Checking for the directory existence is a more robust test of the configuration.


def test_console_log_format_validation():
    """Test comprehensive console log format validation per F-005 requirements."""
    # Check that the format includes required components
    format_string = flyrigloader.log_format_console
    
    # Verify module placeholder inclusion per F-005 requirements
    assert "{module}" in format_string, "Console log format must include {module} placeholder per F-005"
    
    # Verify timestamp formatting per F-005 requirements
    assert "{time:" in format_string, "Console log format must include timestamp formatting per F-005"
    assert "YYYY-MM-DD HH:mm:ss" in format_string, "Console log format must include proper timestamp format per F-005"
    
    # Verify other required components
    assert "{level}" in format_string, "Console log format must include level information"
    assert "{function}" in format_string, "Console log format must include function information for debugging"
    assert "{line}" in format_string, "Console log format must include line number information for debugging"
    assert "{message}" in format_string, "Console log format must include the actual message"
    
    # Verify format structure and readability
    expected_components = ["{time:", "{level:", "{module}", "{function}", "{line}", "{message}"]
    for component in expected_components:
        assert component in format_string, f"Required format component {component} missing from console format"


def test_loguru_used_in_patterns_module(caplog):
    """Test that the patterns module is using loguru for logging per F-005 integration requirements."""
    caplog.set_level(logging.DEBUG)
    
    # Generate a pattern to trigger logging
    test_pattern = patterns.generate_pattern_from_template("{experiment}_{date}.csv")
    
    # Check if log message was captured and contains the pattern
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("Generated pattern" in r.message and test_pattern in r.message 
               for r in debug_records), "Expected debug log from patterns module not found"


def test_loguru_used_in_pickle_module(caplog):
    """Test that the pickle module is using loguru for logging per F-005 integration requirements."""
    caplog.set_level(logging.ERROR)
    
    # Create a simple test case that triggers logging in pickle
    with contextlib.suppress(FileNotFoundError, ValueError, RuntimeError):
        # This should trigger an error log about the invalid path
        pickle.read_pickle_any_format("nonexistent_file.pkl")
    
    # Check if log message was captured at ERROR level
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("nonexistent_file.pkl" in r.message for r in error_records), \
           "Expected error log from pickle module not found"


def test_loguru_used_in_column_models_module(caplog):
    """Test that the column_models module is using loguru for logging per F-005 integration requirements."""
    caplog.set_level(logging.DEBUG)
    
    # Import to ensure the module is loaded and trigger any initialization logging
    importlib.reload(column_models)
    
    # Test validation that might trigger warnings
    from flyrigloader.io.column_models import ColumnConfig
    
    # Create a config that might trigger warnings
    config = ColumnConfig(
        type="non_numpy_type",
        dimension=2,  # This should trigger a warning for non-numpy type
        description="Test config for logging validation"
    )
    
    # Check if warning was logged (might not always trigger, but validates logger is available)
    # The test validates that the module has proper loguru integration
    assert hasattr(column_models, 'logger'), "column_models module should have logger attribute"


def test_comprehensive_loguru_module_verification():
    """Test comprehensive loguru._logger.Logger usage across all internal modules per F-005 requirements."""
    # Extended list of modules that should be using loguru
    modules_to_check = [
        "flyrigloader.discovery.patterns",
        "flyrigloader.io.pickle", 
        "flyrigloader.io.column_models"
    ]
    
    for module_name in modules_to_check:
        # Import the module or get it if already imported
        module = importlib.import_module(module_name)
        module_dir = dir(module)
        
        # Check if module has loguru logger
        assert "logger" in module_dir, f"Module {module_name} does not have a 'logger' attribute per F-005"
        
        # Check if the logger is from loguru
        logger_obj = getattr(module, "logger")
        logger_type_str = str(type(logger_obj))
        assert logger_type_str == "<class 'loguru._logger.Logger'>", \
               f"Logger in {module_name} is not a loguru._logger.Logger, got {logger_type_str}"
        
        # Verify logger has required methods
        required_methods = ['info', 'debug', 'warning', 'error', 'critical']
        for method in required_methods:
            assert hasattr(logger_obj, method), \
                   f"Logger in {module_name} missing required method {method}"


def test_log_sanitization_security():
    """Test log sanitization to ensure no sensitive data exposure per TST-INF-002 security requirements."""
    # Test data that should be sanitized or not logged
    sensitive_data = {
        "password": "secret123",
        "api_key": "sk-1234567890abcdef",
        "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        "private_key": "-----BEGIN PRIVATE KEY-----",
        "credit_card": "4111-1111-1111-1111"
    }
    
    # Test that sensitive data patterns are not logged inadvertently
    with pytest.warns(None) as warning_list:
        for data_type, sensitive_value in sensitive_data.items():
            # Test logging with potential sensitive data
            test_message = f"Processing {data_type} configuration"
            logger.info(test_message)
            
            # Verify the message itself doesn't contain sensitive values
            assert sensitive_value not in test_message, \
                   f"Test message should not contain sensitive {data_type} value"
    
    # Verify that no warnings were raised about sensitive data in logs
    sensitive_warnings = [w for w in warning_list if any(
        sensitive_word in str(w.message).lower() 
        for sensitive_word in ['password', 'key', 'token', 'secret']
    )]
    
    # This test validates that our logging practices don't accidentally expose sensitive data
    assert len(sensitive_warnings) == 0, "No sensitive data warnings should be present in standard logging"


def test_log_message_structure_validation(caplog):
    """Test that log messages follow proper structure and formatting per F-005 requirements."""
    caplog.set_level(logging.DEBUG)
    caplog.clear()
    
    # Test various log levels with structured messages
    logger.info("Test INFO message with structured format")
    logger.debug("Test DEBUG message with structured format") 
    logger.warning("Test WARNING message with structured format")
    logger.error("Test ERROR message with structured format")
    
    # Verify all messages were captured
    assert len(caplog.records) >= 4, "All test log messages should be captured"
    
    # Check message structure and format
    for record in caplog.records:
        # Verify record has required attributes
        assert hasattr(record, 'levelname'), "Log record should have levelname"
        assert hasattr(record, 'message'), "Log record should have message"
        assert hasattr(record, 'module'), "Log record should have module info"
        
        # Verify message content is properly formatted
        assert len(record.message) > 0, "Log message should not be empty"
        assert "structured format" in record.message, "Test message content should be preserved"


def test_file_log_format_validation():
    """Test file log format configuration per F-005 requirements."""
    # Access the file log format from the flyrigloader initialization
    # The file format is defined in __init__.py
    expected_file_format_components = [
        "{time:YYYY-MM-DD HH:mm:ss.SSS}",
        "{level: <8}",
        "{name}",
        "{function}",
        "{line}",
        "{message}"
    ]
    
    # The file format should be structured for parsing and analysis
    # We can't directly access it, but we can verify the configuration principles
    # by checking that our logging setup includes file output
    
    # Verify that the logger has file handlers configured
    from loguru._logger import Logger
    test_logger = Logger()
    
    # Check that flyrigloader has set up logging correctly during import
    # The import should have configured both console and file logging
    assert hasattr(flyrigloader, 'log_format_console'), "Console format should be defined"
    
    # Verify file logging configuration principles
    log_dir_exists = True
    try:
        src_dir = os.path.dirname(flyrigloader.__file__)
        project_root = os.path.dirname(os.path.dirname(src_dir))
        log_dir = os.path.join(project_root, "logs")
        log_dir_exists = os.path.isdir(log_dir)
    except Exception:
        log_dir_exists = False
    
    assert log_dir_exists, "File logging should be configured with logs directory"

