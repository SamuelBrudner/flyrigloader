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


def test_loguru_used_in_patterns_module(caplog):
    """Test that the patterns module is using loguru for logging."""
    # Generate a pattern to trigger logging
    test_pattern = patterns.generate_pattern_from_template("{experiment}_{date}.csv")
    
    # Check if log message was captured and contains the pattern
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("Generated pattern" in r.message and test_pattern in r.message 
               for r in debug_records), "Expected debug log from patterns module not found"


def test_loguru_used_in_pickle_module(caplog):
    """Test that the pickle module is using loguru for logging."""
    # Create a simple test case that triggers logging in pickle
    import contextlib
    
    with contextlib.suppress(FileNotFoundError, ValueError):
        # This should trigger an error log about the invalid path
        pickle.read_pickle_any_format("nonexistent_file.pkl")
    
    # Check if log message was captured at ERROR level
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("nonexistent_file.pkl" in r.message for r in error_records), \
           "Expected error log from pickle module not found"


def test_loguru_imported_in_all_modules():
    """Test that all modules are importing loguru rather than standard logging."""
    # List all modules that should be using loguru
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
        assert "logger" in module_dir, f"Module {module_name} does not have a 'logger' attribute"
        
        # Check if the logger is from loguru
        logger_obj = getattr(module, "logger")
        assert str(type(logger_obj)) == "<class 'loguru._logger.Logger'>", \
               f"Logger in {module_name} is not a loguru Logger"
