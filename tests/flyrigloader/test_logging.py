import pytest
from loguru import logger
import logging
import os

# Ensure the logger configuration from flyrigloader/__init__.py is loaded
# This will set up the console and file sinks
import flyrigloader

# --- Fixture to integrate Loguru with pytest caplog ---
# MOVED to conftest.py to be globally available

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
