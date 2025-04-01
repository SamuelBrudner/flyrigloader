"""
Pytest configuration file.
"""
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

# --- Loguru / Caplog Integration Fixture ---
import pytest
import logging
from loguru import logger

@pytest.fixture(autouse=True) # Run once per function
def capture_loguru_logs_globally(caplog):
    """Fixture to capture Loguru logs into pytest's caplog for all tests."""
    class PropagateHandler(logging.Handler):
        def emit(self, record):
            # Ensure the logger exists in the standard logging hierarchy
            std_logger = logging.getLogger(record.name)
            # Avoid adding handlers repeatedly if fixture runs multiple times somehow
            if not any(isinstance(h, PropagateHandler) for h in std_logger.handlers):
                 # If using caplog, it might manage handlers; let's just handle the record
                 # std_logger.addHandler(self) # Potentially problematic with caplog
                 pass # Rely on caplog capturing from the root logger or specific loggers
            std_logger.handle(record)

    # Add the handler to loguru's sinks
    handler_id = logger.add(PropagateHandler(), format="{message}", level=0)
    
    # Set caplog default level (individual tests can override)
    caplog.set_level(logging.DEBUG)
    
    yield # Run tests
    
    # Remove the handler after the session finishes
    try:
        logger.remove(handler_id)
    except ValueError:
        # Handler may have already been removed in another test
        pass
