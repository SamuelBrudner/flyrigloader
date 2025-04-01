"""
FlyRigLoader - Tools for managing and controlling fly rigs for neuroscience experiments.
"""

__version__ = "0.1.0"

import sys
import os
from loguru import logger

# --- Logger Configuration ---

# Remove default stderr sink to configure our own
logger.remove()

# Console Sink (INFO level)
log_format_console = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add(
    sys.stderr, 
    level="INFO", 
    format=log_format_console,
    colorize=True  # Use colors for console output
)

# File Sink (DEBUG level) - Create logs directory relative to project root if it doesn't exist
# Assuming the script runs from the project root or __init__.py's location relative to it
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # Adjust if needed
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "flyrigloader_{time:YYYYMMDD}.log")

log_format_file = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} - {message}"
)
logger.add(
    log_file_path, 
    rotation="10 MB",  # Rotate log file when it reaches 10 MB
    retention="7 days", # Keep logs for 7 days
    compression="zip", # Compress rotated files
    level="DEBUG", 
    format=log_format_file,
    encoding="utf-8" # Specify encoding for the log file
)

logger.info("--- FlyRigLoader Logger Initialized ---")

# --- End Logger Configuration ---
