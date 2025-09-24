"""Column configuration public API entry points."""

from __future__ import annotations

from typing import Optional

from flyrigloader import logger
from flyrigloader.exceptions import FlyRigLoaderError
from flyrigloader.io.column_models import ColumnConfigDict

from .dependencies import DefaultDependencyProvider, get_dependency_provider

def get_default_column_config(
    _deps: Optional[DefaultDependencyProvider] = None
) -> ColumnConfigDict:
    """
    Get the default column configuration with enhanced testability.
    
    This function supports comprehensive dependency injection for testing scenarios
    through the _deps parameter, enabling pytest.monkeypatch patterns.
    
    Args:
        _deps: Optional dependency provider for testing injection (internal parameter)
    
    Returns:
        ColumnConfigDict with the default configuration
        
    Raises:
        FileNotFoundError: If the default configuration file doesn't exist
        ValueError: If the configuration is invalid
    """
    operation_name = "get_default_column_config"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.info("Loading default column configuration")
    
    try:
        # Load the default configuration with dependency injection
        result = _deps.io.get_config_from_source(None)
        
        if hasattr(result, 'columns'):
            column_count = len(result.columns) if hasattr(result.columns, '__len__') else 0
            logger.info(f"Successfully loaded default configuration with {column_count} columns")
            logger.debug(f"Column names: {list(result.columns.keys()) if hasattr(result.columns, 'keys') else 'N/A'}")
        else:
            logger.info(f"Successfully loaded default configuration, type: {type(result).__name__}")
        
        return result
        
    except Exception as e:
        error_msg = (
            f"Failed to load default column configuration for {operation_name}: {e}. "
            "Please ensure the default configuration file exists and is valid."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e
