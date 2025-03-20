"""
Centralized import handling for optional dependencies.

This module provides unified import handling for optional dependencies,
ensuring consistent error messages and proper fallbacks.
"""

from typing import Tuple, Any, Optional, Dict, List
import importlib
from loguru import logger


def import_optional_dependency(
    dependency_name: str,
    error_message: Optional[str] = None,
    min_version: Optional[str] = None
) -> Tuple[Any, bool]:
    """
    Import an optional dependency with consistent error handling.
    
    Args:
        dependency_name: Name of the dependency to import
        error_message: Custom error message if import fails
        min_version: Minimum required version
        
    Returns:
        Tuple of (module, is_available)
        - module: The imported module or None if not available
        - is_available: Boolean indicating if the dependency is available
    """
    try:
        module = importlib.import_module(dependency_name)
        
        # Check for minimum version if specified
        if min_version is not None:
            version = getattr(module, "__version__", "0.0.0")
            if version < min_version:
                logger.warning(
                    f"Installed {dependency_name} version {version} is older than recommended "
                    f"minimum version {min_version}. Some features may not work correctly."
                )
        
        return module, True
    except ImportError:
        if error_message is None:
            error_message = (
                f"{dependency_name} not installed. Some features may not be available. "
                f"Install with 'pip install {dependency_name}' to use all features."
            )
        logger.warning(error_message)
        return None, False


# Pre-import common optional dependencies
pandera, PANDERA_AVAILABLE = import_optional_dependency(
    "pandera", 
    error_message="Pandera not installed. Install with 'pip install pandera' to use "
    "enhanced schema validation. Basic validation will be used as fallback.",
    min_version="0.10.0"
)


class DependencyManager:
    """
    Manager for handling optional package dependencies.
    
    This class provides methods to check if dependencies are available
    and import them on demand, with appropriate error messages.
    """
    
    def __init__(self):
        """Initialize with empty dependency cache."""
        self._imported_deps: Dict[str, Tuple[Any, bool]] = {}
    
    def require(
        self, 
        dependency_name: str, 
        error_message: Optional[str] = None, 
        min_version: Optional[str] = None
    ) -> Any:
        """
        Require a dependency, raising an error if it's not available.
        
        Args:
            dependency_name: Name of the dependency to import
            error_message: Custom error message if import fails
            min_version: Minimum required version
            
        Returns:
            The imported module
            
        Raises:
            ImportError: If the dependency is not available
        """
        module, available = self.check(
            dependency_name, 
            error_message=error_message, 
            min_version=min_version
        )
        
        if not available:
            if error_message is None:
                error_message = (
                    f"{dependency_name} is required for this operation but not installed. "
                    f"Install with 'pip install {dependency_name}'."
                )
            raise ImportError(error_message)
            
        return module
    
    def check(
        self, 
        dependency_name: str, 
        error_message: Optional[str] = None, 
        min_version: Optional[str] = None
    ) -> Tuple[Any, bool]:
        """
        Check if a dependency is available, importing it if not already cached.
        
        Args:
            dependency_name: Name of the dependency to import
            error_message: Custom error message if import fails
            min_version: Minimum required version
            
        Returns:
            Tuple of (module, is_available)
        """
        if dependency_name not in self._imported_deps:
            self._imported_deps[dependency_name] = import_optional_dependency(
                dependency_name, 
                error_message=error_message,
                min_version=min_version
            )
            
        return self._imported_deps[dependency_name]


# Create a singleton instance for use throughout the package
dependencies = DependencyManager()