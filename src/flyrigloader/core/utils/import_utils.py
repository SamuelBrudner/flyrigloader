"""
Centralized import handling for optional dependencies.

This module provides unified import handling for optional dependencies,
ensuring consistent error messages and proper fallbacks.
"""

from typing import Tuple, Any, Optional, Dict, List
import importlib
from loguru import logger
from packaging import version as pkg_version


def import_optional_dependency(
    dependency_name: str, 
    error_message: Optional[str] = None,
    min_version: Optional[str] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Import an optional dependency, gracefully handling its absence.
    
    This follows the flyrigloader standardized error handling pattern by returning
    a tuple with metadata containing a success flag.
    
    Args:
        dependency_name: Name of the dependency to import
        error_message: Custom error message if dependency is not available
        min_version: Minimum version required (string like "1.0.0")
        
    Returns:
        Tuple of (module or None, metadata_dict)
        If import fails, module will be None and metadata will contain error information
    """
    try:
        module = importlib.import_module(dependency_name)
        
        # Check version if requested
        if min_version and hasattr(module, "__version__") and pkg_version.parse(module.__version__) < pkg_version.parse(min_version):
            error_message = (
                f"{dependency_name} version {module.__version__} is installed, but "
                f"version {min_version} or higher is required. Please upgrade with "
                f"'pip install --upgrade {dependency_name}>={min_version}'"
            )
            logger.warning(error_message)
            return None, {
                "success": False,
                "error": f"Insufficient version of {dependency_name}",
                "found_version": str(module.__version__),
                "required_version": min_version
            }
        
        return module, {"success": True}
    except ImportError:
        if error_message is None:
            error_message = (
                f"{dependency_name} not installed. Some features may not be available. "
                f"Install with 'pip install {dependency_name}' to use all features."
            )
        logger.warning(error_message)
        return None, {
            "success": False,
            "error": error_message,
            "missing_dependency": dependency_name
        }


# Pre-import common optional dependencies
pandera, pandera_meta = import_optional_dependency(
    "pandera", 
    error_message="Pandera not installed. Install with 'pip install pandera' to use "
    "enhanced schema validation. Basic validation will be used as fallback.",
    min_version="0.10.0"
)
PANDERA_AVAILABLE = pandera_meta["success"]


class DependencyManager:
    """
    Manager for handling optional package dependencies.
    
    This class provides methods to check if dependencies are available
    and import them on demand, with appropriate error messages.
    """
    
    def __init__(self):
        """Initialize with empty dependency cache."""
        self._imported_deps: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
    
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
        module, metadata = self.check(
            dependency_name, 
            error_message=error_message, 
            min_version=min_version
        )
        
        if not metadata["success"]:
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
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Check if a dependency is available, importing it if not already cached.
        
        Args:
            dependency_name: Name of the dependency to import
            error_message: Custom error message if import fails
            min_version: Minimum required version
            
        Returns:
            Tuple of (module, metadata)
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


# Utility functions for commonly used dependencies

def has_pandera() -> bool:
    """Check if pandera is available.
    
    Returns:
        bool: True if pandera is installed and available
    """
    return PANDERA_AVAILABLE


def import_pandera():
    """Import pandera if available.
    
    Returns:
        Any: The pandera module if available, None otherwise
    """
    return pandera
