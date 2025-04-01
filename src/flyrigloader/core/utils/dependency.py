"""
Dependency injection and management system.

This module provides a thread-safe dependency management system with
explicit dependency injection capabilities.
"""

import threading
from typing import Any, Dict, Optional, Tuple, TypeVar, Type, Callable

from .import_utils import import_optional_dependency

T = TypeVar('T')


class DependencyContainer:
    """
    Thread-safe container for managing dependencies with explicit injection.
    
    This class provides methods to register, resolve, and require dependencies,
    ensuring thread-safety and proper dependency management.
    """
    
    def __init__(self):
        """Initialize the dependency container."""
        self._dependencies: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._singletons: Dict[str, Any] = {}
        self._imported_deps: Dict[str, Tuple[Any, bool]] = {}
        self._lock = threading.RLock()  # Reentrant lock for thread safety
    
    def register(self, key: str, dependency: Any) -> None:
        """
        Register a dependency instance with the container.
        
        Args:
            key: The key to register the dependency under
            dependency: The dependency instance
        """
        with self._lock:
            self._dependencies[key] = dependency
    
    def register_factory(self, key: str, factory: Callable[[], Any]) -> None:
        """
        Register a factory function that creates the dependency when needed.
        
        Args:
            key: The key to register the factory under
            factory: A callable that returns a new instance of the dependency
        """
        with self._lock:
            self._factories[key] = factory
    
    def register_singleton(self, key: str, factory: Callable[[], Any]) -> None:
        """
        Register a singleton factory that creates the dependency once when first needed.
        
        Args:
            key: The key to register the singleton under
            factory: A callable that returns a new instance of the dependency
        """
        with self._lock:
            # Just store the factory, actual instance will be created on first resolve
            self._factories[key] = factory
            # Mark this as a singleton
            self._singletons[key] = None
    
    def register_singleton_instance(self, key: str, instance: Any) -> None:
        """
        Register an existing instance as a singleton.
        
        Args:
            key: The key to register the singleton under
            instance: The singleton instance
        """
        with self._lock:
            self._singletons[key] = instance
    
    def resolve(self, key: str) -> Optional[Any]:
        """
        Resolve a dependency by key, returning None if not found.
        
        Args:
            key: The key to resolve
            
        Returns:
            The dependency or None if not found
        """
        with self._lock:
            # Check if it's a singleton and already created
            if key in self._singletons and self._singletons[key] is not None:
                return self._singletons[key]
            
            # Check if it's a direct dependency
            if key in self._dependencies:
                return self._dependencies[key]
            
            # Check if it's a factory
            if key in self._factories:
                instance = self._factories[key]()
                
                # If it's a singleton, store the instance
                if key in self._singletons:
                    self._singletons[key] = instance
                
                return instance
            
            return None
    
    def require(self, key: str) -> Any:
        """
        Require a dependency, raising KeyError if not found.
        
        Args:
            key: The key to resolve
            
        Returns:
            The dependency
            
        Raises:
            KeyError: If the dependency is not found
        """
        instance = self.resolve(key)
        if instance is None:
            raise KeyError(f"Dependency '{key}' not found in container")
        return instance
    
    def import_dependency(
        self, 
        dependency_name: str, 
        error_message: Optional[str] = None, 
        min_version: Optional[str] = None
    ) -> Tuple[Any, bool]:
        """
        Import an optional dependency and cache the result.
        
        Args:
            dependency_name: Name of the dependency to import
            error_message: Custom error message if import fails
            min_version: Minimum required version
            
        Returns:
            Tuple of (module, is_available)
        """
        with self._lock:
            if dependency_name not in self._imported_deps:
                self._imported_deps[dependency_name] = import_optional_dependency(
                    dependency_name, 
                    error_message=error_message,
                    min_version=min_version
                )
            
            return self._imported_deps[dependency_name]
    
    def require_import(
        self, 
        dependency_name: str, 
        error_message: Optional[str] = None, 
        min_version: Optional[str] = None
    ) -> Any:
        """
        Require an optional dependency, raising ImportError if not available.
        
        Args:
            dependency_name: Name of the dependency to import
            error_message: Custom error message if import fails
            min_version: Minimum required version
            
        Returns:
            The imported module
            
        Raises:
            ImportError: If the dependency is not available
        """
        module, available = self.import_dependency(
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


class ThreadSafeDependencyManager:
    """
    Thread-safe singleton dependency manager for backward compatibility.
    
    This class wraps DependencyContainer to provide a thread-safe singleton
    pattern while maintaining the original DependencyManager interface.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    cls._instance = super(ThreadSafeDependencyManager, cls).__new__(cls)
                    # Initialize the instance within the same lock
                    cls._instance._container = DependencyContainer()
                    cls._instance._initialized = True
        return cls._instance
    
    def __init__(self):
        # No initialization needed here - it's done in __new__
        pass
    
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
        return self._container.import_dependency(
            dependency_name, 
            error_message=error_message, 
            min_version=min_version
        )
    
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
        return self._container.require_import(
            dependency_name, 
            error_message=error_message, 
            min_version=min_version
        )
    
    @property
    def container(self) -> DependencyContainer:
        """
        Get the underlying dependency container.
        
        Returns:
            The dependency container
        """
        return self._container


# Create thread-safe singleton instances for use throughout the package
# For backward compatibility, keep the original name
dependencies = ThreadSafeDependencyManager()

# Provide direct access to the container for explicit DI usage
container = dependencies.container
