"""
Registry infrastructure for FlyRigLoader providing extensible plugin-style architecture.

This module implements the centralized registry pattern that serves as a first-class
architectural component, enabling dynamic registration of file format handlers and
column validation schemas without requiring modifications to core code.

Key Features:
- Thread-safe singleton implementation for registry instances
- Plugin-style extensibility with O(1) lookup performance
- Automatic plugin discovery through entry points
- Runtime registration support for third-party extensions
- Priority-based ordering for format handlers
- Comprehensive error handling with domain-specific exceptions

Architecture Integration:
The registry pattern aligns with the SOLID Open/Closed principle, allowing the system
to be open for extension but closed for modification. This approach replaces hardcoded
mappings with dynamic plugin-style extensibility.
"""

import threading
from abc import ABC, abstractmethod
from typing import (
    Any, Dict, List, Optional, Protocol, Type, TypeVar, Union,
    runtime_checkable
)
from pathlib import Path
import importlib.metadata
import warnings
from functools import wraps


# Type variables for generic registry implementation
T = TypeVar('T')
RegistryItem = TypeVar('RegistryItem')


# Protocol definitions for extensible handlers
@runtime_checkable
class BaseLoader(Protocol):
    """Protocol interface for custom loader implementations.
    
    All loaders registered with LoaderRegistry must implement this interface
    to ensure consistent behavior across different file formats.
    """
    
    def load(self, path: Path) -> Any:
        """Load raw data from file without transformation.
        
        Args:
            path: Path to the file to load
            
        Returns:
            Raw data object (typically dict or list)
            
        Raises:
            LoadError: If file cannot be loaded or is corrupted
        """
        ...
    
    def supports_extension(self, extension: str) -> bool:
        """Check if loader supports given file extension.
        
        Args:
            extension: File extension including dot (e.g., '.pkl')
            
        Returns:
            True if loader can handle this extension
        """
        ...
    
    @property
    def priority(self) -> int:
        """Priority for this loader when multiple loaders support same extension.
        
        Returns:
            Integer priority (higher values = higher priority)
        """
        ...


@runtime_checkable
class BaseSchema(Protocol):
    """Protocol interface for custom schema validators.
    
    All schemas registered with SchemaRegistry must implement this interface
    to ensure consistent validation behavior across different data types.
    """
    
    def validate(self, data: Any) -> Dict[str, Any]:
        """Validate data against schema and return validated result.
        
        Args:
            data: Raw data to validate
            
        Returns:
            Validated data dictionary
            
        Raises:
            TransformError: If data does not match schema
        """
        ...
    
    @property
    def schema_name(self) -> str:
        """Name identifying this schema."""
        ...
    
    @property
    def supported_types(self) -> List[str]:
        """List of data types this schema can validate."""
        ...


# Base registry implementation with shared functionality
class BaseRegistry(ABC):
    """Abstract base class for all registry implementations.
    
    Provides thread-safe singleton pattern and common registry operations.
    All concrete registries inherit from this base to ensure consistent
    behavior and thread safety across the system.
    """
    
    _instances: Dict[str, 'BaseRegistry'] = {}
    _lock = threading.Lock()
    
    def __new__(cls):
        """Implement thread-safe singleton pattern.
        
        Ensures only one instance of each registry type exists,
        with proper thread safety for concurrent access.
        """
        with cls._lock:
            class_name = cls.__name__
            if class_name not in cls._instances:
                cls._instances[class_name] = super().__new__(cls)
            return cls._instances[class_name]
    
    def __init__(self):
        """Initialize registry with thread-safe storage."""
        if not hasattr(self, '_initialized'):
            self._registry: Dict[str, Any] = {}
            self._priority_map: Dict[str, int] = {}
            self._registry_lock = threading.RLock()
            self._initialized = True
    
    def clear(self) -> None:
        """Clear all registered items from registry.
        
        Thread-safe operation that removes all registered items.
        Primarily used for testing and cleanup scenarios.
        """
        with self._registry_lock:
            self._registry.clear()
            self._priority_map.clear()
    
    def _get_sorted_items(self) -> List[tuple]:
        """Get registry items sorted by priority (highest first).
        
        Returns:
            List of (key, item) tuples sorted by priority
        """
        with self._registry_lock:
            return sorted(
                self._registry.items(),
                key=lambda x: self._priority_map.get(x[0], 0),
                reverse=True
            )
    
    @abstractmethod
    def _validate_item(self, item: Any) -> bool:
        """Validate that item conforms to registry requirements.
        
        Args:
            item: Item to validate
            
        Returns:
            True if item is valid for this registry
        """
        pass


class LoaderRegistry(BaseRegistry):
    """Registry for file format loaders providing O(1) lookup by extension.
    
    Manages file format handlers through a centralized registry pattern,
    enabling plugin-style extensibility where new data formats can be
    registered without modifying core code.
    
    Features:
    - Thread-safe singleton implementation
    - Priority-based ordering for format handlers
    - Automatic plugin discovery through entry points
    - Runtime registration support for third-party extensions
    - Comprehensive validation of loader implementations
    
    Example:
        >>> registry = LoaderRegistry()
        >>> registry.register_loader('.pkl', PickleLoader, priority=10)
        >>> loader = registry.get_loader_for_extension('.pkl')
        >>> data = loader.load(Path('data.pkl'))
    """
    
    def __init__(self):
        """Initialize LoaderRegistry with built-in loaders."""
        super().__init__()
        if not hasattr(self, '_loaders_initialized'):
            self._discover_plugins()
            self._loaders_initialized = True
    
    def register_loader(
        self,
        extension: str,
        loader_class: Type[BaseLoader],
        priority: int = 0
    ) -> None:
        """Register a loader for specific file extension.
        
        Args:
            extension: File extension including dot (e.g., '.pkl')
            loader_class: Loader class implementing BaseLoader protocol
            priority: Priority for this loader (higher = higher priority)
            
        Raises:
            ValueError: If extension is invalid or loader doesn't implement BaseLoader
            TypeError: If loader_class is not a valid loader implementation
        """
        if not extension.startswith('.'):
            raise ValueError(f"Extension must start with dot: {extension}")
        
        if not self._validate_item(loader_class):
            raise TypeError(f"Loader {loader_class} must implement BaseLoader protocol")
        
        with self._registry_lock:
            # Check if extension already registered with higher priority
            if extension in self._registry:
                current_priority = self._priority_map.get(extension, 0)
                if priority <= current_priority:
                    warnings.warn(
                        f"Loader for {extension} already registered with higher priority "
                        f"({current_priority} vs {priority}). Use higher priority to override.",
                        UserWarning
                    )
                    return
            
            # Register the loader
            self._registry[extension] = loader_class
            self._priority_map[extension] = priority
    
    def get_loader_for_extension(self, extension: str) -> Optional[Type[BaseLoader]]:
        """Get loader class for file extension with O(1) lookup.
        
        Args:
            extension: File extension including dot (e.g., '.pkl')
            
        Returns:
            Loader class if registered, None otherwise
        """
        if extension is None:
            return None
            
        if not extension.startswith('.'):
            extension = f'.{extension}'
        
        with self._registry_lock:
            return self._registry.get(extension)
    
    def get_all_loaders(self) -> Dict[str, Type[BaseLoader]]:
        """Get all registered loaders ordered by priority.
        
        Returns:
            Dictionary mapping extension to loader class, ordered by priority
        """
        with self._registry_lock:
            # Return copy to prevent external modification
            return dict(self._get_sorted_items())
    
    def unregister_loader(self, extension: str) -> bool:
        """Unregister loader for specific extension.
        
        Args:
            extension: File extension to unregister
            
        Returns:
            True if loader was removed, False if not found
        """
        if not extension.startswith('.'):
            extension = f'.{extension}'
        
        with self._registry_lock:
            if extension in self._registry:
                del self._registry[extension]
                self._priority_map.pop(extension, None)
                return True
            return False
    
    def _validate_item(self, item: Any) -> bool:
        """Validate that item implements BaseLoader protocol.
        
        Args:
            item: Item to validate
            
        Returns:
            True if item is valid loader
        """
        if not isinstance(item, type):
            return False
        
        # Check if class implements BaseLoader protocol by checking required methods
        required_methods = ['load', 'supports_extension']
        required_properties = ['priority']
        
        try:
            # Check if all required methods exist
            for method in required_methods:
                if not hasattr(item, method):
                    return False
            
            # Check if all required properties exist by creating instance
            instance = item()
            for prop in required_properties:
                if not hasattr(instance, prop):
                    return False
            
            return True
        except (TypeError, AttributeError):
            return False
    
    def _discover_plugins(self) -> None:
        """Discover and register plugins through entry points.
        
        Automatically discovers loaders registered through setuptools entry points
        under the 'flyrigloader.loaders' group.
        """
        try:
            # Discover entry points for loaders
            entry_points = importlib.metadata.entry_points()
            
            # Handle different entry_points API versions
            if hasattr(entry_points, 'select'):
                # New API (Python 3.10+)
                loader_entries = entry_points.select(group='flyrigloader.loaders')
            else:
                # Legacy API
                loader_entries = entry_points.get('flyrigloader.loaders', [])
            
            for entry_point in loader_entries:
                try:
                    loader_class = entry_point.load()
                    # Entry point name should be the extension
                    extension = entry_point.name
                    if not extension.startswith('.'):
                        extension = f'.{extension}'
                    
                    # Register with default priority
                    self.register_loader(extension, loader_class, priority=0)
                    
                except Exception as e:
                    warnings.warn(
                        f"Failed to load plugin loader {entry_point.name}: {e}",
                        UserWarning
                    )
        except Exception as e:
            # Plugin discovery failures shouldn't break the registry
            warnings.warn(f"Plugin discovery failed: {e}", UserWarning)


class SchemaRegistry(BaseRegistry):
    """Registry for column validation schemas providing O(1) lookup by name.
    
    Manages column validation schemas through a centralized registry pattern,
    enabling plugin-style extensibility where new schemas can be registered
    without modifying core code.
    
    Features:
    - Thread-safe singleton implementation
    - Schema versioning and compatibility checking
    - Automatic plugin discovery through entry points
    - Runtime registration support for third-party schemas
    - Comprehensive validation of schema implementations
    
    Example:
        >>> registry = SchemaRegistry()
        >>> registry.register_schema('experiment', ExperimentSchema)
        >>> schema = registry.get_schema('experiment')
        >>> validated_data = schema.validate(raw_data)
    """
    
    def __init__(self):
        """Initialize SchemaRegistry with built-in schemas."""
        super().__init__()
        if not hasattr(self, '_schemas_initialized'):
            self._discover_plugins()
            self._schemas_initialized = True
    
    def register_schema(
        self,
        name: str,
        schema_class: Type[BaseSchema],
        priority: int = 0
    ) -> None:
        """Register a schema with given name.
        
        Args:
            name: Schema name for lookup
            schema_class: Schema class implementing BaseSchema protocol
            priority: Priority for this schema (higher = higher priority)
            
        Raises:
            ValueError: If name is invalid or schema doesn't implement BaseSchema
            TypeError: If schema_class is not a valid schema implementation
        """
        if not name or not isinstance(name, str):
            raise ValueError(f"Schema name must be non-empty string: {name}")
        
        if not self._validate_item(schema_class):
            raise TypeError(f"Schema {schema_class} must implement BaseSchema protocol")
        
        with self._registry_lock:
            # Check if schema already registered with higher priority
            if name in self._registry:
                current_priority = self._priority_map.get(name, 0)
                if priority <= current_priority:
                    warnings.warn(
                        f"Schema {name} already registered with higher priority "
                        f"({current_priority} vs {priority}). Use higher priority to override.",
                        UserWarning
                    )
                    return
            
            # Register the schema
            self._registry[name] = schema_class
            self._priority_map[name] = priority
    
    def get_schema(self, name: str) -> Optional[Type[BaseSchema]]:
        """Get schema class by name with O(1) lookup.
        
        Args:
            name: Schema name
            
        Returns:
            Schema class if registered, None otherwise
        """
        with self._registry_lock:
            return self._registry.get(name)
    
    def get_all_schemas(self) -> Dict[str, Type[BaseSchema]]:
        """Get all registered schemas ordered by priority.
        
        Returns:
            Dictionary mapping schema name to schema class, ordered by priority
        """
        with self._registry_lock:
            # Return copy to prevent external modification
            return dict(self._get_sorted_items())
    
    def unregister_schema(self, name: str) -> bool:
        """Unregister schema by name.
        
        Args:
            name: Schema name to unregister
            
        Returns:
            True if schema was removed, False if not found
        """
        with self._registry_lock:
            if name in self._registry:
                del self._registry[name]
                self._priority_map.pop(name, None)
                return True
            return False
    
    def _validate_item(self, item: Any) -> bool:
        """Validate that item implements BaseSchema protocol.
        
        Args:
            item: Item to validate
            
        Returns:
            True if item is valid schema
        """
        if not isinstance(item, type):
            return False
        
        # Check if class implements BaseSchema protocol by checking required methods
        required_methods = ['validate']
        required_properties = ['schema_name', 'supported_types']
        
        try:
            # Check if all required methods exist
            for method in required_methods:
                if not hasattr(item, method):
                    return False
            
            # Check if all required properties exist by creating instance
            instance = item()
            for prop in required_properties:
                if not hasattr(instance, prop):
                    return False
            
            return True
        except (TypeError, AttributeError):
            return False
    
    def _discover_plugins(self) -> None:
        """Discover and register plugins through entry points.
        
        Automatically discovers schemas registered through setuptools entry points
        under the 'flyrigloader.schemas' group.
        """
        try:
            # Discover entry points for schemas
            entry_points = importlib.metadata.entry_points()
            
            # Handle different entry_points API versions
            if hasattr(entry_points, 'select'):
                # New API (Python 3.10+)
                schema_entries = entry_points.select(group='flyrigloader.schemas')
            else:
                # Legacy API
                schema_entries = entry_points.get('flyrigloader.schemas', [])
            
            for entry_point in schema_entries:
                try:
                    schema_class = entry_point.load()
                    # Entry point name should be the schema name
                    schema_name = entry_point.name
                    
                    # Register with default priority
                    self.register_schema(schema_name, schema_class, priority=0)
                    
                except Exception as e:
                    warnings.warn(
                        f"Failed to load plugin schema {entry_point.name}: {e}",
                        UserWarning
                    )
        except Exception as e:
            # Plugin discovery failures shouldn't break the registry
            warnings.warn(f"Plugin discovery failed: {e}", UserWarning)


# Utility functions for registry management
def register_loader(extension: str, loader_class: Type[BaseLoader], priority: int = 0) -> None:
    """Convenience function to register a loader.
    
    Args:
        extension: File extension including dot (e.g., '.pkl')
        loader_class: Loader class implementing BaseLoader protocol
        priority: Priority for this loader (higher = higher priority)
    """
    registry = LoaderRegistry()
    registry.register_loader(extension, loader_class, priority)


def register_schema(name: str, schema_class: Type[BaseSchema], priority: int = 0) -> None:
    """Convenience function to register a schema.
    
    Args:
        name: Schema name for lookup
        schema_class: Schema class implementing BaseSchema protocol
        priority: Priority for this schema (higher = higher priority)
    """
    registry = SchemaRegistry()
    registry.register_schema(name, schema_class, priority)


def get_loader_for_extension(extension: str) -> Optional[Type[BaseLoader]]:
    """Convenience function to get loader by extension.
    
    Args:
        extension: File extension including dot (e.g., '.pkl')
        
    Returns:
        Loader class if registered, None otherwise
    """
    registry = LoaderRegistry()
    return registry.get_loader_for_extension(extension)


def get_schema(name: str) -> Optional[Type[BaseSchema]]:
    """Convenience function to get schema by name.
    
    Args:
        name: Schema name
        
    Returns:
        Schema class if registered, None otherwise
    """
    registry = SchemaRegistry()
    return registry.get_schema(name)


# Registry decorator for automatic registration
def loader_for(extension: str, priority: int = 0):
    """Decorator to automatically register loader for extension.
    
    Args:
        extension: File extension including dot (e.g., '.pkl')
        priority: Priority for this loader (higher = higher priority)
        
    Example:
        @loader_for('.pkl', priority=10)
        class PickleLoader:
            def load(self, path: Path) -> Any:
                # Implementation
                pass
    """
    def decorator(cls):
        register_loader(extension, cls, priority)
        return cls
    return decorator


def schema_for(name: str, priority: int = 0):
    """Decorator to automatically register schema by name.
    
    Args:
        name: Schema name for lookup
        priority: Priority for this schema (higher = higher priority)
        
    Example:
        @schema_for('experiment', priority=10)
        class ExperimentSchema:
            def validate(self, data: Any) -> Dict[str, Any]:
                # Implementation
                pass
    """
    def decorator(cls):
        register_schema(name, cls, priority)
        return cls
    return decorator


# Export main classes and functions
__all__ = [
    'LoaderRegistry',
    'SchemaRegistry',
    'BaseLoader',
    'BaseSchema',
    'register_loader',
    'register_schema',
    'get_loader_for_extension',
    'get_schema',
    'loader_for',
    'schema_for'
]