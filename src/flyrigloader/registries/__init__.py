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
from enum import Enum
import logging

from flyrigloader.exceptions import RegistryError


# Type variables for generic registry implementation
T = TypeVar('T')
RegistryItem = TypeVar('RegistryItem')

# Logger for comprehensive registration event tracking
logger = logging.getLogger(__name__)


class RegistryPriority(Enum):
    """
    Priority enumeration system for loader and schema registration.
    
    Implements the priority hierarchy BUILTIN < USER < PLUGIN < OVERRIDE
    as specified in Section 0.2.1 registry enhancement requirements.
    Higher numeric values indicate higher priority.
    
    Attributes:
        BUILTIN: Built-in loaders provided with FlyRigLoader (priority 10)
        USER: User-registered loaders via API calls (priority 20)
        PLUGIN: Plugin-based loaders discovered via entry points (priority 30)
        OVERRIDE: Explicit override loaders with highest priority (priority 40)
    """
    BUILTIN = 10
    USER = 20
    PLUGIN = 30
    OVERRIDE = 40
    
    def __lt__(self, other):
        """Support comparison for priority resolution."""
        if isinstance(other, RegistryPriority):
            return self.value < other.value
        return NotImplemented
    
    def __le__(self, other):
        """Support comparison for priority resolution."""
        if isinstance(other, RegistryPriority):
            return self.value <= other.value
        return NotImplemented
    
    def __gt__(self, other):
        """Support comparison for priority resolution."""
        if isinstance(other, RegistryPriority):
            return self.value > other.value
        return NotImplemented
    
    def __ge__(self, other):
        """Support comparison for priority resolution."""
        if isinstance(other, RegistryPriority):
            return self.value >= other.value
        return NotImplemented
    
    @classmethod
    def from_priority_value(cls, priority: int) -> 'RegistryPriority':
        """
        Convert integer priority to RegistryPriority enum.
        
        Args:
            priority: Integer priority value
            
        Returns:
            Closest RegistryPriority enum value
        """
        if priority >= cls.OVERRIDE.value:
            return cls.OVERRIDE
        elif priority >= cls.PLUGIN.value:
            return cls.PLUGIN
        elif priority >= cls.USER.value:
            return cls.USER
        else:
            return cls.BUILTIN


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
        """Initialize registry with thread-safe storage and enhanced lifecycle tracking."""
        if not hasattr(self, '_initialized'):
            self._registry: Dict[str, Any] = {}
            self._priority_map: Dict[str, int] = {}
            self._priority_enum_map: Dict[str, RegistryPriority] = {}
            self._registration_metadata: Dict[str, Dict[str, Any]] = {}
            self._registry_lock = threading.RLock()
            self._initialized = True
            
            # Log registry initialization with comprehensive context
            logger.info(
                f"Initialized {self.__class__.__name__} registry with enhanced lifecycle tracking",
                extra={
                    'registry_type': self.__class__.__name__,
                    'thread_id': threading.current_thread().ident,
                    'initialization_time': threading.current_thread().name
                }
            )
    
    def clear(self) -> None:
        """Clear all registered items from registry.
        
        Thread-safe operation that removes all registered items.
        Primarily used for testing and cleanup scenarios.
        """
        with self._registry_lock:
            cleared_count = len(self._registry)
            self._registry.clear()
            self._priority_map.clear()
            self._priority_enum_map.clear()
            self._registration_metadata.clear()
            
            # Log comprehensive registry clear event
            logger.info(
                f"Cleared {cleared_count} items from {self.__class__.__name__} registry",
                extra={
                    'registry_type': self.__class__.__name__,
                    'cleared_count': cleared_count,
                    'thread_id': threading.current_thread().ident
                }
            )
    
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
    
    def get_registration_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive metadata for a registered item.
        
        Args:
            key: Registry key to get metadata for
            
        Returns:
            Registration metadata dictionary or None if not found
        """
        with self._registry_lock:
            return self._registration_metadata.get(key, {}).copy()
    
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all registered items.
        
        Returns:
            Dictionary mapping keys to their registration metadata
        """
        with self._registry_lock:
            return {k: v.copy() for k, v in self._registration_metadata.items()}
    
    def get_priority_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get priority information for a registered item.
        
        Args:
            key: Registry key to get priority info for
            
        Returns:
            Priority information dictionary
        """
        with self._registry_lock:
            if key not in self._registry:
                return None
            
            return {
                'numeric_priority': self._priority_map.get(key, 0),
                'priority_enum': self._priority_enum_map.get(key, RegistryPriority.BUILTIN),
                'priority_name': self._priority_enum_map.get(key, RegistryPriority.BUILTIN).name
            }
    
    def _store_registration_metadata(
        self, 
        key: str, 
        item: Any, 
        priority: int,
        priority_enum: RegistryPriority,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store comprehensive metadata for registered item.
        
        Args:
            key: Registry key
            item: Registered item
            priority: Numeric priority
            priority_enum: Priority enumeration
            additional_metadata: Additional metadata to store
        """
        import time
        
        metadata = {
            'registration_time': time.time(),
            'thread_id': threading.current_thread().ident,
            'thread_name': threading.current_thread().name,
            'numeric_priority': priority,
            'priority_enum': priority_enum,
            'priority_name': priority_enum.name,
            'item_type': type(item).__name__,
            'item_module': getattr(type(item), '__module__', 'unknown')
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        self._registration_metadata[key] = metadata

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
        priority: int = 0,
        priority_enum: Optional[RegistryPriority] = None,
        source: str = "api"
    ) -> None:
        """Register a loader for specific file extension with comprehensive lifecycle tracking.
        
        Args:
            extension: File extension including dot (e.g., '.pkl')
            loader_class: Loader class implementing BaseLoader protocol
            priority: Priority for this loader (higher = higher priority)
            priority_enum: Priority enumeration (auto-detected if not provided)
            source: Registration source ('api', 'plugin', 'entry_point', 'decorator')
            
        Raises:
            RegistryError: If registration fails due to conflicts or validation errors
            ValueError: If extension is invalid
            TypeError: If loader_class is not a valid loader implementation
        """
        # Validate extension format
        if not extension.startswith('.'):
            error = RegistryError(
                f"Extension must start with dot: {extension}",
                error_code="REGISTRY_001",
                context={'extension': extension, 'source': source}
            )
            logger.error(f"Extension validation failed: {error}")
            raise error
        
        # Validate loader implementation
        if not self._validate_item(loader_class):
            error = RegistryError(
                f"Loader {loader_class} must implement BaseLoader protocol",
                error_code="REGISTRY_001",
                context={
                    'loader_class': loader_class.__name__,
                    'extension': extension,
                    'source': source
                }
            )
            logger.error(f"Loader validation failed: {error}")
            raise error
        
        # Determine priority enum if not provided
        if priority_enum is None:
            priority_enum = RegistryPriority.from_priority_value(priority)
        
        with self._registry_lock:
            try:
                # Handle existing registrations with atomic conflict resolution
                registration_action = "registered"
                previous_loader = None
                
                if extension in self._registry:
                    current_priority = self._priority_map.get(extension, 0)
                    current_enum = self._priority_enum_map.get(extension, RegistryPriority.BUILTIN)
                    previous_loader = self._registry[extension]
                    
                    # Check priority-based resolution
                    if priority_enum < current_enum or (priority_enum == current_enum and priority <= current_priority):
                        # Log skip with detailed context
                        logger.warning(
                            f"Skipped registration of loader for {extension}: existing loader has higher priority",
                            extra={
                                'extension': extension,
                                'new_loader': loader_class.__name__,
                                'existing_loader': previous_loader.__name__ if previous_loader else 'unknown',
                                'new_priority': priority,
                                'existing_priority': current_priority,
                                'new_priority_enum': priority_enum.name,
                                'existing_priority_enum': current_enum.name,
                                'source': source,
                                'action': 'skipped'
                            }
                        )
                        return
                    else:
                        registration_action = "replaced"
                        logger.info(
                            f"Replacing existing loader for {extension} with higher priority loader",
                            extra={
                                'extension': extension,
                                'previous_loader': previous_loader.__name__ if previous_loader else 'unknown',
                                'new_loader': loader_class.__name__,
                                'previous_priority': current_priority,
                                'new_priority': priority,
                                'previous_priority_enum': current_enum.name,
                                'new_priority_enum': priority_enum.name,
                                'source': source
                            }
                        )
                
                # Perform atomic registration
                self._registry[extension] = loader_class
                self._priority_map[extension] = priority
                self._priority_enum_map[extension] = priority_enum
                
                # Store comprehensive metadata
                self._store_registration_metadata(
                    extension, 
                    loader_class, 
                    priority, 
                    priority_enum,
                    {
                        'source': source,
                        'extension': extension,
                        'loader_name': loader_class.__name__,
                        'previous_loader': previous_loader.__name__ if previous_loader else None,
                        'action': registration_action
                    }
                )
                
                # Log successful registration with comprehensive context
                logger.info(
                    f"Successfully {registration_action} loader for {extension}",
                    extra={
                        'extension': extension,
                        'loader_class': loader_class.__name__,
                        'priority': priority,
                        'priority_enum': priority_enum.name,
                        'source': source,
                        'action': registration_action,
                        'registry_size': len(self._registry)
                    }
                )
                
            except Exception as e:
                # Handle registration failures with comprehensive error context
                error = RegistryError(
                    f"Failed to register loader for {extension}: {str(e)}",
                    error_code="REGISTRY_001",
                    context={
                        'extension': extension,
                        'loader_class': loader_class.__name__,
                        'priority': priority,
                        'priority_enum': priority_enum.name,
                        'source': source,
                        'original_error': str(e)
                    }
                )
                logger.error(f"Loader registration failed: {error}")
                raise error
    
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
        """Unregister loader for specific extension with comprehensive lifecycle tracking.
        
        Args:
            extension: File extension to unregister
            
        Returns:
            True if loader was removed, False if not found
        """
        if not extension.startswith('.'):
            extension = f'.{extension}'
        
        with self._registry_lock:
            if extension in self._registry:
                # Capture metadata before removal
                removed_loader = self._registry[extension]
                metadata = self.get_registration_metadata(extension)
                
                # Perform atomic removal
                del self._registry[extension]
                self._priority_map.pop(extension, None)
                self._priority_enum_map.pop(extension, None)
                self._registration_metadata.pop(extension, None)
                
                # Log comprehensive unregistration event
                logger.info(
                    f"Unregistered loader for extension {extension}",
                    extra={
                        'extension': extension,
                        'removed_loader': removed_loader.__name__ if removed_loader else 'unknown',
                        'previous_metadata': metadata,
                        'remaining_count': len(self._registry),
                        'action': 'unregistered'
                    }
                )
                
                return True
                
            else:
                logger.warning(
                    f"Attempted to unregister non-existent loader for extension {extension}",
                    extra={
                        'extension': extension,
                        'action': 'unregister_failed'
                    }
                )
                return False
    
    def get_loader_capabilities(self, extension: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive capability information for a loader.
        
        Args:
            extension: File extension to get capabilities for
            
        Returns:
            Dictionary containing loader capabilities and metadata
        """
        if not extension.startswith('.'):
            extension = f'.{extension}'
        
        with self._registry_lock:
            loader_class = self._registry.get(extension)
            if not loader_class:
                return None
            
            try:
                # Create instance to introspect capabilities
                loader_instance = loader_class()
                
                capabilities = {
                    'extension': extension,
                    'loader_class': loader_class.__name__,
                    'loader_module': getattr(loader_class, '__module__', 'unknown'),
                    'supports_extension': getattr(loader_instance, 'supports_extension', lambda x: True)(extension),
                    'priority': self._priority_map.get(extension, 0),
                    'priority_enum': self._priority_enum_map.get(extension, RegistryPriority.BUILTIN),
                    'priority_name': self._priority_enum_map.get(extension, RegistryPriority.BUILTIN).name,
                    'registration_metadata': self.get_registration_metadata(extension),
                    'loader_priority': getattr(loader_instance, 'priority', 0) if hasattr(loader_instance, 'priority') else 0
                }
                
                # Add additional introspection if available
                if hasattr(loader_instance, '__dict__'):
                    capabilities['instance_attributes'] = list(loader_instance.__dict__.keys())
                
                return capabilities
                
            except Exception as e:
                logger.warning(
                    f"Failed to introspect loader capabilities for {extension}: {str(e)}",
                    extra={
                        'extension': extension,
                        'loader_class': loader_class.__name__ if loader_class else 'unknown',
                        'error': str(e)
                    }
                )
                return {
                    'extension': extension,
                    'loader_class': loader_class.__name__ if loader_class else 'unknown',
                    'error': str(e),
                    'capabilities_available': False
                }
    
    def get_all_loader_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Get capabilities for all registered loaders.
        
        Returns:
            Dictionary mapping extensions to their capability information
        """
        with self._registry_lock:
            capabilities = {}
            for extension in self._registry.keys():
                capability_info = self.get_loader_capabilities(extension)
                if capability_info:
                    capabilities[extension] = capability_info
            return capabilities
    
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
        """Discover and register plugins through entry points with comprehensive lifecycle tracking.
        
        Automatically discovers loaders registered through setuptools entry points
        under the 'flyrigloader.loaders' group with enhanced error handling and logging.
        """
        discovered_count = 0
        failed_count = 0
        
        try:
            # Log plugin discovery start
            logger.info(
                f"Starting plugin discovery for {self.__class__.__name__}",
                extra={
                    'registry_type': self.__class__.__name__,
                    'discovery_phase': 'start'
                }
            )
            
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
                    logger.info(
                        f"Loading plugin entry point: {entry_point.name}",
                        extra={
                            'entry_point_name': entry_point.name,
                            'entry_point_value': entry_point.value,
                            'discovery_phase': 'loading'
                        }
                    )
                    
                    loader_class = entry_point.load()
                    
                    # Entry point name should be the extension
                    extension = entry_point.name
                    if not extension.startswith('.'):
                        extension = f'.{extension}'
                    
                    # Register with PLUGIN priority for entry point discovered loaders
                    self.register_loader(
                        extension, 
                        loader_class, 
                        priority=RegistryPriority.PLUGIN.value,
                        priority_enum=RegistryPriority.PLUGIN,
                        source="entry_point"
                    )
                    
                    discovered_count += 1
                    
                    logger.info(
                        f"Successfully loaded plugin loader: {entry_point.name}",
                        extra={
                            'entry_point_name': entry_point.name,
                            'loader_class': loader_class.__name__,
                            'extension': extension,
                            'discovery_phase': 'success'
                        }
                    )
                    
                except Exception as e:
                    failed_count += 1
                    logger.warning(
                        f"Failed to load plugin loader {entry_point.name}: {str(e)}",
                        extra={
                            'entry_point_name': entry_point.name,
                            'error': str(e),
                            'error_type': type(e).__name__,
                            'discovery_phase': 'error'
                        }
                    )
                    
                    warnings.warn(
                        f"Failed to load plugin loader {entry_point.name}: {e}",
                        UserWarning
                    )
                    
        except Exception as e:
            # Plugin discovery failures shouldn't break the registry
            logger.error(
                f"Plugin discovery failed for {self.__class__.__name__}: {str(e)}",
                extra={
                    'registry_type': self.__class__.__name__,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'discovery_phase': 'failed'
                }
            )
            warnings.warn(f"Plugin discovery failed: {e}", UserWarning)
        
        finally:
            # Log comprehensive discovery summary
            logger.info(
                f"Plugin discovery completed for {self.__class__.__name__}",
                extra={
                    'registry_type': self.__class__.__name__,
                    'discovered_count': discovered_count,
                    'failed_count': failed_count,
                    'total_registered': len(self._registry),
                    'discovery_phase': 'complete'
                }
            )


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
        priority: int = 0,
        priority_enum: Optional[RegistryPriority] = None,
        source: str = "api"
    ) -> None:
        """Register a schema with comprehensive lifecycle tracking.
        
        Args:
            name: Schema name for lookup
            schema_class: Schema class implementing BaseSchema protocol
            priority: Priority for this schema (higher = higher priority)
            priority_enum: Priority enumeration (auto-detected if not provided)
            source: Registration source ('api', 'plugin', 'entry_point', 'decorator')
            
        Raises:
            RegistryError: If registration fails due to conflicts or validation errors
            ValueError: If name is invalid
            TypeError: If schema_class is not a valid schema implementation
        """
        # Validate schema name
        if not name or not isinstance(name, str):
            error = RegistryError(
                f"Schema name must be non-empty string: {name}",
                error_code="REGISTRY_001",
                context={'schema_name': name, 'source': source}
            )
            logger.error(f"Schema name validation failed: {error}")
            raise error
        
        # Validate schema implementation
        if not self._validate_item(schema_class):
            error = RegistryError(
                f"Schema {schema_class} must implement BaseSchema protocol",
                error_code="REGISTRY_001",
                context={
                    'schema_class': schema_class.__name__,
                    'schema_name': name,
                    'source': source
                }
            )
            logger.error(f"Schema validation failed: {error}")
            raise error
        
        # Determine priority enum if not provided
        if priority_enum is None:
            priority_enum = RegistryPriority.from_priority_value(priority)
        
        with self._registry_lock:
            try:
                # Handle existing registrations with atomic conflict resolution
                registration_action = "registered"
                previous_schema = None
                
                if name in self._registry:
                    current_priority = self._priority_map.get(name, 0)
                    current_enum = self._priority_enum_map.get(name, RegistryPriority.BUILTIN)
                    previous_schema = self._registry[name]
                    
                    # Check priority-based resolution
                    if priority_enum < current_enum or (priority_enum == current_enum and priority <= current_priority):
                        # Log skip with detailed context
                        logger.warning(
                            f"Skipped registration of schema {name}: existing schema has higher priority",
                            extra={
                                'schema_name': name,
                                'new_schema': schema_class.__name__,
                                'existing_schema': previous_schema.__name__ if previous_schema else 'unknown',
                                'new_priority': priority,
                                'existing_priority': current_priority,
                                'new_priority_enum': priority_enum.name,
                                'existing_priority_enum': current_enum.name,
                                'source': source,
                                'action': 'skipped'
                            }
                        )
                        return
                    else:
                        registration_action = "replaced"
                        logger.info(
                            f"Replacing existing schema {name} with higher priority schema",
                            extra={
                                'schema_name': name,
                                'previous_schema': previous_schema.__name__ if previous_schema else 'unknown',
                                'new_schema': schema_class.__name__,
                                'previous_priority': current_priority,
                                'new_priority': priority,
                                'previous_priority_enum': current_enum.name,
                                'new_priority_enum': priority_enum.name,
                                'source': source
                            }
                        )
                
                # Perform atomic registration
                self._registry[name] = schema_class
                self._priority_map[name] = priority
                self._priority_enum_map[name] = priority_enum
                
                # Store comprehensive metadata
                self._store_registration_metadata(
                    name, 
                    schema_class, 
                    priority, 
                    priority_enum,
                    {
                        'source': source,
                        'schema_name': name,
                        'schema_class_name': schema_class.__name__,
                        'previous_schema': previous_schema.__name__ if previous_schema else None,
                        'action': registration_action
                    }
                )
                
                # Log successful registration with comprehensive context
                logger.info(
                    f"Successfully {registration_action} schema {name}",
                    extra={
                        'schema_name': name,
                        'schema_class': schema_class.__name__,
                        'priority': priority,
                        'priority_enum': priority_enum.name,
                        'source': source,
                        'action': registration_action,
                        'registry_size': len(self._registry)
                    }
                )
                
            except Exception as e:
                # Handle registration failures with comprehensive error context
                error = RegistryError(
                    f"Failed to register schema {name}: {str(e)}",
                    error_code="REGISTRY_001",
                    context={
                        'schema_name': name,
                        'schema_class': schema_class.__name__,
                        'priority': priority,
                        'priority_enum': priority_enum.name,
                        'source': source,
                        'original_error': str(e)
                    }
                )
                logger.error(f"Schema registration failed: {error}")
                raise error
    
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
        """Unregister schema by name with comprehensive lifecycle tracking.
        
        Args:
            name: Schema name to unregister
            
        Returns:
            True if schema was removed, False if not found
        """
        with self._registry_lock:
            if name in self._registry:
                # Capture metadata before removal
                removed_schema = self._registry[name]
                metadata = self.get_registration_metadata(name)
                
                # Perform atomic removal
                del self._registry[name]
                self._priority_map.pop(name, None)
                self._priority_enum_map.pop(name, None)
                self._registration_metadata.pop(name, None)
                
                # Log comprehensive unregistration event
                logger.info(
                    f"Unregistered schema {name}",
                    extra={
                        'schema_name': name,
                        'removed_schema': removed_schema.__name__ if removed_schema else 'unknown',
                        'previous_metadata': metadata,
                        'remaining_count': len(self._registry),
                        'action': 'unregistered'
                    }
                )
                
                return True
                
            else:
                logger.warning(
                    f"Attempted to unregister non-existent schema {name}",
                    extra={
                        'schema_name': name,
                        'action': 'unregister_failed'
                    }
                )
                return False
    
    def get_schema_capabilities(self, name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive capability information for a schema.
        
        Args:
            name: Schema name to get capabilities for
            
        Returns:
            Dictionary containing schema capabilities and metadata
        """
        with self._registry_lock:
            schema_class = self._registry.get(name)
            if not schema_class:
                return None
            
            try:
                # Create instance to introspect capabilities
                schema_instance = schema_class()
                
                capabilities = {
                    'schema_name': name,
                    'schema_class': schema_class.__name__,
                    'schema_module': getattr(schema_class, '__module__', 'unknown'),
                    'priority': self._priority_map.get(name, 0),
                    'priority_enum': self._priority_enum_map.get(name, RegistryPriority.BUILTIN),
                    'priority_name': self._priority_enum_map.get(name, RegistryPriority.BUILTIN).name,
                    'registration_metadata': self.get_registration_metadata(name),
                    'schema_name_property': getattr(schema_instance, 'schema_name', name) if hasattr(schema_instance, 'schema_name') else name,
                    'supported_types': getattr(schema_instance, 'supported_types', []) if hasattr(schema_instance, 'supported_types') else []
                }
                
                # Add additional introspection if available
                if hasattr(schema_instance, '__dict__'):
                    capabilities['instance_attributes'] = list(schema_instance.__dict__.keys())
                
                return capabilities
                
            except Exception as e:
                logger.warning(
                    f"Failed to introspect schema capabilities for {name}: {str(e)}",
                    extra={
                        'schema_name': name,
                        'schema_class': schema_class.__name__ if schema_class else 'unknown',
                        'error': str(e)
                    }
                )
                return {
                    'schema_name': name,
                    'schema_class': schema_class.__name__ if schema_class else 'unknown',
                    'error': str(e),
                    'capabilities_available': False
                }
    
    def get_all_schema_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Get capabilities for all registered schemas.
        
        Returns:
            Dictionary mapping schema names to their capability information
        """
        with self._registry_lock:
            capabilities = {}
            for name in self._registry.keys():
                capability_info = self.get_schema_capabilities(name)
                if capability_info:
                    capabilities[name] = capability_info
            return capabilities
    
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
        """Discover and register plugins through entry points with comprehensive lifecycle tracking.
        
        Automatically discovers schemas registered through setuptools entry points
        under the 'flyrigloader.schemas' group with enhanced error handling and logging.
        """
        discovered_count = 0
        failed_count = 0
        
        try:
            # Log plugin discovery start
            logger.info(
                f"Starting plugin discovery for {self.__class__.__name__}",
                extra={
                    'registry_type': self.__class__.__name__,
                    'discovery_phase': 'start'
                }
            )
            
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
                    logger.info(
                        f"Loading plugin entry point: {entry_point.name}",
                        extra={
                            'entry_point_name': entry_point.name,
                            'entry_point_value': entry_point.value,
                            'discovery_phase': 'loading'
                        }
                    )
                    
                    schema_class = entry_point.load()
                    # Entry point name should be the schema name
                    schema_name = entry_point.name
                    
                    # Register with PLUGIN priority for entry point discovered schemas
                    self.register_schema(
                        schema_name, 
                        schema_class, 
                        priority=RegistryPriority.PLUGIN.value,
                        priority_enum=RegistryPriority.PLUGIN,
                        source="entry_point"
                    )
                    
                    discovered_count += 1
                    
                    logger.info(
                        f"Successfully loaded plugin schema: {entry_point.name}",
                        extra={
                            'entry_point_name': entry_point.name,
                            'schema_class': schema_class.__name__,
                            'schema_name': schema_name,
                            'discovery_phase': 'success'
                        }
                    )
                    
                except Exception as e:
                    failed_count += 1
                    logger.warning(
                        f"Failed to load plugin schema {entry_point.name}: {str(e)}",
                        extra={
                            'entry_point_name': entry_point.name,
                            'error': str(e),
                            'error_type': type(e).__name__,
                            'discovery_phase': 'error'
                        }
                    )
                    
                    warnings.warn(
                        f"Failed to load plugin schema {entry_point.name}: {e}",
                        UserWarning
                    )
                    
        except Exception as e:
            # Plugin discovery failures shouldn't break the registry
            logger.error(
                f"Plugin discovery failed for {self.__class__.__name__}: {str(e)}",
                extra={
                    'registry_type': self.__class__.__name__,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'discovery_phase': 'failed'
                }
            )
            warnings.warn(f"Plugin discovery failed: {e}", UserWarning)
        
        finally:
            # Log comprehensive discovery summary
            logger.info(
                f"Plugin discovery completed for {self.__class__.__name__}",
                extra={
                    'registry_type': self.__class__.__name__,
                    'discovered_count': discovered_count,
                    'failed_count': failed_count,
                    'total_registered': len(self._registry),
                    'discovery_phase': 'complete'
                }
            )


# Utility functions for registry management
def register_loader(
    extension: str, 
    loader_class: Type[BaseLoader], 
    priority: int = 0,
    priority_enum: Optional[RegistryPriority] = None
) -> None:
    """Convenience function to register a loader with enhanced priority support.
    
    Args:
        extension: File extension including dot (e.g., '.pkl')
        loader_class: Loader class implementing BaseLoader protocol
        priority: Priority for this loader (higher = higher priority)
        priority_enum: Priority enumeration (auto-detected if not provided)
    """
    registry = LoaderRegistry()
    registry.register_loader(extension, loader_class, priority, priority_enum, source="api")


def register_schema(
    name: str, 
    schema_class: Type[BaseSchema], 
    priority: int = 0,
    priority_enum: Optional[RegistryPriority] = None
) -> None:
    """Convenience function to register a schema with enhanced priority support.
    
    Args:
        name: Schema name for lookup
        schema_class: Schema class implementing BaseSchema protocol
        priority: Priority for this schema (higher = higher priority)
        priority_enum: Priority enumeration (auto-detected if not provided)
    """
    registry = SchemaRegistry()
    registry.register_schema(name, schema_class, priority, priority_enum, source="api")


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


def get_registered_loaders() -> Dict[str, Dict[str, Any]]:
    """Get comprehensive information about all registered loaders.
    
    This function provides registry introspection capabilities to expose
    loader metadata per Section 5.2.1 requirements.
    
    Returns:
        Dictionary mapping extensions to loader information including:
        - loader_class: Name of the loader class
        - priority: Numeric priority value
        - priority_enum: Priority enumeration level
        - registration_metadata: Comprehensive registration context
        - capabilities: Loader capability information when available
    """
    registry = LoaderRegistry()
    
    loaders_info = {}
    with registry._registry_lock:
        for extension, loader_class in registry._registry.items():
            loader_info = {
                'extension': extension,
                'loader_class': loader_class.__name__,
                'loader_module': getattr(loader_class, '__module__', 'unknown'),
                'priority': registry._priority_map.get(extension, 0),
                'priority_enum': registry._priority_enum_map.get(extension, RegistryPriority.BUILTIN),
                'priority_name': registry._priority_enum_map.get(extension, RegistryPriority.BUILTIN).name,
                'registration_metadata': registry.get_registration_metadata(extension),
                'capabilities': registry.get_loader_capabilities(extension)
            }
            loaders_info[extension] = loader_info
    
    logger.info(
        f"Retrieved information for {len(loaders_info)} registered loaders",
        extra={
            'loader_count': len(loaders_info),
            'extensions': list(loaders_info.keys())
        }
    )
    
    return loaders_info


def get_loader_capabilities(extension: str) -> Optional[Dict[str, Any]]:
    """Get comprehensive capability information for a specific loader.
    
    This function exposes loader metadata introspection per Section 0.2.1
    registry architecture requirements.
    
    Args:
        extension: File extension to get capabilities for (e.g., '.pkl')
        
    Returns:
        Dictionary containing loader capabilities and metadata, or None if not found
    """
    registry = LoaderRegistry()
    capabilities = registry.get_loader_capabilities(extension)
    
    if capabilities:
        logger.info(
            f"Retrieved capabilities for loader {extension}",
            extra={
                'extension': extension,
                'loader_class': capabilities.get('loader_class', 'unknown')
            }
        )
    else:
        logger.warning(
            f"No capabilities found for loader {extension}",
            extra={'extension': extension}
        )
    
    return capabilities


def get_registered_schemas() -> Dict[str, Dict[str, Any]]:
    """Get comprehensive information about all registered schemas.
    
    Returns:
        Dictionary mapping schema names to schema information including:
        - schema_class: Name of the schema class
        - priority: Numeric priority value
        - priority_enum: Priority enumeration level
        - registration_metadata: Comprehensive registration context
        - capabilities: Schema capability information when available
    """
    registry = SchemaRegistry()
    
    schemas_info = {}
    with registry._registry_lock:
        for name, schema_class in registry._registry.items():
            schema_info = {
                'schema_name': name,
                'schema_class': schema_class.__name__,
                'schema_module': getattr(schema_class, '__module__', 'unknown'),
                'priority': registry._priority_map.get(name, 0),
                'priority_enum': registry._priority_enum_map.get(name, RegistryPriority.BUILTIN),
                'priority_name': registry._priority_enum_map.get(name, RegistryPriority.BUILTIN).name,
                'registration_metadata': registry.get_registration_metadata(name),
                'capabilities': registry.get_schema_capabilities(name)
            }
            schemas_info[name] = schema_info
    
    logger.info(
        f"Retrieved information for {len(schemas_info)} registered schemas",
        extra={
            'schema_count': len(schemas_info),
            'schema_names': list(schemas_info.keys())
        }
    )
    
    return schemas_info


# Registry decorators for automatic registration
def loader_for(extension: str, priority: int = 0, priority_enum: Optional[RegistryPriority] = None):
    """Decorator to automatically register loader for extension with enhanced priority support.
    
    Args:
        extension: File extension including dot (e.g., '.pkl')
        priority: Priority for this loader (higher = higher priority)
        priority_enum: Priority enumeration (auto-detected if not provided)
        
    Example:
        @loader_for('.pkl', priority=10)
        class PickleLoader:
            def load(self, path: Path) -> Any:
                # Implementation
                pass
    """
    def decorator(cls):
        register_loader(extension, cls, priority, priority_enum)
        return cls
    return decorator


def schema_for(name: str, priority: int = 0, priority_enum: Optional[RegistryPriority] = None):
    """Decorator to automatically register schema by name with enhanced priority support.
    
    Args:
        name: Schema name for lookup
        priority: Priority for this schema (higher = higher priority)
        priority_enum: Priority enumeration (auto-detected if not provided)
        
    Example:
        @schema_for('experiment', priority=10)
        class ExperimentSchema:
            def validate(self, data: Any) -> Dict[str, Any]:
                # Implementation
                pass
    """
    def decorator(cls):
        register_schema(name, cls, priority, priority_enum)
        return cls
    return decorator


def auto_register(
    registry_type: str = "auto",
    key: Optional[str] = None,
    priority: int = 0,
    priority_enum: Optional[RegistryPriority] = None
):
    """Auto-registration decorator for automatic entry-point discovery registration.
    
    This decorator enables zero-code plugin extensions by automatically registering
    classes with appropriate registries based on their implementation and entry-point
    discovery per Section 0.3.1 requirements.
    
    Args:
        registry_type: Type of registry ('loader', 'schema', 'auto')
        key: Registration key (extension for loaders, name for schemas)
        priority: Priority for registration (higher = higher priority)
        priority_enum: Priority enumeration (auto-detected if not provided)
        
    Returns:
        Decorator function that registers the class automatically
        
    Example:
        @auto_register(registry_type="loader", key=".pkl", priority=20)
        class MyPickleLoader:
            def load(self, path: Path) -> Any:
                # Implementation
                pass
                
        @auto_register(registry_type="schema", key="experiment")
        class MyExperimentSchema:
            def validate(self, data: Any) -> Dict[str, Any]:
                # Implementation
                pass
                
        @auto_register()  # Auto-detect type and key
        class AutoDetectedLoader:
            def load(self, path: Path) -> Any:
                # Implementation - will auto-detect as loader
                pass
    """
    def decorator(cls):
        try:
            # Determine priority enum if not provided
            effective_priority_enum = priority_enum or RegistryPriority.from_priority_value(priority)
            
            # Auto-detect registry type if needed
            if registry_type == "auto":
                detected_type = _auto_detect_registry_type(cls)
                logger.info(
                    f"Auto-detected registry type '{detected_type}' for class {cls.__name__}",
                    extra={
                        'class_name': cls.__name__,
                        'detected_type': detected_type,
                        'auto_registration': True
                    }
                )
            else:
                detected_type = registry_type
            
            # Register based on detected or specified type
            if detected_type == "loader":
                # Auto-detect extension if not provided
                effective_key = key or _auto_detect_loader_extension(cls)
                
                if effective_key:
                    register_loader(
                        effective_key, 
                        cls, 
                        priority, 
                        effective_priority_enum
                    )
                    logger.info(
                        f"Auto-registered loader {cls.__name__} for extension {effective_key}",
                        extra={
                            'class_name': cls.__name__,
                            'extension': effective_key,
                            'priority': priority,
                            'priority_enum': effective_priority_enum.name,
                            'auto_registration': True
                        }
                    )
                else:
                    logger.warning(
                        f"Could not auto-detect extension for loader {cls.__name__}",
                        extra={'class_name': cls.__name__, 'auto_registration': True}
                    )
                    
            elif detected_type == "schema":
                # Auto-detect schema name if not provided
                effective_key = key or _auto_detect_schema_name(cls)
                
                if effective_key:
                    register_schema(
                        effective_key, 
                        cls, 
                        priority, 
                        effective_priority_enum
                    )
                    logger.info(
                        f"Auto-registered schema {cls.__name__} with name {effective_key}",
                        extra={
                            'class_name': cls.__name__,
                            'schema_name': effective_key,
                            'priority': priority,
                            'priority_enum': effective_priority_enum.name,
                            'auto_registration': True
                        }
                    )
                else:
                    logger.warning(
                        f"Could not auto-detect name for schema {cls.__name__}",
                        extra={'class_name': cls.__name__, 'auto_registration': True}
                    )
            else:
                error = RegistryError(
                    f"Unknown registry type '{detected_type}' for auto-registration of {cls.__name__}",
                    error_code="REGISTRY_009",
                    context={
                        'class_name': cls.__name__,
                        'registry_type': detected_type,
                        'auto_registration': True
                    }
                )
                logger.error(f"Auto-registration failed: {error}")
                raise error
                
        except Exception as e:
            error = RegistryError(
                f"Auto-registration decorator failed for {cls.__name__}: {str(e)}",
                error_code="REGISTRY_009",
                context={
                    'class_name': cls.__name__,
                    'registry_type': registry_type,
                    'key': key,
                    'original_error': str(e),
                    'auto_registration': True
                }
            )
            logger.error(f"Auto-registration decorator failed: {error}")
            raise error
        
        return cls
    return decorator


def _auto_detect_registry_type(cls) -> str:
    """Auto-detect registry type based on class interface.
    
    Args:
        cls: Class to analyze
        
    Returns:
        Detected registry type ('loader' or 'schema')
    """
    # Check for loader interface
    if (hasattr(cls, 'load') and hasattr(cls, 'supports_extension') and 
        hasattr(cls, 'priority')):
        return "loader"
    
    # Check for schema interface
    if (hasattr(cls, 'validate') and hasattr(cls, 'schema_name') and 
        hasattr(cls, 'supported_types')):
        return "schema"
    
    # Default to loader if unclear
    return "loader"


def _auto_detect_loader_extension(cls) -> Optional[str]:
    """Auto-detect file extension for loader class.
    
    Args:
        cls: Loader class to analyze
        
    Returns:
        Detected file extension or None
    """
    # Try to extract from class name
    class_name = cls.__name__.lower()
    
    # Common patterns
    if 'pickle' in class_name or 'pkl' in class_name:
        return '.pkl'
    elif 'json' in class_name:
        return '.json'
    elif 'csv' in class_name:
        return '.csv'
    elif 'yaml' in class_name or 'yml' in class_name:
        return '.yaml'
    elif 'parquet' in class_name:
        return '.parquet'
    
    # Try to inspect class attributes or methods for hints
    try:
        instance = cls()
        if hasattr(instance, 'extension'):
            return instance.extension
        if hasattr(instance, 'file_extension'):
            return instance.file_extension
        if hasattr(instance, 'supported_extensions') and instance.supported_extensions:
            return instance.supported_extensions[0]
    except Exception:
        pass
    
    return None


def _auto_detect_schema_name(cls) -> Optional[str]:
    """Auto-detect schema name for schema class.
    
    Args:
        cls: Schema class to analyze
        
    Returns:
        Detected schema name or None
    """
    # Try to extract from class name
    class_name = cls.__name__
    
    # Remove common suffixes
    if class_name.endswith('Schema'):
        name = class_name[:-6]  # Remove 'Schema'
    elif class_name.endswith('Validator'):
        name = class_name[:-9]  # Remove 'Validator'
    else:
        name = class_name
    
    # Convert to lowercase and snake_case
    import re
    name = re.sub('([A-Z]+)', r'_\1', name).lower().strip('_')
    
    # Try to inspect class attributes for explicit name
    try:
        instance = cls()
        if hasattr(instance, 'schema_name') and instance.schema_name:
            return instance.schema_name
        if hasattr(instance, 'name') and instance.name:
            return instance.name
    except Exception:
        pass
    
    return name if name else None


# Export main classes and functions
__all__ = [
    # Core registry classes
    'LoaderRegistry',
    'SchemaRegistry',
    'BaseLoader',
    'BaseSchema',
    'BaseRegistry',
    
    # Priority enumeration
    'RegistryPriority',
    
    # Registration functions
    'register_loader',
    'register_schema',
    
    # Lookup functions
    'get_loader_for_extension',
    'get_schema',
    
    # Registry introspection functions (new exports)
    'get_registered_loaders',
    'get_loader_capabilities',
    'get_registered_schemas',
    
    # Registration decorators
    'loader_for',
    'schema_for',
    'auto_register',  # New export
    
    # Exception handling
    'RegistryError'
]