# FlyRigLoader Extension Guide

This guide provides comprehensive documentation for extending FlyRigLoader through registry-based patterns, enabling plugin-style extensibility without modifying core code.

## Table of Contents

1. [Overview](#overview)
2. [Registry Architecture](#registry-architecture)
3. [Creating Custom Loaders](#creating-custom-loaders)
4. [Creating Custom Schemas](#creating-custom-schemas)
5. [Plugin Discovery Mechanism](#plugin-discovery-mechanism)
6. [Third-Party Extension Development](#third-party-extension-development)
7. [Advanced Registration Patterns](#advanced-registration-patterns)
8. [Testing Extensions](#testing-extensions)
9. [Best Practices](#best-practices)

## Overview

FlyRigLoader implements a registry-based extensibility pattern that aligns with the SOLID Open/Closed principle, allowing the system to be open for extension but closed for modification. This architecture enables:

> **First time extending the loader?** Start with the [Architecture Overview](architecture.md) to understand how discovery, IO, and registries cooperate before wiring a new plugin.

- **Plugin-style extensibility**: New file formats and validation schemas can be added without modifying core code
- **Thread-safe singleton implementation**: All registries use thread-safe singleton patterns with O(1) lookup performance
- **Automatic plugin discovery**: Extensions can be discovered automatically through setuptools entry points
- **Runtime registration**: Dynamic registration of handlers at runtime
- **Priority-based ordering**: Multiple handlers can coexist with priority-based selection

### Key Benefits

- **Separation of concerns**: Clean separation between discovery, loading, and transformation
- **Testability**: All extension points support dependency injection for testing
- **Performance**: O(1) lookup performance for all registry operations
- **Extensibility**: Plugin architecture enables third-party extensions
- **Maintainability**: Clear interfaces and protocols for all extension points

## Registry Architecture

FlyRigLoader provides two primary registries for extensibility:

### LoaderRegistry

The `LoaderRegistry` manages file format handlers, enabling automatic format detection and pluggable loader registration.

```python
from flyrigloader.registries import LoaderRegistry

# Get the singleton instance
registry = LoaderRegistry()

# Register a new loader
registry.register_loader('.custom', CustomLoader, priority=10)

# Get loader for extension
loader_class = registry.get_loader_for_extension('.pkl')

# Get all registered loaders
all_loaders = registry.get_all_loaders()
```

### SchemaRegistry

The `SchemaRegistry` manages column validation schemas, enabling dynamic schema registration and validation.

```python
from flyrigloader.registries import SchemaRegistry

# Get the singleton instance
registry = SchemaRegistry()

# Register a new schema
registry.register_schema('experiment', ExperimentSchema, priority=10)

# Get schema by name
schema_class = registry.get_schema('experiment')

# Get all registered schemas
all_schemas = registry.get_all_schemas()
```

### Thread-Safe Singleton Implementation

Both registries implement thread-safe singleton patterns to ensure consistent state across the application:

```python
# These will return the same instance
registry1 = LoaderRegistry()
registry2 = LoaderRegistry()
assert registry1 is registry2  # True

# Thread-safe access
import threading

def worker():
    registry = LoaderRegistry()
    registry.register_loader('.worker', WorkerLoader)

# Multiple threads can safely access the registry
threads = [threading.Thread(target=worker) for _ in range(5)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
```

## Creating Custom Loaders

To create a custom loader, implement the `BaseLoader` protocol:

### BaseLoader Protocol

```python
from typing import Protocol, runtime_checkable
from pathlib import Path
from abc import abstractmethod

@runtime_checkable
class BaseLoader(Protocol):
    """Protocol interface for custom loader implementations."""
    
    def load(self, path: Path) -> Any:
        """Load raw data from file without transformation."""
        ...
    
    def supports_extension(self, extension: str) -> bool:
        """Check if loader supports given file extension."""
        ...
    
    @property
    def priority(self) -> int:
        """Priority for this loader (higher values = higher priority)."""
        ...
```

### Example Custom Loader Implementation

```python
from flyrigloader.registries import BaseLoader, LoaderRegistry
from flyrigloader.exceptions import LoadError
from pathlib import Path
import json

class JsonLoader:
    """Custom loader for JSON files."""
    
    def load(self, path: Path) -> dict:
        """Load JSON data from file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise LoadError(
                f"Failed to parse JSON file: {path}",
                error_code="LOAD_004",
                context={
                    "file_path": str(path),
                    "json_error": str(e)
                }
            )
        except Exception as e:
            raise LoadError(
                f"Failed to load JSON file: {path}",
                error_code="LOAD_001",
                context={
                    "file_path": str(path),
                    "error": str(e)
                }
            )
    
    def supports_extension(self, extension: str) -> bool:
        """Check if this loader supports JSON files."""
        return extension.lower() in ['.json', '.jsonl']
    
    @property
    def priority(self) -> int:
        """Priority for JSON loader."""
        return 5  # Medium priority

# Register the loader
registry = LoaderRegistry()
registry.register_loader('.json', JsonLoader, priority=5)
```

### Decorator-Based Registration

Use the `@loader_for` decorator for automatic registration:

```python
from flyrigloader.registries import loader_for

@loader_for('.json', priority=5)
class JsonLoader:
    """Custom JSON loader with automatic registration."""
    
    def load(self, path: Path) -> dict:
        with open(path, 'r') as f:
            return json.load(f)
    
    def supports_extension(self, extension: str) -> bool:
        return extension.lower() in ['.json', '.jsonl']
    
    @property
    def priority(self) -> int:
        return 5
```

### Advanced Loader Example

```python
from flyrigloader.registries import BaseLoader, register_loader
from flyrigloader.exceptions import LoadError
import h5py
import numpy as np

class HDF5Loader:
    """Advanced loader for HDF5 files with error handling."""
    
    def __init__(self):
        self.supported_extensions = {'.h5', '.hdf5', '.hdf'}
    
    def load(self, path: Path) -> dict:
        """Load HDF5 data with comprehensive error handling."""
        try:
            data = {}
            with h5py.File(path, 'r') as f:
                # Recursively load all datasets
                data = self._load_group(f)
            return data
        except OSError as e:
            raise LoadError(
                f"Cannot open HDF5 file: {path}",
                error_code="LOAD_002",
                context={
                    "file_path": str(path),
                    "hdf5_error": str(e)
                }
            )
    
    def _load_group(self, group) -> dict:
        """Recursively load HDF5 groups and datasets."""
        result = {}
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                result[key] = self._load_group(item)
            elif isinstance(item, h5py.Dataset):
                result[key] = item[:]
        return result
    
    def supports_extension(self, extension: str) -> bool:
        return extension.lower() in self.supported_extensions
    
    @property
    def priority(self) -> int:
        return 15  # High priority for HDF5 files

# Register with explicit priority
register_loader('.h5', HDF5Loader, priority=15)
register_loader('.hdf5', HDF5Loader, priority=15)
```

## Creating Custom Schemas

To create a custom schema, implement the `BaseSchema` protocol:

### BaseSchema Protocol

```python
from typing import Protocol, runtime_checkable, Dict, Any, List
from abc import abstractmethod

@runtime_checkable
class BaseSchema(Protocol):
    """Protocol interface for custom schema validators."""
    
    def validate(self, data: Any) -> Dict[str, Any]:
        """Validate data against schema and return validated result."""
        ...
    
    @property
    def schema_name(self) -> str:
        """Name identifying this schema."""
        ...
    
    @property
    def supported_types(self) -> List[str]:
        """List of data types this schema can validate."""
        ...
```

### Example Custom Schema Implementation

```python
from flyrigloader.registries import BaseSchema, SchemaRegistry
from flyrigloader.exceptions import TransformError
from typing import Dict, Any, List
import numpy as np

class ExperimentSchema:
    """Custom schema for experiment data validation."""
    
    def __init__(self):
        self._name = 'experiment'
        self._types = ['experiment', 'behavioral', 'neural']
    
    def validate(self, data: Any) -> Dict[str, Any]:
        """Validate experiment data format."""
        if not isinstance(data, dict):
            raise TransformError(
                "Experiment data must be a dictionary",
                error_code="TRANSFORM_001",
                context={
                    "data_type": type(data).__name__,
                    "schema_name": self._name
                }
            )
        
        # Required fields for experiment schema
        required_fields = ['timestamp', 'subject_id', 'session_id']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            raise TransformError(
                f"Missing required fields: {missing_fields}",
                error_code="TRANSFORM_006",
                context={
                    "missing_fields": missing_fields,
                    "available_fields": list(data.keys()),
                    "schema_name": self._name
                }
            )
        
        # Validate timestamp format
        if not isinstance(data['timestamp'], (int, float, np.number)):
            raise TransformError(
                "Timestamp must be numeric",
                error_code="TRANSFORM_002",
                context={
                    "timestamp_type": type(data['timestamp']).__name__,
                    "schema_name": self._name
                }
            )
        
        # Return validated data with normalized timestamp
        validated_data = data.copy()
        validated_data['timestamp'] = float(data['timestamp'])
        
        return validated_data
    
    @property
    def schema_name(self) -> str:
        return self._name
    
    @property
    def supported_types(self) -> List[str]:
        return self._types.copy()

# Register the schema
registry = SchemaRegistry()
registry.register_schema('experiment', ExperimentSchema, priority=10)
```

### Decorator-Based Schema Registration

```python
from flyrigloader.registries import schema_for

@schema_for('behavioral', priority=8)
class BehavioralSchema:
    """Schema for behavioral experiment data."""
    
    def validate(self, data: Any) -> Dict[str, Any]:
        # Validation logic specific to behavioral data
        if not isinstance(data, dict):
            raise TransformError("Behavioral data must be a dictionary")
        
        # Validate behavioral-specific fields
        required_fields = ['trial_data', 'stimulus_info', 'response_data']
        for field in required_fields:
            if field not in data:
                raise TransformError(f"Missing required field: {field}")
        
        return data
    
    @property
    def schema_name(self) -> str:
        return 'behavioral'
    
    @property
    def supported_types(self) -> List[str]:
        return ['behavioral', 'trial', 'stimulus']
```

### Advanced Schema with Pydantic Integration

```python
from flyrigloader.registries import BaseSchema, register_schema
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
import numpy as np

class NeuralDataModel(BaseModel):
    """Pydantic model for neural data validation."""
    
    spike_times: List[float] = Field(..., description="Spike timestamps")
    unit_id: int = Field(..., ge=0, description="Unit identifier")
    channel: int = Field(..., ge=0, description="Recording channel")
    sampling_rate: float = Field(..., gt=0, description="Sampling rate in Hz")
    recording_duration: float = Field(..., gt=0, description="Recording duration in seconds")
    
    @validator('spike_times')
    def validate_spike_times(cls, v):
        if not all(isinstance(t, (int, float)) for t in v):
            raise ValueError("All spike times must be numeric")
        if not all(t >= 0 for t in v):
            raise ValueError("All spike times must be non-negative")
        return sorted(v)  # Ensure sorted order

class NeuralSchema:
    """Advanced schema using Pydantic for neural data validation."""
    
    def validate(self, data: Any) -> Dict[str, Any]:
        """Validate neural data using Pydantic model."""
        try:
            # Convert numpy arrays to lists if needed
            if isinstance(data.get('spike_times'), np.ndarray):
                data['spike_times'] = data['spike_times'].tolist()
            
            # Validate using Pydantic model
            validated = NeuralDataModel(**data)
            return validated.dict()
            
        except Exception as e:
            raise TransformError(
                f"Neural data validation failed: {str(e)}",
                error_code="TRANSFORM_001",
                context={
                    "validation_error": str(e),
                    "schema_name": "neural"
                }
            )
    
    @property
    def schema_name(self) -> str:
        return 'neural'
    
    @property
    def supported_types(self) -> List[str]:
        return ['neural', 'spike', 'electrophysiology']

# Register the advanced schema
register_schema('neural', NeuralSchema, priority=12)
```

## Plugin Discovery Mechanism

FlyRigLoader supports automatic plugin discovery through setuptools entry points, enabling third-party packages to register extensions automatically.

### Entry Points Configuration

In your plugin package's `setup.py` or `pyproject.toml`, define entry points:

#### setup.py

```python
from setuptools import setup

setup(
    name='flyrigloader-json-plugin',
    version='1.0.0',
    packages=['flyrigloader_json'],
    entry_points={
        'flyrigloader.loaders': [
            'json = flyrigloader_json.loader:JsonLoader',
            'jsonl = flyrigloader_json.loader:JsonLinesLoader',
        ],
        'flyrigloader.schemas': [
            'json_schema = flyrigloader_json.schema:JsonSchema',
            'experiment_v2 = flyrigloader_json.schema:ExperimentSchemaV2',
        ],
    },
)
```

#### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "flyrigloader-json-plugin"
version = "1.0.0"
description = "JSON support plugin for FlyRigLoader"

[project.entry-points."flyrigloader.loaders"]
json = "flyrigloader_json.loader:JsonLoader"
jsonl = "flyrigloader_json.loader:JsonLinesLoader"

[project.entry-points."flyrigloader.schemas"]
json_schema = "flyrigloader_json.schema:JsonSchema"
experiment_v2 = "flyrigloader_json.schema:ExperimentSchemaV2"
```

### Entry Point Discovery Implementation

The registries automatically discover and load entry points:

```python
import importlib.metadata
from flyrigloader.registries import LoaderRegistry, SchemaRegistry

# This happens automatically when registries are initialized
def discover_plugins():
    """Discover and register plugins through entry points."""
    
    # Discover loader plugins
    entry_points = importlib.metadata.entry_points()
    
    # Handle different entry_points API versions
    if hasattr(entry_points, 'select'):
        # New API (Python 3.10+)
        loader_entries = entry_points.select(group='flyrigloader.loaders')
        schema_entries = entry_points.select(group='flyrigloader.schemas')
    else:
        # Legacy API
        loader_entries = entry_points.get('flyrigloader.loaders', [])
        schema_entries = entry_points.get('flyrigloader.schemas', [])
    
    # Register discovered loaders
    registry = LoaderRegistry()
    for entry_point in loader_entries:
        try:
            loader_class = entry_point.load()
            extension = entry_point.name
            if not extension.startswith('.'):
                extension = f'.{extension}'
            registry.register_loader(extension, loader_class, priority=0)
        except Exception as e:
            print(f"Warning: Failed to load plugin loader {entry_point.name}: {e}")
    
    # Register discovered schemas
    schema_registry = SchemaRegistry()
    for entry_point in schema_entries:
        try:
            schema_class = entry_point.load()
            schema_name = entry_point.name
            schema_registry.register_schema(schema_name, schema_class, priority=0)
        except Exception as e:
            print(f"Warning: Failed to load plugin schema {entry_point.name}: {e}")
```

### Manual Plugin Registration

You can also register plugins manually at runtime:

```python
from flyrigloader.registries import LoaderRegistry, SchemaRegistry

# Runtime loader registration
def register_my_plugin():
    """Register custom plugin at runtime."""
    
    # Register loader
    loader_registry = LoaderRegistry()
    loader_registry.register_loader('.custom', CustomLoader, priority=10)
    
    # Register schema
    schema_registry = SchemaRegistry()
    schema_registry.register_schema('custom', CustomSchema, priority=10)
    
    print("Plugin registered successfully!")

# Call during application initialization
register_my_plugin()
```

## Third-Party Extension Development

This section provides complete examples for developing third-party extensions.

### Complete JSON Plugin Example

Create a package structure:

```
flyrigloader-json-plugin/
├── pyproject.toml
├── src/
│   └── flyrigloader_json/
│       ├── __init__.py
│       ├── loader.py
│       └── schema.py
└── tests/
    ├── __init__.py
    └── test_plugin.py
```

#### src/flyrigloader_json/loader.py

```python
"""JSON loader plugin for FlyRigLoader."""

from pathlib import Path
from typing import Any, Dict, List, Union
import json
import jsonlines
from flyrigloader.exceptions import LoadError

class JsonLoader:
    """Loader for standard JSON files."""
    
    def load(self, path: Path) -> Union[Dict, List]:
        """Load JSON data from file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise LoadError(
                f"Invalid JSON format in file: {path}",
                error_code="LOAD_004",
                context={
                    "file_path": str(path),
                    "json_error": str(e),
                    "line_number": e.lineno,
                    "column_number": e.colno
                }
            )
        except Exception as e:
            raise LoadError(
                f"Failed to load JSON file: {path}",
                error_code="LOAD_001",
                context={
                    "file_path": str(path),
                    "error": str(e)
                }
            )
    
    def supports_extension(self, extension: str) -> bool:
        """Check if this loader supports JSON files."""
        return extension.lower() == '.json'
    
    @property
    def priority(self) -> int:
        """Priority for JSON loader."""
        return 5

class JsonLinesLoader:
    """Loader for JSON Lines (.jsonl) files."""
    
    def load(self, path: Path) -> List[Dict]:
        """Load JSON Lines data from file."""
        try:
            data = []
            with jsonlines.open(path, 'r') as reader:
                for line in reader:
                    data.append(line)
            return data
        except Exception as e:
            raise LoadError(
                f"Failed to load JSON Lines file: {path}",
                error_code="LOAD_001",
                context={
                    "file_path": str(path),
                    "error": str(e)
                }
            )
    
    def supports_extension(self, extension: str) -> bool:
        """Check if this loader supports JSON Lines files."""
        return extension.lower() == '.jsonl'
    
    @property
    def priority(self) -> int:
        """Priority for JSON Lines loader."""
        return 5
```

#### src/flyrigloader_json/schema.py

```python
"""JSON schema validation plugin for FlyRigLoader."""

from typing import Any, Dict, List
from flyrigloader.exceptions import TransformError
import jsonschema

class JsonSchema:
    """Schema validator for JSON data."""
    
    def __init__(self):
        self._name = 'json_schema'
        self._types = ['json', 'generic']
        
        # Default JSON schema for validation
        self._default_schema = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "number"},
                "data": {"type": "object"}
            },
            "required": ["timestamp", "data"]
        }
    
    def validate(self, data: Any) -> Dict[str, Any]:
        """Validate JSON data against schema."""
        try:
            # Validate using jsonschema
            jsonschema.validate(data, self._default_schema)
            
            # Return validated data
            return data
            
        except jsonschema.ValidationError as e:
            raise TransformError(
                f"JSON schema validation failed: {e.message}",
                error_code="TRANSFORM_001",
                context={
                    "validation_error": e.message,
                    "schema_path": list(e.absolute_path),
                    "schema_name": self._name
                }
            )
        except Exception as e:
            raise TransformError(
                f"JSON validation error: {str(e)}",
                error_code="TRANSFORM_001",
                context={
                    "error": str(e),
                    "schema_name": self._name
                }
            )
    
    @property
    def schema_name(self) -> str:
        return self._name
    
    @property
    def supported_types(self) -> List[str]:
        return self._types.copy()

class ExperimentSchemaV2:
    """Advanced schema for experiment data (version 2)."""
    
    def __init__(self):
        self._name = 'experiment_v2'
        self._types = ['experiment', 'behavioral', 'neural']
        
        # Enhanced schema with more fields
        self._schema = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "number"},
                "subject_id": {"type": "string"},
                "session_id": {"type": "string"},
                "experiment_type": {"type": "string"},
                "metadata": {"type": "object"},
                "trial_data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "trial_number": {"type": "integer"},
                            "stimulus": {"type": "string"},
                            "response": {"type": "string"},
                            "reaction_time": {"type": "number"}
                        },
                        "required": ["trial_number", "stimulus", "response"]
                    }
                }
            },
            "required": ["timestamp", "subject_id", "session_id", "experiment_type"]
        }
    
    def validate(self, data: Any) -> Dict[str, Any]:
        """Validate experiment data against enhanced schema."""
        try:
            jsonschema.validate(data, self._schema)
            
            # Additional custom validation
            if data.get('experiment_type') not in ['behavioral', 'neural', 'combined']:
                raise TransformError(
                    f"Invalid experiment type: {data.get('experiment_type')}",
                    error_code="TRANSFORM_001",
                    context={
                        "experiment_type": data.get('experiment_type'),
                        "valid_types": ['behavioral', 'neural', 'combined']
                    }
                )
            
            return data
            
        except jsonschema.ValidationError as e:
            raise TransformError(
                f"Experiment schema validation failed: {e.message}",
                error_code="TRANSFORM_001",
                context={
                    "validation_error": e.message,
                    "schema_path": list(e.absolute_path),
                    "schema_name": self._name
                }
            )
    
    @property
    def schema_name(self) -> str:
        return self._name
    
    @property
    def supported_types(self) -> List[str]:
        return self._types.copy()
```

#### tests/test_plugin.py

```python
"""Tests for JSON plugin."""

import json
import pytest
from pathlib import Path
from flyrigloader_json.loader import JsonLoader, JsonLinesLoader
from flyrigloader_json.schema import JsonSchema, ExperimentSchemaV2
from flyrigloader.exceptions import LoadError, TransformError

class TestJsonLoader:
    """Tests for JSON loader."""
    
    def test_load_valid_json(self, tmp_path):
        """Test loading valid JSON file."""
        # Create test JSON file
        data = {"test": "data", "number": 42}
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(data))
        
        # Test loader
        loader = JsonLoader()
        result = loader.load(json_file)
        
        assert result == data
    
    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON file."""
        # Create invalid JSON file
        json_file = tmp_path / "invalid.json"
        json_file.write_text('{"invalid": json}')
        
        # Test loader
        loader = JsonLoader()
        with pytest.raises(LoadError) as exc_info:
            loader.load(json_file)
        
        assert "Invalid JSON format" in str(exc_info.value)
        assert exc_info.value.error_code == "LOAD_004"
    
    def test_supports_extension(self):
        """Test extension support check."""
        loader = JsonLoader()
        assert loader.supports_extension('.json') is True
        assert loader.supports_extension('.JSON') is True
        assert loader.supports_extension('.txt') is False
    
    def test_priority(self):
        """Test loader priority."""
        loader = JsonLoader()
        assert loader.priority == 5

class TestJsonSchema:
    """Tests for JSON schema validator."""
    
    def test_validate_valid_data(self):
        """Test validation of valid data."""
        schema = JsonSchema()
        data = {
            "timestamp": 1234567890.0,
            "data": {"measurement": "value"}
        }
        
        result = schema.validate(data)
        assert result == data
    
    def test_validate_invalid_data(self):
        """Test validation of invalid data."""
        schema = JsonSchema()
        data = {"invalid": "data"}  # Missing required fields
        
        with pytest.raises(TransformError) as exc_info:
            schema.validate(data)
        
        assert "JSON schema validation failed" in str(exc_info.value)
        assert exc_info.value.error_code == "TRANSFORM_001"
    
    def test_schema_properties(self):
        """Test schema properties."""
        schema = JsonSchema()
        assert schema.schema_name == 'json_schema'
        assert 'json' in schema.supported_types
        assert 'generic' in schema.supported_types
```

### Package Installation and Usage

After creating your plugin package:

```bash
# Install the plugin
pip install flyrigloader-json-plugin

# The plugin is automatically discovered and registered
```

Use the plugin in your code:

```python
from flyrigloader.io.loaders import load_data_file
from flyrigloader.registries import get_schema

# Load JSON file (automatically uses JsonLoader)
data = load_data_file('experiment.json')

# Validate with custom schema
schema = get_schema('experiment_v2')
if schema:
    validator = schema()
    validated_data = validator.validate(data)
```

## Advanced Registration Patterns

### Metaclass-Based Registration

The registry system supports metaclass-based registration for automatic discovery:

```python
from flyrigloader.registries import LoaderRegistry, SchemaRegistry

class AutoRegisterLoaderMeta(type):
    """Metaclass for automatic loader registration."""
    
    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)
        
        # Auto-register if extension is defined
        if hasattr(cls, 'EXTENSION') and hasattr(cls, 'PRIORITY'):
            registry = LoaderRegistry()
            registry.register_loader(cls.EXTENSION, cls, cls.PRIORITY)
        
        return cls

class AutoRegisterLoader(metaclass=AutoRegisterLoaderMeta):
    """Base class for auto-registering loaders."""
    pass

# Usage
class CsvLoader(AutoRegisterLoader):
    EXTENSION = '.csv'
    PRIORITY = 8
    
    def load(self, path: Path) -> dict:
        # Implementation
        pass
    
    def supports_extension(self, extension: str) -> bool:
        return extension.lower() == '.csv'
    
    @property
    def priority(self) -> int:
        return self.PRIORITY

# CsvLoader is automatically registered when class is defined
```

### Conditional Registration

Register loaders conditionally based on available dependencies:

```python
from flyrigloader.registries import LoaderRegistry
import importlib.util

def register_optional_loaders():
    """Register loaders based on available dependencies."""
    registry = LoaderRegistry()
    
    # Register HDF5 loader if h5py is available
    if importlib.util.find_spec('h5py') is not None:
        from .loaders.hdf5 import HDF5Loader
        registry.register_loader('.h5', HDF5Loader, priority=15)
        registry.register_loader('.hdf5', HDF5Loader, priority=15)
    
    # Register NetCDF loader if netcdf4 is available
    if importlib.util.find_spec('netCDF4') is not None:
        from .loaders.netcdf import NetCDFLoader
        registry.register_loader('.nc', NetCDFLoader, priority=12)
    
    # Register Arrow loader if pyarrow is available
    if importlib.util.find_spec('pyarrow') is not None:
        from .loaders.arrow import ArrowLoader
        registry.register_loader('.arrow', ArrowLoader, priority=10)
        registry.register_loader('.parquet', ArrowLoader, priority=10)

# Call during module initialization
register_optional_loaders()
```

### Factory Pattern Registration

Use factory patterns for complex loader creation:

```python
from flyrigloader.registries import LoaderRegistry
from typing import Type, Dict, Any

class LoaderFactory:
    """Factory for creating configured loaders."""
    
    def __init__(self):
        self._configurations: Dict[str, Dict[str, Any]] = {}
    
    def register_configuration(self, extension: str, config: Dict[str, Any]):
        """Register configuration for a loader."""
        self._configurations[extension] = config
    
    def create_loader(self, extension: str, loader_class: Type) -> Any:
        """Create configured loader instance."""
        config = self._configurations.get(extension, {})
        return loader_class(**config)

# Usage
factory = LoaderFactory()
factory.register_configuration('.custom', {
    'compression': 'gzip',
    'encoding': 'utf-8',
    'buffer_size': 8192
})

# Register factory-created loader
registry = LoaderRegistry()
registry.register_loader('.custom', 
    lambda: factory.create_loader('.custom', CustomLoader),
    priority=10
)
```

## Testing Extensions

### Testing Custom Loaders

```python
import pytest
from pathlib import Path
from flyrigloader.registries import LoaderRegistry
from flyrigloader.exceptions import LoadError
from your_package.loaders import CustomLoader

class TestCustomLoader:
    """Test suite for custom loader."""
    
    def setup_method(self):
        """Setup test environment."""
        self.loader = CustomLoader()
        self.registry = LoaderRegistry()
    
    def test_load_valid_file(self, tmp_path):
        """Test loading valid file."""
        # Create test file
        test_file = tmp_path / "test.custom"
        test_file.write_text("test data")
        
        # Test loading
        result = self.loader.load(test_file)
        assert result is not None
    
    def test_load_nonexistent_file(self, tmp_path):
        """Test loading nonexistent file."""
        test_file = tmp_path / "nonexistent.custom"
        
        with pytest.raises(LoadError) as exc_info:
            self.loader.load(test_file)
        
        assert exc_info.value.error_code == "LOAD_001"
    
    def test_supports_extension(self):
        """Test extension support."""
        assert self.loader.supports_extension('.custom') is True
        assert self.loader.supports_extension('.other') is False
    
    def test_priority(self):
        """Test loader priority."""
        assert isinstance(self.loader.priority, int)
        assert self.loader.priority >= 0
    
    def test_registry_integration(self):
        """Test integration with registry."""
        self.registry.register_loader('.custom', CustomLoader, priority=10)
        
        loader_class = self.registry.get_loader_for_extension('.custom')
        assert loader_class == CustomLoader
    
    def test_cleanup_registry(self):
        """Test registry cleanup."""
        self.registry.register_loader('.custom', CustomLoader, priority=10)
        assert self.registry.get_loader_for_extension('.custom') is not None
        
        self.registry.unregister_loader('.custom')
        assert self.registry.get_loader_for_extension('.custom') is None
```

### Testing Schema Validators

```python
import pytest
from flyrigloader.registries import SchemaRegistry
from flyrigloader.exceptions import TransformError
from your_package.schemas import CustomSchema

class TestCustomSchema:
    """Test suite for custom schema."""
    
    def setup_method(self):
        """Setup test environment."""
        self.schema = CustomSchema()
        self.registry = SchemaRegistry()
    
    def test_validate_valid_data(self):
        """Test validation of valid data."""
        valid_data = {
            "field1": "value1",
            "field2": 42,
            "field3": [1, 2, 3]
        }
        
        result = self.schema.validate(valid_data)
        assert result == valid_data
    
    def test_validate_invalid_data(self):
        """Test validation of invalid data."""
        invalid_data = {"invalid": "data"}
        
        with pytest.raises(TransformError) as exc_info:
            self.schema.validate(invalid_data)
        
        assert exc_info.value.error_code.startswith("TRANSFORM_")
    
    def test_schema_properties(self):
        """Test schema properties."""
        assert isinstance(self.schema.schema_name, str)
        assert len(self.schema.schema_name) > 0
        assert isinstance(self.schema.supported_types, list)
        assert len(self.schema.supported_types) > 0
    
    def test_registry_integration(self):
        """Test integration with schema registry."""
        self.registry.register_schema('custom', CustomSchema, priority=10)
        
        schema_class = self.registry.get_schema('custom')
        assert schema_class == CustomSchema
```

### Mock Testing with Registry

```python
import pytest
from unittest.mock import Mock, patch
from flyrigloader.registries import LoaderRegistry, SchemaRegistry

class TestRegistryMocking:
    """Test registry behavior with mocking."""
    
    def test_mock_loader_registration(self):
        """Test registry with mock loader."""
        registry = LoaderRegistry()
        
        # Create mock loader
        mock_loader = Mock()
        mock_loader.priority = 10
        mock_loader.supports_extension.return_value = True
        mock_loader.load.return_value = {"test": "data"}
        
        # Register mock
        registry.register_loader('.mock', mock_loader, priority=10)
        
        # Test registry behavior
        loader_class = registry.get_loader_for_extension('.mock')
        assert loader_class == mock_loader
        
        # Test loader usage
        loader_instance = loader_class()
        result = loader_instance.load(Path('test.mock'))
        assert result == {"test": "data"}
    
    @patch('flyrigloader.registries.importlib.metadata.entry_points')
    def test_entry_point_discovery(self, mock_entry_points):
        """Test entry point discovery with mocking."""
        # Mock entry points
        mock_entry_point = Mock()
        mock_entry_point.name = 'mock'
        mock_entry_point.load.return_value = Mock()
        
        mock_entry_points.return_value.select.return_value = [mock_entry_point]
        
        # Create registry (triggers discovery)
        registry = LoaderRegistry()
        registry._discover_plugins()
        
        # Verify entry point was loaded
        mock_entry_point.load.assert_called_once()
```

## Best Practices



### 1. Error Handling

Always use FlyRigLoader's exception hierarchy:

```python
from flyrigloader.exceptions import LoadError, TransformError

class MyLoader:
    def load(self, path: Path) -> Any:
        try:
            # Loading logic
            return data
        except FileNotFoundError:
            raise LoadError(
                f"File not found: {path}",
                error_code="LOAD_001",
                context={"file_path": str(path)}
            )
        except Exception as e:
            raise LoadError(
                f"Failed to load file: {path}",
                error_code="LOAD_002",
                context={
                    "file_path": str(path),
                    "original_error": str(e)
                }
            )
```

### 2. Performance Considerations

Optimize for O(1) registry lookups:

```python
class EfficientLoader:
    def __init__(self):
        # Cache expensive computations
        self._extension_cache = {'.ext1', '.ext2', '.ext3'}
    
    def supports_extension(self, extension: str) -> bool:
        # O(1) lookup instead of expensive computation
        return extension.lower() in self._extension_cache
    
    def load(self, path: Path) -> Any:
        # Minimize file I/O operations
        with open(path, 'rb') as f:
            return self._load_efficiently(f)
```

### 3. Thread Safety

Ensure thread-safe implementations:

```python
import threading

class ThreadSafeLoader:
    def __init__(self):
        self._lock = threading.RLock()
        self._cache = {}
    
    def load(self, path: Path) -> Any:
        with self._lock:
            if path in self._cache:
                return self._cache[path]
            
            data = self._load_data(path)
            self._cache[path] = data
            return data
```

### 4. Validation and Type Safety

Use comprehensive type hints and validation:

```python
from typing import Union, Dict, Any, List, Optional
from pathlib import Path

class TypeSafeLoader:
    def load(self, path: Path) -> Union[Dict[str, Any], List[Any]]:
        """Load data with clear return type specification."""
        if not isinstance(path, Path):
            raise TypeError(f"Expected Path, got {type(path)}")
        
        # Implementation with clear types
        return self._load_with_types(path)
    
    def supports_extension(self, extension: str) -> bool:
        """Type-safe extension checking."""
        if not isinstance(extension, str):
            return False
        
        return extension.lower() in self._supported_extensions
```

### 5. Documentation and Examples

Provide comprehensive documentation:

```python
class WellDocumentedLoader:
    """
    Loader for XYZ format files.
    
    This loader handles XYZ format files with the following features:
    - Automatic compression detection
    - Metadata extraction
    - Error recovery
    
    Examples:
        >>> loader = WellDocumentedLoader()
        >>> data = loader.load(Path('data.xyz'))
        >>> print(data.keys())
        ['timestamp', 'measurements', 'metadata']
    
    Supported Extensions:
        - .xyz: Standard XYZ format
        - .xyz.gz: Compressed XYZ format
    
    Error Codes:
        - LOAD_001: File not found
        - LOAD_002: Invalid format
        - LOAD_003: Compression error
    """
    
    def load(self, path: Path) -> Dict[str, Any]:
        """
        Load XYZ format file.
        
        Args:
            path: Path to XYZ file
            
        Returns:
            Dictionary with keys 'timestamp', 'measurements', 'metadata'
            
        Raises:
            LoadError: If file cannot be loaded (see error codes above)
        """
        # Implementation
        pass
```

### 6. Testing Strategy

Implement comprehensive tests:

```python
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from flyrigloader.registries import LoaderRegistry

class TestMyLoader:
    """Comprehensive test suite for MyLoader."""
    
    @pytest.fixture
    def loader(self):
        """Create loader instance for testing."""
        return MyLoader()
    
    @pytest.fixture
    def registry(self):
        """Create clean registry for testing."""
        registry = LoaderRegistry()
        yield registry
        registry.clear()  # Cleanup
    
    def test_load_success(self, loader, tmp_path):
        """Test successful loading."""
        # Create test file
        test_file = tmp_path / "test.ext"
        test_file.write_text("test data")
        
        # Test loading
        result = loader.load(test_file)
        assert result is not None
    
    def test_load_error_handling(self, loader, tmp_path):
        """Test error handling."""
        nonexistent_file = tmp_path / "nonexistent.ext"
        
        with pytest.raises(LoadError) as exc_info:
            loader.load(nonexistent_file)
        
        assert exc_info.value.error_code == "LOAD_001"
    
    def test_registry_integration(self, loader, registry):
        """Test registry integration."""
        registry.register_loader('.ext', MyLoader, priority=10)
        
        loader_class = registry.get_loader_for_extension('.ext')
        assert loader_class == MyLoader
    
    @patch('your_package.loaders.expensive_operation')
    def test_with_mocking(self, mock_operation, loader):
        """Test with mocked dependencies."""
        mock_operation.return_value = "mocked result"
        
        # Test loader behavior with mocked dependency
        result = loader.load(Path('test.ext'))
        assert result == "mocked result"
        mock_operation.assert_called_once()
```

This comprehensive extension guide provides everything needed to create robust, maintainable extensions for FlyRigLoader using the registry-based architecture. The patterns and examples shown here ensure compatibility with the existing system while maintaining high code quality and performance standards.