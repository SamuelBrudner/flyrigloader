# DiscoveryOptions API Documentation

**Version**: 2.0.0  
**Status**: Implementation In Progress  
**Created**: 2025-09-30

---

## Overview

`DiscoveryOptions` is a frozen dataclass that consolidates file discovery parameters into a single, type-safe, immutable configuration object. This simplifies API signatures from 10+ parameters down to 4, while maintaining full backward compatibility.

---

## Motivation

### Problem

Current API functions have too many parameters:

```python
def load_experiment_files(
    config=None,                    # Configuration
    experiment_name="",             # Required
    config_path=None,               # Alternative to config
    base_directory=None,            # Optional override
    pattern="*.*",                  # Discovery param 1
    recursive=True,                 # Discovery param 2
    extensions=None,                # Discovery param 3
    extract_metadata=False,         # Discovery param 4
    parse_dates=False,              # Discovery param 5
    _deps=None                      # Internal testing
)
```

**Issues:**
- Hard to remember parameter order
- Difficult to extend without breaking signatures
- Type safety requires complex Union types
- Poor discoverability in IDEs
- Hard to create reusable configurations

### Solution

Consolidate discovery parameters into a single options object:

```python
def load_experiment_files(
    config: ProjectConfig,
    experiment_name: str,
    base_directory: Optional[Path] = None,
    options: DiscoveryOptions = DiscoveryOptions.defaults()
)
```

**Benefits:**
- ✅ 60% fewer parameters (10 → 4)
- ✅ Type-safe and validated
- ✅ Immutable (thread-safe)
- ✅ Reusable configurations
- ✅ Easy to extend
- ✅ Better IDE support

---

## Class Definition

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

@dataclass(frozen=True)
class DiscoveryOptions:
    """
    Immutable configuration for file discovery operations.
    
    This dataclass consolidates all file discovery parameters into a single,
    type-safe object. Being frozen makes it immutable and thread-safe.
    
    Attributes:
        pattern: Glob pattern for file matching (default: "*.*")
        recursive: Search subdirectories recursively (default: True)
        extensions: Filter by file extensions, e.g., ['.pkl', '.csv'] (default: None)
        extract_metadata: Extract metadata from filenames using config patterns (default: False)
        parse_dates: Parse dates from filename components (default: False)
    """
    
    pattern: str = "*.*"
    recursive: bool = True
    extensions: Optional[List[str]] = None
    extract_metadata: bool = False
    parse_dates: bool = False
    
    def __post_init__(self):
        """Validate options after initialization."""
        # Pattern validation
        if not isinstance(self.pattern, str):
            raise ValueError(
                f"pattern must be a string, got {type(self.pattern).__name__}",
                recovery_hint="Provide a glob pattern string. Example: '*.pkl' or 'data_*.csv'"
            )
        
        if not self.pattern.strip():
            raise ValueError(
                "pattern cannot be empty",
                recovery_hint="Provide a valid glob pattern. Example: '*.*' for all files"
            )
        
        # Extensions validation
        if self.extensions is not None:
            if not isinstance(self.extensions, list):
                raise ValueError(
                    f"extensions must be a list, got {type(self.extensions).__name__}",
                    recovery_hint="Provide a list of extension strings. Example: ['.pkl', '.csv']"
                )
            
            for ext in self.extensions:
                if not isinstance(ext, str):
                    raise ValueError(
                        f"Each extension must be a string, got {type(ext).__name__}",
                        recovery_hint="Use string extensions. Example: ['.pkl', '.csv'] not [.pkl, .csv]"
                    )
```

---

## Factory Methods

### `DiscoveryOptions.defaults()`

Returns default discovery options suitable for most use cases.

```python
@classmethod
def defaults(cls) -> 'DiscoveryOptions':
    """
    Create default discovery options.
    
    Returns:
        DiscoveryOptions with pattern='*.*', recursive=True, no metadata extraction
        
    Example:
        >>> options = DiscoveryOptions.defaults()
        >>> options.pattern
        '*.*'
        >>> options.recursive
        True
    """
    return cls()
```

**Use Case:** General file discovery without metadata extraction.

---

### `DiscoveryOptions.minimal(pattern)`

Minimal options with just a pattern - useful for simple discovery.

```python
@classmethod
def minimal(cls, pattern: str = "*.*") -> 'DiscoveryOptions':
    """
    Create minimal discovery options with just a pattern.
    
    Args:
        pattern: Glob pattern for file matching
        
    Returns:
        DiscoveryOptions with specified pattern, recursive=True
        
    Example:
        >>> options = DiscoveryOptions.minimal("*.pkl")
        >>> options.pattern
        '*.pkl'
        >>> options.extract_metadata
        False
    """
    return cls(pattern=pattern, recursive=True)
```

**Use Case:** Quick file discovery with a specific pattern.

---

### `DiscoveryOptions.with_metadata(pattern, parse_dates)`

Options configured for metadata extraction from filenames.

```python
@classmethod
def with_metadata(
    cls, 
    pattern: str = "*.*",
    parse_dates: bool = True,
    recursive: bool = True
) -> 'DiscoveryOptions':
    """
    Create options configured for metadata extraction.
    
    Args:
        pattern: Glob pattern for file matching
        parse_dates: Whether to parse dates from filenames
        recursive: Search recursively
        
    Returns:
        DiscoveryOptions with metadata extraction enabled
        
    Example:
        >>> options = DiscoveryOptions.with_metadata("*.pkl", parse_dates=True)
        >>> options.extract_metadata
        True
        >>> options.parse_dates
        True
    """
    return cls(
        pattern=pattern,
        recursive=recursive,
        extract_metadata=True,
        parse_dates=parse_dates
    )
```

**Use Case:** Extracting experimental metadata from filenames.

---

### `DiscoveryOptions.with_filtering(pattern, extensions, recursive)`

Options configured for extension-based filtering.

```python
@classmethod
def with_filtering(
    cls,
    pattern: str = "*.*",
    extensions: Optional[List[str]] = None,
    recursive: bool = True
) -> 'DiscoveryOptions':
    """
    Create options configured for file type filtering.
    
    Args:
        pattern: Glob pattern for file matching
        extensions: List of file extensions to include
        recursive: Search recursively
        
    Returns:
        DiscoveryOptions with extension filtering
        
    Example:
        >>> options = DiscoveryOptions.with_filtering("*.*", ['.pkl', '.csv'])
        >>> options.extensions
        ['.pkl', '.csv']
    """
    return cls(
        pattern=pattern,
        recursive=recursive,
        extensions=extensions
    )
```

**Use Case:** Finding files of specific types.

---

## Usage Examples

### Basic Usage

```python
from flyrigloader.discovery.options import DiscoveryOptions
from flyrigloader import load_experiment_files

# Use defaults
files = load_experiment_files(
    config=my_config,
    experiment_name="thermal_preference",
    options=DiscoveryOptions.defaults()
)

# Use minimal with pattern
files = load_experiment_files(
    config=my_config,
    experiment_name="thermal_preference",
    options=DiscoveryOptions.minimal("*.pkl")
)
```

### Metadata Extraction

```python
# Extract metadata from filenames
options = DiscoveryOptions.with_metadata(
    pattern="exp_*.pkl",
    parse_dates=True
)

files = load_experiment_files(
    config=my_config,
    experiment_name="thermal_preference",
    options=options
)

# files is now a dict with metadata
# {
#     'path/to/exp_2024-01-15.pkl': {
#         'date': datetime(2024, 1, 15),
#         'experiment': 'thermal_preference',
#         ...
#     }
# }
```

### Extension Filtering

```python
# Find only pickle and CSV files
options = DiscoveryOptions.with_filtering(
    pattern="data_*",
    extensions=['.pkl', '.csv'],
    recursive=True
)

files = load_experiment_files(
    config=my_config,
    experiment_name="thermal_preference",
    options=options
)
```

### Custom Configuration

```python
# Create custom options
options = DiscoveryOptions(
    pattern="experiment_*.pkl",
    recursive=False,  # Only current directory
    extensions=['.pkl'],
    extract_metadata=True,
    parse_dates=False  # Don't parse dates
)

files = load_experiment_files(
    config=my_config,
    experiment_name="thermal_preference",
    options=options
)
```

### Reusable Configurations

```python
# Define once, reuse many times
STANDARD_DISCOVERY = DiscoveryOptions.with_metadata(
    pattern="*.pkl",
    parse_dates=True
)

# Use across multiple experiments
files1 = load_experiment_files(config, "exp1", options=STANDARD_DISCOVERY)
files2 = load_experiment_files(config, "exp2", options=STANDARD_DISCOVERY)
files3 = load_dataset_files(config, "dataset1", options=STANDARD_DISCOVERY)
```

---

## Migration Guide

### From Old API (v1.x)

**Old Code:**
```python
files = load_experiment_files(
    config=config,
    experiment_name="thermal_preference",
    pattern="*.pkl",
    recursive=True,
    extract_metadata=True,
    parse_dates=True
)
```

**New Code (v2.0):**
```python
files = load_experiment_files(
    config=config,
    experiment_name="thermal_preference",
    options=DiscoveryOptions.with_metadata("*.pkl", parse_dates=True)
)
```

### Backward Compatibility

The old API will continue to work in v1.x with deprecation warnings:

```python
# Still works, but shows deprecation warning
files = load_experiment_files(
    config=config,
    experiment_name="thermal_preference",
    pattern="*.pkl",  # DeprecationWarning
    extract_metadata=True  # DeprecationWarning
)
```

**Deprecation Timeline:**
- v1.x: Old API works with warnings
- v2.0: Both APIs supported
- v3.0: Old API removed (options-only)

---

## Validation Rules

### Pattern Validation
- ✅ Must be a non-empty string
- ✅ Valid glob pattern syntax
- ❌ Cannot be None
- ❌ Cannot be empty or whitespace-only

### Extensions Validation
- ✅ Must be a list of strings (if provided)
- ✅ Each extension must be a string
- ✅ Can include or omit leading dot (normalized internally)
- ❌ Cannot be a single string (use list)

### Boolean Flags
- ✅ `recursive`, `extract_metadata`, `parse_dates` must be bool
- ✅ Default to safe values (recursive=True, extraction=False)

---

## Error Handling

All validation errors include recovery hints:

```python
# Invalid pattern type
try:
    options = DiscoveryOptions(pattern=123)
except ValueError as e:
    print(e)
    # "pattern must be a string, got int"
    # recovery_hint: "Provide a glob pattern string. Example: '*.pkl' or 'data_*.csv'"

# Invalid extensions type
try:
    options = DiscoveryOptions(extensions=".pkl")  # Should be a list
except ValueError as e:
    print(e)
    # "extensions must be a list, got str"
    # recovery_hint: "Provide a list of extension strings. Example: ['.pkl', '.csv']"
```

---

## Thread Safety

`DiscoveryOptions` is immutable (frozen dataclass), making it **thread-safe** and safe to share across async operations:

```python
# Safe to share across threads/async tasks
SHARED_OPTIONS = DiscoveryOptions.with_metadata("*.pkl")

async def process_experiment(exp_name: str):
    # Safe to use shared options
    files = await async_load_experiment_files(
        config=config,
        experiment_name=exp_name,
        options=SHARED_OPTIONS  # No mutation possible
    )
```

---

## Testing Utilities

### Test Fixtures

```python
import pytest
from flyrigloader.discovery.options import DiscoveryOptions

@pytest.fixture
def minimal_options():
    """Minimal options for basic discovery."""
    return DiscoveryOptions.minimal("*.pkl")

@pytest.fixture
def metadata_options():
    """Options with metadata extraction."""
    return DiscoveryOptions.with_metadata(parse_dates=True)

@pytest.fixture
def filtered_options():
    """Options with extension filtering."""
    return DiscoveryOptions.with_filtering(extensions=['.pkl', '.csv'])
```

### Mock Options

```python
# Create test-specific options
test_options = DiscoveryOptions(
    pattern="test_*.pkl",
    recursive=False,
    extract_metadata=False
)
```

---

## API Reference

### Class: `DiscoveryOptions`

**Attributes:**
- `pattern: str` - Glob pattern for file matching
- `recursive: bool` - Search subdirectories recursively
- `extensions: Optional[List[str]]` - File extensions filter
- `extract_metadata: bool` - Extract metadata from filenames
- `parse_dates: bool` - Parse dates from filename components

**Methods:**

#### Class Methods
- `defaults() -> DiscoveryOptions` - Default options
- `minimal(pattern: str) -> DiscoveryOptions` - Minimal options with pattern
- `with_metadata(pattern: str, parse_dates: bool, recursive: bool) -> DiscoveryOptions` - Metadata extraction options
- `with_filtering(pattern: str, extensions: List[str], recursive: bool) -> DiscoveryOptions` - Filtering options

#### Instance Methods
- `__post_init__() -> None` - Validation (called automatically)

---

## See Also

- [API Simplification Overview](API_SIMPLIFICATION.md)
- [Migration Guide](MIGRATION_GUIDE.md) 
- [API Reference](API_REFERENCE.md)
- [Semantic Model Improvements](SEMANTIC_MODEL_IMPROVEMENTS_SUMMARY.md)

---

**Next Steps:**
1. ✅ Documentation complete
2. ⏳ Write comprehensive tests
3. ⏳ Implement DiscoveryOptions class
4. ⏳ Update API functions
5. ⏳ Create migration utilities
