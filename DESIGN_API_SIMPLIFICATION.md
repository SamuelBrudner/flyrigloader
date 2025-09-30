# API Simplification Plan for flyrigloader.io.column_models

## Current Problems

1. **DependencyContainer in public API** - Only used for testing, leaks test concerns
2. **Polymorphic `get_config_from_source()`** - Does 4 different things based on type
3. **Duplicate class methods** - `ColumnConfig.load_column_config()` vs `load_column_config()`
4. **3,100 lines of duplicate tests** across 3 files

## Proposed Clean API

### Public API (what users see)

```python
# Load from YAML file
config = load_column_config(path: str) -> ColumnConfigDict

# Validate a dictionary
config = ColumnConfigDict.model_validate(data: dict) -> ColumnConfigDict

# Get default config
config = get_default_config() -> ColumnConfigDict

# Transform data
df = make_dataframe_from_config(
    exp_matrix: dict,
    config: ColumnConfigDict | str,  # Accept model or path for convenience
    metadata: dict | None = None
) -> pd.DataFrame
```

### Internal API (hidden, for testing)

```python
# Keep but mark as private
_get_dependency_container() -> DependencyContainer
_set_dependency_container(container) -> None

# Or use pytest fixtures directly instead
```

## Changes Required

### 1. Remove `dependencies` parameter from public functions

**Before:**
```python
def load_column_config(
    config_path: str, 
    dependencies: Optional[DependencyContainer] = None
) -> ColumnConfigDict:
```

**After:**
```python
def load_column_config(config_path: str) -> ColumnConfigDict:
    """Load and validate column configuration from YAML file."""
    deps = _get_dependency_container()  # Internal only
    # ... rest stays same
```

### 2. Remove polymorphic `get_config_from_source`

**Before:**
```python
get_config_from_source(None | str | dict | ColumnConfigDict) -> ColumnConfigDict
```

**After:**
```python
# Remove entirely - users should call specific functions:
# - load_column_config(path) for files
# - ColumnConfigDict.model_validate(dict) for dicts
# - get_default_config() for default
# - Just use config directly if they already have it
```

### 3. Add simple `get_default_config()` function

```python
def get_default_config() -> ColumnConfigDict:
    """Load the default column configuration."""
    default_path = _get_default_config_path()
    return load_column_config(default_path)
```

### 4. Remove class method duplicates

Remove:
- `ColumnConfig.load_column_config()` - redundant
- `ColumnConfig.get_config_from_source()` - redundant

Keep only the standalone functions.

### 5. Update exports

**column_models.py __all__:**
```python
__all__ = [
    # Models
    'ColumnConfig',
    'ColumnConfigDict',
    'ColumnDimension',
    'SpecialHandlerType',
    
    # Functions
    'load_column_config',
    'get_default_config',
    
    # Schema registry (advanced)
    'register_schema',
    'get_schema_registry',
]
```

**io/__init__.py __all__:**
```python
__all__ = [
    # Loading
    "read_pickle_any_format",
    "load_experimental_data",
    
    # Transformation
    "make_dataframe_from_config",
    
    # Classes (rarely needed directly)
    "PickleLoader",
    "DataFrameTransformer",
]
```

## Migration Path

1. ✅ Mark old functions as deprecated with warnings
2. ✅ Add new simplified functions
3. ✅ Update internal callers to use new API
4. ✅ Update tests to use new API
5. ✅ Remove deprecated functions in next major version

## Test Consolidation

**New structure:**
- `test_config_models.py` - Pydantic validation only (~300 lines)
- `test_config_loading.py` - YAML I/O and file operations (~200 lines)
- `test_dataframe_transform.py` - DataFrame creation (~400 lines)
- `test_pickle_io.py` - Pickle loading (~400 lines)

**Total: ~1,300 lines** (down from 4,143)

## Benefits

- ✅ **Clear semantic model**: Load → Validate → Transform
- ✅ **Easy to learn**: 3 main functions instead of 10+
- ✅ **Testable without mocking**: Real files, real validation
- ✅ **Fail fast**: Type hints enforce correct usage
- ✅ **Less code**: 1,300 test lines vs 4,143
