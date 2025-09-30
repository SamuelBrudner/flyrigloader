# API Simplification Guide

**Version**: 2.0.0  
**Status**: Implementation In Progress  
**Created**: 2025-09-30

---

## Overview

This guide documents the API simplification initiative for FlyRigLoader v2.0, focusing on reducing parameter count, improving type safety, and enhancing usability through the `DiscoveryOptions` dataclass.

---

## Motivation

### Current State (v1.x)

The main API functions have accumulated too many parameters over time:

```python
def load_experiment_files(
    config: Optional[Union[Dict[str, Any], Any]] = None,
    experiment_name: str = "",
    config_path: Optional[Union[str, Path]] = None,
    base_directory: Optional[Union[str, Path]] = None,
    pattern: str = "*.*",
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
    extract_metadata: bool = False,
    parse_dates: bool = False,
    _deps: Optional[DefaultDependencyProvider] = None
) -> Union[List[str], Dict[str, Dict[str, Any]]]:
```

**Problems:**
1. **10 parameters** - hard to remember, error-prone
2. **Complex type unions** - `Union[Dict, Any]`, `Union[str, Path]`
3. **Poor discoverability** - which parameters go together?
4. **Hard to extend** - adding features requires new parameters
5. **Reusability** - can't easily save and reuse configurations

### Target State (v2.0)

Simplified, type-safe API with consolidated options:

```python
def load_experiment_files(
    config: ProjectConfig,
    experiment_name: str,
    base_directory: Optional[Path] = None,
    options: DiscoveryOptions = DiscoveryOptions.defaults()
) -> Union[List[str], Dict[str, Dict[str, Any]]]:
```

**Benefits:**
1. **4 parameters** - 60% reduction
2. **Type-safe** - Pydantic models, no unions
3. **Clear intent** - options object groups related params
4. **Extensible** - add fields to DiscoveryOptions without breaking API
5. **Reusable** - share options across calls

---

## Design Principles

### 1. **Consolidate Related Parameters**

Discovery-related parameters are grouped into `DiscoveryOptions`:
- `pattern` → `options.pattern`
- `recursive` → `options.recursive`
- `extensions` → `options.extensions`
- `extract_metadata` → `options.extract_metadata`
- `parse_dates` → `options.parse_dates`

### 2. **Prefer Explicit Types Over Unions**

```python
# Before: Ambiguous unions
config: Optional[Union[Dict[str, Any], Any]] = None

# After: Explicit Pydantic model
config: ProjectConfig
```

### 3. **Immutability for Safety**

```python
@dataclass(frozen=True)
class DiscoveryOptions:
    # Frozen = immutable = thread-safe
    pattern: str = "*.*"
    recursive: bool = True
```

### 4. **Factory Methods for Common Patterns**

```python
# Defaults
DiscoveryOptions.defaults()

# Minimal
DiscoveryOptions.minimal("*.pkl")

# Metadata extraction
DiscoveryOptions.with_metadata(parse_dates=True)

# Extension filtering
DiscoveryOptions.with_filtering(extensions=['.pkl', '.csv'])
```

---

## API Transformations

### `load_experiment_files()`

#### Before (v1.x) - 10 Parameters

```python
files = load_experiment_files(
    config=config_dict,              # 1. Config (dict or object)
    experiment_name="thermal_pref",  # 2. Required name
    config_path=None,                # 3. Alternative to config
    base_directory=None,             # 4. Optional override
    pattern="*.pkl",                 # 5. Discovery param
    recursive=True,                  # 6. Discovery param
    extensions=['.pkl'],             # 7. Discovery param
    extract_metadata=True,           # 8. Discovery param
    parse_dates=True,                # 9. Discovery param
    _deps=None                       # 10. Internal testing
)
```

#### After (v2.0) - 4 Parameters

```python
files = load_experiment_files(
    config=config,                   # 1. Pydantic model
    experiment_name="thermal_pref",  # 2. Required name
    base_directory=None,             # 3. Optional override
    options=DiscoveryOptions.with_metadata(  # 4. All discovery params
        pattern="*.pkl",
        parse_dates=True
    )
)
```

**Improvement:**
- ✅ 60% fewer parameters (10 → 4)
- ✅ Grouped related parameters
- ✅ Type-safe configuration
- ✅ Reusable options object

---

### `load_dataset_files()`

#### Before (v1.x)

```python
files = load_dataset_files(
    config=config_dict,
    dataset_name="dataset1",
    config_path=None,
    base_directory=None,
    pattern="*.pkl",
    recursive=True,
    extensions=None,
    extract_metadata=False,
    parse_dates=False,
    _deps=None
)
```

#### After (v2.0)

```python
files = load_dataset_files(
    config=config,
    dataset_name="dataset1",
    base_directory=None,
    options=DiscoveryOptions.minimal("*.pkl")
)
```

---

## Common Usage Patterns

### Pattern 1: Simple Discovery

**Before:**
```python
files = load_experiment_files(
    config=config,
    experiment_name="exp1",
    pattern="*.pkl",
    recursive=True
)
```

**After:**
```python
files = load_experiment_files(
    config=config,
    experiment_name="exp1",
    options=DiscoveryOptions.minimal("*.pkl")
)
```

---

### Pattern 2: With Metadata Extraction

**Before:**
```python
files = load_experiment_files(
    config=config,
    experiment_name="exp1",
    pattern="exp_*.pkl",
    extract_metadata=True,
    parse_dates=True,
    recursive=True
)
```

**After:**
```python
files = load_experiment_files(
    config=config,
    experiment_name="exp1",
    options=DiscoveryOptions.with_metadata(
        pattern="exp_*.pkl",
        parse_dates=True
    )
)
```

---

### Pattern 3: Extension Filtering

**Before:**
```python
files = load_experiment_files(
    config=config,
    experiment_name="exp1",
    pattern="data_*",
    extensions=['.pkl', '.csv'],
    recursive=True
)
```

**After:**
```python
files = load_experiment_files(
    config=config,
    experiment_name="exp1",
    options=DiscoveryOptions.with_filtering(
        pattern="data_*",
        extensions=['.pkl', '.csv']
    )
)
```

---

### Pattern 4: Reusable Configuration

**Before:**
```python
# Have to repeat parameters every time
files1 = load_experiment_files(config, "exp1", pattern="*.pkl", extract_metadata=True, parse_dates=True)
files2 = load_experiment_files(config, "exp2", pattern="*.pkl", extract_metadata=True, parse_dates=True)
files3 = load_experiment_files(config, "exp3", pattern="*.pkl", extract_metadata=True, parse_dates=True)
```

**After:**
```python
# Define once, reuse many times
STANDARD_OPTS = DiscoveryOptions.with_metadata("*.pkl", parse_dates=True)

files1 = load_experiment_files(config, "exp1", options=STANDARD_OPTS)
files2 = load_experiment_files(config, "exp2", options=STANDARD_OPTS)
files3 = load_experiment_files(config, "exp3", options=STANDARD_OPTS)
```

---

## Migration Strategy

### Phase 1: Deprecation (v1.x)

Old API continues to work but emits deprecation warnings:

```python
import warnings

def load_experiment_files(
    config: Optional[Union[Dict, ProjectConfig]] = None,
    experiment_name: str = "",
    config_path: Optional[Union[str, Path]] = None,
    base_directory: Optional[Union[str, Path]] = None,
    pattern: str = "*.*",
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
    extract_metadata: bool = False,
    parse_dates: bool = False,
    options: Optional[DiscoveryOptions] = None,
    _deps: Optional[DefaultDependencyProvider] = None
):
    # Detect old API usage
    if pattern != "*.*" or not recursive or extensions or extract_metadata or parse_dates:
        warnings.warn(
            "Individual discovery parameters (pattern, recursive, extensions, extract_metadata, parse_dates) "
            "are deprecated. Use the 'options' parameter with DiscoveryOptions instead. "
            "See DISCOVERY_OPTIONS.md for migration guide.",
            DeprecationWarning,
            stacklevel=2
        )
    
    # Convert old params to options if needed
    if options is None:
        options = DiscoveryOptions(
            pattern=pattern,
            recursive=recursive,
            extensions=extensions,
            extract_metadata=extract_metadata,
            parse_dates=parse_dates
        )
```

### Phase 2: Support Both (v2.0)

Both APIs fully supported, old API deprecated:

```python
# Old API - works with deprecation warning
files = load_experiment_files(config, "exp1", pattern="*.pkl", extract_metadata=True)

# New API - recommended
files = load_experiment_files(config, "exp1", options=DiscoveryOptions.with_metadata("*.pkl"))
```

### Phase 3: Remove Old (v3.0)

Old individual parameters removed, options-only:

```python
def load_experiment_files(
    config: ProjectConfig,
    experiment_name: str,
    base_directory: Optional[Path] = None,
    options: DiscoveryOptions = DiscoveryOptions.defaults()
):
    # Only new API supported
```

---

## Backward Compatibility

### Automatic Conversion

Old code continues to work through automatic conversion:

```python
# User calls with old API
files = load_experiment_files(
    config=config,
    experiment_name="exp1",
    pattern="*.pkl",
    extract_metadata=True
)

# Internally converted to:
options = DiscoveryOptions(
    pattern="*.pkl",
    extract_metadata=True,
    recursive=True,  # default
    extensions=None,  # default
    parse_dates=False  # default
)
files = _load_experiment_files_impl(config, "exp1", None, options)
```

### Deprecation Warnings

Clear, actionable warnings guide users:

```python
DeprecationWarning: Individual discovery parameters are deprecated.

Old code:
    load_experiment_files(config, "exp1", pattern="*.pkl", extract_metadata=True)

New code:
    load_experiment_files(config, "exp1", options=DiscoveryOptions.with_metadata("*.pkl"))

See docs/DISCOVERY_OPTIONS.md for full migration guide.
```

---

## Benefits Summary

### Developer Experience

| Aspect | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Parameters** | 10 | 4 | -60% |
| **Type Safety** | Weak (unions) | Strong (Pydantic) | ✅ |
| **Discoverability** | Poor (flat params) | Good (grouped) | ✅ |
| **Reusability** | None | High (objects) | ✅ |
| **Extensibility** | Hard (breaks API) | Easy (add fields) | ✅ |

### Code Quality

| Metric | Before | After |
|--------|--------|-------|
| **Cyclomatic Complexity** | High | Lower |
| **Test Coverage** | Harder (many params) | Easier (objects) |
| **Maintainability** | Difficult | Improved |
| **Documentation** | Scattered | Centralized |

---

## Implementation Checklist

### Documentation (Current Phase)
- [x] Create DISCOVERY_OPTIONS.md
- [x] Create API_SIMPLIFICATION.md
- [ ] Update API_REFERENCE.md
- [ ] Update EXAMPLES.md
- [ ] Create MIGRATION_GUIDE.md

### Testing (Next Phase)
- [ ] Unit tests for DiscoveryOptions
- [ ] Integration tests with new API
- [ ] Backward compatibility tests
- [ ] Deprecation warning tests

### Implementation
- [ ] Create discovery/options.py
- [ ] Add DiscoveryOptions class
- [ ] Add factory methods
- [ ] Add validation

### Integration
- [ ] Update load_experiment_files()
- [ ] Update load_dataset_files()
- [ ] Add deprecation warnings
- [ ] Add automatic conversion

### Migration
- [ ] Create migration utilities
- [ ] Add deprecation warnings
- [ ] Update examples
- [ ] Update documentation

---

## Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| **Documentation** | 1-2 hours | ✅ Complete |
| **Tests** | 2-3 hours | ⏳ Next |
| **Implementation** | 2-3 hours | ⏳ Pending |
| **Integration** | 2-3 hours | ⏳ Pending |
| **Migration** | 1-2 hours | ⏳ Pending |
| **Total** | 8-13 hours | - |

---

## See Also

- [DiscoveryOptions Documentation](DISCOVERY_OPTIONS.md)
- [Semantic Model Improvements](SEMANTIC_MODEL_IMPROVEMENTS_SUMMARY.md)
- [Implementation Roadmap](IMPLEMENTATION_ROADMAP.md)
- [API Reference](API_REFERENCE.md)

---

**Status**: Phase 1 (Documentation) Complete  
**Next**: Phase 2 (Write Tests)
