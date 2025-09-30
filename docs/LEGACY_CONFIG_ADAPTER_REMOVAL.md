# LegacyConfigAdapter Removal Strategy

**Decision**: REMOVE in v2.0.0  
**Status**: Deprecation in v1.x, Full Removal in v2.0.0  
**Date**: 2025-09-30

## Executive Summary

The `LegacyConfigAdapter` adds **significant complexity** (300+ LOC, dual maintenance burden) to provide **minimal value** (backward-compatible dict access to Pydantic models). This document proposes a phased removal strategy.

---

## Current State Analysis

### What LegacyConfigAdapter Provides

```python
# Current (with adapter)
adapter = LegacyConfigAdapter(config_dict)
adapter["project"]["directories"]["major_data_directory"]  # Dict-style access
adapter.get_model("project")  # Get underlying Pydantic model
adapter.validate_all()  # Validate all sections
```

### Complexity Metrics

| Metric | Value | Impact |
|--------|-------|--------|
| **Lines of Code** | ~300 | Maintenance burden |
| **Methods** | 15+ | API surface area |
| **Test Coverage** | 25+ tests | Test maintenance |
| **Import Usage** | 42 locations in `api.py` | High coupling |
| **Type Complexity** | `Union[Dict, LegacyConfigAdapter]` everywhere | Type annotation bloat |

---

## Problems with LegacyConfigAdapter

### 1. **Premature Backward Compatibility**

This is **pre-v1.0 software**. There's no "legacy" to support yet!

```python
# The adapter warns users:
warnings.warn(
    "Dictionary-based configuration format is deprecated...",
    DeprecationWarning
)
```

**Issue**: Deprecating something that was never released is unnecessary complexity.

### 2. **Pydantic Already Provides Dict Access**

```python
# WITHOUT adapter (using built-in Pydantic)
config_model = ProjectConfig(**config_dict)
config_model.model_dump()  # → dict
config_model.directories  # → direct attribute access

# WITH adapter (unnecessary wrapper)
adapter = LegacyConfigAdapter(config_dict)
adapter["project"]["directories"]  # → nested dict access
```

**Pydantic's `.model_dump()` provides dict conversion when needed.**

### 3. **Dual Maintenance Burden**

Every code path must handle **both** types:

```python
# From api.py Protocol definitions
def get_ignore_patterns(
    self, 
    config: Union[Dict[str, Any], LegacyConfigAdapter],  # Dual type
    experiment: Optional[str] = None
) -> List[str]:
    ...
```

**This doubling of type complexity ripples through the entire codebase.**

### 4. **Test Explosion**

Tests must cover both configurations:

```python
@pytest.fixture(params=['dict', 'legacy_adapter'])
def config_in_both_formats(request, realistic_config_dict):
    if request.param == 'dict':
        return realistic_config_dict
    else:
        return LegacyConfigAdapter(realistic_config_dict)  # Extra test paths
```

**25+ tests** specifically for adapter functionality that could be eliminated.

---

## Proposed Removal Strategy

### **Phase 1: Deprecation (Current v1.x)**

✅ **Already in place** (see line 874-880 in models.py):

```python
warnings.warn(
    "Dictionary-based configuration format is deprecated. "
    "Use create_config() builder and Pydantic models for new configurations.",
    DeprecationWarning,
    stacklevel=2
)
```

**Actions**:
- ✅ Keep deprecation warning
- ✅ Document migration path in README
- ✅ Add migration guide (this document)

---

### **Phase 2: Migration Tools (v1.5)** 🔄

**Provide tools to help users migrate:**

#### 2.1. CLI Migration Tool

```bash
# Auto-convert YAML configs to use Pydantic models
flyrigloader migrate-config old_config.yaml new_config.yaml
```

#### 2.2. Python Migration Helper

```python
from flyrigloader.migration import migrate_config

# Old code (with adapter)
config = load_config("config.yaml")  # Returns LegacyConfigAdapter
data_dir = config["project"]["directories"]["major_data_directory"]

# New code (without adapter)
config = load_config("config.yaml")  # Returns Pydantic models directly
data_dir = config.project.directories["major_data_directory"]
# OR
data_dir = config.project.model_dump()["directories"]["major_data_directory"]
```

#### 2.3. Documentation Updates

Create comprehensive migration guide with examples for all common patterns.

---

### **Phase 3: Removal (v2.0.0)** 🗑️

**Breaking Change**: Remove `LegacyConfigAdapter` entirely

#### 3.1. Code Changes

**Remove from `config/models.py`**:
```python
# DELETE:
class LegacyConfigAdapter(MutableMapping):
    ...  # 300 lines removed
```

**Simplify `api.py` type signatures**:
```python
# BEFORE:
def load_config(path: Union[str, Path]) -> Union[Dict[str, Any], LegacyConfigAdapter]:
    ...

# AFTER:
def load_config(path: Union[str, Path]) -> ProjectConfig:
    """Load configuration and return validated Pydantic model."""
    ...
```

**Remove dual-type handling**:
```python
# BEFORE:
config: Union[Dict[str, Any], LegacyConfigAdapter]

# AFTER:
config: ProjectConfig  # Single type, clear semantics
```

#### 3.2. Test Cleanup

**Remove adapter-specific tests**:
- `test_legacy_config_adapter.py` (entire file)
- Adapter-related fixtures
- Parametrized `config_in_both_formats` tests

**Estimated test reduction**: ~25 tests, ~1000 LOC

#### 3.3. API Surface Reduction

**Before (v1.x)**:
- 15+ adapter methods
- Dual type handling everywhere
- Complex Union types

**After (v2.0)**:
- Direct Pydantic model access
- Single type annotations
- `.model_dump()` for dict access when needed

---

## Migration Guide for Users

### Pattern 1: Dict-Style Access

**Old Code**:
```python
config = load_config("config.yaml")  # Returns LegacyConfigAdapter
data_dir = config["project"]["directories"]["major_data_directory"]
```

**New Code (Option 1: Direct Attribute)**:
```python
config = load_config("config.yaml")  # Returns ProjectConfig
data_dir = config.directories["major_data_directory"]
```

**New Code (Option 2: Convert to Dict)**:
```python
config = load_config("config.yaml")
config_dict = config.model_dump()  # Convert to dict
data_dir = config_dict["project"]["directories"]["major_data_directory"]
```

---

### Pattern 2: Iteration

**Old Code**:
```python
for key, value in config.items():  # Adapter provides .items()
    print(f"{key}: {value}")
```

**New Code**:
```python
config_dict = config.model_dump()  # Convert once
for key, value in config_dict.items():
    print(f"{key}: {value}")
```

---

### Pattern 3: Validation

**Old Code**:
```python
adapter = LegacyConfigAdapter(config_dict)
is_valid = adapter.validate_all()  # Custom validation method
```

**New Code**:
```python
try:
    config = ProjectConfig(**config_dict)  # Pydantic validates on creation
    is_valid = True
except ValidationError as e:
    print(f"Validation failed: {e}")
    is_valid = False
```

---

### Pattern 4: Get with Default

**Old Code**:
```python
datasets = config.get("datasets", {})  # Adapter provides .get()
```

**New Code**:
```python
datasets = config.datasets if hasattr(config, 'datasets') else {}
# OR
datasets = getattr(config, 'datasets', {})
```

---

## Benefits of Removal

### 1. **Reduced Complexity**

| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| **LOC** | ~300 | 0 | -300 |
| **Test LOC** | ~1000 | 0 | -1000 |
| **API Methods** | 15+ | 0 | -15 |
| **Type Unions** | Everywhere | None | -100% |

### 2. **Clearer API**

```python
# BEFORE: Ambiguous, dual-type
def load_config(path: str) -> Union[Dict, LegacyConfigAdapter]:
    ...

# AFTER: Clear, single type
def load_config(path: str) -> ProjectConfig:
    """Load and validate configuration as Pydantic model."""
    ...
```

### 3. **Standard Pydantic Idioms**

```python
# Users familiar with Pydantic will immediately understand:
config = load_config("config.yaml")
config.directories  # Direct attribute access (standard)
config.model_dump()  # Convert to dict (standard Pydantic)
config.model_validate(data)  # Validation (standard Pydantic)
```

### 4. **Reduced Maintenance**

- No adapter-specific bugs
- No dual-type handling edge cases
- No special documentation for adapter semantics

---

## Risks and Mitigation

### Risk 1: Breaking Existing User Code

**Severity**: Medium (pre-v1.0, limited user base)

**Mitigation**:
- Phased deprecation (v1.x → v2.0)
- Clear migration guide (this document)
- CLI migration tool
- Comprehensive examples

---

### Risk 2: Users Expect Dict Behavior

**Severity**: Low (Pydantic provides `.model_dump()`)

**Mitigation**:
- Document `.model_dump()` as dict conversion method
- Show examples in migration guide
- Highlight in v2.0 release notes

---

## Timeline

| Phase | Version | Date | Actions |
|-------|---------|------|---------|
| **Deprecation** | v1.0 | Current | ✅ Warning in place |
| **Migration Tools** | v1.5 | +3 months | CLI tool, docs |
| **Removal** | v2.0 | +6 months | Delete adapter, simplify types |

---

## Decision

**REMOVE** LegacyConfigAdapter in v2.0.0

**Rationale**:
1. Pre-v1.0 software doesn't need "legacy" support
2. Pydantic already provides dict conversion (`.model_dump()`)
3. Adapter adds 300+ LOC and dual maintenance burden
4. Simplification aligns with "minimal until required" philosophy

**Recommendation**: 
- Keep deprecation warning in v1.x
- Provide migration guide and tools in v1.5
- Remove entirely in v2.0
- Document Pydantic idioms as the standard approach

---

## Related Documentation

- [Pydantic Model Usage](PYDANTIC_MODELS.md)
- [Configuration Migration](CONFIG_MIGRATION.md)
- [API Simplification](API_SIMPLIFICATION.md)
