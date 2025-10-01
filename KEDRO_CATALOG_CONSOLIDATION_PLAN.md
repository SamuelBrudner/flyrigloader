# Kedro Catalog Functions Consolidation Plan

## Current State: 8 Functions

### 1. `create_flyrigloader_catalog_entry()` - Lines 67-233
**Purpose:** Create a single catalog entry  
**Usage:** Basic building block  
**Keep:** ✅ Core functionality

### 2. `validate_catalog_config()` - Lines 235-411
**Purpose:** Validate catalog configuration  
**Usage:** Used in tests, validation workflow  
**Keep:** ✅ Important validation

### 3. `get_dataset_parameters()` - Lines 413-550
**Purpose:** Extract parameters from dataset config  
**Usage:** **Unclear** - Helper for other functions?  
**Action:** ⚠️ **INVESTIGATE** - May be internal only

### 4. `generate_catalog_template()` - Lines 552-713
**Purpose:** Generate catalog templates (multi_experiment, workflow, etc.)  
**Usage:** Template generation  
**Keep:** ✅ Useful for quick setup

### 5. `create_multi_experiment_catalog()` - Lines 715-867
**Purpose:** Create catalog for multiple experiments  
**Usage:** **Overlap with generate_catalog_template?**  
**Action:** 🔄 **CONSOLIDATE** with template generation

### 6. `inject_catalog_parameters()` - Lines 869-994
**Purpose:** Inject parameters into existing catalog  
**Usage:** **Niche** - Dynamic parameter injection  
**Action:** ❌ **CANDIDATE FOR REMOVAL** or make internal

### 7. `create_workflow_catalog_entries()` - Lines 996-1187
**Purpose:** Create complete workflow catalog  
**Usage:** **Overlap with generate_catalog_template(template_type="workflow")?**  
**Action:** 🔄 **CONSOLIDATE** with template generation

### 8. `validate_catalog_against_schema()` - Lines 1189-1432
**Purpose:** Validate catalog against schema requirements  
**Usage:** **Overlap with validate_catalog_config?**  
**Action:** 🔄 **CONSOLIDATE** with validation

---

## Consolidation Strategy

### Proposed 4-Function API

#### 1. **`create_catalog_entry()`** ← Keep as-is
```python
def create_catalog_entry(
    dataset_name: str,
    config_path: Union[str, Path],
    experiment_name: str,
    **options
) -> Dict[str, Any]:
    """Create a single catalog entry."""
```
**Rename:** `create_flyrigloader_catalog_entry` → `create_catalog_entry` (shorter)

---

#### 2. **`validate_catalog()`** ← Consolidate validation functions
```python
def validate_catalog(
    catalog_config: Dict[str, Any],
    *,
    config_path: Optional[Union[str, Path]] = None,
    schema_requirements: Optional[Dict[str, Any]] = None,
    strict: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate catalog configuration.
    
    Combines functionality from:
    - validate_catalog_config()
    - validate_catalog_against_schema()
    """
```

**Changes:**
- Merge `validate_catalog_config` + `validate_catalog_against_schema`
- Single entry point for all validation
- Return (is_valid, errors) tuple

---

#### 3. **`generate_catalog()`** ← Consolidate template/multi/workflow functions
```python
def generate_catalog(
    template_type: str,  # "single", "multi", "workflow"
    base_config_path: Union[str, Path],
    experiments: Union[str, List[str]],
    *,
    dataset_prefix: str = "experiment",
    output_format: str = "dict",  # "dict" or "yaml"
    **options
) -> Union[Dict[str, Any], str]:
    """
    Generate catalog configurations from templates.
    
    Combines functionality from:
    - generate_catalog_template()
    - create_multi_experiment_catalog()
    - create_workflow_catalog_entries()
    
    Examples:
        # Single experiment
        catalog = generate_catalog("single", "config.yaml", "baseline")
        
        # Multiple experiments
        catalog = generate_catalog("multi", "config.yaml", ["baseline", "treatment"])
        
        # Complete workflow
        catalog = generate_catalog("workflow", "config.yaml", ["baseline", "treatment"])
    """
```

**Changes:**
- Unified interface for all template generation
- `template_type` parameter selects behavior
- Replaces 3 separate functions

---

#### 4. **`update_catalog()`** ← Simplified parameter injection
```python
def update_catalog(
    catalog_config: Dict[str, Any],
    updates: Dict[str, Any],
    *,
    target_datasets: Optional[List[str]] = None,
    merge_strategy: str = "override"  # "override", "merge", "preserve"
) -> Dict[str, Any]:
    """
    Update catalog configuration with new parameters.
    
    Simplified version of inject_catalog_parameters().
    
    Args:
        catalog_config: Existing catalog
        updates: Parameters to inject
        target_datasets: Specific datasets to update (None = all)
        merge_strategy: How to handle conflicts
    """
```

**Changes:**
- Simplified from `inject_catalog_parameters`
- Clearer name: "update" vs "inject"
- More intuitive interface

---

## Detailed Consolidation Steps

### Step 1: Create New Unified Functions

1. **`validate_catalog()`** - Merge validation
   - Copy logic from `validate_catalog_config`
   - Integrate schema validation from `validate_catalog_against_schema`
   - Add new parameter: `schema_requirements`
   - Keep backward compatibility

2. **`generate_catalog()`** - Merge template generation
   - Dispatcher based on `template_type`
   - Refactor common logic into helpers
   - Map old function calls to new interface:
     - `generate_catalog_template` → `generate_catalog`
     - `create_multi_experiment_catalog` → `generate_catalog(..., template_type="multi")`
     - `create_workflow_catalog_entries` → `generate_catalog(..., template_type="workflow")`

3. **`update_catalog()`** - Simplify parameter injection
   - Simplify `inject_catalog_parameters` logic
   - Remove complex branching
   - Focus on common use cases

### Step 2: Add Deprecation Warnings

Keep old function names with deprecation warnings:
```python
def create_multi_experiment_catalog(*args, **kwargs):
    """Deprecated: Use generate_catalog(template_type='multi', ...) instead."""
    warnings.warn(
        "create_multi_experiment_catalog is deprecated. "
        "Use generate_catalog(template_type='multi', ...) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return generate_catalog(template_type="multi", *args, **kwargs)
```

### Step 3: Update Tests

- Update imports
- Update function calls
- Add tests for new unified interface
- Keep backward compatibility tests

### Step 4: Update Documentation

- Update `docs/kedro_integration.md`
- Update `docs/KEDRO_API_REFERENCE.md`
- Add migration guide

---

## Benefits of Consolidation

### For Users
1. **Simpler API** - 4 functions instead of 8
2. **Clearer names** - More intuitive function names
3. **Consistent interface** - Similar parameter patterns
4. **Better documentation** - Easier to find what you need

### For Maintainers
1. **Less code** - ~400 lines reduction (estimated)
2. **Shared logic** - Common code extracted to helpers
3. **Easier testing** - Fewer edge cases
4. **Better organization** - Logical grouping

---

## Backward Compatibility Strategy

### Phase 1: Add New Functions (This PR)
- Create new unified functions
- Keep all old functions working
- Add deprecation warnings to old functions

### Phase 2: Migration Period (Next Release)
- Update examples in documentation
- Provide migration guide
- Keep old functions with warnings

### Phase 3: Removal (2-3 Releases Later)
- Remove deprecated functions
- Clean up internal helpers

---

## Internal Helper Functions (New)

These will be **private** (prefixed with `_`):

```python
def _create_single_entry(...) -> Dict:
    """Helper for single catalog entry."""
    
def _create_multi_entries(...) -> Dict:
    """Helper for multi-experiment catalog."""
    
def _create_workflow_entries(...) -> Dict:
    """Helper for workflow catalog."""
    
def _validate_schema(...) -> List[str]:
    """Helper for schema validation."""
    
def _merge_configs(...) -> Dict:
    """Helper for config merging."""
```

---

## Implementation Priority

### High Priority (Do Now)
1. ✅ Create `generate_catalog()` - Most impactful consolidation
2. ✅ Create `validate_catalog()` - Simplifies validation

### Medium Priority (Next)
3. ⚠️ Create `update_catalog()` - Useful but less critical
4. ⚠️ Rename `create_flyrigloader_catalog_entry` → `create_catalog_entry`

### Low Priority (Future)
5. Remove `get_dataset_parameters()` if unused externally
6. Extract internal helpers

---

## Questions to Answer Before Proceeding

1. **Is `get_dataset_parameters()` used externally?**
   - If no → make it internal `_get_dataset_parameters()`
   - If yes → keep in public API

2. **Is `inject_catalog_parameters()` really needed?**
   - Complex function with niche use case
   - Could be replaced with simpler dict merge

3. **Should we do this consolidation in one PR or multiple?**
   - Recommendation: Do it incrementally
   - PR 1: Add new functions with deprecation
   - PR 2: Update tests
   - PR 3: Remove old functions (later)

---

## Decision: Proceed with Consolidation?

**Recommendation:** YES, but incrementally

**Rationale:**
- API is too complex (8 functions)
- Clear overlap between functions
- Users will benefit from simpler interface
- Backward compatibility maintained

**Next Steps:**
1. Implement `generate_catalog()` first (biggest win)
2. Implement `validate_catalog()` second
3. Add deprecation warnings
4. Update tests
5. Update documentation

**Estimated Effort:** 4-6 hours
**Risk:** Low (backward compatibility maintained)
**Impact:** High (significantly improves API usability)

---

## Approval Needed

Before proceeding with implementation, please confirm:

- [ ] Agree with consolidation approach
- [ ] Agree with new function names
- [ ] Agree with deprecation strategy
- [ ] Agree with phased rollout

Any concerns or alternative suggestions?
