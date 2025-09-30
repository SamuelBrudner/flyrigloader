# Semantic Model & Contract Testing Gaps

## Current State Analysis

You asked an excellent question: "Are we missing contract guard tests, semantic invariant tests? Do we have a clear semantic model?"

**Answer: YES, we are missing these, but now we have a plan.**

## What We've Accomplished

✅ **Cleaned up tests** - Removed duplicates, simplified patterns  
✅ **Simplified API** - Clear, focused functions  
✅ **100% test pass rate** - All 129 tests passing  
✅ **Basic property tests** - Some Hypothesis tests exist  
✅ **API documentation** - Usage examples in docs/API_COLUMN_CONFIG.md  

## What We're Missing

### 1. Formal Semantic Model ✅ NOW DOCUMENTED
**Status:** Created `docs/SEMANTIC_MODEL.md`

This document defines:
- Core domain concepts (ColumnConfig, ColumnConfigDict, exp_matrix, DataFrame)
- System invariants (INV-1 through INV-5)
- Operation contracts (preconditions, postconditions, raises)
- Composition properties (idempotency, determinism, etc.)
- Failure modes

### 2. Contract Guard Tests ❌ NOT IMPLEMENTED

**What they are:** Tests that verify functions honor their contracts

**Example:**
```python
def test_load_column_config_contract(valid_yaml_path):
    """Verify load_column_config honors its contract."""
    # PRECONDITION: path exists
    assert os.path.exists(valid_yaml_path)
    
    # OPERATION
    result = load_column_config(valid_yaml_path)
    
    # POSTCONDITION: returns valid ColumnConfigDict
    assert isinstance(result, ColumnConfigDict)
    assert len(result.columns) > 0
    
    # POSTCONDITION: all columns valid
    for col in result.columns.values():
        assert isinstance(col, ColumnConfig)
```

**What we need:**
- Precondition violation tests (what happens with invalid inputs?)
- Postcondition verification tests (does output meet guarantees?)
- Contract documentation in docstrings

### 3. Semantic Invariant Tests ❌ NOT IMPLEMENTED

**What they are:** Tests that verify properties ALWAYS hold

**Example:**
```python
def test_dataframe_row_count_invariant(random_valid_data):
    """INV-5: DataFrame row count ALWAYS equals input array length."""
    config = get_default_config()
    input_length = len(random_valid_data['t'])
    
    df = make_dataframe_from_config(random_valid_data, config)
    
    # This MUST be true for ANY valid input
    assert len(df) == input_length
```

**What we need:**
- INV-1: Configuration validity test
- INV-2: Data alignment test
- INV-3: Schema compliance test  
- INV-4: Dimension compliance test
- INV-5: Output consistency test

### 4. Composition Property Tests ❌ MINIMAL

**What they are:** Tests that verify operations compose correctly

**Example:**
```python
def test_config_loading_idempotent(temp_config_file):
    """Loading same config twice produces equivalent results."""
    config1 = load_column_config(temp_config_file)
    config2 = load_column_config(temp_config_file)
    
    # Idempotency property
    assert config1.columns.keys() == config2.columns.keys()
    assert all(
        config1.columns[k].type == config2.columns[k].type 
        for k in config1.columns
    )
```

**What we need:**
- Idempotency tests (f(f(x)) == f(x))
- Determinism tests (f(x) always gives same result)
- Commutability tests (where applicable)
- Associativity tests (where applicable)

## Recommended Implementation Plan

### Phase 1: Contract Documentation (1-2 hours)
Add formal contracts to docstrings:

```python
def load_column_config(path: str) -> ColumnConfigDict:
    """
    Load and validate column configuration from YAML file.
    
    Contract:
        Preconditions:
            - path is non-empty string
            - File at path exists
            - File contains valid YAML matching ColumnConfigDict schema
            
        Postconditions:
            - Returns valid ColumnConfigDict instance
            - result.columns is non-empty dict
            - All columns satisfy ColumnConfig schema
            
        Raises:
            - FileNotFoundError: if path doesn't exist
            - ValidationError: if YAML structure invalid
            - yaml.YAMLError: if YAML syntax invalid
            
        Invariants Preserved:
            - INV-1: Configuration Validity
    """
```

### Phase 2: Contract Guard Tests (2-3 hours)
Create `tests/flyrigloader/io/test_contracts.py`:

- Test each function's preconditions
- Test each function's postconditions
- Test each function's exception guarantees
- ~20-30 tests

### Phase 3: Semantic Invariant Tests (2-3 hours)
Create `tests/flyrigloader/io/test_invariants.py`:

- Test INV-1 through INV-5
- Use property-based testing (Hypothesis)
- Verify invariants hold for arbitrary valid inputs
- ~10-15 tests

### Phase 4: Composition Property Tests (2-3 hours)
Enhance `tests/flyrigloader/io/test_properties.py`:

- Idempotency tests
- Determinism tests
- Order independence tests
- ~15-20 tests

**Total effort:** ~8-12 hours for comprehensive contract/invariant testing

## Benefits

### Improved Confidence
- Know that contracts are honored
- Know that invariants always hold
- Know how system composes

### Better Documentation
- Contracts in docstrings are executable specifications
- Semantic model documents the "why" not just the "how"
- Clear failure modes documented

### Easier Maintenance
- Contract tests catch regressions immediately
- Invariant tests verify system properties
- Property tests find edge cases

### Higher Quality
- Move from "tests pass" to "contracts proven"
- From "it works" to "it's correct"
- From "examples" to "formal specification"

## Example Test File Structure

```
tests/flyrigloader/io/
├── test_column_config.py          # Existing - basic functionality
├── test_dataframe_transform.py    # Existing - transformation tests
├── test_pickle.py                 # Existing - I/O tests
├── test_contracts.py              # NEW - contract guard tests
├── test_invariants.py             # NEW - semantic invariant tests
└── test_properties.py             # NEW - composition property tests
```

## Quick Wins (1-2 hours)

If you want to start immediately, here are the highest-value tests to add first:

### 1. INV-5 Test (Output Consistency)
```python
@given(st.integers(min_value=1, max_value=1000))
def test_output_row_count_invariant(row_count):
    """DataFrame ALWAYS has same row count as input."""
    config = get_default_config()
    exp_matrix = create_valid_matrix(row_count, config)
    df = make_dataframe_from_config(exp_matrix, config)
    assert len(df) == row_count
```

### 2. Idempotency Test
```python
def test_config_loading_idempotent(tmp_path):
    """Loading same config twice gives equivalent results."""
    config_file = create_test_config(tmp_path)
    config1 = load_column_config(str(config_file))
    config2 = load_column_config(str(config_file))
    assert config1.model_dump() == config2.model_dump()
```

### 3. Precondition Test
```python
def test_make_dataframe_rejects_mismatched_lengths():
    """Contract: all arrays must have same length."""
    config = simple_config()
    bad_matrix = {
        't': np.zeros(100),
        'x': np.zeros(50)  # Different length!
    }
    with pytest.raises(TransformError):
        make_dataframe_from_config(bad_matrix, config)
```

## Summary

You've identified a real gap in our testing strategy. We have good coverage of "does it work?" but not "does it honor its contracts?" or "do invariants always hold?"

The semantic model document (`docs/SEMANTIC_MODEL.md`) now provides the foundation. The next step is implementing the contract and invariant tests.

**Recommendation:** Start with the 3 quick wins above (1-2 hours) to get immediate value, then decide if you want to invest in the full contract test suite (8-12 hours).

---

**Created:** 2025-09-30  
**Status:** Documentation complete, tests not yet implemented  
**Priority:** Medium-High (improves quality significantly)
