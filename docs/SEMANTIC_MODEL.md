# Semantic Model: Column Configuration System

## Overview

The column configuration system provides a formal schema for validating and transforming experimental data into pandas DataFrames. This document defines the core domain concepts, their relationships, and the contracts that govern their behavior.

## Domain Concepts

### ColumnConfig
**Definition:** Schema specification for a single column in experimental data.

**Invariants:**
- `type` is always a valid type string
- `dimension` (if specified) is 1, 2, or 3
- `required` is boolean
- If `alias` is specified, it's a non-empty string
- If `default_value` is specified, column is not required

**Example:**
```python
ColumnConfig(
    type='numpy.ndarray',
    dimension=1,
    required=True,
    description='Time values in seconds'
)
```

### ColumnConfigDict
**Definition:** Complete schema for all columns in an experimental dataset.

**Invariants:**
- `columns` is a non-empty dictionary
- All values in `columns` are valid ColumnConfig instances
- `special_handlers` maps handler names to function names
- All referenced special handlers are defined in `special_handlers`

**Example:**
```python
ColumnConfigDict(
    columns={'t': ColumnConfig(...), 'x': ColumnConfig(...)},
    special_handlers={}
)
```

### Experimental Matrix
**Definition:** Raw experimental data as a dictionary mapping column names to numpy arrays.

**Invariants:**
- Keys are non-empty strings
- Values are numpy arrays
- **All arrays have the same length** (data alignment invariant)

**Example:**
```python
exp_matrix = {
    't': np.array([0, 1, 2, 3]),
    'x': np.array([0.1, 0.2, 0.3, 0.4])
}
```

### DataFrame
**Definition:** Validated, transformed tabular data.

**Invariants:**
- Row count equals input array length
- Column types match configuration
- Required columns are present
- No columns are None unless specified as optional with default None

## Core System Invariants

These properties MUST hold for all valid operations:

### INV-1: Configuration Validity
**Statement:** Every ColumnConfigDict instance is structurally valid according to Pydantic schema.

**Verification:** Pydantic validation at construction time.

**Guaranteed by:** Pydantic BaseModel validation.

### INV-2: Data Alignment
**Statement:** All arrays in an experimental matrix have the same length.

**Verification:** Check at transformation time.

**Example:**
```python
# Valid - all length 100
{'t': np.zeros(100), 'x': np.zeros(100)}

# Invalid - mismatched lengths
{'t': np.zeros(100), 'x': np.zeros(50)}  # Raises error
```

### INV-3: Schema Compliance
**Statement:** All required columns in configuration exist in experimental matrix.

**Verification:** Check before transformation.

**Example:**
```python
# Config requires 't' and 'x'
# Matrix must have both
exp_matrix = {'t': [...], 'x': [...]}  # Valid
exp_matrix = {'t': [...]}  # Invalid - missing 'x'
```

### INV-4: Dimension Compliance
**Statement:** Array dimensions match configuration specification.

**Verification:** Check during transformation.

**Example:**
```python
# Config specifies dimension=1
# Array must be 1D
config.columns['t'].dimension == 1
exp_matrix['t'].shape == (100,)  # Valid
exp_matrix['t'].shape == (100, 5)  # Invalid
```

### INV-5: Output Consistency
**Statement:** DataFrame row count always equals input array length.

**Verification:** Assert after transformation.

**Example:**
```python
input_length = len(exp_matrix['t'])
df = make_dataframe_from_config(exp_matrix, config)
assert len(df) == input_length  # ALWAYS true
```

## Operations & Contracts

### load_column_config

```python
def load_column_config(path: str) -> ColumnConfigDict
```

**Preconditions:**
- `path` is a non-empty string
- File at `path` exists
- File contains valid YAML
- YAML structure matches ColumnConfigDict schema

**Postconditions:**
- Returns valid ColumnConfigDict instance
- Result satisfies INV-1 (Configuration Validity)
- `result.columns` is non-empty

**Raises:**
- `FileNotFoundError` if path doesn't exist
- `ValidationError` if YAML structure invalid
- `yaml.YAMLError` if YAML syntax invalid

**Invariants Preserved:**
- INV-1: Output is always valid

**Example Contract Test:**
```python
def test_load_column_config_contract(valid_yaml_path):
    # PRE: path exists
    assert os.path.exists(valid_yaml_path)
    
    # OPERATION
    result = load_column_config(valid_yaml_path)
    
    # POST: valid ColumnConfigDict
    assert isinstance(result, ColumnConfigDict)
    assert len(result.columns) > 0
    
    # POST: all columns are valid
    for col in result.columns.values():
        assert isinstance(col, ColumnConfig)
```

### get_default_config

```python
def get_default_config() -> ColumnConfigDict
```

**Preconditions:**
- None (always callable)

**Postconditions:**
- Returns valid ColumnConfigDict instance
- Result is idempotent (same result each call)
- Result has standard flyrigloader columns

**Raises:**
- Never raises (guaranteed to succeed)

**Invariants Preserved:**
- INV-1: Output is always valid

**Properties:**
- **Deterministic:** Multiple calls return equivalent configs
- **Idempotent:** `get_default_config() == get_default_config()`

### make_dataframe_from_config

```python
def make_dataframe_from_config(
    exp_matrix: Dict[str, np.ndarray],
    config: ColumnConfigDict | str,
    metadata: Dict[str, Any] | None = None
) -> pd.DataFrame
```

**Preconditions:**
- `exp_matrix` is a dictionary with string keys
- All values in `exp_matrix` are numpy arrays
- All arrays in `exp_matrix` have same length (INV-2)
- `exp_matrix` contains all required columns from `config`
- Array dimensions match config specifications (INV-4)
- `config` is valid ColumnConfigDict or path to config file
- If `metadata` provided, it's a dictionary

**Postconditions:**
- Returns pandas DataFrame
- `len(result) == len(exp_matrix[first_column])` (INV-5)
- All required columns are present in result
- Column types match config specification
- If metadata provided, metadata columns are added

**Raises:**
- `TransformError` if preconditions violated
- `ValidationError` if config invalid
- `ValueError` if data shape mismatch

**Invariants Preserved:**
- INV-2: Data alignment maintained
- INV-3: Schema compliance verified
- INV-4: Dimensions validated
- INV-5: Output consistency ensured

**Properties:**
- **Deterministic:** Same inputs always produce same output
- **Column-Order Independent:** Column order in exp_matrix doesn't affect result
- **Metadata Additive:** Adding metadata doesn't change data columns

**Example Contract Test:**
```python
def test_make_dataframe_contract(valid_exp_matrix, valid_config):
    # PRE: all arrays same length
    lengths = [len(arr) for arr in valid_exp_matrix.values()]
    assert len(set(lengths)) == 1
    input_length = lengths[0]
    
    # PRE: required columns present
    required = [name for name, col in valid_config.columns.items() if col.required]
    assert all(name in valid_exp_matrix for name in required)
    
    # OPERATION
    result = make_dataframe_from_config(valid_exp_matrix, valid_config)
    
    # POST: is DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # POST: row count preserved (INV-5)
    assert len(result) == input_length
    
    # POST: required columns present
    assert all(name in result.columns for name in required)
```

## Composition Properties

### Property 1: Config Loading Idempotency
**Statement:** Loading the same config file twice produces equivalent results.

```python
config1 = load_column_config(path)
config2 = load_column_config(path)
assert config1.columns.keys() == config2.columns.keys()
```

### Property 2: Transformation Determinism
**Statement:** Same config + data always produces same DataFrame.

```python
df1 = make_dataframe_from_config(exp_matrix, config)
df2 = make_dataframe_from_config(exp_matrix, config)
assert df1.equals(df2)
```

### Property 3: Column Order Independence
**Statement:** Column order in exp_matrix doesn't affect transformation.

```python
matrix1 = {'t': [...], 'x': [...]}
matrix2 = {'x': [...], 't': [...]}  # Different order
df1 = make_dataframe_from_config(matrix1, config)
df2 = make_dataframe_from_config(matrix2, config)
assert set(df1.columns) == set(df2.columns)
```

### Property 4: Metadata Additivity
**Statement:** Adding metadata doesn't affect data columns.

```python
df1 = make_dataframe_from_config(exp_matrix, config)
df2 = make_dataframe_from_config(exp_matrix, config, metadata={'exp': 'test'})
# Data columns unchanged
data_cols = [c for c in df1.columns if c not in metadata]
assert df1[data_cols].equals(df2[data_cols])
```

## Failure Modes

### Well-Defined Failures
These are expected failure modes with clear error messages:

1. **File Not Found**
   - Trigger: Config file doesn't exist
   - Exception: `FileNotFoundError`
   - Recovery: Provide correct path or use default

2. **Invalid YAML**
   - Trigger: Malformed YAML syntax
   - Exception: `yaml.YAMLError`
   - Recovery: Fix YAML syntax

3. **Schema Violation**
   - Trigger: YAML structure doesn't match ColumnConfigDict schema
   - Exception: `ValidationError`
   - Recovery: Fix config structure

4. **Missing Required Column**
   - Trigger: exp_matrix missing required column
   - Exception: `TransformError`
   - Recovery: Add missing column or mark as optional

5. **Dimension Mismatch**
   - Trigger: Array dimension doesn't match config
   - Exception: `TransformError`
   - Recovery: Reshape array or fix config

6. **Length Mismatch**
   - Trigger: Arrays in exp_matrix have different lengths
   - Exception: `TransformError`
   - Recovery: Ensure all arrays same length

## Testing Strategy

### Contract Tests
Verify preconditions, operations, and postconditions for each function.

### Invariant Tests
Verify system invariants (INV-1 through INV-5) always hold.

### Property Tests
Use Hypothesis to verify composition properties hold for arbitrary valid inputs.

### Example Test Organization:
```
tests/
├── test_contracts.py         # Contract guard tests
├── test_invariants.py         # Semantic invariant tests  
├── test_properties.py         # Property-based tests
└── test_integration.py        # End-to-end workflows
```

## References

- **Pydantic Documentation:** https://docs.pydantic.dev/
- **Contract Testing:** Design by Contract (Bertrand Meyer)
- **Property-Based Testing:** QuickCheck, Hypothesis

---

**Version:** 1.0  
**Last Updated:** 2025-09-30  
**Authors:** flyrigloader team
