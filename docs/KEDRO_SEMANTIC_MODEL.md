# Kedro Integration Semantic Model

## Overview

This document defines the formal semantic model for FlyRigLoader's Kedro integration, including domain concepts, invariants, operation contracts, and composition properties.

## Domain Concepts

### FlyRigLoaderDataSet
**Definition:** Kedro AbstractDataset implementation that provides full data loading with FlyRigLoader's complete discovery and transformation pipeline.

**Invariants:**
- Always returns a pandas DataFrame
- Operations are thread-safe (protected by RLock)
- Read-only (save operations always raise NotImplementedError)
- Configuration is immutable after initialization
- Output DataFrames always include Kedro metadata columns

**Type Signature:**
```python
class FlyRigLoaderDataSet(AbstractDataset[None, pd.DataFrame])
```

### FlyRigManifestDataSet
**Definition:** Kedro AbstractDataset implementation for lightweight file discovery without full data loading.

**Invariants:**
- Always returns a FileManifest object
- Operations are thread-safe (protected by RLock)
- Read-only (save operations always raise NotImplementedError)
- Fast execution (<100ms typical)
- No data loading occurs (memory efficient)

**Type Signature:**
```python
class FlyRigManifestDataSet(AbstractDataset[None, FileManifest])
```

### Configuration
**Definition:** FlyRigLoader configuration loaded from YAML files.

**Invariants:**
- Immutable after load
- Validated against schema
- File path must exist
- Experiment name must be valid

### Kedro Metadata Columns
**Definition:** Additional columns added to DataFrames for Kedro pipeline compatibility.

**Standard Columns:**
- `kedro_run_id`: Unique identifier for pipeline run
- `kedro_node_name`: Name of the node that produced data
- `experiment_name`: Name of the experiment
- `load_timestamp`: Timestamp when data was loaded

## Core System Invariants

### INV-1: Read-Only Operations
**Statement:** All dataset operations are read-only; save operations always raise NotImplementedError.

**Rationale:** Experimental data is immutable at source; datasets only load existing data.

**Verification:**
```python
def test_save_raises_not_implemented():
    dataset = FlyRigLoaderDataSet(config_path="...", experiment_name="...")
    with pytest.raises(NotImplementedError):
        dataset.save(some_dataframe)
```

### INV-2: Thread Safety
**Statement:** All dataset operations are thread-safe and can be called concurrently from multiple Kedro nodes.

**Rationale:** Kedro pipelines may execute nodes in parallel.

**Implementation:** Protected by threading.RLock

**Verification:**
```python
def test_concurrent_load_operations():
    dataset = FlyRigLoaderDataSet(config_path="...", experiment_name="...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(dataset.load) for _ in range(10)]
        results = [f.result() for f in futures]
    # All loads succeed without corruption
    assert all(isinstance(r, pd.DataFrame) for r in results)
```

### INV-3: Configuration Immutability
**Statement:** Configuration parameters are immutable after dataset initialization.

**Rationale:** Ensures consistent behavior across multiple load operations.

**Verification:**
```python
def test_config_immutability():
    dataset = FlyRigLoaderDataSet(config_path="...", experiment_name="test")
    config1 = dataset._config
    dataset.load()
    config2 = dataset._config
    assert config1 is config2  # Same object, not reloaded
```

### INV-4: Output Type Consistency
**Statement:** Dataset load operations always return the expected type (DataFrame or FileManifest).

**Verification:**
```python
def test_output_type_consistency():
    data_dataset = FlyRigLoaderDataSet(config_path="...", experiment_name="...")
    result = data_dataset.load()
    assert isinstance(result, pd.DataFrame)
    
    manifest_dataset = FlyRigManifestDataSet(config_path="...", experiment_name="...")
    result = manifest_dataset.load()
    assert isinstance(result, FileManifest)
```

### INV-5: Metadata Column Presence
**Statement:** DataFrames from FlyRigLoaderDataSet always contain Kedro metadata columns when include_kedro_metadata=True.

**Verification:**
```python
def test_metadata_columns_present():
    dataset = FlyRigLoaderDataSet(
        config_path="...",
        experiment_name="test",
        transform_options={"include_kedro_metadata": True}
    )
    df = dataset.load()
    assert "experiment_name" in df.columns
    assert "load_timestamp" in df.columns
```

## Operation Contracts

### FlyRigLoaderDataSet._load()

```python
def _load(self) -> pd.DataFrame
```

**Preconditions:**
- `config_path` points to existing, valid YAML file
- `experiment_name` is non-empty string
- Experiment exists in configuration
- At least one file matches discovery criteria

**Postconditions:**
- Returns non-empty pandas DataFrame
- DataFrame row count > 0
- All required columns from configuration are present
- If `include_kedro_metadata=True`, metadata columns present
- Thread-safe (can be called concurrently)

**Raises:**
- `ConfigError`: Invalid configuration or missing file
- `DiscoveryError`: No files found matching criteria
- `TransformError`: Data transformation failed
- `FileNotFoundError`: Config file doesn't exist

**Invariants Preserved:**
- INV-1: Read-only (no side effects)
- INV-2: Thread-safe operation
- INV-3: Configuration remains immutable
- INV-4: Returns DataFrame type
- INV-5: Metadata columns present if requested

**Example Contract Test:**
```python
def test_load_contract():
    # PRE: valid config and experiment
    config_path = create_valid_config()
    experiment_name = "baseline_study"
    
    dataset = FlyRigLoaderDataSet(
        config_path=config_path,
        experiment_name=experiment_name
    )
    
    # OPERATION
    result = dataset.load()
    
    # POST: returns DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # POST: non-empty
    assert len(result) > 0
    
    # POST: thread-safe (no errors on concurrent access)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(dataset.load) for _ in range(5)]
        results = [f.result() for f in futures]
        assert all(isinstance(r, pd.DataFrame) for r in results)
```

### FlyRigLoaderDataSet._save()

```python
def _save(self, data: pd.DataFrame) -> None
```

**Preconditions:**
- None (always raises)

**Postconditions:**
- Always raises NotImplementedError
- No side effects

**Raises:**
- `NotImplementedError`: Always (read-only dataset)

**Invariants Preserved:**
- INV-1: Read-only enforced

### FlyRigLoaderDataSet._exists()

```python
def _exists(self) -> bool
```

**Preconditions:**
- None

**Postconditions:**
- Returns True if configuration file exists and is readable
- Returns False otherwise
- No side effects (idempotent)

**Raises:**
- None (catches all exceptions internally)

**Invariants Preserved:**
- INV-1: Read-only (no mutations)
- INV-2: Thread-safe

**Properties:**
- **Idempotent:** Multiple calls return same result
- **Fast:** Typically <10ms

### FlyRigLoaderDataSet._describe()

```python
def _describe(self) -> Dict[str, Any]
```

**Preconditions:**
- None

**Postconditions:**
- Returns dictionary with dataset metadata
- Contains keys: "config_path", "experiment_name", "type"
- No side effects

**Raises:**
- None (catches all exceptions internally)

**Invariants Preserved:**
- INV-1: Read-only
- INV-2: Thread-safe

### FlyRigManifestDataSet._load()

```python
def _load(self) -> FileManifest
```

**Preconditions:**
- `config_path` points to existing, valid YAML file
- `experiment_name` is non-empty string
- Experiment exists in configuration

**Postconditions:**
- Returns FileManifest object
- Manifest contains file metadata (paths, sizes, timestamps)
- No actual data loaded (memory efficient)
- Execution time < 100ms (typically)

**Raises:**
- `ConfigError`: Invalid configuration
- `DiscoveryError`: File discovery failed

**Invariants Preserved:**
- INV-1: Read-only
- INV-2: Thread-safe
- INV-4: Returns FileManifest type

## Composition Properties

### Property 1: Load Idempotency
**Statement:** Loading the same dataset multiple times produces equivalent results.

**Verification:**
```python
def test_load_idempotency():
    dataset = FlyRigLoaderDataSet(config_path="...", experiment_name="test")
    df1 = dataset.load()
    df2 = dataset.load()
    
    # Same structure
    assert list(df1.columns) == list(df2.columns)
    assert len(df1) == len(df2)
    
    # Same data (excluding timestamps)
    cols_to_compare = [c for c in df1.columns if "timestamp" not in c.lower()]
    assert df1[cols_to_compare].equals(df2[cols_to_compare])
```

### Property 2: Concurrent Access Safety
**Statement:** Concurrent load operations from multiple threads produce valid results without corruption.

**Verification:**
```python
def test_concurrent_access_safety():
    dataset = FlyRigLoaderDataSet(config_path="...", experiment_name="test")
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(dataset.load) for _ in range(20)]
        results = [f.result() for f in futures]
    
    # All results valid
    assert all(isinstance(r, pd.DataFrame) for r in results)
    assert all(len(r) > 0 for r in results)
```

### Property 3: Dataset Type Determines Output
**Statement:** Dataset class determines output type regardless of parameters.

**Verification:**
```python
def test_dataset_type_determines_output():
    # Same config, different dataset types
    data_ds = FlyRigLoaderDataSet(config_path="...", experiment_name="test")
    manifest_ds = FlyRigManifestDataSet(config_path="...", experiment_name="test")
    
    result1 = data_ds.load()
    result2 = manifest_ds.load()
    
    assert isinstance(result1, pd.DataFrame)
    assert isinstance(result2, FileManifest)
```

### Property 4: Configuration Independence
**Statement:** Datasets with different configurations operate independently.

**Verification:**
```python
def test_configuration_independence():
    dataset1 = FlyRigLoaderDataSet(config_path="...", experiment_name="exp1")
    dataset2 = FlyRigLoaderDataSet(config_path="...", experiment_name="exp2")
    
    df1 = dataset1.load()
    df2 = dataset2.load()
    
    # Different experiments produce different data
    assert not df1.equals(df2)
```

## Failure Modes

### Well-Defined Failures

1. **Configuration File Not Found**
   - Trigger: config_path points to non-existent file
   - Exception: `FileNotFoundError` or `ConfigError`
   - Recovery: Provide correct path

2. **Invalid YAML Structure**
   - Trigger: Malformed YAML or wrong schema
   - Exception: `ConfigError` with validation details
   - Recovery: Fix YAML structure

3. **Experiment Not Found**
   - Trigger: experiment_name not in configuration
   - Exception: `ConfigError`
   - Recovery: Use valid experiment name or add to config

4. **No Files Discovered**
   - Trigger: No files match discovery criteria
   - Exception: `DiscoveryError`
   - Recovery: Check file patterns, paths, or date ranges

5. **Data Transformation Failed**
   - Trigger: Incompatible data format or schema mismatch
   - Exception: `TransformError`
   - Recovery: Fix data format or update schema

6. **Save Operation Attempted**
   - Trigger: Calling save() on read-only dataset
   - Exception: `NotImplementedError`
   - Recovery: Use different dataset type or don't save

## Relationship to Core FlyRigLoader

### Delegation Model

```
FlyRigLoaderDataSet
    │
    ├─> Config Loading (flyrigloader.config.yaml_config)
    ├─> File Discovery (flyrigloader.discovery.files)
    ├─> Data Loading (flyrigloader.io.loaders)
    └─> DataFrame Transform (flyrigloader.io.transformers)
```

### Kedro-Specific Additions

1. **AbstractDataset Interface**
   - Wraps FlyRigLoader calls in _load(), _exists(), _describe()
   - Adds thread safety with RLock
   - Implements Kedro lifecycle expectations

2. **Metadata Integration**
   - Adds Kedro metadata columns to DataFrames
   - Tracks pipeline execution context
   - Enables data lineage

3. **Error Translation**
   - Catches FlyRigLoader exceptions
   - Translates to Kedro-compatible format
   - Preserves error context

## Testing Strategy

### Contract Tests
Verify preconditions, operations, and postconditions for each method.

Example: `tests/flyrigloader/kedro/test_contracts.py`

### Invariant Tests
Verify system invariants (INV-1 through INV-5) always hold.

Example: `tests/flyrigloader/kedro/test_invariants.py`

### Property Tests
Use Hypothesis to verify composition properties for arbitrary valid inputs.

Example: `tests/flyrigloader/kedro/test_properties.py`

### Integration Tests
Verify end-to-end Kedro pipeline integration.

Example: `tests/flyrigloader/kedro/test_integration.py`

## References

- **Kedro Documentation:** https://docs.kedro.org/
- **AbstractDataset Interface:** https://docs.kedro.org/en/stable/data/data_catalog.html
- **FlyRigLoader Core:** `docs/SEMANTIC_MODEL.md`
- **Column Config Model:** `docs/SEMANTIC_MODEL.md`

---

**Version:** 1.0  
**Last Updated:** 2025-09-30  
**Status:** Active
