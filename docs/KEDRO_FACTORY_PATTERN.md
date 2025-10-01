# Kedro Factory Function Pattern

## Overview

The FlyRigLoader Kedro integration provides two ways to create datasets:

1. **Direct Instantiation** - Explicit class construction
2. **Factory Function** - `create_kedro_dataset()` wrapper

This document clarifies when to use each approach and why both exist.

---

## TL;DR - Quick Guide

**Use Direct Instantiation when:**
- ✅ In `catalog.yml` configuration files
- ✅ You know exactly which dataset type you need
- ✅ Writing simple, explicit code

**Use Factory Function when:**
- ✅ Building dynamic/programmatic pipelines
- ✅ Dataset type is determined at runtime
- ✅ You want a unified API regardless of type

---

## Direct Instantiation Pattern

### Syntax

```python
from flyrigloader.kedro import FlyRigLoaderDataSet, FlyRigManifestDataSet

# For data loading
dataset = FlyRigLoaderDataSet(
    config_path="config/experiments.yaml",
    experiment_name="baseline_study",
    recursive=True,
    extract_metadata=True
)

# For manifest-only operations
manifest = FlyRigManifestDataSet(
    config_path="config/experiments.yaml",
    experiment_name="baseline_study",
    include_stats=True
)
```

### When to Use

1. **In Kedro catalog.yml files:**
   ```yaml
   experiment_data:
     type: flyrigloader.FlyRigLoaderDataSet
     config_path: "${base_dir}/config/experiments.yaml"
     experiment_name: "baseline_study"
     recursive: true
   ```

2. **When you need explicit type checking:**
   ```python
   dataset: FlyRigLoaderDataSet = FlyRigLoaderDataSet(...)
   # IDE knows exactly what type this is
   ```

3. **For simple, one-off usage:**
   ```python
   # Clear and explicit
   data = FlyRigLoaderDataSet("config.yaml", "exp1").load()
   ```

### Advantages

- ✅ **Explicit** - Clear which class you're using
- ✅ **Type-safe** - Better IDE autocomplete and type checking
- ✅ **Simple** - No indirection, direct construction
- ✅ **Fast** - No function call overhead
- ✅ **Standard** - Follows typical Python OOP patterns

### Disadvantages

- ❌ **Verbose** - Longer import statements
- ❌ **Static** - Dataset type must be known at write-time
- ❌ **Repetitive** - Same pattern repeated for each type

---

## Factory Function Pattern

### Syntax

```python
from flyrigloader.kedro import create_kedro_dataset

# Create data loading dataset
dataset = create_kedro_dataset(
    config_path="config/experiments.yaml",
    experiment_name="baseline_study",
    dataset_type="data",  # or "manifest"
    recursive=True
)

# Create manifest dataset
manifest = create_kedro_dataset(
    config_path="config/experiments.yaml",
    experiment_name="baseline_study",
    dataset_type="manifest"
)
```

### When to Use

1. **Dynamic dataset type selection:**
   ```python
   def create_pipeline_dataset(experiment: str, use_manifest: bool):
       dataset_type = "manifest" if use_manifest else "data"
       return create_kedro_dataset(
           "config.yaml",
           experiment,
           dataset_type=dataset_type
       )
   ```

2. **Programmatic pipeline generation:**
   ```python
   # Generate datasets for multiple experiments
   experiments = ["baseline", "treatment_1", "treatment_2"]
   
   datasets = {
       f"{exp}_data": create_kedro_dataset(
           "config.yaml", 
           exp,
           dataset_type="data"
       )
       for exp in experiments
   }
   ```

3. **API consistency across calls:**
   ```python
   # Same function signature regardless of type
   def load_experiment(name: str, as_manifest: bool = False):
       dataset = create_kedro_dataset(
           get_config_path(),
           name,
           dataset_type="manifest" if as_manifest else "data"
       )
       return dataset.load()
   ```

4. **Configuration-driven pipelines:**
   ```python
   pipeline_config = {
       "datasets": [
           {"name": "exp1", "type": "data"},
           {"name": "exp2", "type": "manifest"}
       ]
   }
   
   datasets = [
       create_kedro_dataset(
           "config.yaml",
           ds["name"],
           dataset_type=ds["type"]
       )
       for ds in pipeline_config["datasets"]
   ]
   ```

### Advantages

- ✅ **Dynamic** - Dataset type determined at runtime
- ✅ **Flexible** - Easy to change type based on conditions
- ✅ **Unified API** - One function for all types
- ✅ **Convenient** - Single import for all cases

### Disadvantages

- ❌ **Less explicit** - Type not clear from code
- ❌ **String-based** - `dataset_type` is a string parameter
- ❌ **Indirection** - Extra function call layer
- ❌ **Type checking** - Returns `Union[FlyRigLoaderDataSet, FlyRigManifestDataSet]`

---

## Implementation Details

### Factory Function Signature

```python
def create_kedro_dataset(
    config_path: Union[str, Path],
    experiment_name: str,
    *,
    dataset_type: str = "data",  # "data" or "manifest"
    **options: Any
) -> Union[FlyRigLoaderDataSet, FlyRigManifestDataSet]:
    """
    Factory function for creating Kedro datasets.
    
    Args:
        config_path: Path to FlyRigLoader configuration
        experiment_name: Name of experiment to load
        dataset_type: Type of dataset - "data" or "manifest"
        **options: Additional options passed to dataset constructor
        
    Returns:
        Configured dataset instance
        
    Raises:
        ValueError: If dataset_type is invalid
        
    Example:
        >>> dataset = create_kedro_dataset(
        ...     "config/experiments.yaml",
        ...     "baseline_study",
        ...     dataset_type="data",
        ...     recursive=True
        ... )
    """
    if dataset_type == "data":
        return FlyRigLoaderDataSet(config_path, experiment_name, **options)
    elif dataset_type == "manifest":
        return FlyRigManifestDataSet(config_path, experiment_name, **options)
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")
```

### Lazy Import Pattern

The factory function uses a lazy import pattern to avoid circular dependencies:

```python
# In flyrigloader/kedro/__init__.py
def create_kedro_dataset(*args, **kwargs):
    """Lazy wrapper to avoid circular imports."""
    from flyrigloader.api import create_kedro_dataset as _create_kedro_dataset
    return _create_kedro_dataset(*args, **kwargs)
```

This allows the function to be imported from `flyrigloader.kedro` while being implemented in `flyrigloader.api`.

---

## Comparison Matrix

| Feature | Direct Instantiation | Factory Function |
|---------|---------------------|------------------|
| **Type Safety** | ✅ Strong | ⚠️ Weaker (Union type) |
| **Explicitness** | ✅ Very clear | ⚠️ Less obvious |
| **Runtime Flexibility** | ❌ None | ✅ Full |
| **IDE Support** | ✅ Excellent | ⚠️ Good |
| **Code Length** | ⚠️ Longer | ✅ Shorter |
| **Dynamic Pipelines** | ❌ Difficult | ✅ Easy |
| **Catalog.yml** | ✅ Standard | ❌ Not supported |
| **Type Checking (mypy)** | ✅ Precise | ⚠️ Union type |
| **Learning Curve** | ✅ Obvious | ⚠️ Requires docs |

---

## Best Practices

### ✅ DO: Use Direct Instantiation in Catalog Files

```yaml
# conf/base/catalog.yml
baseline_data:
  type: flyrigloader.FlyRigLoaderDataSet  # ✅ Explicit
  config_path: "${base_dir}/config/experiments.yaml"
  experiment_name: "baseline"
```

### ✅ DO: Use Factory for Dynamic Pipelines

```python
def create_experiment_pipeline(experiments: List[str]):
    """Generate pipeline dynamically."""
    nodes = []
    for exp in experiments:
        # ✅ Factory great for loops
        dataset = create_kedro_dataset(
            "config.yaml",
            exp,
            dataset_type="data"
        )
        nodes.append(create_processing_node(dataset))
    return Pipeline(nodes)
```

### ✅ DO: Use Direct Instantiation for Type Clarity

```python
def process_manifest(manifest: FlyRigManifestDataSet) -> Dict:
    """
    Process a manifest dataset.
    
    Args:
        manifest: Manifest dataset (type is explicit)
    """
    return analyze_manifest(manifest.load())

# ✅ Call site is type-safe
manifest = FlyRigManifestDataSet("config.yaml", "exp1")
result = process_manifest(manifest)
```

### ❌ DON'T: Use Factory in Catalog Files

```yaml
# conf/base/catalog.yml
baseline_data:
  type: flyrigloader.create_kedro_dataset  # ❌ Won't work!
  # Factory functions aren't dataset classes
```

### ❌ DON'T: Use Factory When Type is Known

```python
# ❌ Unnecessary indirection
dataset = create_kedro_dataset(
    "config.yaml",
    "exp1",
    dataset_type="data"  # Why not just use FlyRigLoaderDataSet?
)

# ✅ Better - explicit and clear
dataset = FlyRigLoaderDataSet("config.yaml", "exp1")
```

---

## Real-World Examples

### Example 1: Static Pipeline (Direct Instantiation)

```python
# src/project/pipelines/analysis/nodes.py
from flyrigloader.kedro import FlyRigLoaderDataSet

def analyze_experiment(config_path: str, exp_name: str) -> pd.DataFrame:
    """
    Analyze a single experiment.
    
    Uses direct instantiation because:
    - Dataset type is always the same
    - Function is called with known parameters
    - Type safety is important
    """
    # ✅ Direct instantiation - clear and explicit
    dataset = FlyRigLoaderDataSet(
        config_path=config_path,
        experiment_name=exp_name,
        recursive=True
    )
    
    data = dataset.load()
    return perform_analysis(data)
```

### Example 2: Dynamic Pipeline (Factory Function)

```python
# src/project/pipelines/batch/pipeline.py
from flyrigloader.kedro import create_kedro_dataset
from typing import Dict, List

def generate_batch_pipeline(
    experiments: List[str],
    use_manifests: bool = False
) -> Dict[str, Any]:
    """
    Generate pipeline for multiple experiments.
    
    Uses factory function because:
    - Dataset type varies based on use_manifests flag
    - Number of datasets determined at runtime
    - Programmatic generation is key requirement
    """
    datasets = {}
    
    # ✅ Factory function - runtime flexibility
    dataset_type = "manifest" if use_manifests else "data"
    
    for exp in experiments:
        dataset_name = f"{exp}_{dataset_type}"
        datasets[dataset_name] = create_kedro_dataset(
            config_path="config/experiments.yaml",
            experiment_name=exp,
            dataset_type=dataset_type,
            recursive=True
        )
    
    return datasets
```

### Example 3: Conditional Loading (Factory Function)

```python
# src/project/utils/loaders.py
from flyrigloader.kedro import create_kedro_dataset

def smart_load(
    experiment: str,
    check_size_first: bool = True
) -> Union[pd.DataFrame, FileManifest]:
    """
    Smart loader that checks file size before full load.
    
    Uses factory function because:
    - Conditionally loads manifest first
    - Then decides whether to load full data
    - Type changes based on runtime conditions
    """
    if check_size_first:
        # ✅ First check manifest
        manifest_ds = create_kedro_dataset(
            "config.yaml",
            experiment,
            dataset_type="manifest"
        )
        manifest = manifest_ds.load()
        
        total_size = sum(f.size_bytes for f in manifest.files)
        
        if total_size > 10_000_000:  # 10MB threshold
            print(f"Large dataset ({total_size} bytes), using manifest")
            return manifest
    
    # Load full data
    data_ds = create_kedro_dataset(
        "config.yaml",
        experiment,
        dataset_type="data"
    )
    return data_ds.load()
```

### Example 4: Configuration-Driven (Factory Function)

```python
# src/project/config/pipeline_builder.py
from flyrigloader.kedro import create_kedro_dataset
import yaml

def build_from_config(config_file: str) -> Dict:
    """
    Build datasets from YAML configuration.
    
    Uses factory function because:
    - Dataset types specified in config file
    - Fully dynamic construction
    - No compile-time knowledge of types
    """
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    datasets = {}
    
    # ✅ Factory perfect for config-driven creation
    for ds_config in config["datasets"]:
        dataset = create_kedro_dataset(
            config_path=ds_config["config_path"],
            experiment_name=ds_config["experiment"],
            dataset_type=ds_config["type"],
            **ds_config.get("options", {})
        )
        datasets[ds_config["name"]] = dataset
    
    return datasets
```

---

## Migration Guide

### From Old API (if upgrading)

If you previously used different patterns, here's how to migrate:

```python
# OLD: Direct path to class
from flyrigloader.kedro.datasets import FlyRigLoaderDataSet

# NEW: Import from main package
from flyrigloader.kedro import FlyRigLoaderDataSet

# OLD: Using 'filepath' parameter
dataset = FlyRigLoaderDataSet(
    filepath="config.yaml",  # deprecated
    experiment_name="exp1"
)

# NEW: Using 'config_path' parameter
dataset = FlyRigLoaderDataSet(
    config_path="config.yaml",  # standardized
    experiment_name="exp1"
)

# OLD: Accessing .filepath
path = dataset.filepath  # deprecated, emits warning

# NEW: Accessing .config_path
path = dataset.config_path  # standard
```

---

## Decision Tree

```
Need to create a Kedro dataset?
│
├─ Is this in a catalog.yml file?
│  └─ YES → Use Direct Instantiation
│
├─ Is the dataset type known at write-time?
│  ├─ YES → Prefer Direct Instantiation
│  └─ NO → Use Factory Function
│
├─ Are you building dynamic/programmatic pipelines?
│  └─ YES → Use Factory Function
│
├─ Do you need strong type checking?
│  └─ YES → Use Direct Instantiation
│
└─ Default → Use Direct Instantiation (more explicit)
```

---

## Summary

### Core Principle

**Use the simplest pattern that meets your needs:**

- **Static, known types** → Direct Instantiation
- **Dynamic, runtime types** → Factory Function

Both patterns are valid and supported. Choose based on your specific use case.

### Key Takeaways

1. **Direct Instantiation** is the default choice for most use cases
2. **Factory Function** shines in dynamic/programmatic scenarios
3. **Catalog files** must use Direct Instantiation
4. **Type safety** favors Direct Instantiation
5. **Runtime flexibility** favors Factory Function

---

**Version:** 1.0  
**Last Updated:** 2025-09-30  
**Status:** Active
