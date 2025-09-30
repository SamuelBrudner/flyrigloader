# Metadata Merge and Precedence Rules

**Version**: 1.0.0  
**Last Updated**: 2025-09-30

## Overview

FlyRigLoader collects metadata from **three sources**. This document defines the **explicit merge and precedence rules** when the same metadata key appears in multiple sources.

---

## Metadata Sources (in Precedence Order)

### 1. **Manual Metadata** (HIGHEST PRECEDENCE)
- **Source**: Explicitly passed to API functions via `metadata={}` parameter
- **Format**: Dictionary
- **Example**:
  ```python
  process_experiment_data(
      data_path="file.pkl",
      metadata={"experimenter": "Dr. Smith", "date": "2024-01-15"}  # Highest priority
  )
  ```
- **Use Case**: Override or supplement automatically extracted metadata

### 2. **Filename Extraction Metadata** (MEDIUM PRECEDENCE)
- **Source**: Extracted from filenames using regex patterns
- **Format**: Named capture groups in regex patterns
- **Example**:
  ```yaml
  extraction_patterns:
    - r"(?P<animal_id>mouse_\d+)_(?P<date>\d{8})\.pkl"
  ```
  For file `mouse_001_20240115.pkl`:
  - Extracts: `{"animal_id": "mouse_001", "date": "20240115"}`
- **Use Case**: Automatic metadata extraction from standardized filenames

### 3. **Configuration Metadata** (LOWEST PRECEDENCE)
- **Source**: Defined in YAML configuration under `datasets.{name}.metadata` or `experiments.{name}.metadata`
- **Format**: Dictionary in YAML
- **Example**:
  ```yaml
  datasets:
    my_dataset:
      metadata:
        experiment_type: "baseline"
        rig_id: "rig_001"
        sampling_rate: 60.0
  ```
- **Use Case**: Static metadata shared across all files in a dataset

---

## Precedence Rule (Last-Write-Wins with Priority)

```
Manual Metadata  >  Filename Extraction  >  Configuration Metadata
   (priority 3)        (priority 2)            (priority 1)
```

**Merge Algorithm**:
1. Start with **Configuration Metadata** (base layer)
2. Overlay **Filename Extraction Metadata** (overwrites conflicts)
3. Overlay **Manual Metadata** (overwrites conflicts, final authority)

---

## Examples

### Example 1: Manual Metadata Overrides Everything

**Configuration:**
```yaml
datasets:
  my_dataset:
    metadata:
      date: "2024-01-01"
      condition: "baseline"
```

**Filename:** `mouse_001_20240115_treatment.pkl`

**Extraction Pattern:** `r"(?P<animal_id>mouse_\d+)_(?P<date>\d{8})_(?P<condition>\w+)\.pkl"`

**API Call:**
```python
metadata = process_experiment_data(
    "mouse_001_20240115_treatment.pkl",
    metadata={"date": "2024-01-20", "experimenter": "Dr. Smith"}
)
```

**Result:**
```python
{
    "date": "2024-01-20",           # From manual (overrides extraction & config)
    "condition": "treatment",       # From filename extraction (overrides config)
    "animal_id": "mouse_001",       # From filename extraction (new key)
    "experimenter": "Dr. Smith"     # From manual (new key)
}
```

**Explanation**:
- `date`: Manual wins (2024-01-20) over extraction (20240115) and config (2024-01-01)
- `condition`: Filename extraction wins (treatment) over config (baseline)
- `animal_id`: Only in filename extraction, included
- `experimenter`: Only in manual, included

---

### Example 2: Filename Extraction Overrides Configuration

**Configuration:**
```yaml
datasets:
  my_dataset:
    metadata:
      date: "2024-01-01"
      experimenter: "Default"
```

**Filename:** `data_20240215.pkl`

**Extraction Pattern:** `r"data_(?P<date>\d{8})\.pkl"`

**API Call:**
```python
metadata = load_dataset_files(
    config_path="config.yaml",
    dataset_name="my_dataset",
    extract_metadata=True  # No manual metadata provided
)
# Returns: {"file.pkl": {"date": "20240215", "experimenter": "Default"}}
```

**Result for each file**:
```python
{
    "date": "20240215",        # From filename (overrides config "2024-01-01")
    "experimenter": "Default"  # From config (no override)
}
```

---

### Example 3: Configuration Metadata as Fallback

**Configuration:**
```yaml
datasets:
  my_dataset:
    metadata:
      sampling_rate: 60.0
      rig_id: "rig_001"
      analysis_version: "v1.2.3"
```

**Filename:** `data.pkl` (no extractable metadata)

**API Call:**
```python
files = load_dataset_files(
    config_path="config.yaml",
    dataset_name="my_dataset",
    extract_metadata=True
)
```

**Result**:
```python
{
    "sampling_rate": 60.0,
    "rig_id": "rig_001",
    "analysis_version": "v1.2.3"
}
```

**Explanation**: When no filename extraction or manual metadata is provided, configuration metadata is used as-is.

---

## Special Cases

### Date Parsing (`parse_dates=True`)

When `parse_dates=True` is specified, extracted date strings are **converted to `datetime` objects** and stored in a **separate key**:

**Extraction Result (before parsing)**:
```python
{"date": "20240115"}
```

**After Date Parsing**:
```python
{
    "date": "20240115",                    # Original string preserved
    "parsed_date": datetime(2024, 1, 15)   # New key added
}
```

**Merge Behavior**:
- `date` (string) follows normal precedence rules
- `parsed_date` (datetime) is **always derived from `date`** in the current metadata context
- Manual metadata can override `date` string, which will trigger re-parsing of `parsed_date`

---

### Metadata Type Coercion

Metadata values are **not type-coerced** during merging:

```python
# Configuration metadata
{"trial_count": 10}  # Integer

# Filename extraction
{"trial_count": "10"}  # String

# Result: Filename extraction wins with string type
{"trial_count": "10"}  # String
```

**Rule**: Precedence determines the winner; type is preserved from the winning source.

---

### Empty Metadata Values

Empty or `None` values **do not override** non-empty values:

```python
# Configuration metadata
{"condition": "baseline"}

# Filename extraction fails to extract 'condition'
# (or extraction pattern doesn't match)

# Result: Configuration value is preserved
{"condition": "baseline"}
```

**Rule**: Only **successfully extracted** metadata overwrites lower-precedence sources.

---

## Implementation Notes

### Metadata Merge Function

```python
def merge_metadata(
    config_metadata: Dict[str, Any],
    extracted_metadata: Dict[str, Any],
    manual_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Merge metadata from three sources with defined precedence.
    
    Precedence: manual > extracted > config
    
    Args:
        config_metadata: Metadata from YAML configuration (lowest priority)
        extracted_metadata: Metadata extracted from filename (medium priority)
        manual_metadata: Manually provided metadata (highest priority)
    
    Returns:
        Merged metadata dictionary with conflicts resolved by precedence
    """
    # Start with lowest precedence
    result = config_metadata.copy()
    
    # Overlay medium precedence (overwrites conflicts)
    result.update(extracted_metadata)
    
    # Overlay highest precedence (overwrites conflicts)
    if manual_metadata:
        result.update(manual_metadata)
    
    return result
```

---

## Testing Requirements

All implementations must pass these test scenarios:

1. **Manual override test**: Manual metadata overrides both filename and config
2. **Filename override test**: Filename extraction overrides config but not manual
3. **Config fallback test**: Config metadata used when no extraction/manual provided
4. **Key preservation test**: Keys unique to each source are all included
5. **Empty value test**: Empty/None values don't override non-empty values
6. **Type preservation test**: Winner's type is preserved (no coercion)

---

## Anti-Patterns to Avoid

### ❌ **Anti-Pattern 1: Implicit Merging**
```python
# BAD: No clear precedence
metadata = {**config_meta, **extracted_meta, **manual_meta}  # Last wins, but unclear
```

### ✅ **Correct Pattern: Explicit Precedence**
```python
# GOOD: Documented precedence with clear merge function
metadata = merge_metadata(config_meta, extracted_meta, manual_meta)
```

---

### ❌ **Anti-Pattern 2: Lossy Merging**
```python
# BAD: Lower-precedence sources can override higher precedence
if not manual_meta.get("date"):
    manual_meta["date"] = extracted_meta.get("date")  # Wrong direction
```

### ✅ **Correct Pattern: Preserve Higher Precedence**
```python
# GOOD: Higher precedence always wins
result["date"] = manual_meta.get("date") or extracted_meta.get("date") or config_meta.get("date")
```

---

## Related Documentation

- [Pattern Precedence Rules](PATTERN_PRECEDENCE.md)
- [Date Parsing Formats](DATE_PARSING.md)
- [API Reference](API_REFERENCE.md)
