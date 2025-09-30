# Pattern Precedence and Filtering Rules

**Version**: 1.0.0  
**Last Updated**: 2025-09-30

## Overview

FlyRigLoader uses multiple pattern systems for file discovery and filtering. This document defines the **explicit precedence rules** and interaction semantics to eliminate ambiguity.

---

## Pattern Types

### 1. **Ignore Patterns** (`ignore_patterns`, `ignore_substrings`)
- **Purpose**: Exclude files from discovery
- **Format**: Glob patterns (e.g., `"._*"`, `"temp*"`, `"*backup*"`)
- **Scope**: Global (project-level) or experiment-specific
- **Precedence**: **HIGHEST** - Ignored files are never included, regardless of other patterns

### 2. **Mandatory Substring Filters** (`mandatory_substrings`)
- **Purpose**: Require files to contain specific substrings (OR logic)
- **Format**: Plain strings (e.g., `"baseline"`, `"experiment"`)
- **Scope**: Experiment-specific
- **Precedence**: **HIGH** - Files must match at least one mandatory substring to be included

### 3. **Dataset Patterns** (`datasets.{name}.patterns`)
- **Purpose**: Define which files belong to a dataset
- **Format**: Glob patterns (e.g., `"*baseline*"`, `"exp_*.pkl"`)
- **Scope**: Dataset-specific
- **Precedence**: **MEDIUM** - Files must match dataset patterns to be considered

### 4. **Filename Extraction Patterns** (`extraction_patterns`)
- **Purpose**: Extract metadata from filenames
- **Format**: Regex with named groups (e.g., `r"(?P<date>\d{8})"`)
- **Scope**: Global or dataset/experiment-specific
- **Precedence**: **LOWEST** - Does not affect file inclusion, only metadata extraction

---

## Precedence Rules (Applied in Order)

```
┌─────────────────────────────────────────────┐
│ Step 1: Apply Ignore Patterns (EXCLUDE)    │
│   → If file matches any ignore pattern:    │
│     REJECT file, skip remaining steps      │
└─────────────────────────────────────────────┘
              ↓ (if not ignored)
┌─────────────────────────────────────────────┐
│ Step 2: Apply Mandatory Substrings (FILTER)│
│   → If mandatory_substrings is defined:    │
│     - File must contain ≥1 substring (OR)  │
│     - If no match: REJECT file             │
│   → If not defined: ACCEPT file            │
└─────────────────────────────────────────────┘
              ↓ (if passed)
┌─────────────────────────────────────────────┐
│ Step 3: Apply Dataset Patterns (MATCH)     │
│   → If dataset patterns are defined:       │
│     - File must match ≥1 pattern (OR)      │
│     - If no match: REJECT file             │
│   → If not defined: ACCEPT file            │
└─────────────────────────────────────────────┘
              ↓ (if accepted)
┌─────────────────────────────────────────────┐
│ Step 4: Extract Metadata (NO FILTERING)    │
│   → Apply extraction patterns to filename  │
│   → Multiple patterns can all match        │
│   → Metadata merged (see METADATA_MERGE.md)│
└─────────────────────────────────────────────┘
```

---

## Examples

### Example 1: Ignore Patterns Override Everything

```yaml
project:
  ignore_substrings: ["._", "backup"]

datasets:
  my_dataset:
    patterns: ["*baseline*"]
```

**Files:**
- `baseline_20240101.csv` → ✅ **ACCEPTED** (matches dataset pattern, not ignored)
- `._baseline_20240101.csv` → ❌ **REJECTED** (matches ignore pattern, even though it matches dataset pattern)
- `baseline_backup_20240101.csv` → ❌ **REJECTED** (contains "backup", ignored)

**Rule**: Ignore patterns have **highest precedence**.

---

### Example 2: Mandatory Substrings (OR Logic)

```yaml
experiments:
  my_experiment:
    filters:
      mandatory_experiment_strings: ["baseline", "control"]
    datasets: ["my_dataset"]

datasets:
  my_dataset:
    patterns: ["*.csv"]
```

**Files:**
- `baseline_20240101.csv` → ✅ **ACCEPTED** (contains "baseline")
- `control_20240101.csv` → ✅ **ACCEPTED** (contains "control")
- `treatment_20240101.csv` → ❌ **REJECTED** (contains neither "baseline" nor "control")
- `baseline_control.csv` → ✅ **ACCEPTED** (contains both, only needs one)

**Rule**: At least **one** mandatory substring must match (OR logic).

---

### Example 3: Dataset Patterns (OR Logic)

```yaml
datasets:
  my_dataset:
    patterns: ["*baseline*", "*ctrl*"]
```

**Files:**
- `baseline_20240101.csv` → ✅ **ACCEPTED** (matches `*baseline*`)
- `ctrl_20240101.csv` → ✅ **ACCEPTED** (matches `*ctrl*`)
- `treatment_20240101.csv` → ❌ **REJECTED** (matches no pattern)
- `baseline_ctrl.csv` → ✅ **ACCEPTED** (matches both, only needs one)

**Rule**: At least **one** dataset pattern must match (OR logic).

---

### Example 4: Extraction Patterns Don't Filter

```yaml
project:
  extraction_patterns:
    - r"(?P<date>\d{8})"
    - r"(?P<condition>baseline|treatment)"

datasets:
  my_dataset:
    patterns: ["*.csv"]
```

**Files:**
- `data_20240101.csv` → ✅ **ACCEPTED** (pattern extracts `date=20240101`)
- `data_baseline.csv` → ✅ **ACCEPTED** (pattern extracts `condition=baseline`)
- `data_unknown.csv` → ✅ **ACCEPTED** (no extraction, but still included)

**Rule**: Extraction patterns **never filter** files, they only add metadata.

---

## Edge Cases

### Multiple Pattern Sources

If patterns are defined at **multiple levels**, they are **merged**:

```yaml
project:
  ignore_substrings: ["._", "temp"]

experiments:
  my_experiment:
    filters:
      ignore_substrings: ["backup"]  # Added to project-level
```

**Result**: Files matching `"._"`, `"temp"`, OR `"backup"` are ignored.

**Rule**: Experiment-level patterns **extend** (not replace) project-level patterns.

---

### Empty Pattern Lists

- **Empty `ignore_substrings`**: No files are ignored based on substrings
- **Empty `mandatory_substrings`**: No mandatory filtering (all files pass)
- **Empty `datasets.patterns`**: All files in directory are considered (use with caution!)

---

## Implementation Notes

### Pattern Matching Functions

```python
def _apply_ignore_patterns(files: List[str], patterns: List[str]) -> List[str]:
    """Step 1: Remove files matching ignore patterns (highest precedence)."""
    return [f for f in files if not any(fnmatch(f, p) for p in patterns)]

def _apply_mandatory_filters(files: List[str], substrings: List[str]) -> List[str]:
    """Step 2: Keep only files containing mandatory substrings (OR logic)."""
    if not substrings:
        return files  # No mandatory filters defined
    return [f for f in files if any(s in f for s in substrings)]

def _apply_dataset_patterns(files: List[str], patterns: List[str]) -> List[str]:
    """Step 3: Keep only files matching dataset patterns (OR logic)."""
    if not patterns:
        return files  # No patterns defined
    return [f for f in files if any(fnmatch(f, p) for p in patterns)]

def _extract_metadata(files: List[str], patterns: List[str]) -> Dict[str, Dict[str, Any]]:
    """Step 4: Extract metadata without filtering."""
    # Never removes files, only adds metadata
    ...
```

---

## Testing Requirements

All implementations must pass these test scenarios:

1. **Ignore precedence**: Ignored files never appear, even if they match other patterns
2. **Mandatory substring OR logic**: Any one match is sufficient
3. **Dataset pattern OR logic**: Any one match is sufficient
4. **Extraction non-filtering**: Files without extractable metadata are still included
5. **Pattern merging**: Experiment patterns extend project patterns

---

## Related Documentation

- [Metadata Merge Rules](METADATA_MERGE.md)
- [Dimension Handling](DIMENSION_HANDLING.md)
- [API Reference](API_REFERENCE.md)
