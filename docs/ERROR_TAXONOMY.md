# Error Taxonomy and Exception Hierarchy

**Version**: 2.0.0-dev  
**Last Updated**: 2025-09-30

## Overview

This document defines the **complete exception hierarchy** for FlyRigLoader, establishing clear error handling contracts per Priority 3.1 of the semantic model review. Each exception type has specific semantics, recovery strategies, and usage guidelines.

---

## Design Principles

### 1. **Error Categories Match User Mental Model**

Exceptions are organized by **who needs to fix the problem**:

```
FlyRigLoaderError (Base)
├── ConfigurationError       → User fixes config file
├── DataFormatError          → User checks/fixes data files
├── FileSystemError          → User checks paths/permissions
└── InternalError            → Library bug (report to maintainers)
```

### 2. **Clear Recovery Paths**

Every exception includes:
- **What went wrong** (error message)
- **Why it happened** (context)
- **How to fix it** (recovery suggestion)

### 3. **Fail Fast and Loud**

Following the user's "fail loud and fast" rule:
- Errors are raised immediately at point of detection
- No silent failures or fallback logic
- Clear error messages with actionable information

---

## Exception Hierarchy

### Base Exception

```python
class FlyRigLoaderError(Exception):
    """
    Base exception for all FlyRigLoader errors.
    
    All FlyRigLoader exceptions inherit from this base class,
    allowing users to catch all library-specific errors with a
    single except clause.
    
    Attributes:
        message: Human-readable error description
        context: Dictionary of contextual information
        recovery_hint: Optional suggestion for fixing the error
    """
```

**Usage**:
```python
try:
    # FlyRigLoader operations
    ...
except FlyRigLoaderError as e:
    # Catch all library errors
    logger.error(f"FlyRigLoader error: {e}")
    if e.recovery_hint:
        logger.info(f"Suggestion: {e.recovery_hint}")
```

---

## User Error Categories

These errors indicate problems with user input that must be fixed by the user.

### 1. ConfigurationError

**What**: Invalid or missing configuration

```python
class ConfigurationError(FlyRigLoaderError):
    """
    Configuration file or configuration data is invalid.
    
    Raised when:
    - YAML syntax errors
    - Missing required configuration sections
    - Invalid configuration values
    - Schema validation failures
    - Pattern compilation errors
    
    Recovery: Fix the configuration file and retry.
    """
```

**Examples**:

```python
# Missing required section
raise ConfigurationError(
    "Missing 'project.directories.major_data_directory' in configuration",
    context={"config_path": "config.yaml"},
    recovery_hint="Add 'major_data_directory' under 'project.directories' section"
)

# Invalid regex pattern
raise ConfigurationError(
    f"Invalid regex pattern: '{pattern}'",
    context={"pattern": pattern, "field": "extraction_patterns"},
    recovery_hint="Fix the regex syntax in extraction_patterns"
)

# Empty required field
raise ConfigurationError(
    "Dataset 'rig' field cannot be empty",
    context={"dataset": "baseline_behavior"},
    recovery_hint="Provide a valid rig identifier for the dataset"
)
```

**Subcategories**:

```python
class ConfigurationValidationError(ConfigurationError):
    """Pydantic validation errors during config parsing"""

class ConfigurationMigrationError(ConfigurationError):
    """Errors during config version migration"""
```

---

### 2. DataFormatError

**What**: Data files are corrupted, malformed, or incompatible

```python
class DataFormatError(FlyRigLoaderError):
    """
    Data file format is invalid or incompatible.
    
    Raised when:
    - Pickle file is corrupted
    - Unexpected data structure
    - Missing required fields in experimental matrix
    - Type mismatches (e.g., expecting array, got scalar)
    - Dimension mismatches
    
    Recovery: Check the data file or regenerate it from source.
    """
```

**Examples**:

```python
# Corrupted pickle
raise DataFormatError(
    f"Failed to load pickle file: {path}",
    context={"file_path": str(path), "error": str(original_error)},
    recovery_hint="File may be corrupted. Try regenerating from source data."
)

# Missing required column
raise DataFormatError(
    f"Missing required column 't' (time) in experimental matrix",
    context={"file_path": str(path), "available_columns": list(data.keys())},
    recovery_hint="Ensure your experimental data includes a time array 't'"
)

# Dimension mismatch
raise DataFormatError(
    f"signal_disp dimension {shape} doesn't match time dimension {t_len}",
    context={"file_path": str(path), "shape": shape, "time_length": t_len},
    recovery_hint="Check that signal_disp has one dimension matching the time array length"
)
```

**Subcategories**:

```python
class PickleLoadError(DataFormatError):
    """Failed to unpickle data file"""

class DataSchemaError(DataFormatError):
    """Data doesn't match expected schema"""

class DimensionMismatchError(DataFormatError):
    """Array dimensions incompatible with configuration"""
```

---

### 3. FileSystemError

**What**: File system operations failed

```python
class FileSystemError(FlyRigLoaderError):
    """
    File system access or manipulation failed.
    
    Raised when:
    - File or directory doesn't exist
    - Permission denied
    - Path traversal security violation
    - Disk full
    - Too many open files
    
    Recovery: Check paths, permissions, and disk space.
    """
```

**Examples**:

```python
# File not found
raise FileNotFoundError(
    f"Configuration file not found: {path}",
    context={"config_path": str(path)},
    recovery_hint="Check the file path and ensure the file exists"
)

# Permission denied
raise PermissionError(
    f"Permission denied accessing: {path}",
    context={"path": str(path), "operation": "read"},
    recovery_hint="Check file permissions or run with appropriate privileges"
)

# Path traversal attempt
raise SecurityError(
    f"Path traversal detected: {path}",
    context={"path": str(path), "base_dir": str(base_dir)},
    recovery_hint="Ensure path is within the allowed directory structure"
)
```

**Subcategories**:

```python
class FileNotFoundError(FileSystemError):
    """File or directory doesn't exist"""

class PermissionError(FileSystemError):
    """Insufficient permissions for operation"""

class SecurityError(FileSystemError):
    """Security policy violation (e.g., path traversal)"""
```

---

## Library Error Categories

These errors indicate bugs in FlyRigLoader itself.

### 4. InternalError

**What**: Unexpected internal state or logic error

```python
class InternalError(FlyRigLoaderError):
    """
    Internal library error indicating a bug.
    
    Raised when:
    - Unreachable code is reached
    - Invariant violation
    - Unexpected internal state
    - Programming logic error
    
    Recovery: This is a bug - please report with context to maintainers.
    """
```

**Examples**:

```python
# Unreachable code path
raise InternalError(
    "Reached unreachable code path",
    context={"function": "process_data", "state": state},
    recovery_hint="This is a bug. Please report to FlyRigLoader maintainers."
)

# Invariant violation
raise InternalError(
    f"Invariant violated: expected {expected}, got {actual}",
    context={"expected": expected, "actual": actual, "location": "validation"},
    recovery_hint="This is a library bug. Please file an issue with this error message."
)
```

---

## Special-Purpose Exceptions

### 5. PerformanceWarning

**What**: Operation exceeded performance SLA (warning, not error)

```python
class PerformanceWarning(UserWarning):
    """
    Performance SLA was not met.
    
    Issued when:
    - File loading > 1s per 100MB
    - DataFrame transformation > 500ms per 1M rows
    - Complete workflow > 30 seconds
    
    This is a WARNING, not an exception. Operations complete successfully
    but slower than expected.
    
    Recovery: Check for large files, slow disk I/O, or system resource constraints.
    """
```

**Usage**:

```python
import warnings

if duration > sla_threshold:
    warnings.warn(
        f"Performance SLA violation: {operation} took {duration:.2f}s "
        f"(expected <{sla_threshold:.2f}s)",
        PerformanceWarning,
        stacklevel=2
    )
```

---

## Exception Attributes

All custom exceptions support these attributes:

```python
class FlyRigLoaderError(Exception):
    def __init__(
        self,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        recovery_hint: Optional[str] = None,
        caused_by: Optional[Exception] = None
    ):
        """
        Initialize exception with context.
        
        Args:
            message: Human-readable error description
            context: Dictionary of contextual information for debugging
            recovery_hint: Optional suggestion for fixing the error
            caused_by: Original exception that caused this error (for chaining)
        """
        self.message = message
        self.context = context or {}
        self.recovery_hint = recovery_hint
        self.caused_by = caused_by
        
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format complete error message with context."""
        parts = [self.message]
        
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {ctx_str}")
        
        if self.recovery_hint:
            parts.append(f"Suggestion: {self.recovery_hint}")
        
        if self.caused_by:
            parts.append(f"Caused by: {type(self.caused_by).__name__}: {self.caused_by}")
        
        return "\n".join(parts)
```

---

## Usage Patterns

### Pattern 1: Simple Error

```python
# Minimal - just message
raise ConfigurationError("Missing 'project' section in configuration")
```

### Pattern 2: Error with Context

```python
# Add debugging context
raise DataFormatError(
    f"Missing required column: {col_name}",
    context={
        "file_path": str(path),
        "required_columns": required_cols,
        "available_columns": available_cols
    }
)
```

### Pattern 3: Error with Recovery Hint

```python
# Help user fix the problem
raise ConfigurationError(
    f"Invalid regex pattern: '{pattern}'",
    context={"pattern": pattern, "field": "extraction_patterns"},
    recovery_hint="Check regex syntax - you may be missing an escape character"
)
```

### Pattern 4: Error Chaining

```python
# Chain from original exception
try:
    data = pickle.load(f)
except Exception as e:
    raise DataFormatError(
        f"Failed to load pickle file: {path}",
        context={"file_path": str(path)},
        recovery_hint="File may be corrupted",
        caused_by=e
    ) from e
```

---

## Error Handling Guidelines

### For Library Code

#### DO: Raise Specific Exceptions

```python
# ✅ GOOD: Specific exception with context
if not path.exists():
    raise FileNotFoundError(
        f"Configuration file not found: {path}",
        context={"config_path": str(path)},
        recovery_hint="Check the file path"
    )
```

#### DON'T: Raise Generic Exceptions

```python
# ❌ BAD: Generic exception without context
if not path.exists():
    raise Exception(f"File not found: {path}")
```

#### DO: Fail Fast

```python
# ✅ GOOD: Validate early, fail fast
def load_experiment(config, name):
    if name not in config.experiments:
        raise ConfigurationError(
            f"Experiment '{name}' not found in configuration",
            context={"available_experiments": list(config.experiments.keys())},
            recovery_hint=f"Check experiment name or add '{name}' to config"
        )
    # Continue with validated data
    ...
```

#### DON'T: Silent Fallbacks

```python
# ❌ BAD: Silent fallback hides problems
def load_experiment(config, name):
    if name not in config.experiments:
        logger.warning(f"Experiment {name} not found, using default")
        return {}  # Silent failure
```

### For User Code

#### Catch Specific Exceptions

```python
try:
    data = load_experiment_files(config, "my_experiment")
except ConfigurationError as e:
    # User error - fix config
    print(f"Configuration problem: {e}")
    if e.recovery_hint:
        print(f"Try: {e.recovery_hint}")
    sys.exit(1)
except DataFormatError as e:
    # Data problem - check files
    print(f"Data file problem: {e}")
    print(f"Context: {e.context}")
    sys.exit(1)
except InternalError as e:
    # Library bug - report
    print(f"Internal error (please report): {e}")
    sys.exit(2)
```

#### Catch All FlyRigLoader Errors

```python
try:
    # FlyRigLoader operations
    ...
except FlyRigLoaderError as e:
    # Handle all library errors uniformly
    logger.error(f"FlyRigLoader error: {e}")
    logger.debug(f"Context: {e.context}")
```

---

## Testing Requirements

### Unit Tests

Every error type must have tests for:

1. **Instantiation**: Can create exception with message
2. **Context**: Context dict is stored and formatted
3. **Recovery Hint**: Hint is included in formatted message
4. **Error Chaining**: `caused_by` is preserved

### Integration Tests

Test that errors are raised in appropriate scenarios:

```python
def test_configuration_error_on_missing_section():
    """Test that ConfigurationError is raised for missing section."""
    config = {"project": {}}  # Missing directories
    
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(config)
    
    assert "major_data_directory" in str(exc_info.value)
    assert exc_info.value.recovery_hint is not None
```

---

## Migration from Generic Exceptions

### Current State (v1.x)

```python
# Generic exceptions
raise ValueError("Invalid configuration")
raise RuntimeError("Failed to load file")
raise Exception("Unexpected error")
```

### Target State (v2.0)

```python
# Specific exceptions with context
raise ConfigurationError(
    "Invalid configuration",
    context={...},
    recovery_hint="..."
)

raise DataFormatError(
    "Failed to load file",
    context={...},
    recovery_hint="..."
)

raise InternalError(
    "Unexpected error",
    context={...},
    recovery_hint="This is a bug, please report"
)
```

---

## Related Documentation

- [Exception API Reference](API_REFERENCE.md#exceptions)
- [Error Handling Best Practices](BEST_PRACTICES.md#error-handling)
- [Debugging Guide](DEBUGGING.md)
- [Performance Warnings](PERFORMANCE_SLA.md)
