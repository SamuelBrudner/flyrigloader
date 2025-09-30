# Error Recovery Hints - Implementation Examples

**Purpose**: Guide for adding recovery hints to all exception raises  
**Status**: Implementation in progress

---

## Pattern Examples

### Example 1: Configuration Errors

**Before**:
```python
raise ConfigError(
    "Missing required field 'major_data_directory'",
    error_code="CONFIG_001"
)
```

**After**:
```python
raise ConfigError(
    "Missing required field 'major_data_directory'",
    error_code="CONFIG_001",
    recovery_hint="Add 'major_data_directory' under 'project.directories' in your config.yaml"
)
```

---

### Example 2: File Not Found Errors

**Before**:
```python
raise FileNotFoundError(f"Path not found: {path_str}")
```

**After**:
```python
raise FileNotFoundError(
    f"Path not found: {path_str}",
    recovery_hint=f"Check that the path exists and you have read permissions. Expected path: {path_str}"
)
```

---

### Example 3: Validation Errors

**Before**:
```python
raise ValueError(f"Invalid regex pattern '{pattern}'")
```

**After**:
```python
raise ValueError(
    f"Invalid regex pattern '{pattern}'",
    recovery_hint="Check regex syntax - common issues: unescaped special characters (. * + ? [ ] ( ) { } | \\), unclosed groups"
)
```

---

### Example 4: Security Errors

**Before**:
```python
raise ValueError("Path traversal is not allowed")
```

**After**:
```python
raise ValueError(
    f"Path traversal detected in: {path_str}",
    recovery_hint="Remove '..' and '~' from paths. Use absolute paths or paths relative to major_data_directory only"
)
```

---

### Example 5: Type Errors

**Before**:
```python
raise TypeError(f"Expected string, got {type(value)}")
```

**After**:
```python
raise TypeError(
    f"Expected string for field '{field_name}', got {type(value).__name__}",
    recovery_hint=f"Convert value to string or check configuration schema for correct type"
)
```

---

## Recovery Hint Guidelines

### DO ✅

1. **Be Specific**
   ```python
   recovery_hint="Add 'rig' field to dataset 'baseline_behavior' configuration"
   ```

2. **Provide Examples**
   ```python
   recovery_hint="Use date format YYYY-MM-DD (e.g., '2024-01-15') or YYYY_MM_DD (e.g., '2024_01_15')"
   ```

3. **Suggest Fixes**
   ```python
   recovery_hint="Escape special regex characters: use r'\\.' for literal dot, r'\\*' for literal asterisk"
   ```

4. **Point to Documentation**
   ```python
   recovery_hint="See PATTERN_PRECEDENCE.md for pattern matching rules"
   ```

### DON'T ❌

1. **Be Vague**
   ```python
   recovery_hint="Fix the configuration"  # Too vague
   ```

2. **Blame the User**
   ```python
   recovery_hint="You made a mistake"  # Not helpful
   ```

3. **Duplicate Error Message**
   ```python
   # Error message already says "field missing"
   recovery_hint="The field is missing"  # Redundant
   ```

---

## Priority Files for Update

### High Priority (User-Facing Errors)

1. **src/flyrigloader/config/validators.py**
   - Path validation errors
   - Pattern validation errors
   - Date parsing errors
   - Version validation errors

2. **src/flyrigloader/config/yaml_config.py**
   - YAML parsing errors
   - Configuration loading errors
   - Section missing errors

3. **src/flyrigloader/config/models.py**
   - Pydantic validation errors
   - Field validation errors

### Medium Priority (Data Errors)

4. **src/flyrigloader/io/pickle.py**
   - File loading errors
   - Format detection errors
   - Corruption errors

5. **src/flyrigloader/io/transformers.py**
   - DataFrame transformation errors
   - Column mapping errors
   - Dimension mismatch errors

### Lower Priority (Discovery Errors)

6. **src/flyrigloader/discovery/files.py**
   - File discovery errors
   - Permission errors

7. **src/flyrigloader/discovery/patterns.py**
   - Pattern matching errors

---

## Implementation Checklist

For each error raise:

- [ ] Add `recovery_hint` parameter
- [ ] Provide specific, actionable guidance
- [ ] Include examples where helpful
- [ ] Test the error message readability
- [ ] Update related documentation if needed

---

## Testing Recovery Hints

### Manual Test Pattern

```python
def test_error_recovery_hint():
    """Test that error includes helpful recovery hint."""
    with pytest.raises(ConfigError) as exc_info:
        validate_config(invalid_config)
    
    error = exc_info.value
    assert error.recovery_hint is not None
    assert "Add" in error.recovery_hint or "Check" in error.recovery_hint
    assert len(error.recovery_hint) > 20  # Substantial guidance
```

---

## Progress Tracking

| Module | Total Errors | With Hints | Progress |
|--------|--------------|------------|----------|
| config/validators.py | ~30 | 0 | 0% |
| config/yaml_config.py | ~20 | 0 | 0% |
| config/models.py | ~15 | 0 | 0% |
| io/pickle.py | ~10 | 0 | 0% |
| io/transformers.py | ~8 | 0 | 0% |
| discovery/files.py | ~12 | 0 | 0% |
| discovery/patterns.py | ~5 | 0 | 0% |
| **TOTAL** | **~100** | **0** | **0%** |

---

**Next Steps**:
1. Start with config/validators.py (highest user impact)
2. Update 5-10 errors per session
3. Test each update
4. Document patterns learned
5. Create PR when module complete
