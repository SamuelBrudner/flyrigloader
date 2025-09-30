# Pydantic V2 Test Fixes

**Status**: In Progress  
**Branch**: `fix/pydantic-v2-test-assertions`  
**Created**: 2025-09-30

---

## Overview

Fix test assertion failures caused by Pydantic v2 migration. Tests expect old error message formats but Pydantic v2 raises `ValidationError` with different messages.

---

## Root Causes

### 1. Pydantic v2 Enum Validation Messages Changed

**Before (Pydantic v1)**:
- Custom validators raised `ValueError` with custom messages
- Message: `"Dimension must be 1, 2, or 3"`

**After (Pydantic v2)**:
- Built-in enum validation raises `ValidationError`
- Message: `"Input should be 1, 2 or 3 [type=enum, input_value=..., input_type=...]"`

### 2. ValueError with recovery_hint Parameter

During recovery hints implementation, `ValueError` calls were given `recovery_hint=` parameter:

```python
raise ValueError(
    "Missing required columns: x, y",
    recovery_hint="Add missing columns to exp_matrix"
)
```

**Problem**: Standard `ValueError` doesn't accept `recovery_hint` parameter!  
**Solution**: Use custom `TransformError` exception class instead.

---

## Fixes Required

### Part 1: Test Assertions (test_column_config.py)

**Files**:
- `tests/flyrigloader/io/test_column_config.py`
- `tests/flyrigloader/io/test_pydantic_features.py`

**Changes**:
1. Change `pytest.raises(ValueError)` →  `pytest.raises(ValidationError)`
2. Update `match=` patterns to match Pydantic v2 format
3. Import `ValidationError` from `pydantic`

**Affected Tests (~7 tests)**:
- `test_dimension_validation_invalid_cases` (8 parametrized cases)
- `test_special_handling_validation_invalid_cases` (5 parametrized cases)
- `test_invalid_dimension_validation`
- `test_special_handlers_validation_warning`

**Example Fix**:
```python
# BEFORE
with pytest.raises(ValueError, match="Dimension must be 1, 2, or 3"):
    ColumnConfig(dimension=2.5, ...)

# AFTER
with pytest.raises(ValidationError, match="Input should be 1, 2 or 3"):
    ColumnConfig(dimension=2.5, ...)
```

### Part 2: TransformError Usage (transformers.py)

**File**: `src/flyrigloader/io/transformers.py`

**Problem**: ~15 `ValueError` calls with `recovery_hint=` parameter

**Solution**:
1. Import `TransformError` from `flyrigloader.exceptions`
2. Replace all `ValueError(..., recovery_hint=...)` → `TransformError(..., recovery_hint=...)`
3. Add appropriate error codes (TRANSFORM_006 for missing columns, etc.)

**Affected Locations** (~15 instances):
- Line 118: Handler validation
- Line 328: exp_matrix None check
- Line 336: exp_matrix type check  
- Line 343: exp_matrix empty check
- Line 368: Missing 't' key
- Line 376: Time data type check
- Line 384: Empty time check
- Line 448: **Missing required columns** (main failure)
- Line 667: Missing signal_disp
- Line 673: Missing 't' for signal_disp
- Line 685: signal_disp dimension check
- Line 702: signal_disp shape mismatch
- Line 742: exp_matrix type in extract_columns
- Line 793: Null array check
- Line 801: Array conversion failure
- Line 824: 1D conversion failure

**Example Fix**:
```python
# BEFORE
raise ValueError(
    f"Missing required columns: {', '.join(missing_columns)}",
    recovery_hint=f"Add missing columns to exp_matrix: {', '.join(missing_columns)}"
)

# AFTER
raise TransformError(
    f"Missing required columns: {', '.join(missing_columns)}",
    error_code="TRANSFORM_006",
    recovery_hint=f"Add missing columns to exp_matrix: {', '.join(missing_columns)}"
)
```

### Part 3: Test Updates for TransformError

**Files**:
- `tests/flyrigloader/io/test_column_config.py`
- `tests/flyrigloader/io/test_pydantic_features.py`
- `tests/flyrigloader/io/test_pickle.py`

**Changes**:
1. Import `TransformError` from `flyrigloader.exceptions`
2. Update `pytest.raises(ValueError)` → `pytest.raises(TransformError)` for transformation errors
3. Keep `ValueError` expectations where appropriate (non-transformation errors)

**Test Categories**:
- Missing columns tests → expect `TransformError`
- Data validation tests → expect `TransformError`
- Dimension mismatch tests → expect `TransformError`
- signal_disp handling tests → expect `TransformError`

---

## Implementation Plan

1. ✅ Create feature branch: `fix/pydantic-v2-test-assertions`
2. ⏳ Fix Part 1: Test assertions for Pydantic enum validation
3. ⏳ Fix Part 2: Replace ValueError with TransformError in transformers.py
4. ⏳ Fix Part 3: Update tests to expect TransformError
5. ⏳ Run full test suite
6. ⏳ Commit and create PR

---

## Expected Test Results

**Before Fixes**:
- 141 failures (Pydantic message mismatches + recovery_hint TypeError)
- 582 passing

**After Fixes**:
- 0 new failures
- 723+ passing (all fixed)
- Same core functionality

---

## Error Code Mapping

| Error Type | Error Code | Usage |
|------------|------------|-------|
| Missing columns | TRANSFORM_006 | Required columns not in exp_matrix |
| Type validation | TRANSFORM_002 | exp_matrix wrong type, array conversion |
| Dimension mismatch | TRANSFORM_005 | signal_disp dimensions, 1D conversion |
| Data validation | TRANSFORM_007 | None values, empty arrays, invalid format |
| Handler errors | TRANSFORM_008 | Handler registration/execution |

---

## Notes

- All fixes maintain backward compatibility
- Error messages stay descriptive and actionable  
- Recovery hints preserved with proper exception classes
- No changes to actual business logic
- Only test assertions and exception types changed

---

## Testing Strategy

1. Run Pydantic validation tests first
2. Run transformation error tests second  
3. Run full test suite last
4. Verify no regressions in passing tests

---

**Status**: Ready to implement Part 1, then Part 2, then Part 3
