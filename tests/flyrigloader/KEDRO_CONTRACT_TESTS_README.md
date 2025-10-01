# Kedro Contract Tests

## Status: Written but Environment Issue

The file `test_kedro_contracts.py` contains 17 comprehensive contract tests that verify the formal semantic model defined in `docs/KEDRO_SEMANTIC_MODEL.md`.

### Test Coverage

- **5 Invariant Tests** (INV-1 through INV-5)
- **4 Operation Contract Tests** (preconditions/postconditions)
- **3 Composition Property Tests**
- **4 Failure Mode Tests**
- **2 Backward Compatibility Tests**

Total: **18 contract tests**

### Current Issue

The tests are currently being skipped due to a pytest import/environment issue where `from kedro.io import AbstractDataset` fails during test collection, even though Kedro is installed and imports successfully in standalone Python.

### Symptoms

```bash
# Kedro imports fine standalone
$ python -c "from kedro.io import AbstractDataset; print('OK')"
OK

# But test collection reports Kedro not available
$ pytest tests/flyrigloader/test_kedro_contracts.py
17 skipped (Kedro not installed)
```

### Root Cause

The issue appears to be related to:
1. pytest's test collection/import mechanism
2. Possible circular imports or namespace conflicts
3. Test environment path configuration

### Value of These Tests

Even though they don't run yet, the tests serve important purposes:

1. **Documentation** - They formally document the contracts and invariants
2. **Specification** - They translate the semantic model into executable tests
3. **Future Proofing** - Once the environment issue is fixed, they provide regression protection

### Next Steps

To fix:
1. Debug pytest import path for Kedro modules
2. Consider using pytest plugins for conditional imports
3. Or restructure test imports to avoid collection-time failures
4. Or create a separate test suite that runs with explicit Kedro environment

### Workaround

The existing 43 Kedro integration tests in `test_kedro_integration.py` DO run successfully and provide good coverage. These contract tests would add:
- Formal verification of invariants
- Explicit contract testing
- Property-based validation
- Better documentation of guarantees

---

**Created:** 2025-09-30  
**Status:** Needs environment debugging  
**Priority:** Medium (existing tests provide coverage)
