# Implementation Roadmap

**Status**: Phase 2 - Core Implementation  
**Last Updated**: 2025-09-30

---

## Progress Overview

| Phase | Status | Completion |
|-------|--------|------------|
| Phase 1: Documentation | ✅ Complete | 100% |
| Phase 2: Core Implementation | 🔄 In Progress | 10% |
| Phase 3: Migration Support | ⏳ Pending | 0% |
| Phase 4: v2.0 Release | ⏳ Pending | 0% |

---

## Current Sprint: Core Implementation

### Completed ✅

1. **Test Suite Fix**
   - Fixed property-based test duplicate handling
   - All config tests passing (232/257 passing, 25 pre-existing failures)

### In Progress 🔄

2. **Add Recovery Hints to Error Raises** (Priority 3.1)
   - Target: Add `recovery_hint` parameter to all exception raises
   - Files to update: ~50 error raises across codebase
   - Status: 0% complete

### Next Up ⏳

3. **Implement PerformanceWarning Checks** (Priority 3.2)
   - Add SLA monitoring to critical paths
   - Status: Not started

4. **Integrate DiscoveryOptions** (Priority 2.2)
   - Update API function signatures
   - Status: Design complete, implementation pending

---

## Detailed Tasks

### Task 1: Add Recovery Hints to All Errors

**Goal**: Enhance every exception raise with recovery hints

**Approach**:
1. Find all `raise` statements
2. Add `recovery_hint` parameter
3. Test error messages

**Files to Update** (Priority Order):
- [ ] `src/flyrigloader/config/yaml_config.py` (config errors)
- [ ] `src/flyrigloader/config/models.py` (validation errors)  
- [ ] `src/flyrigloader/discovery/files.py` (discovery errors)
- [ ] `src/flyrigloader/io/pickle.py` (load errors)
- [ ] `src/flyrigloader/io/transformers.py` (transform errors)

**Example Pattern**:
```python
# BEFORE:
raise ConfigError(
    f"Missing required field: {field}",
    error_code="CONFIG_001"
)

# AFTER:
raise ConfigError(
    f"Missing required field: {field}",
    error_code="CONFIG_001",
    recovery_hint=f"Add '{field}' to your configuration file under the project section"
)
```

---

### Task 2: Implement Performance Monitoring

**Goal**: Add PerformanceWarning to critical operations

**Target Operations**:
- [ ] Data loading (`read_pickle_any_format`)
- [ ] DataFrame transformation (`make_dataframe_from_config`)
- [ ] Complete workflows (`load_experiment_files`, `load_dataset_files`)

**Implementation Pattern**:
```python
import warnings
import time

def read_pickle_any_format(file_path: Path):
    start = time.time()
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    
    data = _load_pickle(file_path)
    
    duration = time.time() - start
    sla = max(0.1, file_size_mb / 100)  # 1s per 100MB
    
    if duration > sla:
        warnings.warn(
            f"Performance SLA violation: {file_path.name} took {duration:.2f}s "
            f"(expected <{sla:.2f}s for {file_size_mb:.1f}MB)",
            PerformanceWarning,
            stacklevel=2
        )
    
    return data
```

---

### Task 3: Integrate DiscoveryOptions

**Goal**: Simplify API signatures using DiscoveryOptions dataclass

**Files to Update**:
- [ ] `src/flyrigloader/api.py` - Update function signatures
- [ ] `src/flyrigloader/config/discovery.py` - Use DiscoveryOptions
- [ ] `src/flyrigloader/discovery/files.py` - Accept DiscoveryOptions
- [ ] Tests - Update to use DiscoveryOptions

**Migration Strategy**:
1. Keep old signatures with deprecation warnings (v1.x)
2. Add new signatures accepting DiscoveryOptions
3. Internal functions use DiscoveryOptions only

---

## Success Metrics

### For Each Task

- [ ] All tests passing
- [ ] No performance regression
- [ ] Documentation updated
- [ ] Examples added

### Overall Phase 2

- [ ] 90%+ test coverage maintained
- [ ] All error messages have recovery hints
- [ ] Performance monitoring active
- [ ] API simplified (parameter reduction)

---

## Timeline

| Week | Focus | Deliverables |
|------|-------|--------------|
| Week 1 (Current) | Recovery hints | Config + Discovery modules |
| Week 2 | Recovery hints | IO + Transformation modules |
| Week 3 | Performance monitoring | All critical paths |
| Week 4 | DiscoveryOptions integration | API updates + tests |

---

## Notes

- Maintain backward compatibility in v1.x
- Add deprecation warnings where appropriate
- Update documentation as we go
- Create examples for new patterns

---

**Next Action**: Start with Task 1 - Add recovery hints to config module
