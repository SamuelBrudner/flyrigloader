# Performance SLA Monitoring

**Version**: 2.0.0-dev  
**Last Updated**: 2025-09-30

## Overview

This document defines **Performance Service Level Agreements (SLAs)** for FlyRigLoader operations and how they are monitored and reported. Following Priority 3.2 of the semantic model review, we establish clear performance contracts and warning mechanisms.

---

## Design Philosophy

### Non-Blocking Performance Monitoring

Performance violations **warn but don't fail**:

```python
# ✅ Operation completes successfully, logs warning
if duration > sla_threshold:
    warnings.warn(
        f"Performance SLA violation: operation took {duration:.2f}s",
        PerformanceWarning
    )
    # Operation continues and returns result
return result
```

**Rationale**: Users care more about **correctness** than speed. Performance warnings help identify bottlenecks without breaking workflows.

---

## SLA Definitions

### 1. Data Loading SLA

**Requirement**: ≤ 1 second per 100MB

```python
# Expected performance
file_size_mb = 150
max_load_time = file_size_mb / 100  # 1.5 seconds
```

**Monitoring**:

```python
def read_pickle_any_format(file_path: Path) -> Dict[str, Any]:
    """Load pickle file with performance monitoring."""
    start_time = time.time()
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    
    # Load data
    data = _load_pickle(file_path)
    
    # Check SLA
    duration = time.time() - start_time
    max_time = max(0.1, file_size_mb / 100)  # Minimum 100ms
    
    if duration > max_time:
        warnings.warn(
            f"Data loading SLA violation: {file_path.name} "
            f"({file_size_mb:.1f}MB) took {duration:.2f}s "
            f"(expected <{max_time:.2f}s)",
            PerformanceWarning,
            stacklevel=2
        )
    
    return data
```

**Common Causes**:
- Network-mounted filesystems (NFS, SMB)
- Slow disk I/O
- Large gzipped files requiring decompression
- Memory paging/swapping

---

### 2. DataFrame Transformation SLA

**Requirement**: ≤ 500ms per 1M rows

```python
# Expected performance
row_count = 2_500_000
max_transform_time = (row_count / 1_000_000) * 0.5  # 1.25 seconds
```

**Monitoring**:

```python
def make_dataframe_from_config(
    exp_matrix: Dict[str, np.ndarray],
    config: ColumnConfigDict
) -> pd.DataFrame:
    """Create DataFrame with performance monitoring."""
    start_time = time.time()
    
    # Transform data
    df = _transform_to_dataframe(exp_matrix, config)
    
    # Check SLA
    duration = time.time() - start_time
    row_count = len(df)
    max_time = max(0.05, (row_count / 1_000_000) * 0.5)  # Minimum 50ms
    
    if duration > max_time:
        warnings.warn(
            f"DataFrame transformation SLA violation: {row_count:,} rows "
            f"took {duration:.3f}s (expected <{max_time:.3f}s)",
            PerformanceWarning,
            stacklevel=2
        )
    
    return df
```

**Common Causes**:
- Complex column transformations
- Large 2D arrays requiring special handling
- Memory allocation overhead
- Metadata integration overhead

---

### 3. Complete Workflow SLA

**Requirement**: ≤ 30 seconds end-to-end

**Scope**: Full workflow from config loading to DataFrame output

```python
Config Load → File Discovery → Data Loading → Transformation → DataFrame
```

**Monitoring**:

```python
def load_experiment_files(
    config: ProjectConfig,
    experiment_name: str,
    **kwargs
) -> Union[List[str], Dict[str, Dict[str, Any]]]:
    """Discover experiment files with workflow performance monitoring."""
    workflow_start = time.time()
    
    # Execute workflow
    files = _discover_and_load(config, experiment_name, **kwargs)
    
    # Check workflow SLA
    workflow_duration = time.time() - workflow_start
    
    if workflow_duration > 30.0:
        warnings.warn(
            f"Workflow SLA violation: complete operation took {workflow_duration:.1f}s "
            f"(expected <30.0s). Processed {len(files)} files.",
            PerformanceWarning,
            stacklevel=2
        )
    
    return files
```

**Common Causes**:
- Large number of files
- Slow file discovery (many directories)
- Complex metadata extraction
- Network latency
- System resource constraints

---

## PerformanceWarning Class

```python
class PerformanceWarning(UserWarning):
    """
    Warning issued when operations exceed performance SLAs.
    
    This warning indicates that an operation completed successfully
    but took longer than expected. It does not indicate a failure.
    
    Users can:
    - Ignore warnings (operations are still correct)
    - Log warnings for monitoring
    - Filter warnings in production
    - Investigate bottlenecks for optimization
    
    Example:
        >>> import warnings
        >>> warnings.simplefilter('always', PerformanceWarning)
        >>> # Now all performance warnings will be shown
    """
    pass
```

---

## Warning Control

### Filter All Performance Warnings

```python
import warnings
from flyrigloader.exceptions import PerformanceWarning

# Ignore all performance warnings
warnings.filterwarnings('ignore', category=PerformanceWarning)

# Operations complete without warnings
data = load_experiment_files(...)
```

### Log Performance Warnings

```python
import warnings
import logging

# Configure warning logging
logging.captureWarnings(True)
warnings_logger = logging.getLogger('py.warnings')
warnings_logger.setLevel(logging.INFO)

# Performance warnings logged to file
data = load_experiment_files(...)
```

### Treat Warnings as Errors (Testing)

```python
import warnings

# Convert warnings to errors for testing
warnings.simplefilter('error', PerformanceWarning)

# Now performance violations raise exceptions
try:
    data = load_experiment_files(...)
except PerformanceWarning as e:
    print(f"Performance SLA violated: {e}")
```

### Custom Warning Handler

```python
import warnings

def performance_monitor(message, category, filename, lineno, file=None, line=None):
    """Custom handler for performance warnings."""
    if category == PerformanceWarning:
        # Log to monitoring system
        metrics.record_sla_violation(message)
        print(f"⚠️  Performance: {message}")

warnings.showwarning = performance_monitor
```

---

## Performance Metrics Collection

### Basic Metrics

```python
from dataclasses import dataclass, field
from typing import List
import time

@dataclass
class PerformanceMetrics:
    """Track performance metrics for a session."""
    operation_times: List[float] = field(default_factory=list)
    sla_violations: int = 0
    total_operations: int = 0
    
    def record_operation(self, duration: float, sla_threshold: float):
        """Record operation and check SLA."""
        self.operation_times.append(duration)
        self.total_operations += 1
        
        if duration > sla_threshold:
            self.sla_violations += 1
    
    @property
    def violation_rate(self) -> float:
        """Calculate percentage of SLA violations."""
        if self.total_operations == 0:
            return 0.0
        return (self.sla_violations / self.total_operations) * 100
    
    @property
    def average_time(self) -> float:
        """Calculate average operation time."""
        if not self.operation_times:
            return 0.0
        return sum(self.operation_times) / len(self.operation_times)
```

### Usage Example

```python
metrics = PerformanceMetrics()

for file_path in files:
    start = time.time()
    data = read_pickle_any_format(file_path)
    duration = time.time() - start
    
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    sla = file_size_mb / 100
    metrics.record_operation(duration, sla)

print(f"SLA Violation Rate: {metrics.violation_rate:.1f}%")
print(f"Average Load Time: {metrics.average_time:.3f}s")
```

---

## SLA Testing

### Unit Test Pattern

```python
def test_data_loading_sla(tmp_path, monkeypatch):
    """Test that data loading meets SLA requirements."""
    # Create 100MB test file
    large_file = tmp_path / "large_data.pkl"
    test_data = {"data": np.random.rand(1000, 1000)}
    with open(large_file, 'wb') as f:
        pickle.dump(test_data, f)
    
    # Measure load time
    start = time.time()
    data = read_pickle_any_format(large_file)
    duration = time.time() - start
    
    # Check SLA (1s per 100MB)
    file_size_mb = large_file.stat().st_size / (1024 * 1024)
    max_time = file_size_mb / 100
    
    assert duration <= max_time, (
        f"SLA violation: {duration:.2f}s for {file_size_mb:.1f}MB "
        f"(expected <{max_time:.2f}s)"
    )
```

### Performance Benchmark Test

```python
@pytest.mark.benchmark
def test_transformation_performance_benchmark(benchmark, sample_data):
    """Benchmark DataFrame transformation performance."""
    exp_matrix = sample_data  # 1M rows
    config = get_default_column_config()
    
    # Run benchmark
    result = benchmark(make_dataframe_from_config, exp_matrix, config)
    
    # Verify SLA
    assert benchmark.stats['mean'] < 0.5, "Transformation should be <500ms for 1M rows"
```

---

## Optimization Strategies

### When SLA Violations Occur

#### 1. Profile the Operation

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Operation being profiled
data = load_experiment_files(...)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 slowest functions
```

#### 2. Check System Resources

```python
import psutil

# Memory usage
process = psutil.Process()
print(f"Memory: {process.memory_info().rss / 1024 / 1024:.1f}MB")

# CPU usage
print(f"CPU: {psutil.cpu_percent(interval=1)}%")

# Disk I/O
io_counters = psutil.disk_io_counters()
print(f"Read: {io_counters.read_bytes / 1024 / 1024:.1f}MB")
```

#### 3. Use Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def load_cached_data(file_path: str):
    """Load data with caching for frequently accessed files."""
    return read_pickle_any_format(Path(file_path))
```

#### 4. Parallelize Operations

```python
from concurrent.futures import ThreadPoolExecutor

def load_files_parallel(file_paths: List[Path], max_workers: int = 4):
    """Load multiple files in parallel."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(read_pickle_any_format, file_paths))
```

---

## Performance Reporting

### Generate Performance Report

```python
def generate_performance_report(metrics: PerformanceMetrics) -> str:
    """Generate human-readable performance report."""
    report = [
        "Performance Report",
        "=" * 50,
        f"Total Operations: {metrics.total_operations}",
        f"SLA Violations: {metrics.sla_violations} ({metrics.violation_rate:.1f}%)",
        f"Average Time: {metrics.average_time:.3f}s",
        f"Min Time: {min(metrics.operation_times):.3f}s",
        f"Max Time: {max(metrics.operation_times):.3f}s",
        ""
    ]
    
    if metrics.sla_violations > 0:
        report.append("⚠️  SLA violations detected. Consider:")
        report.append("  - Profiling slow operations")
        report.append("  - Checking disk I/O performance")
        report.append("  - Using caching for frequently accessed files")
        report.append("  - Parallelizing file loading")
    else:
        report.append("✅ All operations met SLA requirements")
    
    return "\n".join(report)
```

---

## Configuration

### Enable/Disable Monitoring

```python
# In config.yaml
performance:
  monitoring_enabled: true
  sla_warnings_enabled: true
  collect_metrics: true
  log_slow_operations: true
  slow_operation_threshold_factor: 1.5  # Warn at 1.5x SLA
```

### Programmatic Control

```python
from flyrigloader.performance import set_monitoring_enabled

# Disable performance monitoring
set_monitoring_enabled(False)

# Operations run without performance checks
data = load_experiment_files(...)

# Re-enable monitoring
set_monitoring_enabled(True)
```

---

## Related Documentation

- [Error Taxonomy](ERROR_TAXONOMY.md) - Exception handling
- [Testing Guide](TESTING_GUIDE.md) - Performance testing
- [Optimization Guide](OPTIMIZATION.md) - Performance tuning
- [API Reference](API_REFERENCE.md) - Function documentation
