# Testing Guidelines for flyrigloader

**Version:** 1.0  
**Last Updated:** 2024-12-17  
**Status:** Authoritative Reference

## Table of Contents

1. [Overview](#overview)
2. [Core Testing Principles](#core-testing-principles)
3. [Centralized Fixture Management](#centralized-fixture-management)
4. [AAA Pattern Implementation](#aaa-pattern-implementation)
5. [Naming Conventions](#naming-conventions)
6. [Edge-Case Testing Standards](#edge-case-testing-standards)
7. [Performance Test Isolation](#performance-test-isolation)
8. [Network Test Management](#network-test-management)
9. [Parallel Execution Configuration](#parallel-execution-configuration)
10. [Automated Style Enforcement](#automated-style-enforcement)
11. [Pull Request Requirements](#pull-request-requirements)
12. [Coverage Requirements](#coverage-requirements)
13. [Examples and Best Practices](#examples-and-best-practices)
14. [Troubleshooting Guide](#troubleshooting-guide)

---

## Overview

This document serves as the **definitive reference** for all testing practices, conventions, and standards within the flyrigloader project. It implements the comprehensive testing strategy outlined in Section 6.6 of the technical specification, ensuring consistent application of best practices across development team contributions and external research community engagement.

### Purpose and Scope

The flyrigloader testing framework is specifically designed for **behavior-focused validation through black-box testing approaches** that emphasize public API contracts and observable system behavior rather than implementation-specific details. This approach addresses the unique requirements of neuroscience research data processing workflows, including functional correctness validation, developer experience optimization, and cross-platform behavioral consistency.

### Key Testing Dimensions

| Testing Dimension | Primary Focus | Tools & Frameworks | Coverage Target |
|---|---|---|---|
| **Unit Testing** | Public API behavior validation and isolated function testing | pytest, pytest-mock, hypothesis | 100% for critical modules |
| **Integration Testing** | Cross-module interaction and workflow validation | pytest, pytest-benchmark | 95% coverage |
| **Performance Testing (Opt-in)** | SLA compliance and benchmark validation (scripts/benchmarks/) | pytest-benchmark, custom metrics | 100% SLA compliance via manual workflow |
| **Property-Based Testing** | Edge case discovery and data validation | hypothesis, custom generators | Comprehensive fuzzing |
| **Developer Experience** | Rapid feedback and development workflow optimization | pytest-xdist, parallel execution | <30s default test suite execution |

---

## Core Testing Principles

### 1. Behavior-Focused Testing

All test cases must interact with the system through **documented public interfaces** rather than internal implementation details. This approach provides robust validation while maintaining flexibility for internal architectural changes and performance optimizations.

**✅ Correct Approach:**
```python
def test_config_loading_success(sample_config_file):
    """Test configuration loading through public API."""
    # ARRANGE
    config_path = sample_config_file
    
    # ACT  
    config = load_config(config_path)
    
    # ASSERT
    assert config is not None
    assert "project" in config
    assert config["project"]["name"] is not None
```

**❌ Incorrect Approach:**
```python
def test_config_loading_internal():
    """Avoid testing internal implementation details."""
    # DON'T TEST PRIVATE ATTRIBUTES
    loader = ConfigLoader()
    loader._load_yaml_file()  # Accessing private method
    assert loader._config is not None  # Accessing private attribute
```

### 2. Test Isolation and Reproducibility

Each test must run in complete isolation with no shared state dependencies. Tests must be deterministic and produce consistent results across different environments and execution orders.

### 3. Rapid Developer Feedback

The default test suite must complete execution in **under 30 seconds** to maintain productive development workflows. This is achieved through:

- **Selective test execution**: Excluding performance, benchmark, network, and slow tests by default
- **Parallel execution**: Using pytest-xdist with 4-worker configuration
- **Efficient fixture management**: Centralized fixtures to eliminate setup duplication

---

## Centralized Fixture Management

### Overview

All test fixtures are centralized in `tests/conftest.py` and shared utilities in `tests/utils.py` to eliminate code duplication across test modules while maintaining consistent testing patterns throughout unit, integration, and performance test layers.

### Fixture Organization Structure

```
tests/
├── conftest.py           # Global fixtures and pytest configuration
├── utils.py              # Shared protocol-based mock implementations
├── coverage/
│   └── pytest.ini       # Comprehensive pytest configuration
└── flyrigloader/
    ├── test_api.py       # API layer tests using centralized fixtures
    ├── test_config/      # Configuration tests
    ├── test_discovery/   # Discovery engine tests
    ├── test_io/          # I/O module tests
    └── integration/      # Integration test suites
```

### Fixture Naming Conventions

All fixtures must follow standardized naming patterns to ensure consistency and clarity:

- **`mock_*`**: Mock objects and simulated components
- **`temp_*`**: Temporary resources requiring cleanup
- **`sample_*`**: Test data and synthetic datasets
- **`fixture_*`**: Complex setup scenarios and configurations

### Core Centralized Fixtures

#### 1. Configuration Fixtures

```python
# From tests/conftest.py
@pytest.fixture(scope="module")
def sample_comprehensive_config_dict():
    """Comprehensive sample configuration for all testing scenarios."""
    return {
        "project": {
            "directories": {
                "major_data_directory": "/research/data/neuroscience",
                "batchfile_directory": "/research/batch_definitions"
            },
            "ignore_substrings": ["backup", "temp", ".DS_Store"],
            "mandatory_substrings": ["experiment_", "data_"],
            "file_extensions": [".csv", ".pkl", ".pickle", ".json"]
        },
        "rigs": {
            "old_opto": {
                "sampling_frequency": 60,
                "mm_per_px": 0.154,
                "arena_diameter_mm": 120
            }
        },
        "datasets": {
            "baseline_behavior": {
                "rig": "old_opto",
                "patterns": ["*baseline*", "*control*"],
                "dates_vials": {
                    "20241220": [1, 2, 3, 4, 5],
                    "20241221": [1, 2, 3]
                }
            }
        }
    }
```

#### 2. Filesystem Fixtures

```python
@pytest.fixture(scope="function")
def temp_cross_platform_dir():
    """Cross-platform temporary directory with proper cleanup."""
    # Handles platform-specific considerations for Windows, Linux, and macOS
    # Ensures proper permissions and Unicode path support
    # Provides comprehensive cleanup even on test failures
```

#### 3. Synthetic Data Fixtures

```python
@pytest.fixture(scope="session")
def test_data_generator():
    """Session-scoped synthetic experimental data generation."""
    # Realistic experimental data patterns (neural recordings, behavioral data)
    # Configurable data sizes and complexity levels
    # Memory-efficient data generation strategies
```

### Shared Mock Implementations

The `tests/utils.py` module provides **protocol-based mock implementations** that are reusable across all test categories:

```python
from tests.utils import (
    create_mock_filesystem,
    create_mock_dataloader,
    create_mock_config_provider,
    generate_edge_case_scenarios,
    validate_test_structure
)

def test_integration_workflow(create_mock_filesystem, create_mock_dataloader):
    """Example using shared mock implementations."""
    # ARRANGE
    filesystem = create_mock_filesystem(
        structure={'files': {'/test/data.pkl': {'size': 1024}}},
        unicode_files=True,
        corrupted_files=False
    )
    dataloader = create_mock_dataloader(['basic', 'network'])
    
    # ACT & ASSERT
    # Test implementation using consistent mock behaviors
```

---

## AAA Pattern Implementation

### Pattern Overview

The **Arrange-Act-Assert (AAA)** pattern is mandatory for all test functions. This pattern promotes readability, maintainability, and consistent test structure across the neuroscience data processing validation suite.

### Structure Requirements

Every test function must clearly separate the three phases:

1. **ARRANGE**: Set up test data and dependencies
2. **ACT**: Execute the function under test
3. **ASSERT**: Verify the results and side effects

### AAA Pattern Template

```python
def test_{module}_{function}_{scenario}():
    """
    Descriptive test documentation.
    
    Validates: [Specific behavior being tested]
    Edge cases: [Any edge cases covered]
    Dependencies: [Mock objects or fixtures used]
    """
    # ARRANGE - Set up test data and dependencies
    test_input = setup_test_input()
    expected_output = define_expected_output()
    mock_dependency = create_mock_dependency()
    
    # ACT - Execute the function under test
    actual_output = function_under_test(test_input)
    
    # ASSERT - Verify the results
    assert actual_output == expected_output
    assert additional_conditions_met(actual_output)
    
    # Optional: Verify mock interactions
    mock_dependency.assert_called_once_with(expected_parameters)
```

### Real-World Examples

#### Unit Test Example

```python
def test_config_yaml_parsing_success(sample_config_file):
    """
    Test YAML configuration parsing with valid input.
    
    Validates: Configuration loading returns expected structure
    Edge cases: None (happy path scenario)
    Dependencies: sample_config_file fixture
    """
    # ARRANGE
    config_path = sample_config_file
    expected_keys = ["project", "rigs", "datasets"]
    
    # ACT
    loaded_config = load_config(config_path)
    
    # ASSERT
    assert loaded_config is not None
    assert isinstance(loaded_config, dict)
    for key in expected_keys:
        assert key in loaded_config
```

#### Integration Test Example

```python
def test_integration_file_discovery_to_data_loading(
    temp_filesystem_structure,
    mock_data_loading_comprehensive
):
    """
    Test integration between file discovery and data loading modules.
    
    Validates: Discovered files can be successfully loaded
    Edge cases: Handles missing files gracefully
    Dependencies: Filesystem and data loading mocks
    """
    # ARRANGE
    data_directory = temp_filesystem_structure["data_root"]
    expected_files = [
        temp_filesystem_structure["baseline_file_1"],
        temp_filesystem_structure["baseline_file_2"]
    ]
    
    # Setup mock data loading responses
    for file_path in expected_files:
        mock_data_loading_comprehensive.add_experimental_matrix(
            str(file_path), n_timepoints=1000
        )
    
    # ACT
    discovered_files = discover_experiment_files(data_directory)
    loaded_data = {}
    for file_path in discovered_files:
        loaded_data[file_path] = load_experimental_data(file_path)
    
    # ASSERT
    assert len(discovered_files) == len(expected_files)
    assert len(loaded_data) == len(expected_files)
    for file_path, data in loaded_data.items():
        assert 't' in data  # Time series data
        assert 'x' in data  # Position data
        assert 'y' in data
```

### AAA Pattern Enforcement

The **flake8-pytest-style** plugin automatically validates AAA pattern compliance:

```ini
# .flake8 configuration
[flake8]
select = PT
pytest-fixture-no-parentheses = true
pytest-mark-no-parentheses = true

# AAA pattern validation rules
# PT024: pytest.mark.parametrize is missing values
# PT025: pytest.mark.parametrize has duplicated values
```

---

## Naming Conventions

### Test Function Naming

All test functions must follow the standardized pattern:

**Pattern**: `test_{module}_{function}_{scenario}`

**Examples**:
- `test_config_yaml_loading_success`
- `test_config_yaml_loading_malformed_file`
- `test_discovery_files_pattern_matching_unicode_paths`
- `test_io_pickle_loading_corrupted_file_error`

### Category-Specific Naming Patterns

#### Unit Tests
- `test_{module}_{function}_{scenario}`
- Example: `test_config_yaml_parsing_success`

#### Integration Tests  
- `test_integration_{workflow}_{scenario}`
- Example: `test_integration_discovery_to_loading_workflow`

#### Performance Tests (scripts/benchmarks/)
- `test_benchmark_{operation}_{constraint}`
- Example: `test_benchmark_data_loading_large_files`

#### Network Tests
- `test_network_{service}_{scenario}`
- Example: `test_network_config_remote_loading_success`

#### Edge-Case Tests
- `test_edge_{category}_{specific_condition}`
- Example: `test_edge_unicode_path_special_characters`

### Fixture Naming Standards

```python
# Mock objects and simulated components
@pytest.fixture
def mock_filesystem_provider():
    """Mock filesystem operations for testing."""

# Temporary resources requiring cleanup  
@pytest.fixture
def temp_experiment_directory():
    """Temporary directory for experiment files."""

# Test data and synthetic datasets
@pytest.fixture  
def sample_neuroscience_timeseries():
    """Sample time series data for testing."""

# Complex setup scenarios
@pytest.fixture
def fixture_integration_test_environment():
    """Complete integration test environment setup."""
```

### Pytest Marker Usage

All tests must be properly categorized using standardized markers:

```python
# Unit test with specific domain marker
@pytest.mark.unit
@pytest.mark.data_loading
def test_pickle_file_loading_success():
    """Unit test for pickle file loading."""

# Integration test with multiple markers
@pytest.mark.integration
@pytest.mark.configuration
@pytest.mark.file_discovery
def test_integration_config_driven_discovery():
    """Integration test for configuration-driven file discovery."""

# Performance test (excluded by default)
@pytest.mark.performance
@pytest.mark.benchmark
def test_benchmark_large_dataset_processing():
    """Performance benchmark for large dataset processing."""

# Network test (excluded by default)
@pytest.mark.network
def test_network_remote_config_loading():
    """Network-dependent configuration loading test."""

# Edge case test
@pytest.mark.edge_case
@pytest.mark.error_handling
def test_edge_corrupted_file_handling():
    """Edge case test for corrupted file handling."""
```

---

## Edge-Case Testing Standards

### Overview

Edge-case testing is critical for neuroscience research workflow validation, ensuring robust handling of boundary conditions, corrupted data scenarios, and platform-specific considerations that researchers may encounter in real-world experimental environments.

### Categories of Edge Cases

#### 1. Unicode Path Handling

Neuroscience researchers often work with international collaborations involving Unicode characters in file paths and names. All file handling code must be tested against Unicode scenarios.

```python
@pytest.mark.edge_case
@pytest.mark.platform_specific
def test_edge_unicode_file_paths_cross_platform(fixture_unicode_path_generator):
    """
    Test Unicode file path handling across platforms.
    
    Validates: File discovery works with Unicode characters
    Edge cases: Various Unicode encodings and special characters
    Platform considerations: Windows path length limitations
    """
    # ARRANGE
    unicode_paths = fixture_unicode_path_generator(num_paths=5)
    test_directory = create_unicode_test_structure(unicode_paths)
    
    # ACT
    discovered_files = discover_files_in_directory(test_directory)
    
    # ASSERT
    assert len(discovered_files) > 0
    for file_path in discovered_files:
        assert file_path.exists()
        # Verify Unicode characters are preserved
        assert any(ord(char) > 127 for char in str(file_path))
```

#### 2. Corrupted File Scenarios

Research data can become corrupted due to hardware failures, network interruptions, or storage issues. The system must handle these gracefully.

```python
@pytest.mark.edge_case
@pytest.mark.error_handling
def test_edge_corrupted_pickle_file_handling(fixture_corrupted_file_scenarios):
    """
    Test handling of corrupted pickle files.
    
    Validates: Graceful error handling for corrupted data
    Edge cases: Various corruption patterns and recovery scenarios
    Research context: Hardware failures during long experiments
    """
    # ARRANGE
    corrupted_file = fixture_corrupted_file_scenarios["corrupted_pickle"]
    expected_error_type = pickle.UnpicklingError
    
    # ACT & ASSERT
    with pytest.raises(expected_error_type) as exc_info:
        load_pickle_file(corrupted_file)
    
    # Verify error message is informative for researchers
    assert "corrupted" in str(exc_info.value).lower()
    assert str(corrupted_file) in str(exc_info.value)
```

#### 3. Boundary Condition Testing

Data processing must handle edge cases in data dimensions, file sizes, and processing parameters.

```python
@pytest.mark.edge_case
@pytest.mark.parametrize("array_size", [0, 1, 2, 1000000])
def test_edge_boundary_array_sizes(array_size, fixture_boundary_condition_data):
    """
    Test data processing with boundary condition array sizes.
    
    Validates: System handles various array sizes correctly
    Edge cases: Empty arrays, single elements, very large arrays
    Research context: Experiments with varying data collection periods
    """
    # ARRANGE
    test_data = fixture_boundary_condition_data["minimal_time_series"]
    if array_size > 0:
        test_data = generate_synthetic_array(size=array_size)
    
    # ACT
    if array_size == 0:
        # Empty array should raise appropriate error
        with pytest.raises(ValueError, match="empty.*array"):
            result = process_time_series_data(test_data)
    else:
        result = process_time_series_data(test_data)
        
        # ASSERT
        assert result is not None
        if array_size == 1:
            assert len(result) == 1
        elif array_size > 1:
            assert len(result) == array_size
```

#### 4. Memory Constraint Scenarios

Large neuroscience datasets can strain system memory. Tests must validate behavior under memory pressure.

```python
@pytest.mark.edge_case
@pytest.mark.memory
def test_edge_memory_pressure_large_dataset(fixture_memory_constraint_scenarios):
    """
    Test system behavior under memory pressure.
    
    Validates: Graceful handling of large datasets
    Edge cases: Memory allocation failures and cleanup
    Research context: Processing multi-hour experimental recordings
    """
    # ARRANGE
    memory_scenario = fixture_memory_constraint_scenarios['large_dataset_generator']
    large_dataset = memory_scenario(size_mb=100)  # 100MB test dataset
    
    # ACT
    with monitor_memory_usage() as memory_monitor:
        result = process_large_dataset(large_dataset)
    
    # ASSERT
    assert result is not None
    # Verify memory usage stayed within reasonable bounds
    assert memory_monitor.peak_usage_mb < 500  # Allow for processing overhead
    # Verify memory was properly cleaned up
    assert memory_monitor.final_usage_mb < memory_monitor.initial_usage_mb + 50
```

### Edge-Case Testing Utilities

The `tests/utils.py` module provides comprehensive edge-case scenario generators:

```python
from tests.utils import generate_edge_case_scenarios

# Generate comprehensive edge-case test scenarios
edge_cases = generate_edge_case_scenarios(
    scenario_types=['unicode', 'boundary', 'corrupted', 'memory', 'concurrent'],
    include_platform_specific=True
)

# Use in tests
@pytest.mark.parametrize("scenario", edge_cases['unicode'])
def test_unicode_edge_cases(scenario):
    """Test various Unicode edge cases."""
    # Test implementation using generated scenarios
```

---

## Performance Test Isolation

### Overview

Performance benchmarks and resource-intensive tests are **isolated in the `scripts/benchmarks/` directory** and excluded from default test execution to maintain rapid developer feedback cycles while providing comprehensive performance validation when required.

### Performance Test Architecture

```
scripts/
└── benchmarks/
    ├── run_benchmarks.py           # CLI runner for manual execution
    ├── data_loading_benchmarks.py  # Data loading SLA validation
    ├── memory_profiling.py         # Memory usage analysis
    ├── concurrent_access_tests.py  # Multi-process validation
    └── performance_regression.py   # Historical comparison
```

### Default Test Execution Configuration

Performance tests are automatically excluded through pytest marker configuration:

```python
# pyproject.toml configuration
[tool.pytest.ini_options]
addopts = [
    # Selective test execution for rapid developer feedback
    "-m", "not performance and not benchmark and not network and not slow",
    # Other configuration...
]

markers = [
    "performance: marks tests as performance-related (excluded by default)",
    "benchmark: marks tests as performance benchmarks (excluded by default)",
    "slow: marks tests as slow - execution time >30 seconds (excluded by default)",
]
```

### Benchmark Execution Methods

#### 1. CLI Runner Execution

```bash
# Execute all performance benchmarks
python scripts/benchmarks/run_benchmarks.py

# Execute specific benchmark categories
python scripts/benchmarks/run_benchmarks.py --category data-loading
python scripts/benchmarks/run_benchmarks.py --category memory-profiling

# Execute with detailed reporting
python scripts/benchmarks/run_benchmarks.py --verbose --report-artifacts
```

#### 2. Manual pytest Execution

```bash
# Run only performance tests
pytest -m "performance or benchmark" --benchmark-only

# Run performance tests with detailed reporting
pytest -m "benchmark" \
    --benchmark-min-rounds=5 \
    --benchmark-disable-gc \
    --benchmark-json=results.json
```

### Performance Test Implementation Standards

#### SLA Validation Tests

```python
@pytest.mark.performance
@pytest.mark.benchmark
def test_benchmark_data_loading_sla_validation():
    """
    Validate data loading meets SLA requirement: <1s per 100MB.
    
    SLA: Data loading must complete within 1 second per 100MB
    Measurement: Statistical analysis with confidence intervals
    Environment: Normalized for CI constraints
    """
    # ARRANGE
    test_data_100mb = generate_test_dataset(size_mb=100)
    sla_threshold_seconds = 1.0
    
    # ACT & ASSERT
    with benchmark_timer("data_loading_100mb") as timer:
        loaded_data = load_experimental_data(test_data_100mb)
    
    # Validate SLA compliance
    assert timer.duration < sla_threshold_seconds
    assert loaded_data is not None
    
    # Log performance metrics for trend analysis
    log_performance_metric("data_loading_100mb", timer.duration, sla_threshold_seconds)
```

#### Memory Profiling Tests

```python
@pytest.mark.performance
@pytest.mark.memory
def test_memory_profiling_large_dataset_processing():
    """
    Profile memory usage for large dataset processing.
    
    Validates: Memory usage patterns and leak detection
    Dataset: 500MB experimental data simulation
    Monitoring: Line-by-line memory analysis
    """
    # ARRANGE
    large_dataset = generate_experimental_dataset(size_mb=500)
    
    # ACT
    with memory_profiler() as profiler:
        for chunk in process_dataset_in_chunks(large_dataset):
            analyze_chunk(chunk)
            # Explicit cleanup to test memory management
            del chunk
            gc.collect()
    
    # ASSERT
    memory_report = profiler.get_report()
    assert memory_report.peak_usage_mb < 1000  # Peak usage limit
    assert memory_report.final_usage_mb < memory_report.initial_usage_mb + 100  # Leak detection
```

### CI/CD Integration

Performance tests are executed through optional GitHub Actions workflows:

```yaml
# .github/workflows/test.yml (excerpt)
- name: Run performance benchmarks (conditional)
  if: ${{ inputs.run_performance_tests != false && matrix.python-version == '3.11' && matrix.os == 'ubuntu-latest' }}
  run: |
    python scripts/benchmarks/run_benchmarks.py --ci-mode --report-artifacts
```

### Performance Baseline Management

```python
# Example performance baseline configuration
PERFORMANCE_BASELINES = {
    'data_loading_per_mb': 0.01,      # 10ms per MB
    'discovery_per_1k_files': 0.5,   # 500ms per 1000 files  
    'transformation_per_1m_rows': 0.5 # 500ms per 1M rows
}

def validate_performance_regression(operation, actual_time, baseline_key):
    """Validate performance against historical baselines."""
    baseline = PERFORMANCE_BASELINES[baseline_key]
    regression_threshold = baseline * 1.2  # 20% regression tolerance
    
    assert actual_time <= regression_threshold, (
        f"Performance regression detected: {operation} took {actual_time:.3f}s, "
        f"baseline: {baseline:.3f}s, threshold: {regression_threshold:.3f}s"
    )
```

---

## Network Test Management

### Overview

Tests requiring external network resources or service dependencies are annotated with `@pytest.mark.network` and automatically skipped during default test execution to ensure consistent CI/CD pipeline performance and eliminate external service dependencies from routine development workflows.

### Network Test Configuration

| Test Category | Default Behavior | Network Flag Behavior | Use Cases |
|---|---|---|---|
| **Unit Tests** | Complete isolation, no network access | Unchanged (network not required) | Core functionality validation |
| **Integration Tests** | Skip @pytest.mark.network tests | Include network-dependent tests | Cross-module workflows |
| **Performance Tests** | Excluded from default execution | Execute via scripts/benchmarks/ | Benchmark validation |

### Network Test Implementation

#### Basic Network Test Pattern

```python
@pytest.mark.network
def test_network_remote_configuration_loading():
    """
    Test loading configuration from network-accessible locations.
    
    Network dependency: Requires external service access
    Execution: Skipped by default, enabled with --run-network
    Timeout: Configured for network latency tolerance
    """
    # ARRANGE
    remote_config_url = "https://example.com/research-config.yaml"
    timeout_seconds = 10
    
    # ACT
    with network_timeout(timeout_seconds):
        config = load_remote_configuration(remote_config_url)
    
    # ASSERT
    assert config is not None
    assert "project" in config
    assert config["project"]["name"] is not None
```

#### Network Test with Retry Logic

```python
@pytest.mark.network
@pytest.mark.integration
def test_network_data_synchronization_with_retry():
    """
    Test data synchronization with retry mechanism.
    
    Network dependency: Remote data repository access
    Resilience: Implements retry logic for transient failures
    Research context: Multi-site collaboration scenarios
    """
    # ARRANGE
    remote_repository = "https://data.example.com/experiments/"
    max_retries = 3
    retry_delay = 1.0
    
    # ACT
    for attempt in range(max_retries):
        try:
            sync_result = synchronize_experimental_data(remote_repository)
            break  # Success, exit retry loop
        except NetworkError as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                raise  # Final attempt failed
    
    # ASSERT
    assert sync_result.success is True
    assert sync_result.files_synchronized > 0
```

### Network Test Execution Commands

```bash
# Default execution (network tests skipped)
pytest

# Enable network-dependent tests
pytest --run-network

# Enable network tests with reduced parallelism for stability
pytest --run-network -n 2

# Full validation including network tests
pytest --run-network --runslow
```

### Network Test Isolation Configuration

```python
# pytest.ini configuration for network test isolation
[pytest]
addopts = 
    # Skip network tests by default
    -m "not network and not performance and not slow"

markers =
    network: marks tests requiring network connectivity (excluded by default)
```

### Network Mock Patterns

For tests that simulate network behavior without actual network access:

```python
@pytest.mark.unit
def test_network_behavior_simulation(mock_external_dependencies_comprehensive):
    """
    Test network behavior through simulation without actual network access.
    
    Approach: Mock network responses for deterministic testing
    Benefits: Fast execution, no external dependencies
    Use case: Network error handling validation
    """
    # ARRANGE
    mock_network = mock_external_dependencies_comprehensive
    mock_network.simulate_network_delay(delay_seconds=2.0)
    mock_network.simulate_network_error(error_type="timeout")
    
    # ACT & ASSERT
    with pytest.raises(NetworkTimeoutError):
        result = fetch_remote_data_with_timeout(timeout=1.0)
```

---

## Parallel Execution Configuration

### Overview

The flyrigloader test suite is optimized for **parallel execution using pytest-xdist with a standardized 4-worker configuration** for both local development and CI environments. This provides optimal performance while maintaining test isolation and deterministic results.

### Standardized Parallelism Settings

| **Environment** | **Configuration** | **Worker Count** | **Execution Context** |
|---|---|---|---|
| **Local Development** | `pytest -n 4` | 4 workers (recommended) | Optimal balance for developer experience |
| **CI Pipeline** | `pytest -n 4` | 4 workers (standardized) | Consistent with GitHub Actions runner capabilities |
| **Benchmark Jobs** | `pytest -n auto` | Auto-detected CPU cores | Maximum performance for resource-intensive tests |
| **Network Testing** | `pytest -n 2 --run-network` | 2 workers (reduced) | Conservative approach for external dependencies |

### Configuration Implementation

#### pyproject.toml Configuration

```toml
[tool.pytest.ini_options]
addopts = [
    # Parallel execution with standardized 4-worker setup
    "-n", "4",
    # Test distribution strategy for load balancing
    "--dist=worksteal",
    # Other configuration options...
]
```

#### Execution Examples

```bash
# Standard parallel execution (4 workers)
pytest

# Manual worker count specification
pytest -n 4

# Auto-detect CPU cores for benchmarks
pytest -n auto -m "benchmark"

# Reduced parallelism for network tests
pytest -n 2 --run-network

# Sequential execution for debugging
pytest -n 0  # or pytest without -n flag
```

### Worker-Specific Configuration

The test infrastructure automatically configures worker-specific settings for isolation and reproducibility:

```python
# From tests/conftest.py
@pytest.fixture(scope="session", autouse=True)
def configure_pytest_xdist_parallel_execution():
    """
    Session-scoped autouse fixture for pytest-xdist optimization.
    
    Features:
    - Automatic worker detection and configuration
    - Test isolation guarantees for parallel execution
    - Shared fixture management across workers
    - Load balancing optimization
    - Memory usage monitoring per worker
    """
    worker_id = os.environ.get('PYTEST_XDIST_WORKER')
    
    if worker_id:
        # Configure worker-specific settings
        logger.info(f"Initializing pytest-xdist worker: {worker_id}")
        
        # Set worker-specific random seed for reproducible tests
        worker_num = int(worker_id.replace('gw', '')) if 'gw' in worker_id else 0
        random.seed(42 + worker_num)
        if np:
            np.random.seed(42 + worker_num)
        
        # Configure worker-specific temporary directory
        worker_tempdir = Path(tempfile.gettempdir()) / f"pytest_worker_{worker_id}"
        worker_tempdir.mkdir(exist_ok=True)
```

### Test Isolation Guarantees

Each test runs in complete isolation with no shared state dependencies:

```python
@pytest.fixture(scope="function")
def pytest_xdist_worker_info():
    """
    Provide worker information for tests requiring worker-specific behavior.
    
    Returns:
        Dict: Worker information including ID, process info, and capabilities
    """
    worker_id = os.environ.get('PYTEST_XDIST_WORKER', 'master')
    
    return {
        'worker_id': worker_id,
        'is_worker': worker_id != 'master',
        'worker_number': int(worker_id.replace('gw', '')) if 'gw' in worker_id else 0,
        'process_id': os.getpid(),
        'parallel_execution': worker_id != 'master'
    }
```

### Load Balancing Strategy

The `--dist=worksteal` strategy provides optimal load balancing:

- **Work Stealing**: Workers can take tests from other workers' queues when idle
- **Dynamic Distribution**: Tests are distributed based on execution time
- **Efficient Resource Utilization**: Minimizes idle time across workers

### Performance Benefits

Parallel execution provides significant performance improvements:

```python
# Performance comparison metrics
EXECUTION_TIME_COMPARISON = {
    'sequential_execution': '45-60 seconds',
    'parallel_4_workers': '12-18 seconds',
    'speedup_factor': '3.5x average',
    'target_achievement': '<30 seconds consistently'
}
```

---

## Automated Style Enforcement

### Overview

The flyrigloader project uses **flake8-pytest-style integration** for automated enforcement of testing conventions including AAA patterns, naming standards, and fixture usage compliance through pre-commit hooks and CI pipeline validation.

### Tool Integration

#### flake8-pytest-style Configuration

```ini
# .flake8 configuration file
[flake8]
# Enhanced pytest style configuration for AAA patterns and naming conventions
extend-ignore = ["E203", "W503"]
max-line-length = 100
per-file-ignores = [
    "tests/*:PT009,PT027"  # Allow longer parametrize names and fixtures in tests
]

# pytest-style plugin configuration for enforcing testing standards
pytest-fixture-no-parentheses = true
pytest-mark-no-parentheses = true
pytest-parametrize-names-type = "tuple"
pytest-parametrize-values-type = "tuple"

# Pytest style enforcement rules
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings  
    "F",      # pyflakes
    "PT",     # flake8-pytest-style
]
```

#### Pre-commit Hook Integration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/PyCQA/flake8
    rev: 7.2.0
    hooks:
      - id: flake8
        additional_dependencies:
          - flake8-pytest-style==2.1.0
        args: 
          - --extend-ignore=E203,W503
          - --max-line-length=100
          - --pytest-fixture-no-parentheses
          - --pytest-mark-no-parentheses
  
  - repo: local
    hooks:
      - id: pytest-aaa-validation
        name: Pytest AAA Pattern Enforcement
        entry: python -m flake8 --select=PT
        language: system
        files: ^tests/.*\.py$
        description: Enforce Arrange-Act-Assert patterns in test functions
      
      - id: pytest-naming-validation  
        name: Pytest Naming Convention Check
        entry: python -m flake8 --select=PT001,PT006,PT007
        language: system
        files: ^tests/.*\.py$
        description: Validate pytest naming conventions and parametrize usage
```

### Validation Rules

#### Core Validation Rules

| **Validation Rule** | **pytest-style Code** | **Enforcement Level** | **Example Violation** |
|---|---|---|---|
| **Test Function Naming** | PT001 | Blocking | `def unit_test_config()` → `def test_config_loading_success()` |
| **Fixture Naming** | PT003 | Blocking | `@pytest.fixture() def data()` → `@pytest.fixture def sample_data()` |
| **Parametrize Structure** | PT006, PT007 | Blocking | Inconsistent parameter names and value types |
| **AAA Pattern Enforcement** | PT024, PT025 | Warning | Missing arrange/act/assert structure comments |

#### Detailed Rule Examples

```python
# ✅ Correct: Proper test function naming
def test_config_yaml_loading_success():
    """Test YAML configuration loading success case."""
    pass

# ❌ Incorrect: Invalid test function naming  
def validate_config_loading():  # PT001 violation
    """Missing 'test_' prefix."""
    pass

# ✅ Correct: Proper fixture definition
@pytest.fixture
def sample_config_data():
    """Sample configuration data for testing."""
    return {"project": {"name": "test"}}

# ❌ Incorrect: Fixture with parentheses
@pytest.fixture()  # PT003 violation
def config():
    return {}

# ✅ Correct: Proper parametrize usage
@pytest.mark.parametrize(
    ("input_value", "expected_result"),
    [
        ("valid_input", True),
        ("invalid_input", False),
    ]
)
def test_validation_function(input_value, expected_result):
    assert validate_input(input_value) == expected_result

# ❌ Incorrect: Inconsistent parametrize structure
@pytest.mark.parametrize("input,expected", [  # PT006 violation
    "valid_input", True,  # PT007 violation - should be tuple
    "invalid_input", False,
])
def test_validation(input, expected):
    pass
```

### CI Pipeline Integration

```yaml
# .github/workflows/quality-assurance.yml (excerpt)
- name: Enhanced pytest style validation
  run: |
    echo "::group::Pytest Style Validation"
    flake8 tests/ --select=PT --statistics
    echo "::endgroup::"

- name: AAA pattern enforcement check
  run: |
    echo "::group::AAA Pattern Enforcement"
    python -m flake8 tests/ --select=PT024,PT025 --statistics
    echo "::endgroup::"
```

### Style Validation Commands

```bash
# Run pytest style validation
flake8 tests/ --select=PT

# Check specific rule categories
flake8 tests/ --select=PT001,PT003  # Naming conventions
flake8 tests/ --select=PT024,PT025  # AAA patterns

# Generate detailed style report
flake8 tests/ --select=PT --statistics --tee --output-file=style-report.txt
```

### Automated Fixes

Some style violations can be automatically fixed:

```bash
# Auto-fix fixture parentheses
python -m autoflake --remove-unused-variables --in-place tests/**/*.py

# Format parametrize decorators
python -m black tests/ --line-length=100
```

---

## Pull Request Requirements

### Overview

Enhanced pull request documentation requirements ensure **comprehensive traceability for all test modifications**, supporting research reproducibility and maintaining clear audit trails for experimental data processing validation changes that could impact scientific workflow reliability.

### Mandatory Test Documentation Requirements

#### Complete Test Modification Documentation

Every removed or rewritten test must be comprehensively documented in the PR description with detailed justification including:

1. **Original test purpose and functionality**
2. **Specific reason for modification or removal**
3. **Replacement test identification (if applicable)**
4. **Coverage impact analysis**
5. **Confirmation of maintained functionality**
6. **Explicit mapping to old test names**

#### PR Description Template

```markdown
## Test Modifications Summary

### Tests Removed
- `test_old_function_name()` - Reason: Refactored to use centralized fixture
  - **Original Purpose**: Tested configuration loading with hardcoded values
  - **Replacement**: `test_config_loading_with_centralized_fixture()`
  - **Coverage Impact**: No reduction, same functionality with better fixture management
  - **Functionality Maintained**: ✅ All original validation preserved

### Tests Rewritten  
- `test_implementation_specific_check()` - Reason: Converted from whitebox to blackbox testing
  - **Original Purpose**: Checked internal `_config` attribute
  - **New Implementation**: `test_config_behavior_validation()`
  - **Change Rationale**: Eliminates implementation coupling per Section 6.6.1.1
  - **Behavior Preserved**: ✅ Same validation through public API

### Tests Added
- `test_edge_case_unicode_file_paths()` - Purpose: Enhanced edge-case coverage
  - **Coverage Gap Addressed**: Unicode path handling validation
  - **Testing Category**: Edge-case validation per Section 6.6.2.4
  - **Research Relevance**: International collaboration file sharing

### Coverage Analysis
- **Before**: 89.2% line coverage, 84.1% branch coverage
- **After**: 91.5% line coverage, 87.3% branch coverage
- **Critical Modules**: All maintain 100% coverage requirement
- **New Edge Cases**: 15 additional boundary conditions tested

### Compliance Checklist
- [ ] All test modifications documented with justification
- [ ] Coverage thresholds maintained (≥90% line, ≥85% branch)
- [ ] AAA pattern compliance verified with flake8-pytest-style
- [ ] Naming conventions follow standardized patterns
- [ ] Centralized fixtures used where applicable
- [ ] Performance test isolation maintained (scripts/benchmarks/)
- [ ] Network test annotations applied correctly (@pytest.mark.network)
```

### Test Traceability Matrix

Each PR must include a traceability matrix for complex refactoring:

| Original Test | New Test | Modification Type | Justification | Coverage Impact |
|---|---|---|---|---|
| `test_private_attr_access()` | `test_public_api_behavior()` | Refactor | Remove implementation coupling | No change |
| `test_duplicate_fixture_setup()` | Uses `sample_config_dict` fixture | Consolidation | Eliminate code duplication | Improved |
| `test_slow_performance_check()` | Moved to `scripts/benchmarks/` | Isolation | Maintain rapid feedback | Excluded from default |

### Validation Requirements

#### Pre-PR Checklist

```bash
# 1. Run style validation
flake8 tests/ --select=PT --statistics

# 2. Verify coverage requirements
pytest --cov=src/flyrigloader --cov-fail-under=90

# 3. Check critical module coverage
pytest --cov=src/flyrigloader/io --cov-fail-under=100

# 4. Validate AAA pattern compliance
flake8 tests/ --select=PT024,PT025

# 5. Confirm fixture naming conventions
python -c "
import ast
import glob
fixture_files = glob.glob('tests/**/*.py', recursive=True)
# Custom validation script for fixture naming
"
```

#### Automated PR Validation

```yaml
# .github/workflows/pr-validation.yml
name: PR Test Validation

on:
  pull_request:
    paths: ['tests/**']

jobs:
  validate-test-changes:
    runs-on: ubuntu-latest
    steps:
    - name: Check PR description for test documentation
      run: |
        # Validate PR description includes test modification documentation
        if [[ ! "${{ github.event.pull_request.body }}" =~ "Test Modifications Summary" ]]; then
          echo "❌ PR must document test modifications"
          exit 1
        fi
    
    - name: Validate test traceability
      run: |
        # Check for removed tests without documentation
        git diff --name-only origin/main..HEAD | grep "test_.*\.py" | while read file; do
          if git diff origin/main..HEAD "$file" | grep -q "^-def test_"; then
            echo "⚠️  Test removal detected in $file - ensure PR documents this"
          fi
        done
```

### Research Reproducibility Requirements

#### Documentation Standards for Research Context

All test modifications must consider **research reproducibility implications**:

```markdown
### Research Impact Assessment

#### Experimental Workflow Validation Changes
- **Data Processing Steps Affected**: Configuration loading, file discovery
- **Validation Coverage**: All research data pipeline steps maintain validation
- **Backward Compatibility**: Existing research workflows remain supported
- **Breaking Changes**: None - all changes are internal test improvements

#### Scientific Rigor Maintenance
- **Test Coverage**: Enhanced from 89% to 91% with better edge-case handling
- **Error Detection**: Improved validation of corrupted data scenarios
- **Cross-Platform Validation**: Added Unicode path testing for international collaboration

#### Research Community Impact
- **External Researchers**: No impact on public API usage
- **Data Processing Reliability**: Enhanced through better test coverage
- **Documentation**: Updated testing guidelines reflect new standards
```

---

## Coverage Requirements

### Overview

The flyrigloader project enforces **comprehensive coverage requirements with quality gating** to prevent coverage regression and ensure consistent code quality maintenance across development cycles.

### Global Coverage Targets

| Coverage Type | Threshold | Enforcement | Quality Gate |
|---|---|---|---|
| **Line Coverage** | ≥90% | CI/CD pipeline gating | Mandatory CI gate |
| **Branch Coverage** | ≥85% | Decision point validation | PR blocking gate |
| **Function Coverage** | 100% | Exported API functions | Mandatory CI gate |
| **Edge-Case Coverage** | ≥90% | Boundary conditions | Warning-level gate |

### Critical Module Requirements

**100% Coverage Required:**
- `src/flyrigloader/api.py` - Primary API interface
- `src/flyrigloader/config/yaml_config.py` - Configuration management
- `src/flyrigloader/config/discovery.py` - Discovery configuration
- `src/flyrigloader/discovery/files.py` - File discovery engine
- `src/flyrigloader/io/pickle.py` - Data loading core

### Module-Specific Coverage Targets

| **Module Category** | **Line Coverage** | **Branch Coverage** | **Edge-Case Requirements** | **Quality Gate** |
|---|---|---|---|---|
| **Critical API Modules** | 100% | 100% | All error paths validated | Mandatory CI gate |
| **Core Business Logic** | ≥95% | ≥95% | Exception handling required | PR blocking gate |
| **Utility Modules** | ≥85% | ≥80% | Boundary conditions tested | Warning-level gate |
| **Integration Layers** | ≥90% | ≥85% | Cross-module interactions | Mandatory CI gate |

### Coverage Validation Implementation

#### Automated Coverage Enforcement

```bash
# Global coverage validation
pytest --cov=src/flyrigloader --cov-fail-under=90 --cov-branch

# Critical module validation
pytest --cov=src/flyrigloader/api --cov-fail-under=100
pytest --cov=src/flyrigloader/io --cov-fail-under=100

# Branch coverage enforcement
pytest --cov-branch --cov-report=term-missing
```

#### CI/CD Pipeline Integration

```yaml
# .github/workflows/test.yml (excerpt)
- name: Validate critical module coverage
  run: |
    echo "::group::Critical Module Coverage Validation"
    coverage report --include="*/flyrigloader/io/*" --fail-under=100
    coverage report --include="*/flyrigloader/config/*" --fail-under=100
    coverage report --include="*/flyrigloader/discovery/*" --fail-under=95
    echo "::endgroup::"
```

### Coverage Analysis and Reporting

#### Comprehensive Coverage Reports

```python
# Coverage configuration in pyproject.toml
[tool.coverage.run]
source = ["src/flyrigloader"]
branch = true
parallel = true
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__pycache__/*"
]

[tool.coverage.report]
show_missing = true
fail_under = 90
precision = 2
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if __name__ == .__main__.:"
]
```

#### Coverage Gap Analysis

```bash
# Generate detailed coverage analysis
coverage report --show-missing --precision=2

# Identify uncovered lines
coverage report --include="*/flyrigloader/*" --show-missing

# Generate HTML report for visual analysis
coverage html -d htmlcov-detailed

# Export coverage data for analysis
coverage json -o coverage-analysis.json
```

### Edge-Case Coverage Requirements

#### Boundary Condition Testing

All modules must include comprehensive boundary condition testing:

```python
@pytest.mark.edge_case
@pytest.mark.parametrize("test_input,expected_behavior", [
    ([], "empty_input_handling"),
    ([1], "single_element_processing"),
    (list(range(1000000)), "large_input_processing"),
    (None, "null_input_validation"),
])
def test_edge_case_boundary_conditions(test_input, expected_behavior):
    """
    Comprehensive boundary condition validation.
    
    Coverage requirement: All boundary conditions must be tested
    Edge cases: Empty, single element, large, null inputs
    Quality gate: Contributes to ≥90% edge-case coverage
    """
    # Implementation validates all boundary scenarios
```

#### Error Path Coverage

All error handling paths must be explicitly tested:

```python
@pytest.mark.error_handling
def test_error_path_coverage_comprehensive():
    """
    Validate all error handling paths are covered.
    
    Coverage requirement: 100% of exception paths tested
    Error scenarios: File not found, permission denied, corruption
    Quality gate: Mandatory for critical modules
    """
    # Test all possible error conditions
    error_scenarios = [
        (FileNotFoundError, "nonexistent_file.pkl"),
        (PermissionError, "readonly_file.pkl"),
        (pickle.UnpicklingError, "corrupted_file.pkl"),
    ]
    
    for error_type, test_file in error_scenarios:
        with pytest.raises(error_type):
            load_file_with_error_handling(test_file)
```

### Coverage Regression Prevention

#### Automated Quality Gates

```python
# CI pipeline coverage validation script
def validate_coverage_regression():
    """Prevent coverage regression in CI pipeline."""
    current_coverage = get_current_coverage()
    baseline_coverage = get_baseline_coverage()
    
    if current_coverage < baseline_coverage - 0.5:  # 0.5% tolerance
        raise CoverageRegressionError(
            f"Coverage regression detected: {current_coverage}% < {baseline_coverage}%"
        )
    
    # Update baseline if coverage improved
    if current_coverage > baseline_coverage:
        update_coverage_baseline(current_coverage)
```

---

## Examples and Best Practices

### Complete Test Examples

#### Unit Test Example with Full AAA Pattern

```python
@pytest.mark.unit
@pytest.mark.data_loading
def test_pickle_file_loading_success_comprehensive(
    sample_experimental_matrix,
    temp_cross_platform_dir,
    mock_data_loading_comprehensive
):
    """
    Comprehensive unit test for pickle file loading success scenario.
    
    Validates: Pickle file loading returns expected data structure
    Edge cases: Cross-platform file path handling
    Dependencies: Centralized fixtures for consistent test setup
    Coverage: Contributes to 100% critical module coverage requirement
    """
    # ARRANGE - Set up test data and dependencies
    test_file_path = temp_cross_platform_dir / "test_experiment.pkl"
    expected_data_structure = sample_experimental_matrix
    expected_columns = ['t', 'x', 'y', 'signal']
    
    # Setup mock data loading behavior
    mock_data_loading_comprehensive.add_experimental_matrix(
        str(test_file_path),
        n_timepoints=1000,
        include_signal=True,
        include_metadata=True
    )
    
    # ACT - Execute the function under test
    loaded_data = load_pickle_file(test_file_path)
    
    # ASSERT - Verify the results
    assert loaded_data is not None
    assert isinstance(loaded_data, dict)
    
    # Validate expected data structure
    for column in expected_columns:
        assert column in loaded_data
        assert len(loaded_data[column]) == 1000
    
    # Validate data types and ranges
    assert all(isinstance(t, (int, float)) for t in loaded_data['t'])
    assert all(0 <= x <= 120 for x in loaded_data['x'])  # Arena bounds
    assert all(0 <= y <= 120 for y in loaded_data['y'])
    
    # Verify metadata presence
    assert 'date' in loaded_data
    assert 'exp_name' in loaded_data
    assert 'rig' in loaded_data
```

#### Integration Test Example

```python
@pytest.mark.integration
@pytest.mark.configuration
@pytest.mark.file_discovery
def test_integration_config_driven_file_discovery_workflow(
    sample_comprehensive_config_dict,
    temp_filesystem_structure,
    mock_config_and_discovery_comprehensive
):
    """
    Integration test for configuration-driven file discovery workflow.
    
    Validates: Complete workflow from config loading to file discovery
    Integration: Config module + Discovery module interaction
    Research context: Typical experimental data organization workflow
    """
    # ARRANGE - Set up complete integration environment
    config_file = temp_filesystem_structure["config_file"]
    data_directory = temp_filesystem_structure["data_root"]
    expected_experiment_files = [
        temp_filesystem_structure["baseline_file_1"],
        temp_filesystem_structure["baseline_file_2"],
        temp_filesystem_structure["opto_file_1"]
    ]
    
    # Configure mock behavior for integration
    mocks = mock_config_and_discovery_comprehensive
    mocks["load_config"].return_value = sample_comprehensive_config_dict
    
    # ACT - Execute integrated workflow
    loaded_config = load_config(config_file)
    discovery_patterns = get_discovery_patterns_from_config(loaded_config)
    discovered_files = discover_files_with_patterns(data_directory, discovery_patterns)
    
    # ASSERT - Verify integration results
    assert loaded_config is not None
    assert len(discovery_patterns) > 0
    assert len(discovered_files) >= len(expected_experiment_files)
    
    # Validate discovered files match expected structure
    discovered_file_names = [f.name for f in discovered_files]
    for expected_file in expected_experiment_files:
        assert expected_file.name in discovered_file_names
    
    # Verify configuration patterns were applied correctly
    baseline_files = [f for f in discovered_files if "baseline" in f.name]
    opto_files = [f for f in discovered_files if "opto" in f.name]
    assert len(baseline_files) >= 2  # At least 2 baseline files expected
    assert len(opto_files) >= 1     # At least 1 optogenetic file expected
    
    # Validate ignored files were excluded
    ignored_files = [f for f in discovered_files if any(
        ignore_pattern in f.name 
        for ignore_pattern in loaded_config["project"]["ignore_substrings"]
    )]
    assert len(ignored_files) == 0  # No ignored files should be discovered
```

#### Edge-Case Test Example

```python
@pytest.mark.edge_case
@pytest.mark.error_handling
@pytest.mark.parametrize("corruption_scenario", [
    "truncated_pickle",
    "invalid_header", 
    "binary_contamination",
    "empty_file"
])
def test_edge_case_corrupted_file_comprehensive_handling(
    corruption_scenario,
    fixture_corrupted_file_scenarios,
    temp_cross_platform_dir
):
    """
    Comprehensive edge-case test for corrupted file handling.
    
    Validates: System gracefully handles various file corruption scenarios
    Edge cases: Multiple corruption patterns affecting research data
    Research context: Hardware failures during long experimental sessions
    Error handling: Informative error messages for debugging
    """
    # ARRANGE - Set up corruption scenario
    corrupted_files = fixture_corrupted_file_scenarios
    corrupted_file_path = corrupted_files[corruption_scenario]
    
    # Define expected error behavior based on corruption type
    expected_errors = {
        "truncated_pickle": pickle.UnpicklingError,
        "invalid_header": pickle.UnpicklingError,
        "binary_contamination": UnicodeDecodeError,
        "empty_file": EOFError
    }
    expected_error_type = expected_errors[corruption_scenario]
    
    # ACT & ASSERT - Verify appropriate error handling
    with pytest.raises(expected_error_type) as exc_info:
        load_experimental_data_with_validation(corrupted_file_path)
    
    # Validate error message quality for research users
    error_message = str(exc_info.value).lower()
    assert "corrupted" in error_message or "invalid" in error_message
    assert str(corrupted_file_path) in str(exc_info.value)
    
    # Verify error logging for debugging
    assert any(
        "file corruption detected" in log_record.message.lower()
        for log_record in caplog.records
    )
    
    # Ensure system state remains stable after error
    # (no memory leaks, proper cleanup)
    assert verify_system_stability_after_error()
```

### Performance Test Example (scripts/benchmarks/)

```python
# Located in scripts/benchmarks/data_loading_benchmarks.py
@pytest.mark.benchmark
@pytest.mark.performance
def test_benchmark_data_loading_sla_comprehensive(benchmark):
    """
    Comprehensive performance benchmark for data loading SLA validation.
    
    SLA Requirement: Data loading must complete within 1 second per 100MB
    Measurement: Statistical analysis with confidence intervals
    Environment: Normalized for CI constraints and hardware variations
    Regression: Compared against historical performance baselines
    """
    # ARRANGE - Set up performance test environment
    test_data_sizes = [10, 50, 100, 200]  # MB
    sla_seconds_per_100mb = 1.0
    
    performance_results = []
    
    for data_size_mb in test_data_sizes:
        # Generate test dataset of specified size
        test_dataset = generate_large_test_dataset(size_mb=data_size_mb)
        expected_max_time = (data_size_mb / 100.0) * sla_seconds_per_100mb
        
        # ACT - Execute benchmarked operation
        def load_operation():
            return load_experimental_data(test_dataset)
        
        result = benchmark.pedantic(
            load_operation,
            rounds=5,
            iterations=1,
            warmup_rounds=2
        )
        
        # ASSERT - Validate SLA compliance
        actual_time = benchmark.stats.mean
        assert actual_time <= expected_max_time, (
            f"SLA violation: {data_size_mb}MB took {actual_time:.3f}s, "
            f"expected ≤ {expected_max_time:.3f}s"
        )
        
        performance_results.append({
            'data_size_mb': data_size_mb,
            'actual_time': actual_time,
            'sla_time': expected_max_time,
            'sla_compliance': actual_time <= expected_max_time
        })
    
    # Generate performance analysis report
    generate_performance_report(performance_results, "data_loading_sla")
    
    # Verify all test sizes met SLA requirements
    assert all(result['sla_compliance'] for result in performance_results)
```

### Best Practices Summary

#### 1. Test Structure Best Practices

- **Always use AAA pattern** with clear section separation
- **Write descriptive docstrings** explaining test purpose and context
- **Use centralized fixtures** from `tests/conftest.py` and `tests/utils.py`
- **Follow naming conventions** consistently across all test types

#### 2. Mock and Fixture Best Practices

```python
# ✅ Good: Use centralized mock factories
filesystem_mock = create_mock_filesystem(
    structure={'files': {'/test/data.pkl': {'size': 1024}}},
    unicode_files=True
)

# ✅ Good: Descriptive fixture names with proper scope
@pytest.fixture(scope="function")
def sample_neuroscience_experiment_config():
    """Sample configuration for neuroscience experiment testing."""
    
# ❌ Avoid: Creating mocks inline in test functions
def test_something():
    mock_fs = MagicMock()  # Use centralized mock instead
```

#### 3. Edge-Case Testing Best Practices

- **Test boundary conditions systematically** using parametrized tests
- **Include platform-specific scenarios** for cross-platform compatibility
- **Validate error messages** are informative for research users
- **Test resource cleanup** after error conditions

#### 4. Performance Testing Best Practices

- **Isolate in scripts/benchmarks/** to maintain rapid feedback
- **Use statistical analysis** with multiple rounds and warmup
- **Validate against SLA requirements** with clear assertions
- **Generate trend reports** for regression detection

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Test Execution Issues

**Problem**: Tests fail with "fixture not found" errors
```
FAILED tests/test_example.py::test_function - fixture 'unknown_fixture' not found
```

**Solution**: Verify fixture is defined in centralized locations:
```bash
# Check fixture definitions
grep -r "def unknown_fixture" tests/conftest.py tests/utils.py

# Verify import statements
grep -r "from tests.utils import" tests/
```

#### 2. Coverage Issues

**Problem**: Coverage below 90% threshold
```
FAILED - coverage was 87.3%, expected >= 90%
```

**Solution**: Identify uncovered lines and add tests:
```bash
# Generate detailed coverage report
coverage report --show-missing --include="src/flyrigloader/*"

# Focus on specific modules
coverage report --show-missing --include="src/flyrigloader/module_name.py"

# Add tests for uncovered lines identified in report
```

#### 3. Parallel Execution Issues

**Problem**: Tests fail only in parallel execution
```
FAILED tests/test_example.py::test_function - only fails with -n 4
```

**Solution**: Check for shared state dependencies:
```python
# ❌ Problematic: Global state
global_var = None

def test_function_1():
    global global_var
    global_var = "test1"

def test_function_2():
    global global_var
    assert global_var == "test2"  # Fails in parallel

# ✅ Solution: Use fixtures for isolation
@pytest.fixture
def isolated_state():
    return {"var": None}

def test_function_1(isolated_state):
    isolated_state["var"] = "test1"

def test_function_2(isolated_state):
    isolated_state["var"] = "test2"
```

#### 4. Performance Test Issues

**Problem**: Performance tests run in default execution
```
Tests taking too long - benchmark tests should be excluded
```

**Solution**: Verify marker configuration:
```python
# Check test markers
@pytest.mark.performance  # Should be excluded by default
@pytest.mark.benchmark    # Should be excluded by default
def test_slow_operation():
    pass

# Verify pytest configuration
# pyproject.toml should have:
# addopts = ["-m", "not performance and not benchmark"]
```

#### 5. Style Validation Issues

**Problem**: flake8-pytest-style violations
```
PT001 use @pytest.fixture() over @pytest.fixture
PT024 pytest.mark.parametrize is missing values
```

**Solution**: Apply automatic fixes:
```bash
# Fix fixture parentheses automatically
sed -i 's/@pytest.fixture()/@pytest.fixture/g' tests/**/*.py

# Fix parametrize structure
# Manual fix required - ensure tuples are used for values
```

### Performance Optimization Tips

#### 1. Fixture Optimization

```python
# ✅ Use appropriate fixture scope
@pytest.fixture(scope="session")  # Expensive setup, reuse across tests
def expensive_test_data():
    return generate_large_dataset()

@pytest.fixture(scope="function")  # Test isolation required
def test_specific_data():
    return {"test": "data"}
```

#### 2. Parallel Execution Optimization

```bash
# Optimal worker count for development
pytest -n 4

# For CI with more resources
pytest -n auto

# For debugging (sequential)
pytest -n 0
```

#### 3. Test Selection Optimization

```bash
# Fast feedback - exclude slow tests
pytest -m "not slow and not performance"

# Specific test categories
pytest -m "unit and data_loading"

# Single module testing
pytest tests/test_specific_module.py -v
```

### Debugging Failed Tests

#### 1. Verbose Output

```bash
# Maximum verbosity
pytest -vv --tb=long

# Show local variables in traceback
pytest --tb=long --show-capture=all

# Capture output
pytest -s --capture=no
```

#### 2. Test Isolation

```bash
# Run single test
pytest tests/test_module.py::test_function_name -v

# Run with specific fixtures
pytest tests/test_module.py::test_function_name --setup-show
```

#### 3. Coverage Analysis for Failed Tests

```bash
# Run single test with coverage
pytest tests/test_module.py::test_function_name --cov=src/flyrigloader --cov-report=term-missing

# Generate HTML report for debugging
pytest tests/test_module.py --cov=src/flyrigloader --cov-report=html
```

---

## Conclusion

This comprehensive testing guidelines document serves as the authoritative reference for all testing practices within the flyrigloader project. It implements the testing strategy outlined in Section 6.6 of the technical specification, ensuring behavior-focused validation, comprehensive coverage, and optimal developer experience.

### Key Takeaways

1. **Behavior-Focused Testing**: All tests must validate public API behavior rather than implementation details
2. **Centralized Fixture Management**: Use `tests/conftest.py` and `tests/utils.py` for consistent test infrastructure
3. **AAA Pattern Enforcement**: Mandatory Arrange-Act-Assert structure with automated validation
4. **Performance Test Isolation**: Keep benchmarks in `scripts/benchmarks/` for rapid feedback cycles
5. **Comprehensive Coverage**: Maintain ≥90% line coverage and ≥85% branch coverage with quality gates
6. **Network Test Management**: Use `@pytest.mark.network` annotation for conditional execution
7. **Automated Style Enforcement**: flake8-pytest-style integration ensures consistent conventions

### Compliance Verification

Before submitting any code changes, verify compliance with this document:

```bash
# Run complete validation
pytest --cov=src/flyrigloader --cov-fail-under=90
flake8 tests/ --select=PT
python scripts/benchmarks/run_benchmarks.py --validate-sla
```

### Continuous Improvement

This document will be updated as the testing strategy evolves. All changes must maintain backward compatibility with existing test infrastructure while enhancing the research workflow validation capabilities of the flyrigloader project.

For questions or clarifications regarding these testing guidelines, refer to the technical specification Section 6.6 or consult with the development team through the project's standard communication channels.

**Document Version**: 1.0  
**Effective Date**: 2024-12-17  
**Next Review**: 2025-03-17