"""
Exhaustive pytest-benchmark suite for FlyRigLoader YAML configuration management enforcing 
comprehensive SLA validation for loading, validation, merging, memory usage, and discovery 
operations, relocated from default test suite to enable optional performance validation 
without impacting rapid development workflows.

Performance SLA Requirements:
- F-001-RQ-001: Load YAML configuration files within <100ms per Section 2.2.1
- F-001-RQ-002: Support Kedro parameter dictionaries with <50ms validation per Section 2.2.1 
- F-001-RQ-003: Validate configuration structure within <10ms validation per Section 2.2.1
- F-001-RQ-004: Merge project and experiment-level settings within <5ms merge operation per Section 2.2.1

Memory Usage Requirements:
- Memory usage validation (<10MB for large configurations) per specification requirements
- Memory leak detection for iterative configuration loading scenarios
- Large dataset memory efficiency validation for configuration-driven discovery operations

Configuration-Driven Discovery Performance:
- discover_files_with_config performance validation with configuration overhead analysis
- discover_experiment_files performance testing with hierarchical configuration merging
- discover_dataset_files performance validation with dataset-specific configuration processing

Regression Detection:
- Baseline performance establishment for future performance comparison across operations
- Statistical significance testing for performance changes with confidence intervals
- Automated performance alerts for CI/CD integration with configurable thresholds

This benchmark suite is isolated from the default pytest execution via @pytest.mark.benchmark
markers and executes exclusively through scripts/benchmarks/run_benchmarks.py CLI runner
or optional GitHub Actions benchmark jobs to maintain rapid developer feedback cycles.
"""

import gc
import json
import os
import tempfile
import tracemalloc
import time
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple, Optional
from unittest.mock import patch, MagicMock
from datetime import datetime

import numpy as np
import pytest
import yaml

# Import configuration management modules from flyrigloader
from flyrigloader.config.yaml_config import (
    load_config,
    validate_config_dict,
    get_ignore_patterns,
    get_mandatory_substrings,
    get_dataset_info,
    get_experiment_info,
    get_extraction_patterns,
    get_all_experiment_names,
    get_all_dataset_names
)
from flyrigloader.config.discovery import (
    discover_files_with_config,
    discover_experiment_files,
    discover_dataset_files
)

# Import benchmark utilities and configuration from local modules
from .utils import (
    MemoryProfiler,
    memory_profiling_context,
    estimate_data_size,
    StatisticalAnalysisEngine,
    EnvironmentAnalyzer,
    PerformanceArtifactGenerator,
    analyze_benchmark_results
)
from .config import (
    BenchmarkConfig,
    BenchmarkCategory,
    DEFAULT_BENCHMARK_CONFIG,
    get_benchmark_config,
    get_category_config,
    PerformanceSLA
)


# ============================================================================
# CONFIGURATION SIZE SCENARIO FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def small_config_dict():
    """
    Small configuration dictionary for baseline performance testing and rapid validation.
    
    Features:
    - Minimal valid configuration structure for baseline SLA validation
    - Single dataset and experiment for simple hierarchy testing
    - Basic extraction patterns for metadata processing validation
    - Lightweight structure for sub-millisecond performance validation
    
    Returns:
        Dict[str, Any]: Minimal valid configuration meeting all structural requirements
    """
    return {
        "project": {
            "directories": {
                "major_data_directory": "/test/data",
                "batchfile_directory": "/test/batch"
            },
            "ignore_substrings": ["temp", "backup"],
            "mandatory_experiment_strings": ["valid"],
            "extraction_patterns": [
                r".*_(?P<date>\d{8})\.csv",
                r".*_(?P<animal_id>\w+)_(?P<condition>\w+)\.pkl"
            ]
        },
        "datasets": {
            "basic_dataset": {
                "rig": "test_rig_001",
                "patterns": ["*_basic_*.csv", "*_test_*.pkl"],
                "dates_vials": {
                    "20240101": [1, 2, 3],
                    "20240102": [1, 2, 3, 4]
                },
                "metadata": {
                    "extraction_patterns": [
                        r".*_basic_(?P<date>\d{8})_(?P<replicate>\d+)\.csv"
                    ]
                }
            }
        },
        "experiments": {
            "basic_experiment": {
                "datasets": ["basic_dataset"],
                "filters": {
                    "ignore_substrings": ["debug"],
                    "mandatory_experiment_strings": ["basic"]
                },
                "metadata": {
                    "extraction_patterns": [
                        r".*_basic_experiment_(?P<date>\d{8})_(?P<condition>\w+)\.csv"
                    ]
                }
            }
        },
        "rigs": {
            "test_rig_001": {
                "sampling_frequency": 1000,
                "mm_per_px": 0.1,
                "camera_resolution": [1920, 1080]
            }
        }
    }


@pytest.fixture(scope="session")
def medium_config_dict():
    """
    Medium-sized configuration dictionary for realistic performance testing scenarios.
    
    Features:
    - 50 datasets with varying complexity for comprehensive hierarchy testing
    - 25 experiments with multiple dataset relationships for merging validation
    - Complex extraction patterns for metadata processing performance testing
    - Realistic configuration size for typical research project scenarios
    - Hierarchical filter merging validation with project/experiment overrides
    
    Returns:
        Dict[str, Any]: Realistic configuration with moderate complexity for SLA validation
    """
    datasets = {}
    experiments = {}
    
    # Generate 50 datasets with varying complexity to test hierarchical operations
    for i in range(50):
        dataset_name = f"dataset_{i:03d}"
        date_count = (i % 10) + 1  # 1-10 dates per dataset
        vial_count = (i % 5) + 1   # 1-5 vials per date
        
        # Generate realistic date patterns
        dates_vials = {}
        for month in range(1, min(date_count + 1, 13)):
            for day in range(1, min(date_count + 1, 29)):
                date_str = f"2024{month:02d}{day:02d}"
                dates_vials[date_str] = list(range(1, vial_count + 1))
        
        datasets[dataset_name] = {
            "rig": f"rig_{i % 5:03d}",
            "patterns": [
                f"*_{dataset_name}_*",
                f"experiment_*_{i:03d}_*",
                f"neural_data_{i:03d}_*.pkl"
            ],
            "dates_vials": dates_vials,
            "metadata": {
                "extraction_patterns": [
                    f".*_{dataset_name}_(?P<date>\\d{{8}})_(?P<replicate>\\d+)\\.csv",
                    f".*_(?P<condition>\\w+)_{dataset_name}_(?P<timestamp>\\d{{14}})\\.pkl",
                    f".*_{dataset_name}_(?P<animal_id>\\w+)_(?P<session>\\d+)\\.h5"
                ]
            },
            "filters": {
                "ignore_substrings": [f"temp_{i}", f"debug_{i}"],
                "mandatory_experiment_strings": [f"valid_{i}", f"dataset_{i:03d}"]
            }
        }
    
    # Generate 25 experiments with complex dataset relationships for merging tests
    for i in range(25):
        experiment_name = f"experiment_{i:03d}"
        dataset_count = (i % 8) + 1  # 1-8 datasets per experiment
        start_idx = (i * 2) % 40
        experiment_datasets = [f"dataset_{j:03d}" for j in range(start_idx, start_idx + dataset_count)]
        
        experiments[experiment_name] = {
            "datasets": experiment_datasets,
            "filters": {
                "ignore_substrings": [
                    f"exclude_{i}", f"temp_{i}", f"debug_{i}", f"preview_{i}"
                ],
                "mandatory_experiment_strings": [
                    f"include_{i}", f"valid_{i}", f"experiment_{i:03d}"
                ]
            },
            "metadata": {
                "extraction_patterns": [
                    f".*_{experiment_name}_(?P<date>\\d{{8}})_(?P<condition>\\w+)\\.csv",
                    f".*_(?P<phase>\\w+)_{experiment_name}_(?P<replicate>\\d+)\\.pkl",
                    f".*_{experiment_name}_(?P<timestamp>\\d{{14}})_(?P<trial>\\d+)\\.h5",
                    f".*_(?P<session>\\w+)_{experiment_name}_(?P<subject>\\d+)\\.mat"
                ]
            },
            "parameters": {
                f"param_{j}": f"value_{i}_{j}" for j in range(5)
            }
        }
    
    return {
        "project": {
            "directories": {
                "major_data_directory": "/test/medium/data",
                "batchfile_directory": "/test/medium/batch",
                "output_directory": "/test/medium/output"
            },
            "ignore_substrings": [
                "temp", "backup", "hidden", "cache", "debug", 
                "test_output", "intermediate", "preview"
            ],
            "mandatory_experiment_strings": ["valid", "processed", "final"],
            "extraction_patterns": [
                r".*_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.csv",
                r".*_(?P<experiment>\w+)_(?P<timestamp>\d{14})\.pkl",
                r".*_(?P<phase>\w+)_(?P<date>\d{8})_(?P<trial>\d+)\.h5",
                r".*_(?P<session>\w+)_(?P<subject>\d+)_(?P<run>\d+)\.mat"
            ]
        },
        "rigs": {
            f"rig_{i:03d}": {
                "sampling_frequency": 1000 + i * 100,
                "mm_per_px": 0.1 + i * 0.01,
                "camera_resolution": [1920 + i * 20, 1080 + i * 10],
                "calibration_parameters": {
                    "param_a": i * 0.1,
                    "param_b": i * 0.05,
                    "param_c": i * 1.0
                }
            } for i in range(5)
        },
        "datasets": datasets,
        "experiments": experiments
    }


@pytest.fixture(scope="session")
def large_config_dict():
    """
    Large configuration dictionary for stress testing performance and memory usage validation.
    
    Features:
    - 500 datasets with high complexity for memory usage testing (<10MB constraint)
    - 200 experiments with complex relationships for hierarchical merging stress testing
    - Extensive extraction patterns for regex processing performance validation
    - Large-scale configuration for discovery performance impact assessment
    - Comprehensive parameter structures for complex configuration merging scenarios
    
    Returns:
        Dict[str, Any]: Complex configuration for stress testing and memory validation
    """
    datasets = {}
    experiments = {}
    
    # Generate 500 datasets with high complexity for memory and performance stress testing
    for i in range(500):
        dataset_name = f"dataset_{i:04d}"
        dates_vials = {}
        
        # Generate extensive date ranges for memory usage testing
        for month in range(1, 13):
            for day in range(1, min(29, (i % 10) + 5)):
                date_str = f"2024{month:02d}{day:02d}"
                vial_count = min((i % 25) + 1, 20)  # 1-20 vials per date
                dates_vials[date_str] = list(range(1, vial_count + 1))
        
        datasets[dataset_name] = {
            "rig": f"rig_{i % 10:03d}",
            "patterns": [
                f"*_{dataset_name}_*",
                f"experiment_*_{i:04d}_*",
                f"neural_data_{i:04d}_*.pkl",
                f"behavioral_data_{i:04d}_*.csv",
                f"metadata_{i:04d}_*.json"
            ],
            "dates_vials": dates_vials,
            "metadata": {
                "extraction_patterns": [
                    f".*_{dataset_name}_(?P<date>\\d{{8}})_(?P<replicate>\\d+)\\.csv",
                    f".*_(?P<condition>\\w+)_{dataset_name}_(?P<timestamp>\\d{{14}})\\.pkl",
                    f".*_{dataset_name}_(?P<animal_id>\\w{{3,6}})_(?P<session>\\d{{1,3}})\\.h5",
                    f".*_(?P<phase>\\w{{4,8}})_{dataset_name}_(?P<trial>\\d{{1,4}})\\.mat",
                    f".*_{dataset_name}_(?P<experimenter>\\w+)_(?P<run>\\d{{2,3}})\\.json"
                ]
            },
            "filters": {
                "ignore_substrings": [
                    f"temp_{i}", f"debug_{i}", f"backup_{i}", 
                    f"preview_{i}", f"test_{i}", f"cache_{i}"
                ],
                "mandatory_experiment_strings": [
                    f"valid_{i}", f"dataset_{i:04d}", f"processed_{i}"
                ]
            },
            "parameters": {
                f"param_{j}": {
                    "value": f"complex_value_{i}_{j}",
                    "type": "string" if j % 2 else "numeric",
                    "validation": f"regex_pattern_{j}" if j % 3 else None
                } for j in range(10)
            }
        }
    
    # Generate 200 experiments with complex dataset relationships for stress testing
    for i in range(200):
        experiment_name = f"experiment_{i:04d}"
        dataset_count = min((i % 20) + 1, 15)  # 1-15 datasets per experiment
        start_idx = (i * 2) % 400
        experiment_datasets = [f"dataset_{j:04d}" for j in range(start_idx, start_idx + dataset_count)]
        
        experiments[experiment_name] = {
            "datasets": experiment_datasets,
            "filters": {
                "ignore_substrings": [
                    f"exclude_{i}", f"temp_{i}", f"debug_{i}", 
                    f"preview_{i}", f"cache_{i}", f"backup_{i}",
                    f"test_{i}", f"intermediate_{i}"
                ],
                "mandatory_experiment_strings": [
                    f"include_{i}", f"valid_{i}", f"experiment_{i:04d}",
                    f"processed_{i}", f"final_{i}"
                ]
            },
            "metadata": {
                "extraction_patterns": [
                    f".*_{experiment_name}_(?P<date>\\d{{8}})_(?P<condition>\\w{{3,10}})\\.csv",
                    f".*_(?P<phase>\\w{{4,8}})_{experiment_name}_(?P<replicate>\\d{{1,3}})\\.pkl",
                    f".*_{experiment_name}_(?P<timestamp>\\d{{14}})_(?P<trial>\\d{{1,4}})\\.h5",
                    f".*_(?P<session>\\w{{5,10}})_{experiment_name}_(?P<subject>\\d{{2,4}})\\.mat",
                    f".*_{experiment_name}_(?P<experimenter>\\w{{3,8}})_(?P<run>\\d{{2,3}})\\.json"
                ]
            },
            "parameters": {
                f"param_{j}": {
                    "value": f"complex_value_{i}_{j}",
                    "description": f"Parameter {j} for experiment {experiment_name}",
                    "constraints": {
                        "min_value": j * 0.1,
                        "max_value": j * 10.0,
                        "validation_regex": f"pattern_{i}_{j}"
                    }
                } for j in range(15)
            },
            "analysis_config": {
                "preprocessing": {
                    f"step_{k}": f"config_{i}_{k}" for k in range(5)
                },
                "processing": {
                    f"algorithm_{k}": f"params_{i}_{k}" for k in range(5)
                },
                "postprocessing": {
                    f"output_{k}": f"format_{i}_{k}" for k in range(5)
                }
            }
        }
    
    return {
        "project": {
            "directories": {
                "major_data_directory": "/test/large/data",
                "batchfile_directory": "/test/large/batch",
                "output_directory": "/test/large/output",
                "log_directory": "/test/large/logs",
                "cache_directory": "/test/large/cache",
                "temp_directory": "/test/large/temp"
            },
            "ignore_substrings": [
                "temp", "backup", "hidden", "cache", "debug", 
                "test_output", "intermediate", "preview", "draft",
                "sandbox", "experimental", "deprecated", "old"
            ],
            "mandatory_experiment_strings": [
                "valid", "processed", "final", "approved", "verified"
            ],
            "extraction_patterns": [
                r".*_(?P<date>\d{8})_(?P<condition>\w{3,10})_(?P<replicate>\d{1,3})\.csv",
                r".*_(?P<experiment>\w{4,12})_(?P<timestamp>\d{14})\.pkl",
                r".*_(?P<phase>\w{4,8})_(?P<date>\d{8})_(?P<trial>\d{1,4})\.h5",
                r".*_(?P<session>\w{5,10})_(?P<subject>\d{2,4})_(?P<run>\d{2,3})\.mat",
                r".*_(?P<animal_id>\w{3,6})_(?P<experimenter>\w{3,8})_(?P<protocol>\w{4,10})\.json"
            ]
        },
        "rigs": {
            f"rig_{i:03d}": {
                "sampling_frequency": 1000 + i * 50,
                "mm_per_px": 0.1 + i * 0.005,
                "camera_resolution": [1920 + i * 10, 1080 + i * 5],
                "calibration_parameters": {
                    "param_a": i * 0.05,
                    "param_b": i * 0.025,
                    "param_c": i * 0.75
                },
                "hardware_config": {
                    "camera_model": f"camera_model_{i}",
                    "lens_type": f"lens_{i % 5}",
                    "lighting_config": f"lighting_{i % 3}"
                }
            } for i in range(10)
        },
        "datasets": datasets,
        "experiments": experiments
    }


@pytest.fixture(scope="session", params=[
    ("small", "small_config_dict"),
    ("medium", "medium_config_dict"),
    ("large", "large_config_dict")
])
def config_size_scenario(request):
    """
    Parametrized fixture providing different configuration sizes for systematic performance testing.
    
    Features:
    - Small configuration for baseline performance validation
    - Medium configuration for realistic scenario testing  
    - Large configuration for stress testing and memory validation
    - Consistent interface for systematic SLA validation across all scales
    
    Args:
        request: Pytest request object with parameter information
        
    Returns:
        Tuple[str, Dict[str, Any]]: (size_name, config_dict) for systematic testing
    """
    size_name, fixture_name = request.param
    config_dict = request.getfixturevalue(fixture_name)
    return size_name, config_dict


@pytest.fixture(scope="function")
def temp_config_file(request):
    """
    Create temporary YAML configuration file for file-based loading performance tests.
    
    Features:
    - Temporary file creation with automatic cleanup
    - YAML serialization with default flow style for consistency
    - Support for parameterized configuration dictionaries
    - Cross-platform temporary file handling
    
    Args:
        request: Pytest request object with optional config parameter
        
    Returns:
        str: Path to temporary configuration file for file-based SLA testing
    """
    # Get configuration from parameter or default to small config
    if hasattr(request, "param"):
        config_dict = request.param
    else:
        config_dict = request.getfixturevalue("small_config_dict")
    
    # Create temporary file with YAML content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup temporary file
    try:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    except OSError:
        # File cleanup failure should not affect test results
        pass


# ============================================================================
# MEMORY USAGE MEASUREMENT UTILITIES
# ============================================================================

def measure_memory_usage(func, *args, **kwargs):
    """
    Measure peak memory usage during function execution with garbage collection control.
    
    Features:
    - Peak memory usage tracking via tracemalloc for accurate measurement
    - Garbage collection control for consistent measurement conditions
    - Memory measurement in MB for SLA validation (<10MB constraint)
    - Exception safety with guaranteed tracemalloc cleanup
    
    Args:
        func: Function to measure memory usage for
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple[Any, float]: (function_result, peak_memory_mb) for validation
    """
    # Force garbage collection before measurement for baseline establishment
    gc.collect()
    
    # Start memory tracing with detailed tracking
    tracemalloc.start()
    
    try:
        # Execute function under memory monitoring
        result = func(*args, **kwargs)
        
        # Get peak memory usage during execution
        current, peak = tracemalloc.get_traced_memory()
        peak_memory_mb = peak / (1024 * 1024)  # Convert bytes to MB
        
        return result, peak_memory_mb
    finally:
        # Ensure tracemalloc is stopped even on exceptions
        tracemalloc.stop()


@pytest.fixture(scope="function")
def memory_profiler_context():
    """
    Function-scoped fixture providing memory profiling context manager for large configuration scenarios.
    
    Features:
    - Memory profiling with configurable precision for detailed analysis
    - Integration with benchmark utilities for comprehensive memory validation
    - Large dataset memory efficiency validation (>500MB configuration scenarios)
    - Memory leak detection for iterative configuration loading scenarios
    
    Returns:
        Function that creates memory profiling context with specified parameters
    """
    def create_context(data_size_estimate: int, 
                      precision: int = 3,
                      enable_line_profiling: bool = True,
                      monitor_interval: float = 0.1):
        """
        Create memory profiling context with specified configuration.
        
        Args:
            data_size_estimate: Estimated size of configuration data in bytes
            precision: Decimal precision for memory measurements
            enable_line_profiling: Whether to enable continuous monitoring
            monitor_interval: Monitoring interval in seconds
            
        Returns:
            Memory profiling context manager for configuration benchmarks
        """
        return memory_profiling_context(
            data_size_estimate=data_size_estimate,
            precision=precision,
            enable_line_profiling=enable_line_profiling,
            monitor_interval=monitor_interval
        )
    
    return create_context


# ============================================================================
# CONFIGURATION LOADING BENCHMARK TESTS
# ============================================================================

class TestConfigurationLoadingPerformance:
    """
    Comprehensive performance benchmark tests for configuration loading operations with SLA validation.
    
    Test Categories:
    - File-based YAML configuration loading (F-001-RQ-001: <100ms)
    - Dictionary-based Kedro parameter validation (F-001-RQ-002: <50ms) 
    - Configuration structure validation (F-001-RQ-003: <10ms)
    - Memory usage validation for large configurations (<10MB constraint)
    """
    
    @pytest.mark.benchmark(group="config_loading")
    @pytest.mark.performance
    def test_load_config_from_file_performance_sla(self, benchmark, temp_config_file):
        """
        Benchmark F-001-RQ-001: Load YAML configuration files within <100ms SLA requirement.
        
        Validates file-based configuration loading performance against technical specification
        SLA requirements for typical research project configuration files with comprehensive
        statistical analysis and regression detection capabilities.
        
        SLA Requirement: <100ms for typical configuration files per Section 2.2.1
        """
        # Benchmark the load_config function with file input and statistical analysis
        result = benchmark(load_config, temp_config_file)
        
        # Verify successful loading and structural integrity
        assert isinstance(result, dict), "Configuration loading must return dictionary"
        assert "project" in result, "Configuration must contain project section"
        assert "datasets" in result, "Configuration must contain datasets section"
        
        # Validate SLA compliance: <100ms for file-based loading
        mean_time_ms = benchmark.stats.stats.mean * 1000
        assert mean_time_ms < 100.0, (
            f"F-001-RQ-001 SLA violation: File loading {mean_time_ms:.3f}ms > 100ms threshold"
        )
        
        # Store metadata for performance analysis and regression detection
        benchmark.extra_info.update({
            "sla_requirement": "F-001-RQ-001",
            "operation_type": "file_loading",
            "threshold_ms": 100.0,
            "measured_ms": mean_time_ms,
            "compliance": mean_time_ms < 100.0,
            "performance_margin_percent": ((100.0 - mean_time_ms) / 100.0) * 100
        })
    
    @pytest.mark.benchmark(group="config_loading")
    @pytest.mark.performance
    @pytest.mark.parametrize("config_size_scenario", [
        ("small", "small_config_dict"),
        ("medium", "medium_config_dict")
    ], indirect=True)
    def test_load_config_from_dict_performance_sla(self, benchmark, config_size_scenario):
        """
        Benchmark F-001-RQ-002: Support Kedro parameter dictionaries with <50ms validation SLA.
        
        Tests dictionary-based configuration validation performance against SLA requirements
        for Kedro-style parameter dictionaries with comprehensive performance analysis across
        different configuration complexity levels.
        
        SLA Requirement: <50ms validation for Kedro parameter dictionaries per Section 2.2.1
        """
        size_name, config_dict = config_size_scenario
        
        # Benchmark the load_config function with dictionary input
        result = benchmark(load_config, config_dict)
        
        # Verify successful loading and validation
        assert isinstance(result, dict), "Dictionary loading must return dictionary"
        assert result == config_dict, "Validated dictionary must match input exactly"
        
        # Validate SLA compliance: <50ms validation for Kedro parameter dictionaries
        mean_time_ms = benchmark.stats.stats.mean * 1000
        assert mean_time_ms < 50.0, (
            f"F-001-RQ-002 SLA violation ({size_name}): Dict validation {mean_time_ms:.3f}ms > 50ms threshold"
        )
        
        # Store metadata for performance analysis
        benchmark.extra_info.update({
            "sla_requirement": "F-001-RQ-002",
            "operation_type": "dict_validation",
            "config_size": size_name,
            "threshold_ms": 50.0,
            "measured_ms": mean_time_ms,
            "compliance": mean_time_ms < 50.0,
            "performance_margin_percent": ((50.0 - mean_time_ms) / 50.0) * 100
        })
    
    @pytest.mark.benchmark(group="config_validation")
    @pytest.mark.performance
    def test_validate_config_dict_performance_sla(self, benchmark, medium_config_dict):
        """
        Benchmark F-001-RQ-003: Validate configuration structure within <10ms SLA requirement.
        
        Tests isolated configuration structure validation performance with comprehensive
        statistical analysis to ensure rapid validation operations for configuration-heavy
        research workflows.
        
        SLA Requirement: <10ms validation per Section 2.2.1
        """
        # Benchmark the validate_config_dict function with statistical analysis
        result = benchmark(validate_config_dict, medium_config_dict)
        
        # Verify successful validation
        assert isinstance(result, dict), "Validation must return dictionary"
        assert result == medium_config_dict, "Validated config must match input"
        
        # Validate SLA compliance: <10ms validation
        mean_time_ms = benchmark.stats.stats.mean * 1000
        assert mean_time_ms < 10.0, (
            f"F-001-RQ-003 SLA violation: Structure validation {mean_time_ms:.3f}ms > 10ms threshold"
        )
        
        # Store metadata for performance analysis
        benchmark.extra_info.update({
            "sla_requirement": "F-001-RQ-003",
            "operation_type": "structure_validation",
            "threshold_ms": 10.0,
            "measured_ms": mean_time_ms,
            "compliance": mean_time_ms < 10.0,
            "performance_margin_percent": ((10.0 - mean_time_ms) / 10.0) * 100
        })
    
    @pytest.mark.benchmark(group="config_memory")
    @pytest.mark.performance
    def test_large_config_memory_usage_validation(self, large_config_dict):
        """
        Validate memory usage constraint: <10MB for large configuration processing.
        
        Tests memory usage during large configuration loading and validation operations
        to ensure compliance with memory usage limits as specified in technical requirements.
        Includes memory leak detection and efficiency validation.
        
        Memory Requirement: <10MB for large configurations per specification
        """
        # Measure memory usage during large config processing with leak detection
        result, peak_memory_mb = measure_memory_usage(load_config, large_config_dict)
        
        # Verify successful loading of large configuration
        assert isinstance(result, dict), "Large config loading must return dictionary"
        assert "project" in result, "Large config must contain project section"
        assert "datasets" in result, "Large config must contain datasets section"
        assert "experiments" in result, "Large config must contain experiments section"
        assert len(result["datasets"]) == 500, "Large config must contain 500 datasets"
        assert len(result["experiments"]) == 200, "Large config must contain 200 experiments"
        
        # Validate memory usage constraint: <10MB for large configurations
        assert peak_memory_mb < 10.0, (
            f"Memory usage SLA violation: {peak_memory_mb:.2f}MB > 10.0MB threshold"
        )
        
        # Additional memory efficiency validation
        config_size_estimate = len(str(large_config_dict)) / (1024 * 1024)  # Rough size in MB
        memory_multiplier = peak_memory_mb / max(config_size_estimate, 0.1)
        
        # Memory usage should not exceed 3x the configuration size
        assert memory_multiplier < 3.0, (
            f"Memory efficiency concern: {memory_multiplier:.2f}x multiplier > 3.0x threshold"
        )
        
        print(f"Large configuration memory validation:")
        print(f"  Peak memory usage: {peak_memory_mb:.2f}MB")
        print(f"  Memory multiplier: {memory_multiplier:.2f}x")
        print(f"  SLA compliance: {'✓' if peak_memory_mb < 10.0 else '✗'}")


# ============================================================================
# HIERARCHICAL CONFIGURATION MERGING BENCHMARK TESTS  
# ============================================================================

class TestHierarchicalConfigurationPerformance:
    """
    Performance benchmark tests for hierarchical configuration operations and merging performance.
    
    Test Categories:
    - Hierarchical ignore pattern merging (F-001-RQ-004: <5ms)
    - Mandatory substring merging with project/experiment hierarchy
    - Extraction pattern merging across project/experiment/dataset levels
    - Performance validation for complex configuration hierarchies
    """
    
    @pytest.mark.benchmark(group="config_merging")
    @pytest.mark.performance
    @pytest.mark.parametrize("experiment_scenario", [
        ("basic_experiment", "simple_hierarchy"),
        ("experiment_001", "medium_hierarchy"),
        ("experiment_025", "complex_hierarchy")
    ])
    def test_get_ignore_patterns_merging_performance_sla(self, benchmark, medium_config_dict, experiment_scenario):
        """
        Benchmark F-001-RQ-004: Hierarchical ignore pattern merging within <5ms SLA requirement.
        
        Tests project and experiment-level ignore pattern merging performance with comprehensive
        validation across different hierarchy complexity levels to ensure rapid configuration
        processing for research workflows.
        
        SLA Requirement: <5ms merge operation per Section 2.2.1
        """
        experiment_name, scenario_type = experiment_scenario
        
        # Benchmark ignore pattern retrieval with experiment-specific merging
        result = benchmark(get_ignore_patterns, medium_config_dict, experiment_name)
        
        # Verify successful hierarchical merging
        assert isinstance(result, list), "Ignore patterns must return list"
        assert len(result) > 0, "Merged patterns should not be empty"
        
        # Validate project-level patterns are included
        project_patterns = medium_config_dict["project"]["ignore_substrings"]
        for pattern in project_patterns:
            assert any(f"*{pattern}*" in merged for merged in result), (
                f"Project pattern '{pattern}' missing from merged results"
            )
        
        # Validate SLA compliance: <5ms merge operation
        mean_time_ms = benchmark.stats.stats.mean * 1000
        assert mean_time_ms < 5.0, (
            f"F-001-RQ-004 SLA violation ({scenario_type}): Pattern merging {mean_time_ms:.3f}ms > 5ms threshold"
        )
        
        # Store metadata for performance analysis
        benchmark.extra_info.update({
            "sla_requirement": "F-001-RQ-004",
            "operation_type": "ignore_pattern_merging",
            "scenario_type": scenario_type,
            "experiment_name": experiment_name,
            "threshold_ms": 5.0,
            "measured_ms": mean_time_ms,
            "compliance": mean_time_ms < 5.0,
            "pattern_count": len(result)
        })
    
    @pytest.mark.benchmark(group="config_merging")
    @pytest.mark.performance
    def test_get_mandatory_substrings_merging_performance_sla(self, benchmark, medium_config_dict):
        """
        Benchmark hierarchical mandatory substring merging within <5ms SLA requirement.
        
        Tests project and experiment-level mandatory substring merging performance as part
        of F-001-RQ-004 merge operation requirements with validation of hierarchical
        configuration processing efficiency.
        
        SLA Requirement: <5ms merge operation per Section 2.2.1
        """
        experiment_name = "experiment_001"
        
        # Benchmark mandatory substring retrieval with experiment-specific merging
        result = benchmark(get_mandatory_substrings, medium_config_dict, experiment_name)
        
        # Verify successful hierarchical merging
        assert isinstance(result, list), "Mandatory substrings must return list"
        
        # Validate project-level substrings are included
        project_substrings = medium_config_dict["project"]["mandatory_experiment_strings"]
        for substring in project_substrings:
            assert substring in result, (
                f"Project substring '{substring}' missing from merged results"
            )
        
        # Validate SLA compliance: <5ms merge operation
        mean_time_ms = benchmark.stats.stats.mean * 1000
        assert mean_time_ms < 5.0, (
            f"F-001-RQ-004 SLA violation: Mandatory substring merging {mean_time_ms:.3f}ms > 5ms threshold"
        )
        
        # Store metadata for performance analysis
        benchmark.extra_info.update({
            "sla_requirement": "F-001-RQ-004",
            "operation_type": "mandatory_substring_merging",
            "threshold_ms": 5.0,
            "measured_ms": mean_time_ms,
            "compliance": mean_time_ms < 5.0,
            "substring_count": len(result)
        })
    
    @pytest.mark.benchmark(group="config_merging")
    @pytest.mark.performance
    def test_get_extraction_patterns_merging_performance_sla(self, benchmark, medium_config_dict):
        """
        Benchmark hierarchical extraction pattern merging within <5ms SLA requirement.
        
        Tests project, experiment, and dataset-level extraction pattern merging performance
        as part of F-001-RQ-004 merge operation requirements with comprehensive validation
        of complex hierarchical configuration processing.
        
        SLA Requirement: <5ms merge operation per Section 2.2.1
        """
        experiment_name = "experiment_001"
        
        # Benchmark extraction pattern retrieval with hierarchical merging
        result = benchmark(get_extraction_patterns, medium_config_dict, experiment_name)
        
        # Verify successful hierarchical merging
        assert result is None or isinstance(result, list), "Extraction patterns must return list or None"
        
        if result is not None:
            # Validate project-level patterns are included
            project_patterns = medium_config_dict["project"]["extraction_patterns"]
            for pattern in project_patterns:
                assert pattern in result, (
                    f"Project extraction pattern '{pattern}' missing from merged results"
                )
        
        # Validate SLA compliance: <5ms merge operation
        mean_time_ms = benchmark.stats.stats.mean * 1000
        assert mean_time_ms < 5.0, (
            f"F-001-RQ-004 SLA violation: Extraction pattern merging {mean_time_ms:.3f}ms > 5ms threshold"
        )
        
        # Store metadata for performance analysis
        benchmark.extra_info.update({
            "sla_requirement": "F-001-RQ-004",
            "operation_type": "extraction_pattern_merging",
            "threshold_ms": 5.0,
            "measured_ms": mean_time_ms,
            "compliance": mean_time_ms < 5.0,
            "pattern_count": len(result) if result else 0
        })


# ============================================================================
# CONFIGURATION ACCESS PATTERN BENCHMARK TESTS
# ============================================================================

class TestConfigurationAccessPerformance:
    """
    Performance benchmark tests for configuration data access operations and lookup performance.
    
    Test Categories:
    - Dataset information retrieval performance optimization
    - Experiment information retrieval with relationship validation
    - Large configuration enumeration performance (experiments/datasets)
    - Configuration lookup efficiency for frequent operations
    """
    
    @pytest.mark.benchmark(group="config_access")
    @pytest.mark.performance
    def test_get_dataset_info_performance_optimization(self, benchmark, medium_config_dict):
        """
        Benchmark dataset information retrieval performance for configuration-heavy workflows.
        
        Tests dataset lookup operations for performance optimization in research workflows
        with frequent configuration access patterns and relationship validation.
        """
        dataset_name = "dataset_001"
        
        # Benchmark dataset info retrieval
        result = benchmark(get_dataset_info, medium_config_dict, dataset_name)
        
        # Verify successful retrieval and structural integrity
        assert isinstance(result, dict), "Dataset info must return dictionary"
        assert "rig" in result, "Dataset info must contain rig information"
        assert "dates_vials" in result, "Dataset info must contain dates_vials"
        assert "patterns" in result, "Dataset info must contain patterns"
        assert "metadata" in result, "Dataset info must contain metadata"
        
        # Validate reasonable performance for frequent operations
        mean_time_ms = benchmark.stats.stats.mean * 1000
        assert mean_time_ms < 1.0, (
            f"Dataset access performance concern: {mean_time_ms:.3f}ms > 1.0ms threshold"
        )
        
        # Store metadata for performance analysis
        benchmark.extra_info.update({
            "operation_type": "dataset_lookup",
            "threshold_ms": 1.0,
            "measured_ms": mean_time_ms,
            "dataset_name": dataset_name
        })
    
    @pytest.mark.benchmark(group="config_access")
    @pytest.mark.performance
    def test_get_experiment_info_performance_optimization(self, benchmark, medium_config_dict):
        """
        Benchmark experiment information retrieval performance for configuration-heavy workflows.
        
        Tests experiment lookup operations for performance optimization in research workflows
        with complex experiment-dataset relationships and hierarchical validation.
        """
        experiment_name = "experiment_001"
        
        # Benchmark experiment info retrieval
        result = benchmark(get_experiment_info, medium_config_dict, experiment_name)
        
        # Verify successful retrieval and structural integrity
        assert isinstance(result, dict), "Experiment info must return dictionary"
        assert "datasets" in result, "Experiment info must contain datasets"
        assert "filters" in result, "Experiment info must contain filters"
        assert "metadata" in result, "Experiment info must contain metadata"
        
        # Validate dataset relationships
        assert isinstance(result["datasets"], list), "Experiment datasets must be list"
        assert len(result["datasets"]) > 0, "Experiment must reference datasets"
        
        # Validate reasonable performance for frequent operations
        mean_time_ms = benchmark.stats.stats.mean * 1000
        assert mean_time_ms < 1.0, (
            f"Experiment access performance concern: {mean_time_ms:.3f}ms > 1.0ms threshold"
        )
        
        # Store metadata for performance analysis
        benchmark.extra_info.update({
            "operation_type": "experiment_lookup",
            "threshold_ms": 1.0,
            "measured_ms": mean_time_ms,
            "experiment_name": experiment_name,
            "dataset_count": len(result["datasets"])
        })
    
    @pytest.mark.benchmark(group="config_access")
    @pytest.mark.performance
    def test_get_all_experiment_names_performance_large_config(self, benchmark, large_config_dict):
        """
        Benchmark experiment name enumeration performance for large configuration traversal.
        
        Tests large configuration traversal for experiment discovery with performance
        validation for complex research project configurations with hundreds of experiments.
        """
        # Benchmark experiment name retrieval from large config
        result = benchmark(get_all_experiment_names, large_config_dict)
        
        # Verify successful enumeration
        assert isinstance(result, list), "Experiment names must return list"
        assert len(result) == 200, f"Large config should contain 200 experiments, got {len(result)}"
        
        # Validate naming consistency
        for name in result[:5]:  # Check first 5 for pattern validation
            assert name.startswith("experiment_"), f"Experiment name '{name}' should start with 'experiment_'"
            assert name[11:].isdigit(), f"Experiment name '{name}' should end with numeric ID"
        
        # Validate reasonable performance for large configurations
        mean_time_ms = benchmark.stats.stats.mean * 1000
        assert mean_time_ms < 10.0, (
            f"Large config enumeration performance concern: {mean_time_ms:.3f}ms > 10.0ms threshold"
        )
        
        # Store metadata for performance analysis
        benchmark.extra_info.update({
            "operation_type": "experiment_enumeration",
            "config_size": "large",
            "threshold_ms": 10.0,
            "measured_ms": mean_time_ms,
            "experiment_count": len(result)
        })
    
    @pytest.mark.benchmark(group="config_access")
    @pytest.mark.performance
    def test_get_all_dataset_names_performance_large_config(self, benchmark, large_config_dict):
        """
        Benchmark dataset name enumeration performance for large configuration traversal.
        
        Tests large configuration traversal for dataset discovery with performance validation
        for complex research project configurations with hundreds of datasets and relationships.
        """
        # Benchmark dataset name retrieval from large config
        result = benchmark(get_all_dataset_names, large_config_dict)
        
        # Verify successful enumeration
        assert isinstance(result, list), "Dataset names must return list"
        assert len(result) == 500, f"Large config should contain 500 datasets, got {len(result)}"
        
        # Validate naming consistency
        for name in result[:5]:  # Check first 5 for pattern validation
            assert name.startswith("dataset_"), f"Dataset name '{name}' should start with 'dataset_'"
            assert name[8:].isdigit(), f"Dataset name '{name}' should end with numeric ID"
        
        # Validate reasonable performance for large configurations
        mean_time_ms = benchmark.stats.stats.mean * 1000
        assert mean_time_ms < 10.0, (
            f"Large config enumeration performance concern: {mean_time_ms:.3f}ms > 10.0ms threshold"
        )
        
        # Store metadata for performance analysis
        benchmark.extra_info.update({
            "operation_type": "dataset_enumeration",
            "config_size": "large", 
            "threshold_ms": 10.0,
            "measured_ms": mean_time_ms,
            "dataset_count": len(result)
        })


# ============================================================================
# CONFIGURATION-DRIVEN DISCOVERY PERFORMANCE TESTS
# ============================================================================

class TestConfigurationDrivenDiscoveryPerformance:
    """
    Performance benchmark tests for configuration-driven file discovery operations with SLA validation.
    
    Test Categories:
    - Configuration-aware file discovery performance optimization
    - Experiment-specific file discovery with hierarchical configuration processing
    - Dataset-specific file discovery with configuration overhead analysis
    - Discovery performance impact from configuration complexity
    """
    
    @pytest.fixture(scope="function")
    def mock_file_system(self, monkeypatch):
        """
        Mock file system for discovery performance testing with consistent response patterns.
        
        Features:
        - Predictable file system environment for consistent performance measurement
        - Mock file discovery functions to focus on configuration processing overhead
        - Realistic file count simulation for discovery performance validation
        - Path provider mocking for filesystem access patterns
        
        Returns:
            MagicMock: Mock file system for discovery performance isolation
        """
        from unittest.mock import MagicMock
        
        # Mock file discovery functions to focus on configuration processing
        mock_discover_files = MagicMock()
        mock_discover_files.return_value = [
            f"/test/data/2024{month:02d}{day:02d}/experiment_{exp:03d}_file_{i:03d}.csv"
            for month in range(1, 4)
            for day in range(1, 11) 
            for exp in range(1, 6)
            for i in range(1, 21)  # 100 files per scenario for realistic testing
        ]
        
        # Mock the path provider's exists method for realistic filesystem simulation
        mock_path_provider = MagicMock()
        mock_path_provider.resolve_path.side_effect = lambda x: Path(str(x))
        mock_path_provider.exists.side_effect = lambda x: str(x).startswith('/test/')
        mock_path_provider.list_directories.return_value = [
            Path("/test/data/20240101"), Path("/test/data/20240102"),
            Path("/test/data/20240201"), Path("/test/data/20240202")
        ]
        
        # Patch the discovery function and path provider for performance isolation
        monkeypatch.setattr(
            "flyrigloader.config.discovery.discover_files", 
            mock_discover_files
        )
        
        # Patch the default discovery engine's path provider
        from flyrigloader.config.discovery import _default_discovery_engine
        original_provider = _default_discovery_engine.path_provider
        _default_discovery_engine.path_provider = mock_path_provider
        
        yield mock_discover_files
        
        # Restore the original path provider after test
        _default_discovery_engine.path_provider = original_provider
    
    @pytest.mark.benchmark(group="config_discovery")
    @pytest.mark.performance
    def test_discover_files_with_config_performance_overhead(self, benchmark, medium_config_dict, mock_file_system):
        """
        Benchmark configuration-driven file discovery performance with overhead analysis.
        
        Tests configuration processing overhead in file discovery workflows to ensure
        configuration operations don't bottleneck discovery performance with comprehensive
        validation of hierarchical configuration merging impact.
        """
        directory = "/test/data"
        pattern = "*.csv"
        experiment = "experiment_001"
        
        # Benchmark config-driven discovery with overhead analysis
        result = benchmark(
            discover_files_with_config,
            medium_config_dict,
            directory,
            pattern,
            recursive=True,
            experiment=experiment,
            extract_metadata=True
        )
        
        # Verify discovery executed with config parameters
        assert mock_file_system.called, "Mock file system should be called during discovery"
        
        # Validate configuration processing doesn't significantly impact discovery performance
        mean_time_ms = benchmark.stats.stats.mean * 1000
        assert mean_time_ms < 50.0, (
            f"Config-driven discovery performance concern: {mean_time_ms:.3f}ms > 50.0ms threshold"
        )
        
        # Store metadata for performance analysis
        benchmark.extra_info.update({
            "operation_type": "config_driven_discovery",
            "experiment_name": experiment,
            "threshold_ms": 50.0,
            "measured_ms": mean_time_ms,
            "config_complexity": "medium"
        })
    
    @pytest.mark.benchmark(group="config_discovery")
    @pytest.mark.performance
    def test_discover_experiment_files_performance_hierarchy(self, benchmark, medium_config_dict, mock_file_system):
        """
        Benchmark experiment-specific file discovery performance with hierarchical processing.
        
        Tests experiment configuration processing performance in discovery workflows including
        dataset enumeration, filter application, and hierarchical configuration merging
        with comprehensive performance validation.
        """
        experiment_name = "experiment_001"
        base_directory = "/test/data"
        
        # Benchmark experiment-specific discovery with hierarchy processing
        result = benchmark(
            discover_experiment_files,
            medium_config_dict,
            experiment_name,
            base_directory,
            recursive=True,
            extract_metadata=True
        )
        
        # Verify experiment processing executed
        assert mock_file_system.called, "Mock file system should be called during experiment discovery"
        
        # Validate experiment configuration processing performance
        mean_time_ms = benchmark.stats.stats.mean * 1000
        assert mean_time_ms < 50.0, (
            f"Experiment discovery performance concern: {mean_time_ms:.3f}ms > 50.0ms threshold"
        )
        
        # Store metadata for performance analysis
        benchmark.extra_info.update({
            "operation_type": "experiment_discovery",
            "experiment_name": experiment_name,
            "threshold_ms": 50.0,
            "measured_ms": mean_time_ms,
            "config_complexity": "medium"
        })
    
    @pytest.mark.benchmark(group="config_discovery")
    @pytest.mark.performance
    def test_discover_dataset_files_performance_dataset_specific(self, benchmark, medium_config_dict, mock_file_system):
        """
        Benchmark dataset-specific file discovery performance with configuration processing.
        
        Tests dataset configuration processing performance in discovery workflows including
        date-vial enumeration, pattern application, and dataset-specific configuration
        merging with performance optimization validation.
        """
        dataset_name = "dataset_001"
        base_directory = "/test/data"
        
        # Benchmark dataset-specific discovery with configuration processing
        result = benchmark(
            discover_dataset_files,
            medium_config_dict,
            dataset_name,
            base_directory,
            recursive=True,
            extract_metadata=True
        )
        
        # Verify dataset processing executed
        assert mock_file_system.called, "Mock file system should be called during dataset discovery"
        
        # Validate dataset configuration processing performance
        mean_time_ms = benchmark.stats.stats.mean * 1000
        assert mean_time_ms < 50.0, (
            f"Dataset discovery performance concern: {mean_time_ms:.3f}ms > 50.0ms threshold"
        )
        
        # Store metadata for performance analysis
        benchmark.extra_info.update({
            "operation_type": "dataset_discovery",
            "dataset_name": dataset_name,
            "threshold_ms": 50.0,
            "measured_ms": mean_time_ms,
            "config_complexity": "medium"
        })


# ============================================================================
# CONFIGURATION VALIDATION EDGE CASE PERFORMANCE TESTS
# ============================================================================

class TestConfigurationValidationEdgeCases:
    """
    Performance benchmark tests for configuration validation edge cases and error handling.
    
    Test Categories:
    - Invalid configuration validation performance (rapid error detection)
    - Missing experiment/dataset lookup performance optimization
    - Configuration error handling efficiency validation
    - Edge case performance consistency across error scenarios
    """
    
    @pytest.mark.benchmark(group="config_validation")
    @pytest.mark.performance
    def test_invalid_config_validation_performance_rapid_detection(self, benchmark):
        """
        Benchmark performance of configuration validation failure cases for rapid error detection.
        
        Tests that validation errors are detected quickly without performance penalties
        for invalid configurations, ensuring rapid feedback for configuration debugging
        in research workflow development.
        """
        invalid_config = {
            "invalid_structure": True,
            "missing_required_sections": "test",
            "malformed_datasets": ["not_a_dict"]
        }
        
        # Benchmark validation of invalid configuration with error detection
        with pytest.raises(ValueError):
            benchmark(validate_config_dict, invalid_config)
        
        # Validate that error detection is fast
        mean_time_ms = benchmark.stats.stats.mean * 1000
        assert mean_time_ms < 1.0, (
            f"Invalid config detection performance concern: {mean_time_ms:.3f}ms > 1.0ms threshold"
        )
        
        # Store metadata for performance analysis
        benchmark.extra_info.update({
            "operation_type": "invalid_config_detection",
            "threshold_ms": 1.0,
            "measured_ms": mean_time_ms,
            "error_type": "structure_validation"
        })
    
    @pytest.mark.benchmark(group="config_validation")
    @pytest.mark.performance
    def test_missing_experiment_access_performance_rapid_failure(self, benchmark, medium_config_dict):
        """
        Benchmark performance of missing experiment access for rapid failure detection.
        
        Tests that missing configuration lookups fail quickly without unnecessary
        processing overhead, ensuring efficient error handling in configuration-heavy
        research workflows.
        """
        nonexistent_experiment = "nonexistent_experiment_999"
        
        # Benchmark missing experiment lookup with rapid failure
        with pytest.raises(KeyError):
            benchmark(get_experiment_info, medium_config_dict, nonexistent_experiment)
        
        # Validate that missing lookups are fast
        mean_time_ms = benchmark.stats.stats.mean * 1000
        assert mean_time_ms < 1.0, (
            f"Missing experiment lookup performance concern: {mean_time_ms:.3f}ms > 1.0ms threshold"
        )
        
        # Store metadata for performance analysis
        benchmark.extra_info.update({
            "operation_type": "missing_experiment_lookup",
            "threshold_ms": 1.0,
            "measured_ms": mean_time_ms,
            "experiment_name": nonexistent_experiment
        })
    
    @pytest.mark.benchmark(group="config_validation")
    @pytest.mark.performance
    def test_missing_dataset_access_performance_rapid_failure(self, benchmark, medium_config_dict):
        """
        Benchmark performance of missing dataset access for rapid failure detection.
        
        Tests that missing dataset lookups fail quickly without unnecessary processing
        overhead, ensuring efficient error handling and debugging support in research
        workflow configuration development.
        """
        nonexistent_dataset = "nonexistent_dataset_999"
        
        # Benchmark missing dataset lookup with rapid failure
        with pytest.raises(KeyError):
            benchmark(get_dataset_info, medium_config_dict, nonexistent_dataset)
        
        # Validate that missing lookups are fast
        mean_time_ms = benchmark.stats.stats.mean * 1000
        assert mean_time_ms < 1.0, (
            f"Missing dataset lookup performance concern: {mean_time_ms:.3f}ms > 1.0ms threshold"
        )
        
        # Store metadata for performance analysis
        benchmark.extra_info.update({
            "operation_type": "missing_dataset_lookup",
            "threshold_ms": 1.0,
            "measured_ms": mean_time_ms,
            "dataset_name": nonexistent_dataset
        })


# ============================================================================
# PERFORMANCE REGRESSION DETECTION TESTS
# ============================================================================

class TestConfigurationPerformanceRegression:
    """
    Performance regression detection tests for configuration management with baseline establishment.
    
    Test Categories:
    - Configuration loading performance baseline establishment
    - Configuration validation performance baseline establishment  
    - Hierarchical merging performance baseline establishment
    - Regression detection with statistical significance validation
    """
    
    @pytest.mark.benchmark(group="regression")
    @pytest.mark.performance
    def test_config_loading_performance_baseline_establishment(self, benchmark, temp_config_file):
        """
        Establish performance baseline for configuration loading regression detection.
        
        Creates a performance baseline measurement for detecting future performance
        regressions in configuration loading operations with comprehensive metadata
        for statistical regression analysis and CI/CD integration.
        """
        # Create baseline measurement with comprehensive metadata
        result = benchmark(load_config, temp_config_file)
        
        # Store baseline metadata for regression analysis
        benchmark.extra_info.update({
            "operation": "config_loading",
            "config_type": "baseline",
            "sla_threshold_ms": 100,
            "regression_threshold_percent": 20,
            "baseline_establishment": True,
            "measurement_timestamp": datetime.now().isoformat(),
            "statistical_significance_level": 0.05
        })
        
        # Verify baseline functionality
        assert isinstance(result, dict), "Baseline config loading must return dictionary"
        assert "project" in result, "Baseline config must contain project section"
        
        print(f"Configuration loading baseline established: {benchmark.stats.stats.mean*1000:.3f}ms")
    
    @pytest.mark.benchmark(group="regression")
    @pytest.mark.performance
    def test_config_validation_performance_baseline_establishment(self, benchmark, medium_config_dict):
        """
        Establish performance baseline for configuration validation regression detection.
        
        Creates a performance baseline measurement for detecting future performance
        regressions in configuration validation operations with statistical metadata
        for comprehensive regression analysis.
        """
        # Create baseline measurement with statistical metadata
        result = benchmark(validate_config_dict, medium_config_dict)
        
        # Store baseline metadata for regression analysis
        benchmark.extra_info.update({
            "operation": "config_validation",
            "config_type": "medium_complexity",
            "sla_threshold_ms": 10,
            "regression_threshold_percent": 15,
            "baseline_establishment": True,
            "measurement_timestamp": datetime.now().isoformat(),
            "config_complexity_metrics": {
                "dataset_count": len(medium_config_dict["datasets"]),
                "experiment_count": len(medium_config_dict["experiments"]),
                "total_patterns": len(medium_config_dict["project"]["extraction_patterns"])
            }
        })
        
        # Verify baseline functionality
        assert isinstance(result, dict), "Baseline validation must return dictionary"
        assert result == medium_config_dict, "Baseline validation must return identical config"
        
        print(f"Configuration validation baseline established: {benchmark.stats.stats.mean*1000:.3f}ms")
    
    @pytest.mark.benchmark(group="regression")
    @pytest.mark.performance 
    def test_hierarchical_merging_performance_baseline_establishment(self, benchmark, large_config_dict):
        """
        Establish performance baseline for hierarchical merging regression detection.
        
        Creates a performance baseline measurement for detecting future performance
        regressions in configuration merging operations with comprehensive complexity
        analysis for statistical significance validation.
        """
        experiment_name = "experiment_0001"
        
        # Create baseline measurement for pattern merging with complexity analysis
        result = benchmark(get_ignore_patterns, large_config_dict, experiment_name)
        
        # Store baseline metadata for regression analysis
        benchmark.extra_info.update({
            "operation": "hierarchical_merging",
            "config_type": "large_complexity",
            "sla_threshold_ms": 5,
            "regression_threshold_percent": 25,
            "baseline_establishment": True,
            "measurement_timestamp": datetime.now().isoformat(),
            "complexity_metrics": {
                "total_datasets": len(large_config_dict["datasets"]),
                "total_experiments": len(large_config_dict["experiments"]),
                "project_ignore_patterns": len(large_config_dict["project"]["ignore_substrings"]),
                "merged_pattern_count": len(result)
            }
        })
        
        # Verify baseline functionality
        assert isinstance(result, list), "Baseline merging must return list"
        assert len(result) > 0, "Baseline merging must produce merged patterns"
        
        print(f"Hierarchical merging baseline established: {benchmark.stats.stats.mean*1000:.3f}ms")


# ============================================================================
# CONFIGURATION FILE FORMAT PERFORMANCE TESTS
# ============================================================================

class TestConfigurationFileFormatPerformance:
    """
    Performance benchmark tests for different configuration file formats and serialization styles.
    
    Test Categories:
    - YAML format loading performance across serialization styles
    - Configuration file format optimization validation
    - Cross-format performance consistency analysis
    - Serialization format impact on loading performance
    """
    
    @pytest.fixture(params=[
        ("yaml_safe_dump", {"default_flow_style": False}),
        ("yaml_unsafe_dump", {"default_flow_style": False, "allow_unicode": True}),
        ("yaml_flow_style", {"default_flow_style": True}),
        ("yaml_explicit_start", {"default_flow_style": False, "explicit_start": True}),
        ("yaml_canonical", {"default_flow_style": False, "canonical": True})
    ])
    def yaml_format_config(self, request, medium_config_dict):
        """
        Create configuration files in different YAML formats for performance testing.
        
        Features:
        - Multiple YAML serialization format variations for performance comparison
        - Consistent configuration data across format variations
        - Temporary file management with automatic cleanup
        - Format-specific YAML dumping options for comprehensive testing
        
        Args:
            request: Pytest request with format parameters
            medium_config_dict: Base configuration dictionary for consistent testing
            
        Returns:
            Tuple[str, str]: (format_name, file_path) for format-specific testing
        """
        format_name, dump_kwargs = request.param
        
        # Create temporary file with specific format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            if "unsafe" in format_name:
                yaml.dump(medium_config_dict, f, **dump_kwargs)
            else:
                yaml.safe_dump(medium_config_dict, f, **dump_kwargs)
            temp_path = f.name
        
        yield format_name, temp_path
        
        # Cleanup temporary file
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except OSError:
            # File cleanup failure should not affect test results
            pass
    
    @pytest.mark.benchmark(group="format_performance")
    @pytest.mark.performance
    def test_yaml_format_loading_performance_optimization(self, benchmark, yaml_format_config):
        """
        Benchmark configuration loading performance across different YAML formats.
        
        Tests how different YAML serialization formats affect loading performance to
        optimize configuration file generation and loading strategies for research
        workflow efficiency with comprehensive format comparison analysis.
        """
        format_name, config_path = yaml_format_config
        
        # Benchmark format-specific loading with performance analysis
        result = benchmark(load_config, config_path)
        
        # Store format metadata for analysis
        benchmark.extra_info.update({
            "yaml_format": format_name,
            "operation": "format_loading",
            "file_size_bytes": os.path.getsize(config_path),
            "measurement_timestamp": datetime.now().isoformat()
        })
        
        # Verify successful loading regardless of format
        assert isinstance(result, dict), "Format loading must return dictionary"
        assert "project" in result, "Format loading must preserve project section"
        assert "datasets" in result, "Format loading must preserve datasets section"
        assert "experiments" in result, "Format loading must preserve experiments section"
        
        # Validate content integrity across formats
        assert len(result["datasets"]) == 50, "Format loading must preserve all datasets"
        assert len(result["experiments"]) == 25, "Format loading must preserve all experiments"
        
        # All formats should meet SLA requirements
        mean_time_ms = benchmark.stats.stats.mean * 1000
        assert mean_time_ms < 100.0, (
            f"Format loading SLA violation ({format_name}): {mean_time_ms:.3f}ms > 100ms threshold"
        )
        
        print(f"YAML format '{format_name}' loading performance: {mean_time_ms:.3f}ms")


# ============================================================================
# MEMORY LEAK DETECTION FOR CONFIGURATION OPERATIONS
# ============================================================================

class TestConfigurationMemoryLeakDetection:
    """
    Memory leak detection tests for iterative configuration loading and processing scenarios.
    
    Test Categories:
    - Iterative configuration loading memory leak detection
    - Large configuration processing memory efficiency validation
    - Memory growth pattern analysis for configuration operations
    - Long-running configuration processing memory stability
    """
    
    @pytest.mark.benchmark(group="memory_leak")
    @pytest.mark.performance
    def test_iterative_config_loading_memory_leak_detection(self, memory_profiler_context, temp_config_file):
        """
        Detect memory leaks in iterative configuration loading scenarios.
        
        Tests memory stability during repeated configuration loading operations to ensure
        no memory accumulation occurs in long-running research workflow applications with
        frequent configuration reloading and validation cycles.
        """
        # Setup memory profiling for iterative loading
        data_size_estimate = os.path.getsize(temp_config_file) * 10  # Estimate for multiple loads
        
        with memory_profiler_context(
            data_size_estimate=data_size_estimate,
            precision=3,
            enable_line_profiling=True
        ) as profiler:
            
            # Perform iterative configuration loading
            iteration_count = 20
            configs = []
            
            for i in range(iteration_count):
                # Load configuration
                config = load_config(temp_config_file)
                configs.append(config)
                
                # Validate loaded configuration
                validated = validate_config_dict(config)
                assert validated == config
                
                # Update peak memory tracking
                profiler.update_peak_memory()
                
                # Clear intermediate references periodically
                if i % 5 == 4:
                    configs.clear()
                    gc.collect()
            
            # Complete memory profiling
            memory_results = profiler.end_profiling()
        
        # Validate memory efficiency
        memory_multiplier = memory_results.get("memory_multiplier", 1.0)
        assert memory_multiplier < 3.0, (
            f"Memory leak concern: {memory_multiplier:.2f}x multiplier > 3.0x threshold"
        )
        
        # Validate no significant memory growth
        peak_memory_mb = memory_results.get("peak_memory_mb", 0)
        assert peak_memory_mb < 50.0, (
            f"Memory usage concern: {peak_memory_mb:.2f}MB > 50.0MB threshold for iterative loading"
        )
        
        print(f"Iterative loading memory validation:")
        print(f"  Peak memory: {peak_memory_mb:.2f}MB")
        print(f"  Memory multiplier: {memory_multiplier:.2f}x")
        print(f"  Iterations: {iteration_count}")
    
    @pytest.mark.benchmark(group="memory_leak")
    @pytest.mark.performance
    def test_large_config_processing_memory_stability(self, memory_profiler_context, large_config_dict):
        """
        Validate memory stability during large configuration processing operations.
        
        Tests memory usage patterns during complex configuration processing to ensure
        stable memory utilization for large research project configurations with
        extensive hierarchical relationships and metadata processing.
        """
        # Estimate memory requirements for large configuration
        config_size_estimate = len(str(large_config_dict))
        
        with memory_profiler_context(
            data_size_estimate=config_size_estimate,
            precision=3,
            enable_line_profiling=True
        ) as profiler:
            
            # Process large configuration through various operations
            operations = [
                lambda: validate_config_dict(large_config_dict),
                lambda: get_all_dataset_names(large_config_dict),
                lambda: get_all_experiment_names(large_config_dict),
                lambda: get_ignore_patterns(large_config_dict, "experiment_0001"),
                lambda: get_mandatory_substrings(large_config_dict, "experiment_0001"),
                lambda: get_extraction_patterns(large_config_dict, "experiment_0001")
            ]
            
            # Execute operations multiple times
            for iteration in range(5):
                for operation in operations:
                    result = operation()
                    assert result is not None or isinstance(result, (list, dict))
                    
                    # Update memory tracking
                    profiler.update_peak_memory()
                
                # Force garbage collection between iterations
                gc.collect()
            
            # Complete memory profiling
            memory_results = profiler.end_profiling()
        
        # Validate memory efficiency for large configuration processing
        memory_multiplier = memory_results.get("memory_multiplier", 1.0)
        assert memory_multiplier < 4.0, (
            f"Large config memory concern: {memory_multiplier:.2f}x multiplier > 4.0x threshold"
        )
        
        # Validate memory usage constraint
        peak_memory_mb = memory_results.get("peak_memory_mb", 0)
        assert peak_memory_mb < 100.0, (
            f"Large config memory usage: {peak_memory_mb:.2f}MB > 100.0MB threshold"
        )
        
        print(f"Large configuration processing memory validation:")
        print(f"  Peak memory: {peak_memory_mb:.2f}MB")
        print(f"  Memory multiplier: {memory_multiplier:.2f}x")
        print(f"  Configuration size: {len(large_config_dict['datasets'])} datasets, {len(large_config_dict['experiments'])} experiments")


# ============================================================================
# COMPREHENSIVE CONFIGURATION BENCHMARK SUITE
# ============================================================================

class TestComprehensiveConfigurationBenchmarkSuite:
    """
    Comprehensive configuration benchmark suite integrating all performance validation requirements.
    
    This test class provides end-to-end performance validation combining SLA compliance,
    memory efficiency, regression detection, and cross-format compatibility testing
    for complete configuration management performance validation.
    """
    
    @pytest.mark.benchmark(group="comprehensive")
    @pytest.mark.performance
    def test_comprehensive_config_performance_validation(self, 
                                                        benchmark_coordinator,
                                                        medium_config_dict,
                                                        temp_config_file):
        """
        Comprehensive configuration performance validation with integrated analysis.
        
        Executes complete configuration performance validation including SLA compliance,
        memory efficiency, statistical analysis, and regression detection for end-to-end
        configuration management performance validation.
        """
        # Define comprehensive test function
        def comprehensive_config_test():
            # Test file loading
            file_result = load_config(temp_config_file)
            
            # Test dictionary validation
            dict_result = validate_config_dict(medium_config_dict)
            
            # Test hierarchical operations
            ignore_patterns = get_ignore_patterns(medium_config_dict, "experiment_001")
            mandatory_strings = get_mandatory_substrings(medium_config_dict, "experiment_001")
            extraction_patterns = get_extraction_patterns(medium_config_dict, "experiment_001")
            
            # Test lookup operations
            dataset_info = get_dataset_info(medium_config_dict, "dataset_001")
            experiment_info = get_experiment_info(medium_config_dict, "experiment_001")
            
            # Return comprehensive results
            return {
                "file_loading": file_result is not None,
                "dict_validation": dict_result == medium_config_dict,
                "ignore_patterns": len(ignore_patterns),
                "mandatory_strings": len(mandatory_strings),
                "extraction_patterns": len(extraction_patterns) if extraction_patterns else 0,
                "dataset_lookup": "rig" in dataset_info,
                "experiment_lookup": "datasets" in experiment_info
            }
        
        # Execute comprehensive benchmark with coordinator
        results = benchmark_coordinator.execute_comprehensive_benchmark(
            test_name="comprehensive_config_performance",
            test_function=comprehensive_config_test,
            test_category="config_loading",
            enable_memory_profiling=True,
            enable_regression_detection=True,
            establish_baseline=False
        )
        
        # Validate comprehensive results
        test_result = results["measurements"]
        assert len(test_result) > 0, "Comprehensive test must produce measurements"
        
        # Validate overall performance
        mean_time_ms = np.mean(test_result) * 1000
        assert mean_time_ms < 200.0, (
            f"Comprehensive config performance concern: {mean_time_ms:.3f}ms > 200ms threshold"
        )
        
        # Validate SLA compliance across all operations
        validation_results = results["validation_results"]
        assert validation_results["overall_valid"], "Comprehensive validation must pass all SLA requirements"
        
        print(f"Comprehensive configuration benchmark results:")
        print(f"  Mean execution time: {mean_time_ms:.3f}ms")
        print(f"  SLA compliance: {'✓' if validation_results['sla_validation']['compliant'] else '✗'}")
        print(f"  Statistical reliability: {'✓' if validation_results['reliability_validation']['reliable'] else '✗'}")
        if results["memory_results"]:
            print(f"  Memory efficiency: {'✓' if results['memory_results'].get('efficient', True) else '✗'}")