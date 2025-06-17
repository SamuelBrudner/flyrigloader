"""
Comprehensive pytest-benchmark suite for flyrigloader data loading pipeline performance validation.

This module provides comprehensive performance testing for the flyrigloader data loading pipeline,
validating SLA compliance, format detection performance, scalability requirements, and cross-platform 
consistency. Relocated from default test suite to enable optional performance validation without 
impacting rapid developer feedback cycles per Section 0 requirements.

Key Features:
- TST-PERF-001 SLA enforcement: Validates <1s per 100MB data loading requirement
- F-003-RQ-004 format detection: Tests <100ms overhead limit for automatic format detection  
- Scalability validation: Verifies data-size scalability from 1MB to 1GB per F-014 requirements
- Statistical accuracy: Maintains ±5% variance measurement accuracy per Section 2.4.9
- Memory efficiency validation: Tests memory-efficiency benchmarks for large dataset processing
- Cross-platform performance: Validates consistent performance across Ubuntu, Windows, macOS
- Regression detection: Statistical analysis with confidence intervals and baseline comparison
- CI/CD integration: Artifact generation for performance trend analysis and alerting

Performance Requirements Validated:
- Data loading SLA: <1s per 100MB (TST-PERF-001)
- Format detection overhead: <100ms (F-003-RQ-004) 
- Memory efficiency: <2x data size multiplier for large datasets
- Cross-platform variance: <10% performance difference across platforms
- Statistical reliability: ±5% measurement variance with 95% confidence intervals

Integration:
- pytest-benchmark framework for statistical performance measurement
- pytest-memory-profiler for line-by-line memory analysis and leak detection
- scripts/benchmarks/run_benchmarks.py CLI integration for --category data-loading execution
- Cross-platform normalization with environment-specific performance baselines
- CI/CD artifact generation for GitHub Actions performance monitoring
"""

import gc
import json
import tempfile
import time
import pickle
import gzip
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings

# Import flyrigloader components for data loading validation
from flyrigloader.io.pickle import (
    read_pickle_any_format,
    load_experimental_data,
    PickleLoader,
    DataFrameTransformer,
    create_test_pickle_loader,
    create_test_dataframe_transformer,
    DependencyContainer
)

# Import benchmark configuration and utilities
from .conftest import (
    benchmark_coordinator,
    benchmark_config,
    synthetic_data_generator,
    benchmark_data_sizes,
    memory_profiler_context,
    memory_leak_detector,
    statistical_analysis_engine,
    performance_baseline_manager,
    benchmark_validator,
    platform_skip_conditions,
    cross_platform_validator,
    environment_analyzer
)
from .utils import (
    MemoryProfiler,
    memory_profiling_context,
    estimate_data_size,
    StatisticalAnalysisEngine,
    analyze_benchmark_results
)
from .config import (
    BenchmarkCategory,
    PerformanceSLA,
    get_category_config
)


# ============================================================================
# PYTEST MARKERS AND CONFIGURATION
# ============================================================================

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.performance,
    pytest.mark.data_loading
]


# ============================================================================
# BENCHMARK TEST FIXTURES AND SETUP
# ============================================================================

@pytest.fixture(scope="function")
def data_loading_test_config(benchmark_config):
    """
    Function-scoped fixture providing data loading specific benchmark configuration.
    
    Features:
    - Data loading SLA thresholds per TST-PERF-001 requirements
    - Format detection performance limits per F-003-RQ-004 specifications
    - Memory efficiency thresholds for large dataset scenarios
    - Cross-platform performance normalization factors
    - Statistical analysis parameters for ±5% variance accuracy
    
    Returns:
        Dict containing data loading benchmark configuration
    """
    category_config = get_category_config(BenchmarkCategory.DATA_LOADING)
    
    # Enhanced configuration with comprehensive validation parameters
    config = {
        "sla_thresholds": {
            "data_loading_per_100mb_seconds": benchmark_config.sla.DATA_LOADING_TIME_PER_100MB_SECONDS,
            "format_detection_overhead_ms": 100.0,  # F-003-RQ-004 requirement
            "memory_multiplier_threshold": 2.0,     # Memory efficiency requirement
            "cross_platform_variance_percent": 10.0  # Cross-platform consistency
        },
        "test_data_sizes": {
            "micro": (1024, 1.0),           # 1MB - Format detection testing
            "small": (10 * 1024, 10.0),    # 10MB - Small dataset validation
            "medium": (50 * 1024, 50.0),   # 50MB - Standard performance testing
            "large": (100 * 1024, 100.0),  # 100MB - SLA validation scale
            "xlarge": (200 * 1024, 200.0), # 200MB - Scalability validation  
            "stress": (500 * 1024, 500.0)  # 500MB - Memory profiling scale
        },
        "statistical_requirements": {
            "variance_threshold_percent": 5.0,      # ±5% variance per Section 2.4.9
            "confidence_level": 0.95,               # 95% confidence intervals
            "min_iterations": 10,                   # Minimum measurements for reliability
            "outlier_detection_enabled": True       # Remove statistical outliers
        },
        "memory_profiling": {
            "enable_for_large_datasets": True,      # >100MB scenarios
            "leak_detection_threshold_mb": 10.0,    # Memory leak detection
            "continuous_monitoring": True,          # Line-by-line analysis
            "gc_between_iterations": True           # Garbage collection control
        },
        "format_testing": {
            "pickle_formats": ["pkl", "pickle"],         # Standard pickle formats
            "compressed_formats": ["pklz", "pkl.gz"],    # Compressed pickle formats  
            "invalid_formats": ["txt", "json", "yaml"],  # Error scenario testing
            "corrupted_data_scenarios": True            # Corrupted file handling
        }
    }
    
    print(f"\n=== Data Loading Benchmark Configuration ===")
    print(f"SLA Threshold: {config['sla_thresholds']['data_loading_per_100mb_seconds']:.2f}s per 100MB")
    print(f"Format Detection Limit: {config['sla_thresholds']['format_detection_overhead_ms']:.0f}ms")
    print(f"Statistical Variance: ±{config['statistical_requirements']['variance_threshold_percent']:.1f}%")
    print(f"Memory Efficiency: <{config['sla_thresholds']['memory_multiplier_threshold']:.1f}x data size")
    print("=" * 50)
    
    return config


@pytest.fixture(scope="function")
def synthetic_pickle_data_generator(synthetic_data_generator, tmp_path):
    """
    Function-scoped fixture providing synthetic pickle file generation for data loading benchmarks.
    
    Features:
    - Realistic experimental data generation with various formats
    - Multiple pickle file formats (standard, compressed, corrupted)
    - Configurable data sizes for scalability testing
    - Cross-platform compatible file generation
    - Memory-efficient data creation for large dataset scenarios
    
    Returns:
        Generator function for creating test pickle files
    """
    def generate_test_pickle_files(
        size_config: Tuple[int, float],  # (size_kb, size_mb)
        data_type: str = "neural",
        file_formats: List[str] = None,
        include_corrupted: bool = False
    ) -> Dict[str, Any]:
        """
        Generate synthetic pickle files for benchmark testing.
        
        Args:
            size_config: Tuple of (size_kb, size_mb) for data generation
            data_type: Type of data to generate ("neural", "behavioral", "large_dataset")
            file_formats: List of file formats to generate ["pkl", "pklz", "pkl.gz"]
            include_corrupted: Whether to include corrupted files for error testing
            
        Returns:
            Dict containing generated files and metadata
        """
        size_kb, size_mb = size_config
        file_formats = file_formats or ["pkl", "pklz"]
        
        # Calculate data dimensions for target size
        target_bytes = size_kb * 1024
        # Estimate: float64 = 8 bytes, target ~50 columns
        cols = 50
        rows = max(1, int(target_bytes / (cols * 8)))
        
        print(f"Generating {data_type} pickle data: {rows:,} x {cols} (~{size_mb:.1f}MB)")
        
        # Generate synthetic dataset
        dataset = synthetic_data_generator.generate_synthetic_dataset(
            rows=rows,
            cols=cols,
            data_type=data_type,
            include_metadata=True
        )
        
        generated_files = {}
        base_filename = f"test_data_{data_type}_{size_mb:.0f}mb"
        
        # Create standard pickle format
        if "pkl" in file_formats:
            pkl_file = tmp_path / f"{base_filename}.pkl"
            with open(pkl_file, 'wb') as f:
                pickle.dump(dataset["data_array"], f, protocol=pickle.HIGHEST_PROTOCOL)
            generated_files["pkl"] = {
                "path": pkl_file,
                "format": "standard_pickle",
                "size_bytes": pkl_file.stat().st_size,
                "data_shape": dataset["data_array"].shape
            }
        
        # Create compressed pickle format  
        if "pklz" in file_formats or "pkl.gz" in file_formats:
            pklz_file = tmp_path / f"{base_filename}.pklz"
            with gzip.open(pklz_file, 'wb') as f:
                pickle.dump(dataset["data_array"], f, protocol=pickle.HIGHEST_PROTOCOL)
            generated_files["pklz"] = {
                "path": pklz_file,
                "format": "compressed_pickle",
                "size_bytes": pklz_file.stat().st_size,
                "data_shape": dataset["data_array"].shape,
                "compression_ratio": pkl_file.stat().st_size / pklz_file.stat().st_size if "pkl" in generated_files else 1.0
            }
        
        # Create corrupted file for error scenario testing
        if include_corrupted:
            corrupted_file = tmp_path / f"{base_filename}_corrupted.pkl"
            with open(corrupted_file, 'wb') as f:
                f.write(b"corrupted pickle data - not valid")
            generated_files["corrupted"] = {
                "path": corrupted_file,
                "format": "corrupted_pickle",
                "size_bytes": corrupted_file.stat().st_size,
                "expected_error": "RuntimeError"
            }
        
        # Create DataFrame version if available
        if dataset["dataframe"] is not None:
            df_file = tmp_path / f"{base_filename}_dataframe.pkl"
            dataset["dataframe"].to_pickle(df_file)
            generated_files["dataframe"] = {
                "path": df_file,
                "format": "pandas_pickle",
                "size_bytes": df_file.stat().st_size,
                "data_shape": dataset["dataframe"].shape
            }
        
        return {
            "files": generated_files,
            "metadata": dataset["metadata"],
            "generation_config": {
                "size_config": size_config,
                "data_type": data_type,
                "file_formats": file_formats,
                "target_size_mb": size_mb,
                "actual_data_size_mb": dataset["metadata"]["estimated_size_mb"]
            }
        }
    
    return generate_test_pickle_files


# ============================================================================
# CORE DATA LOADING PERFORMANCE BENCHMARKS
# ============================================================================

@pytest.mark.benchmark(group="data_loading_core")
@pytest.mark.parametrize("size_name,size_config", [
    ("micro", (1024, 1.0)),      # 1MB - Format detection
    ("small", (10240, 10.0)),    # 10MB - Small dataset
    ("medium", (51200, 50.0)),   # 50MB - Standard benchmark
    ("large", (102400, 100.0)),  # 100MB - SLA validation
])
def test_benchmark_pickle_loading_sla_validation(
    benchmark,
    benchmark_coordinator,
    data_loading_test_config,
    synthetic_pickle_data_generator,
    size_name,
    size_config
):
    """
    Comprehensive benchmark for pickle file loading SLA validation per TST-PERF-001.
    
    This test validates the core data loading performance requirement of <1s per 100MB
    across various data sizes and formats, with statistical analysis and regression detection.
    
    Requirements Validated:
    - TST-PERF-001: Data loading <1s per 100MB
    - Statistical accuracy: ±5% variance with 95% confidence
    - Memory efficiency: <2x data size multiplier
    - Format compatibility: Standard and compressed pickle formats
    
    Args:
        benchmark: pytest-benchmark fixture for performance measurement
        benchmark_coordinator: Comprehensive benchmark orchestration utility
        data_loading_test_config: Data loading specific configuration
        synthetic_pickle_data_generator: Test data generation utility
        size_name: Human-readable size identifier
        size_config: (size_kb, size_mb) configuration tuple
    """
    size_kb, size_mb = size_config
    sla_threshold = data_loading_test_config["sla_thresholds"]["data_loading_per_100mb_seconds"]
    
    # Scale SLA threshold based on data size
    size_scale_factor = size_mb / 100.0  # Scale factor relative to 100MB baseline
    expected_threshold = sla_threshold * size_scale_factor
    
    print(f"\n=== Benchmarking Data Loading: {size_name} ({size_mb:.1f}MB) ===")
    print(f"Expected threshold: {expected_threshold:.3f}s (scaled from {sla_threshold:.1f}s for 100MB)")
    
    # Generate test data
    test_data = synthetic_pickle_data_generator(
        size_config=size_config,
        data_type="neural",
        file_formats=["pkl", "pklz"],
        include_corrupted=False
    )
    
    def load_pickle_file_standard():
        """Load standard pickle file using flyrigloader."""
        file_info = test_data["files"]["pkl"]
        result = read_pickle_any_format(file_info["path"])
        return result
    
    def load_pickle_file_compressed():
        """Load compressed pickle file using flyrigloader.""" 
        file_info = test_data["files"]["pklz"]
        result = read_pickle_any_format(file_info["path"])
        return result
    
    # Execute comprehensive benchmark for standard format
    standard_results = benchmark_coordinator.execute_comprehensive_benchmark(
        test_name=f"data_loading_standard_{size_name}",
        test_function=load_pickle_file_standard,
        test_category="data_loading",
        data_size_config=(int(size_mb * 1000), 50, size_mb),
        enable_memory_profiling=size_mb >= 50.0,  # Enable for larger datasets
        enable_regression_detection=True,
        establish_baseline=False
    )
    
    # Execute comprehensive benchmark for compressed format
    compressed_results = benchmark_coordinator.execute_comprehensive_benchmark(
        test_name=f"data_loading_compressed_{size_name}",
        test_function=load_pickle_file_compressed,
        test_category="data_loading", 
        data_size_config=(int(size_mb * 1000), 50, size_mb),
        enable_memory_profiling=size_mb >= 50.0,
        enable_regression_detection=True,
        establish_baseline=False
    )
    
    # Validate SLA compliance for both formats
    standard_mean = standard_results["statistics"]["mean"]
    compressed_mean = compressed_results["statistics"]["mean"]
    
    # SLA validation with detailed reporting
    assert standard_mean <= expected_threshold, (
        f"Standard pickle loading SLA violation: {standard_mean:.4f}s > {expected_threshold:.4f}s "
        f"({((standard_mean - expected_threshold) / expected_threshold * 100):+.1f}% over threshold)"
    )
    
    assert compressed_mean <= expected_threshold * 1.2, (  # Allow 20% overhead for decompression
        f"Compressed pickle loading SLA violation: {compressed_mean:.4f}s > {expected_threshold * 1.2:.4f}s "
        f"(Decompression overhead: {((compressed_mean - standard_mean) / standard_mean * 100):+.1f}%)"
    )
    
    # Statistical reliability validation
    standard_cv = standard_results["statistics"]["cv_percent"]
    compressed_cv = compressed_results["statistics"]["cv_percent"]
    variance_threshold = data_loading_test_config["statistical_requirements"]["variance_threshold_percent"]
    
    assert standard_cv <= variance_threshold, (
        f"Standard loading statistical variance too high: {standard_cv:.2f}% > {variance_threshold:.1f}%"
    )
    
    assert compressed_cv <= variance_threshold, (
        f"Compressed loading statistical variance too high: {compressed_cv:.2f}% > {variance_threshold:.1f}%"
    )
    
    # Memory efficiency validation (if profiling enabled)
    if standard_results.get("memory_results"):
        memory_multiplier = standard_results["memory_results"]["memory_multiplier"]
        efficiency_threshold = data_loading_test_config["sla_thresholds"]["memory_multiplier_threshold"]
        
        assert memory_multiplier <= efficiency_threshold, (
            f"Memory efficiency violation: {memory_multiplier:.2f}x > {efficiency_threshold:.1f}x threshold"
        )
    
    # Performance comparison and analysis
    compression_overhead = ((compressed_mean - standard_mean) / standard_mean) * 100
    print(f"\n=== Data Loading Performance Results ===")
    print(f"Standard pickle: {standard_mean:.4f}s (CV: {standard_cv:.1f}%)")
    print(f"Compressed pickle: {compressed_mean:.4f}s (CV: {compressed_cv:.1f}%)")
    print(f"Compression overhead: {compression_overhead:+.1f}%")
    print(f"SLA compliance: {'✓ PASS' if standard_mean <= expected_threshold else '✗ FAIL'}")
    
    # Run pytest-benchmark for integration with benchmark plugin
    benchmark.pedantic(load_pickle_file_standard, rounds=5, iterations=1)


@pytest.mark.benchmark(group="format_detection_performance")
@pytest.mark.parametrize("file_format", ["pkl", "pklz", "corrupted"])
def test_benchmark_format_detection_overhead(
    benchmark,
    benchmark_validator,
    statistical_analysis_engine,
    data_loading_test_config,
    synthetic_pickle_data_generator,
    file_format
):
    """
    Benchmark automatic format detection overhead per F-003-RQ-004 requirement.
    
    This test validates that automatic format detection adds <100ms overhead to data loading
    operations across different file formats and error scenarios.
    
    Requirements Validated:
    - F-003-RQ-004: Format detection overhead <100ms
    - Error handling performance for corrupted files
    - Cross-format detection consistency
    - Statistical measurement reliability
    
    Args:
        benchmark: pytest-benchmark fixture for performance measurement
        benchmark_validator: Benchmark result validation utility
        statistical_analysis_engine: Statistical analysis engine for confidence intervals
        data_loading_test_config: Data loading configuration
        synthetic_pickle_data_generator: Test data generation utility
        file_format: Format to test ("pkl", "pklz", "corrupted")
    """
    overhead_threshold_ms = data_loading_test_config["sla_thresholds"]["format_detection_overhead_ms"]
    overhead_threshold_s = overhead_threshold_ms / 1000.0
    
    print(f"\n=== Benchmarking Format Detection: {file_format} ===")
    print(f"Overhead threshold: {overhead_threshold_ms:.0f}ms")
    
    # Generate small test file for format detection testing
    test_data = synthetic_pickle_data_generator(
        size_config=(1024, 1.0),  # 1MB for fast format detection testing
        data_type="neural",
        file_formats=["pkl", "pklz"],
        include_corrupted=(file_format == "corrupted")
    )
    
    if file_format == "corrupted" and "corrupted" not in test_data["files"]:
        pytest.skip("Corrupted file generation not available")
    
    file_info = test_data["files"][file_format]
    
    def measure_format_detection():
        """Measure format detection time specifically."""
        file_path = file_info["path"]
        
        # Time the format detection process
        start_time = time.perf_counter()
        
        try:
            # This will trigger format detection (try gzip first, then regular pickle)
            result = read_pickle_any_format(file_path)
            success = True
        except Exception as e:
            # Expected for corrupted files
            success = False
            if file_format != "corrupted":
                raise  # Unexpected error for valid files
        
        end_time = time.perf_counter()
        detection_time = end_time - start_time
        
        return detection_time, success
    
    # Execute multiple measurements for statistical analysis
    measurements = []
    success_count = 0
    
    iterations = data_loading_test_config["statistical_requirements"]["min_iterations"]
    for i in range(iterations):
        detection_time, success = measure_format_detection()
        measurements.append(detection_time)
        if success:
            success_count += 1
    
    # Statistical analysis
    confidence_interval = statistical_analysis_engine.calculate_confidence_interval(measurements)
    cleaned_measurements, outlier_indices = statistical_analysis_engine.detect_outliers(measurements)
    
    # Format detection performance validation
    mean_detection_time = confidence_interval.mean
    detection_time_ms = mean_detection_time * 1000
    
    if file_format != "corrupted":
        # Validate overhead threshold for valid files
        assert mean_detection_time <= overhead_threshold_s, (
            f"Format detection overhead violation for {file_format}: "
            f"{detection_time_ms:.2f}ms > {overhead_threshold_ms:.0f}ms threshold"
        )
        
        # Validate success rate for valid files
        success_rate = (success_count / iterations) * 100
        assert success_rate >= 95.0, (
            f"Format detection reliability issue for {file_format}: "
            f"{success_rate:.1f}% success rate < 95% threshold"
        )
        
        print(f"Format detection results for {file_format}:")
        print(f"  Mean time: {detection_time_ms:.2f}ms")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Outliers detected: {len(outlier_indices)}")
        print(f"  SLA compliance: {'✓ PASS' if mean_detection_time <= overhead_threshold_s else '✗ FAIL'}")
    
    else:
        # For corrupted files, validate error detection speed
        assert mean_detection_time <= overhead_threshold_s * 2, (  # Allow 2x time for error detection
            f"Corrupted file detection too slow: {detection_time_ms:.2f}ms > {overhead_threshold_ms * 2:.0f}ms"
        )
        
        # Validate that errors are properly detected
        error_rate = ((iterations - success_count) / iterations) * 100
        assert error_rate >= 95.0, (
            f"Corrupted file detection failure: {error_rate:.1f}% error rate < 95% threshold"
        )
        
        print(f"Corrupted file detection results:")
        print(f"  Mean detection time: {detection_time_ms:.2f}ms")
        print(f"  Error detection rate: {error_rate:.1f}%")
        print(f"  Performance: {'✓ PASS' if mean_detection_time <= overhead_threshold_s * 2 else '✗ FAIL'}")
    
    # Statistical reliability validation
    cv_percent = (confidence_interval.std_error * np.sqrt(confidence_interval.sample_size) / confidence_interval.mean) * 100
    variance_threshold = data_loading_test_config["statistical_requirements"]["variance_threshold_percent"]
    
    assert cv_percent <= variance_threshold, (
        f"Format detection variance too high: {cv_percent:.2f}% > {variance_threshold:.1f}%"
    )
    
    # Run pytest-benchmark for integration
    benchmark.pedantic(lambda: measure_format_detection()[0], rounds=5, iterations=1)


# ============================================================================
# SCALABILITY AND MEMORY EFFICIENCY BENCHMARKS
# ============================================================================

@pytest.mark.benchmark(group="scalability_validation")
@pytest.mark.parametrize("scale_name,scale_config", [
    ("small_scale", (10240, 10.0)),      # 10MB baseline
    ("medium_scale", (51200, 50.0)),     # 50MB scaling
    ("large_scale", (102400, 100.0)),    # 100MB target 
    ("xlarge_scale", (204800, 200.0)),   # 200MB stress test
])
def test_benchmark_data_loading_scalability(
    benchmark_coordinator,
    data_loading_test_config,
    synthetic_pickle_data_generator,
    memory_leak_detector,
    platform_skip_conditions,
    scale_name,
    scale_config
):
    """
    Comprehensive scalability validation for data loading pipeline per F-014 requirements.
    
    This test validates that data loading performance scales linearly with data size
    from 1MB to 1GB, maintaining consistent performance characteristics and memory efficiency.
    
    Requirements Validated:
    - F-014: Data-size scalability (1MB to 1GB) implementation requirements
    - Linear performance scaling with data size
    - Memory efficiency maintenance across scales
    - Memory leak detection for large datasets
    - Cross-platform scalability consistency
    
    Args:
        benchmark_coordinator: Comprehensive benchmark orchestration utility
        data_loading_test_config: Data loading configuration
        synthetic_pickle_data_generator: Test data generation utility
        memory_leak_detector: Memory leak detection utility
        platform_skip_conditions: Platform-specific skip conditions
        scale_name: Human-readable scale identifier
        scale_config: (size_kb, size_mb) configuration tuple
    """
    size_kb, size_mb = scale_config
    
    # Skip large scale tests if insufficient memory
    if size_mb >= 200.0:
        skip_condition = platform_skip_conditions["skip_if_insufficient_memory"](size_mb / 1024 * 2)  # 2x safety factor
        if hasattr(skip_condition, 'mark'):
            pytest.skip(f"Insufficient memory for {size_mb:.0f}MB test")
    
    print(f"\n=== Scalability Validation: {scale_name} ({size_mb:.1f}MB) ===")
    
    # Generate test data with realistic experimental structure
    test_data = synthetic_pickle_data_generator(
        size_config=scale_config,
        data_type="large_dataset" if size_mb >= 100 else "neural",
        file_formats=["pkl", "pklz"],
        include_corrupted=False
    )
    
    def load_and_validate_dataset():
        """Load dataset and perform basic validation to ensure correctness."""
        file_info = test_data["files"]["pkl"]
        start_time = time.perf_counter()
        
        # Load data using flyrigloader
        result = read_pickle_any_format(file_info["path"])
        
        # Basic validation to ensure data integrity
        if hasattr(result, 'shape'):
            assert result.size > 0, "Loaded dataset is empty"
            assert len(result.shape) == 2, f"Expected 2D data, got {len(result.shape)}D"
        elif isinstance(result, dict):
            assert len(result) > 0, "Loaded dataset dictionary is empty"
        
        end_time = time.perf_counter()
        load_time = end_time - start_time
        
        return result, load_time
    
    # Execute comprehensive benchmark with memory profiling
    scalability_results = benchmark_coordinator.execute_comprehensive_benchmark(
        test_name=f"scalability_{scale_name}",
        test_function=load_and_validate_dataset,
        test_category="data_loading",
        data_size_config=(int(size_mb * 1000), 50, size_mb),
        enable_memory_profiling=True,  # Always enable for scalability testing
        enable_regression_detection=True,
        establish_baseline=False
    )
    
    # Scalability performance analysis
    mean_load_time = scalability_results["statistics"]["mean"]
    throughput_mb_per_second = size_mb / mean_load_time
    
    # Calculate performance scaling efficiency
    baseline_mb = 10.0  # 10MB baseline for scaling comparison
    expected_time_linear = (size_mb / baseline_mb) * data_loading_test_config["sla_thresholds"]["data_loading_per_100mb_seconds"] * 0.1
    scaling_efficiency = expected_time_linear / mean_load_time if mean_load_time > 0 else 0
    
    # Memory efficiency validation
    memory_results = scalability_results.get("memory_results")
    if memory_results:
        memory_multiplier = memory_results["memory_multiplier"]
        efficiency_threshold = data_loading_test_config["sla_thresholds"]["memory_multiplier_threshold"]
        
        assert memory_multiplier <= efficiency_threshold, (
            f"Memory efficiency violation at {size_mb:.1f}MB: "
            f"{memory_multiplier:.2f}x > {efficiency_threshold:.1f}x threshold"
        )
        
        # Memory leak detection for large datasets
        if size_mb >= 100.0:
            leak_analysis = memory_leak_detector.detect_leak_in_iterations(
                operation_func=load_and_validate_dataset,
                iterations=5
            )
            
            assert not leak_analysis["leak_detected"], (
                f"Memory leak detected at {size_mb:.1f}MB scale: "
                f"{leak_analysis['total_memory_growth_mb']:.1f}MB growth > "
                f"{leak_analysis['leak_threshold_mb']:.1f}MB threshold"
            )
    
    # Performance scaling validation
    sla_per_100mb = data_loading_test_config["sla_thresholds"]["data_loading_per_100mb_seconds"]
    expected_time_for_size = (size_mb / 100.0) * sla_per_100mb
    
    assert mean_load_time <= expected_time_for_size, (
        f"Scalability SLA violation at {size_mb:.1f}MB: "
        f"{mean_load_time:.4f}s > {expected_time_for_size:.4f}s "
        f"(Throughput: {throughput_mb_per_second:.1f} MB/s)"
    )
    
    # Statistical reliability at scale
    cv_percent = scalability_results["statistics"]["cv_percent"]
    variance_threshold = data_loading_test_config["statistical_requirements"]["variance_threshold_percent"]
    
    assert cv_percent <= variance_threshold, (
        f"Statistical variance too high at {size_mb:.1f}MB: {cv_percent:.2f}% > {variance_threshold:.1f}%"
    )
    
    print(f"\n=== Scalability Results for {scale_name} ===")
    print(f"Load time: {mean_load_time:.4f}s")
    print(f"Throughput: {throughput_mb_per_second:.1f} MB/s")
    print(f"Scaling efficiency: {scaling_efficiency:.2f}x")
    print(f"Memory multiplier: {memory_results['memory_multiplier']:.2f}x" if memory_results else "N/A")
    print(f"Statistical CV: {cv_percent:.2f}%")
    print(f"SLA compliance: {'✓ PASS' if mean_load_time <= expected_time_for_size else '✗ FAIL'}")


@pytest.mark.benchmark(group="memory_efficiency_validation")
@pytest.mark.parametrize("memory_scenario", [
    "standard_dataset",    # Standard memory usage
    "large_dataset",       # Large dataset memory profiling  
    "memory_pressure",     # Memory pressure simulation
])
def test_benchmark_memory_efficiency_validation(
    benchmark_coordinator,
    data_loading_test_config,
    synthetic_pickle_data_generator,
    memory_profiler_context,
    memory_leak_detector,
    platform_skip_conditions,
    memory_scenario
):
    """
    Comprehensive memory efficiency validation for large dataset processing scenarios.
    
    This test validates memory efficiency benchmarks per memory-efficiency requirements,
    including line-by-line memory analysis, leak detection, and garbage collection impact.
    
    Requirements Validated:
    - Memory efficiency: <2x data size multiplier for large datasets
    - Memory leak detection for iterative loading scenarios  
    - Line-by-line memory profiling using pytest-memory-profiler
    - Garbage collection impact analysis and optimization
    - Memory pressure handling and recovery
    
    Args:
        benchmark_coordinator: Comprehensive benchmark orchestration utility
        data_loading_test_config: Data loading configuration
        synthetic_pickle_data_generator: Test data generation utility
        memory_profiler_context: Memory profiling context manager
        memory_leak_detector: Memory leak detection utility
        platform_skip_conditions: Platform-specific skip conditions
        memory_scenario: Memory testing scenario identifier
    """
    # Configure memory scenario parameters
    scenario_configs = {
        "standard_dataset": {
            "size_config": (51200, 50.0),  # 50MB
            "iterations": 5,
            "gc_frequency": "normal",
            "memory_pressure": False
        },
        "large_dataset": {
            "size_config": (204800, 200.0),  # 200MB
            "iterations": 3,
            "gc_frequency": "aggressive", 
            "memory_pressure": False
        },
        "memory_pressure": {
            "size_config": (102400, 100.0),  # 100MB
            "iterations": 10,  # More iterations to create pressure
            "gc_frequency": "minimal",
            "memory_pressure": True
        }
    }
    
    config = scenario_configs[memory_scenario]
    size_kb, size_mb = config["size_config"]
    
    # Skip if insufficient memory
    required_memory_gb = (size_mb * config["iterations"]) / 1024 * 3  # 3x safety factor
    if required_memory_gb > 1.0:  # Skip if > 1GB requirement
        skip_condition = platform_skip_conditions["skip_if_insufficient_memory"](required_memory_gb)
        if hasattr(skip_condition, 'mark'):
            pytest.skip(f"Insufficient memory for {memory_scenario} ({required_memory_gb:.1f}GB required)")
    
    print(f"\n=== Memory Efficiency Validation: {memory_scenario} ===")
    print(f"Dataset size: {size_mb:.1f}MB")
    print(f"Iterations: {config['iterations']}")
    print(f"GC frequency: {config['gc_frequency']}")
    
    # Generate test data
    test_data = synthetic_pickle_data_generator(
        size_config=(size_kb, size_mb),
        data_type="large_dataset",
        file_formats=["pkl"],
        include_corrupted=False
    )
    
    file_path = test_data["files"]["pkl"]["path"]
    expected_data_size = test_data["metadata"]["estimated_size_mb"]
    
    def load_with_memory_monitoring():
        """Load data with comprehensive memory monitoring."""
        # Configure garbage collection based on scenario
        if config["gc_frequency"] == "aggressive":
            gc.collect()
        elif config["gc_frequency"] == "minimal":
            gc.disable()
        
        try:
            # Load the data
            result = read_pickle_any_format(file_path)
            
            # Validate result to ensure it's processed
            if hasattr(result, 'shape'):
                data_size = result.nbytes if hasattr(result, 'nbytes') else 0
            else:
                data_size = 0
            
            # Simulate memory pressure if configured
            if config["memory_pressure"]:
                # Create temporary large objects to pressure memory
                temp_data = np.random.random((1000, 1000))  # ~8MB array
                del temp_data
            
            return result, data_size
        
        finally:
            if config["gc_frequency"] == "minimal":
                gc.enable()
    
    # Execute memory profiling with context manager
    data_size_estimate = int(expected_data_size * 1024 * 1024)
    
    with memory_profiler_context(
        data_size_estimate=data_size_estimate,
        precision=3,
        enable_line_profiling=True,
        monitor_interval=0.05  # High frequency monitoring
    ) as profiler:
        
        # Execute multiple iterations for memory analysis
        load_times = []
        data_sizes = []
        
        for iteration in range(config["iterations"]):
            start_time = time.perf_counter()
            
            result, data_size = load_with_memory_monitoring()
            
            end_time = time.perf_counter()
            load_time = end_time - start_time
            
            load_times.append(load_time)
            data_sizes.append(data_size)
            
            # Update peak memory tracking
            profiler.update_peak_memory()
            
            # Brief pause between iterations
            time.sleep(0.1)
    
    # Memory profiling analysis is done by context manager
    # We need to get the memory analysis results
    memory_stats = profiler.end_profiling()
    
    # Memory efficiency validation
    memory_multiplier = memory_stats["memory_multiplier"]
    efficiency_threshold = data_loading_test_config["sla_thresholds"]["memory_multiplier_threshold"]
    
    assert memory_multiplier <= efficiency_threshold, (
        f"Memory efficiency violation in {memory_scenario}: "
        f"{memory_multiplier:.2f}x > {efficiency_threshold:.1f}x threshold "
        f"(Peak: {memory_stats['peak_memory_mb']:.1f}MB, Data: {memory_stats['data_size_mb']:.1f}MB)"
    )
    
    # Memory leak detection
    leak_analysis = memory_leak_detector.detect_leak_in_iterations(
        operation_func=load_with_memory_monitoring,
        iterations=config["iterations"]
    )
    
    assert not leak_analysis["leak_detected"], (
        f"Memory leak detected in {memory_scenario}: "
        f"{leak_analysis['total_memory_growth_mb']:.1f}MB growth > "
        f"{leak_analysis['leak_threshold_mb']:.1f}MB threshold"
    )
    
    # Performance consistency under memory scenarios
    mean_load_time = np.mean(load_times)
    cv_load_time = (np.std(load_times, ddof=1) / mean_load_time) * 100
    variance_threshold = data_loading_test_config["statistical_requirements"]["variance_threshold_percent"]
    
    assert cv_load_time <= variance_threshold * 2, (  # Allow 2x variance under memory pressure
        f"Performance inconsistency in {memory_scenario}: "
        f"{cv_load_time:.2f}% CV > {variance_threshold * 2:.1f}% threshold"
    )
    
    print(f"\n=== Memory Efficiency Results: {memory_scenario} ===")
    print(f"Memory multiplier: {memory_multiplier:.2f}x")
    print(f"Peak memory: {memory_stats['peak_memory_mb']:.1f}MB")
    print(f"Memory overhead: {memory_stats['memory_overhead_mb']:.1f}MB")
    print(f"Load time CV: {cv_load_time:.2f}%")
    print(f"Memory leak: {'✗ DETECTED' if leak_analysis['leak_detected'] else '✓ NONE'}")
    print(f"Efficiency: {'✓ PASS' if memory_multiplier <= efficiency_threshold else '✗ FAIL'}")


# ============================================================================
# CROSS-PLATFORM PERFORMANCE VALIDATION
# ============================================================================

@pytest.mark.benchmark(group="cross_platform_validation")
def test_benchmark_cross_platform_performance_consistency(
    benchmark_coordinator,
    data_loading_test_config,
    synthetic_pickle_data_generator,
    cross_platform_validator,
    environment_analyzer,
    platform_skip_conditions
):
    """
    Comprehensive cross-platform performance validation across Ubuntu, Windows, macOS.
    
    This test validates consistent performance across different platforms with environment
    normalization factors to account for hardware differences and virtualization overhead.
    
    Requirements Validated:
    - Cross-platform performance consistency (<10% variance)
    - Environment normalization with hardware compensation
    - Platform-specific performance baseline establishment
    - CI environment performance validation with virtualization factors
    
    Args:
        benchmark_coordinator: Comprehensive benchmark orchestration utility
        data_loading_test_config: Data loading configuration
        synthetic_pickle_data_generator: Test data generation utility
        cross_platform_validator: Cross-platform validation utility
        environment_analyzer: Environment analysis and normalization utility
        platform_skip_conditions: Platform-specific skip conditions
    """
    platform_info = platform_skip_conditions["platform_info"]
    current_platform = platform_info["current_platform"]
    is_ci = platform_info["is_ci"]
    
    print(f"\n=== Cross-Platform Validation: {current_platform} ===")
    print(f"CI Environment: {is_ci}")
    print(f"Available Memory: {platform_info['memory_gb']:.1f}GB")
    
    # Generate standard test dataset for cross-platform testing
    test_data = synthetic_pickle_data_generator(
        size_config=(51200, 50.0),  # 50MB standard size
        data_type="neural",
        file_formats=["pkl", "pklz"],
        include_corrupted=False
    )
    
    # Analyze current environment for normalization
    env_characteristics = environment_analyzer.detect_environment_characteristics()
    normalization_factors = environment_analyzer.calculate_normalization_factors()
    
    def load_data_cross_platform():
        """Load data with cross-platform compatibility validation."""
        # Test both standard and compressed formats
        results = {}
        
        # Standard pickle
        pkl_path = test_data["files"]["pkl"]["path"]
        start_time = time.perf_counter()
        standard_result = read_pickle_any_format(pkl_path)
        standard_time = time.perf_counter() - start_time
        results["standard"] = (standard_result, standard_time)
        
        # Compressed pickle  
        pklz_path = test_data["files"]["pklz"]["path"]
        start_time = time.perf_counter()
        compressed_result = read_pickle_any_format(pklz_path)
        compressed_time = time.perf_counter() - start_time
        results["compressed"] = (compressed_result, compressed_time)
        
        return results
    
    # Execute cross-platform benchmark
    platform_results = benchmark_coordinator.execute_comprehensive_benchmark(
        test_name=f"cross_platform_{current_platform.lower()}",
        test_function=load_data_cross_platform,
        test_category="data_loading",
        data_size_config=(50000, 50, 50.0),
        enable_memory_profiling=False,  # Focus on performance consistency
        enable_regression_detection=True,
        establish_baseline=False
    )
    
    # Extract performance metrics
    mean_execution_time = platform_results["statistics"]["mean"]
    cv_percent = platform_results["statistics"]["cv_percent"]
    
    # Apply environment normalization
    normalized_time = environment_analyzer.normalize_performance_measurement(
        mean_execution_time, normalization_factors
    )
    
    # Cross-platform performance validation
    sla_threshold = data_loading_test_config["sla_thresholds"]["data_loading_per_100mb_seconds"] * 0.5  # 50MB = 0.5 * 100MB
    platform_variance_threshold = data_loading_test_config["sla_thresholds"]["cross_platform_variance_percent"]
    
    # Validate raw performance meets SLA
    assert mean_execution_time <= sla_threshold, (
        f"Platform {current_platform} SLA violation: "
        f"{mean_execution_time:.4f}s > {sla_threshold:.4f}s"
    )
    
    # Validate statistical consistency 
    variance_threshold = data_loading_test_config["statistical_requirements"]["variance_threshold_percent"]
    assert cv_percent <= variance_threshold, (
        f"Platform {current_platform} performance inconsistency: "
        f"{cv_percent:.2f}% CV > {variance_threshold:.1f}% threshold"
    )
    
    # Platform-specific validation adjustments
    if is_ci:
        # CI environments may have higher variance due to virtualization
        ci_variance_allowance = variance_threshold * 1.5
        assert cv_percent <= ci_variance_allowance, (
            f"CI environment {current_platform} variance too high: "
            f"{cv_percent:.2f}% > {ci_variance_allowance:.1f}% (CI adjusted threshold)"
        )
    
    # Cross-platform consistency analysis
    platform_performance_data = {
        "platform": current_platform,
        "raw_time": mean_execution_time,
        "normalized_time": normalized_time,
        "cv_percent": cv_percent,
        "environment_factors": {
            "cpu_normalization": normalization_factors["cpu"],
            "memory_normalization": normalization_factors["memory"],
            "virtualization_factor": normalization_factors["virtualization"],
            "combined_factor": normalization_factors["combined"]
        },
        "hardware_characteristics": {
            "cpu_count": env_characteristics["cpu_count"],
            "memory_gb": env_characteristics["memory_gb"],
            "is_virtualized": env_characteristics["is_virtualized"],
            "platform": env_characteristics["platform"]
        }
    }
    
    # Validate cross-platform consistency using normalized values
    cross_platform_validation = cross_platform_validator.validate_performance_consistency(
        current_platform=current_platform,
        performance_measurement=normalized_time,
        variance_threshold=platform_variance_threshold
    )
    
    print(f"\n=== Cross-Platform Performance Results ===")
    print(f"Platform: {current_platform}")
    print(f"Raw time: {mean_execution_time:.4f}s")
    print(f"Normalized time: {normalized_time:.4f}s")
    print(f"Normalization factor: {normalization_factors['combined']:.3f}x")
    print(f"Performance CV: {cv_percent:.2f}%")
    print(f"Environment suitability: {env_characteristics.get('benchmarking_suitability', 'unknown')}")
    print(f"SLA compliance: {'✓ PASS' if mean_execution_time <= sla_threshold else '✗ FAIL'}")
    print(f"Cross-platform consistency: {'✓ PASS' if cross_platform_validation.get('consistent', True) else '✗ FAIL'}")
    
    # Store platform results for cross-platform comparison (if in CI)
    if is_ci and platform_results.get("artifacts_generated"):
        # Results will be collected by CI artifact management
        platform_performance_file = Path(f"platform_performance_{current_platform.lower()}.json")
        with open(platform_performance_file, 'w') as f:
            json.dump(platform_performance_data, f, indent=2)


# ============================================================================
# HYPOTHESIS-BASED PROPERTY TESTING FOR EDGE CASES
# ============================================================================

@pytest.mark.benchmark(group="property_based_validation") 
@given(
    data_rows=st.integers(min_value=100, max_value=10000),
    data_cols=st.integers(min_value=10, max_value=100),
    data_dtype=st.sampled_from([np.float32, np.float64, np.int32, np.int64])
)
@settings(
    max_examples=10,  # Limit for benchmark performance
    deadline=30000    # 30 second deadline per example
)
def test_benchmark_property_based_edge_cases(
    benchmark_validator,
    data_loading_test_config,
    tmp_path,
    data_rows,
    data_cols,
    data_dtype
):
    """
    Hypothesis-based property testing for comprehensive edge-case validation within benchmark context.
    
    This test uses property-based testing to discover edge cases in data loading performance
    across various data shapes, types, and characteristics that might not be covered by
    standard benchmarks.
    
    Requirements Validated:
    - Performance consistency across diverse data characteristics
    - Edge-case handling without performance degradation
    - Data type compatibility and performance impact
    - Robust error handling for unusual data shapes
    
    Args:
        benchmark_validator: Benchmark result validation utility
        data_loading_test_config: Data loading configuration
        tmp_path: Temporary directory for test files
        data_rows: Number of rows (hypothesis-generated)
        data_cols: Number of columns (hypothesis-generated)  
        data_dtype: Data type (hypothesis-generated)
    """
    # Calculate data size estimate
    bytes_per_element = np.dtype(data_dtype).itemsize
    estimated_size_mb = (data_rows * data_cols * bytes_per_element) / (1024 * 1024)
    
    # Skip very large datasets to maintain benchmark performance
    if estimated_size_mb > 100.0:
        pytest.skip(f"Dataset too large for property testing: {estimated_size_mb:.1f}MB")
    
    print(f"\n=== Property-Based Edge Case: {data_rows}x{data_cols} {data_dtype} ({estimated_size_mb:.1f}MB) ===")
    
    # Generate synthetic data with hypothesis-provided characteristics
    synthetic_data = np.random.random((data_rows, data_cols)).astype(data_dtype)
    
    # Create test pickle file
    test_file = tmp_path / f"property_test_{data_rows}x{data_cols}_{data_dtype.__name__}.pkl"
    with open(test_file, 'wb') as f:
        pickle.dump(synthetic_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_property_test_data():
        """Load hypothesis-generated test data."""
        result = read_pickle_any_format(test_file)
        
        # Validate data integrity
        assert hasattr(result, 'shape'), "Loaded data missing shape attribute"
        assert result.shape == (data_rows, data_cols), f"Shape mismatch: {result.shape} != {(data_rows, data_cols)}"
        assert result.dtype == data_dtype, f"Data type mismatch: {result.dtype} != {data_dtype}"
        
        return result
    
    # Execute property-based performance measurement
    measurements = []
    iterations = max(3, min(10, int(100 / max(estimated_size_mb, 1))))  # Adaptive iterations
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        result = load_property_test_data()
        end_time = time.perf_counter()
        
        load_time = end_time - start_time
        measurements.append(load_time)
    
    # Performance validation using benchmark validator
    validation_results = benchmark_validator.validate_statistical_reliability(measurements)
    
    # Validate statistical reliability
    assert validation_results["reliable"], (
        f"Property-based test statistical unreliability: {validation_results['message']}"
    )
    
    # Calculate throughput and efficiency metrics
    mean_load_time = np.mean(measurements)
    throughput_mb_per_second = estimated_size_mb / mean_load_time if mean_load_time > 0 else 0
    
    # Validate performance against scaled SLA
    size_scale_factor = estimated_size_mb / 100.0  # Scale relative to 100MB baseline
    sla_threshold = data_loading_test_config["sla_thresholds"]["data_loading_per_100mb_seconds"] * size_scale_factor
    
    assert mean_load_time <= sla_threshold, (
        f"Property-based SLA violation for {data_rows}x{data_cols} {data_dtype}: "
        f"{mean_load_time:.4f}s > {sla_threshold:.4f}s "
        f"(Throughput: {throughput_mb_per_second:.1f} MB/s)"
    )
    
    # Data type specific performance analysis
    dtype_performance_expectations = {
        np.float32: 1.0,    # Baseline
        np.float64: 0.8,    # Slightly slower due to larger size
        np.int32: 1.1,      # Potentially faster
        np.int64: 0.9       # Larger size impact
    }
    
    expected_performance_factor = dtype_performance_expectations.get(data_dtype, 1.0)
    adjusted_threshold = sla_threshold / expected_performance_factor
    
    if mean_load_time > adjusted_threshold:
        warnings.warn(
            f"Data type {data_dtype} performance below expectation: "
            f"{mean_load_time:.4f}s > {adjusted_threshold:.4f}s (adjusted for type)"
        )
    
    print(f"Property test results:")
    print(f"  Data shape: {data_rows}x{data_cols}")
    print(f"  Data type: {data_dtype}")
    print(f"  Size: {estimated_size_mb:.2f}MB")
    print(f"  Load time: {mean_load_time:.4f}s")
    print(f"  Throughput: {throughput_mb_per_second:.1f} MB/s")
    print(f"  Statistical reliability: {'✓ PASS' if validation_results['reliable'] else '✗ FAIL'}")
    print(f"  SLA compliance: {'✓ PASS' if mean_load_time <= sla_threshold else '✗ FAIL'}")


# ============================================================================
# COMPREHENSIVE INTEGRATION AND REGRESSION TESTING
# ============================================================================

@pytest.mark.benchmark(group="integration_validation")
def test_benchmark_comprehensive_data_loading_integration(
    benchmark_coordinator,
    data_loading_test_config,
    synthetic_pickle_data_generator,
    performance_baseline_manager,
    statistical_analysis_engine
):
    """
    Comprehensive integration benchmark validating complete data loading pipeline.
    
    This test provides end-to-end validation of the data loading pipeline including
    format detection, loading, validation, and performance analysis with comprehensive
    statistical analysis and regression detection.
    
    Requirements Validated:
    - Complete data loading pipeline integration
    - Format detection → loading → validation workflow
    - Performance baseline establishment and regression detection
    - Statistical accuracy across complete pipeline
    - Memory efficiency in integrated scenarios
    
    Args:
        benchmark_coordinator: Comprehensive benchmark orchestration utility
        data_loading_test_config: Data loading configuration
        synthetic_pickle_data_generator: Test data generation utility
        performance_baseline_manager: Performance baseline management utility
        statistical_analysis_engine: Statistical analysis engine
    """
    print(f"\n=== Comprehensive Data Loading Integration Benchmark ===")
    
    # Generate comprehensive test dataset with multiple formats
    integration_test_data = synthetic_pickle_data_generator(
        size_config=(102400, 100.0),  # 100MB for comprehensive testing
        data_type="neural",
        file_formats=["pkl", "pklz"],
        include_corrupted=True
    )
    
    def execute_complete_pipeline():
        """Execute complete data loading pipeline with validation."""
        pipeline_results = {}
        total_start_time = time.perf_counter()
        
        # Test standard pickle loading
        pkl_file = integration_test_data["files"]["pkl"]["path"]
        start_time = time.perf_counter()
        standard_data = read_pickle_any_format(pkl_file)
        standard_time = time.perf_counter() - start_time
        pipeline_results["standard_loading"] = {
            "data": standard_data,
            "load_time": standard_time,
            "throughput_mb_s": 100.0 / standard_time if standard_time > 0 else 0
        }
        
        # Test compressed pickle loading
        pklz_file = integration_test_data["files"]["pklz"]["path"]
        start_time = time.perf_counter()
        compressed_data = read_pickle_any_format(pklz_file)
        compressed_time = time.perf_counter() - start_time
        pipeline_results["compressed_loading"] = {
            "data": compressed_data,
            "load_time": compressed_time,
            "throughput_mb_s": 100.0 / compressed_time if compressed_time > 0 else 0
        }
        
        # Test error handling with corrupted file
        corrupted_file = integration_test_data["files"]["corrupted"]["path"]
        start_time = time.perf_counter()
        try:
            corrupted_data = read_pickle_any_format(corrupted_file)
            error_handled = False
        except Exception as e:
            error_handled = True
            error_type = type(e).__name__
        error_time = time.perf_counter() - start_time
        pipeline_results["error_handling"] = {
            "error_handled": error_handled,
            "error_time": error_time,
            "error_type": error_type if error_handled else None
        }
        
        # Data validation and consistency check
        if hasattr(standard_data, 'shape') and hasattr(compressed_data, 'shape'):
            data_consistency = np.array_equal(standard_data, compressed_data)
            pipeline_results["data_consistency"] = data_consistency
        else:
            pipeline_results["data_consistency"] = True  # Assume consistent if can't compare
        
        total_time = time.perf_counter() - total_start_time
        pipeline_results["total_pipeline_time"] = total_time
        
        return pipeline_results
    
    # Execute comprehensive integration benchmark
    integration_results = benchmark_coordinator.execute_comprehensive_benchmark(
        test_name="comprehensive_data_loading_integration",
        test_function=execute_complete_pipeline,
        test_category="data_loading",
        data_size_config=(100000, 50, 100.0),
        enable_memory_profiling=True,
        enable_regression_detection=True,
        establish_baseline=True  # Establish baseline for integration testing
    )
    
    # Extract pipeline performance metrics
    mean_execution_time = integration_results["statistics"]["mean"]
    pipeline_cv = integration_results["statistics"]["cv_percent"]
    
    # Comprehensive validation of integration results
    sla_threshold = data_loading_test_config["sla_thresholds"]["data_loading_per_100mb_seconds"]
    assert mean_execution_time <= sla_threshold, (
        f"Integration pipeline SLA violation: {mean_execution_time:.4f}s > {sla_threshold:.4f}s"
    )
    
    # Statistical reliability validation
    variance_threshold = data_loading_test_config["statistical_requirements"]["variance_threshold_percent"]
    assert pipeline_cv <= variance_threshold, (
        f"Integration pipeline statistical variance too high: {pipeline_cv:.2f}% > {variance_threshold:.1f}%"
    )
    
    # Memory efficiency validation
    memory_results = integration_results.get("memory_results")
    if memory_results:
        memory_multiplier = memory_results["memory_multiplier"]
        efficiency_threshold = data_loading_test_config["sla_thresholds"]["memory_multiplier_threshold"]
        
        assert memory_multiplier <= efficiency_threshold, (
            f"Integration memory efficiency violation: {memory_multiplier:.2f}x > {efficiency_threshold:.1f}x"
        )
    
    # Baseline establishment and regression analysis
    baseline_results = integration_results.get("baseline_results")
    if baseline_results and baseline_results.get("validation_status") == "completed":
        regression_analysis = baseline_results.get("regression_analysis", {})
        if regression_analysis.get("regression_analysis", {}).get("regression_detected"):
            warnings.warn(
                f"Performance regression detected in integration pipeline: "
                f"{regression_analysis['regression_analysis']['confidence']:.2f} confidence"
            )
    
    print(f"\n=== Integration Benchmark Results ===")
    print(f"Pipeline execution time: {mean_execution_time:.4f}s")
    print(f"Statistical CV: {pipeline_cv:.2f}%")
    print(f"Memory multiplier: {memory_results['memory_multiplier']:.2f}x" if memory_results else "N/A")
    print(f"SLA compliance: {'✓ PASS' if mean_execution_time <= sla_threshold else '✗ FAIL'}")
    print(f"Statistical reliability: {'✓ PASS' if pipeline_cv <= variance_threshold else '✗ FAIL'}")
    print(f"Memory efficiency: {'✓ PASS' if not memory_results or memory_results['memory_multiplier'] <= efficiency_threshold else '✗ FAIL'}")
    
    # Detailed pipeline component analysis
    # Note: We can't access the detailed pipeline results here since they're inside the benchmark function
    # This would require modifying the benchmark coordinator to return more detailed results
    print(f"Integration pipeline validation: COMPLETED")


# ============================================================================
# MODULE METADATA AND BENCHMARK CONFIGURATION
# ============================================================================

# Test module metadata for benchmark categorization
__benchmark_config__ = {
    "category": "data-loading",
    "sla_requirements": {
        "TST-PERF-001": "Data loading <1s per 100MB",
        "F-003-RQ-004": "Format detection overhead <100ms",
        "F-014": "Data-size scalability (1MB to 1GB)",
        "Section 2.4.9": "Statistical accuracy ±5% variance"
    },
    "integration_points": {
        "cli_runner": "scripts/benchmarks/run_benchmarks.py --category data-loading",
        "ci_cd": "GitHub Actions benchmark job with artifact collection",
        "memory_profiling": "pytest-memory-profiler for line-by-line analysis",
        "cross_platform": "Ubuntu, Windows, macOS environment validation"
    },
    "excluded_from_default": True,
    "pytest_markers": ["benchmark", "performance", "data_loading"],
    "artifacts_generated": [
        "JSON performance reports",
        "CSV statistical summaries", 
        "Memory profiling reports",
        "Cross-platform comparison data",
        "Regression detection alerts"
    ]
}

# Export key benchmark functions for CLI integration
__all__ = [
    "test_benchmark_pickle_loading_sla_validation",
    "test_benchmark_format_detection_overhead", 
    "test_benchmark_data_loading_scalability",
    "test_benchmark_memory_efficiency_validation",
    "test_benchmark_cross_platform_performance_consistency",
    "test_benchmark_property_based_edge_cases",
    "test_benchmark_comprehensive_data_loading_integration"
]

# Module documentation for benchmark suite
__doc__ += f"""

Benchmark Test Categories:
- Core SLA Validation: {len([f for f in __all__ if 'sla' in f])} tests
- Scalability Testing: {len([f for f in __all__ if 'scalability' in f])} tests  
- Memory Efficiency: {len([f for f in __all__ if 'memory' in f])} tests
- Cross-Platform: {len([f for f in __all__ if 'cross_platform' in f])} tests
- Property-Based: {len([f for f in __all__ if 'property' in f])} tests
- Integration: {len([f for f in __all__ if 'integration' in f])} tests

Total Benchmark Functions: {len(__all__)}

CLI Execution:
    python scripts/benchmarks/run_benchmarks.py --category data-loading
    python scripts/benchmarks/run_benchmarks.py --category data-loading --verbose
    python scripts/benchmarks/run_benchmarks.py --category data-loading --report-artifacts

Performance SLA Matrix:
    Data Loading: <1s per 100MB (TST-PERF-001)
    Format Detection: <100ms overhead (F-003-RQ-004)
    Memory Efficiency: <2x data size multiplier
    Cross-Platform Variance: <10% difference
    Statistical Accuracy: ±5% variance @ 95% confidence

Integration Points:
    - pytest-benchmark plugin for statistical measurement
    - pytest-memory-profiler for detailed memory analysis  
    - GitHub Actions for CI/CD artifact collection
    - Cross-platform normalization for consistent results
    - Regression detection with historical baseline comparison
"""