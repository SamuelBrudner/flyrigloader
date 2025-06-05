"""
Performance benchmark test suite for DataFrame transformation operations.

This module implements comprehensive performance testing for DataFrame transformation
operations against TST-PERF-002 SLA requirement of <500ms per 1M rows. Validates
make_dataframe_from_config function performance across various experimental data
scenarios including multi-dimensional array handling, special handlers (signal_disp),
metadata integration, and time alignment validation.

Test Coverage:
- TST-PERF-002: DataFrame transformation within 500ms per 1M rows
- F-006-RQ-001: Convert exp_matrix to DataFrame within performance constraints  
- F-006-RQ-002: Handle multi-dimensional arrays with optimized transformation performance
- F-006-RQ-003: Execute custom transformation functions within <50ms per handler
- F-006-RQ-004: Add metadata columns to DataFrame within <10ms merge time
- Section 2.4.5: Memory efficiency <2x data size overhead

Statistical Measurement:
Uses pytest-benchmark for reliable performance metrics with automatic calibration,
statistical analysis, and regression detection across different data complexities.

Integration with CI/CD:
Benchmarks integrate with automated testing pipeline to ensure consistent DataFrame
transformation performance and prevent performance regression during library updates.
"""

import gc
import psutil
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings

# Import flyrigloader modules for testing
from flyrigloader.io.pickle import make_dataframe_from_config
from flyrigloader.io.column_models import (
    ColumnConfig, 
    ColumnConfigDict, 
    SpecialHandlerType,
    ColumnDimension
)


# ============================================================================
# PERFORMANCE TEST DATA GENERATION FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def benchmark_data_generator():
    """
    Session-scoped fixture providing optimized synthetic data generation
    for performance benchmarking across various dataset scales.
    
    Features:
    - Memory-efficient data generation strategies
    - Realistic experimental data patterns (neural recordings, behavioral data)
    - Configurable complexity levels for performance testing
    - Cross-platform optimized generation
    """
    class BenchmarkDataGenerator:
        def __init__(self):
            self.random_seed = 42
            np.random.seed(self.random_seed)
        
        def generate_time_series(self, n_points: int, sampling_freq: float = 60.0) -> np.ndarray:
            """Generate realistic time series data for performance testing."""
            return np.linspace(0, n_points / sampling_freq, n_points, dtype=np.float64)
        
        def generate_position_data(self, n_points: int, arena_diameter: float = 120.0) -> Tuple[np.ndarray, np.ndarray]:
            """Generate realistic position tracking data with movement patterns."""
            # Use memory-efficient generation for large datasets
            x_pos = np.random.uniform(-arena_diameter/2, arena_diameter/2, n_points).astype(np.float32)
            y_pos = np.random.uniform(-arena_diameter/2, arena_diameter/2, n_points).astype(np.float32)
            
            # Add realistic movement correlation
            for i in range(1, min(1000, n_points)):  # Limit correlation computation for large datasets
                x_pos[i] = 0.95 * x_pos[i-1] + 0.05 * x_pos[i]
                y_pos[i] = 0.95 * y_pos[i-1] + 0.05 * y_pos[i]
            
            return x_pos, y_pos
        
        def generate_signal_data(self, n_points: int, n_channels: int = 16) -> np.ndarray:
            """Generate multi-channel signal data optimized for benchmarking."""
            # Use float32 for memory efficiency in large datasets
            return np.random.normal(0, 1, (n_channels, n_points)).astype(np.float32)
        
        def generate_exp_matrix(self, 
                              n_points: int,
                              include_signal_disp: bool = True,
                              include_metadata: bool = True,
                              complexity_level: str = "medium") -> Dict[str, Any]:
            """
            Generate comprehensive experimental data matrix for performance testing.
            
            Args:
                n_points: Number of time points to generate
                include_signal_disp: Whether to include multi-dimensional signal data
                include_metadata: Whether to include metadata fields
                complexity_level: Data complexity ("low", "medium", "high")
            
            Returns:
                Dict containing experimental data matrix
            """
            # Base time series data
            time_data = self.generate_time_series(n_points)
            x_pos, y_pos = self.generate_position_data(n_points)
            
            matrix = {
                't': time_data,
                'x': x_pos,
                'y': y_pos
            }
            
            # Add complexity based on level
            if complexity_level in ["medium", "high"]:
                # Add velocity and derived measures
                dt = np.diff(time_data, prepend=time_data[0])
                dx = np.diff(x_pos, prepend=x_pos[0])
                dy = np.diff(y_pos, prepend=y_pos[0])
                
                matrix.update({
                    'vx': (dx / dt).astype(np.float32),
                    'vy': (dy / dt).astype(np.float32),
                    'speed': np.sqrt(dx**2 + dy**2).astype(np.float32),
                    'dtheta': np.arctan2(dy, dx).astype(np.float32)
                })
            
            if complexity_level == "high":
                # Add additional derived measures
                matrix.update({
                    'distance_from_center': np.sqrt(x_pos**2 + y_pos**2).astype(np.float32),
                    'cumulative_distance': np.cumsum(np.sqrt(dx**2 + dy**2)).astype(np.float32),
                    'signal': np.random.normal(0, 1, n_points).astype(np.float32)
                })
            
            # Add multi-dimensional signal data for special handler testing
            if include_signal_disp:
                n_channels = 16 if complexity_level != "high" else 32
                matrix['signal_disp'] = self.generate_signal_data(n_points, n_channels)
            
            return matrix
    
    return BenchmarkDataGenerator()


@pytest.fixture(scope="function", params=[
    ("small", 1000, "Small dataset: 1K timepoints"),
    ("medium", 100000, "Medium dataset: 100K timepoints"), 
    ("large", 1000000, "Large dataset: 1M timepoints"),
    ("extra_large", 2000000, "Extra large dataset: 2M timepoints")
])
def benchmark_dataset_scale(request, benchmark_data_generator):
    """
    Parametrized fixture providing datasets at different scales for performance testing.
    
    Scales:
    - Small: 1K timepoints (~60KB) - baseline performance
    - Medium: 100K timepoints (~6MB) - typical experimental session
    - Large: 1M timepoints (~60MB) - TST-PERF-002 SLA target
    - Extra Large: 2M timepoints (~120MB) - stress testing
    """
    scale_name, n_points, description = request.param
    
    # Force garbage collection before generating test data
    gc.collect()
    
    # Generate test data
    exp_matrix = benchmark_data_generator.generate_exp_matrix(
        n_points=n_points,
        include_signal_disp=True,
        complexity_level="medium"
    )
    
    # Calculate expected memory usage and SLA timing
    base_data_size_mb = (n_points * 8 * 4) / (1024 * 1024)  # Approximate size for 4 float64 columns
    expected_sla_time = (n_points / 1_000_000) * 0.5  # 500ms per 1M rows
    
    return {
        "scale_name": scale_name,
        "n_points": n_points,
        "description": description,
        "exp_matrix": exp_matrix,
        "expected_data_size_mb": base_data_size_mb,
        "expected_sla_time_seconds": expected_sla_time
    }


@pytest.fixture(scope="function")
def benchmark_column_config():
    """
    Optimized column configuration for performance benchmarking.
    
    Provides a streamlined configuration focused on performance-critical
    operations including signal_disp special handling and metadata integration.
    """
    config_dict = {
        "columns": {
            "t": ColumnConfig(
                type="numpy.ndarray",
                dimension=ColumnDimension.ONE_D,
                required=True,
                description="Time values"
            ),
            "x": ColumnConfig(
                type="numpy.ndarray", 
                dimension=ColumnDimension.ONE_D,
                required=True,
                description="X position"
            ),
            "y": ColumnConfig(
                type="numpy.ndarray",
                dimension=ColumnDimension.ONE_D, 
                required=True,
                description="Y position"
            ),
            "vx": ColumnConfig(
                type="numpy.ndarray",
                dimension=ColumnDimension.ONE_D,
                required=False,
                description="X velocity"
            ),
            "vy": ColumnConfig(
                type="numpy.ndarray",
                dimension=ColumnDimension.ONE_D,
                required=False,
                description="Y velocity"
            ),
            "speed": ColumnConfig(
                type="numpy.ndarray",
                dimension=ColumnDimension.ONE_D,
                required=False,
                description="Speed magnitude"
            ),
            "dtheta": ColumnConfig(
                type="numpy.ndarray",
                dimension=ColumnDimension.ONE_D,
                required=False,
                description="Change in heading"
            ),
            "signal_disp": ColumnConfig(
                type="numpy.ndarray",
                dimension=ColumnDimension.TWO_D,
                required=False,
                description="Multi-channel signal data",
                special_handling=SpecialHandlerType.TRANSFORM_TIME_DIMENSION
            ),
            "experiment_id": ColumnConfig(
                type="string",
                required=False,
                is_metadata=True,
                description="Experiment identifier"
            ),
            "animal_id": ColumnConfig(
                type="string",
                required=False,
                is_metadata=True,
                description="Animal identifier"
            ),
            "condition": ColumnConfig(
                type="string",
                required=False,
                is_metadata=True,
                description="Experimental condition"
            )
        },
        "special_handlers": {
            "transform_to_match_time_dimension": "_handle_signal_disp"
        }
    }
    
    return ColumnConfigDict.model_validate(config_dict)


@pytest.fixture(scope="function")
def benchmark_metadata():
    """Sample metadata for performance testing metadata integration."""
    return {
        "experiment_id": "PERF_TEST_001",
        "animal_id": "mouse_benchmark_001", 
        "condition": "performance_testing",
        "date": "20241201",
        "rig": "benchmark_rig",
        "sampling_rate": 60.0,
        "arena_diameter_mm": 120.0
    }


@pytest.fixture(scope="function")
def memory_monitor():
    """
    Fixture for monitoring memory usage during benchmark tests.
    
    Provides utilities for measuring memory efficiency and detecting
    memory leaks during DataFrame transformation operations.
    """
    class MemoryMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.baseline_memory = None
            self.peak_memory = None
        
        def start_monitoring(self):
            """Start memory monitoring and establish baseline."""
            gc.collect()  # Clean up before baseline
            self.baseline_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
            self.peak_memory = self.baseline_memory
        
        def update_peak(self):
            """Update peak memory usage."""
            current_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
            self.peak_memory = max(self.peak_memory, current_memory)
        
        def get_memory_overhead(self, data_size_mb: float) -> float:
            """Calculate memory overhead ratio compared to data size."""
            if self.baseline_memory is None:
                raise ValueError("Memory monitoring not started")
            
            self.update_peak()
            memory_used = self.peak_memory - self.baseline_memory
            return memory_used / data_size_mb if data_size_mb > 0 else 0
        
        def assert_memory_efficiency(self, data_size_mb: float, max_overhead: float = 2.0):
            """Assert memory efficiency meets SLA requirements."""
            overhead_ratio = self.get_memory_overhead(data_size_mb)
            assert overhead_ratio <= max_overhead, (
                f"Memory overhead {overhead_ratio:.2f}x exceeds maximum {max_overhead}x "
                f"(used {self.peak_memory - self.baseline_memory:.1f}MB for {data_size_mb:.1f}MB data)"
            )
    
    return MemoryMonitor()


# ============================================================================
# CORE DATAFRAME TRANSFORMATION BENCHMARKS
# ============================================================================

@pytest.mark.benchmark(group="dataframe_transformation")
def test_benchmark_make_dataframe_from_config_sla_validation(
    benchmark, 
    benchmark_dataset_scale,
    benchmark_column_config,
    memory_monitor
):
    """
    Benchmark make_dataframe_from_config against TST-PERF-002 SLA requirements.
    
    Validates DataFrame transformation performance of <500ms per 1M rows across
    different dataset scales with comprehensive memory efficiency monitoring.
    
    Performance Requirements:
    - TST-PERF-002: <500ms per 1M rows DataFrame transformation
    - Section 2.4.5: <2x data size memory overhead
    - F-006-RQ-001: Efficient exp_matrix to DataFrame conversion
    
    Test Coverage:
    - Small datasets: Performance baseline establishment
    - Medium datasets: Typical experimental session scale
    - Large datasets: SLA compliance validation  
    - Extra large datasets: Scalability verification
    """
    # Setup test data
    exp_matrix = benchmark_dataset_scale["exp_matrix"]
    n_points = benchmark_dataset_scale["n_points"]
    expected_sla_time = benchmark_dataset_scale["expected_sla_time_seconds"]
    data_size_mb = benchmark_dataset_scale["expected_data_size_mb"]
    
    # Start memory monitoring
    memory_monitor.start_monitoring()
    
    # Define benchmark function
    def transform_dataframe():
        """Core transformation operation being benchmarked."""
        return make_dataframe_from_config(
            exp_matrix=exp_matrix,
            config_source=benchmark_column_config,
            metadata=None,
            skip_columns=None
        )
    
    # Execute benchmark with statistical measurement
    result = benchmark.pedantic(
        transform_dataframe,
        iterations=5,  # Sufficient for statistical significance
        rounds=3,      # Multiple rounds for consistency
        warmup_rounds=1  # Warm up for consistent timing
    )
    
    # Validate SLA compliance
    avg_time = benchmark.stats.stats.mean
    assert avg_time <= expected_sla_time, (
        f"SLA violation: DataFrame transformation took {avg_time:.3f}s, "
        f"expected ≤{expected_sla_time:.3f}s for {n_points:,} rows"
    )
    
    # Validate memory efficiency (Section 2.4.5)
    memory_monitor.assert_memory_efficiency(data_size_mb, max_overhead=2.0)
    
    # Validate result integrity
    assert isinstance(result, pd.DataFrame), "Result must be a pandas DataFrame"
    assert len(result) == n_points, f"DataFrame length mismatch: expected {n_points}, got {len(result)}"
    
    # Log performance metrics for tracking
    print(f"\nPerformance Metrics for {benchmark_dataset_scale['description']}:")
    print(f"  Execution time: {avg_time:.3f}s (SLA: ≤{expected_sla_time:.3f}s)")
    print(f"  Rows per second: {n_points/avg_time:,.0f}")
    print(f"  Memory overhead: {memory_monitor.get_memory_overhead(data_size_mb):.1f}x")


@pytest.mark.benchmark(group="multidimensional_arrays")
def test_benchmark_signal_disp_transformation_performance(
    benchmark,
    benchmark_data_generator,
    benchmark_column_config,
    memory_monitor
):
    """
    Benchmark multi-dimensional array handling performance for signal_disp data.
    
    Validates F-006-RQ-002 requirements for optimized transformation of 2D signal
    arrays with special handler processing. Tests signal_disp transformation 
    across different channel counts and data sizes.
    
    Performance Requirements:
    - F-006-RQ-002: Optimized multi-dimensional array transformation
    - F-006-RQ-003: Special handler execution <50ms per handler
    - Vectorized operations for memory efficiency
    """
    # Generate test data with large signal_disp array
    n_points = 100000  # 100K timepoints for consistent testing
    n_channels = 32    # Larger channel count for stress testing
    
    exp_matrix = {
        't': benchmark_data_generator.generate_time_series(n_points),
        'x': np.random.uniform(-60, 60, n_points).astype(np.float32),
        'y': np.random.uniform(-60, 60, n_points).astype(np.float32),
        'signal_disp': benchmark_data_generator.generate_signal_data(n_points, n_channels)
    }
    
    # Calculate expected data size for multi-dimensional array
    signal_disp_size_mb = (n_channels * n_points * 4) / (1024 * 1024)  # float32 = 4 bytes
    
    # Start memory monitoring
    memory_monitor.start_monitoring()
    
    # Define benchmark function focusing on signal_disp transformation
    def transform_signal_disp():
        """Benchmark signal_disp-specific transformation operations."""
        return make_dataframe_from_config(
            exp_matrix=exp_matrix,
            config_source=benchmark_column_config,
            metadata=None,
            skip_columns=['vx', 'vy', 'speed', 'dtheta']  # Focus on signal_disp
        )
    
    # Execute benchmark
    result = benchmark.pedantic(
        transform_signal_disp,
        iterations=3,
        rounds=2,
        warmup_rounds=1
    )
    
    # Validate special handler performance (F-006-RQ-003: <50ms per handler)
    handler_time = benchmark.stats.stats.mean
    assert handler_time <= 0.05, (
        f"Special handler performance violation: signal_disp transformation took "
        f"{handler_time:.3f}s, expected ≤0.050s"
    )
    
    # Validate memory efficiency for multi-dimensional data
    memory_monitor.assert_memory_efficiency(signal_disp_size_mb, max_overhead=2.0)
    
    # Validate signal_disp transformation correctness
    assert isinstance(result, pd.DataFrame), "Result must be a pandas DataFrame"
    assert 'signal_disp' in result.columns, "signal_disp column must be present"
    assert len(result) == n_points, f"DataFrame length mismatch: expected {n_points}, got {len(result)}"
    
    # Validate signal_disp series structure (should be Series of arrays)
    signal_disp_series = result['signal_disp']
    assert isinstance(signal_disp_series, pd.Series), "signal_disp must be a pandas Series"
    
    # Check first few entries to ensure proper array structure
    for i in range(min(3, len(signal_disp_series))):
        signal_array = signal_disp_series.iloc[i]
        assert isinstance(signal_array, np.ndarray), f"signal_disp[{i}] must be numpy array"
        assert signal_array.shape[0] == n_channels, f"signal_disp[{i}] must have {n_channels} channels"
    
    print(f"\nSignal_disp Transformation Metrics:")
    print(f"  Channels: {n_channels}, Timepoints: {n_points:,}")
    print(f"  Transformation time: {handler_time:.3f}s")
    print(f"  Data throughput: {(n_channels * n_points) / handler_time / 1e6:.1f}M elements/sec")


@pytest.mark.benchmark(group="metadata_integration")
def test_benchmark_metadata_integration_performance(
    benchmark,
    benchmark_data_generator,
    benchmark_column_config,
    benchmark_metadata,
    memory_monitor
):
    """
    Benchmark metadata integration performance for F-006-RQ-004 requirements.
    
    Validates metadata column addition performance with <10ms merge time
    across different dataset sizes and metadata complexity levels.
    
    Performance Requirements:
    - F-006-RQ-004: Add metadata columns within <10ms merge time
    - Efficient metadata broadcasting across large DataFrames
    - Memory-efficient metadata storage
    """
    # Generate test data
    n_points = 500000  # 500K timepoints for metadata stress testing
    exp_matrix = benchmark_data_generator.generate_exp_matrix(
        n_points=n_points,
        include_signal_disp=False,  # Focus on metadata performance
        complexity_level="low"
    )
    
    # Calculate baseline data size
    base_data_size_mb = (n_points * 8 * 3) / (1024 * 1024)  # 3 float64 columns
    
    # Start memory monitoring
    memory_monitor.start_monitoring()
    
    # Define benchmark function focusing on metadata integration
    def transform_with_metadata():
        """Benchmark metadata integration operations."""
        return make_dataframe_from_config(
            exp_matrix=exp_matrix,
            config_source=benchmark_column_config,
            metadata=benchmark_metadata,
            skip_columns=['vx', 'vy', 'speed', 'dtheta', 'signal_disp']
        )
    
    # Execute benchmark
    result = benchmark.pedantic(
        transform_with_metadata,
        iterations=5,
        rounds=3,
        warmup_rounds=1
    )
    
    # Validate metadata integration performance (F-006-RQ-004: <10ms merge time)
    merge_time = benchmark.stats.stats.mean
    assert merge_time <= 0.01, (
        f"Metadata integration performance violation: merge took {merge_time:.3f}s, "
        f"expected ≤0.010s"
    )
    
    # Validate memory efficiency with metadata
    memory_monitor.assert_memory_efficiency(base_data_size_mb, max_overhead=2.0)
    
    # Validate metadata integration correctness
    assert isinstance(result, pd.DataFrame), "Result must be a pandas DataFrame"
    assert len(result) == n_points, f"DataFrame length mismatch"
    
    # Check that all metadata fields are present and correctly populated
    for key, expected_value in benchmark_metadata.items():
        if key in benchmark_column_config.columns and benchmark_column_config.columns[key].is_metadata:
            assert key in result.columns, f"Metadata column '{key}' missing from result"
            
            # Validate that metadata is properly broadcasted
            unique_values = result[key].unique()
            assert len(unique_values) == 1, f"Metadata column '{key}' should have single value"
            assert unique_values[0] == expected_value, f"Metadata value mismatch for '{key}'"
    
    print(f"\nMetadata Integration Metrics:")
    print(f"  Rows: {n_points:,}, Metadata fields: {len(benchmark_metadata)}")
    print(f"  Integration time: {merge_time:.4f}s")
    print(f"  Metadata throughput: {len(benchmark_metadata) * n_points / merge_time / 1e6:.1f}M assignments/sec")


@pytest.mark.benchmark(group="vectorized_operations")
def test_benchmark_vectorized_operations_performance(
    benchmark,
    benchmark_data_generator,
    benchmark_column_config,
    memory_monitor
):
    """
    Benchmark vectorized operations performance for large-scale data processing.
    
    Validates efficient vectorized operations for derived calculations and
    data transformations across multiple columns simultaneously.
    
    Performance Requirements:
    - Vectorized NumPy operations for optimal performance
    - Memory-efficient batch processing
    - Scalable performance across dataset sizes
    """
    # Generate complex test data with derived calculations
    n_points = 750000  # 750K timepoints for vectorized operations testing
    exp_matrix = benchmark_data_generator.generate_exp_matrix(
        n_points=n_points,
        include_signal_disp=False,
        complexity_level="high"  # Include all derived measures
    )
    
    # Calculate data size for multiple derived columns
    data_size_mb = (n_points * 8 * 8) / (1024 * 1024)  # ~8 float64 columns
    
    # Start memory monitoring  
    memory_monitor.start_monitoring()
    
    # Define benchmark function for vectorized operations
    def transform_vectorized():
        """Benchmark vectorized transformation operations."""
        return make_dataframe_from_config(
            exp_matrix=exp_matrix,
            config_source=benchmark_column_config,
            metadata=None,
            skip_columns=['signal_disp']  # Focus on vectorized numerical operations
        )
    
    # Execute benchmark
    result = benchmark.pedantic(
        transform_vectorized,
        iterations=3,
        rounds=2,
        warmup_rounds=1
    )
    
    # Validate vectorized operations performance
    vectorized_time = benchmark.stats.stats.mean
    operations_per_second = n_points / vectorized_time
    
    # Expect high throughput for vectorized operations (>1M operations/sec)
    assert operations_per_second >= 1_000_000, (
        f"Vectorized operations performance below threshold: "
        f"{operations_per_second:,.0f} ops/sec, expected ≥1,000,000"
    )
    
    # Validate memory efficiency for vectorized operations
    memory_monitor.assert_memory_efficiency(data_size_mb, max_overhead=2.0)
    
    # Validate vectorized transformation correctness
    assert isinstance(result, pd.DataFrame), "Result must be a pandas DataFrame"
    assert len(result) == n_points, f"DataFrame length mismatch"
    
    # Verify presence of derived columns
    expected_columns = ['t', 'x', 'y', 'vx', 'vy', 'speed', 'dtheta']
    for col in expected_columns:
        if col in exp_matrix:  # Only check columns present in input
            assert col in result.columns, f"Derived column '{col}' missing"
    
    print(f"\nVectorized Operations Metrics:")
    print(f"  Rows: {n_points:,}, Columns processed: {len(result.columns)}")
    print(f"  Processing time: {vectorized_time:.3f}s")
    print(f"  Throughput: {operations_per_second:,.0f} rows/sec")


# ============================================================================
# TIME ALIGNMENT AND VALIDATION BENCHMARKS
# ============================================================================

@pytest.mark.benchmark(group="time_alignment")
def test_benchmark_time_alignment_validation_performance(
    benchmark,
    benchmark_data_generator,
    benchmark_column_config,
    memory_monitor
):
    """
    Benchmark time alignment validation performance for F-006-RQ-005 requirements.
    
    Validates efficient time dimension validation across multiple arrays ensuring
    all data arrays properly align with the time dimension without performance penalty.
    
    Performance Requirements:
    - F-006-RQ-005: Efficient time alignment validation
    - Fast dimension checking across multiple arrays
    - Memory-efficient validation operations
    """
    # Generate test data with potential alignment challenges
    n_points = 600000  # 600K timepoints
    exp_matrix = benchmark_data_generator.generate_exp_matrix(
        n_points=n_points,
        include_signal_disp=True,
        complexity_level="high"
    )
    
    # Add potential alignment challenges (arrays with different orientations)
    n_channels = 20
    signal_data = benchmark_data_generator.generate_signal_data(n_points, n_channels)
    
    # Test both orientations to stress time alignment validation
    exp_matrix['signal_disp'] = signal_data  # (channels, timepoints)
    exp_matrix['signal_disp_alt'] = signal_data.T  # (timepoints, channels) - should be transposed
    
    # Calculate data size including alignment validation overhead
    data_size_mb = (n_points * 8 * 6 + n_channels * n_points * 4 * 2) / (1024 * 1024)
    
    # Start memory monitoring
    memory_monitor.start_monitoring()
    
    # Define benchmark function for time alignment validation
    def validate_time_alignment():
        """Benchmark time alignment validation operations."""
        return make_dataframe_from_config(
            exp_matrix=exp_matrix,
            config_source=benchmark_column_config,
            metadata=None,
            skip_columns=['signal_disp_alt']  # Test with standard signal_disp only
        )
    
    # Execute benchmark
    result = benchmark.pedantic(
        validate_time_alignment,
        iterations=3,
        rounds=2,
        warmup_rounds=1
    )
    
    # Validate time alignment performance
    alignment_time = benchmark.stats.stats.mean
    
    # Time alignment should be fast (validation overhead <10% of total time)
    expected_max_time = 0.1  # 100ms for time alignment validation
    assert alignment_time <= expected_max_time, (
        f"Time alignment validation too slow: {alignment_time:.3f}s, "
        f"expected ≤{expected_max_time:.3f}s"
    )
    
    # Validate memory efficiency during alignment validation
    memory_monitor.assert_memory_efficiency(data_size_mb, max_overhead=2.0)
    
    # Validate time alignment correctness
    assert isinstance(result, pd.DataFrame), "Result must be a pandas DataFrame"
    assert len(result) == n_points, f"Time alignment failed: expected {n_points} rows"
    
    # Verify time column integrity
    assert 't' in result.columns, "Time column 't' must be present"
    time_column = result['t']
    assert len(time_column) == n_points, "Time column length mismatch"
    assert time_column.dtype in [np.float64, np.float32], "Time column must be numeric"
    
    # Verify signal_disp alignment (should be Series of arrays)
    if 'signal_disp' in result.columns:
        signal_disp = result['signal_disp']
        assert isinstance(signal_disp, pd.Series), "signal_disp must be pandas Series"
        assert len(signal_disp) == n_points, "signal_disp length must match time dimension"
        
        # Check array structure in signal_disp
        first_signal = signal_disp.iloc[0]
        assert isinstance(first_signal, np.ndarray), "signal_disp elements must be numpy arrays"
        assert first_signal.shape[0] == n_channels, f"signal_disp arrays must have {n_channels} channels"
    
    print(f"\nTime Alignment Validation Metrics:")
    print(f"  Timepoints: {n_points:,}, Channels: {n_channels}")
    print(f"  Alignment validation time: {alignment_time:.3f}s")
    print(f"  Validation rate: {n_points / alignment_time:,.0f} timepoints/sec")


# ============================================================================
# MEMORY EFFICIENCY AND STRESS TESTING
# ============================================================================

@pytest.mark.benchmark(group="memory_efficiency")
def test_benchmark_memory_efficiency_stress_test(
    benchmark,
    benchmark_data_generator,
    benchmark_column_config,
    memory_monitor
):
    """
    Stress test memory efficiency under high memory pressure scenarios.
    
    Validates Section 2.4.5 memory efficiency requirements with large datasets
    and complex transformations while maintaining <2x data size overhead.
    
    Performance Requirements:
    - Section 2.4.5: Memory efficiency <2x data size overhead
    - Efficient memory usage under pressure
    - Garbage collection optimization
    """
    # Generate large dataset for memory stress testing
    n_points = 1_500_000  # 1.5M timepoints for memory stress
    
    # Create complex experimental matrix with all features
    exp_matrix = benchmark_data_generator.generate_exp_matrix(
        n_points=n_points,
        include_signal_disp=True,
        complexity_level="high"
    )
    
    # Calculate total expected data size
    base_columns = 8  # t, x, y, vx, vy, speed, dtheta, signal, distance_from_center, cumulative_distance
    signal_channels = 16
    data_size_mb = (
        (n_points * 8 * base_columns) +  # Base columns (float64)
        (signal_channels * n_points * 4)  # signal_disp (float32)
    ) / (1024 * 1024)
    
    # Force garbage collection before test
    gc.collect()
    
    # Start memory monitoring
    memory_monitor.start_monitoring()
    
    # Define memory-intensive transformation
    def memory_intensive_transform():
        """Benchmark memory efficiency under stress conditions."""
        # Include metadata to add memory pressure
        metadata = {
            "experiment_id": "MEMORY_STRESS_TEST",
            "animal_id": f"stress_test_animal_{np.random.randint(1000)}",
            "condition": "memory_stress"
        }
        
        result = make_dataframe_from_config(
            exp_matrix=exp_matrix,
            config_source=benchmark_column_config,
            metadata=metadata,
            skip_columns=None  # Include all columns for maximum memory pressure
        )
        
        # Force intermediate garbage collection
        gc.collect()
        return result
    
    # Execute memory stress benchmark
    result = benchmark.pedantic(
        memory_intensive_transform,
        iterations=2,  # Fewer iterations for memory stress test
        rounds=1,
        warmup_rounds=0
    )
    
    # Validate memory efficiency under stress (Section 2.4.5)
    memory_overhead = memory_monitor.get_memory_overhead(data_size_mb)
    assert memory_overhead <= 2.0, (
        f"Memory efficiency violation under stress: {memory_overhead:.2f}x overhead, "
        f"expected ≤2.0x (used {memory_monitor.peak_memory - memory_monitor.baseline_memory:.1f}MB "
        f"for {data_size_mb:.1f}MB data)"
    )
    
    # Validate transformation correctness under memory pressure
    assert isinstance(result, pd.DataFrame), "Result must be a pandas DataFrame"
    assert len(result) == n_points, f"DataFrame length incorrect under memory pressure"
    
    # Verify all expected columns are present
    expected_base_cols = ['t', 'x', 'y', 'vx', 'vy', 'speed', 'dtheta']
    for col in expected_base_cols:
        if col in exp_matrix:
            assert col in result.columns, f"Column '{col}' missing under memory pressure"
    
    # Verify signal_disp handling under memory pressure
    if 'signal_disp' in result.columns:
        signal_col = result['signal_disp']
        assert isinstance(signal_col, pd.Series), "signal_disp structure corrupted under memory pressure"
        assert len(signal_col) == n_points, "signal_disp length incorrect under memory pressure"
    
    # Verify metadata integration under memory pressure
    assert 'experiment_id' in result.columns, "Metadata integration failed under memory pressure"
    
    # Final garbage collection and memory check
    gc.collect()
    
    stress_time = benchmark.stats.stats.mean
    print(f"\nMemory Efficiency Stress Test Metrics:")
    print(f"  Dataset size: {n_points:,} timepoints ({data_size_mb:.1f}MB)")
    print(f"  Transformation time: {stress_time:.3f}s")
    print(f"  Memory overhead: {memory_overhead:.2f}x")
    print(f"  Peak memory usage: {memory_monitor.peak_memory:.1f}MB")


# ============================================================================
# PROPERTY-BASED PERFORMANCE TESTING
# ============================================================================

@pytest.mark.benchmark(group="property_based")
@given(
    n_points=st.integers(min_value=1000, max_value=100000),
    n_channels=st.integers(min_value=4, max_value=32),
    include_derived=st.booleans()
)
@settings(max_examples=10, deadline=None)
def test_property_based_transformation_performance(
    benchmark,
    benchmark_data_generator,
    benchmark_column_config,
    memory_monitor,
    n_points,
    n_channels,
    include_derived
):
    """
    Property-based performance testing across diverse data configurations.
    
    Uses Hypothesis to generate diverse test scenarios and validate performance
    characteristics across different data sizes, channel counts, and complexity levels.
    
    Performance Requirements:
    - Consistent performance scaling across data configurations
    - Robust performance under diverse input conditions
    - Memory efficiency across property space
    """
    # Generate test data based on hypothesis parameters
    exp_matrix = {
        't': benchmark_data_generator.generate_time_series(n_points),
        'x': np.random.uniform(-60, 60, n_points).astype(np.float32),
        'y': np.random.uniform(-60, 60, n_points).astype(np.float32),
        'signal_disp': benchmark_data_generator.generate_signal_data(n_points, n_channels)
    }
    
    # Add derived measures based on property
    if include_derived:
        dt = np.diff(exp_matrix['t'], prepend=exp_matrix['t'][0])
        dx = np.diff(exp_matrix['x'], prepend=exp_matrix['x'][0])
        dy = np.diff(exp_matrix['y'], prepend=exp_matrix['y'][0])
        
        exp_matrix.update({
            'vx': (dx / dt).astype(np.float32),
            'vy': (dy / dt).astype(np.float32),
            'speed': np.sqrt(dx**2 + dy**2).astype(np.float32)
        })
    
    # Calculate expected data size
    base_cols = 3 + (3 if include_derived else 0)  # Base + optional derived
    data_size_mb = (
        (n_points * 4 * base_cols) +  # Base columns (float32)
        (n_channels * n_points * 4)  # signal_disp (float32)
    ) / (1024 * 1024)
    
    # Start memory monitoring
    memory_monitor.start_monitoring()
    
    # Define property-based transformation
    def property_transform():
        """Property-based transformation testing."""
        return make_dataframe_from_config(
            exp_matrix=exp_matrix,
            config_source=benchmark_column_config,
            metadata=None,
            skip_columns=None
        )
    
    # Execute property-based benchmark (single iteration for property testing)
    result = benchmark.pedantic(
        property_transform,
        iterations=1,
        rounds=1,
        warmup_rounds=0
    )
    
    # Validate property-based performance characteristics
    transform_time = benchmark.stats.stats.mean
    
    # Performance should scale linearly with data size
    expected_max_time = (n_points / 1_000_000) * 0.5  # 500ms per 1M rows scaling
    if n_points >= 10000:  # Only apply SLA to reasonable dataset sizes
        assert transform_time <= expected_max_time * 2, (  # Allow 2x tolerance for property testing
            f"Property-based performance violation: {transform_time:.3f}s > "
            f"{expected_max_time * 2:.3f}s for {n_points:,} rows"
        )
    
    # Validate memory efficiency (relaxed for property testing)
    if data_size_mb > 1.0:  # Only test memory efficiency for larger datasets
        memory_monitor.assert_memory_efficiency(data_size_mb, max_overhead=3.0)  # Relaxed for property testing
    
    # Validate transformation correctness
    assert isinstance(result, pd.DataFrame), "Property-based result must be DataFrame"
    assert len(result) == n_points, f"Property-based length mismatch: expected {n_points}, got {len(result)}"
    
    # Validate core columns presence
    assert 't' in result.columns, "Time column missing in property-based test"
    assert 'x' in result.columns, "X position missing in property-based test"
    assert 'y' in result.columns, "Y position missing in property-based test"
    
    # Validate signal_disp handling
    if 'signal_disp' in result.columns:
        signal_col = result['signal_disp']
        assert isinstance(signal_col, pd.Series), "Property-based signal_disp must be Series"
        assert len(signal_col) == n_points, "Property-based signal_disp length mismatch"


# ============================================================================
# REGRESSION AND COMPARISON BENCHMARKS  
# ============================================================================

@pytest.mark.benchmark(group="regression_prevention")
def test_benchmark_performance_regression_baseline(
    benchmark,
    benchmark_data_generator,
    benchmark_column_config,
    memory_monitor
):
    """
    Establish performance regression baseline for continuous monitoring.
    
    Provides consistent baseline measurements for detecting performance
    regressions in CI/CD pipeline with standardized test conditions.
    
    Performance Requirements:
    - Stable baseline for regression detection
    - Consistent measurement conditions
    - Representative workload patterns
    """
    # Standard baseline configuration
    n_points = 500000  # 500K timepoints - representative workload
    baseline_exp_matrix = benchmark_data_generator.generate_exp_matrix(
        n_points=n_points,
        include_signal_disp=True,
        complexity_level="medium"
    )
    
    # Calculate baseline data size
    data_size_mb = (n_points * 8 * 5 + 16 * n_points * 4) / (1024 * 1024)  # ~40MB
    
    # Start memory monitoring
    memory_monitor.start_monitoring()
    
    # Define baseline transformation
    def baseline_transform():
        """Baseline transformation for regression detection."""
        return make_dataframe_from_config(
            exp_matrix=baseline_exp_matrix,
            config_source=benchmark_column_config,
            metadata={"baseline_test": "regression_prevention"},
            skip_columns=None
        )
    
    # Execute baseline benchmark with high precision
    result = benchmark.pedantic(
        baseline_transform,
        iterations=10,  # High iteration count for stable baseline
        rounds=5,      # Multiple rounds for statistical significance
        warmup_rounds=2  # Adequate warmup for stable measurements
    )
    
    # Store baseline metrics for regression tracking
    baseline_time = benchmark.stats.stats.mean
    baseline_stddev = benchmark.stats.stats.stddev
    baseline_memory_overhead = memory_monitor.get_memory_overhead(data_size_mb)
    
    # Validate baseline consistency (low variance)
    coefficient_of_variation = baseline_stddev / baseline_time
    assert coefficient_of_variation <= 0.1, (
        f"Baseline measurement too variable: CV={coefficient_of_variation:.3f}, expected ≤0.1"
    )
    
    # Validate baseline memory efficiency
    memory_monitor.assert_memory_efficiency(data_size_mb, max_overhead=2.0)
    
    # Validate baseline transformation correctness
    assert isinstance(result, pd.DataFrame), "Baseline result must be DataFrame"
    assert len(result) == n_points, f"Baseline length mismatch"
    
    # Store baseline metrics for CI/CD tracking
    baseline_metrics = {
        "execution_time_seconds": baseline_time,
        "standard_deviation": baseline_stddev,
        "coefficient_of_variation": coefficient_of_variation,
        "memory_overhead_ratio": baseline_memory_overhead,
        "data_size_mb": data_size_mb,
        "rows_per_second": n_points / baseline_time,
        "memory_usage_mb": memory_monitor.peak_memory - memory_monitor.baseline_memory
    }
    
    print(f"\nPerformance Regression Baseline Metrics:")
    print(f"  Execution time: {baseline_time:.4f}s ± {baseline_stddev:.4f}s")
    print(f"  Coefficient of variation: {coefficient_of_variation:.3f}")
    print(f"  Memory overhead: {baseline_memory_overhead:.2f}x")
    print(f"  Throughput: {n_points / baseline_time:,.0f} rows/sec")
    
    # Save baseline metrics for regression detection in CI/CD
    # (In real implementation, this would be saved to a file or database)
    
    return baseline_metrics


if __name__ == "__main__":
    # Allow direct execution for development testing
    pytest.main([__file__, "-v", "--benchmark-only", "--benchmark-disable-gc"])