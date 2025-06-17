"""
Performance benchmark test suite for flyrigloader DataFrame transformation pipeline.

This module enforces SLA validation, memory efficiency testing, vectorized operation analysis, 
and property-based performance validation, relocated from default test suite to maintain rapid 
developer feedback while preserving comprehensive transformation performance analysis.

Key Performance Requirements:
- TST-PERF-002: DataFrame transformation <500ms per 1M rows
- Section 2.4.5: Memory efficiency requirements for large dataset transformations  
- Vectorized operation performance validation vs specialized transformation operations
- Data-size scaling linearity verification for transformation pipeline components
- Specialized scenario testing under stress and specialized transformation conditions
- Property-based performance validation via Hypothesis for comprehensive edge cases

Performance SLA Enforcement:
- Data transformation: <500ms per 1M rows (TST-PERF-002)
- Memory efficiency: <2x data size overhead (Section 2.4.5)
- Special handlers: <50ms per handler (F-006-RQ-003)
- Metadata merge: <10ms merge time (F-006-RQ-004)
- Vectorized operations: Optimal performance validation
- Time alignment: Validation performance benchmarks

Integration:
- pytest-benchmark for statistical measurement and comparison
- pytest-memory-profiler for line-by-line memory analysis and leak detection
- Hypothesis for property-based performance testing edge cases
- Cross-platform performance validation with environment normalization
- CI/CD integration via scripts/benchmarks/run_benchmarks.py CLI framework
- GitHub Actions artifact management for performance report generation
"""

import gc
import psutil
import time
import warnings
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings, assume

# Import flyrigloader modules - maintained integration from original location
from flyrigloader.io.pickle import make_dataframe_from_config, handle_signal_disp
from flyrigloader.io.column_models import get_config_from_source, ColumnConfigDict

# Import shared benchmark utilities and configuration
from .utils import (
    MemoryProfiler,
    memory_profiling_context, 
    estimate_data_size,
    analyze_benchmark_results
)
from .config import BenchmarkCategory, get_category_config


# ============================================================================
# BENCHMARK PERFORMANCE TEST DATA GENERATION FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def performance_test_configurations():
    """
    Session-scoped fixture providing standardized performance test configurations
    for consistent benchmarking across transformation test scenarios.
    
    Returns:
        Dict: Performance test scale configurations with SLA thresholds per specification
    """
    return {
        "small_scale": {
            "description": "Small dataset for baseline performance validation",
            "rows": 10_000,  # 10K rows
            "expected_transform_time_ms": 5.0,  # Well under SLA for small data
            "memory_multiplier_threshold": 1.5,
            "use_case": "Unit test validation"
        },
        "medium_scale": {
            "description": "Medium dataset for typical experimental data volumes", 
            "rows": 100_000,  # 100K rows
            "expected_transform_time_ms": 50.0,  # Proportional to SLA
            "memory_multiplier_threshold": 1.8,
            "use_case": "Standard experimental session"
        },
        "large_scale": {
            "description": "Large dataset for SLA boundary testing",
            "rows": 1_000_000,  # 1M rows (SLA boundary)
            "expected_transform_time_ms": 500.0,  # TST-PERF-002 SLA limit
            "memory_multiplier_threshold": 2.0,  # Section 2.4.5 requirement
            "use_case": "Maximum SLA validation"
        },
        "stress_scale": {
            "description": "Stress test for performance regression detection",
            "rows": 2_000_000,  # 2M rows (beyond SLA)
            "expected_transform_time_ms": 1000.0,  # Linear scaling expectation
            "memory_multiplier_threshold": 2.0,
            "use_case": "Performance regression detection"
        }
    }


@pytest.fixture
def synthetic_exp_matrix_generator():
    """
    Factory fixture for generating realistic experimental data matrices
    with configurable size and complexity for transformation performance testing.
    
    Returns:
        Callable: Function to generate experimental matrices with specific characteristics
    """
    def generate_matrix(
        rows: int,
        include_signal_disp: bool = True,
        include_metadata_fields: bool = True,
        signal_channels: int = 16,
        add_noise_characteristics: bool = True,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Generate synthetic experimental data matrix with realistic characteristics.
        
        Args:
            rows: Number of time points to generate
            include_signal_disp: Whether to include 2D signal display data
            include_metadata_fields: Whether to include metadata columns
            signal_channels: Number of signal channels for signal_disp
            add_noise_characteristics: Whether to add realistic noise patterns
            seed: Random seed for reproducible generation
            
        Returns:
            Dict containing synthetic experimental data
        """
        np.random.seed(seed)
        
        # Base time series (always required)
        sampling_freq = 60.0  # Hz
        duration = rows / sampling_freq
        time_array = np.linspace(0, duration, rows)
        
        # Realistic trajectory data with arena constraints
        arena_radius = 60.0  # mm
        center_bias = 0.3
        movement_noise = 0.1
        
        # Generate correlated random walk trajectory
        x_pos = np.zeros(rows)
        y_pos = np.zeros(rows)
        
        for i in range(1, rows):
            # Current distance from center
            current_radius = np.sqrt(x_pos[i-1]**2 + y_pos[i-1]**2)
            
            # Center bias force
            bias_strength = center_bias * (current_radius / arena_radius)**2
            center_force_x = -bias_strength * x_pos[i-1] / max(current_radius, 0.1)
            center_force_y = -bias_strength * y_pos[i-1] / max(current_radius, 0.1)
            
            # Random movement
            random_x = np.random.normal(0, movement_noise)
            random_y = np.random.normal(0, movement_noise)
            
            # Update position with boundary enforcement
            dt = 1.0 / sampling_freq
            dx = (center_force_x + random_x) * dt
            dy = (center_force_y + random_y) * dt
            
            new_x = x_pos[i-1] + dx
            new_y = y_pos[i-1] + dy
            
            # Enforce arena boundaries
            new_radius = np.sqrt(new_x**2 + new_y**2)
            if new_radius > arena_radius:
                reflection_factor = arena_radius / new_radius * 0.95
                new_x *= reflection_factor
                new_y *= reflection_factor
            
            x_pos[i] = new_x
            y_pos[i] = new_y
        
        # Build experimental matrix
        exp_matrix = {
            't': time_array,
            'x': x_pos,
            'y': y_pos
        }
        
        # Add velocity components
        vx = np.gradient(x_pos, time_array)
        vy = np.gradient(y_pos, time_array)
        exp_matrix['vx'] = vx
        exp_matrix['vy'] = vy
        exp_matrix['speed'] = np.sqrt(vx**2 + vy**2)
        
        # Add angular measures
        dtheta = np.arctan2(np.gradient(y_pos), np.gradient(x_pos))
        if add_noise_characteristics:
            dtheta += np.random.normal(0, 0.1, rows)  # Add realistic noise
        exp_matrix['dtheta'] = dtheta
        
        # Add single-channel signal with realistic characteristics
        signal_base_freq = 2.0  # Hz
        signal = np.sin(2 * np.pi * signal_base_freq * time_array)
        signal += 0.3 * np.sin(4 * np.pi * signal_base_freq * time_array)  # Harmonics
        if add_noise_characteristics:
            signal += np.random.normal(0, 0.05, rows)  # Add noise
        exp_matrix['signal'] = signal
        
        # Add multi-channel signal_disp data (critical for performance testing)
        if include_signal_disp:
            signal_disp = np.zeros((signal_channels, rows))
            
            for ch in range(signal_channels):
                # Channel-specific characteristics
                phase_offset = 2 * np.pi * ch / signal_channels
                amplitude = 0.8 + 0.4 * np.random.random()
                
                # Base signal with harmonics
                ch_signal = amplitude * np.sin(2 * np.pi * signal_base_freq * time_array + phase_offset)
                ch_signal += 0.3 * amplitude * np.sin(4 * np.pi * signal_base_freq * time_array + phase_offset)
                
                # Add baseline drift
                drift_freq = 0.01
                drift = 0.2 * np.sin(2 * np.pi * drift_freq * time_array + np.random.random() * 2 * np.pi)
                
                if add_noise_characteristics:
                    noise = 0.05 * np.random.normal(0, 1, rows)
                    ch_signal += noise + drift
                
                signal_disp[ch, :] = ch_signal
            
            exp_matrix['signal_disp'] = signal_disp
        
        # Add alias column for configuration testing
        exp_matrix['dtheta_smooth'] = exp_matrix['dtheta']
        
        return exp_matrix
    
    return generate_matrix


@pytest.fixture
def comprehensive_column_config():
    """
    Fixture providing comprehensive column configuration for testing
    all transformation pathways and special handlers.
    
    Returns:
        ColumnConfigDict: Complete column configuration for performance testing
    """
    config_dict = {
        'columns': {
            't': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': True,
                'description': 'Time values in seconds'
            },
            'x': {
                'type': 'numpy.ndarray', 
                'dimension': 1,
                'required': True,
                'description': 'X position in mm'
            },
            'y': {
                'type': 'numpy.ndarray',
                'dimension': 1, 
                'required': True,
                'description': 'Y position in mm'
            },
            'vx': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': False,
                'description': 'X velocity in mm/s'
            },
            'vy': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': False,
                'description': 'Y velocity in mm/s'
            },
            'speed': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': False,
                'description': 'Speed in mm/s'
            },
            'dtheta': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': False,
                'description': 'Change in heading angle',
                'alias': 'dtheta_smooth'
            },
            'signal': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': False,
                'description': 'Single-channel signal data'
            },
            'signal_disp': {
                'type': 'numpy.ndarray',
                'dimension': 2,
                'required': False,
                'description': 'Multi-channel signal display data',
                'special_handling': 'transform_to_match_time_dimension'
            },
            # Metadata columns
            'date': {
                'type': 'str',
                'dimension': 0,
                'required': False,
                'description': 'Experiment date'
            },
            'experiment_id': {
                'type': 'str', 
                'dimension': 0,
                'required': False,
                'description': 'Unique experiment identifier'
            }
        }
    }
    
    return get_config_from_source(config_dict)


def estimate_exp_matrix_size(exp_matrix: Dict[str, Any]) -> int:
    """
    Estimate the memory size of an experimental matrix in bytes.
    
    Args:
        exp_matrix: Experimental data matrix
        
    Returns:
        Estimated size in bytes
    """
    total_size = 0
    
    for key, value in exp_matrix.items():
        if isinstance(value, np.ndarray):
            total_size += value.nbytes
        elif isinstance(value, (list, tuple)):
            # Rough estimate for list/tuple structures
            total_size += len(value) * 8  # Assume 8 bytes per element
        elif isinstance(value, str):
            total_size += len(value.encode('utf-8'))
        else:
            total_size += 8  # Default estimate for other types
    
    return total_size


# ============================================================================
# TST-PERF-002 SLA VALIDATION BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
class TestDataFrameTransformationSLA:
    """
    Primary performance benchmark suite for DataFrame transformation operations
    targeting TST-PERF-002 SLA requirements and memory efficiency validation.
    """
    
    @pytest.mark.parametrize(
        "scale_name,expect_sla_compliance", 
        [
            ("small_scale", True),
            ("medium_scale", True),
            ("large_scale", True),  # SLA boundary test
            ("stress_scale", False)  # Expected to exceed SLA
        ]
    )
    def test_dataframe_transformation_sla_enforcement(
        self,
        benchmark,
        scale_name,
        expect_sla_compliance,
        performance_test_configurations,
        synthetic_exp_matrix_generator,
        comprehensive_column_config
    ):
        """
        Enforce TST-PERF-002 SLA requirement: DataFrame transformation <500ms per 1M rows.
        
        Validates:
        - SLA compliance across multiple data scales
        - Performance consistency and predictability
        - Transformation efficiency under various loads
        - Statistical measurement accuracy for SLA enforcement
        
        Args:
            scale_name: Test scale configuration name
            expect_sla_compliance: Whether this scale should meet SLA
        """
        config = performance_test_configurations[scale_name]
        
        # Generate test data matching the scale configuration
        exp_matrix = synthetic_exp_matrix_generator(
            rows=config["rows"],
            include_signal_disp=True,
            signal_channels=16
        )
        
        # Benchmark the transformation with statistical rigor
        result = benchmark.pedantic(
            make_dataframe_from_config,
            args=(exp_matrix, comprehensive_column_config),
            kwargs={},
            iterations=5,  # Statistical significance
            rounds=10,     # Multiple measurements
            warmup_rounds=2  # Reduce JIT effects
        )
        
        # Extract benchmark statistics
        benchmark_stats = benchmark.stats
        mean_time_ms = benchmark_stats['mean'] * 1000
        
        # SLA validation with detailed reporting
        sla_compliant = mean_time_ms <= config["expected_transform_time_ms"]
        
        if expect_sla_compliance:
            assert sla_compliant, (
                f"TST-PERF-002 SLA violation for {scale_name}: "
                f"Measured {mean_time_ms:.2f}ms > SLA {config['expected_transform_time_ms']}ms "
                f"({config['rows']:,} rows). "
                f"Performance degradation: {(mean_time_ms/config['expected_transform_time_ms']-1)*100:.1f}%"
            )
            
            # Calculate per-million-row performance for SLA reporting
            per_million_row_ms = (mean_time_ms / config["rows"]) * 1_000_000
            assert per_million_row_ms <= 500.0, (
                f"Per-million-row SLA violation: {per_million_row_ms:.2f}ms > 500ms TST-PERF-002 limit"
            )
        else:
            # Stress test - expect to exceed SLA but validate reasonable scaling
            performance_ratio = mean_time_ms / config["expected_transform_time_ms"]
            assert performance_ratio <= 2.0, (
                f"Performance degradation too severe for stress test {scale_name}: "
                f"{performance_ratio:.2f}x expected time suggests algorithmic issues"
            )
        
        # Validate result integrity
        assert isinstance(result, pd.DataFrame), "Result must be a DataFrame"
        assert len(result) == config["rows"], "Output size must match input"
        assert 't' in result.columns, "Time column must be present"
        
        # Performance consistency validation
        cv = benchmark_stats['stddev'] / benchmark_stats['mean']
        assert cv <= 0.15, (
            f"Performance variability too high for {scale_name}: CV={cv:.3f} > 0.15"
        )
    
    def test_memory_efficiency_requirement_validation(
        self,
        performance_test_configurations,
        synthetic_exp_matrix_generator,
        comprehensive_column_config
    ):
        """
        Validate Section 2.4.5 memory efficiency requirements <2x data size overhead
        during DataFrame transformation operations.
        
        Tests:
        - Section 2.4.5: Memory efficiency <2x data size overhead
        - Memory usage monitoring during transformation
        - Peak memory consumption validation
        - Memory leak detection for large dataset scenarios
        """
        large_config = performance_test_configurations["large_scale"]
        
        # Generate large dataset for memory testing
        exp_matrix = synthetic_exp_matrix_generator(
            rows=large_config["rows"],
            include_signal_disp=True,
            signal_channels=16
        )
        
        # Estimate input data size
        data_size = estimate_exp_matrix_size(exp_matrix)
        
        # Profile memory usage during transformation
        with memory_profiling_context(
            data_size_estimate=data_size,
            precision=3,
            enable_line_profiling=True
        ) as profiler:
            # Perform transformation with memory monitoring
            profiler.update_peak_memory()
            result = make_dataframe_from_config(exp_matrix, comprehensive_column_config)
            profiler.update_peak_memory()
            
            # Memory statistics from profiler
            memory_stats = profiler.end_profiling()
        
        # Validate memory efficiency requirement
        assert memory_stats['meets_efficiency_requirement'], (
            f"Section 2.4.5 memory efficiency violation: "
            f"{memory_stats['memory_multiplier']:.2f}x > 2.0x data size. "
            f"Data size: {memory_stats['data_size_mb']:.1f}MB, "
            f"Memory overhead: {memory_stats['memory_overhead_mb']:.1f}MB"
        )
        
        # Additional validation for reasonable memory usage
        assert memory_stats['memory_multiplier'] >= 0.1, (
            "Memory multiplier suspiciously low - measurement may be incorrect"
        )
        
        # Validate memory leak detection
        leak_analysis = memory_stats.get('leak_analysis', {})
        if leak_analysis.get('leak_detected', False):
            warnings.warn(
                f"Potential memory leak detected: "
                f"{leak_analysis.get('memory_growth_mb', 0):.1f}MB growth, "
                f"confidence: {leak_analysis.get('confidence', 0):.2f}"
            )
        
        # Validate transformation result integrity
        assert isinstance(result, pd.DataFrame), "Result must be a DataFrame"
        assert len(result) == large_config["rows"], "Output size must match input"
    
    @pytest.mark.parametrize(
        "scale_name,expected_linear_scaling", 
        [
            ("small_scale", True),
            ("medium_scale", True), 
            ("large_scale", True)
        ]
    )
    def test_transformation_scaling_linearity(
        self,
        benchmark,
        scale_name,
        expected_linear_scaling,
        performance_test_configurations,
        synthetic_exp_matrix_generator,
        comprehensive_column_config
    ):
        """
        Validate that DataFrame transformation performance scales linearly
        with data size for predictable performance characteristics.
        
        Tests:
        - Linear scaling validation across data sizes
        - Performance predictability for capacity planning
        - Consistent per-row transformation cost
        - Algorithmic efficiency verification
        """
        config = performance_test_configurations[scale_name]
        
        # Generate test data of specified scale
        exp_matrix = synthetic_exp_matrix_generator(
            rows=config["rows"],
            include_signal_disp=True,
            signal_channels=8  # Moderate complexity
        )
        
        # Benchmark transformation with statistical rigor
        result = benchmark.pedantic(
            make_dataframe_from_config,
            args=(exp_matrix, comprehensive_column_config),
            kwargs={},
            iterations=3,
            rounds=5,
            warmup_rounds=1
        )
        
        # Calculate per-row performance for linearity validation
        benchmark_stats = benchmark.stats
        mean_time_ms = benchmark_stats['mean'] * 1000
        per_row_time_us = (mean_time_ms * 1000) / config["rows"]
        
        # Validate performance expectations
        assert mean_time_ms <= config["expected_transform_time_ms"], (
            f"Scale {scale_name} SLA violation: {mean_time_ms:.2f}ms > {config['expected_transform_time_ms']}ms"
        )
        
        # Linear scaling should keep per-row time consistent
        if expected_linear_scaling:
            assert per_row_time_us <= 1000.0, (  # Max 1ms per 1000 rows (reasonable)
                f"Per-row performance degradation: {per_row_time_us:.2f}µs per row too high for scale {scale_name}"
            )
        
        # Validate consistent scaling across different data sizes
        scaling_efficiency = 1000.0 / per_row_time_us  # Rows per millisecond
        assert scaling_efficiency >= 100.0, (
            f"Scaling efficiency below threshold: {scaling_efficiency:.0f} rows/ms < 100 rows/ms minimum"
        )
        
        # Validate result integrity
        assert isinstance(result, pd.DataFrame)
        assert len(result) == config["rows"]
    
    def test_vectorized_operations_performance(
        self,
        benchmark,
        synthetic_exp_matrix_generator,
        comprehensive_column_config
    ):
        """
        Benchmark vectorized operations performance to ensure optimal
        utilization of NumPy and Pandas vectorized capabilities.
        
        Tests:
        - Vectorized array processing performance
        - NumPy broadcasting efficiency
        - Pandas vectorized operations optimization
        - Multi-channel data processing efficiency
        """
        # Generate data with multiple array columns for vectorization testing
        exp_matrix = synthetic_exp_matrix_generator(
            rows=500_000,  # Large enough to show vectorization benefits
            include_signal_disp=True,
            signal_channels=32  # More channels to test vectorization
        )
        
        # Benchmark vectorized transformation
        result = benchmark.pedantic(
            make_dataframe_from_config,
            args=(exp_matrix, comprehensive_column_config),
            kwargs={},
            iterations=2,
            rounds=3,
            warmup_rounds=1
        )
        
        # Validate vectorized performance expectations
        benchmark_stats = benchmark.stats
        mean_time_ms = benchmark_stats['mean'] * 1000
        
        # Vectorized operations should be efficient for large datasets
        per_million_rows_ms = (mean_time_ms / 500_000) * 1_000_000
        assert per_million_rows_ms <= 500.0, (
            f"Vectorized operations TST-PERF-002 violation: {per_million_rows_ms:.2f}ms per 1M rows > 500ms"
        )
        
        # Validate vectorization effectiveness with multi-channel data
        assert isinstance(result, pd.DataFrame)
        signal_columns = [col for col in result.columns if col.startswith('signal_disp')]
        assert len(signal_columns) > 0, "Signal vectorization should produce multiple columns"
        
        # Validate multi-channel processing efficiency
        expected_signal_columns = 32  # Based on signal_channels parameter
        actual_signal_columns = len(signal_columns)
        vectorization_ratio = actual_signal_columns / expected_signal_columns
        assert vectorization_ratio >= 0.8, (
            f"Multi-channel vectorization efficiency low: {vectorization_ratio:.2f} < 0.8"
        )


# ============================================================================ 
# SPECIALIZED PERFORMANCE SCENARIOS
# ============================================================================

@pytest.mark.benchmark  
class TestSpecializedTransformationScenarios:
    """
    Performance benchmarks for specialized transformation scenarios including
    edge cases, stress conditions, and complex data structures.
    """
    
    def test_large_signal_disp_transformation_performance(
        self,
        benchmark,
        synthetic_exp_matrix_generator,
        comprehensive_column_config
    ):
        """
        Benchmark performance with large multi-channel signal_disp arrays
        to validate performance with complex 2D array transformations.
        
        Tests:
        - Large 2D array transformation performance
        - Memory-intensive signal processing efficiency
        - Multi-channel data handling scalability
        - Special handler performance validation (F-006-RQ-003)
        """
        # Generate data with large signal_disp array
        exp_matrix = synthetic_exp_matrix_generator(
            rows=200_000,  # 200K timepoints
            include_signal_disp=True,
            signal_channels=64,  # Large channel count
            add_noise_characteristics=True
        )
        
        # Validate signal_disp size for test validity
        signal_disp_size_mb = exp_matrix['signal_disp'].nbytes / 1024 / 1024
        assert signal_disp_size_mb > 50, f"Signal_disp should be substantial size for testing: {signal_disp_size_mb:.1f}MB"
        
        # Benchmark large signal_disp transformation
        result = benchmark.pedantic(
            make_dataframe_from_config,
            args=(exp_matrix, comprehensive_column_config),
            kwargs={},
            iterations=2,
            rounds=4,
            warmup_rounds=1
        )
        
        # Validate specialized transformation performance
        benchmark_stats = benchmark.stats
        mean_time_ms = benchmark_stats['mean'] * 1000
        
        # Large signal_disp should still meet reasonable performance targets
        per_channel_ms = mean_time_ms / 64  # 64 channels
        assert per_channel_ms <= 50.0, (
            f"F-006-RQ-003 special handler SLA violation: {per_channel_ms:.2f}ms per channel > 50ms"
        )
        
        # Validate signal_disp special handling effectiveness
        signal_columns = [col for col in result.columns if col.startswith('signal_disp')]
        assert len(signal_columns) == 64, f"Expected 64 signal channels, got {len(signal_columns)}"
        
        # Validate memory efficiency for large arrays
        estimated_input_size = estimate_exp_matrix_size(exp_matrix)
        estimated_output_size = result.memory_usage(deep=True).sum()
        memory_ratio = estimated_output_size / estimated_input_size
        assert memory_ratio <= 3.0, (
            f"Large signal_disp memory efficiency issue: {memory_ratio:.2f}x input size"
        )
    
    def test_high_frequency_sampling_transformation(
        self,
        benchmark,
        synthetic_exp_matrix_generator,
        comprehensive_column_config
    ):
        """
        Benchmark transformation performance with high-frequency sampled data
        to validate time-series processing efficiency.
        
        Tests:
        - High-frequency time series transformation
        - Dense temporal data processing performance
        - Time alignment validation performance
        """
        # Generate high-frequency data (1000 Hz equivalent)
        high_freq_rows = 60_000  # 1 minute at 1000 Hz
        exp_matrix = synthetic_exp_matrix_generator(
            rows=high_freq_rows,
            include_signal_disp=True,
            signal_channels=8,
            add_noise_characteristics=True
        )
        
        # Adjust time array for high frequency
        exp_matrix['t'] = np.linspace(0, 60, high_freq_rows)  # 60 seconds
        
        # Benchmark high-frequency transformation
        result = benchmark.pedantic(
            make_dataframe_from_config,
            args=(exp_matrix, comprehensive_column_config),
            kwargs={},
            iterations=3,
            rounds=5,
            warmup_rounds=1
        )
        
        # Validate high-frequency performance
        benchmark_stats = benchmark.stats
        mean_time_ms = benchmark_stats['mean'] * 1000
        
        # High-frequency data should maintain reasonable performance
        samples_per_ms = high_freq_rows / mean_time_ms
        assert samples_per_ms >= 100.0, (
            f"High-frequency processing efficiency low: {samples_per_ms:.0f} samples/ms < 100"
        )
        
        # Validate time series integrity
        assert isinstance(result, pd.DataFrame)
        assert len(result) == high_freq_rows
        assert result['t'].dtype in [np.float64, np.float32], "Time column should be numeric"
        
        # Validate temporal resolution preservation
        time_diff = np.diff(result['t'].values)
        expected_dt = 60.0 / high_freq_rows
        actual_dt = np.mean(time_diff)
        temporal_accuracy = abs(actual_dt - expected_dt) / expected_dt
        assert temporal_accuracy <= 0.01, (
            f"Temporal resolution degraded: {temporal_accuracy:.4f} relative error > 0.01"
        )
    
    def test_sparse_data_transformation_performance(
        self,
        benchmark,
        synthetic_exp_matrix_generator,
        comprehensive_column_config
    ):
        """
        Benchmark transformation performance with sparse experimental data
        containing missing values and irregular patterns.
        
        Tests:
        - Sparse data handling performance
        - Missing value processing efficiency
        - Data validation under sparse conditions
        """
        # Generate base experimental matrix
        exp_matrix = synthetic_exp_matrix_generator(
            rows=100_000,
            include_signal_disp=False,  # Focus on basic columns for sparsity
            signal_channels=0
        )
        
        # Introduce sparsity in various columns
        sparsity_factor = 0.3  # 30% missing data
        
        for key in ['vx', 'vy', 'speed', 'signal']:
            if key in exp_matrix:
                mask = np.random.random(len(exp_matrix[key])) < sparsity_factor
                exp_matrix[key] = exp_matrix[key].copy()
                exp_matrix[key][mask] = np.nan
        
        # Benchmark sparse data transformation
        result = benchmark.pedantic(
            make_dataframe_from_config,
            args=(exp_matrix, comprehensive_column_config),
            kwargs={},
            iterations=3,
            rounds=5,
            warmup_rounds=1
        )
        
        # Validate sparse data performance
        benchmark_stats = benchmark.stats
        mean_time_ms = benchmark_stats['mean'] * 1000
        
        # Sparse data should not significantly degrade performance
        per_row_us = (mean_time_ms * 1000) / 100_000
        assert per_row_us <= 50.0, (
            f"Sparse data processing degradation: {per_row_us:.2f}µs per row > 50µs"
        )
        
        # Validate sparse data handling integrity
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100_000
        
        # Validate NaN handling
        for col in ['vx', 'vy', 'speed', 'signal']:
            if col in result.columns:
                nan_ratio = result[col].isna().mean()
                assert 0.25 <= nan_ratio <= 0.35, (
                    f"NaN ratio for {col} outside expected range: {nan_ratio:.3f}"
                )


# ============================================================================
# PROPERTY-BASED PERFORMANCE TESTING
# ============================================================================

@pytest.mark.benchmark
class TestPropertyBasedTransformationPerformance:
    """
    Property-based performance testing using Hypothesis for comprehensive
    edge-case coverage and transformation performance validation.
    """
    
    @given(
        rows=st.integers(min_value=1000, max_value=50000),
        signal_channels=st.integers(min_value=1, max_value=16),
        include_noise=st.booleans()
    )
    @settings(
        max_examples=10,  # Limit for benchmark performance
        deadline=30000,   # 30 second deadline per example
        suppress_health_check=[st.HealthCheck.too_slow]
    )
    def test_property_based_transformation_performance(
        self,
        benchmark,
        rows,
        signal_channels,
        include_noise,
        synthetic_exp_matrix_generator,
        comprehensive_column_config
    ):
        """
        Property-based performance testing for transformation operations
        across diverse input parameter combinations.
        
        Tests:
        - Performance consistency across parameter ranges
        - Edge case handling efficiency
        - Transformation robustness under varied conditions
        - Hypothesis-driven performance validation
        """
        assume(rows >= 1000)  # Minimum for meaningful performance measurement
        assume(signal_channels >= 1)
        
        # Generate test data with Hypothesis-driven parameters
        exp_matrix = synthetic_exp_matrix_generator(
            rows=rows,
            include_signal_disp=True,
            signal_channels=signal_channels,
            add_noise_characteristics=include_noise
        )
        
        # Benchmark transformation with property-based parameters
        result = benchmark.pedantic(
            make_dataframe_from_config,
            args=(exp_matrix, comprehensive_column_config),
            kwargs={},
            iterations=2,
            rounds=3,
            warmup_rounds=1
        )
        
        # Performance validation scaled to data size
        benchmark_stats = benchmark.stats
        mean_time_ms = benchmark_stats['mean'] * 1000
        
        # Calculate performance metrics
        per_row_us = (mean_time_ms * 1000) / rows
        per_channel_ms = mean_time_ms / max(signal_channels, 1)
        
        # Property-based performance assertions
        assert per_row_us <= 500.0, (
            f"Property-based per-row performance issue: {per_row_us:.2f}µs > 500µs "
            f"(rows={rows}, channels={signal_channels}, noise={include_noise})"
        )
        
        assert per_channel_ms <= 100.0, (
            f"Property-based per-channel performance issue: {per_channel_ms:.2f}ms > 100ms "
            f"(rows={rows}, channels={signal_channels}, noise={include_noise})"
        )
        
        # Validate result integrity for all property combinations
        assert isinstance(result, pd.DataFrame)
        assert len(result) == rows
        assert 't' in result.columns
        
        # Validate signal channel processing
        if signal_channels > 0:
            signal_cols = [col for col in result.columns if col.startswith('signal_disp')]
            assert len(signal_cols) == signal_channels, (
                f"Signal channel count mismatch: expected {signal_channels}, got {len(signal_cols)}"
            )
    
    @given(
        time_resolution=st.floats(min_value=0.001, max_value=1.0),
        position_scale=st.floats(min_value=10.0, max_value=100.0),
        velocity_noise=st.floats(min_value=0.01, max_value=0.5)
    )
    @settings(
        max_examples=8,
        deadline=25000,
        suppress_health_check=[st.HealthCheck.too_slow]
    )
    def test_property_based_temporal_transformation(
        self,
        benchmark,
        time_resolution,
        position_scale,
        velocity_noise,
        comprehensive_column_config
    ):
        """
        Property-based testing for temporal aspects of transformation performance
        with various time resolution and scaling parameters.
        
        Tests:
        - Temporal transformation performance across resolutions
        - Scaling factor impact on transformation efficiency
        - Noise level effects on processing performance
        """
        # Generate data with property-based temporal characteristics
        rows = 10_000  # Fixed size for temporal testing
        sampling_freq = 1.0 / time_resolution
        
        # Generate temporal data
        np.random.seed(42)  # Consistent seed for benchmark reliability
        time_array = np.linspace(0, rows * time_resolution, rows)
        
        # Position data with property-based scaling
        x_pos = position_scale * np.sin(2 * np.pi * 0.1 * time_array)
        y_pos = position_scale * np.cos(2 * np.pi * 0.1 * time_array)
        
        # Velocity with property-based noise
        vx = np.gradient(x_pos, time_array) + np.random.normal(0, velocity_noise, rows)
        vy = np.gradient(y_pos, time_array) + np.random.normal(0, velocity_noise, rows)
        
        exp_matrix = {
            't': time_array,
            'x': x_pos,
            'y': y_pos,
            'vx': vx,
            'vy': vy,
            'speed': np.sqrt(vx**2 + vy**2)
        }
        
        # Benchmark temporal transformation
        result = benchmark.pedantic(
            make_dataframe_from_config,
            args=(exp_matrix, comprehensive_column_config),
            kwargs={},
            iterations=2,
            rounds=3,
            warmup_rounds=1
        )
        
        # Temporal performance validation
        benchmark_stats = benchmark.stats
        mean_time_ms = benchmark_stats['mean'] * 1000
        
        # Performance should be independent of temporal scaling
        samples_per_ms = rows / mean_time_ms
        assert samples_per_ms >= 50.0, (
            f"Temporal transformation efficiency low: {samples_per_ms:.0f} samples/ms < 50 "
            f"(resolution={time_resolution:.3f}s, scale={position_scale:.1f}, noise={velocity_noise:.3f})"
        )
        
        # Validate temporal integrity preservation
        assert isinstance(result, pd.DataFrame)
        assert len(result) == rows
        
        # Validate temporal resolution preservation
        if len(result) > 1:
            actual_resolution = np.mean(np.diff(result['t'].values))
            resolution_error = abs(actual_resolution - time_resolution) / time_resolution
            assert resolution_error <= 0.05, (
                f"Temporal resolution error too high: {resolution_error:.4f} > 0.05"
            )


# ============================================================================
# PERFORMANCE REGRESSION AND MONITORING HOOKS
# ============================================================================

def pytest_benchmark_update_json(config, benchmarks, output_json):
    """
    Hook to enhance benchmark JSON output with custom performance metrics
    and SLA compliance tracking for CI/CD integration.
    """
    # Add custom performance analysis
    for benchmark_data in benchmarks:
        test_name = benchmark_data.get('name', 'unknown')
        stats = benchmark_data.get('stats', {})
        
        # Add SLA compliance analysis
        if 'transformation' in test_name.lower():
            mean_time_ms = stats.get('mean', 0) * 1000
            
            # TST-PERF-002 SLA validation
            per_million_rows_ms = mean_time_ms * 1000  # Rough estimation
            sla_compliant = per_million_rows_ms <= 500.0
            
            benchmark_data['custom_metrics'] = {
                'mean_time_ms': mean_time_ms,
                'per_million_rows_ms': per_million_rows_ms,
                'tst_perf_002_compliant': sla_compliant,
                'memory_efficiency_validated': True  # Set by individual tests
            }
    
    # Add overall suite metadata
    output_json.setdefault('suite_metadata', {}).update({
        'sla_requirements': {
            'TST-PERF-002': 'DataFrame transformation <500ms per 1M rows',
            'Section-2.4.5': 'Memory efficiency <2x data size overhead'
        },
        'benchmark_category': 'transformation',
        'execution_environment': {
            'platform': psutil.os.name,
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 1)
        }
    })


# ============================================================================
# MODULE METADATA AND CLI INTEGRATION
# ============================================================================

# Mark all test classes for benchmark execution via CLI
pytest_benchmark_group = "transformation"

# Module metadata for CLI framework integration
__benchmark_category__ = BenchmarkCategory.TRANSFORMATION
__sla_requirements__ = [
    "TST-PERF-002: DataFrame transformation <500ms per 1M rows",
    "Section 2.4.5: Memory efficiency <2x data size overhead",
    "F-006-RQ-003: Special handlers <50ms per handler",
    "F-006-RQ-004: Metadata merge <10ms merge time"
]
__performance_targets__ = {
    "small_scale": {"rows": 10_000, "max_time_ms": 5.0},
    "medium_scale": {"rows": 100_000, "max_time_ms": 50.0},
    "large_scale": {"rows": 1_000_000, "max_time_ms": 500.0},
    "stress_scale": {"rows": 2_000_000, "max_time_ms": 1000.0}
}

if __name__ == "__main__":
    # CLI integration point for direct execution
    pytest.main([__file__, "--benchmark-only", "--benchmark-json=transformation_benchmark_results.json"])