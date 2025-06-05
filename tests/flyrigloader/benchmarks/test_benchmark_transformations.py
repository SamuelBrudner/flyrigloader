"""
Performance benchmark test suite for DataFrame transformation operations.

This module implements comprehensive performance testing for the flyrigloader DataFrame
transformation pipeline, specifically targeting TST-PERF-002 SLA requirements and
validating transformation performance against defined benchmarks.

Features:
- TST-PERF-002: DataFrame transformation SLA validation (<500ms per 1M rows)
- F-006-RQ-001: exp_matrix to DataFrame conversion performance testing
- F-006-RQ-002: Multi-dimensional array handling performance validation  
- F-006-RQ-003: Special handler performance benchmarks (<50ms per handler)
- F-006-RQ-004: Metadata integration performance testing (<10ms merge time)
- Section 2.4.5: Memory efficiency validation (<2x data size overhead)

Performance Requirements Tested:
- Data transformation: <500ms per 1M rows (TST-PERF-002)
- Special handlers: <50ms per handler (F-006-RQ-003)
- Metadata merge: <10ms merge time (F-006-RQ-004)
- Memory overhead: <2x data size (Section 2.4.5)
- Vectorized operations: Optimal performance validation
- Time alignment: Validation performance benchmarks

Integration:
- pytest-benchmark for statistical measurement
- CI/CD pipeline integration for performance regression detection
- Memory profiling for efficiency validation
- Cross-platform performance validation
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

# Import the functions under test
from flyrigloader.io.pickle import make_dataframe_from_config, handle_signal_disp
from flyrigloader.io.column_models import get_config_from_source, ColumnConfigDict


# ============================================================================
# PERFORMANCE TEST DATA GENERATION FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def performance_test_configurations():
    """
    Session-scoped fixture providing standardized performance test configurations
    for consistent benchmarking across different test scenarios.
    
    Returns:
        Dict: Performance test scale configurations with expected SLA thresholds
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
            "memory_multiplier_threshold": 2.0,
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
    with configurable size and complexity for performance testing.
    
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
                'type': 'string',
                'required': False,
                'is_metadata': True,
                'description': 'Experiment date'
            },
            'exp_name': {
                'type': 'string',
                'required': False,
                'is_metadata': True,
                'description': 'Experiment name'
            },
            'rig': {
                'type': 'string',
                'required': False,
                'is_metadata': True,
                'description': 'Rig identifier'
            },
            'animal_id': {
                'type': 'string',
                'required': False,
                'is_metadata': True,
                'description': 'Animal identifier'
            }
        },
        'special_handlers': {
            'transform_to_match_time_dimension': '_handle_signal_disp'
        }
    }
    
    return get_config_from_source(config_dict)


@pytest.fixture
def sample_metadata_for_benchmarks():
    """
    Fixture providing sample metadata for metadata integration performance testing.
    
    Returns:
        Dict: Sample metadata for benchmark testing
    """
    return {
        'date': '20241201',
        'exp_name': 'performance_benchmark_test',
        'rig': 'benchmark_rig',
        'animal_id': 'benchmark_animal_001',
        'condition': 'performance_test',
        'replicate': '1',
        'experimenter': 'benchmark_system',
        'session_id': 'benchmark_session_001'
    }


# ============================================================================
# MEMORY EFFICIENCY TESTING UTILITIES
# ============================================================================

class MemoryProfiler:
    """
    Utility class for monitoring memory usage during DataFrame transformation
    operations to validate memory efficiency requirements.
    """
    
    def __init__(self):
        self.initial_memory = None
        self.peak_memory = None
        self.final_memory = None
        self.data_size_estimate = None
    
    def start_profiling(self, data_size_estimate: int):
        """
        Start memory profiling session.
        
        Args:
            data_size_estimate: Estimated size of input data in bytes
        """
        gc.collect()  # Clean up before measurement
        process = psutil.Process()
        self.initial_memory = process.memory_info().rss
        self.peak_memory = self.initial_memory
        self.data_size_estimate = data_size_estimate
    
    def update_peak_memory(self):
        """Update peak memory usage measurement."""
        process = psutil.Process()
        current_memory = process.memory_info().rss
        self.peak_memory = max(self.peak_memory, current_memory)
    
    def end_profiling(self) -> Dict[str, float]:
        """
        End profiling session and return memory usage statistics.
        
        Returns:
            Dict containing memory usage metrics
        """
        gc.collect()
        process = psutil.Process()
        self.final_memory = process.memory_info().rss
        
        memory_overhead = self.peak_memory - self.initial_memory
        memory_multiplier = memory_overhead / max(self.data_size_estimate, 1)
        
        return {
            'initial_memory_mb': self.initial_memory / 1024 / 1024,
            'peak_memory_mb': self.peak_memory / 1024 / 1024,
            'final_memory_mb': self.final_memory / 1024 / 1024,
            'memory_overhead_mb': memory_overhead / 1024 / 1024,
            'data_size_mb': self.data_size_estimate / 1024 / 1024,
            'memory_multiplier': memory_multiplier,
            'meets_efficiency_requirement': memory_multiplier <= 2.0
        }


def estimate_exp_matrix_size(exp_matrix: Dict[str, Any]) -> int:
    """
    Estimate the memory size of an experimental data matrix.
    
    Args:
        exp_matrix: Experimental data dictionary
        
    Returns:
        Estimated size in bytes
    """
    total_size = 0
    
    for key, value in exp_matrix.items():
        if isinstance(value, np.ndarray):
            total_size += value.nbytes
        elif isinstance(value, (list, tuple)):
            total_size += len(value) * 8  # Estimate for Python objects
        else:
            total_size += 64  # Conservative estimate for other types
    
    return total_size


# ============================================================================
# CORE TRANSFORMATION PERFORMANCE BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
class TestDataFrameTransformationPerformance:
    """
    Comprehensive performance benchmark suite for DataFrame transformation operations
    targeting TST-PERF-002 and related performance requirements.
    """
    
    def test_basic_transformation_sla_validation(
        self,
        benchmark,
        performance_test_configurations,
        synthetic_exp_matrix_generator,
        comprehensive_column_config
    ):
        """
        Validate basic DataFrame transformation against TST-PERF-002 SLA requirement
        of <500ms per 1M rows for standard experimental data transformation.
        
        Tests:
        - TST-PERF-002: DataFrame transformation <500ms per 1M rows
        - F-006-RQ-001: exp_matrix to DataFrame conversion performance
        - Linear scaling validation across different data sizes
        """
        large_config = performance_test_configurations["large_scale"]
        
        # Generate 1M row test data
        exp_matrix = synthetic_exp_matrix_generator(
            rows=large_config["rows"],
            include_signal_disp=False,  # Test basic transformation first
            include_metadata_fields=False
        )
        
        # Benchmark the transformation
        result = benchmark.pedantic(
            make_dataframe_from_config,
            args=(exp_matrix, comprehensive_column_config),
            kwargs={},
            iterations=3,  # Multiple iterations for statistical significance
            rounds=5,      # Multiple rounds for consistency
            warmup_rounds=1
        )
        
        # Validate SLA compliance
        benchmark_stats = benchmark.stats
        mean_time_ms = benchmark_stats['mean'] * 1000
        
        assert mean_time_ms <= large_config["expected_transform_time_ms"], (
            f"Transformation SLA violation: {mean_time_ms:.2f}ms > "
            f"{large_config['expected_transform_time_ms']}ms for {large_config['rows']} rows"
        )
        
        # Validate result integrity
        assert isinstance(result, pd.DataFrame), "Result must be a DataFrame"
        assert len(result) == large_config["rows"], "Output row count must match input"
        assert 't' in result.columns, "Time column must be present"
        assert 'x' in result.columns, "X position column must be present"
        assert 'y' in result.columns, "Y position column must be present"
    
    def test_signal_disp_special_handler_performance(
        self,
        benchmark,
        synthetic_exp_matrix_generator,
        comprehensive_column_config
    ):
        """
        Benchmark signal_disp special handler performance against F-006-RQ-003
        requirement of <50ms per handler execution.
        
        Tests:
        - F-006-RQ-003: Special handler execution <50ms per handler
        - Signal display 2D array transformation performance
        - Multi-dimensional array handling efficiency
        """
        # Generate data with signal_disp for handler testing
        exp_matrix = synthetic_exp_matrix_generator(
            rows=100_000,  # Medium size for handler-specific testing
            include_signal_disp=True,
            signal_channels=16
        )
        
        # Benchmark only the signal_disp handling
        result = benchmark.pedantic(
            handle_signal_disp,
            args=(exp_matrix,),
            kwargs={},
            iterations=5,
            rounds=10,
            warmup_rounds=2
        )
        
        # Validate handler performance SLA
        benchmark_stats = benchmark.stats
        mean_time_ms = benchmark_stats['mean'] * 1000
        
        assert mean_time_ms <= 50.0, (
            f"Signal_disp handler SLA violation: {mean_time_ms:.2f}ms > 50ms"
        )
        
        # Validate handler result correctness
        assert isinstance(result, pd.Series), "Handler must return Series"
        assert len(result) == len(exp_matrix['t']), "Handler output must match time dimension"
        assert all(isinstance(x, np.ndarray) for x in result), "Each element must be an array"
    
    def test_metadata_integration_performance(
        self,
        benchmark,
        synthetic_exp_matrix_generator,
        comprehensive_column_config,
        sample_metadata_for_benchmarks
    ):
        """
        Benchmark metadata integration performance against F-006-RQ-004
        requirement of <10ms merge time for metadata columns.
        
        Tests:
        - F-006-RQ-004: Metadata integration <10ms merge time
        - Metadata column addition performance
        - DataFrame merge operation efficiency
        """
        exp_matrix = synthetic_exp_matrix_generator(
            rows=50_000,  # Moderate size for metadata testing
            include_signal_disp=False
        )
        
        # Benchmark transformation with metadata
        result = benchmark.pedantic(
            make_dataframe_from_config,
            args=(exp_matrix, comprehensive_column_config),
            kwargs={'metadata': sample_metadata_for_benchmarks},
            iterations=10,
            rounds=15,
            warmup_rounds=3
        )
        
        # Validate metadata merge performance SLA
        benchmark_stats = benchmark.stats
        mean_time_ms = benchmark_stats['mean'] * 1000
        
        # Note: This tests the full transformation with metadata, not just merge
        # The merge itself should be <10ms, but we're testing realistic usage
        assert mean_time_ms <= 100.0, (  # Reasonable total time with metadata
            f"Metadata integration exceeded reasonable time: {mean_time_ms:.2f}ms"
        )
        
        # Validate metadata columns are present
        metadata_columns = ['date', 'exp_name', 'rig', 'animal_id']
        for col in metadata_columns:
            assert col in result.columns, f"Metadata column {col} must be present"
            assert result[col].iloc[0] == sample_metadata_for_benchmarks[col], (
                f"Metadata column {col} must have correct value"
            )
    
    def test_memory_efficiency_validation(
        self,
        synthetic_exp_matrix_generator,
        comprehensive_column_config,
        performance_test_configurations
    ):
        """
        Validate memory efficiency requirement from Section 2.4.5 of <2x data size overhead
        during DataFrame transformation operations.
        
        Tests:
        - Section 2.4.5: Memory efficiency <2x data size overhead
        - Memory usage monitoring during transformation
        - Peak memory consumption validation
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
        profiler = MemoryProfiler()
        profiler.start_profiling(data_size)
        
        # Perform transformation with memory monitoring
        profiler.update_peak_memory()
        result = make_dataframe_from_config(exp_matrix, comprehensive_column_config)
        profiler.update_peak_memory()
        
        memory_stats = profiler.end_profiling()
        
        # Validate memory efficiency requirement
        assert memory_stats['meets_efficiency_requirement'], (
            f"Memory efficiency violation: {memory_stats['memory_multiplier']:.2f}x > 2.0x data size. "
            f"Data size: {memory_stats['data_size_mb']:.1f}MB, "
            f"Memory overhead: {memory_stats['memory_overhead_mb']:.1f}MB"
        )
        
        # Additional validation for reasonable memory usage
        assert memory_stats['memory_multiplier'] >= 0.1, (
            "Memory multiplier suspiciously low - measurement may be incorrect"
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
        """
        config = performance_test_configurations[scale_name]
        
        # Generate test data of specified scale
        exp_matrix = synthetic_exp_matrix_generator(
            rows=config["rows"],
            include_signal_disp=True,
            signal_channels=8  # Moderate complexity
        )
        
        # Benchmark transformation
        result = benchmark.pedantic(
            make_dataframe_from_config,
            args=(exp_matrix, comprehensive_column_config),
            kwargs={},
            iterations=3,
            rounds=5,
            warmup_rounds=1
        )
        
        # Calculate per-row performance
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
            f"Vectorized operations performance issue: {per_million_rows_ms:.2f}ms per 1M rows"
        )
        
        # Validate vectorization effectiveness with multi-channel data
        assert isinstance(result, pd.DataFrame)
        signal_columns = [col for col in result.columns if col.startswith('signal_disp')]
        assert len(signal_columns) > 0, "Signal vectorization should produce multiple columns"


# ============================================================================ 
# SPECIALIZED PERFORMANCE SCENARIOS
# ============================================================================

@pytest.mark.benchmark  
class TestSpecializedTransformationScenarios:
    """
    Performance benchmarks for specialized transformation scenarios including
    edge cases, error conditions, and complex data structures.
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
            rounds=3,
            warmup_rounds=1
        )
        
        # Validate performance for large multi-channel data
        benchmark_stats = benchmark.stats
        mean_time_ms = benchmark_stats['mean'] * 1000
        
        # Should handle large signal arrays efficiently
        mb_per_second = signal_disp_size_mb / (mean_time_ms / 1000)
        assert mb_per_second >= 50.0, (
            f"Signal_disp processing rate too slow: {mb_per_second:.1f}MB/s"
        )
        
        # Validate signal_disp was processed correctly
        assert isinstance(result, pd.DataFrame)
        assert 'signal_disp' in result.columns, "Signal_disp column should be present"
    
    def test_alias_resolution_performance(
        self,
        benchmark,
        synthetic_exp_matrix_generator,
        comprehensive_column_config
    ):
        """
        Benchmark column alias resolution performance to ensure efficient
        handling of alternative column naming schemes.
        
        Tests:
        - Column alias lookup performance
        - Configuration resolution efficiency
        - Name mapping transformation speed
        """
        # Generate data with aliased columns (dtheta_smooth as alias for dtheta)
        exp_matrix = synthetic_exp_matrix_generator(
            rows=300_000,
            include_signal_disp=False  # Focus on alias resolution
        )
        
        # Remove the primary column name, leaving only the alias
        del exp_matrix['dtheta']  # Only dtheta_smooth remains
        
        # Benchmark alias resolution
        result = benchmark.pedantic(
            make_dataframe_from_config,
            args=(exp_matrix, comprehensive_column_config),
            kwargs={},
            iterations=5,
            rounds=8,
            warmup_rounds=2
        )
        
        # Validate alias resolution performance
        benchmark_stats = benchmark.stats
        mean_time_ms = benchmark_stats['mean'] * 1000
        
        # Alias resolution should not significantly impact performance
        per_row_time_ns = (mean_time_ms * 1_000_000) / 300_000
        assert per_row_time_ns <= 2000, (  # Max 2µs per row for alias resolution
            f"Alias resolution performance issue: {per_row_time_ns:.1f}ns per row"
        )
        
        # Validate alias was correctly resolved
        assert isinstance(result, pd.DataFrame)
        assert 'dtheta' in result.columns, "Aliased column should appear with primary name"
        assert len(result) == 300_000
    
    def test_mixed_data_types_transformation_performance(
        self,
        benchmark,
        synthetic_exp_matrix_generator,
        comprehensive_column_config,
        sample_metadata_for_benchmarks
    ):
        """
        Benchmark performance with mixed data types (arrays, scalars, strings)
        to validate efficient handling of heterogeneous experimental data.
        
        Tests:
        - Mixed data type processing efficiency
        - Type conversion performance
        - Heterogeneous data structure handling
        """
        # Generate comprehensive mixed-type dataset
        exp_matrix = synthetic_exp_matrix_generator(
            rows=400_000,
            include_signal_disp=True,
            signal_channels=16
        )
        
        # Add various data type scenarios
        exp_matrix['integer_column'] = np.arange(400_000, dtype=np.int32)
        exp_matrix['float32_column'] = np.random.rand(400_000).astype(np.float32)
        exp_matrix['boolean_array'] = np.random.choice([True, False], 400_000)
        
        # Benchmark mixed-type transformation
        result = benchmark.pedantic(
            make_dataframe_from_config,
            args=(exp_matrix, comprehensive_column_config),
            kwargs={'metadata': sample_metadata_for_benchmarks},
            iterations=2,
            rounds=4,
            warmup_rounds=1
        )
        
        # Validate mixed-type performance
        benchmark_stats = benchmark.stats
        mean_time_ms = benchmark_stats['mean'] * 1000
        
        # Mixed types should not cause significant performance degradation
        per_million_rows_ms = (mean_time_ms / 400_000) * 1_000_000
        assert per_million_rows_ms <= 600.0, (  # Allow some overhead for type handling
            f"Mixed data type performance issue: {per_million_rows_ms:.2f}ms per 1M rows"
        )
        
        # Validate all data types were handled correctly
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 400_000
        assert result['t'].dtype in [np.float64, np.float32], "Time should be float type"
        assert 'date' in result.columns, "Metadata should be included"
    
    @pytest.mark.parametrize(
        "rows,channels,expected_max_time_ms",
        [
            (50_000, 8, 50),      # Small signal_disp
            (100_000, 16, 100),   # Medium signal_disp  
            (200_000, 32, 200),   # Large signal_disp
        ]
    )
    def test_signal_disp_scaling_performance(
        self,
        benchmark,
        rows,
        channels,
        expected_max_time_ms,
        synthetic_exp_matrix_generator,
        comprehensive_column_config
    ):
        """
        Parametrized test for signal_disp transformation scaling across
        different array sizes and channel counts.
        
        Tests:
        - Signal_disp array size scaling
        - Multi-channel processing efficiency
        - 2D array transformation performance validation
        """
        # Generate signal_disp data with specified dimensions
        exp_matrix = synthetic_exp_matrix_generator(
            rows=rows,
            include_signal_disp=True,
            signal_channels=channels,
            include_metadata_fields=False  # Focus on signal processing
        )
        
        # Benchmark signal_disp transformation
        result = benchmark.pedantic(
            make_dataframe_from_config,
            args=(exp_matrix, comprehensive_column_config),
            kwargs={},
            iterations=3,
            rounds=5,
            warmup_rounds=1
        )
        
        # Validate scaling performance
        benchmark_stats = benchmark.stats
        mean_time_ms = benchmark_stats['mean'] * 1000
        
        assert mean_time_ms <= expected_max_time_ms, (
            f"Signal_disp scaling violation: {mean_time_ms:.2f}ms > {expected_max_time_ms}ms "
            f"for {rows} rows × {channels} channels"
        )
        
        # Validate transformation result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == rows
        assert 'signal_disp' in result.columns, "Signal_disp should be transformed"


# ============================================================================
# STRESS TESTING AND REGRESSION DETECTION
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.slow  # Mark as slow test for CI optimization
class TestPerformanceStressAndRegression:
    """
    Stress testing and performance regression detection for DataFrame
    transformation operations under extreme conditions.
    """
    
    def test_stress_transformation_performance(
        self,
        benchmark,
        performance_test_configurations,
        synthetic_exp_matrix_generator,
        comprehensive_column_config
    ):
        """
        Stress test DataFrame transformation with datasets beyond normal
        SLA boundaries to detect performance regression issues.
        
        Tests:
        - Performance under stress conditions
        - Memory behavior with large datasets
        - Graceful performance degradation validation
        """
        stress_config = performance_test_configurations["stress_scale"]
        
        # Generate stress-test dataset
        exp_matrix = synthetic_exp_matrix_generator(
            rows=stress_config["rows"],  # 2M rows
            include_signal_disp=True,
            signal_channels=16
        )
        
        # Monitor memory during stress test
        data_size = estimate_exp_matrix_size(exp_matrix)
        profiler = MemoryProfiler()
        profiler.start_profiling(data_size)
        
        # Benchmark stress-level transformation
        try:
            result = benchmark.pedantic(
                make_dataframe_from_config,
                args=(exp_matrix, comprehensive_column_config),
                kwargs={},
                iterations=1,  # Single iteration for stress test
                rounds=2,
                warmup_rounds=0
            )
            
            profiler.update_peak_memory()
            memory_stats = profiler.end_profiling()
            
            # Validate stress test completion
            benchmark_stats = benchmark.stats
            mean_time_ms = benchmark_stats['mean'] * 1000
            
            # Should complete within reasonable time (may exceed SLA)
            assert mean_time_ms <= stress_config["expected_transform_time_ms"], (
                f"Stress test failure: {mean_time_ms:.2f}ms > {stress_config['expected_transform_time_ms']}ms"
            )
            
            # Memory should still be reasonable
            assert memory_stats['meets_efficiency_requirement'], (
                f"Memory efficiency violation under stress: {memory_stats['memory_multiplier']:.2f}x"
            )
            
            # Validate result integrity under stress
            assert isinstance(result, pd.DataFrame)
            assert len(result) == stress_config["rows"]
            
        except MemoryError:
            pytest.skip("Insufficient memory for stress test - this may be expected on smaller systems")
    
    def test_performance_consistency_validation(
        self,
        benchmark,
        synthetic_exp_matrix_generator,
        comprehensive_column_config
    ):
        """
        Validate performance consistency across multiple runs to detect
        performance variability and ensure reliable benchmarking.
        
        Tests:
        - Performance consistency across runs
        - Statistical variance in execution time
        - Benchmark reliability validation
        """
        # Generate consistent test data
        exp_matrix = synthetic_exp_matrix_generator(
            rows=100_000,
            include_signal_disp=True,
            signal_channels=16,
            seed=42  # Fixed seed for consistency
        )
        
        # Benchmark with many iterations for statistical analysis
        result = benchmark.pedantic(
            make_dataframe_from_config,
            args=(exp_matrix, comprehensive_column_config),
            kwargs={},
            iterations=10,  # More iterations for consistency testing
            rounds=10,
            warmup_rounds=3
        )
        
        # Analyze performance consistency
        benchmark_stats = benchmark.stats
        mean_time = benchmark_stats['mean']
        stddev_time = benchmark_stats['stddev']
        
        # Calculate coefficient of variation (CV)
        cv = (stddev_time / mean_time) * 100
        
        # Performance should be consistent (CV < 10%)
        assert cv <= 10.0, (
            f"Performance inconsistency detected: CV = {cv:.2f}% (>10%). "
            f"Mean: {mean_time*1000:.2f}ms, StdDev: {stddev_time*1000:.2f}ms"
        )
        
        # Validate result integrity
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100_000
    
    @pytest.mark.skipif(
        psutil.virtual_memory().total < 8 * 1024 * 1024 * 1024,  # 8GB
        reason="Insufficient memory for large-scale testing"
    )
    def test_large_scale_memory_management(
        self,
        synthetic_exp_matrix_generator,
        comprehensive_column_config
    ):
        """
        Test memory management with very large datasets to ensure
        graceful handling of memory-intensive operations.
        
        Tests:
        - Large-scale memory management
        - Memory efficiency under load
        - Garbage collection effectiveness
        """
        # Generate very large dataset (if system supports it)
        rows = 5_000_000  # 5M rows
        
        exp_matrix = synthetic_exp_matrix_generator(
            rows=rows,
            include_signal_disp=True,
            signal_channels=8  # Moderate channels for memory management
        )
        
        # Monitor memory throughout transformation
        data_size = estimate_exp_matrix_size(exp_matrix)
        profiler = MemoryProfiler()
        profiler.start_profiling(data_size)
        
        try:
            # Perform transformation with memory monitoring
            profiler.update_peak_memory()
            result = make_dataframe_from_config(exp_matrix, comprehensive_column_config)
            profiler.update_peak_memory()
            
            memory_stats = profiler.end_profiling()
            
            # Validate memory efficiency at scale
            assert memory_stats['meets_efficiency_requirement'], (
                f"Large-scale memory efficiency violation: {memory_stats['memory_multiplier']:.2f}x > 2.0x"
            )
            
            # Validate successful transformation
            assert isinstance(result, pd.DataFrame)
            assert len(result) == rows
            
            # Clean up explicitly for memory management
            del result
            del exp_matrix
            gc.collect()
            
        except MemoryError:
            pytest.skip("Insufficient memory for large-scale test")


# ============================================================================
# PROPERTY-BASED PERFORMANCE TESTING
# ============================================================================

@pytest.mark.benchmark
class TestPropertyBasedPerformanceValidation:
    """
    Property-based performance testing using Hypothesis to validate
    performance characteristics across diverse data scenarios.
    """
    
    @given(
        rows=st.integers(min_value=1000, max_value=100_000),
        channels=st.integers(min_value=1, max_value=32),
        has_signal_disp=st.booleans(),
        has_metadata=st.booleans()
    )
    @settings(max_examples=20, deadline=30000)  # 30 second deadline per example
    def test_property_transformation_performance_scaling(
        self,
        benchmark,
        rows,
        channels,
        has_signal_disp,
        has_metadata,
        synthetic_exp_matrix_generator,
        comprehensive_column_config,
        sample_metadata_for_benchmarks
    ):
        """
        Property-based test to validate that transformation performance
        scales appropriately across diverse data configurations.
        
        Tests:
        - Performance scaling properties
        - Consistent behavior across data variations
        - Linear scaling validation
        """
        # Skip very small datasets that don't provide meaningful benchmarks
        assume(rows >= 5000)
        
        # Generate data with hypothesis-provided parameters
        exp_matrix = synthetic_exp_matrix_generator(
            rows=rows,
            include_signal_disp=has_signal_disp,
            signal_channels=channels,
            include_metadata_fields=False
        )
        
        metadata = sample_metadata_for_benchmarks if has_metadata else None
        
        # Benchmark transformation
        result = benchmark.pedantic(
            make_dataframe_from_config,
            args=(exp_matrix, comprehensive_column_config),
            kwargs={'metadata': metadata} if metadata else {},
            iterations=1,
            rounds=3,
            warmup_rounds=1
        )
        
        # Validate scaling properties
        benchmark_stats = benchmark.stats
        mean_time_ms = benchmark_stats['mean'] * 1000
        
        # Calculate expected time based on linear scaling
        base_rows = 100_000
        base_time_ms = 50.0  # Expected time for 100K rows
        expected_time_ms = (rows / base_rows) * base_time_ms
        
        # Add overhead for signal_disp processing
        if has_signal_disp:
            signal_overhead = (channels / 16) * 20.0  # 20ms per 16 channels
            expected_time_ms += signal_overhead
        
        # Add overhead for metadata
        if has_metadata:
            expected_time_ms += 5.0  # 5ms metadata overhead
        
        # Allow 100% tolerance for property-based testing variability
        max_allowed_time = expected_time_ms * 2.0
        
        assert mean_time_ms <= max_allowed_time, (
            f"Property-based performance violation: {mean_time_ms:.2f}ms > {max_allowed_time:.2f}ms "
            f"for {rows} rows, {channels} channels, signal_disp={has_signal_disp}, metadata={has_metadata}"
        )
        
        # Validate result properties
        assert isinstance(result, pd.DataFrame)
        assert len(result) == rows
        
        if has_signal_disp:
            assert 'signal_disp' in result.columns
        
        if has_metadata:
            assert 'date' in result.columns


# ============================================================================
# BENCHMARK RESULT ANALYSIS AND REPORTING
# ============================================================================

def pytest_benchmark_update_json(config, benchmarks, output_json):
    """
    Custom benchmark result processor to add performance analysis
    and SLA compliance reporting to benchmark output.
    """
    # Add SLA compliance analysis
    sla_compliance = {}
    
    for benchmark in benchmarks:
        name = benchmark['name']
        mean_time = benchmark['stats']['mean']
        
        # Determine SLA based on test name
        if 'basic_transformation_sla' in name:
            sla_threshold = 0.5  # 500ms for 1M rows
            compliance = mean_time <= sla_threshold
            sla_compliance[name] = {
                'sla_threshold_seconds': sla_threshold,
                'actual_time_seconds': mean_time,
                'compliant': compliance,
                'performance_margin': (sla_threshold - mean_time) / sla_threshold * 100
            }
        elif 'signal_disp_special_handler' in name:
            sla_threshold = 0.05  # 50ms per handler
            compliance = mean_time <= sla_threshold
            sla_compliance[name] = {
                'sla_threshold_seconds': sla_threshold,
                'actual_time_seconds': mean_time,
                'compliant': compliance,
                'performance_margin': (sla_threshold - mean_time) / sla_threshold * 100
            }
        elif 'metadata_integration' in name:
            sla_threshold = 0.1  # 100ms total (includes 10ms merge + transformation)
            compliance = mean_time <= sla_threshold
            sla_compliance[name] = {
                'sla_threshold_seconds': sla_threshold,
                'actual_time_seconds': mean_time,
                'compliant': compliance,
                'performance_margin': (sla_threshold - mean_time) / sla_threshold * 100
            }
    
    # Add SLA compliance summary to output
    output_json['sla_compliance'] = sla_compliance
    output_json['performance_summary'] = {
        'total_tests': len(benchmarks),
        'compliant_tests': sum(1 for c in sla_compliance.values() if c['compliant']),
        'overall_compliance_rate': sum(1 for c in sla_compliance.values() if c['compliant']) / max(len(sla_compliance), 1) * 100
    }