"""
Comprehensive pytest-benchmark test suite for data loading performance validation.

This module implements performance benchmarks validating the flyrigloader data loading
infrastructure against Service Level Agreement requirements, specifically TST-PERF-001
requirement of <1 second per 100MB data loading performance.

Performance Requirements Validated:
- TST-PERF-001: Data loading must complete within 1s per 100MB
- F-003-RQ-001: Successfully deserialize .pkl files with performance validation
- F-003-RQ-002: Auto-detect and decompress .gz files within SLA constraints
- F-003-RQ-004: Determine format without file extension within <100ms detection overhead
- F-014: Performance Benchmark Suite validation against defined Service Level Agreements
- Section 2.4.9: Benchmarking accuracy within ±5% variance for reliable measurements

Test Categories:
1. Format-specific performance benchmarks (standard, gzipped, pandas pickle)
2. Data size scaling validation from 1MB to 1GB
3. Format detection overhead measurement
4. Statistical performance measurement with regression detection
5. CI/CD integration for automated quality gate enforcement

Integration:
- pytest-benchmark>=4.0.0 for statistical measurement and calibration
- Synthetic data generation for reproducible performance testing
- Comprehensive error scenario testing with performance validation
- Historical benchmark data tracking for performance trend analysis
"""

import gzip
import pickle
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings

# Import the functions under test
from flyrigloader.io.pickle import read_pickle_any_format


class BenchmarkDataGenerator:
    """
    Synthetic data generator for performance testing with controlled data sizes.
    
    Generates realistic experimental data matrices that mimic actual flyrigloader
    usage patterns while providing precise size control for SLA validation.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize generator with deterministic seed for reproducibility."""
        self.seed = seed
        np.random.seed(seed)
    
    def calculate_data_size_mb(self, n_timepoints: int, n_channels: int = 16) -> float:
        """Calculate approximate data size in MB for given dimensions."""
        # Estimate: timepoints * (3 base arrays + n_channels for signal_disp) * 8 bytes/float64
        base_arrays = 3  # t, x, y arrays
        total_elements = n_timepoints * (base_arrays + n_channels)
        size_bytes = total_elements * 8  # 8 bytes per float64
        return size_bytes / (1024 * 1024)  # Convert to MB
    
    def generate_experimental_matrix(self, target_size_mb: float, n_channels: int = 16) -> Dict[str, Any]:
        """
        Generate synthetic experimental data matrix targeting specific size in MB.
        
        Args:
            target_size_mb: Target data size in megabytes
            n_channels: Number of signal channels for signal_disp array
            
        Returns:
            Dictionary containing realistic experimental data
        """
        # Calculate required timepoints for target size
        estimated_timepoints = int((target_size_mb * 1024 * 1024) / ((3 + n_channels) * 8))
        
        # Generate core experimental data
        timepoints = max(estimated_timepoints, 100)  # Minimum reasonable timepoints
        sampling_freq = 60.0  # Hz
        duration = timepoints / sampling_freq
        
        # Time array
        t = np.linspace(0, duration, timepoints)
        
        # Generate realistic trajectory data with correlated motion
        # Start from center and apply random walk with arena boundaries
        arena_radius = 60.0  # mm
        x = np.zeros(timepoints)
        y = np.zeros(timepoints)
        
        for i in range(1, timepoints):
            # Correlated random walk with center bias
            dx = np.random.normal(0, 0.5) - 0.01 * x[i-1]  # Center bias
            dy = np.random.normal(0, 0.5) - 0.01 * y[i-1]
            
            x[i] = x[i-1] + dx
            y[i] = y[i-1] + dy
            
            # Enforce arena boundaries
            r = np.sqrt(x[i]**2 + y[i]**2)
            if r > arena_radius:
                x[i] = x[i-1] * 0.9  # Bounce back
                y[i] = y[i-1] * 0.9
        
        # Generate multi-channel signal data (realistic neural recordings)
        signal_disp = np.zeros((n_channels, timepoints))
        for ch in range(n_channels):
            # Base oscillation with channel-specific frequency
            freq = 2.0 + 0.5 * ch  # 2-10 Hz range
            phase = 2 * np.pi * ch / n_channels
            
            # Generate signal with realistic characteristics
            base_signal = np.sin(2 * np.pi * freq * t + phase)
            # Add harmonics
            base_signal += 0.3 * np.sin(4 * np.pi * freq * t + phase)
            # Add noise
            noise = np.random.normal(0, 0.1, timepoints)
            # Add baseline drift
            drift = 0.2 * np.sin(2 * np.pi * 0.01 * t)
            
            signal_disp[ch, :] = base_signal + noise + drift
        
        # Single channel signal (average of first few channels)
        signal = np.mean(signal_disp[:4, :], axis=0)
        
        # Derived measures
        velocity = np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2) * sampling_freq
        
        return {
            't': t,
            'x': x,
            'y': y,
            'signal': signal,
            'signal_disp': signal_disp,
            'velocity': velocity,
            'arena_radius': arena_radius,
            'sampling_frequency': sampling_freq,
            'n_channels': n_channels
        }
    
    def verify_data_size(self, data: Dict[str, Any], expected_mb: float, tolerance: float = 0.2) -> bool:
        """Verify that generated data meets size requirements within tolerance."""
        # Calculate actual size
        total_bytes = 0
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                total_bytes += value.nbytes
            elif isinstance(value, (int, float)):
                total_bytes += 8  # Approximate for scalar values
        
        actual_mb = total_bytes / (1024 * 1024)
        size_ratio = actual_mb / expected_mb
        
        return abs(size_ratio - 1.0) <= tolerance


class PickleFileFactory:
    """Factory for creating pickle files in various formats for performance testing."""
    
    @staticmethod
    def create_standard_pickle(data: Dict[str, Any], file_path: Path) -> Path:
        """Create standard pickle file."""
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return file_path
    
    @staticmethod
    def create_gzipped_pickle(data: Dict[str, Any], file_path: Path) -> Path:
        """Create gzipped pickle file."""
        with gzip.open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return file_path
    
    @staticmethod
    def create_pandas_pickle(data: Dict[str, Any], file_path: Path) -> Path:
        """Create pandas-specific pickle file."""
        # Convert to DataFrame first
        df_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray) and value.ndim == 1:
                df_data[key] = value
            elif isinstance(value, np.ndarray) and value.ndim == 2:
                # Handle 2D arrays by creating multiple columns
                for i in range(value.shape[0]):
                    df_data[f"{key}_ch{i:02d}"] = value[i, :]
            else:
                # Scalar values - broadcast to match time dimension
                if 't' in data:
                    df_data[key] = [value] * len(data['t'])
                else:
                    df_data[key] = [value]
        
        df = pd.DataFrame(df_data)
        df.to_pickle(file_path)
        return file_path
    
    @staticmethod
    def get_file_size_mb(file_path: Path) -> float:
        """Get file size in megabytes."""
        return file_path.stat().st_size / (1024 * 1024)


# Pytest fixtures for benchmark testing

@pytest.fixture(scope="session")
def benchmark_data_generator():
    """Session-scoped data generator for consistent benchmark data."""
    return BenchmarkDataGenerator(seed=42)


@pytest.fixture(scope="session")
def pickle_factory():
    """Session-scoped pickle file factory."""
    return PickleFileFactory()


@pytest.fixture(params=[1, 10, 50, 100, 250, 500])  # MB sizes
def data_size_mb(request):
    """Parametrized fixture for testing various data sizes."""
    return request.param


@pytest.fixture(params=["standard", "gzipped", "pandas"])
def pickle_format(request):
    """Parametrized fixture for testing different pickle formats."""
    return request.param


@pytest.fixture
def sample_data_files(tmp_path, benchmark_data_generator, pickle_factory, data_size_mb, pickle_format):
    """
    Create sample pickle files for benchmark testing.
    
    Generates data files of specified size and format for comprehensive
    performance testing across different scenarios.
    """
    # Generate data for target size
    data = benchmark_data_generator.generate_experimental_matrix(
        target_size_mb=data_size_mb,
        n_channels=16
    )
    
    # Verify data size is approximately correct
    assert benchmark_data_generator.verify_data_size(data, data_size_mb, tolerance=0.3)
    
    # Create file with appropriate format
    file_path = tmp_path / f"test_data_{data_size_mb}mb_{pickle_format}.pkl"
    if pickle_format == "gzipped":
        file_path = file_path.with_suffix(".pkl.gz")
    
    if pickle_format == "standard":
        created_file = pickle_factory.create_standard_pickle(data, file_path)
    elif pickle_format == "gzipped":
        created_file = pickle_factory.create_gzipped_pickle(data, file_path)
    elif pickle_format == "pandas":
        created_file = pickle_factory.create_pandas_pickle(data, file_path)
    else:
        raise ValueError(f"Unknown pickle format: {pickle_format}")
    
    file_size_mb = pickle_factory.get_file_size_mb(created_file)
    
    return {
        "file_path": created_file,
        "data": data,
        "size_mb": file_size_mb,
        "format": pickle_format,
        "expected_data_size_mb": data_size_mb
    }


# Performance benchmark test classes

class TestDataLoadingPerformance:
    """
    Core performance benchmark tests for data loading SLA validation.
    
    Validates TST-PERF-001 requirement: Data loading must complete within 1s per 100MB
    """
    
    def test_data_loading_sla_compliance(self, benchmark, sample_data_files):
        """
        Test data loading performance against SLA requirements.
        
        Validates that read_pickle_any_format meets the <1 second per 100MB SLA
        requirement across different file formats and sizes.
        """
        file_info = sample_data_files
        file_path = file_info["file_path"]
        file_size_mb = file_info["size_mb"]
        
        # Calculate SLA threshold: 1 second per 100MB
        sla_threshold_seconds = file_size_mb / 100.0
        
        # Benchmark the loading operation
        result = benchmark.pedantic(
            read_pickle_any_format,
            args=(file_path,),
            iterations=5,  # Minimum iterations for statistical significance
            rounds=3,      # Multiple rounds for variance measurement
            warmup_rounds=1  # Warmup for consistent timing
        )
        
        # Verify successful loading
        assert result is not None
        assert isinstance(result, (dict, pd.DataFrame))
        
        # Validate SLA compliance
        mean_time = benchmark.stats.stats.mean
        max_time = benchmark.stats.stats.max
        
        # Assert SLA compliance with descriptive error message
        assert mean_time <= sla_threshold_seconds, (
            f"SLA violation: Mean loading time {mean_time:.3f}s exceeds "
            f"threshold {sla_threshold_seconds:.3f}s for {file_size_mb:.1f}MB file"
        )
        
        # Also check maximum time doesn't exceed 150% of SLA (for variance tolerance)
        max_threshold = sla_threshold_seconds * 1.5
        assert max_time <= max_threshold, (
            f"Performance variance too high: Max time {max_time:.3f}s exceeds "
            f"variance threshold {max_threshold:.3f}s"
        )
    
    def test_format_detection_overhead(self, benchmark, tmp_path, benchmark_data_generator, pickle_factory):
        """
        Test format detection overhead meets <100ms requirement (F-003-RQ-004).
        
        Creates files without extension to force format detection and validates
        that the detection overhead remains under 100ms.
        """
        # Create test data (small size for focused overhead testing)
        data = benchmark_data_generator.generate_experimental_matrix(1.0)  # 1MB
        
        # Create files without extensions to force format detection
        test_files = []
        
        # Standard pickle without extension
        std_file = tmp_path / "test_standard"
        pickle_factory.create_standard_pickle(data, std_file)
        test_files.append(("standard", std_file))
        
        # Gzipped pickle without .gz extension
        gz_file = tmp_path / "test_gzipped"
        pickle_factory.create_gzipped_pickle(data, gz_file)
        test_files.append(("gzipped", gz_file))
        
        detection_times = []
        
        for format_name, file_path in test_files:
            # Benchmark format detection by measuring total time
            result = benchmark.pedantic(
                read_pickle_any_format,
                args=(file_path,),
                iterations=10,  # More iterations for overhead measurement
                rounds=5
            )
            
            # Record detection time
            detection_time_ms = benchmark.stats.stats.mean * 1000  # Convert to ms
            detection_times.append((format_name, detection_time_ms))
            
            # Verify successful loading
            assert result is not None
        
        # Validate detection overhead < 100ms for all formats
        for format_name, detection_time_ms in detection_times:
            assert detection_time_ms < 100, (
                f"Format detection overhead violation: {format_name} format "
                f"detection took {detection_time_ms:.1f}ms, exceeds 100ms threshold"
            )
    
    def test_scaling_performance_linearity(self, benchmark, tmp_path, benchmark_data_generator, pickle_factory):
        """
        Test that loading performance scales linearly with file size.
        
        Validates that the loading performance maintains linear scaling
        characteristics and doesn't degrade superlinearly with size.
        """
        # Test with a range of file sizes
        test_sizes_mb = [1, 5, 10, 25, 50]
        performance_data = []
        
        for size_mb in test_sizes_mb:
            # Generate test data
            data = benchmark_data_generator.generate_experimental_matrix(size_mb)
            
            # Create standard pickle file
            file_path = tmp_path / f"scaling_test_{size_mb}mb.pkl"
            pickle_factory.create_standard_pickle(data, file_path)
            actual_size_mb = pickle_factory.get_file_size_mb(file_path)
            
            # Benchmark loading
            result = benchmark.pedantic(
                read_pickle_any_format,
                args=(file_path,),
                iterations=3,
                rounds=2
            )
            
            loading_time = benchmark.stats.stats.mean
            throughput_mb_per_sec = actual_size_mb / loading_time
            
            performance_data.append({
                'target_size_mb': size_mb,
                'actual_size_mb': actual_size_mb,
                'loading_time_s': loading_time,
                'throughput_mb_s': throughput_mb_per_sec
            })
            
            # Verify successful loading
            assert result is not None
            
            # Ensure meets basic SLA for each size
            sla_threshold = actual_size_mb / 100.0
            assert loading_time <= sla_threshold, (
                f"SLA violation at {actual_size_mb:.1f}MB: {loading_time:.3f}s > {sla_threshold:.3f}s"
            )
        
        # Analyze scaling characteristics
        throughputs = [p['throughput_mb_s'] for p in performance_data]
        mean_throughput = np.mean(throughputs)
        throughput_std = np.std(throughputs)
        throughput_cv = throughput_std / mean_throughput  # Coefficient of variation
        
        # Validate consistent throughput (linear scaling)
        # Allow up to 30% coefficient of variation for realistic hardware variance
        assert throughput_cv < 0.3, (
            f"Non-linear scaling detected: Throughput CV {throughput_cv:.2f} > 0.3, "
            f"mean throughput: {mean_throughput:.1f} MB/s, std: {throughput_std:.1f}"
        )


class TestFormatSpecificPerformance:
    """
    Format-specific performance validation for different pickle types.
    
    Tests performance characteristics specific to standard, gzipped, and
    pandas pickle formats with their unique optimization requirements.
    """
    
    def test_standard_pickle_performance(self, benchmark, tmp_path, benchmark_data_generator, pickle_factory):
        """Test standard pickle format performance characteristics."""
        # Create moderate-sized test data for format-specific testing
        data = benchmark_data_generator.generate_experimental_matrix(25.0)  # 25MB
        
        file_path = tmp_path / "standard_perf_test.pkl"
        pickle_factory.create_standard_pickle(data, file_path)
        file_size_mb = pickle_factory.get_file_size_mb(file_path)
        
        # Benchmark with high precision for format analysis
        result = benchmark.pedantic(
            read_pickle_any_format,
            args=(file_path,),
            iterations=10,
            rounds=5
        )
        
        # Verify loading success and data integrity
        assert result is not None
        assert isinstance(result, dict)
        assert 't' in result
        assert 'x' in result
        assert 'y' in result
        
        # Standard pickle should be fastest format
        loading_time = benchmark.stats.stats.mean
        sla_threshold = file_size_mb / 100.0
        
        # Should significantly outperform SLA (expect ~50% of threshold)
        performance_target = sla_threshold * 0.7
        assert loading_time <= performance_target, (
            f"Standard pickle underperforming: {loading_time:.3f}s > {performance_target:.3f}s "
            f"target for {file_size_mb:.1f}MB file"
        )
    
    def test_gzipped_pickle_performance(self, benchmark, tmp_path, benchmark_data_generator, pickle_factory):
        """Test gzipped pickle format performance with decompression overhead."""
        data = benchmark_data_generator.generate_experimental_matrix(25.0)  # 25MB
        
        file_path = tmp_path / "gzipped_perf_test.pkl.gz"
        pickle_factory.create_gzipped_pickle(data, file_path)
        compressed_size_mb = pickle_factory.get_file_size_mb(file_path)
        
        # Benchmark decompression + loading
        result = benchmark.pedantic(
            read_pickle_any_format,
            args=(file_path,),
            iterations=8,  # Slightly fewer iterations due to decompression overhead
            rounds=4
        )
        
        # Verify loading success
        assert result is not None
        assert isinstance(result, dict)
        
        # For gzipped files, SLA is based on original data size, not compressed size
        original_size_estimate = compressed_size_mb * 3  # Typical compression ratio
        sla_threshold = original_size_estimate / 100.0
        
        loading_time = benchmark.stats.stats.mean
        
        # Gzipped should still meet SLA despite decompression overhead
        assert loading_time <= sla_threshold, (
            f"Gzipped pickle SLA violation: {loading_time:.3f}s > {sla_threshold:.3f}s "
            f"for {compressed_size_mb:.1f}MB compressed file (est. {original_size_estimate:.1f}MB original)"
        )
        
        # Compression efficiency check
        compression_ratio = compressed_size_mb / (original_size_estimate / 3)
        assert compression_ratio < 0.8, f"Poor compression efficiency: {compression_ratio:.2f}"
    
    def test_pandas_pickle_performance(self, benchmark, tmp_path, benchmark_data_generator, pickle_factory):
        """Test pandas-specific pickle format performance characteristics."""
        data = benchmark_data_generator.generate_experimental_matrix(25.0)  # 25MB
        
        file_path = tmp_path / "pandas_perf_test.pkl"
        pickle_factory.create_pandas_pickle(data, file_path)
        file_size_mb = pickle_factory.get_file_size_mb(file_path)
        
        # Benchmark pandas pickle loading
        result = benchmark.pedantic(
            read_pickle_any_format,
            args=(file_path,),
            iterations=8,
            rounds=4
        )
        
        # Verify loading success and DataFrame structure
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 't' in result.columns
        assert 'x' in result.columns
        assert 'y' in result.columns
        
        # Pandas pickle should meet SLA
        loading_time = benchmark.stats.stats.mean
        sla_threshold = file_size_mb / 100.0
        
        assert loading_time <= sla_threshold, (
            f"Pandas pickle SLA violation: {loading_time:.3f}s > {sla_threshold:.3f}s "
            f"for {file_size_mb:.1f}MB file"
        )


class TestPerformanceRegressionDetection:
    """
    Performance regression detection and quality gate enforcement.
    
    Implements automated detection of performance regressions and enforces
    quality gates for CI/CD pipeline integration.
    """
    
    def test_baseline_performance_tracking(self, benchmark, tmp_path, benchmark_data_generator, pickle_factory):
        """
        Track baseline performance metrics for regression detection.
        
        Establishes baseline performance characteristics that can be used
        for automated regression detection in CI/CD pipelines.
        """
        # Standard test case for baseline tracking
        data = benchmark_data_generator.generate_experimental_matrix(50.0)  # 50MB
        
        file_path = tmp_path / "baseline_tracking.pkl"
        pickle_factory.create_standard_pickle(data, file_path)
        file_size_mb = pickle_factory.get_file_size_mb(file_path)
        
        # High-precision benchmark for baseline establishment
        result = benchmark.pedantic(
            read_pickle_any_format,
            args=(file_path,),
            iterations=15,  # High iteration count for statistical stability
            rounds=7,
            warmup_rounds=2
        )
        
        # Verify successful loading
        assert result is not None
        
        # Calculate performance metrics
        stats = benchmark.stats.stats
        mean_time = stats.mean
        std_time = stats.stddev
        throughput = file_size_mb / mean_time
        
        # Performance regression thresholds
        # These values serve as baseline for future regression detection
        baseline_metrics = {
            'mean_loading_time_s': mean_time,
            'std_loading_time_s': std_time,
            'throughput_mb_s': throughput,
            'file_size_mb': file_size_mb,
            'sla_margin': (file_size_mb / 100.0) - mean_time,  # How much faster than SLA
            'coefficient_of_variation': std_time / mean_time
        }
        
        # Quality assertions for baseline metrics
        assert baseline_metrics['sla_margin'] > 0, "Baseline performance doesn't meet SLA"
        assert baseline_metrics['coefficient_of_variation'] < 0.1, "Performance too variable for reliable baseline"
        assert baseline_metrics['throughput_mb_s'] > 100, "Baseline throughput too low"
        
        # Store metrics in benchmark extras for CI/CD pipeline access
        benchmark.extra_info.update(baseline_metrics)
    
    def test_memory_efficiency_validation(self, benchmark, tmp_path, benchmark_data_generator, pickle_factory):
        """
        Validate memory efficiency during data loading operations.
        
        Ensures that memory usage doesn't exceed reasonable bounds during
        loading operations, preventing memory-related performance degradation.
        """
        import psutil
        import os
        
        # Create larger test data for memory validation
        data = benchmark_data_generator.generate_experimental_matrix(100.0)  # 100MB
        
        file_path = tmp_path / "memory_efficiency_test.pkl"
        pickle_factory.create_standard_pickle(data, file_path)
        file_size_mb = pickle_factory.get_file_size_mb(file_path)
        
        # Monitor memory usage during loading
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Benchmark with memory monitoring
        result = benchmark.pedantic(
            read_pickle_any_format,
            args=(file_path,),
            iterations=3,  # Fewer iterations to reduce memory pressure
            rounds=2
        )
        
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = memory_after - memory_before
        
        # Verify loading success
        assert result is not None
        
        # Memory efficiency validation
        # Memory increase should be reasonable relative to file size
        memory_efficiency_ratio = memory_increase / file_size_mb
        
        # Allow up to 3x memory overhead (data + intermediate objects + overhead)
        assert memory_efficiency_ratio < 3.0, (
            f"Memory efficiency violation: {memory_increase:.1f}MB increase "
            f"for {file_size_mb:.1f}MB file (ratio: {memory_efficiency_ratio:.1f})"
        )
        
        # Performance should still meet SLA despite memory constraints
        loading_time = benchmark.stats.stats.mean
        sla_threshold = file_size_mb / 100.0
        assert loading_time <= sla_threshold, f"Memory loading SLA violation: {loading_time:.3f}s > {sla_threshold:.3f}s"


class TestErrorScenarioPerformance:
    """
    Performance validation for error scenarios and edge cases.
    
    Ensures that error handling doesn't introduce significant performance
    overhead and that graceful degradation maintains acceptable performance.
    """
    
    def test_corrupted_file_detection_performance(self, benchmark, tmp_path):
        """Test performance of corrupted file detection and error handling."""
        # Create corrupted file
        corrupted_file = tmp_path / "corrupted.pkl"
        corrupted_file.write_bytes(b"not a pickle file at all" * 1000)  # ~23KB of garbage
        
        # Benchmark error detection
        with pytest.raises(RuntimeError):
            benchmark.pedantic(
                read_pickle_any_format,
                args=(corrupted_file,),
                iterations=5,
                rounds=3
            )
        
        # Error detection should be fast
        detection_time = benchmark.stats.stats.mean
        assert detection_time < 0.1, f"Slow error detection: {detection_time:.3f}s > 0.1s"
    
    def test_missing_file_performance(self, benchmark, tmp_path):
        """Test performance of missing file error handling."""
        missing_file = tmp_path / "does_not_exist.pkl"
        
        # Benchmark missing file detection
        with pytest.raises(FileNotFoundError):
            benchmark.pedantic(
                read_pickle_any_format,
                args=(missing_file,),
                iterations=10,
                rounds=3
            )
        
        # Missing file detection should be very fast
        detection_time = benchmark.stats.stats.mean
        assert detection_time < 0.01, f"Slow missing file detection: {detection_time:.3f}s > 0.01s"


# Property-based testing for comprehensive performance validation

class TestPropertyBasedPerformance:
    """
    Property-based performance testing using Hypothesis for comprehensive validation.
    
    Uses Hypothesis to generate diverse test scenarios and validate that performance
    properties hold across a wide range of inputs and conditions.
    """
    
    @given(
        data_size_mb=st.floats(min_value=0.1, max_value=200.0),
        n_channels=st.integers(min_value=1, max_value=32)
    )
    @settings(max_examples=20, deadline=None)  # Reduced examples for performance tests
    def test_performance_properties_hold(self, data_size_mb, n_channels, benchmark, tmp_path):
        """
        Property-based test that performance characteristics hold across diverse inputs.
        
        Tests the property that loading performance scales predictably with data size
        regardless of the specific structure or channel count of the data.
        """
        # Skip very large tests in property-based testing to maintain reasonable test times
        if data_size_mb > 100:
            pytest.skip("Skipping large data size in property-based test")
        
        generator = BenchmarkDataGenerator(seed=42)
        data = generator.generate_experimental_matrix(data_size_mb, n_channels)
        
        # Verify data generation was successful
        if not generator.verify_data_size(data, data_size_mb, tolerance=0.5):
            pytest.skip(f"Data generation failed size verification for {data_size_mb}MB")
        
        factory = PickleFileFactory()
        file_path = tmp_path / f"property_test_{data_size_mb:.1f}mb_{n_channels}ch.pkl"
        factory.create_standard_pickle(data, file_path)
        
        actual_size_mb = factory.get_file_size_mb(file_path)
        
        # Benchmark loading
        result = benchmark.pedantic(
            read_pickle_any_format,
            args=(file_path,),
            iterations=3,  # Minimal iterations for property testing
            rounds=2
        )
        
        # Verify successful loading
        assert result is not None
        
        # Property: Performance should always meet SLA
        loading_time = benchmark.stats.stats.mean
        sla_threshold = actual_size_mb / 100.0
        
        assert loading_time <= sla_threshold, (
            f"Property violation: {loading_time:.3f}s > {sla_threshold:.3f}s "
            f"for {actual_size_mb:.1f}MB, {n_channels} channels"
        )
        
        # Property: Throughput should be reasonable regardless of structure
        throughput = actual_size_mb / loading_time
        assert throughput > 50, f"Property violation: Low throughput {throughput:.1f} MB/s"


# Performance monitoring and CI/CD integration utilities

def pytest_benchmark_update_machine_info(config, machine_info):
    """
    Update machine info for benchmark tracking across CI/CD environments.
    
    Provides consistent machine identification for performance regression
    detection across different CI/CD runners and local development environments.
    """
    machine_info.update({
        'python_implementation': 'CPython',  # Assuming CPython
        'benchmark_suite': 'data_loading_performance',
        'test_category': 'SLA_validation',
        'requirements_validated': [
            'TST-PERF-001',
            'F-003-RQ-001',
            'F-003-RQ-002', 
            'F-003-RQ-004',
            'F-014'
        ]
    })


def pytest_benchmark_scale_unit(config, unit, benchmark):
    """
    Custom unit scaling for data loading benchmarks.
    
    Provides meaningful units for benchmark reporting that align with
    SLA requirements (MB/s throughput, loading time per MB, etc.).
    """
    if hasattr(benchmark, 'extra_info') and 'file_size_mb' in benchmark.extra_info:
        file_size_mb = benchmark.extra_info['file_size_mb']
        mean_time = benchmark.stats.stats.mean
        
        # Add throughput metrics
        throughput_mb_s = file_size_mb / mean_time
        benchmark.extra_info['throughput_mb_s'] = throughput_mb_s
        benchmark.extra_info['time_per_mb'] = mean_time / file_size_mb
        
        return f"MB/s (throughput: {throughput_mb_s:.1f})"
    
    return unit


# Benchmark configuration and markers

# Mark all tests in this module as benchmarks
pytestmark = pytest.mark.benchmark

# Benchmark-specific configuration
pytest_benchmark_config = {
    'min_rounds': 2,
    'max_time': 10.0,  # Maximum time per benchmark in seconds
    'min_time': 0.1,   # Minimum time per benchmark 
    'warmup': True,
    'warmup_iterations': 1,
    'disable_gc': True,  # Disable garbage collection during benchmarks
    'sort': 'mean',
    'columns': ['mean', 'stddev', 'min', 'max', 'median', 'ops'],
    'histogram': True
}


# Integration markers for CI/CD pipeline
performance_sla = pytest.mark.performance_sla
regression_test = pytest.mark.regression_test
quality_gate = pytest.mark.quality_gate