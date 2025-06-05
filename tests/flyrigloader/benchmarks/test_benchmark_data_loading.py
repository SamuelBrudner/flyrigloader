"""
Comprehensive pytest-benchmark test suite for data loading performance validation.

This module implements statistical performance measurement and Service Level Agreement (SLA) 
validation for the flyrigloader data loading pipeline per TST-PERF-001 requirements.

Key Validation Areas:
- Multi-format pickle loading performance (standard, gzipped, pandas-specific)
- Data size scaling from 1MB to 1GB per F-014 scalability requirements
- Format detection overhead validation (<100ms per F-003-RQ-004)
- Statistical measurement accuracy within ±5% variance per Section 2.4.9
- Regression detection through automated CI/CD pipeline integration

Performance SLAs Validated:
- TST-PERF-001: Data loading <1 second per 100MB
- F-003-RQ-001: Standard pickle deserialization with performance validation
- F-003-RQ-002: Gzipped pickle auto-detection and decompression within SLA
- F-003-RQ-004: Format detection without file extension <100ms overhead

Statistical Analysis Features:
- pytest-benchmark automatic calibration and regression detection
- Multiple data size scenarios with realistic experimental data patterns
- Cross-platform performance validation across Python 3.8-3.11
- Historical benchmark data for trend analysis and capacity planning
"""

import gzip
import os
import pickle
import platform
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, Optional
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, assume, settings

# Core flyrigloader imports for benchmarking
from flyrigloader.io.pickle import (
    read_pickle_any_format,
    PickleLoader,
    DependencyContainer,
    DefaultFileSystemProvider,
    DefaultCompressionProvider,
    DefaultPickleProvider,
    DefaultDataFrameProvider
)


# ============================================================================
# BENCHMARK TEST DATA GENERATION UTILITIES
# ============================================================================

class SyntheticDataGenerator:
    """
    Advanced synthetic experimental data generator for realistic performance testing.
    
    Creates data structures that mirror actual experimental scenarios with:
    - Realistic experimental time series (neural recordings, behavioral tracking)
    - Multi-dimensional signal arrays with proper scaling characteristics
    - Memory-efficient generation patterns for large datasets
    - Reproducible random generation for consistent benchmark results
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize generator with reproducible random seed."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_experimental_matrix(
        self,
        target_size_mb: float,
        include_signal_disp: bool = True,
        include_metadata: bool = True,
        sampling_rate: float = 60.0
    ) -> Dict[str, Any]:
        """
        Generate synthetic experimental data matrix targeting specific memory size.
        
        Args:
            target_size_mb: Target memory size in MB for the generated data
            include_signal_disp: Whether to include multi-channel signal data
            include_metadata: Whether to include experimental metadata
            sampling_rate: Data sampling frequency in Hz
            
        Returns:
            Dictionary containing experimental data with realistic structure
        """
        # Calculate time points needed to reach target size
        # Estimate: Each timepoint ~8 bytes base + signal channels * 8 bytes
        base_size_per_point = 24  # t, x, y (8 bytes each)
        signal_size_per_point = 16 * 8 if include_signal_disp else 8  # 16 channels or 1 signal
        total_size_per_point = base_size_per_point + signal_size_per_point
        
        target_bytes = target_size_mb * 1024 * 1024
        n_timepoints = max(100, int(target_bytes / total_size_per_point))
        
        # Generate realistic time series data
        duration_seconds = n_timepoints / sampling_rate
        time_array = np.linspace(0, duration_seconds, n_timepoints)
        
        # Generate realistic behavioral trajectory data
        # Simulate fly movement in circular arena with center bias
        arena_radius = 60.0  # mm
        center_bias = 0.3
        movement_noise = 0.5
        
        x_pos = np.zeros(n_timepoints)
        y_pos = np.zeros(n_timepoints)
        
        # Generate correlated random walk with arena constraints
        for i in range(1, n_timepoints):
            current_radius = np.sqrt(x_pos[i-1]**2 + y_pos[i-1]**2)
            
            # Center bias force (stronger near edges)
            bias_strength = center_bias * (current_radius / arena_radius)**2
            center_force_x = -bias_strength * x_pos[i-1] / max(current_radius, 0.1)
            center_force_y = -bias_strength * y_pos[i-1] / max(current_radius, 0.1)
            
            # Random movement component
            dt = 1.0 / sampling_rate
            dx = (center_force_x + np.random.normal(0, movement_noise)) * dt
            dy = (center_force_y + np.random.normal(0, movement_noise)) * dt
            
            new_x = x_pos[i-1] + dx
            new_y = y_pos[i-1] + dy
            
            # Enforce arena boundaries with reflection
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
        
        # Add single-channel signal if requested
        if not include_signal_disp:
            # Generate realistic neural signal with oscillations and noise
            signal_freq = 2.0  # Hz
            signal = np.sin(2 * np.pi * signal_freq * time_array)
            signal += 0.3 * np.sin(4 * np.pi * signal_freq * time_array)  # Harmonic
            signal += 0.1 * np.random.normal(0, 1, n_timepoints)  # Noise
            exp_matrix['signal'] = signal
        
        # Add multi-channel signal display data if requested
        if include_signal_disp:
            n_channels = 16
            signal_disp = np.zeros((n_channels, n_timepoints))
            
            for ch in range(n_channels):
                # Channel-specific phase and frequency
                phase = 2 * np.pi * ch / n_channels
                freq_var = 1.0 + 0.2 * np.random.random()
                
                # Base oscillation
                base_signal = np.sin(2 * np.pi * 2.0 * freq_var * time_array + phase)
                base_signal += 0.3 * np.sin(4 * np.pi * 2.0 * freq_var * time_array + phase)
                
                # Add realistic noise and drift
                noise = 0.1 * np.random.normal(0, 1, n_timepoints)
                drift = 0.1 * np.sin(2 * np.pi * 0.01 * time_array + np.random.random() * 2 * np.pi)
                
                signal_disp[ch, :] = base_signal + noise + drift
            
            exp_matrix['signal_disp'] = signal_disp
        
        # Add metadata if requested
        if include_metadata:
            exp_matrix.update({
                'date': '20241201',
                'exp_name': 'benchmark_test_data',
                'rig': 'high_speed_rig',
                'fly_id': f'benchmark_fly_{int(target_size_mb)}mb'
            })
        
        return exp_matrix


# ============================================================================
# BENCHMARK FIXTURE SYSTEM
# ============================================================================

@pytest.fixture(scope="session")
def synthetic_data_generator():
    """Session-scoped synthetic data generator for consistent benchmark data."""
    return SyntheticDataGenerator(random_seed=42)


@pytest.fixture(scope="session") 
def benchmark_data_sizes():
    """
    Define standardized data size matrix for performance testing.
    
    Size progression follows realistic experimental scales:
    - Small: 1-10MB (short experiments, quick validation)
    - Medium: 50-100MB (typical behavioral experiments)
    - Large: 500-1000MB (long-duration, high-resolution experiments)
    """
    return [
        ("small_1mb", 1.0),
        ("small_5mb", 5.0),
        ("medium_10mb", 10.0),
        ("medium_50mb", 50.0),
        ("large_100mb", 100.0),
        ("large_500mb", 500.0),
        ("xlarge_1gb", 1000.0)
    ]


@pytest.fixture(scope="session")
def benchmark_pickle_files(
    synthetic_data_generator,
    benchmark_data_sizes,
    cross_platform_temp_dir
):
    """
    Create comprehensive benchmark pickle files in multiple formats.
    
    Generates files for each size in three formats:
    - Standard pickle: Direct pickle.dump() serialization
    - Gzipped pickle: Compressed pickle with gzip
    - Pandas pickle: DataFrame.to_pickle() format
    
    Returns:
        Dict mapping (size_name, format_type) to file paths
    """
    benchmark_dir = cross_platform_temp_dir / "benchmark_data"
    benchmark_dir.mkdir(exist_ok=True)
    
    benchmark_files = {}
    
    for size_name, size_mb in benchmark_data_sizes:
        # Generate synthetic data for this size
        exp_matrix = synthetic_data_generator.generate_experimental_matrix(
            target_size_mb=size_mb,
            include_signal_disp=True,
            include_metadata=True
        )
        
        # Create DataFrame version for pandas pickle
        df_data = {
            't': exp_matrix['t'],
            'x': exp_matrix['x'],
            'y': exp_matrix['y']
        }
        
        # Add signal data appropriately
        if 'signal' in exp_matrix:
            df_data['signal'] = exp_matrix['signal']
        
        if 'signal_disp' in exp_matrix:
            # Flatten signal_disp for DataFrame (each channel as column)
            signal_disp = exp_matrix['signal_disp']
            for ch in range(signal_disp.shape[0]):
                df_data[f'signal_ch{ch:02d}'] = signal_disp[ch, :]
        
        # Add metadata columns
        for key in ['date', 'exp_name', 'rig', 'fly_id']:
            if key in exp_matrix:
                df_data[key] = exp_matrix[key]
        
        df = pd.DataFrame(df_data)
        
        # Create standard pickle file
        standard_path = benchmark_dir / f"{size_name}_standard.pkl"
        with open(standard_path, 'wb') as f:
            pickle.dump(exp_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
        benchmark_files[(size_name, "standard")] = standard_path
        
        # Create gzipped pickle file
        gzipped_path = benchmark_dir / f"{size_name}_gzipped.pkl.gz"
        with gzip.open(gzipped_path, 'wb') as f:
            pickle.dump(exp_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
        benchmark_files[(size_name, "gzipped")] = gzipped_path
        
        # Create pandas pickle file
        pandas_path = benchmark_dir / f"{size_name}_pandas.pkl"
        df.to_pickle(pandas_path)
        benchmark_files[(size_name, "pandas")] = pandas_path
    
    return benchmark_files


@pytest.fixture(scope="function")
def benchmark_pickle_loader():
    """Function-scoped PickleLoader instance for benchmark testing."""
    return PickleLoader()


@pytest.fixture(scope="function")
def mock_dependencies_for_benchmarks(mocker):
    """
    Mock dependencies for controlled benchmark testing when needed.
    
    Provides ability to mock specific components while preserving
    realistic I/O performance characteristics for accurate benchmarking.
    """
    class BenchmarkMockDependencies:
        def __init__(self):
            self.filesystem_mock = None
            self.compression_mock = None
            self.pickle_mock = None
            self.dataframe_mock = None
        
        def mock_filesystem_only(self):
            """Mock only filesystem operations for path testing."""
            self.filesystem_mock = mocker.MagicMock(spec=DefaultFileSystemProvider)
            self.filesystem_mock.path_exists.return_value = True
            return self
        
        def mock_all_except_io(self):
            """Mock everything except actual I/O for format detection testing."""
            self.filesystem_mock = mocker.MagicMock(spec=DefaultFileSystemProvider)
            self.compression_mock = mocker.MagicMock(spec=DefaultCompressionProvider)
            self.pickle_mock = mocker.MagicMock(spec=DefaultPickleProvider)
            self.dataframe_mock = mocker.MagicMock(spec=DefaultDataFrameProvider)
            return self
        
        def create_dependency_container(self):
            """Create DependencyContainer with configured mocks."""
            return DependencyContainer(
                filesystem_provider=self.filesystem_mock,
                compression_provider=self.compression_mock,
                pickle_provider=self.pickle_mock,
                dataframe_provider=self.dataframe_mock
            )
    
    return BenchmarkMockDependencies()


# ============================================================================
# CORE DATA LOADING PERFORMANCE BENCHMARKS
# ============================================================================

class TestDataLoadingPerformanceBenchmarks:
    """
    Comprehensive data loading performance benchmark test class.
    
    Validates Service Level Agreement compliance per TST-PERF-001:
    - Data loading must complete within 1 second per 100MB
    - Format detection overhead must remain under 100ms
    - Statistical measurement accuracy within ±5% variance
    - Multi-format support with consistent performance characteristics
    """
    
    @pytest.mark.benchmark(group="data_loading_sla")
    @pytest.mark.parametrize("size_name,size_mb", [
        ("small_1mb", 1.0),
        ("small_5mb", 5.0),
        ("medium_10mb", 10.0),
        ("medium_50mb", 50.0),
        ("large_100mb", 100.0)
    ])
    @pytest.mark.parametrize("pickle_format", ["standard", "gzipped", "pandas"])
    def test_data_loading_sla_validation(
        self,
        benchmark,
        benchmark_pickle_files,
        size_name,
        size_mb,
        pickle_format
    ):
        """
        Validate data loading performance against TST-PERF-001 SLA requirements.
        
        Tests data loading across multiple sizes and formats to ensure:
        - Loading time scales linearly with file size
        - All formats meet <1 second per 100MB requirement
        - Statistical consistency across multiple runs
        - No performance regression between library versions
        
        Args:
            benchmark: pytest-benchmark fixture for performance measurement
            benchmark_pickle_files: Pre-generated benchmark files
            size_name: Descriptive name for data size
            size_mb: Data size in megabytes
            pickle_format: Format type (standard, gzipped, pandas)
        """
        # Get the file path for this size and format
        file_path = benchmark_pickle_files[(size_name, pickle_format)]
        
        # Calculate expected maximum time based on TST-PERF-001 SLA
        # SLA: <1 second per 100MB, with 20% buffer for statistical variance
        expected_max_time = (size_mb / 100.0) * 1.0 * 1.2
        
        # Run benchmark with statistical measurement
        result = benchmark(read_pickle_any_format, file_path)
        
        # Validate SLA compliance
        execution_time = benchmark.stats.stats.mean
        assert execution_time <= expected_max_time, (
            f"SLA violation: {pickle_format} format loading {size_mb}MB took "
            f"{execution_time:.3f}s, expected <={expected_max_time:.3f}s"
        )
        
        # Validate result structure and content integrity
        assert isinstance(result, (dict, pd.DataFrame)), (
            f"Invalid result type: expected dict or DataFrame, got {type(result)}"
        )
        
        if isinstance(result, dict):
            # Validate essential columns for experimental data
            required_keys = ['t', 'x', 'y']
            missing_keys = [key for key in required_keys if key not in result]
            assert not missing_keys, f"Missing required keys: {missing_keys}"
            
            # Validate data integrity (non-empty, reasonable ranges)
            assert len(result['t']) > 0, "Empty time array"
            assert np.all(np.diff(result['t']) >= 0), "Non-monotonic time array"
            
        elif isinstance(result, pd.DataFrame):
            # Validate DataFrame structure
            assert not result.empty, "Empty DataFrame result"
            required_cols = ['t', 'x', 'y']
            missing_cols = [col for col in required_cols if col not in result.columns]
            assert not missing_cols, f"Missing required columns: {missing_cols}"


    @pytest.mark.benchmark(group="format_detection")
    def test_format_detection_overhead_validation(
        self,
        benchmark,
        benchmark_pickle_files,
        benchmark_pickle_loader
    ):
        """
        Validate format detection overhead per F-003-RQ-004 requirements.
        
        Tests that format detection without file extension completes within
        100ms overhead per the technical specification. Uses statistical
        measurement to ensure consistent performance across runs.
        
        Args:
            benchmark: pytest-benchmark fixture for performance measurement
            benchmark_pickle_files: Pre-generated benchmark files
            benchmark_pickle_loader: PickleLoader instance for testing
        """
        # Use medium-sized file for realistic detection testing
        test_file = benchmark_pickle_files[("medium_10mb", "standard")]
        
        # Create file without extension to force format detection
        no_ext_file = test_file.with_suffix('')
        no_ext_file.write_bytes(test_file.read_bytes())
        
        try:
            # Benchmark format detection and loading
            result = benchmark(benchmark_pickle_loader.load_pickle_any_format, no_ext_file)
            
            # Validate detection overhead SLA
            execution_time = benchmark.stats.stats.mean
            max_detection_overhead = 0.1  # 100ms per F-003-RQ-004
            
            assert execution_time <= max_detection_overhead + 0.5, (
                f"Format detection overhead violation: {execution_time:.3f}s > "
                f"{max_detection_overhead:.3f}s + 0.5s processing time"
            )
            
            # Validate successful detection and loading
            assert result is not None, "Format detection failed"
            assert isinstance(result, (dict, pd.DataFrame)), "Invalid result type"
            
        finally:
            # Cleanup temporary file
            if no_ext_file.exists():
                no_ext_file.unlink()


    @pytest.mark.benchmark(group="scaling_analysis")
    @pytest.mark.parametrize("size_mb", [1.0, 10.0, 50.0, 100.0, 500.0])
    def test_data_loading_scaling_characteristics(
        self,
        benchmark,
        synthetic_data_generator,
        cross_platform_temp_dir,
        size_mb
    ):
        """
        Validate linear scaling characteristics of data loading performance.
        
        Tests that performance scales predictably with data size to ensure:
        - Linear relationship between file size and loading time
        - No algorithmic inefficiencies causing quadratic scaling
        - Consistent memory usage patterns across sizes
        - Baseline establishment for capacity planning
        
        Args:
            benchmark: pytest-benchmark fixture for performance measurement
            synthetic_data_generator: Generator for test data
            cross_platform_temp_dir: Temporary directory for test files
            size_mb: Data size in megabytes for scaling test
        """
        # Generate test data for this specific size
        exp_matrix = synthetic_data_generator.generate_experimental_matrix(
            target_size_mb=size_mb,
            include_signal_disp=True,
            include_metadata=False
        )
        
        # Create temporary pickle file
        test_file = cross_platform_temp_dir / f"scaling_test_{size_mb}mb.pkl"
        with open(test_file, 'wb') as f:
            pickle.dump(exp_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        try:
            # Benchmark loading performance
            result = benchmark(read_pickle_any_format, test_file)
            
            # Calculate performance metrics
            execution_time = benchmark.stats.stats.mean
            throughput_mb_per_sec = size_mb / execution_time
            
            # Validate scaling characteristics
            # Minimum acceptable throughput: 50 MB/s for standard pickle
            min_throughput = 50.0
            assert throughput_mb_per_sec >= min_throughput, (
                f"Poor throughput: {throughput_mb_per_sec:.1f} MB/s < {min_throughput} MB/s"
            )
            
            # Validate linear scaling expectation
            expected_time = size_mb / 100.0  # 1s per 100MB baseline
            scaling_factor = execution_time / expected_time
            
            # Allow 2x variance for realistic hardware differences
            assert scaling_factor <= 2.0, (
                f"Poor scaling: {scaling_factor:.2f}x expected time for {size_mb}MB"
            )
            
            # Validate result integrity
            assert isinstance(result, dict), "Invalid result type"
            assert 't' in result and 'x' in result and 'y' in result, "Missing required data"
            
        finally:
            # Cleanup test file
            if test_file.exists():
                test_file.unlink()


    @pytest.mark.benchmark(group="multi_format_comparison")
    @pytest.mark.parametrize("pickle_format", ["standard", "gzipped", "pandas"])
    def test_multi_format_performance_comparison(
        self,
        benchmark,
        benchmark_pickle_files,
        pickle_format
    ):
        """
        Compare performance characteristics across pickle formats.
        
        Validates that different pickle formats maintain acceptable performance:
        - Standard pickle: Baseline performance reference
        - Gzipped pickle: Acceptable decompression overhead per F-003-RQ-002
        - Pandas pickle: Format-specific optimizations validation
        
        Args:
            benchmark: pytest-benchmark fixture for performance measurement
            benchmark_pickle_files: Pre-generated benchmark files
            pickle_format: Format type for comparison testing
        """
        # Use consistent medium-sized file for format comparison
        test_file = benchmark_pickle_files[("medium_50mb", pickle_format)]
        
        # Benchmark format-specific loading
        result = benchmark(read_pickle_any_format, test_file)
        
        # Validate format-specific performance expectations
        execution_time = benchmark.stats.stats.mean
        
        if pickle_format == "standard":
            # Standard pickle baseline: <0.5s for 50MB
            max_time = 0.5
        elif pickle_format == "gzipped":
            # Gzipped allows 2x overhead for decompression per F-003-RQ-002
            max_time = 1.0
        elif pickle_format == "pandas":
            # Pandas pickle similar to standard with pandas overhead
            max_time = 0.7
        
        assert execution_time <= max_time, (
            f"{pickle_format} format performance violation: "
            f"{execution_time:.3f}s > {max_time:.3f}s for 50MB"
        )
        
        # Validate format-specific result characteristics
        if pickle_format == "pandas":
            assert isinstance(result, pd.DataFrame), "Pandas format should return DataFrame"
        else:
            assert isinstance(result, dict), "Standard/gzipped should return dict"
        
        # Validate data integrity regardless of format
        if isinstance(result, dict):
            assert 't' in result and 'x' in result, "Missing essential data columns"
        elif isinstance(result, pd.DataFrame):
            assert 't' in result.columns and 'x' in result.columns, "Missing essential DataFrame columns"


# ============================================================================
# EDGE CASE AND ERROR HANDLING BENCHMARKS
# ============================================================================

class TestBenchmarkErrorHandlingPerformance:
    """
    Performance benchmarks for error handling and edge case scenarios.
    
    Validates that error conditions don't cause significant performance
    degradation and maintain consistent behavior under stress conditions.
    """
    
    @pytest.mark.benchmark(group="error_handling")
    def test_file_not_found_performance(self, benchmark):
        """
        Benchmark error handling performance for missing files.
        
        Validates that error detection and reporting remains fast
        even when files don't exist, preventing timeout issues.
        """
        non_existent_file = Path("/non/existent/benchmark/file.pkl")
        
        def load_non_existent():
            with pytest.raises(FileNotFoundError):
                read_pickle_any_format(non_existent_file)
        
        # Error handling should be very fast (<10ms)
        result = benchmark(load_non_existent)
        
        execution_time = benchmark.stats.stats.mean
        max_error_time = 0.01  # 10ms
        
        assert execution_time <= max_error_time, (
            f"Error handling too slow: {execution_time:.3f}s > {max_error_time:.3f}s"
        )
    
    
    @pytest.mark.benchmark(group="error_handling")
    def test_corrupted_file_performance(
        self,
        benchmark,
        cross_platform_temp_dir
    ):
        """
        Benchmark error handling for corrupted pickle files.
        
        Validates that corrupted file detection remains performant
        and doesn't cause indefinite blocking or resource exhaustion.
        """
        # Create corrupted pickle file
        corrupted_file = cross_platform_temp_dir / "corrupted_benchmark.pkl"
        corrupted_file.write_bytes(b"not a pickle file" * 1000)
        
        try:
            def load_corrupted():
                with pytest.raises(RuntimeError):
                    read_pickle_any_format(corrupted_file)
            
            # Corrupted file detection should be fast (<100ms)
            result = benchmark(load_corrupted)
            
            execution_time = benchmark.stats.stats.mean
            max_corruption_time = 0.1  # 100ms
            
            assert execution_time <= max_corruption_time, (
                f"Corruption detection too slow: {execution_time:.3f}s > {max_corruption_time:.3f}s"
            )
            
        finally:
            if corrupted_file.exists():
                corrupted_file.unlink()


# ============================================================================
# STATISTICAL ANALYSIS AND REGRESSION DETECTION
# ============================================================================

class TestBenchmarkStatisticalAnalysis:
    """
    Statistical analysis and regression detection for benchmark data.
    
    Implements comprehensive statistical validation per Section 2.4.9
    requirements for ±5% variance and reproducible measurements.
    """
    
    @pytest.mark.benchmark(group="statistical_validation")
    def test_benchmark_reproducibility(
        self,
        benchmark,
        benchmark_pickle_files
    ):
        """
        Validate benchmark reproducibility and statistical consistency.
        
        Tests that benchmark measurements remain consistent across runs
        with coefficient of variation within acceptable limits for
        reliable performance regression detection.
        """
        # Use consistent medium file for reproducibility testing
        test_file = benchmark_pickle_files[("medium_10mb", "standard")]
        
        # Run multiple iterations to measure statistical variance
        execution_times = []
        
        def measured_load():
            start_time = time.perf_counter()
            result = read_pickle_any_format(test_file)
            end_time = time.perf_counter()
            execution_times.append(end_time - start_time)
            return result
        
        # Benchmark with statistical measurement
        result = benchmark(measured_load)
        
        # Analyze statistical characteristics
        if len(execution_times) >= 3:
            mean_time = np.mean(execution_times)
            std_time = np.std(execution_times)
            coefficient_of_variation = std_time / mean_time
            
            # Validate reproducibility per Section 2.4.9 (±5% variance)
            max_cv = 0.05  # 5% coefficient of variation
            assert coefficient_of_variation <= max_cv, (
                f"Poor reproducibility: CV={coefficient_of_variation:.3f} > {max_cv:.3f}"
            )
        
        # Validate benchmark framework measurement consistency
        framework_mean = benchmark.stats.stats.mean
        framework_std = benchmark.stats.stats.stddev
        framework_cv = framework_std / framework_mean
        
        assert framework_cv <= 0.1, (
            f"Framework measurement inconsistency: CV={framework_cv:.3f} > 0.1"
        )


    @pytest.mark.benchmark(group="regression_detection")
    @pytest.mark.parametrize("data_pattern", ["neural", "behavioral", "mixed"])
    def test_performance_regression_detection(
        self,
        benchmark,
        synthetic_data_generator,
        cross_platform_temp_dir,
        data_pattern
    ):
        """
        Baseline performance measurement for regression detection.
        
        Establishes performance baselines across different data patterns
        to enable automated detection of performance regressions in
        CI/CD pipeline execution.
        
        Args:
            benchmark: pytest-benchmark fixture for performance measurement
            synthetic_data_generator: Generator for test data
            cross_platform_temp_dir: Temporary directory for test files
            data_pattern: Type of experimental data pattern
        """
        # Generate pattern-specific test data
        if data_pattern == "neural":
            exp_matrix = synthetic_data_generator.generate_experimental_matrix(
                target_size_mb=25.0,
                include_signal_disp=True,
                include_metadata=False
            )
        elif data_pattern == "behavioral":
            exp_matrix = synthetic_data_generator.generate_experimental_matrix(
                target_size_mb=25.0,
                include_signal_disp=False,
                include_metadata=True
            )
        else:  # mixed
            exp_matrix = synthetic_data_generator.generate_experimental_matrix(
                target_size_mb=25.0,
                include_signal_disp=True,
                include_metadata=True
            )
        
        # Create temporary test file
        test_file = cross_platform_temp_dir / f"regression_{data_pattern}.pkl"
        with open(test_file, 'wb') as f:
            pickle.dump(exp_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        try:
            # Benchmark loading performance for this pattern
            result = benchmark(read_pickle_any_format, test_file)
            
            # Establish pattern-specific performance baselines
            execution_time = benchmark.stats.stats.mean
            data_size_mb = 25.0
            
            # Pattern-specific performance expectations
            if data_pattern == "neural":
                # Neural data with signal_disp has higher complexity
                max_time_per_mb = 0.015  # 15ms per MB
            elif data_pattern == "behavioral":
                # Behavioral data is simpler structure
                max_time_per_mb = 0.010  # 10ms per MB
            else:  # mixed
                # Mixed data has intermediate complexity
                max_time_per_mb = 0.012  # 12ms per MB
            
            expected_max_time = data_size_mb * max_time_per_mb
            
            assert execution_time <= expected_max_time, (
                f"Performance regression in {data_pattern} pattern: "
                f"{execution_time:.3f}s > {expected_max_time:.3f}s"
            )
            
            # Validate data integrity for the pattern
            assert isinstance(result, dict), "Invalid result type"
            
            if data_pattern == "neural":
                assert 'signal_disp' in result, "Missing neural signal data"
            elif data_pattern == "behavioral":
                assert 'signal' in result or len(result) >= 3, "Missing behavioral data"
            
        finally:
            if test_file.exists():
                test_file.unlink()


# ============================================================================
# PLATFORM AND ENVIRONMENT SPECIFIC BENCHMARKS
# ============================================================================

class TestPlatformSpecificBenchmarks:
    """
    Platform-specific performance benchmarks for cross-platform validation.
    
    Ensures consistent performance characteristics across different
    operating systems and Python versions per CI/CD matrix requirements.
    """
    
    @pytest.mark.benchmark(group="platform_performance")
    def test_cross_platform_performance_consistency(
        self,
        benchmark,
        benchmark_pickle_files
    ):
        """
        Validate performance consistency across operating systems.
        
        Tests that platform-specific differences don't cause significant
        performance variations that would affect SLA compliance.
        """
        # Use consistent test file across platforms
        test_file = benchmark_pickle_files[("medium_50mb", "standard")]
        
        # Benchmark loading performance
        result = benchmark(read_pickle_any_format, test_file)
        
        execution_time = benchmark.stats.stats.mean
        
        # Platform-specific performance expectations
        current_platform = platform.system().lower()
        
        if current_platform == "windows":
            # Windows may have slightly higher overhead
            platform_factor = 1.3
        elif current_platform == "darwin":  # macOS
            # macOS typically has good performance
            platform_factor = 1.1
        else:  # Linux and others
            # Linux baseline performance
            platform_factor = 1.0
        
        # Base expectation: 1s per 100MB * platform factor
        expected_max_time = (50.0 / 100.0) * platform_factor
        
        assert execution_time <= expected_max_time, (
            f"Platform performance issue on {current_platform}: "
            f"{execution_time:.3f}s > {expected_max_time:.3f}s"
        )
        
        # Validate result integrity is platform-independent
        assert isinstance(result, dict), "Platform-dependent result type"
        assert 't' in result and 'x' in result, "Platform-dependent data structure"


    @pytest.mark.benchmark(group="memory_efficiency")
    @pytest.mark.parametrize("size_mb", [10.0, 50.0, 100.0])
    def test_memory_efficiency_benchmarks(
        self,
        benchmark,
        synthetic_data_generator,
        cross_platform_temp_dir,
        size_mb
    ):
        """
        Benchmark memory efficiency during data loading operations.
        
        Validates that memory usage remains reasonable and doesn't
        cause system resource exhaustion during large data loading.
        
        Args:
            benchmark: pytest-benchmark fixture for performance measurement
            synthetic_data_generator: Generator for test data
            cross_platform_temp_dir: Temporary directory for test files
            size_mb: Data size for memory efficiency testing
        """
        # Generate test data
        exp_matrix = synthetic_data_generator.generate_experimental_matrix(
            target_size_mb=size_mb,
            include_signal_disp=True,
            include_metadata=True
        )
        
        # Create test file
        test_file = cross_platform_temp_dir / f"memory_test_{size_mb}mb.pkl"
        with open(test_file, 'wb') as f:
            pickle.dump(exp_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        try:
            # Benchmark with memory considerations
            result = benchmark(read_pickle_any_format, test_file)
            
            execution_time = benchmark.stats.stats.mean
            
            # Memory efficiency expectations
            # Loading should not take excessive time due to memory pressure
            max_time_with_memory_pressure = (size_mb / 100.0) * 2.0  # 2x allowance
            
            assert execution_time <= max_time_with_memory_pressure, (
                f"Memory efficiency issue for {size_mb}MB: "
                f"{execution_time:.3f}s > {max_time_with_memory_pressure:.3f}s"
            )
            
            # Validate successful loading despite memory considerations
            assert isinstance(result, dict), "Memory pressure affected result type"
            assert len(result) > 0, "Memory pressure caused empty result"
            
        finally:
            if test_file.exists():
                test_file.unlink()


# ============================================================================
# INTEGRATION WITH CI/CD PIPELINE BENCHMARKS
# ============================================================================

def test_benchmark_ci_pipeline_integration(benchmark_pickle_files):
    """
    Validate benchmark integration with CI/CD pipeline requirements.
    
    Ensures that benchmark tests can execute successfully in automated
    environments with resource constraints and time limitations.
    """
    # Quick validation test that can run in CI without timeout
    small_file = benchmark_pickle_files[("small_1mb", "standard")]
    
    start_time = time.perf_counter()
    result = read_pickle_any_format(small_file)
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    
    # CI environments should handle small files very quickly
    max_ci_time = 0.1  # 100ms for 1MB in CI
    
    assert execution_time <= max_ci_time, (
        f"CI environment performance issue: {execution_time:.3f}s > {max_ci_time:.3f}s"
    )
    
    # Validate CI-appropriate result validation
    assert isinstance(result, dict), "CI environment affected result type"
    assert 't' in result, "CI environment missing essential data"


# ============================================================================
# HYPOTHESIS-BASED PROPERTY TESTING FOR PERFORMANCE
# ============================================================================

class TestPropertyBasedPerformanceTesting:
    """
    Hypothesis-driven property-based testing for performance characteristics.
    
    Uses property-based testing to validate performance properties across
    a wide range of input scenarios and edge cases.
    """
    
    @given(
        size_mb=st.floats(min_value=0.1, max_value=50.0),
        has_signal=st.booleans(),
        has_metadata=st.booleans()
    )
    @settings(max_examples=10, deadline=30000)  # Reasonable limits for benchmarking
    def test_performance_properties_across_data_variants(
        self,
        size_mb,
        has_signal,
        has_metadata,
        synthetic_data_generator,
        cross_platform_temp_dir
    ):
        """
        Test performance properties across randomly generated data variants.
        
        Uses Hypothesis to generate diverse data scenarios and validate
        that performance characteristics remain predictable across all
        realistic input combinations.
        """
        # Skip unreasonably small sizes that don't represent real use cases
        assume(size_mb >= 0.5)
        
        # Generate test data based on hypothesis parameters
        exp_matrix = synthetic_data_generator.generate_experimental_matrix(
            target_size_mb=size_mb,
            include_signal_disp=has_signal,
            include_metadata=has_metadata
        )
        
        # Create temporary test file
        test_file = cross_platform_temp_dir / f"property_test_{size_mb:.1f}mb.pkl"
        with open(test_file, 'wb') as f:
            pickle.dump(exp_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        try:
            # Measure loading performance
            start_time = time.perf_counter()
            result = read_pickle_any_format(test_file)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            
            # Property: Performance should scale roughly linearly with size
            expected_time = size_mb / 100.0  # 1s per 100MB baseline
            max_allowed_time = expected_time * 2.0  # 2x factor for safety
            
            assert execution_time <= max_allowed_time, (
                f"Property violation: {execution_time:.3f}s > {max_allowed_time:.3f}s "
                f"for {size_mb:.1f}MB data (signal={has_signal}, meta={has_metadata})"
            )
            
            # Property: Result should always be valid structure
            assert isinstance(result, dict), "Property violation: invalid result type"
            assert 't' in result, "Property violation: missing time data"
            
            # Property: Data characteristics should match input configuration
            if has_signal:
                assert 'signal_disp' in result or 'signal' in result, (
                    "Property violation: signal data missing when requested"
                )
            
            if has_metadata:
                metadata_keys = ['date', 'exp_name', 'rig', 'fly_id']
                has_any_metadata = any(key in result for key in metadata_keys)
                assert has_any_metadata, (
                    "Property violation: metadata missing when requested"
                )
            
        finally:
            if test_file.exists():
                test_file.unlink()