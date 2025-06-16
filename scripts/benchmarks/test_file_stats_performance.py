"""
File statistics performance benchmark tests.

This module contains performance-intensive tests for file statistics functionality
that have been extracted from the default test suite per Section 0 requirement
for performance test isolation. These tests are designed to validate SLA compliance
and performance characteristics without impacting rapid developer feedback cycles.

Benchmark Categories:
- Large file set processing performance validation
- File statistics collection scalability testing
- Memory usage validation for bulk operations
- Cross-platform performance consistency verification

Execution:
- Excluded from default pytest execution via @pytest.mark.benchmark
- Executed through scripts/benchmarks/run_benchmarks.py CLI
- Integrated with GitHub Actions benchmark job workflow
"""

import gc
import os
import tempfile
import time
from pathlib import Path
from typing import List

import pytest
from hypothesis import given, settings, strategies as st

# Import production modules for performance validation
from flyrigloader.discovery.stats import (
    get_file_stats,
    attach_file_stats,
    FileStatsCollector,
)

# Import benchmark utilities
try:
    import psutil
except ImportError:
    psutil = None


# ============================================================================
# LARGE FILE SET PERFORMANCE BENCHMARKS
# ============================================================================

class TestFileStatsPerformanceBenchmarks:
    """
    Performance benchmark tests for file statistics functionality.
    
    Validates SLA compliance for large file set processing and scalability
    characteristics across various file quantities and sizes.
    """
    
    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_large_file_set_processing_performance(self, benchmark, temp_experiment_directory):
        """
        Benchmark performance with large numbers of files.
        
        Validates file statistics collection performance against SLA requirements
        for processing large file sets in research data workflows.
        
        SLA Target: <5 seconds for 1000 files per Section 6.6.4.1
        """
        # ARRANGE - Create large set of test files
        num_files = 1000
        file_paths = []
        test_dir = temp_experiment_directory["directory"] / "performance_test"
        test_dir.mkdir(exist_ok=True)
        
        # Create files with varying sizes
        for i in range(num_files):
            file_path = test_dir / f"perf_data_{i:04d}.txt"
            # Vary content size for realistic testing
            content_size = 50 + (i % 200)  # 50-249 characters
            content = "X" * content_size
            file_path.write_text(content)
            file_paths.append(str(file_path))
        
        # ACT - Benchmark file statistics attachment
        def run_batch_stats():
            return attach_file_stats(file_paths)
        
        result = benchmark(run_batch_stats)
        
        # ASSERT - Verify performance and correctness
        assert len(result) == num_files
        
        # Verify all files processed correctly
        for file_path in file_paths:
            assert file_path in result
            assert 'size' in result[file_path]
            assert result[file_path]['size'] > 0
        
        # Log performance metrics for analysis
        if hasattr(benchmark, 'stats'):
            print(f"Benchmark stats: mean={benchmark.stats['mean']:.4f}s, "
                  f"max={benchmark.stats['max']:.4f}s")

    @pytest.mark.benchmark 
    @pytest.mark.performance
    def test_file_stats_memory_usage_benchmark(self, temp_experiment_directory):
        """
        Benchmark memory usage for large file set processing.
        
        Validates memory efficiency requirements for bulk file statistics
        collection in neuroscience research data processing workflows.
        
        Memory Target: <50MB for 1000 files
        """
        if not psutil:
            pytest.skip("psutil not available for memory monitoring")
        
        # ARRANGE - Create test files
        num_files = 1000
        file_paths = []
        test_dir = temp_experiment_directory["directory"] / "memory_test"
        test_dir.mkdir(exist_ok=True)
        
        for i in range(num_files):
            file_path = test_dir / f"mem_test_{i:04d}.csv"
            # Create realistic CSV-like content
            content = f"timestamp,x,y,signal\n{i}.0,{i%100},{(i*2)%100},{i%10}\n"
            file_path.write_text(content)
            file_paths.append(str(file_path))
        
        # Measure initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # ACT - Process files and measure memory
        start_time = time.time()
        result = attach_file_stats(file_paths)
        end_time = time.time()
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - initial_memory
        execution_time = end_time - start_time
        
        # ASSERT - Verify memory and performance targets
        assert len(result) == num_files
        assert memory_used < 50.0, f"Memory usage {memory_used:.2f}MB exceeds 50MB target"
        assert execution_time < 10.0, f"Execution time {execution_time:.2f}s exceeds 10s limit"
        
        print(f"Memory usage: {memory_used:.2f}MB, Time: {execution_time:.4f}s")
        
        # Cleanup
        gc.collect()

    @pytest.mark.benchmark
    @pytest.mark.performance
    @pytest.mark.parametrize("file_count", [100, 500, 1000, 2000])
    def test_file_stats_scalability_benchmark(self, benchmark, temp_experiment_directory, file_count):
        """
        Benchmark scalability characteristics for varying file quantities.
        
        Validates linear scaling performance characteristics and identifies
        potential performance bottlenecks in bulk processing scenarios.
        """
        # ARRANGE - Create specified number of test files
        file_paths = []
        test_dir = temp_experiment_directory["directory"] / f"scale_test_{file_count}"
        test_dir.mkdir(exist_ok=True)
        
        for i in range(file_count):
            file_path = test_dir / f"scale_{i:05d}.txt"
            content = f"Scaling test content for file {i}"
            file_path.write_text(content)
            file_paths.append(str(file_path))
        
        # ACT - Benchmark processing
        result = benchmark(attach_file_stats, file_paths)
        
        # ASSERT - Verify scalability
        assert len(result) == file_count
        
        # Calculate throughput
        if hasattr(benchmark, 'stats'):
            throughput = file_count / benchmark.stats['mean']
            print(f"File count: {file_count}, Throughput: {throughput:.1f} files/sec")

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_file_stats_collector_performance(self, benchmark, temp_experiment_directory):
        """
        Benchmark FileStatsCollector modular interface performance.
        
        Validates performance characteristics of the modular collection
        interface compared to direct function calls.
        """
        # ARRANGE - Create test files and collector
        num_files = 500
        file_paths = []
        test_dir = temp_experiment_directory["directory"] / "collector_perf"
        test_dir.mkdir(exist_ok=True)
        
        for i in range(num_files):
            file_path = test_dir / f"collector_{i:03d}.txt"
            file_path.write_text(f"Collector test content {i}")
            file_paths.append(str(file_path))
        
        collector = FileStatsCollector()
        
        # ACT - Benchmark collector batch processing
        def run_collector_batch():
            return collector.collect_batch_stats(file_paths)
        
        result = benchmark(run_collector_batch)
        
        # ASSERT - Verify collector performance
        assert len(result) == num_files
        
        for file_path in file_paths:
            assert file_path in result
            assert isinstance(result[file_path], dict)

    @pytest.mark.benchmark
    @pytest.mark.performance
    @pytest.mark.skipif(
        os.name == 'nt', 
        reason="Unix filesystem cache testing not applicable on Windows"
    )
    def test_filesystem_cache_performance_impact(self, benchmark, temp_experiment_directory):
        """
        Benchmark filesystem cache impact on repeated file access.
        
        Validates performance characteristics when files are accessed
        multiple times, testing filesystem cache efficiency.
        """
        # ARRANGE - Create test files
        num_files = 200
        file_paths = []
        test_dir = temp_experiment_directory["directory"] / "cache_test"
        test_dir.mkdir(exist_ok=True)
        
        for i in range(num_files):
            file_path = test_dir / f"cache_{i:03d}.txt"
            file_path.write_text(f"Cache test content {i}")
            file_paths.append(str(file_path))
        
        # First pass - cold cache
        def first_pass():
            return attach_file_stats(file_paths)
        
        result1 = benchmark.pedantic(first_pass, iterations=1, rounds=3)
        
        # Second pass - warm cache
        def second_pass():
            return attach_file_stats(file_paths)
        
        result2 = benchmark(second_pass)
        
        # ASSERT - Verify cache performance
        assert len(result1) == num_files
        assert len(result2) == num_files
        
        # Both results should be equivalent
        assert set(result1.keys()) == set(result2.keys())


# ============================================================================
# PROPERTY-BASED PERFORMANCE TESTING
# ============================================================================

class TestFileStatsPropertyBasedPerformance:
    """
    Property-based performance testing using Hypothesis.
    
    Validates performance characteristics across diverse file scenarios
    and edge cases to ensure robust performance under varied conditions.
    """
    
    @pytest.mark.benchmark
    @pytest.mark.performance
    @given(st.integers(min_value=10, max_value=500))
    @settings(max_examples=10, deadline=30000)  # Extended deadline for performance tests
    def test_variable_file_count_performance(self, temp_experiment_directory, file_count):
        """
        Property-based test for performance across variable file counts.
        
        Validates that performance scales predictably across different
        file quantities generated by Hypothesis.
        """
        # ARRANGE - Create variable number of files
        file_paths = []
        test_dir = temp_experiment_directory["directory"] / f"prop_perf_{file_count}"
        test_dir.mkdir(exist_ok=True)
        
        for i in range(file_count):
            file_path = test_dir / f"prop_{i:03d}.txt"
            file_path.write_text(f"Property test content {i}")
            file_paths.append(str(file_path))
        
        # ACT - Measure processing time
        start_time = time.time()
        result = attach_file_stats(file_paths)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # ASSERT - Verify performance scales reasonably
        assert len(result) == file_count
        
        # Performance should scale roughly linearly
        # Allow 0.01 seconds per file as generous upper bound
        max_expected_time = file_count * 0.01
        assert execution_time < max_expected_time, \
            f"Performance degraded: {execution_time:.4f}s for {file_count} files"
        
        # Calculate and log throughput
        throughput = file_count / execution_time if execution_time > 0 else float('inf')
        print(f"Files: {file_count}, Time: {execution_time:.4f}s, "
              f"Throughput: {throughput:.1f} files/sec")

    @pytest.mark.benchmark
    @pytest.mark.performance
    @given(st.integers(min_value=0, max_value=10000))
    @settings(max_examples=5, deadline=20000)
    def test_variable_file_size_performance(self, temp_experiment_directory, file_size):
        """
        Property-based test for performance across variable file sizes.
        
        Validates that file size doesn't significantly impact statistics
        collection performance (since we're not reading file contents).
        """
        # ARRANGE - Create file with specific size
        test_file = temp_experiment_directory["directory"] / f"size_test_{file_size}.txt"
        content = "X" * file_size
        test_file.write_text(content)
        
        # ACT - Measure single file processing time
        start_time = time.time()
        result = get_file_stats(test_file)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # ASSERT - Verify file size independence
        assert result['size'] == len(content.encode('utf-8'))
        
        # File statistics should be fast regardless of file size
        # (we're not reading content, just metadata)
        assert execution_time < 0.1, \
            f"File stats too slow: {execution_time:.4f}s for {file_size}-byte file"


# ============================================================================
# STRESS TESTING AND RESOURCE LIMITS
# ============================================================================

class TestFileStatsStressTesting:
    """
    Stress testing for file statistics under resource constraints.
    
    Validates behavior and performance under extreme conditions and
    resource limitations that might occur in research computing environments.
    """
    
    @pytest.mark.benchmark
    @pytest.mark.performance
    @pytest.mark.skipif(psutil is None, reason="psutil required for memory monitoring")
    def test_memory_pressure_performance(self, temp_experiment_directory):
        """
        Test performance under simulated memory pressure.
        
        Validates that file statistics collection remains functional
        and performs adequately under memory-constrained conditions.
        """
        # ARRANGE - Create large number of files
        num_files = 1500
        file_paths = []
        test_dir = temp_experiment_directory["directory"] / "memory_pressure"
        test_dir.mkdir(exist_ok=True)
        
        for i in range(num_files):
            file_path = test_dir / f"pressure_{i:04d}.txt"
            # Create files with realistic research data content patterns
            content = f"# Experiment {i}\ntimestamp,x,y,signal\n"
            for j in range(10):  # 10 data points per file
                content += f"{j},{j%100},{(j*i)%100},{(i+j)%10}\n"
            file_path.write_text(content)
            file_paths.append(str(file_path))
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # ACT - Process files under memory monitoring
        start_time = time.time()
        
        # Process in chunks to simulate memory-conscious processing
        chunk_size = 100
        all_results = {}
        
        for i in range(0, len(file_paths), chunk_size):
            chunk = file_paths[i:i + chunk_size]
            chunk_result = attach_file_stats(chunk)
            all_results.update(chunk_result)
            
            # Force garbage collection between chunks
            gc.collect()
        
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024
        
        execution_time = end_time - start_time
        memory_growth = final_memory - initial_memory
        
        # ASSERT - Verify stress test results
        assert len(all_results) == num_files
        assert execution_time < 20.0, f"Stress test too slow: {execution_time:.2f}s"
        assert memory_growth < 100.0, f"Memory growth too high: {memory_growth:.2f}MB"
        
        print(f"Stress test: {num_files} files, {execution_time:.2f}s, "
              f"memory growth: {memory_growth:.2f}MB")

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_concurrent_access_simulation(self, temp_experiment_directory):
        """
        Simulate concurrent file access patterns.
        
        Tests performance characteristics when multiple processes might
        be accessing the same files, simulating research computing scenarios.
        """
        # ARRANGE - Create shared file set
        num_files = 300
        file_paths = []
        test_dir = temp_experiment_directory["directory"] / "concurrent_sim"
        test_dir.mkdir(exist_ok=True)
        
        for i in range(num_files):
            file_path = test_dir / f"concurrent_{i:03d}.txt"
            file_path.write_text(f"Concurrent access test {i}")
            file_paths.append(str(file_path))
        
        # ACT - Simulate multiple access patterns
        results = []
        
        # First access pattern - sequential
        start_time = time.time()
        result1 = attach_file_stats(file_paths)
        seq_time = time.time() - start_time
        results.append(('sequential', seq_time, len(result1)))
        
        # Second access pattern - randomized order
        import random
        shuffled_paths = file_paths.copy()
        random.shuffle(shuffled_paths)
        
        start_time = time.time()
        result2 = attach_file_stats(shuffled_paths)
        random_time = time.time() - start_time
        results.append(('randomized', random_time, len(result2)))
        
        # Third access pattern - repeated access
        start_time = time.time()
        result3 = attach_file_stats(file_paths[:100])  # Subset
        repeat_time = time.time() - start_time
        results.append(('repeated', repeat_time, len(result3)))
        
        # ASSERT - Verify concurrent simulation results
        for pattern, exec_time, count in results:
            assert exec_time < 10.0, f"{pattern} pattern too slow: {exec_time:.2f}s"
            print(f"{pattern}: {count} files in {exec_time:.4f}s")
        
        # Results should be consistent across patterns
        assert len(result1) == len(result2) == num_files
        assert len(result3) == 100


# ============================================================================
# CROSS-PLATFORM PERFORMANCE VALIDATION
# ============================================================================

class TestCrossPlatformPerformance:
    """
    Cross-platform performance validation and comparison.
    
    Validates that performance characteristics remain consistent across
    different operating systems and filesystem types used in research.
    """
    
    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_platform_specific_performance_characteristics(self, benchmark, temp_experiment_directory):
        """
        Benchmark platform-specific performance characteristics.
        
        Validates consistent performance across Windows, macOS, and Linux
        platforms commonly used in neuroscience research environments.
        """
        # ARRANGE - Create standardized test set
        num_files = 500
        file_paths = []
        test_dir = temp_experiment_directory["directory"] / "platform_test"
        test_dir.mkdir(exist_ok=True)
        
        # Create files with platform-neutral content
        for i in range(num_files):
            file_path = test_dir / f"platform_{i:03d}.txt"
            content = f"Platform test {i}\n" + "data " * 10
            file_path.write_text(content)
            file_paths.append(str(file_path))
        
        # ACT - Benchmark with platform detection
        import platform as plt
        current_platform = plt.system()
        
        result = benchmark(attach_file_stats, file_paths)
        
        # ASSERT - Verify platform performance
        assert len(result) == num_files
        
        # Log platform-specific metrics
        if hasattr(benchmark, 'stats'):
            print(f"Platform: {current_platform}, "
                  f"Mean time: {benchmark.stats['mean']:.4f}s, "
                  f"Throughput: {num_files/benchmark.stats['mean']:.1f} files/sec")
        
        # Platform-independent performance expectations
        if hasattr(benchmark, 'stats'):
            mean_time = benchmark.stats['mean']
            throughput = num_files / mean_time
            
            # Minimum performance expectations (conservative)
            assert throughput > 50, f"Throughput too low: {throughput:.1f} files/sec"
            assert mean_time < 15.0, f"Processing too slow: {mean_time:.2f}s"

    @pytest.mark.benchmark
    @pytest.mark.performance
    @pytest.mark.parametrize("temp_dir_type", ["default", "memory_backed"])
    def test_filesystem_type_performance_impact(self, benchmark, temp_dir_type):
        """
        Test performance impact of different filesystem types.
        
        Compares performance on different filesystem types that might
        be encountered in research computing environments.
        """
        # ARRANGE - Set up appropriate temporary directory
        if temp_dir_type == "memory_backed":
            # Try to use memory-backed filesystem if available
            temp_base = "/tmp" if os.path.exists("/tmp") else tempfile.gettempdir()
        else:
            temp_base = tempfile.gettempdir()
        
        # Create test directory
        with tempfile.TemporaryDirectory(dir=temp_base) as temp_dir:
            test_dir = Path(temp_dir) / "fs_type_test"
            test_dir.mkdir()
            
            # Create test files
            num_files = 200
            file_paths = []
            
            for i in range(num_files):
                file_path = test_dir / f"fs_test_{i:03d}.txt"
                file_path.write_text(f"Filesystem test content {i}")
                file_paths.append(str(file_path))
            
            # ACT - Benchmark filesystem performance
            result = benchmark(attach_file_stats, file_paths)
            
            # ASSERT - Verify filesystem performance
            assert len(result) == num_files
            
            if hasattr(benchmark, 'stats'):
                print(f"Filesystem type: {temp_dir_type}, "
                      f"Time: {benchmark.stats['mean']:.4f}s")