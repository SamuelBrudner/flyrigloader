"""
Performance benchmark test suite for file discovery operations.

This module validates file discovery engine performance against SLA requirements
including recursive traversal within <5 seconds for 10,000 files, linear scaling
performance (O(n) complexity), and efficient pattern matching operations.

Tests cover:
- File discovery SLA validation per F-002-RQ-001
- Extension-based filtering performance per F-002-RQ-002
- Pattern-based exclusion performance per F-002-RQ-003
- Mandatory substring filtering performance per F-002-RQ-004
- Metadata extraction performance per F-007
- Linear scaling validation per Section 2.4.2
- Parallel directory scanning benchmarks
- Cross-platform storage type performance
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from unittest.mock import patch

import pytest
import numpy as np
from loguru import logger

from flyrigloader.discovery.files import discover_files, FileDiscoverer, get_latest_file
from flyrigloader.discovery.patterns import PatternMatcher, match_files_to_patterns
from flyrigloader.config.discovery import discover_files_with_config


class TestDirectorySetup:
    """Helper class for creating benchmark test directories with realistic file structures."""
    
    @staticmethod
    def create_large_directory_structure(
        base_dir: Path,
        num_files: int,
        subdirectory_depth: int = 3,
        files_per_subdir: int = 100
    ) -> List[str]:
        """
        Create a large directory structure for performance testing.
        
        Args:
            base_dir: Base directory for file creation
            num_files: Total number of files to create
            subdirectory_depth: Maximum depth of subdirectories
            files_per_subdir: Target files per subdirectory
            
        Returns:
            List of created file paths
        """
        created_files = []
        files_created = 0
        
        # Create realistic experimental file patterns
        animals = ["mouse", "rat", "fly"]
        conditions = ["baseline", "treatment", "control", "stim"]
        dates = ["20240101", "20240102", "20240103", "20240104", "20240105"]
        replicates = range(1, 6)
        extensions = [".csv", ".pkl", ".pkl.gz", ".mat"]
        
        # Create nested directory structure
        for depth in range(subdirectory_depth):
            for subdir_idx in range(10):  # 10 subdirs per depth level
                if files_created >= num_files:
                    break
                    
                subdir_path = base_dir / f"depth_{depth}" / f"subdir_{subdir_idx}"
                subdir_path.mkdir(parents=True, exist_ok=True)
                
                # Create files in this subdirectory
                files_in_this_dir = min(files_per_subdir, num_files - files_created)
                
                for file_idx in range(files_in_this_dir):
                    # Create realistic filename patterns
                    animal = np.random.choice(animals)
                    condition = np.random.choice(conditions)
                    date = np.random.choice(dates)
                    replicate = np.random.choice(replicates)
                    extension = np.random.choice(extensions)
                    
                    # Generate different filename patterns for testing
                    if file_idx % 3 == 0:
                        # Pattern: animal_date_condition_replicate.ext
                        filename = f"{animal}_{date}_{condition}_{replicate}{extension}"
                    elif file_idx % 3 == 1:
                        # Pattern: date_animal_condition_replicate.ext
                        filename = f"{date}_{animal}_{condition}_{replicate}{extension}"
                    else:
                        # Pattern: exp001_animal_condition.ext
                        filename = f"exp{file_idx:03d}_{animal}_{condition}{extension}"
                    
                    file_path = subdir_path / filename
                    
                    # Create empty file (sufficient for discovery testing)
                    file_path.touch()
                    created_files.append(str(file_path))
                    files_created += 1
                    
                    if files_created >= num_files:
                        break
                        
                if files_created >= num_files:
                    break
                    
            if files_created >= num_files:
                break
        
        logger.info(f"Created {files_created} test files in {base_dir}")
        return created_files
    
    @staticmethod
    def create_pattern_test_files(base_dir: Path, count: int) -> List[str]:
        """Create files specifically for pattern matching performance tests."""
        created_files = []
        
        patterns = [
            "mouse_{date}_{condition}_{replicate}.csv",
            "{date}_rat_{condition}_{replicate}.csv",
            "exp{exp_id}_{animal}_{condition}.csv",
            "data_{session}_{animal}_{date}.pkl",
            "analysis_{type}_{date}_{version}.pkl.gz"
        ]
        
        for i in range(count):
            # Select pattern and fill in values
            pattern_idx = i % len(patterns)
            pattern = patterns[pattern_idx]
            
            if pattern_idx == 0:  # mouse pattern
                filename = pattern.format(
                    date=f"2024010{(i % 9) + 1}",
                    condition=f"cond{i % 5}",
                    replicate=i % 10
                )
            elif pattern_idx == 1:  # rat pattern
                filename = pattern.format(
                    date=f"2024010{(i % 9) + 1}",
                    condition=f"ctrl{i % 3}",
                    replicate=i % 8
                )
            elif pattern_idx == 2:  # experiment pattern
                filename = pattern.format(
                    exp_id=f"{i % 100:03d}",
                    animal=["mouse", "rat", "fly"][i % 3],
                    condition=["baseline", "treatment"][i % 2]
                )
            elif pattern_idx == 3:  # data pattern
                filename = pattern.format(
                    session=f"s{i % 20:02d}",
                    animal=["mouse", "rat"][i % 2],
                    date=f"20240{(i % 3) + 1}01"
                )
            else:  # analysis pattern
                filename = pattern.format(
                    type=["raw", "processed", "summary"][i % 3],
                    date=f"20240101",
                    version=f"v{i % 5}"
                )
            
            file_path = base_dir / filename
            file_path.touch()
            created_files.append(str(file_path))
        
        return created_files


@pytest.fixture(scope="session")
def large_test_directory():
    """Create a large test directory structure for performance benchmarks."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = Path(temp_dir)
        logger.info(f"Setting up large test directory in {base_path}")
        
        # Create 10,000 files for SLA testing
        files = TestDirectorySetup.create_large_directory_structure(
            base_path, num_files=10000, subdirectory_depth=4, files_per_subdir=50
        )
        
        yield {
            "base_dir": str(base_path),
            "files": files,
            "file_count": len(files)
        }


@pytest.fixture(scope="session")
def pattern_test_directory():
    """Create directory optimized for pattern matching performance tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = Path(temp_dir)
        logger.info(f"Setting up pattern test directory in {base_path}")
        
        # Create 5,000 files with various patterns
        files = TestDirectorySetup.create_pattern_test_files(base_path, 5000)
        
        yield {
            "base_dir": str(base_path),
            "files": files,
            "file_count": len(files)
        }


@pytest.fixture
def sample_extraction_patterns():
    """Provide sample extraction patterns for metadata benchmarking."""
    return [
        r".*/(mouse)_(\d{8})_(\w+)_(\d+)\.csv",
        r".*/(\d{8})_(rat)_(\w+)_(\d+)\.csv",
        r".*/(exp\d+)_(\w+)_(\w+)\.csv",
        r".*/data_(\w+)_(\w+)_(\d{8})\.pkl",
        r".*/analysis_(\w+)_(\d{8})_(\w+)\.pkl\.gz"
    ]


@pytest.fixture
def benchmark_config():
    """Configuration for benchmark test parameters."""
    return {
        "sla_time_limit": 5.0,  # 5 seconds for 10,000 files per F-002-RQ-001
        "acceptable_variance": 0.1,  # 10% variance in performance
        "min_iterations": 3,  # Minimum benchmark iterations
        "warmup_iterations": 1,  # Warmup runs before measurement
    }


class TestFileDiscoverySLAValidation:
    """
    Test file discovery performance against defined SLA requirements.
    
    Validates F-002-RQ-001: Recursive directory traversal within <5s for 10,000 files.
    """
    
    def test_discover_files_sla_10k_files(self, benchmark, large_test_directory, benchmark_config):
        """
        Benchmark file discovery for 10,000 files against SLA requirement.
        
        Tests:
        - F-002-RQ-001: <5 seconds for 10,000 files
        - Section 2.4.2: Linear scaling with file count
        """
        base_dir = large_test_directory["base_dir"]
        expected_count = large_test_directory["file_count"]
        sla_limit = benchmark_config["sla_time_limit"]
        
        def discover_all_files():
            """Function to benchmark."""
            return discover_files(
                directory=base_dir,
                pattern="*",
                recursive=True
            )
        
        # Execute benchmark
        result = benchmark.pedantic(
            discover_all_files,
            iterations=benchmark_config["min_iterations"],
            warmup_rounds=benchmark_config["warmup_iterations"]
        )
        
        # Validate SLA compliance
        execution_time = benchmark.stats.mean
        assert execution_time < sla_limit, (
            f"File discovery took {execution_time:.2f}s, exceeding SLA limit of {sla_limit}s "
            f"for {expected_count} files"
        )
        
        # Validate result correctness
        assert len(result) >= expected_count * 0.95, (
            f"Expected ~{expected_count} files, got {len(result)}"
        )
        
        logger.info(
            f"✓ SLA Validation: Discovered {len(result)} files in {execution_time:.3f}s "
            f"(SLA: <{sla_limit}s)"
        )
    
    def test_discover_files_with_extensions_sla(self, benchmark, large_test_directory, benchmark_config):
        """
        Benchmark extension-based filtering performance.
        
        Tests:
        - F-002-RQ-002: Extension-based filtering with O(n) complexity
        """
        base_dir = large_test_directory["base_dir"]
        extensions = ["csv", "pkl", "pkl.gz"]
        
        def discover_with_extensions():
            """Function to benchmark."""
            return discover_files(
                directory=base_dir,
                pattern="*",
                recursive=True,
                extensions=extensions
            )
        
        # Execute benchmark
        result = benchmark.pedantic(
            discover_with_extensions,
            iterations=benchmark_config["min_iterations"],
            warmup_rounds=benchmark_config["warmup_iterations"]
        )
        
        # Validate performance (should be similar to base discovery)
        execution_time = benchmark.stats.mean
        assert execution_time < benchmark_config["sla_time_limit"] * 1.2, (
            f"Extension filtering took {execution_time:.2f}s, exceeding acceptable limit"
        )
        
        # Validate filtering correctness
        for file_path in result[:10]:  # Sample check
            file_ext = Path(file_path).suffix.lstrip('.')
            # Handle compound extensions like .pkl.gz
            if file_path.endswith('.pkl.gz'):
                assert 'pkl.gz' in extensions
            else:
                assert file_ext in extensions, f"File {file_path} has unexpected extension"
        
        logger.info(
            f"✓ Extension Filtering: {len(result)} files in {execution_time:.3f}s"
        )
    
    def test_discover_files_with_ignore_patterns_sla(self, benchmark, large_test_directory, benchmark_config):
        """
        Benchmark pattern-based exclusion performance.
        
        Tests:
        - F-002-RQ-003: Pattern-based file exclusion with O(n*m) complexity
        """
        base_dir = large_test_directory["base_dir"]
        ignore_patterns = ["*temp*", "*backup*", "*test*", "*_old"]
        
        def discover_with_ignore_patterns():
            """Function to benchmark."""
            return discover_files(
                directory=base_dir,
                pattern="*",
                recursive=True,
                ignore_patterns=ignore_patterns
            )
        
        # Execute benchmark
        result = benchmark.pedantic(
            discover_with_ignore_patterns,
            iterations=benchmark_config["min_iterations"],
            warmup_rounds=benchmark_config["warmup_iterations"]
        )
        
        # Validate O(n*m) performance is acceptable
        execution_time = benchmark.stats.mean
        assert execution_time < benchmark_config["sla_time_limit"] * 1.5, (
            f"Ignore pattern filtering took {execution_time:.2f}s, exceeding acceptable limit"
        )
        
        # Validate ignore patterns are applied
        for file_path in result[:20]:  # Sample check
            filename = Path(file_path).name
            for pattern in ignore_patterns:
                pattern_simple = pattern.replace('*', '')
                assert pattern_simple not in filename, (
                    f"File {filename} should be ignored by pattern {pattern}"
                )
        
        logger.info(
            f"✓ Ignore Patterns: {len(result)} files in {execution_time:.3f}s"
        )
    
    def test_discover_files_with_mandatory_substrings_sla(self, benchmark, large_test_directory, benchmark_config):
        """
        Benchmark mandatory substring filtering performance.
        
        Tests:
        - F-002-RQ-004: Mandatory substring filtering with O(n*k) complexity
        """
        base_dir = large_test_directory["base_dir"]
        mandatory_substrings = ["mouse", "rat", "exp"]  # Common in our test files
        
        def discover_with_mandatory_substrings():
            """Function to benchmark."""
            return discover_files(
                directory=base_dir,
                pattern="*",
                recursive=True,
                mandatory_substrings=mandatory_substrings
            )
        
        # Execute benchmark
        result = benchmark.pedantic(
            discover_with_mandatory_substrings,
            iterations=benchmark_config["min_iterations"],
            warmup_rounds=benchmark_config["warmup_iterations"]
        )
        
        # Validate O(n*k) performance is acceptable
        execution_time = benchmark.stats.mean
        assert execution_time < benchmark_config["sla_time_limit"] * 1.3, (
            f"Mandatory substring filtering took {execution_time:.2f}s"
        )
        
        # Validate mandatory substrings are enforced
        for file_path in result[:20]:  # Sample check
            filename = Path(file_path).name
            has_required = any(substring in filename for substring in mandatory_substrings)
            assert has_required, (
                f"File {filename} should contain at least one of: {mandatory_substrings}"
            )
        
        logger.info(
            f"✓ Mandatory Substrings: {len(result)} files in {execution_time:.3f}s"
        )


class TestLinearScalingValidation:
    """
    Validate linear scaling performance (O(n) complexity) with file count.
    
    Tests Section 2.4.2: Linear scaling with file count requirements.
    """
    
    @pytest.mark.parametrize("file_count", [1000, 2000, 5000, 10000])
    def test_linear_scaling_performance(self, benchmark, file_count):
        """
        Test linear scaling of file discovery performance.
        
        Validates that performance scales linearly with file count per Section 2.4.2.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            
            # Create the specified number of files
            files = TestDirectorySetup.create_large_directory_structure(
                base_path, num_files=file_count, subdirectory_depth=2
            )
            
            def discover_files_linear():
                """Function to benchmark."""
                return discover_files(
                    directory=str(base_path),
                    pattern="*",
                    recursive=True
                )
            
            # Execute benchmark
            result = benchmark.pedantic(
                discover_files_linear,
                iterations=2,
                warmup_rounds=1
            )
            
            execution_time = benchmark.stats.mean
            
            # Calculate performance per file (should be roughly constant for linear scaling)
            time_per_file = execution_time / file_count
            
            # Validate linear scaling assumption (time per file should be < 0.001s)
            assert time_per_file < 0.001, (
                f"Time per file ({time_per_file:.6f}s) suggests non-linear scaling for {file_count} files"
            )
            
            # Validate result correctness
            assert len(result) >= file_count * 0.95, (
                f"Expected ~{file_count} files, got {len(result)}"
            )
            
            logger.info(
                f"✓ Linear Scaling [{file_count:>5} files]: {execution_time:.3f}s "
                f"({time_per_file:.6f}s/file)"
            )
    
    def test_complexity_comparison(self, benchmark):
        """
        Compare performance across different file counts to validate O(n) complexity.
        """
        file_counts = [1000, 3000, 5000]
        performance_data = []
        
        for file_count in file_counts:
            with tempfile.TemporaryDirectory() as temp_dir:
                base_path = Path(temp_dir)
                
                # Create test files
                TestDirectorySetup.create_large_directory_structure(
                    base_path, num_files=file_count, subdirectory_depth=2
                )
                
                # Measure performance
                start_time = time.time()
                result = discover_files(
                    directory=str(base_path),
                    pattern="*",
                    recursive=True
                )
                execution_time = time.time() - start_time
                
                time_per_file = execution_time / file_count
                performance_data.append((file_count, execution_time, time_per_file))
                
                logger.info(f"Complexity test [{file_count} files]: {execution_time:.3f}s")
        
        # Validate that time per file remains roughly constant (linear scaling)
        times_per_file = [data[2] for data in performance_data]
        max_time_per_file = max(times_per_file)
        min_time_per_file = min(times_per_file)
        
        # Allow for 50% variance in time per file (accounting for filesystem overhead)
        variance_ratio = max_time_per_file / min_time_per_file
        assert variance_ratio < 2.0, (
            f"Performance variance ratio ({variance_ratio:.2f}) suggests non-linear scaling. "
            f"Times per file: {times_per_file}"
        )
        
        logger.info(f"✓ Complexity Validation: Variance ratio {variance_ratio:.2f} < 2.0")


class TestMetadataExtractionPerformance:
    """
    Test metadata extraction performance benchmarks.
    
    Validates F-007: Metadata extraction performance requirements.
    """
    
    def test_pattern_matching_performance(self, benchmark, pattern_test_directory, sample_extraction_patterns):
        """
        Benchmark regex pattern matching performance for metadata extraction.
        
        Tests F-007: Metadata extraction system performance.
        """
        files = pattern_test_directory["files"][:1000]  # Use subset for focused testing
        
        def extract_metadata_patterns():
            """Function to benchmark."""
            return match_files_to_patterns(files, sample_extraction_patterns)
        
        # Execute benchmark
        result = benchmark.pedantic(
            extract_metadata_patterns,
            iterations=3,
            warmup_rounds=1
        )
        
        execution_time = benchmark.stats.mean
        
        # Validate performance (should handle 1000 files quickly)
        assert execution_time < 1.0, (
            f"Pattern matching took {execution_time:.2f}s for {len(files)} files"
        )
        
        # Validate extraction results
        assert len(result) > 0, "Pattern matching should extract metadata from some files"
        
        # Sample validation of extracted metadata
        for file_path, metadata in list(result.items())[:5]:
            assert isinstance(metadata, dict), f"Metadata should be dict, got {type(metadata)}"
            assert len(metadata) > 0, f"Metadata should not be empty for {file_path}"
        
        logger.info(
            f"✓ Pattern Matching: {len(result)} matches in {execution_time:.3f}s "
            f"from {len(files)} files"
        )
    
    def test_date_parsing_performance(self, benchmark, pattern_test_directory):
        """
        Benchmark date parsing performance during metadata extraction.
        """
        files = pattern_test_directory["files"][:500]
        
        discoverer = FileDiscoverer(
            extract_patterns=[r".*_(\d{8})_.*"],  # Simple date extraction pattern
            parse_dates=True
        )
        
        def extract_with_date_parsing():
            """Function to benchmark."""
            return discoverer.extract_metadata(files)
        
        # Execute benchmark
        result = benchmark.pedantic(
            extract_with_date_parsing,
            iterations=3,
            warmup_rounds=1
        )
        
        execution_time = benchmark.stats.mean
        
        # Validate date parsing performance
        assert execution_time < 2.0, (
            f"Date parsing took {execution_time:.2f}s for {len(files)} files"
        )
        
        # Validate date parsing results
        files_with_dates = sum(1 for metadata in result.values() if 'parsed_date' in metadata)
        logger.info(
            f"✓ Date Parsing: {files_with_dates} dates parsed in {execution_time:.3f}s"
        )
    
    def test_file_stats_performance(self, benchmark, pattern_test_directory):
        """
        Benchmark file statistics collection performance.
        """
        files = pattern_test_directory["files"][:1000]
        
        discoverer = FileDiscoverer(include_stats=True)
        
        def collect_file_stats():
            """Function to benchmark."""
            return discoverer.extract_metadata(files)
        
        # Execute benchmark
        result = benchmark.pedantic(
            collect_file_stats,
            iterations=3,
            warmup_rounds=1
        )
        
        execution_time = benchmark.stats.mean
        
        # Validate file stats performance
        assert execution_time < 1.5, (
            f"File stats collection took {execution_time:.2f}s for {len(files)} files"
        )
        
        # Validate stats collection results
        files_with_stats = sum(
            1 for metadata in result.values() 
            if any(key in metadata for key in ['size', 'mtime', 'ctime'])
        )
        
        logger.info(
            f"✓ File Stats: {files_with_stats} files with stats in {execution_time:.3f}s"
        )


class TestParallelDirectoryScanningBenchmarks:
    """
    Test parallel directory scanning performance for large hierarchies.
    
    Validates Section 2.4.2: Parallel directory scanning scalability considerations.
    """
    
    def test_deep_directory_structure_performance(self, benchmark):
        """
        Benchmark performance on deep directory structures.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            
            # Create deep directory structure (6 levels deep)
            files = TestDirectorySetup.create_large_directory_structure(
                base_path, num_files=2000, subdirectory_depth=6, files_per_subdir=30
            )
            
            def discover_deep_structure():
                """Function to benchmark."""
                return discover_files(
                    directory=str(base_path),
                    pattern="*",
                    recursive=True
                )
            
            # Execute benchmark
            result = benchmark.pedantic(
                discover_deep_structure,
                iterations=3,
                warmup_rounds=1
            )
            
            execution_time = benchmark.stats.mean
            
            # Validate deep structure performance
            assert execution_time < 3.0, (
                f"Deep directory scanning took {execution_time:.2f}s for 6-level structure"
            )
            
            assert len(result) >= len(files) * 0.95, (
                f"Expected ~{len(files)} files, got {len(result)}"
            )
            
            logger.info(
                f"✓ Deep Structure: {len(result)} files in {execution_time:.3f}s "
                f"(6 levels deep)"
            )
    
    def test_multiple_directory_search_performance(self, benchmark):
        """
        Benchmark performance when searching multiple directories simultaneously.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            
            # Create multiple separate directory trees
            search_dirs = []
            total_files = 0
            
            for i in range(5):  # 5 separate directory trees
                subdir = base_path / f"tree_{i}"
                subdir.mkdir()
                files = TestDirectorySetup.create_large_directory_structure(
                    subdir, num_files=500, subdirectory_depth=2
                )
                search_dirs.append(str(subdir))
                total_files += len(files)
            
            def discover_multiple_directories():
                """Function to benchmark."""
                return discover_files(
                    directory=search_dirs,  # List of directories
                    pattern="*",
                    recursive=True
                )
            
            # Execute benchmark
            result = benchmark.pedantic(
                discover_multiple_directories,
                iterations=3,
                warmup_rounds=1
            )
            
            execution_time = benchmark.stats.mean
            
            # Validate multiple directory performance
            assert execution_time < 2.0, (
                f"Multiple directory search took {execution_time:.2f}s for {len(search_dirs)} dirs"
            )
            
            assert len(result) >= total_files * 0.95, (
                f"Expected ~{total_files} files, got {len(result)}"
            )
            
            logger.info(
                f"✓ Multiple Directories: {len(result)} files in {execution_time:.3f}s "
                f"({len(search_dirs)} directories)"
            )


class TestStorageTypePerformanceBenchmarks:
    """
    Test performance across different storage types and filesystem configurations.
    
    Validates Section 2.4.2: Performance requirements for different storage types.
    """
    
    def test_local_filesystem_baseline_performance(self, benchmark, large_test_directory):
        """
        Establish baseline performance on local filesystem.
        """
        base_dir = large_test_directory["base_dir"]
        
        def discover_local_filesystem():
            """Function to benchmark."""
            return discover_files(
                directory=base_dir,
                pattern="*",
                recursive=True
            )
        
        # Execute benchmark
        result = benchmark.pedantic(
            discover_local_filesystem,
            iterations=5,  # More iterations for baseline
            warmup_rounds=2
        )
        
        execution_time = benchmark.stats.mean
        std_dev = benchmark.stats.stddev
        
        # Store baseline metrics for comparison
        baseline_time_per_file = execution_time / len(result)
        
        logger.info(
            f"✓ Local Filesystem Baseline: {len(result)} files in {execution_time:.3f}s ± {std_dev:.3f}s "
            f"({baseline_time_per_file:.6f}s/file)"
        )
        
        # Validate baseline performance meets SLA
        assert execution_time < 5.0, (
            f"Baseline performance {execution_time:.2f}s exceeds 5s SLA"
        )
    
    @pytest.mark.skipif(
        os.name == 'nt', 
        reason="Filesystem performance test optimized for Unix-like systems"
    )
    def test_filesystem_cache_effects(self, benchmark):
        """
        Test filesystem cache effects on discovery performance.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            
            # Create test files
            files = TestDirectorySetup.create_large_directory_structure(
                base_path, num_files=3000, subdirectory_depth=3
            )
            
            def discover_with_cache_warming():
                """Function to benchmark after cache warming."""
                # First call warms the filesystem cache
                discover_files(directory=str(base_path), pattern="*", recursive=True)
                
                # Second call benefits from cache
                return discover_files(directory=str(base_path), pattern="*", recursive=True)
            
            # Execute benchmark
            result = benchmark.pedantic(
                discover_with_cache_warming,
                iterations=3,
                warmup_rounds=1
            )
            
            execution_time = benchmark.stats.mean
            
            logger.info(
                f"✓ Filesystem Cache: {len(result)} files in {execution_time:.3f}s "
                f"(cache-warmed)"
            )
            
            # Cache-warmed performance should be faster
            assert execution_time < 1.5, (
                f"Cache-warmed performance {execution_time:.2f}s should be < 1.5s"
            )


class TestConfigurationDrivenDiscoveryBenchmarks:
    """
    Test performance of configuration-driven discovery operations.
    
    Validates discovery.py performance with configuration filtering.
    """
    
    def test_config_driven_discovery_performance(self, benchmark, large_test_directory):
        """
        Benchmark configuration-driven discovery with filtering.
        """
        base_dir = large_test_directory["base_dir"]
        
        # Sample configuration for testing
        test_config = {
            "project": {
                "ignore_substrings": ["temp", "backup"],
                "extraction_patterns": [
                    r".*_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.csv"
                ]
            }
        }
        
        def config_driven_discovery():
            """Function to benchmark."""
            return discover_files_with_config(
                config=test_config,
                directory=base_dir,
                pattern="*",
                recursive=True,
                extract_metadata=True
            )
        
        # Execute benchmark
        result = benchmark.pedantic(
            config_driven_discovery,
            iterations=3,
            warmup_rounds=1
        )
        
        execution_time = benchmark.stats.mean
        
        # Validate config-driven performance
        assert execution_time < 6.0, (
            f"Config-driven discovery took {execution_time:.2f}s, should be < 6s"
        )
        
        # Validate metadata extraction occurred
        if isinstance(result, dict):
            files_with_metadata = sum(
                1 for metadata in result.values() 
                if len(metadata) > 1  # More than just 'path'
            )
            logger.info(
                f"✓ Config-Driven Discovery: {len(result)} files, "
                f"{files_with_metadata} with metadata in {execution_time:.3f}s"
            )
        else:
            logger.info(
                f"✓ Config-Driven Discovery: {len(result)} files in {execution_time:.3f}s"
            )


class TestPerformanceRegressionValidation:
    """
    Validate performance regression detection and SLA enforcement.
    """
    
    def test_performance_regression_detection(self, benchmark, pattern_test_directory):
        """
        Test performance regression detection capabilities.
        """
        files = pattern_test_directory["files"][:2000]
        
        def baseline_discovery():
            """Baseline discovery function."""
            return discover_files(
                directory=pattern_test_directory["base_dir"],
                pattern="*.csv",
                recursive=True
            )
        
        # Execute benchmark
        result = benchmark.pedantic(
            baseline_discovery,
            iterations=5,
            warmup_rounds=2
        )
        
        execution_time = benchmark.stats.mean
        std_dev = benchmark.stats.stddev
        
        # Calculate performance bounds for regression detection
        upper_bound = execution_time + (2 * std_dev)  # 95% confidence interval
        
        logger.info(
            f"✓ Regression Baseline: {len(result)} files in {execution_time:.3f}s ± {std_dev:.3f}s "
            f"(upper bound: {upper_bound:.3f}s)"
        )
        
        # Validate statistical consistency
        assert std_dev / execution_time < 0.2, (
            f"Performance variance too high: {std_dev / execution_time:.2%} > 20%"
        )
    
    def test_sla_enforcement_validation(self, benchmark):
        """
        Validate SLA enforcement for various discovery scenarios.
        """
        scenarios = [
            {"file_count": 1000, "sla_limit": 1.0, "description": "Small dataset"},
            {"file_count": 5000, "sla_limit": 3.0, "description": "Medium dataset"},
            {"file_count": 10000, "sla_limit": 5.0, "description": "Large dataset (SLA)"},
        ]
        
        for scenario in scenarios:
            with tempfile.TemporaryDirectory() as temp_dir:
                base_path = Path(temp_dir)
                
                # Create test files
                files = TestDirectorySetup.create_large_directory_structure(
                    base_path, 
                    num_files=scenario["file_count"], 
                    subdirectory_depth=3
                )
                
                # Measure performance
                start_time = time.time()
                result = discover_files(
                    directory=str(base_path),
                    pattern="*",
                    recursive=True
                )
                execution_time = time.time() - start_time
                
                # Validate SLA compliance
                assert execution_time < scenario["sla_limit"], (
                    f"{scenario['description']}: {execution_time:.2f}s exceeds "
                    f"SLA limit of {scenario['sla_limit']}s"
                )
                
                logger.info(
                    f"✓ SLA Validation [{scenario['description']}]: "
                    f"{len(result)} files in {execution_time:.3f}s "
                    f"(SLA: <{scenario['sla_limit']}s)"
                )


# Additional utility functions for comprehensive testing

def test_get_latest_file_performance(benchmark, pattern_test_directory):
    """
    Benchmark get_latest_file utility function performance.
    """
    files = pattern_test_directory["files"][:1000]
    
    def find_latest_file():
        """Function to benchmark."""
        return get_latest_file(files)
    
    # Execute benchmark
    result = benchmark.pedantic(
        find_latest_file,
        iterations=5,
        warmup_rounds=1
    )
    
    execution_time = benchmark.stats.mean
    
    # Validate utility function performance
    assert execution_time < 0.1, (
        f"get_latest_file took {execution_time:.3f}s for {len(files)} files"
    )
    
    assert result is not None, "get_latest_file should return a file path"
    assert result in files, "Latest file should be from the input list"
    
    logger.info(
        f"✓ get_latest_file: Found latest from {len(files)} files in {execution_time:.4f}s"
    )


def test_file_discoverer_initialization_performance(benchmark, sample_extraction_patterns):
    """
    Benchmark FileDiscoverer initialization performance.
    """
    def initialize_discoverer():
        """Function to benchmark."""
        return FileDiscoverer(
            extract_patterns=sample_extraction_patterns,
            parse_dates=True,
            include_stats=True
        )
    
    # Execute benchmark
    result = benchmark.pedantic(
        initialize_discoverer,
        iterations=10,
        warmup_rounds=2
    )
    
    execution_time = benchmark.stats.mean
    
    # Validate initialization performance
    assert execution_time < 0.01, (
        f"FileDiscoverer initialization took {execution_time:.4f}s"
    )
    
    assert isinstance(result, FileDiscoverer), "Should return FileDiscoverer instance"
    assert result.pattern_matcher is not None, "Pattern matcher should be initialized"
    
    logger.info(
        f"✓ FileDiscoverer Initialization: {execution_time:.4f}s"
    )


if __name__ == "__main__":
    # Allow running benchmarks directly for development
    pytest.main([__file__, "-v", "--benchmark-only"])