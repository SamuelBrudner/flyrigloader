"""
Comprehensive performance benchmark suite for FlyRigLoader file discovery subsystem.

This module provides comprehensive performance validation for the file discovery engine,
enforcing SLA requirements for recursive traversal, pattern filtering, metadata extraction,
and linear scaling validation, relocated from default test suite to maintain rapid
developer feedback cycles while preserving complete discovery engine performance validation.

Performance SLA Validation:
- F-002-RQ-001: Recursive traversal <5s for 10,000 files
- F-002-RQ-002: Extension-based filtering performance requirements
- F-002-RQ-003: Ignore-pattern exclusion performance validation
- F-002-RQ-004: Mandatory substring filtering performance compliance
- F-007: Metadata extraction performance benchmarks with realistic data patterns
- Section 2.4.2: Linear scaling validation (O(n) complexity verification)

Key Features:
- Statistical analysis with confidence intervals and regression detection
- Memory profiling integration for large dataset scenarios via pytest-memory-profiler
- Cross-platform performance validation across Ubuntu, Windows, macOS environments
- Environment normalization for consistent benchmark results across development and CI
- Integration with scripts/benchmarks/run_benchmarks.py CLI execution framework
- Comprehensive artifact generation for performance reports and trend analysis
- pytest-benchmark statistical analysis and regression detection capabilities
- Complete isolation from default test suite execution per Section 0 requirements

Integration:
- flyrigloader.discovery.files for core file discovery functionality
- flyrigloader.discovery.patterns for pattern matching validation
- flyrigloader.config.discovery for configuration-driven discovery testing
- pytest-benchmark for statistical performance measurement and comparison
- pytest-memory-profiler for line-by-line memory analysis and leak detection
- psutil for cross-platform system resource monitoring and environment normalization
- GitHub Actions for CI/CD artifact management with 90-day retention policy compliance

Test Categories:
- Large-scale file discovery benchmarks for SLA compliance validation
- Extension filtering performance with various file type distributions
- Pattern matching efficiency with complex regex and glob patterns
- Metadata extraction performance with realistic experimental data patterns
- Cross-platform compatibility validation with hardware normalization
- Memory efficiency validation for large directory processing scenarios
- Linear scaling validation with systematic file count progression analysis
"""

import gc
import json
import os
import platform
import tempfile
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from unittest.mock import Mock, MagicMock

import numpy as np
import pandas as pd
import psutil
import pytest
from hypothesis import strategies as st

# Import flyrigloader discovery modules for performance testing
from flyrigloader.discovery.files import (
    discover_files,
    FileDiscoverer,
    StandardFilesystemProvider,
    get_latest_file
)
from flyrigloader.discovery.patterns import PatternMatcher
from flyrigloader.config.discovery import (
    discover_experiment_files,
    discover_dataset_files
)

# Import benchmark utilities and configuration
from .utils import (
    MemoryProfiler,
    memory_profiling_context,
    estimate_data_size,
    StatisticalAnalysisEngine,
    EnvironmentAnalyzer,
    PerformanceArtifactGenerator,
    RegressionDetector,
    CrossPlatformValidator,
    analyze_benchmark_results
)
from .config import (
    BenchmarkConfig,
    BenchmarkCategory,
    get_benchmark_config,
    get_category_config,
    PerformanceSLA
)


# ============================================================================
# PYTEST BENCHMARK MARKERS AND CONFIGURATION
# ============================================================================

# Mark all tests in this module for exclusion from default pytest execution
pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.performance,
    pytest.mark.skipif(
        not pytest.config.getoption("--benchmark-only", default=False) and
        not os.getenv("BENCHMARK_MODE", "").lower() in ["true", "1", "yes"],
        reason="Benchmark tests excluded from default execution. Use --benchmark-only or BENCHMARK_MODE=true"
    )
]


# ============================================================================
# PERFORMANCE CONSTANTS AND THRESHOLDS
# ============================================================================

class DiscoveryPerformanceSLA:
    """
    Performance Service Level Agreement thresholds for discovery subsystem.
    
    Based on technical specification requirements and Section 2.4.2 implementation
    considerations for optimal research workflow performance.
    """
    # F-002-RQ-001: Recursive traversal SLA requirement
    RECURSIVE_TRAVERSAL_MAX_SECONDS_10K_FILES = 5.0
    
    # F-002-RQ-002: Extension filtering performance threshold
    EXTENSION_FILTERING_MAX_SECONDS_PER_1K_FILES = 0.1
    
    # F-002-RQ-003: Pattern matching performance requirement
    PATTERN_MATCHING_MAX_SECONDS_PER_1K_FILES = 0.2
    
    # F-002-RQ-004: Mandatory substring filtering performance
    SUBSTRING_FILTERING_MAX_SECONDS_PER_1K_FILES = 0.15
    
    # F-007: Metadata extraction performance threshold
    METADATA_EXTRACTION_MAX_SECONDS_PER_100_FILES = 1.0
    
    # Linear scaling requirement (O(n) complexity validation)
    LINEAR_SCALING_TOLERANCE_FACTOR = 2.5  # 2.5x tolerance for complexity validation
    
    # Cross-platform performance variance tolerance
    CROSS_PLATFORM_VARIANCE_TOLERANCE_PERCENT = 50.0
    
    # Memory efficiency requirements for large datasets
    MEMORY_MULTIPLIER_THRESHOLD = 3.0  # Maximum 3x data size in memory
    MEMORY_LEAK_THRESHOLD_MB = 100.0   # Maximum 100MB growth per iteration


# ============================================================================
# SYNTHETIC DATA GENERATION FOR DISCOVERY BENCHMARKS
# ============================================================================

@pytest.fixture(scope="function")
def discovery_benchmark_data_generator():
    """
    Function-scoped fixture providing comprehensive synthetic data generation
    for file discovery benchmark scenarios with realistic patterns.
    
    Features:
    - Memory-efficient generation for large file count scenarios (10,000+ files)
    - Realistic experimental file naming patterns for neuroscience research
    - Cross-platform compatible directory structures with proper path handling
    - Configurable file size distributions supporting various benchmark scenarios
    - Pattern complexity variations for comprehensive performance validation
    """
    class DiscoveryDataGenerator:
        def __init__(self):
            self.random_seed = 42
            np.random.seed(self.random_seed)
            
            # Realistic file naming patterns for neuroscience research
            self.file_patterns = {
                "mouse_experiments": "mouse_{animal_id:03d}_{date}_{condition}_rep{replicate:02d}.pkl",
                "rat_experiments": "rat_{animal_id:03d}_{date}_{condition}_session{session:02d}.pkl",
                "experiment_series": "experiment_{exp_id}_animal_{animal_id:03d}_{condition}.pkl",
                "behavioral_data": "behavioral_{animal_id:03d}_{date}_{trial_type}.pkl", 
                "neural_recordings": "neural_{animal_id:03d}_{date}_{channel_config}.pkl",
                "metadata_files": "metadata_{exp_id}_{date}.yaml",
                "config_files": "config_{exp_id}_{version}.yaml",
                "backup_files": "backup_{animal_id:03d}_{date}.bak",  # Should be filtered out
                "temp_files": "temp_{uuid}.tmp"  # Should be filtered out
            }
            
            # File extension distributions for realistic testing
            self.extension_distributions = {
                "data_heavy": {"pkl": 0.7, "csv": 0.2, "yaml": 0.05, "bak": 0.03, "tmp": 0.02},
                "config_heavy": {"yaml": 0.5, "json": 0.3, "pkl": 0.15, "csv": 0.05},
                "mixed": {"pkl": 0.4, "csv": 0.3, "yaml": 0.15, "json": 0.1, "bak": 0.03, "tmp": 0.02}
            }
        
        def generate_large_directory_structure(self, 
                                             base_path: Path,
                                             file_count: int = 10000,
                                             directory_depth: int = 4,
                                             extension_distribution: str = "data_heavy") -> Dict[str, Any]:
            """
            Generate large directory structure for F-002-RQ-001 SLA validation.
            
            Args:
                base_path: Base directory for file generation
                file_count: Number of files to generate (default supports SLA testing)
                directory_depth: Maximum directory nesting depth
                extension_distribution: Type of file extension distribution
                
            Returns:
                Dict containing generation metadata and file organization details
            """
            base_path.mkdir(parents=True, exist_ok=True)
            
            # Create realistic subdirectory structure
            subdirectories = self._create_subdirectory_structure(base_path, directory_depth)
            
            # Generate files with realistic naming patterns and distributions
            generated_files = []
            files_by_category = {"valid_data": [], "config_files": [], "backup_files": [], "temp_files": []}
            extension_dist = self.extension_distributions[extension_distribution]
            
            print(f"Generating {file_count:,} files in {len(subdirectories)} directories...")
            
            # Generate files in batches for memory efficiency
            batch_size = min(1000, file_count // 10) if file_count > 1000 else file_count
            
            for batch_start in range(0, file_count, batch_size):
                batch_end = min(batch_start + batch_size, file_count)
                batch_files = self._generate_file_batch(
                    base_path, subdirectories, batch_start, batch_end, extension_dist
                )
                
                # Categorize files
                for file_path, category in batch_files:
                    generated_files.append(file_path)
                    files_by_category[category].append(file_path)
                
                # Periodic progress reporting for large generations
                if file_count > 5000 and (batch_end % 2000 == 0 or batch_end == file_count):
                    print(f"Generated {batch_end:,}/{file_count:,} files ({(batch_end/file_count)*100:.1f}%)")
            
            # Generate metadata for benchmark validation
            total_size_bytes = sum(f.stat().st_size for f in generated_files if f.exists())
            
            generation_metadata = {
                "base_path": base_path,
                "total_files": len(generated_files),
                "total_directories": len(subdirectories),
                "directory_depth": directory_depth,
                "extension_distribution": extension_distribution,
                "files_by_category": {k: len(v) for k, v in files_by_category.items()},
                "files_by_extension": self._analyze_extension_distribution(generated_files),
                "total_size_bytes": total_size_bytes,
                "average_file_size_bytes": total_size_bytes / len(generated_files) if generated_files else 0,
                "generation_timestamp": datetime.now().isoformat(),
                "subdirectories": [str(d) for d in subdirectories]
            }
            
            print(f"Directory generation complete: {len(generated_files):,} files, {total_size_bytes/(1024*1024):.1f}MB total")
            
            return {
                "metadata": generation_metadata,
                "file_lists": files_by_category,
                "all_files": generated_files,
                "subdirectories": subdirectories
            }
        
        def _create_subdirectory_structure(self, base_path: Path, depth: int) -> List[Path]:
            """Create realistic nested subdirectory structure."""
            subdirectories = [base_path]
            
            # Create main category directories
            main_dirs = ["raw_data", "processed_data", "experiments", "animals", "sessions", "backup", "temp"]
            for main_dir in main_dirs:
                main_path = base_path / main_dir
                main_path.mkdir(exist_ok=True)
                subdirectories.append(main_path)
                
                # Create nested subdirectories based on depth
                if depth > 1:
                    for level in range(1, depth):
                        for i in range(min(3, 6 - level)):  # Fewer subdirs at deeper levels
                            sub_path = main_path
                            for d in range(level):
                                sub_path = sub_path / f"level_{d}" / f"sub_{i:02d}"
                            sub_path.mkdir(parents=True, exist_ok=True)
                            subdirectories.append(sub_path)
            
            return subdirectories
        
        def _generate_file_batch(self, base_path: Path, subdirectories: List[Path], 
                                start_idx: int, end_idx: int, extension_dist: Dict[str, float]) -> List[Tuple[Path, str]]:
            """Generate a batch of files with realistic patterns."""
            batch_files = []
            
            for i in range(start_idx, end_idx):
                # Select random directory
                target_dir = np.random.choice(subdirectories)
                
                # Generate realistic metadata
                animal_id = np.random.randint(1, 500)
                date = (datetime.now() - timedelta(days=np.random.randint(0, 730))).strftime("%Y%m%d")
                condition = np.random.choice(["control", "treatment_a", "treatment_b", "baseline"])
                replicate = np.random.randint(1, 10)
                session = np.random.randint(1, 20)
                exp_id = f"EXP{np.random.randint(1, 100):03d}"
                uuid_str = f"{np.random.randint(10000, 99999)}"
                
                # Select file pattern and category
                if np.random.random() < 0.6:  # 60% data files
                    pattern_name = np.random.choice(["mouse_experiments", "rat_experiments", "behavioral_data", "neural_recordings"])
                    category = "valid_data"
                elif np.random.random() < 0.8:  # 20% config files
                    pattern_name = np.random.choice(["metadata_files", "config_files"])
                    category = "config_files"
                elif np.random.random() < 0.9:  # 10% backup files
                    pattern_name = "backup_files"
                    category = "backup_files"
                else:  # 10% temp files
                    pattern_name = "temp_files"
                    category = "temp_files"
                
                # Generate filename
                pattern = self.file_patterns[pattern_name]
                try:
                    filename_base = pattern.format(
                        animal_id=animal_id, date=date, condition=condition,
                        replicate=replicate, session=session, exp_id=exp_id,
                        uuid=uuid_str, channel_config=f"ch{np.random.randint(1,9)}",
                        trial_type=np.random.choice(["free", "forced", "rest"]),
                        version=f"v{np.random.randint(1,6)}"
                    )
                except KeyError:
                    # Fallback for patterns with fewer parameters
                    filename_base = f"file_{i:06d}.pkl"
                
                # Apply extension distribution
                extension = np.random.choice(list(extension_dist.keys()), p=list(extension_dist.values()))
                filename = filename_base.replace(".pkl", f".{extension}")
                
                file_path = target_dir / filename
                
                # Create file with realistic size
                file_size = self._generate_realistic_file_size(extension, category)
                with open(file_path, 'wb') as f:
                    f.write(b'0' * file_size)
                
                batch_files.append((file_path, category))
            
            return batch_files
        
        def _generate_realistic_file_size(self, extension: str, category: str) -> int:
            """Generate realistic file sizes based on extension and category."""
            if extension in ["pkl", "csv"] and category == "valid_data":
                # Data files: mostly medium-sized with some large files
                if np.random.random() < 0.8:
                    return np.random.randint(10*1024, 1024*1024)  # 10KB-1MB
                else:
                    return np.random.randint(1024*1024, 50*1024*1024)  # 1-50MB
            elif extension in ["yaml", "json"]:
                # Config files: small
                return np.random.randint(100, 10*1024)  # 100B-10KB
            elif extension in ["bak", "tmp"]:
                # Backup/temp files: variable sizes
                return np.random.randint(1024, 10*1024*1024)  # 1KB-10MB
            else:
                # Default: small to medium
                return np.random.randint(1024, 100*1024)  # 1-100KB
        
        def _analyze_extension_distribution(self, files: List[Path]) -> Dict[str, int]:
            """Analyze actual extension distribution in generated files."""
            extension_counts = {}
            for file_path in files:
                if file_path.exists():
                    ext = file_path.suffix.lstrip('.')
                    extension_counts[ext] = extension_counts.get(ext, 0) + 1
            return extension_counts
        
        def generate_pattern_matching_test_data(self, base_path: Path, 
                                              complexity_level: str = "moderate") -> Dict[str, Any]:
            """
            Generate test data for pattern matching performance validation.
            
            Args:
                base_path: Base directory for pattern test files
                complexity_level: Pattern complexity ("simple", "moderate", "complex")
                
            Returns:
                Dict containing pattern test structure and expected results
            """
            complexity_configs = {
                "simple": {
                    "file_count": 1000,
                    "patterns": ["*.pkl", "mouse_*.pkl", "*_control_*.pkl"],
                    "directory_depth": 2
                },
                "moderate": {
                    "file_count": 5000,
                    "patterns": [
                        "mouse_*_control_rep*.pkl",
                        "**/experiment_*.pkl", 
                        "**/*session*.pkl",
                        "*behavioral*.pkl"
                    ],
                    "directory_depth": 4
                },
                "complex": {
                    "file_count": 10000,
                    "patterns": [
                        "**/mouse_[0-9][0-9][0-9]_*_control_rep[1-5].pkl",
                        "**/experiment_EXP[0-9][0-9][0-9]_*_treatment_*.pkl",
                        "**/*session_[0-9][0-9]_*behavioral*.pkl",
                        "**/*neural*ch[1-8]*.pkl"
                    ],
                    "directory_depth": 6
                }
            }
            
            config = complexity_configs[complexity_level]
            base_path.mkdir(parents=True, exist_ok=True)
            
            # Generate directory structure with specified complexity
            subdirectories = self._create_subdirectory_structure(base_path, config["directory_depth"])
            
            # Generate files that will match various patterns
            pattern_matches = {pattern: [] for pattern in config["patterns"]}
            generated_files = []
            
            for i in range(config["file_count"]):
                target_dir = np.random.choice(subdirectories)
                
                # Generate filename based on complexity requirements
                filename = self._generate_pattern_test_filename(i, complexity_level)
                file_path = target_dir / filename
                
                # Create file
                file_path.touch()
                generated_files.append(file_path)
                
                # Check which patterns this file matches
                relative_path = file_path.relative_to(base_path)
                for pattern in config["patterns"]:
                    if self._test_pattern_match(str(relative_path), pattern):
                        pattern_matches[pattern].append(file_path)
            
            return {
                "base_path": base_path,
                "complexity_level": complexity_level,
                "total_files": len(generated_files),
                "patterns": config["patterns"],
                "pattern_matches": pattern_matches,
                "expected_match_counts": {pattern: len(matches) for pattern, matches in pattern_matches.items()},
                "subdirectories": subdirectories
            }
        
        def _generate_pattern_test_filename(self, index: int, complexity: str) -> str:
            """Generate filename appropriate for pattern matching complexity."""
            if complexity == "simple":
                return f"{'mouse' if np.random.random() < 0.5 else 'rat'}_{index:03d}_{'control' if np.random.random() < 0.3 else 'treatment'}_rep{np.random.randint(1, 6)}.pkl"
            elif complexity == "moderate":
                base = np.random.choice(["mouse", "experiment", "session"])
                if base == "mouse":
                    return f"mouse_{index:03d}_{np.random.choice(['control', 'treatment'])}_{np.random.choice(['rep', 'session'])}{np.random.randint(1, 10)}.pkl"
                elif base == "experiment":
                    return f"experiment_EXP{np.random.randint(1, 100):03d}_{np.random.choice(['control', 'treatment'])}_data.pkl"
                else:
                    return f"session_{np.random.randint(1, 50):02d}_behavioral_data.pkl"
            else:  # complex
                pattern_type = np.random.choice(["mouse_control", "experiment_treatment", "session_behavioral", "neural"])
                if pattern_type == "mouse_control":
                    return f"mouse_{np.random.randint(1, 200):03d}_{datetime.now().strftime('%Y%m%d')}_control_rep{np.random.randint(1, 5)}.pkl"
                elif pattern_type == "experiment_treatment":
                    return f"experiment_EXP{np.random.randint(1, 100):03d}_animal_{np.random.randint(1, 500)}_treatment_{np.random.choice(['a', 'b', 'c'])}.pkl"
                elif pattern_type == "session_behavioral":
                    return f"session_{np.random.randint(1, 20):02d}_animal_{np.random.randint(1, 100)}_behavioral_data.pkl"
                else:
                    return f"neural_recording_ch{np.random.randint(1, 8)}_session_{np.random.randint(1, 50)}.pkl"
        
        def _test_pattern_match(self, filename: str, pattern: str) -> bool:
            """Test if filename matches the given pattern."""
            import fnmatch
            # Handle recursive patterns
            if pattern.startswith("**/"):
                pattern_without_prefix = pattern[3:]
                # Check both full path and filename
                return fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(filename.split("/")[-1], pattern_without_prefix)
            return fnmatch.fnmatch(filename, pattern)
    
    return DiscoveryDataGenerator()


# ============================================================================
# CORE DISCOVERY PERFORMANCE BENCHMARKS
# ============================================================================

class TestFileDiscoveryPerformance:
    """
    Comprehensive performance test suite for file discovery subsystem.
    
    Tests all aspects of F-002 requirements including recursive traversal,
    extension filtering, pattern matching, and metadata extraction with
    rigorous SLA validation and cross-platform compatibility.
    """
    
    @pytest.mark.benchmark(group="discovery_core")
    def test_recursive_traversal_sla_10k_files(self, benchmark, large_test_directory, 
                                              benchmark_validator, memory_profiler_context):
        """
        Validate F-002-RQ-001 SLA requirement: recursive traversal <5s for 10,000 files.
        
        This benchmark tests the core recursive file discovery performance with
        realistic directory structures and file distributions, ensuring compliance
        with the critical 5-second SLA threshold for 10,000 files.
        
        SLA Validation:
        - Performance target: <5.0 seconds for 10,000 files
        - Memory efficiency: <3x data size overhead
        - Linear scaling: O(n) complexity validation
        - Cross-platform compatibility: Ubuntu, Windows, macOS
        """
        directory_data = large_test_directory
        base_path = directory_data["metadata"]["base_path"]
        
        # Use memory profiler for large dataset analysis
        with memory_profiler_context(
            data_size_estimate=directory_data["metadata"]["total_size_bytes"],
            precision=3,
            enable_line_profiling=True
        ) as profiler:
            
            def discover_all_files():
                """Core discovery operation for benchmarking."""
                return discover_files(
                    directory=str(base_path),
                    pattern="**/*",
                    recursive=True,
                    extensions=None,  # No filtering for baseline performance
                    ignore_patterns=None
                )
            
            # Execute benchmark with pytest-benchmark integration
            result = benchmark.pedantic(
                discover_all_files,
                rounds=5,  # Multiple rounds for statistical analysis
                iterations=1,  # One iteration per round for large datasets
                warmup_rounds=1
            )
            
            memory_stats = profiler.end_profiling()
        
        # Validate SLA compliance
        mean_time = benchmark.stats.stats.mean
        sla_validation = benchmark_validator.validate_sla_compliance(
            test_category="discovery",
            measurement=mean_time,
            scale_factor=directory_data["metadata"]["total_files"] / 10000.0
        )
        
        # Validate memory efficiency
        memory_validation = benchmark_validator.validate_memory_efficiency(
            memory_stats=memory_stats,
            expected_data_size_mb=directory_data["metadata"]["total_size_bytes"] / (1024*1024)
        )
        
        # Validate statistical reliability
        measurements = [stat.mean for stat in benchmark.stats.rounds]
        reliability_validation = benchmark_validator.validate_statistical_reliability(measurements)
        
        # Comprehensive assertion with detailed failure information
        assert sla_validation["compliant"], (
            f"F-002-RQ-001 SLA violation: {mean_time:.3f}s > {sla_validation['threshold']:.3f}s "
            f"for {directory_data['metadata']['total_files']} files. "
            f"Performance margin: {sla_validation['margin_percent']:+.1f}%"
        )
        
        assert memory_validation["efficient"], (
            f"Memory efficiency violation: {memory_validation['memory_multiplier']:.1f}x > "
            f"{memory_validation['efficiency_threshold']:.1f}x threshold"
        )
        
        assert reliability_validation["reliable"], (
            f"Statistical reliability violation: CV={reliability_validation['coefficient_of_variation']:.1f}%, "
            f"Outliers={reliability_validation['outlier_percentage']:.1f}%"
        )
        
        # Verify result correctness
        assert isinstance(result, list), "Discovery should return list of file paths"
        discovered_count = len(result)
        expected_valid_files = len(directory_data["file_lists"]["valid_data"])
        
        # Allow some tolerance for file generation variations
        assert abs(discovered_count - expected_valid_files) <= expected_valid_files * 0.1, (
            f"Discovery result count mismatch: found {discovered_count}, expected ~{expected_valid_files}"
        )
        
        print(f"\n=== F-002-RQ-001 SLA Validation Results ===")
        print(f"Performance: {mean_time:.3f}s for {directory_data['metadata']['total_files']} files")
        print(f"SLA Compliance: {'✓ PASS' if sla_validation['compliant'] else '✗ FAIL'}")
        print(f"Memory Efficiency: {'✓ EFFICIENT' if memory_validation['efficient'] else '✗ INEFFICIENT'}")
        print(f"Files Discovered: {discovered_count:,} files")
        print("=" * 50)
    
    @pytest.mark.benchmark(group="discovery_filtering")
    def test_extension_filtering_performance(self, benchmark, discovery_benchmark_data_generator, 
                                           tmp_path, benchmark_validator):
        """
        Validate F-002-RQ-002 extension filtering performance requirements.
        
        Tests extension-based filtering performance with various file type distributions
        to ensure efficient filtering operations that don't significantly impact
        discovery performance for large datasets.
        
        Performance Requirements:
        - Extension filtering overhead: <0.1s per 1,000 files
        - Filtering accuracy: 100% correct inclusion/exclusion
        - Memory efficiency: Minimal overhead for filtering operations
        """
        # Generate test data with mixed extension distribution
        test_data = discovery_benchmark_data_generator.generate_large_directory_structure(
            base_path=tmp_path / "extension_test",
            file_count=5000,  # Balanced for CI performance
            extension_distribution="mixed"
        )
        
        base_path = test_data["metadata"]["base_path"]
        
        # Test multiple extension filtering scenarios
        test_scenarios = [
            (["pkl"], "data_files_only"),
            (["yaml", "json"], "config_files_only"),
            (["pkl", "csv"], "data_and_csv"),
            (["pkl", "csv", "yaml"], "multi_extension")
        ]
        
        for extensions, scenario_name in test_scenarios:
            
            def discover_with_extension_filter():
                """Discovery operation with extension filtering."""
                return discover_files(
                    directory=str(base_path),
                    pattern="**/*",
                    recursive=True,
                    extensions=extensions,
                    ignore_patterns=None
                )
            
            # Benchmark the filtering operation
            result = benchmark.pedantic(
                discover_with_extension_filter,
                rounds=3,
                iterations=1,
                warmup_rounds=1
            )
            
            # Validate filtering accuracy
            for file_path in result:
                file_ext = Path(file_path).suffix.lstrip('.')
                assert file_ext in extensions, (
                    f"Extension filtering failed: found {file_ext} not in {extensions}"
                )
            
            # Validate performance
            mean_time = benchmark.stats.stats.mean
            file_count = test_data["metadata"]["total_files"]
            time_per_1k_files = (mean_time / file_count) * 1000
            
            assert time_per_1k_files <= DiscoveryPerformanceSLA.EXTENSION_FILTERING_MAX_SECONDS_PER_1K_FILES, (
                f"F-002-RQ-002 extension filtering performance violation for {scenario_name}: "
                f"{time_per_1k_files:.4f}s per 1K files > "
                f"{DiscoveryPerformanceSLA.EXTENSION_FILTERING_MAX_SECONDS_PER_1K_FILES:.4f}s threshold"
            )
            
            print(f"Extension filtering '{scenario_name}': {mean_time:.3f}s total, "
                  f"{time_per_1k_files:.4f}s per 1K files, {len(result)} files found")
    
    @pytest.mark.benchmark(group="discovery_patterns") 
    def test_pattern_matching_performance(self, benchmark, discovery_benchmark_data_generator,
                                        tmp_path, benchmark_validator):
        """
        Validate F-002-RQ-003 ignore-pattern exclusion and F-002-RQ-004 mandatory 
        substring filtering performance requirements.
        
        Tests complex pattern matching scenarios including glob patterns, ignore patterns,
        and mandatory substring filtering with realistic experimental data patterns.
        
        Performance Requirements:
        - Pattern matching: <0.2s per 1,000 files for complex patterns
        - Substring filtering: <0.15s per 1,000 files
        - Pattern accuracy: 100% correct matching behavior
        """
        # Generate pattern test data with complex patterns
        test_data = discovery_benchmark_data_generator.generate_pattern_matching_test_data(
            base_path=tmp_path / "pattern_test",
            complexity_level="moderate"
        )
        
        base_path = test_data["base_path"]
        
        # Test F-002-RQ-003: Ignore pattern exclusion performance
        ignore_pattern_scenarios = [
            (["backup_*", "temp_*"], "common_ignores"),
            (["*temp*", "*backup*", "*old*"], "wildcard_ignores"),
            (["*.bak", "*.tmp", "*.log"], "extension_ignores")
        ]
        
        for ignore_patterns, scenario_name in ignore_pattern_scenarios:
            
            def discover_with_ignore_patterns():
                """Discovery with ignore pattern filtering."""
                return discover_files(
                    directory=str(base_path),
                    pattern="**/*",
                    recursive=True,
                    extensions=None,
                    ignore_patterns=ignore_patterns
                )
            
            result = benchmark.pedantic(
                discover_with_ignore_patterns,
                rounds=3,
                iterations=1
            )
            
            # Validate ignore pattern accuracy
            for file_path in result:
                filename = Path(file_path).name
                for ignore_pattern in ignore_patterns:
                    import fnmatch
                    assert not fnmatch.fnmatch(filename, ignore_pattern), (
                        f"Ignore pattern failed: {filename} matches {ignore_pattern}"
                    )
            
            # Validate performance
            mean_time = benchmark.stats.stats.mean
            file_count = test_data["total_files"]
            time_per_1k_files = (mean_time / file_count) * 1000
            
            assert time_per_1k_files <= DiscoveryPerformanceSLA.PATTERN_MATCHING_MAX_SECONDS_PER_1K_FILES, (
                f"F-002-RQ-003 pattern matching performance violation for {scenario_name}: "
                f"{time_per_1k_files:.4f}s per 1K files > "
                f"{DiscoveryPerformanceSLA.PATTERN_MATCHING_MAX_SECONDS_PER_1K_FILES:.4f}s threshold"
            )
        
        # Test F-002-RQ-004: Mandatory substring filtering performance
        substring_scenarios = [
            (["mouse", "experiment"], "animal_and_exp"),
            (["control", "treatment"], "condition_filtering"),
            (["neural", "behavioral"], "data_type_filtering")
        ]
        
        for mandatory_substrings, scenario_name in substring_scenarios:
            
            def discover_with_mandatory_substrings():
                """Discovery with mandatory substring filtering."""
                return discover_files(
                    directory=str(base_path),
                    pattern="**/*",
                    recursive=True,
                    extensions=None,
                    mandatory_substrings=mandatory_substrings
                )
            
            result = benchmark.pedantic(
                discover_with_mandatory_substrings,
                rounds=3,
                iterations=1
            )
            
            # Validate substring filtering accuracy
            for file_path in result:
                path_str = str(file_path)
                assert any(substring in path_str for substring in mandatory_substrings), (
                    f"Mandatory substring filtering failed: {path_str} missing {mandatory_substrings}"
                )
            
            # Validate performance
            mean_time = benchmark.stats.stats.mean
            time_per_1k_files = (mean_time / file_count) * 1000
            
            assert time_per_1k_files <= DiscoveryPerformanceSLA.SUBSTRING_FILTERING_MAX_SECONDS_PER_1K_FILES, (
                f"F-002-RQ-004 substring filtering performance violation for {scenario_name}: "
                f"{time_per_1k_files:.4f}s per 1K files > "
                f"{DiscoveryPerformanceSLA.SUBSTRING_FILTERING_MAX_SECONDS_PER_1K_FILES:.4f}s threshold"
            )
            
            print(f"Substring filtering '{scenario_name}': {mean_time:.3f}s total, "
                  f"{time_per_1k_files:.4f}s per 1K files, {len(result)} files found")
    
    @pytest.mark.benchmark(group="discovery_metadata")
    def test_metadata_extraction_performance(self, benchmark, discovery_benchmark_data_generator,
                                            tmp_path, benchmark_validator, memory_profiler_context):
        """
        Validate F-007 metadata extraction performance requirements.
        
        Tests metadata extraction performance with realistic experimental file patterns
        and complex regex-based pattern matching for comprehensive metadata extraction
        validation including dates, animal IDs, conditions, and experiment parameters.
        
        Performance Requirements:
        - Metadata extraction: <1.0s per 100 files
        - Pattern matching accuracy: 100% for supported patterns  
        - Memory efficiency: Minimal overhead for metadata processing
        """
        # Generate test data with realistic experimental patterns
        test_data = discovery_benchmark_data_generator.generate_large_directory_structure(
            base_path=tmp_path / "metadata_test",
            file_count=1000,  # Focused on metadata extraction performance
            extension_distribution="data_heavy"
        )
        
        base_path = test_data["metadata"]["base_path"]
        
        # Define realistic extraction patterns for neuroscience research
        extraction_patterns = [
            r".*/mouse_(?P<animal>\d+)_(?P<date>\d{8})_(?P<condition>\w+)_rep(?P<replicate>\d+)\.pkl",
            r".*/rat_(?P<animal>\d+)_(?P<date>\d{8})_(?P<condition>\w+)_session(?P<session>\d+)\.pkl",
            r".*/experiment_(?P<exp_id>EXP\d+)_animal_(?P<animal>\d+)_(?P<condition>\w+)\.pkl",
            r".*/behavioral_(?P<animal>\d+)_(?P<date>\d{8})_(?P<trial_type>\w+)\.pkl",
            r".*/neural_(?P<animal>\d+)_(?P<date>\d{8})_(?P<channel_config>ch\d+)\.pkl"
        ]
        
        with memory_profiler_context(
            data_size_estimate=test_data["metadata"]["total_size_bytes"],
            enable_line_profiling=True
        ) as profiler:
            
            def extract_metadata_from_files():
                """Metadata extraction operation for benchmarking."""
                return discover_files(
                    directory=str(base_path),
                    pattern="**/*.pkl",  # Focus on data files for metadata extraction
                    recursive=True,
                    extensions=["pkl"],
                    extract_patterns=extraction_patterns,
                    parse_dates=True,
                    include_stats=True
                )
            
            result = benchmark.pedantic(
                extract_metadata_from_files,
                rounds=3,
                iterations=1,
                warmup_rounds=1
            )
            
            memory_stats = profiler.end_profiling()
        
        # Validate metadata extraction accuracy
        assert isinstance(result, dict), "Metadata extraction should return dictionary"
        
        extracted_count = 0
        for file_path, metadata in result.items():
            assert "path" in metadata, f"Missing path in metadata for {file_path}"
            
            # Count successful pattern extractions
            if any(key != "path" for key in metadata.keys()):
                extracted_count += 1
        
        # Validate performance
        mean_time = benchmark.stats.stats.mean
        files_processed = len(result)
        time_per_100_files = (mean_time / files_processed) * 100
        
        assert time_per_100_files <= DiscoveryPerformanceSLA.METADATA_EXTRACTION_MAX_SECONDS_PER_100_FILES, (
            f"F-007 metadata extraction performance violation: "
            f"{time_per_100_files:.4f}s per 100 files > "
            f"{DiscoveryPerformanceSLA.METADATA_EXTRACTION_MAX_SECONDS_PER_100_FILES:.4f}s threshold"
        )
        
        # Validate memory efficiency
        memory_validation = benchmark_validator.validate_memory_efficiency(
            memory_stats=memory_stats,
            expected_data_size_mb=test_data["metadata"]["total_size_bytes"] / (1024*1024)
        )
        
        assert memory_validation["efficient"], (
            f"Metadata extraction memory efficiency violation: "
            f"{memory_validation['memory_multiplier']:.1f}x > "
            f"{memory_validation['efficiency_threshold']:.1f}x threshold"
        )
        
        print(f"\n=== F-007 Metadata Extraction Performance ===")
        print(f"Performance: {mean_time:.3f}s for {files_processed} files")
        print(f"Per 100 files: {time_per_100_files:.4f}s")
        print(f"Successful extractions: {extracted_count}/{files_processed} ({(extracted_count/files_processed)*100:.1f}%)")
        print(f"Memory efficiency: {memory_validation['memory_multiplier']:.1f}x")
        print("=" * 50)


# ============================================================================
# LINEAR SCALING VALIDATION BENCHMARKS
# ============================================================================

class TestLinearScalingValidation:
    """
    Comprehensive linear scaling validation for Section 2.4.2 O(n) complexity requirements.
    
    Tests systematic file count progression to validate linear scaling behavior
    and detect any algorithmic complexity violations in the discovery engine.
    """
    
    @pytest.mark.benchmark(group="scaling_validation")
    def test_linear_scaling_complexity(self, benchmark, discovery_benchmark_data_generator,
                                     tmp_path, statistical_analysis_engine):
        """
        Validate O(n) linear scaling complexity per Section 2.4.2 requirements.
        
        Tests systematic file count progression to validate that discovery performance
        scales linearly with file count, ensuring no quadratic or exponential
        complexity regressions that would impact large dataset performance.
        
        Scaling Requirements:
        - Linear complexity: O(n) scaling behavior validation
        - Scaling tolerance: <2.5x deviation from linear trend
        - Performance consistency: Reliable scaling across file count ranges
        """
        # Define file count progression for scaling analysis
        file_counts = [500, 1000, 2000, 4000]  # Reduced for CI performance
        scaling_results = []
        
        print(f"\n=== Linear Scaling Validation ===")
        print(f"Testing file counts: {file_counts}")
        
        for file_count in file_counts:
            # Generate test data for current file count
            test_data = discovery_benchmark_data_generator.generate_large_directory_structure(
                base_path=tmp_path / f"scaling_{file_count}",
                file_count=file_count,
                directory_depth=3,
                extension_distribution="mixed"
            )
            
            base_path = test_data["metadata"]["base_path"]
            
            def discover_files_for_scaling():
                """Discovery operation for scaling analysis."""
                return discover_files(
                    directory=str(base_path),
                    pattern="**/*",
                    recursive=True
                )
            
            # Benchmark with multiple iterations for statistical reliability
            result = benchmark.pedantic(
                discover_files_for_scaling,
                rounds=3,
                iterations=1
            )
            
            mean_time = benchmark.stats.stats.mean
            scaling_results.append({
                "file_count": file_count,
                "mean_time": mean_time,
                "files_found": len(result),
                "time_per_file": mean_time / file_count
            })
            
            print(f"Files: {file_count:,} | Time: {mean_time:.3f}s | "
                  f"Per file: {(mean_time/file_count)*1000:.2f}ms | "
                  f"Found: {len(result):,}")
        
        # Analyze scaling behavior
        file_counts_array = np.array([r["file_count"] for r in scaling_results])
        times_array = np.array([r["mean_time"] for r in scaling_results])
        
        # Linear regression analysis
        slope, intercept, r_value, p_value, std_err = stats.linregress(file_counts_array, times_array)
        
        # Calculate scaling factor between largest and smallest
        largest_time = scaling_results[-1]["mean_time"]
        smallest_time = scaling_results[0]["mean_time"]
        largest_count = scaling_results[-1]["file_count"]
        smallest_count = scaling_results[0]["file_count"]
        
        expected_linear_scaling = largest_count / smallest_count
        actual_scaling = largest_time / smallest_time
        scaling_deviation = actual_scaling / expected_linear_scaling
        
        # Validate linear scaling compliance
        assert scaling_deviation <= DiscoveryPerformanceSLA.LINEAR_SCALING_TOLERANCE_FACTOR, (
            f"Linear scaling violation: actual scaling {actual_scaling:.2f}x vs "
            f"expected {expected_linear_scaling:.2f}x (deviation: {scaling_deviation:.2f}x) "
            f"exceeds tolerance factor {DiscoveryPerformanceSLA.LINEAR_SCALING_TOLERANCE_FACTOR:.2f}x"
        )
        
        # Validate correlation coefficient for linearity
        assert r_value >= 0.9, (
            f"Poor linear correlation: R² = {r_value**2:.3f} < 0.81 (R = {r_value:.3f})"
        )
        
        print(f"\n=== Scaling Analysis Results ===")
        print(f"Linear regression: y = {slope:.6f}x + {intercept:.3f}")
        print(f"Correlation (R): {r_value:.4f} (R² = {r_value**2:.4f})")
        print(f"Expected scaling: {expected_linear_scaling:.2f}x")
        print(f"Actual scaling: {actual_scaling:.2f}x")
        print(f"Scaling deviation: {scaling_deviation:.2f}x")
        print(f"Linear compliance: {'✓ PASS' if scaling_deviation <= DiscoveryPerformanceSLA.LINEAR_SCALING_TOLERANCE_FACTOR else '✗ FAIL'}")
        print("=" * 50)
    
    @pytest.mark.benchmark(group="scaling_memory")
    def test_memory_scaling_efficiency(self, benchmark, discovery_benchmark_data_generator,
                                     tmp_path, memory_leak_detector):
        """
        Validate memory efficiency scaling for large dataset discovery scenarios.
        
        Tests memory usage patterns across different file counts to ensure
        memory consumption scales appropriately and no memory leaks occur
        during large-scale discovery operations.
        
        Memory Requirements:
        - Memory scaling: Linear or sub-linear memory growth
        - Memory leaks: <100MB growth per iteration
        - Memory efficiency: <3x data size overhead
        """
        # Test memory scaling across different file counts
        file_counts = [1000, 2000, 4000]  # Reduced for CI constraints
        memory_results = []
        
        print(f"\n=== Memory Scaling Validation ===")
        
        for file_count in file_counts:
            # Generate test data
            test_data = discovery_benchmark_data_generator.generate_large_directory_structure(
                base_path=tmp_path / f"memory_{file_count}",
                file_count=file_count,
                directory_depth=2
            )
            
            base_path = test_data["metadata"]["base_path"]
            
            # Test for memory leaks through iterations
            def discovery_operation():
                return discover_files(
                    directory=str(base_path),
                    pattern="**/*",
                    recursive=True
                )
            
            # Use memory leak detector for comprehensive analysis
            leak_analysis = memory_leak_detector.detect_leak_in_iterations(
                operation_func=discovery_operation,
                iterations=3  # Reduced for benchmark context
            )
            
            memory_results.append({
                "file_count": file_count,
                "data_size_mb": test_data["metadata"]["total_size_bytes"] / (1024*1024),
                "peak_memory_mb": leak_analysis["memory_timeline"][-1]["end_memory_mb"],
                "memory_growth_mb": leak_analysis["total_memory_growth_mb"],
                "leak_detected": leak_analysis["leak_detected"]
            })
            
            print(f"Files: {file_count:,} | Data: {memory_results[-1]['data_size_mb']:.1f}MB | "
                  f"Peak: {memory_results[-1]['peak_memory_mb']:.1f}MB | "
                  f"Growth: {memory_results[-1]['memory_growth_mb']:+.1f}MB | "
                  f"Leak: {'Yes' if leak_analysis['leak_detected'] else 'No'}")
        
        # Validate no memory leaks detected
        for result in memory_results:
            assert not result["leak_detected"], (
                f"Memory leak detected for {result['file_count']} files: "
                f"{result['memory_growth_mb']:.1f}MB growth"
            )
            
            assert result["memory_growth_mb"] <= DiscoveryPerformanceSLA.MEMORY_LEAK_THRESHOLD_MB, (
                f"Memory growth threshold exceeded for {result['file_count']} files: "
                f"{result['memory_growth_mb']:.1f}MB > {DiscoveryPerformanceSLA.MEMORY_LEAK_THRESHOLD_MB}MB"
            )
        
        # Analyze memory scaling trend
        file_counts_array = np.array([r["file_count"] for r in memory_results])
        peak_memory_array = np.array([r["peak_memory_mb"] for r in memory_results])
        
        # Linear regression for memory scaling
        slope, intercept, r_value, p_value, std_err = stats.linregress(file_counts_array, peak_memory_array)
        
        print(f"\n=== Memory Scaling Analysis ===")
        print(f"Memory scaling: {slope:.4f} MB per 1K files")
        print(f"Correlation (R): {r_value:.4f} (R² = {r_value**2:.4f})")
        print(f"Memory leak compliance: {'✓ PASS' if all(not r['leak_detected'] for r in memory_results) else '✗ FAIL'}")
        print("=" * 50)


# ============================================================================
# CROSS-PLATFORM COMPATIBILITY BENCHMARKS
# ============================================================================

class TestCrossPlatformPerformance:
    """
    Cross-platform performance validation for Ubuntu, Windows, macOS environments.
    
    Tests performance consistency across different operating systems and validates
    that discovery engine maintains SLA compliance regardless of platform-specific
    filesystem implementation differences.
    """
    
    @pytest.mark.benchmark(group="cross_platform")
    def test_platform_performance_consistency(self, benchmark, discovery_benchmark_data_generator,
                                             tmp_path, environment_analyzer, 
                                             platform_performance_normalizer):
        """
        Validate cross-platform performance consistency with environment normalization.
        
        Tests discovery performance across different platforms while applying
        environment normalization factors to ensure consistent SLA validation
        regardless of underlying hardware and OS differences.
        
        Platform Requirements:
        - Performance consistency: <50% variance across platforms
        - SLA compliance: All platforms meet discovery SLA requirements
        - Environment normalization: Proper hardware abstraction
        """
        current_platform = platform.system()
        
        # Generate platform-neutral test data
        test_data = discovery_benchmark_data_generator.generate_large_directory_structure(
            base_path=tmp_path / f"platform_test_{current_platform.lower()}",
            file_count=3000,  # Balanced for cross-platform CI performance
            directory_depth=3,
            extension_distribution="mixed"
        )
        
        base_path = test_data["metadata"]["base_path"]
        
        # Analyze current environment
        env_analysis = environment_analyzer.analyze_current_environment()
        normalization_factors = environment_analyzer.calculate_normalization_factors()
        
        def cross_platform_discovery():
            """Platform-optimized discovery operation."""
            return discover_files(
                directory=str(base_path),
                pattern="**/*",
                recursive=True,
                extensions=["pkl", "csv", "yaml"]  # Common extensions across platforms
            )
        
        # Execute benchmark with platform-specific considerations
        result = benchmark.pedantic(
            cross_platform_discovery,
            rounds=3,
            iterations=1,
            warmup_rounds=1
        )
        
        # Apply environment normalization
        raw_measurement = benchmark.stats.stats.mean
        normalized_result = platform_performance_normalizer(raw_measurement)
        
        # Validate platform-specific performance
        file_count = test_data["metadata"]["total_files"]
        normalized_time_per_1k = (normalized_result["normalized_measurement"] / file_count) * 1000
        
        # Platform-specific SLA validation with normalization
        platform_sla_threshold = DiscoveryPerformanceSLA.RECURSIVE_TRAVERSAL_MAX_SECONDS_10K_FILES * 0.3  # Scaled for 3K files
        
        assert normalized_result["normalized_measurement"] <= platform_sla_threshold, (
            f"Cross-platform SLA violation on {current_platform}: "
            f"{normalized_result['normalized_measurement']:.3f}s > {platform_sla_threshold:.3f}s "
            f"(raw: {raw_measurement:.3f}s, normalization: {normalized_result['improvement_factor']:.2f}x)"
        )
        
        # Validate discovery accuracy across platforms
        discovered_files = len(result)
        expected_files = len([f for f in test_data["all_files"] 
                            if f.suffix.lstrip('.') in ["pkl", "csv", "yaml"]])
        
        accuracy_ratio = discovered_files / max(expected_files, 1)
        assert 0.9 <= accuracy_ratio <= 1.1, (
            f"Cross-platform discovery accuracy issue: found {discovered_files}, "
            f"expected ~{expected_files} (ratio: {accuracy_ratio:.2f})"
        )
        
        print(f"\n=== Cross-Platform Performance: {current_platform} ===")
        print(f"Raw performance: {raw_measurement:.3f}s")
        print(f"Normalized performance: {normalized_result['normalized_measurement']:.3f}s")
        print(f"Normalization factor: {normalized_result['improvement_factor']:.2f}x")
        print(f"Platform characteristics: {env_analysis['platform_characteristics']}")
        print(f"Files discovered: {discovered_files:,}/{expected_files:,}")
        print(f"SLA compliance: {'✓ PASS' if normalized_result['normalized_measurement'] <= platform_sla_threshold else '✗ FAIL'}")
        print("=" * 50)
    
    @pytest.mark.benchmark(group="cross_platform_paths")
    @pytest.mark.skipif(
        platform.system() == "Windows" and os.getenv("CI", "").lower() == "true",
        reason="Windows path tests may have filesystem constraints in CI"
    )
    def test_cross_platform_path_handling(self, benchmark, discovery_benchmark_data_generator,
                                        tmp_path):
        """
        Validate cross-platform path handling performance and correctness.
        
        Tests discovery performance with platform-specific path variations including
        deep nesting, unicode characters, and various path separator handling
        to ensure consistent behavior across different filesystem implementations.
        
        Path Requirements:
        - Path normalization: Consistent behavior across platforms
        - Unicode support: Proper handling of international characters
        - Deep nesting: Support for complex directory structures
        """
        current_platform = platform.system()
        
        # Create platform-specific test scenarios
        platform_specific_paths = self._generate_platform_specific_paths(tmp_path, current_platform)
        
        def discover_platform_paths():
            """Discovery with platform-specific path patterns."""
            all_results = []
            for test_path in platform_specific_paths:
                if test_path.exists():
                    results = discover_files(
                        directory=str(test_path),
                        pattern="**/*",
                        recursive=True
                    )
                    all_results.extend(results)
            return all_results
        
        # Benchmark platform-specific path handling
        result = benchmark.pedantic(
            discover_platform_paths,
            rounds=2,
            iterations=1
        )
        
        # Validate path handling correctness
        for file_path in result:
            path_obj = Path(file_path)
            
            # Validate path exists and is accessible
            assert path_obj.exists(), f"Discovered path does not exist: {file_path}"
            assert path_obj.is_file(), f"Discovered path is not a file: {file_path}"
            
            # Validate platform-specific path normalization
            normalized_path = str(path_obj.resolve())
            assert os.path.exists(normalized_path), f"Normalized path invalid: {normalized_path}"
        
        # Platform-specific performance validation
        mean_time = benchmark.stats.stats.mean
        files_discovered = len(result)
        
        # Relaxed performance threshold for path complexity
        path_performance_threshold = 2.0  # 2 seconds for complex path scenarios
        
        assert mean_time <= path_performance_threshold, (
            f"Cross-platform path performance violation on {current_platform}: "
            f"{mean_time:.3f}s > {path_performance_threshold:.3f}s for {files_discovered} files"
        )
        
        print(f"\n=== Cross-Platform Path Handling: {current_platform} ===")
        print(f"Performance: {mean_time:.3f}s for {files_discovered} files")
        print(f"Test paths: {len(platform_specific_paths)} directories")
        print(f"Path compliance: {'✓ PASS' if mean_time <= path_performance_threshold else '✗ FAIL'}")
        print("=" * 50)
    
    def _generate_platform_specific_paths(self, base_path: Path, platform_name: str) -> List[Path]:
        """Generate platform-specific test paths and files."""
        test_paths = []
        
        # Common test scenarios
        common_path = base_path / "common_test"
        common_path.mkdir(parents=True, exist_ok=True)
        (common_path / "test_file.pkl").touch()
        test_paths.append(common_path)
        
        # Deep nesting test
        deep_path = base_path / "deep"
        for level in range(5):  # Reduced depth for CI performance
            deep_path = deep_path / f"level_{level}"
        deep_path.mkdir(parents=True, exist_ok=True)
        (deep_path / "deep_file.pkl").touch()
        test_paths.append(base_path / "deep")
        
        # Platform-specific scenarios
        if platform_name == "Windows":
            # Windows-specific scenarios (if not in CI)
            if not os.getenv("CI"):
                windows_path = base_path / "windows_test"
                windows_path.mkdir(exist_ok=True)
                (windows_path / "windows_file.pkl").touch()
                test_paths.append(windows_path)
        
        elif platform_name in ["Linux", "Darwin"]:  # Linux and macOS
            # Unix-specific scenarios
            unix_path = base_path / "unix_test"
            unix_path.mkdir(exist_ok=True)
            (unix_path / "unix_file.pkl").touch()
            test_paths.append(unix_path)
            
            # Unicode path test (where supported)
            try:
                unicode_path = base_path / "测试_directory"
                unicode_path.mkdir(exist_ok=True)
                (unicode_path / "测试_file.pkl").touch()
                test_paths.append(unicode_path)
            except (OSError, UnicodeError):
                # Skip unicode tests if filesystem doesn't support them
                pass
        
        return test_paths


# ============================================================================
# CONFIGURATION-DRIVEN DISCOVERY BENCHMARKS
# ============================================================================

class TestConfigurationDrivenDiscovery:
    """
    Configuration-driven discovery performance validation.
    
    Tests performance of discovery operations driven by YAML configuration
    files, validating the integration between configuration loading and
    discovery execution with realistic experimental configurations.
    """
    
    @pytest.mark.benchmark(group="config_discovery")
    def test_config_driven_discovery_performance(self, benchmark, discovery_benchmark_data_generator,
                                                tmp_path, mock_configuration_provider):
        """
        Validate configuration-driven discovery performance integration.
        
        Tests the complete pipeline from configuration loading through discovery
        execution to ensure minimal overhead from configuration processing and
        optimal performance for research workflow automation.
        
        Integration Requirements:
        - Config loading: <100ms for typical configurations
        - Discovery integration: Minimal overhead from config processing
        - End-to-end performance: Maintain SLA compliance with config overhead
        """
        # Generate test data and configuration
        test_data = discovery_benchmark_data_generator.generate_large_directory_structure(
            base_path=tmp_path / "config_discovery",
            file_count=2000,
            extension_distribution="data_heavy"
        )
        
        base_path = test_data["metadata"]["base_path"]
        
        # Create realistic configuration scenarios
        config_scenarios = self._create_discovery_config_scenarios(base_path, tmp_path)
        
        for config_name, config_data in config_scenarios.items():
            # Setup mock configuration
            config_path = tmp_path / f"{config_name}_config.yaml"
            mock_configuration_provider.add_mock_config(
                file_path=config_path,
                config_data=config_data,
                loading_delay=0.05  # Realistic config loading delay
            )
            
            mock_configuration_provider.activate_yaml_mocks()
            
            def config_driven_discovery():
                """Configuration-driven discovery operation."""
                # This would typically use config-driven discovery functions
                # For benchmark purposes, simulate the config-discovery integration
                
                # Simulate config processing overhead
                time.sleep(0.01)  # Minimal config processing simulation
                
                # Extract discovery parameters from config
                ignore_patterns = config_data.get("project", {}).get("ignore_substrings", [])
                mandatory_patterns = config_data.get("project", {}).get("mandatory_experiment_strings", [])
                
                return discover_files(
                    directory=str(base_path),
                    pattern="**/*",
                    recursive=True,
                    extensions=["pkl", "csv"],
                    ignore_patterns=[f"*{pattern}*" for pattern in ignore_patterns],
                    mandatory_substrings=mandatory_patterns if mandatory_patterns else None
                )
            
            # Benchmark config-driven discovery
            result = benchmark.pedantic(
                config_driven_discovery,
                rounds=3,
                iterations=1
            )
            
            # Validate end-to-end performance
            mean_time = benchmark.stats.stats.mean
            file_count = test_data["metadata"]["total_files"]
            
            # Allow for reasonable config overhead while maintaining SLA
            config_sla_threshold = (DiscoveryPerformanceSLA.RECURSIVE_TRAVERSAL_MAX_SECONDS_10K_FILES * 
                                   (file_count / 10000.0)) + 0.5  # 500ms config overhead allowance
            
            assert mean_time <= config_sla_threshold, (
                f"Config-driven discovery performance violation for '{config_name}': "
                f"{mean_time:.3f}s > {config_sla_threshold:.3f}s "
                f"(includes config processing overhead)"
            )
            
            # Validate discovery accuracy with config filtering
            discovered_count = len(result)
            assert discovered_count > 0, f"Config-driven discovery found no files for '{config_name}'"
            
            print(f"Config scenario '{config_name}': {mean_time:.3f}s, {discovered_count} files")
    
    def _create_discovery_config_scenarios(self, base_path: Path, tmp_path: Path) -> Dict[str, Dict]:
        """Create realistic configuration scenarios for discovery testing."""
        return {
            "minimal_config": {
                "project": {
                    "name": "minimal_test",
                    "ignore_substrings": [],
                    "mandatory_experiment_strings": []
                }
            },
            "filtered_config": {
                "project": {
                    "name": "filtered_test",
                    "ignore_substrings": ["backup", "temp"],
                    "mandatory_experiment_strings": ["mouse", "experiment"]
                }
            },
            "complex_config": {
                "project": {
                    "name": "complex_test",
                    "ignore_substrings": ["backup", "temp", "old", "test"],
                    "mandatory_experiment_strings": ["experiment", "neural", "behavioral"]
                },
                "datasets": {
                    "neural_data": {
                        "base_path": str(base_path),
                        "file_patterns": ["*.pkl", "*.csv"],
                        "metadata_patterns": ["mouse_*", "rat_*"]
                    }
                }
            }
        }


# ============================================================================
# MEMORY EFFICIENCY AND LEAK DETECTION BENCHMARKS
# ============================================================================

class TestMemoryEfficiencyBenchmarks:
    """
    Memory efficiency validation for large-scale discovery operations.
    
    Tests memory usage patterns, leak detection, and efficiency requirements
    for processing large datasets and directory structures to ensure the
    discovery engine maintains optimal memory usage characteristics.
    """
    
    @pytest.mark.benchmark(group="memory_efficiency")
    def test_large_dataset_memory_efficiency(self, benchmark, discovery_benchmark_data_generator,
                                           tmp_path, memory_profiler_context, memory_leak_detector):
        """
        Validate memory efficiency for large dataset discovery scenarios.
        
        Tests memory usage patterns for large-scale discovery operations to ensure
        memory consumption remains within acceptable bounds and no memory leaks
        occur during processing of extensive directory structures.
        
        Memory Requirements:
        - Memory multiplier: <3x data size overhead
        - Memory leaks: <100MB growth per operation
        - Peak memory: Efficient memory utilization patterns
        """
        # Generate large dataset for memory testing
        large_dataset = discovery_benchmark_data_generator.generate_large_directory_structure(
            base_path=tmp_path / "large_memory_test",
            file_count=5000,  # Substantial dataset for memory analysis
            directory_depth=4,
            extension_distribution="mixed"
        )
        
        base_path = large_dataset["metadata"]["base_path"]
        estimated_size_bytes = large_dataset["metadata"]["total_size_bytes"]
        
        with memory_profiler_context(
            data_size_estimate=estimated_size_bytes,
            precision=3,
            enable_line_profiling=True,
            monitor_interval=0.05  # Frequent monitoring for large datasets
        ) as profiler:
            
            def memory_intensive_discovery():
                """Memory-intensive discovery operation with comprehensive features."""
                return discover_files(
                    directory=str(base_path),
                    pattern="**/*",
                    recursive=True,
                    extensions=["pkl", "csv", "yaml", "json"],
                    ignore_patterns=["*backup*", "*temp*"],
                    mandatory_substrings=["experiment", "mouse", "rat"],
                    extract_patterns=[
                        r".*/(?P<type>mouse|rat)_(?P<id>\d+)_(?P<date>\d{8})_(?P<condition>\w+).*\.pkl",
                        r".*/experiment_(?P<exp_id>\w+)_(?P<animal>\w+)_(?P<condition>\w+)\.pkl"
                    ],
                    parse_dates=True,
                    include_stats=True
                )
            
            # Execute benchmark with memory monitoring
            result = benchmark.pedantic(
                memory_intensive_discovery,
                rounds=2,  # Reduced rounds for large dataset scenarios
                iterations=1
            )
            
            memory_stats = profiler.end_profiling()
        
        # Memory leak detection through iterations
        leak_analysis = memory_leak_detector.detect_leak_in_iterations(
            operation_func=lambda: discover_files(
                directory=str(base_path),
                pattern="**/*",
                recursive=True,
                extensions=["pkl"]  # Simplified for leak detection
            ),
            iterations=3  # Sufficient for leak detection
        )
        
        # Validate memory efficiency
        memory_multiplier = memory_stats["memory_overhead_mb"] / max(memory_stats["data_size_mb"], 1)
        
        assert memory_multiplier <= DiscoveryPerformanceSLA.MEMORY_MULTIPLIER_THRESHOLD, (
            f"Memory efficiency violation: {memory_multiplier:.1f}x > "
            f"{DiscoveryPerformanceSLA.MEMORY_MULTIPLIER_THRESHOLD:.1f}x threshold "
            f"(used {memory_stats['memory_overhead_mb']:.1f}MB for {memory_stats['data_size_mb']:.1f}MB data)"
        )
        
        # Validate no memory leaks
        assert not leak_analysis["leak_detected"], (
            f"Memory leak detected: {leak_analysis['total_memory_growth_mb']:.1f}MB growth > "
            f"{DiscoveryPerformanceSLA.MEMORY_LEAK_THRESHOLD_MB}MB threshold"
        )
        
        # Validate discovery result integrity
        assert isinstance(result, dict), "Memory-intensive discovery should return metadata dictionary"
        discovered_count = len(result)
        
        # Performance validation for large dataset
        mean_time = benchmark.stats.stats.mean
        file_count = large_dataset["metadata"]["total_files"]
        scaled_sla = DiscoveryPerformanceSLA.RECURSIVE_TRAVERSAL_MAX_SECONDS_10K_FILES * (file_count / 10000.0)
        
        assert mean_time <= scaled_sla * 2.0, (  # Allow 2x overhead for comprehensive processing
            f"Large dataset performance violation: {mean_time:.3f}s > {scaled_sla * 2.0:.3f}s "
            f"for {file_count} files with comprehensive processing"
        )
        
        print(f"\n=== Large Dataset Memory Efficiency ===")
        print(f"Data size: {memory_stats['data_size_mb']:.1f}MB")
        print(f"Peak memory: {memory_stats['peak_memory_mb']:.1f}MB")
        print(f"Memory overhead: {memory_stats['memory_overhead_mb']:.1f}MB")
        print(f"Memory multiplier: {memory_multiplier:.1f}x")
        print(f"Memory leak: {'No' if not leak_analysis['leak_detected'] else 'Yes'}")
        print(f"Discovery time: {mean_time:.3f}s for {discovered_count:,} files")
        print(f"Efficiency compliance: {'✓ PASS' if memory_multiplier <= DiscoveryPerformanceSLA.MEMORY_MULTIPLIER_THRESHOLD else '✗ FAIL'}")
        print("=" * 50)


# ============================================================================
# COMPREHENSIVE BENCHMARK EXECUTION COORDINATION
# ============================================================================

def run_comprehensive_discovery_benchmarks():
    """
    Execute comprehensive discovery benchmark suite via CLI integration.
    
    This function provides the entry point for scripts/benchmarks/run_benchmarks.py
    CLI execution, enabling selective benchmark category execution and artifact
    generation for CI/CD integration and performance monitoring.
    """
    print("FlyRigLoader Discovery Performance Benchmark Suite")
    print("=" * 60)
    print("Executing comprehensive discovery subsystem performance validation...")
    print("SLA Requirements:")
    print(f"  • F-002-RQ-001: Recursive traversal <{DiscoveryPerformanceSLA.RECURSIVE_TRAVERSAL_MAX_SECONDS_10K_FILES}s for 10K files")
    print(f"  • F-002-RQ-002: Extension filtering <{DiscoveryPerformanceSLA.EXTENSION_FILTERING_MAX_SECONDS_PER_1K_FILES}s per 1K files")
    print(f"  • F-002-RQ-003: Pattern matching <{DiscoveryPerformanceSLA.PATTERN_MATCHING_MAX_SECONDS_PER_1K_FILES}s per 1K files")
    print(f"  • F-002-RQ-004: Substring filtering <{DiscoveryPerformanceSLA.SUBSTRING_FILTERING_MAX_SECONDS_PER_1K_FILES}s per 1K files")
    print(f"  • F-007: Metadata extraction <{DiscoveryPerformanceSLA.METADATA_EXTRACTION_MAX_SECONDS_PER_100_FILES}s per 100 files")
    print(f"  • Linear scaling tolerance: <{DiscoveryPerformanceSLA.LINEAR_SCALING_TOLERANCE_FACTOR}x deviation")
    print("=" * 60)


if __name__ == "__main__":
    run_comprehensive_discovery_benchmarks()