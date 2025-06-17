"""
Centralized pytest fixture configuration for the flyrigloader benchmark test suite.

This module provides comprehensive fixture management for performance testing, statistical
analysis, memory profiling, cross-platform compatibility, and CI/CD artifact generation
while maintaining complete isolation from the default test suite execution.

Key Features:
- Consolidated fixture management from all benchmark test modules per Section 0 requirements
- Performance test environment setup with consistent benchmark execution across platforms
- Statistical analysis support with confidence intervals and regression detection
- Memory profiling integration with pytest-memory-profiler for large dataset scenarios
- Cross-platform compatibility testing across Ubuntu, Windows, macOS environments
- CI/CD artifact support for JSON/CSV performance reports and trend analysis
- Mock provider factories for benchmark isolation and dependency injection
- Session-scoped configuration management for comprehensive performance validation

Integration:
- pytest-benchmark for statistical performance measurement and comparison
- pytest-memory-profiler for line-by-line memory analysis and leak detection
- psutil for cross-platform system resource monitoring and environment normalization
- scipy.stats for confidence interval calculation and regression detection significance testing
- GitHub Actions for CI/CD artifact management with 90-day retention policy compliance
- Cross-platform performance validation with hardware abstraction and normalization factors

Fixture Categories:
- Configuration fixtures: Session-scoped benchmark configuration and environment setup
- Data generation fixtures: Synthetic test data for large dataset scenarios and memory profiling
- Mock provider fixtures: Centralized mock implementations for filesystem and data loading
- Memory profiling fixtures: Memory leak detection and efficiency validation utilities
- Performance analysis fixtures: Statistical analysis engines and regression detection utilities
- Artifact generation fixtures: Performance report generation and CI/CD integration support
- Environment fixtures: Cross-platform normalization and benchmark execution coordination
"""

import gc
import json
import os
import platform
import tempfile
import threading
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Generator, Tuple
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import psutil
import pytest
from hypothesis import strategies as st

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
    CICDIntegrationManager,
    BenchmarkUtilsCoordinator,
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

# Test data directory imports (will be created as needed)
try:
    # These modules may not exist yet - we'll create fixtures to support them
    from tests.conftest import test_data_generator as base_test_data_generator
except ImportError:
    base_test_data_generator = None


# ============================================================================
# SESSION-SCOPED CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def benchmark_config():
    """
    Session-scoped benchmark configuration providing unified access to all
    performance SLA thresholds, statistical analysis parameters, and execution settings.
    
    Features:
    - Auto-detection of CI vs local environment with appropriate threshold adjustments
    - Cross-platform performance normalization factors for consistent results
    - Memory profiling configuration with pytest-memory-profiler integration
    - Statistical analysis parameters for confidence intervals and regression detection
    - CI/CD integration settings for GitHub Actions artifact management
    
    Returns:
        BenchmarkConfig: Comprehensive configuration instance for benchmark execution
    """
    # Auto-detect environment and configure appropriately
    config = get_benchmark_config("auto")
    
    # Log configuration details for benchmark traceability
    env_characteristics = config.environment_normalization.detect_environment_characteristics()
    
    print(f"\n=== Benchmark Configuration Initialized ===")
    print(f"Environment: {'CI' if env_characteristics['is_ci_environment'] else 'Local'}")
    print(f"Platform: {env_characteristics['platform']}")
    print(f"CPU: {env_characteristics['cpu_count']} cores @ {env_characteristics['cpu_frequency_mhz']:.0f}MHz")
    print(f"Memory: {env_characteristics['memory_gb']:.1f}GB ({env_characteristics['memory_usage_percent']:.1f}% used)")
    print(f"Virtualized: {env_characteristics['is_virtualized']}")
    print(f"Normalization Factor: {env_characteristics['normalization_factors']['combined']:.3f}")
    print("=" * 50)
    
    return config


@pytest.fixture(scope="session")  
def comprehensive_column_config():
    """
    Session-scoped fixture providing comprehensive column configuration for DataFrame
    transformation benchmarks supporting various experimental data scenarios.
    
    Features:
    - Neural recording column definitions with realistic data types
    - Behavioral tracking column schemas for experimental validation
    - Metadata column configurations for experiment tracking
    - Performance-optimized column arrangements for transformation benchmarks
    - Cross-platform data type compatibility validation
    
    Returns:
        Dict: Comprehensive column configuration supporting all benchmark scenarios
    """
    return {
        "neural_recording": {
            "timestamp": "datetime64[ns]",
            "channel_1": "float64",
            "channel_2": "float64", 
            "channel_3": "float64",
            "channel_4": "float64",
            "sync_signal": "bool",
            "trial_number": "int32",
            "condition": "category"
        },
        "behavioral_tracking": {
            "timestamp": "datetime64[ns]",
            "x_position": "float32",
            "y_position": "float32",
            "velocity": "float32",
            "orientation": "float32",
            "behavior_state": "category",
            "trial_id": "int32",
            "session_id": "category"
        },
        "experimental_metadata": {
            "animal_id": "category",
            "experiment_date": "datetime64[ns]",
            "experimenter": "category",
            "condition": "category",
            "replicate": "int16",
            "session_duration_minutes": "int16",
            "notes": "string"
        },
        "large_dataset_columns": {
            # Optimized for memory profiling scenarios >500MB
            f"signal_{i:03d}": "float32" for i in range(100)
        }
    }


@pytest.fixture(scope="session")
def benchmark_execution_environment(benchmark_config):
    """
    Session-scoped fixture establishing consistent benchmark execution environment
    with performance normalization and statistical analysis capabilities.
    
    Features:
    - Environment normalization for cross-platform consistency
    - Statistical analysis engine initialization with confidence interval support
    - Memory profiling configuration for large dataset scenarios
    - Performance baseline establishment for regression detection
    - CI/CD integration setup for artifact generation and alerting
    
    Returns:
        Dict: Comprehensive execution environment with analysis utilities
    """
    # Initialize all analysis components
    statistical_engine = StatisticalAnalysisEngine(benchmark_config.statistical_analysis)
    environment_analyzer = EnvironmentAnalyzer(benchmark_config.environment_normalization)
    artifact_generator = PerformanceArtifactGenerator(benchmark_config.cicd_integration)
    regression_detector = RegressionDetector(statistical_engine, benchmark_config.statistical_analysis)
    cross_platform_validator = CrossPlatformValidator(benchmark_config.environment_normalization)
    cicd_manager = CICDIntegrationManager(benchmark_config.cicd_integration)
    
    # Analyze current environment
    env_report = environment_analyzer.generate_environment_report()
    
    # Calculate normalization factors
    normalization_factors = environment_analyzer.calculate_normalization_factors()
    
    # Setup artifact directories if in CI environment
    artifact_dirs = None
    if cicd_manager.is_github_actions:
        artifact_dirs = cicd_manager.setup_artifact_directories()
    
    execution_env = {
        "statistical_engine": statistical_engine,
        "environment_analyzer": environment_analyzer,
        "artifact_generator": artifact_generator,
        "regression_detector": regression_detector,
        "cross_platform_validator": cross_platform_validator,
        "cicd_manager": cicd_manager,
        "environment_report": env_report,
        "normalization_factors": normalization_factors,
        "artifact_directories": artifact_dirs,
        "baseline_performance_targets": {
            "data_loading_per_100mb_seconds": benchmark_config.sla.DATA_LOADING_TIME_PER_100MB_SECONDS,
            "transformation_per_1m_rows_ms": benchmark_config.sla.DATAFRAME_TRANSFORM_TIME_PER_1M_ROWS_MS,
            "discovery_per_10k_files_seconds": benchmark_config.sla.FILE_DISCOVERY_TIME_FOR_10K_FILES_SECONDS,
            "config_loading_ms": benchmark_config.sla.CONFIG_LOADING_TIME_MS
        }
    }
    
    print(f"\n=== Benchmark Execution Environment Ready ===")
    print(f"Environment Suitability: {env_report['benchmarking_suitability']['level']}")
    print(f"Normalization Factor: {normalization_factors['combined']:.3f}")
    print(f"CI Integration: {'Enabled' if cicd_manager.is_github_actions else 'Disabled'}")
    print(f"Statistical Engine: Initialized with {benchmark_config.statistical_analysis.confidence_level*100}% confidence")
    print("=" * 50)
    
    return execution_env


# ============================================================================
# DATA GENERATION FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def synthetic_data_generator(benchmark_config):
    """
    Session-scoped fixture providing comprehensive synthetic data generation
    for benchmark testing scenarios including large dataset support.
    
    Features:
    - Memory-efficient data generation for datasets >500MB
    - Realistic experimental data patterns for neuroscience research
    - Configurable data sizes supporting all benchmark categories
    - Cross-platform compatible data generation with consistent results
    - Performance-optimized data structures for transformation benchmarks
    
    Returns:
        SyntheticDataGenerator: Comprehensive data generation utility
    """
    class SyntheticDataGenerator:
        def __init__(self, config: BenchmarkConfig):
            self.config = config
            self.random_seed = 42
            np.random.seed(self.random_seed)
            
            # Memory-efficient data generation settings
            self.chunk_size = 10_000  # Process data in chunks for large datasets
            self.data_type_defaults = {
                "neural": np.float32,      # Optimize memory for neural data
                "behavioral": np.float32,   # Optimize memory for behavioral data  
                "metadata": object,         # Flexible type for metadata
                "large_dataset": np.float32 # Memory-optimized for >500MB scenarios
            }
        
        def generate_large_test_directory(self, 
                                        base_path: Path, 
                                        file_count: int = 1000,
                                        size_distribution: str = "realistic") -> Dict[str, Any]:
            """
            Generate large test directory structure for file discovery benchmarks.
            
            Args:
                base_path: Base directory path for test file generation
                file_count: Number of files to generate (default supports F-002-RQ-001 SLA)
                size_distribution: Distribution pattern ("realistic", "uniform", "mixed")
                
            Returns:
                Dict containing directory structure and metadata for benchmark validation
            """
            base_path.mkdir(parents=True, exist_ok=True)
            
            # Create realistic subdirectory structure
            subdirs = [
                "raw_data", "processed", "backup", "temp", "archive",
                "experiments/exp_001", "experiments/exp_002", "experiments/exp_003",
                "animals/mouse_cohort", "animals/rat_cohort",
                "sessions/2024_q1", "sessions/2024_q2"
            ]
            
            for subdir in subdirs:
                (base_path / subdir).mkdir(parents=True, exist_ok=True)
            
            # Generate files with realistic naming patterns
            file_patterns = [
                "mouse_{animal_id}_{date}_{condition}_rep{replicate}.pkl",
                "rat_{animal_id}_{date}_{condition}_session{replicate}.pkl", 
                "experiment_{exp_id}_{date}_{condition}.pkl",
                "behavioral_data_{animal_id}_{date}.pkl",
                "neural_recording_{animal_id}_{date}_{session}.pkl",
                "metadata_{exp_id}_{date}.yaml",
                "config_{exp_id}.yaml",
                "backup_{animal_id}_{date}.bak",  # Should be ignored by discovery
                "temp_{uuid}.tmp"  # Should be ignored by discovery
            ]
            
            generated_files = []
            files_by_category = {"valid": [], "ignored": []}
            
            for i in range(file_count):
                # Select pattern and subdir
                pattern = np.random.choice(file_patterns)
                subdir = np.random.choice(subdirs)
                
                # Generate realistic metadata
                animal_id = f"animal_{np.random.randint(1, 100):03d}"
                date = (datetime.now() - timedelta(days=np.random.randint(0, 365))).strftime("%Y%m%d")
                condition = np.random.choice(["control", "treatment_a", "treatment_b"])
                replicate = np.random.randint(1, 6)
                exp_id = f"EXP{np.random.randint(1, 50):03d}"
                session = f"session_{np.random.randint(1, 10):02d}"
                uuid_str = f"{np.random.randint(10000, 99999)}"
                
                # Format filename
                filename = pattern.format(
                    animal_id=animal_id,
                    date=date,
                    condition=condition,
                    replicate=replicate,
                    exp_id=exp_id,
                    session=session,
                    uuid=uuid_str
                )
                
                file_path = base_path / subdir / filename
                
                # Create file with realistic size
                if size_distribution == "realistic":
                    # Most files small, few large ones
                    if np.random.random() < 0.8:
                        size = np.random.randint(1024, 10*1024)  # 1-10KB
                    elif np.random.random() < 0.95:
                        size = np.random.randint(10*1024, 1024*1024)  # 10KB-1MB  
                    else:
                        size = np.random.randint(1024*1024, 100*1024*1024)  # 1-100MB
                elif size_distribution == "uniform":
                    size = np.random.randint(1024, 10*1024*1024)  # 1KB-10MB uniform
                else:  # mixed
                    size = np.random.choice([1024, 10*1024, 1024*1024, 10*1024*1024])
                
                # Create file with specified size
                with open(file_path, 'wb') as f:
                    f.write(b'0' * size)
                
                generated_files.append(file_path)
                
                # Categorize for discovery testing
                if any(ignore_pattern in filename for ignore_pattern in ["backup", "temp"]):
                    files_by_category["ignored"].append(file_path)
                else:
                    files_by_category["valid"].append(file_path)
            
            return {
                "directory": base_path,
                "total_files": len(generated_files),
                "files_by_category": files_by_category,
                "subdirectories": subdirs,
                "size_distribution": size_distribution,
                "generation_metadata": {
                    "patterns_used": file_patterns,
                    "total_size_bytes": sum(f.stat().st_size for f in generated_files),
                    "avg_file_size_bytes": np.mean([f.stat().st_size for f in generated_files]),
                    "max_file_size_bytes": max(f.stat().st_size for f in generated_files)
                }
            }
        
        def generate_pattern_test_directory(self, 
                                          base_path: Path,
                                          complexity_level: str = "moderate") -> Dict[str, Any]:
            """
            Generate directory structure for pattern matching benchmark validation.
            
            Args:
                base_path: Base directory for pattern test files
                complexity_level: Pattern complexity ("simple", "moderate", "complex")
                
            Returns:
                Dict containing pattern test structure and expected matching results
            """
            base_path.mkdir(parents=True, exist_ok=True)
            
            # Define pattern complexity configurations
            complexity_configs = {
                "simple": {
                    "patterns": ["*.pkl", "mouse_*.pkl", "*_control_*.pkl"],
                    "file_count": 100,
                    "nesting_depth": 2
                },
                "moderate": {
                    "patterns": ["mouse_*_control_rep*.pkl", "**/experiment_*.pkl", "**/*session*.pkl"],
                    "file_count": 500,
                    "nesting_depth": 4
                },
                "complex": {
                    "patterns": [
                        "**/mouse_[0-9][0-9][0-9]_*_control_rep[1-5].pkl",
                        "**/experiment_EXP[0-9][0-9][0-9]_*_treatment_*.pkl",
                        "**/*session_[0-9][0-9]_*behavioral*.pkl"
                    ],
                    "file_count": 1000,
                    "nesting_depth": 6
                }
            }
            
            config = complexity_configs[complexity_level]
            
            # Create nested directory structure
            for depth in range(config["nesting_depth"]):
                for i in range(3):  # 3 subdirs per level
                    subdir_path = base_path
                    for d in range(depth + 1):
                        subdir_path = subdir_path / f"level_{d}" / f"subdir_{i}"
                    subdir_path.mkdir(parents=True, exist_ok=True)
            
            # Generate files matching different patterns
            generated_files = []
            pattern_matches = {pattern: [] for pattern in config["patterns"]}
            
            for i in range(config["file_count"]):
                # Choose random directory level
                depth = np.random.randint(0, config["nesting_depth"])
                subdir_idx = np.random.randint(0, 3)
                
                file_dir = base_path
                for d in range(depth + 1):
                    file_dir = file_dir / f"level_{d}" / f"subdir_{subdir_idx}"
                
                # Generate filename based on complexity
                if complexity_level == "simple":
                    filename = f"{'mouse' if np.random.random() < 0.5 else 'rat'}_{i:03d}_{'control' if np.random.random() < 0.3 else 'treatment'}_rep{np.random.randint(1, 6)}.pkl"
                elif complexity_level == "moderate":
                    filename = f"{'mouse' if np.random.random() < 0.7 else 'experiment'}_{i:03d}_{np.random.choice(['control', 'treatment'])}_{np.random.choice(['rep', 'session'])}{np.random.randint(1, 10)}.pkl"
                else:  # complex
                    animal_id = np.random.randint(1, 200)
                    exp_id = f"EXP{np.random.randint(1, 100):03d}"
                    session_id = np.random.randint(1, 20)
                    filename = f"{'mouse' if np.random.random() < 0.6 else 'experiment'}_{animal_id:03d}_{exp_id}_{np.random.choice(['control', 'treatment_a', 'treatment_b'])}_session_{session_id:02d}_{'behavioral' if np.random.random() < 0.4 else 'neural'}.pkl"
                
                file_path = file_dir / filename
                file_path.touch()
                generated_files.append(file_path)
                
                # Track which patterns this file matches
                relative_path = file_path.relative_to(base_path)
                for pattern in config["patterns"]:
                    # Simple pattern matching simulation
                    if self._matches_pattern(str(relative_path), pattern):
                        pattern_matches[pattern].append(file_path)
            
            return {
                "directory": base_path,
                "complexity_level": complexity_level,
                "total_files": len(generated_files),
                "nesting_depth": config["nesting_depth"],
                "patterns": config["patterns"],
                "pattern_matches": pattern_matches,
                "expected_match_counts": {pattern: len(matches) for pattern, matches in pattern_matches.items()}
            }
        
        def _matches_pattern(self, filename: str, pattern: str) -> bool:
            """Simple pattern matching for benchmark testing."""
            import fnmatch
            # Handle recursive patterns
            if pattern.startswith("**/"):
                pattern = pattern[3:]
                return fnmatch.fnmatch(filename.split("/")[-1], pattern) or any(
                    fnmatch.fnmatch(part, pattern) for part in filename.split("/")
                )
            return fnmatch.fnmatch(filename, pattern)
        
        def generate_benchmark_data_sizes(self) -> Dict[str, Tuple[int, int, float]]:
            """
            Generate data size configurations for systematic benchmark testing.
            
            Returns:
                Dict mapping size names to (rows, cols, estimated_mb) tuples
            """
            return {
                "tiny": (1_000, 10, 0.1),           # ~100KB - Unit test scale
                "small": (10_000, 25, 2.4),         # ~2.4MB - Quick validation  
                "medium": (100_000, 50, 48.0),      # ~48MB - Standard benchmark
                "large": (1_000_000, 50, 480.0),    # ~480MB - SLA validation scale
                "xlarge": (2_000_000, 100, 1900.0), # ~1.9GB - Memory profiling scale
                "stress": (5_000_000, 100, 4700.0)  # ~4.7GB - Stress testing (if memory allows)
            }
        
        def generate_synthetic_dataset(self, 
                                     rows: int, 
                                     cols: int,
                                     data_type: str = "neural",
                                     include_metadata: bool = True) -> Dict[str, Any]:
            """
            Generate synthetic dataset for benchmark testing with memory efficiency.
            
            Args:
                rows: Number of rows to generate
                cols: Number of columns to generate
                data_type: Type of data to generate ("neural", "behavioral", "large_dataset")
                include_metadata: Whether to include experimental metadata
                
            Returns:
                Dict containing synthetic dataset and generation metadata
            """
            # Estimate memory usage
            estimated_size_mb = (rows * cols * 4) / (1024 * 1024)  # float32 = 4 bytes
            
            print(f"Generating {data_type} dataset: {rows:,} x {cols} ({estimated_size_mb:.1f}MB)")
            
            # Generate data based on type
            if data_type == "neural":
                # Simulate neural recording with realistic signal patterns
                base_freq = 10  # Base frequency for signal
                data = np.zeros((rows, cols), dtype=self.data_type_defaults[data_type])
                
                # Generate in chunks to manage memory
                chunk_size = min(self.chunk_size, rows)
                for start_idx in range(0, rows, chunk_size):
                    end_idx = min(start_idx + chunk_size, rows)
                    chunk_rows = end_idx - start_idx
                    
                    # Generate base neural signal patterns
                    time_points = np.linspace(start_idx, end_idx, chunk_rows)
                    for col in range(cols):
                        # Each channel has slightly different frequency and amplitude
                        freq = base_freq + col * 0.5
                        amplitude = 1.0 + col * 0.1
                        signal = amplitude * np.sin(2 * np.pi * freq * time_points / 1000)
                        noise = np.random.normal(0, 0.1, chunk_rows)
                        data[start_idx:end_idx, col] = signal + noise
                
            elif data_type == "behavioral":
                # Simulate behavioral tracking data
                data = np.random.exponential(scale=2.0, size=(rows, cols)).astype(self.data_type_defaults[data_type])
                # Add some structure (position coordinates, velocities, etc.)
                if cols >= 4:
                    # First few columns are position data with continuity
                    for i in range(min(4, cols)):
                        data[:, i] = np.cumsum(np.random.normal(0, 0.1, rows))
                
            else:  # large_dataset or default
                # Memory-optimized large dataset generation
                data = np.random.normal(loc=0, scale=1, size=(rows, cols)).astype(self.data_type_defaults[data_type])
            
            # Generate column names
            if data_type == "neural":
                column_names = [f"channel_{i:03d}" for i in range(cols)]
            elif data_type == "behavioral":
                base_names = ["x_pos", "y_pos", "velocity", "orientation", "state"]
                column_names = [base_names[i % len(base_names)] + f"_{i//len(base_names)}" for i in range(cols)]
            else:
                column_names = [f"feature_{i:03d}" for i in range(cols)]
            
            # Create DataFrame if pandas available and dataset not too large
            dataframe = None
            if pd is not None and estimated_size_mb < 1000:  # Only create DataFrame for <1GB
                try:
                    dataframe = pd.DataFrame(data, columns=column_names[:len(column_names)])
                    if include_metadata:
                        # Add metadata columns
                        dataframe['timestamp'] = pd.date_range('2024-01-01', periods=rows, freq='100ms')
                        dataframe['trial_id'] = np.random.randint(1, max(rows//1000, 2), rows)
                        dataframe['condition'] = np.random.choice(['control', 'treatment'], rows)
                except MemoryError:
                    print(f"Warning: DataFrame creation failed due to memory constraints")
                    dataframe = None
            
            # Generate metadata
            metadata = {
                "data_type": data_type,
                "shape": (rows, cols),
                "estimated_size_mb": estimated_size_mb,
                "actual_size_bytes": data.nbytes,
                "dtype": str(data.dtype),
                "column_names": column_names,
                "generation_time": datetime.now().isoformat(),
                "memory_usage": {
                    "peak_during_generation": psutil.Process().memory_info().rss / (1024*1024),
                    "data_array_mb": data.nbytes / (1024*1024),
                    "dataframe_mb": dataframe.memory_usage(deep=True).sum() / (1024*1024) if dataframe is not None else 0
                }
            }
            
            if include_metadata and data_type in ["neural", "behavioral"]:
                metadata["experimental_metadata"] = {
                    "animal_ids": [f"animal_{i:03d}" for i in range(1, min(11, rows//100 + 1))],
                    "conditions": ["control", "treatment_a", "treatment_b"],
                    "date_range": "2024-01-01 to 2024-12-31",
                    "sampling_rate_hz": 1000 if data_type == "neural" else 30
                }
            
            return {
                "data_array": data,
                "dataframe": dataframe,
                "metadata": metadata,
                "column_names": column_names
            }
    
    return SyntheticDataGenerator(benchmark_config)


@pytest.fixture(scope="function")
def large_test_directory(tmp_path, synthetic_data_generator):
    """
    Function-scoped fixture providing large test directory for file discovery benchmarks.
    
    Creates realistic directory structure with 1000+ files to test F-002-RQ-001 SLA
    requirement of <5s discovery time for 10,000 files.
    """
    return synthetic_data_generator.generate_large_test_directory(
        base_path=tmp_path / "large_test_dir",
        file_count=1000,  # Reduced for CI performance, can be increased locally
        size_distribution="realistic"
    )


@pytest.fixture(scope="function") 
def pattern_test_directory(tmp_path, synthetic_data_generator):
    """
    Function-scoped fixture providing pattern matching test directory structure.
    
    Creates complex nested directory with various file patterns for testing
    discovery pattern matching performance and accuracy.
    """
    return synthetic_data_generator.generate_pattern_test_directory(
        base_path=tmp_path / "pattern_test_dir",
        complexity_level="moderate"
    )


@pytest.fixture(scope="function")
def benchmark_data_sizes(synthetic_data_generator):
    """
    Function-scoped fixture providing standardized data size configurations.
    
    Returns consistent data size mappings for systematic benchmark testing
    across all performance test categories.
    """
    return synthetic_data_generator.generate_benchmark_data_sizes()


# ============================================================================
# MEMORY PROFILING FIXTURES
# ============================================================================

@pytest.fixture(scope="function")
def memory_profiler_context():
    """
    Function-scoped fixture providing memory profiling context manager
    for large dataset memory analysis and leak detection.
    
    Features:
    - Integration with pytest-memory-profiler for line-by-line analysis
    - Memory leak detection for iterative loading scenarios
    - Large dataset memory efficiency validation (>500MB scenarios)
    - Cross-platform memory monitoring via psutil integration
    
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
            data_size_estimate: Estimated size of data being processed in bytes
            precision: Decimal precision for memory measurements
            enable_line_profiling: Whether to enable continuous monitoring
            monitor_interval: Monitoring interval in seconds
            
        Returns:
            Memory profiling context manager
        """
        return memory_profiling_context(
            data_size_estimate=data_size_estimate,
            precision=precision,
            enable_line_profiling=enable_line_profiling,
            monitor_interval=monitor_interval
        )
    
    return create_context


@pytest.fixture(scope="function")
def memory_leak_detector(benchmark_config):
    """
    Function-scoped fixture providing memory leak detection utilities
    for large dataset processing validation.
    
    Features:
    - Iterative memory leak detection for loading/unloading cycles
    - Memory growth threshold validation per configuration
    - Garbage collection impact analysis
    - Memory baseline establishment and comparison
    
    Returns:
        MemoryLeakDetector: Utility for detecting memory leaks in benchmark scenarios
    """
    class MemoryLeakDetector:
        def __init__(self, config: BenchmarkConfig):
            self.config = config
            self.memory_config = config.memory_profiling
            self.baseline_measurements = []
            self.process = psutil.Process()
        
        def establish_baseline(self, measurement_count: int = 3):
            """Establish memory baseline with multiple measurements."""
            self.baseline_measurements = []
            for _ in range(measurement_count):
                gc.collect()  # Force garbage collection
                time.sleep(0.1)  # Allow system to settle
                memory_mb = self.process.memory_info().rss / (1024 * 1024)
                self.baseline_measurements.append(memory_mb)
            
            baseline_avg = np.mean(self.baseline_measurements)
            baseline_std = np.std(self.baseline_measurements)
            
            print(f"Memory baseline established: {baseline_avg:.1f}MB ± {baseline_std:.1f}MB")
            return baseline_avg, baseline_std
        
        def detect_leak_in_iterations(self, 
                                    operation_func,
                                    iterations: int = None,
                                    *args, **kwargs) -> Dict[str, Any]:
            """
            Detect memory leaks through iterative operation execution.
            
            Args:
                operation_func: Function to execute repeatedly
                iterations: Number of iterations (uses config default if None)
                *args, **kwargs: Arguments to pass to operation_func
                
            Returns:
                Dict containing leak detection analysis
            """
            iterations = iterations or self.memory_config.leak_detection_iterations
            
            # Establish baseline if not already done
            if not self.baseline_measurements:
                self.establish_baseline()
            
            baseline_avg = np.mean(self.baseline_measurements)
            memory_timeline = []
            
            print(f"Starting memory leak detection over {iterations} iterations...")
            
            for i in range(iterations):
                # Execute operation
                start_time = time.time()
                start_memory = self.process.memory_info().rss / (1024 * 1024)
                
                try:
                    result = operation_func(*args, **kwargs)
                except Exception as e:
                    print(f"Operation failed at iteration {i}: {e}")
                    break
                
                end_memory = self.process.memory_info().rss / (1024 * 1024)
                end_time = time.time()
                
                # Force garbage collection between iterations if configured
                if self.memory_config.gc_collection_between_iterations:
                    gc.collect()
                    time.sleep(0.05)  # Brief pause for GC
                    post_gc_memory = self.process.memory_info().rss / (1024 * 1024)
                else:
                    post_gc_memory = end_memory
                
                memory_timeline.append({
                    "iteration": i,
                    "start_memory_mb": start_memory,
                    "end_memory_mb": end_memory,
                    "post_gc_memory_mb": post_gc_memory,
                    "duration_seconds": end_time - start_time,
                    "memory_growth_mb": post_gc_memory - baseline_avg
                })
                
                # Early detection of significant leaks
                if post_gc_memory - baseline_avg > self.memory_config.memory_leak_detection_threshold_mb * 2:
                    print(f"WARNING: Significant memory growth detected at iteration {i}")
            
            # Analyze memory growth pattern
            final_memory = memory_timeline[-1]["post_gc_memory_mb"] if memory_timeline else baseline_avg
            total_growth = final_memory - baseline_avg
            
            # Calculate leak detection results
            leak_detected = total_growth > self.memory_config.memory_leak_detection_threshold_mb
            
            # Trend analysis
            growth_values = [entry["memory_growth_mb"] for entry in memory_timeline]
            if len(growth_values) > 2:
                # Linear regression to detect growth trend
                x = np.arange(len(growth_values))
                coefficients = np.polyfit(x, growth_values, 1)
                growth_rate_mb_per_iteration = coefficients[0]
                trend_significant = abs(growth_rate_mb_per_iteration) > 0.1  # >0.1MB per iteration
            else:
                growth_rate_mb_per_iteration = 0
                trend_significant = False
            
            leak_analysis = {
                "leak_detected": leak_detected,
                "total_memory_growth_mb": total_growth,
                "leak_threshold_mb": self.memory_config.memory_leak_detection_threshold_mb,
                "baseline_memory_mb": baseline_avg,
                "final_memory_mb": final_memory,
                "iterations_completed": len(memory_timeline),
                "growth_rate_mb_per_iteration": growth_rate_mb_per_iteration,
                "trend_significant": trend_significant,
                "memory_timeline": memory_timeline,
                "analysis_summary": {
                    "status": "LEAK_DETECTED" if leak_detected else "NO_LEAK",
                    "confidence": min(abs(total_growth) / self.memory_config.memory_leak_detection_threshold_mb, 2.0),
                    "recommendations": []
                }
            }
            
            # Generate recommendations
            if leak_detected:
                leak_analysis["analysis_summary"]["recommendations"].append(
                    "Memory leak detected - review resource cleanup in tested operation"
                )
                if growth_rate_mb_per_iteration > 1.0:
                    leak_analysis["analysis_summary"]["recommendations"].append(
                        "High memory growth rate - investigate object retention and garbage collection"
                    )
            else:
                leak_analysis["analysis_summary"]["recommendations"].append(
                    "No significant memory leak detected - memory usage within acceptable limits"
                )
            
            print(f"Memory leak detection complete: {leak_analysis['analysis_summary']['status']}")
            print(f"Total growth: {total_growth:.1f}MB, Rate: {growth_rate_mb_per_iteration:.3f}MB/iteration")
            
            return leak_analysis
    
    return MemoryLeakDetector(benchmark_config)


# ============================================================================
# MOCK PROVIDER FIXTURES
# ============================================================================

@pytest.fixture(scope="function")
def mock_filesystem_provider(mocker):
    """
    Function-scoped fixture providing centralized mock filesystem implementations
    for benchmark testing with cross-platform compatibility.
    
    Features:
    - Cross-platform path mocking with realistic filesystem behavior
    - File existence and permission simulation for error scenario testing
    - Directory traversal mocking for discovery performance testing
    - Configurable file sizes and metadata for realistic benchmarking
    
    Returns:
        MockFilesystemProvider: Centralized filesystem mocking utility
    """
    class MockFilesystemProvider:
        def __init__(self, mocker):
            self.mocker = mocker
            self.mock_files = {}
            self.mock_directories = {}
            self.access_counts = {}  # Track file access for performance analysis
            
        def add_mock_file(self, 
                         path: Union[str, Path],
                         size: int = 1024,
                         content: Any = None,
                         mtime: Optional[datetime] = None,
                         accessible: bool = True) -> Path:
            """Add mock file with specified properties."""
            path = Path(path)
            self.mock_files[str(path)] = {
                "size": size,
                "content": content,
                "mtime": mtime or datetime.now(),
                "accessible": accessible,
                "access_count": 0
            }
            self.access_counts[str(path)] = 0
            
            # Ensure parent directories exist
            parent = path.parent
            while parent != Path('.') and str(parent) != str(parent.parent):
                if str(parent) not in self.mock_directories:
                    self.add_mock_directory(parent)
                parent = parent.parent
            
            return path
        
        def add_mock_directory(self, 
                             path: Union[str, Path],
                             accessible: bool = True) -> Path:
            """Add mock directory with specified properties."""
            path = Path(path)
            self.mock_directories[str(path)] = {
                "accessible": accessible,
                "access_count": 0
            }
            return path
        
        def create_realistic_file_structure(self, 
                                          base_path: Path,
                                          file_count: int = 100) -> Dict[str, Any]:
            """Create realistic file structure for discovery benchmarks."""
            base_path_str = str(base_path)
            self.add_mock_directory(base_path)
            
            # Create subdirectories
            subdirs = ["raw_data", "processed", "backup", "experiments", "configs"]
            for subdir in subdirs:
                subdir_path = base_path / subdir
                self.add_mock_directory(subdir_path)
            
            # Create files with realistic patterns
            created_files = []
            for i in range(file_count):
                subdir = np.random.choice(subdirs)
                filename = f"experiment_{i:03d}_data.pkl"
                file_path = base_path / subdir / filename
                
                # Realistic file size distribution
                if np.random.random() < 0.7:
                    size = np.random.randint(1024, 100*1024)  # Small files
                else:
                    size = np.random.randint(1024*1024, 50*1024*1024)  # Larger files
                
                self.add_mock_file(file_path, size=size)
                created_files.append(file_path)
            
            return {
                "base_path": base_path,
                "subdirectories": subdirs,
                "files": created_files,
                "total_files": len(created_files)
            }
        
        def mock_path_exists(self, path):
            """Mock pathlib.Path.exists() with access tracking."""
            path_str = str(path)
            self.access_counts[path_str] = self.access_counts.get(path_str, 0) + 1
            return (path_str in self.mock_files and self.mock_files[path_str]["accessible"]) or \
                   (path_str in self.mock_directories and self.mock_directories[path_str]["accessible"])
        
        def mock_path_is_file(self, path):
            """Mock pathlib.Path.is_file() with access tracking."""
            path_str = str(path)
            self.access_counts[path_str] = self.access_counts.get(path_str, 0) + 1
            return path_str in self.mock_files and self.mock_files[path_str]["accessible"]
        
        def mock_path_is_dir(self, path):
            """Mock pathlib.Path.is_dir() with access tracking."""
            path_str = str(path)
            self.access_counts[path_str] = self.access_counts.get(path_str, 0) + 1
            return path_str in self.mock_directories and self.mock_directories[path_str]["accessible"]
        
        def mock_path_iterdir(self, path):
            """Mock pathlib.Path.iterdir() for directory traversal."""
            path_str = str(path)
            self.access_counts[path_str] = self.access_counts.get(path_str, 0) + 1
            
            if path_str not in self.mock_directories:
                raise FileNotFoundError(f"Directory not found: {path}")
            
            # Return all files and subdirectories in this path
            children = []
            for item_path in list(self.mock_files.keys()) + list(self.mock_directories.keys()):
                item = Path(item_path)
                if item.parent == path:
                    children.append(item)
            
            return children
        
        def mock_path_stat(self, path):
            """Mock pathlib.Path.stat() with file metadata."""
            path_str = str(path)
            self.access_counts[path_str] = self.access_counts.get(path_str, 0) + 1
            
            if path_str in self.mock_files:
                file_info = self.mock_files[path_str]
                file_info["access_count"] += 1
                
                mock_stat = MagicMock()
                mock_stat.st_size = file_info["size"]
                mock_stat.st_mtime = file_info["mtime"].timestamp()
                mock_stat.st_mode = 0o100644  # Regular file permissions
                return mock_stat
            
            raise FileNotFoundError(f"File not found: {path}")
        
        def activate_all_mocks(self):
            """Activate all filesystem mocks."""
            self.mocker.patch('pathlib.Path.exists', side_effect=self.mock_path_exists)
            self.mocker.patch('pathlib.Path.is_file', side_effect=self.mock_path_is_file)
            self.mocker.patch('pathlib.Path.is_dir', side_effect=self.mock_path_is_dir)
            self.mocker.patch('pathlib.Path.iterdir', side_effect=self.mock_path_iterdir)
            self.mocker.patch('pathlib.Path.stat', side_effect=self.mock_path_stat)
        
        def get_access_statistics(self) -> Dict[str, Any]:
            """Get filesystem access statistics for performance analysis."""
            total_accesses = sum(self.access_counts.values())
            most_accessed = max(self.access_counts.items(), key=lambda x: x[1]) if self.access_counts else ("none", 0)
            
            return {
                "total_filesystem_accesses": total_accesses,
                "unique_paths_accessed": len(self.access_counts),
                "most_accessed_path": most_accessed[0],
                "max_access_count": most_accessed[1],
                "access_distribution": dict(self.access_counts)
            }
    
    return MockFilesystemProvider(mocker)


@pytest.fixture(scope="function")
def mock_configuration_provider(mocker):
    """
    Function-scoped fixture providing mock configuration loading for benchmark testing.
    
    Features:
    - YAML configuration loading simulation with realistic data structures
    - Configuration validation error simulation for error path testing
    - Performance-oriented configuration generation for various test scenarios
    - Cross-platform configuration compatibility testing
    
    Returns:
        MockConfigurationProvider: Centralized configuration mocking utility
    """
    class MockConfigurationProvider:
        def __init__(self, mocker):
            self.mocker = mocker
            self.mock_configs = {}
            self.loading_delays = {}  # Simulate loading time for performance tests
            
        def add_mock_config(self, 
                           file_path: Union[str, Path],
                           config_data: Dict[str, Any],
                           loading_delay: float = 0.0,
                           should_raise: Optional[Exception] = None):
            """Add mock configuration with specified properties."""
            path_str = str(file_path)
            self.mock_configs[path_str] = {
                "data": config_data,
                "loading_delay": loading_delay,
                "should_raise": should_raise,
                "access_count": 0
            }
        
        def create_benchmark_configs(self, base_path: Path) -> Dict[str, Path]:
            """Create various configuration scenarios for benchmark testing."""
            configs = {}
            
            # Small configuration - fast loading
            small_config = {
                "project": {"name": "small_test", "mandatory_experiment_strings": []},
                "datasets": {"test_data": {"dates_vials": {"20240101": ["mouse_001"]}}}
            }
            small_path = base_path / "small_config.yaml"
            self.add_mock_config(small_path, small_config, loading_delay=0.001)
            configs["small"] = small_path
            
            # Medium configuration - moderate complexity
            medium_config = {
                "project": {
                    "name": "medium_test",
                    "mandatory_experiment_strings": ["exp_", "mouse_"],
                    "ignore_substrings": ["backup", "temp"]
                },
                "datasets": {
                    "neural_data": {
                        "dates_vials": {
                            f"2024{month:02d}{day:02d}": [f"mouse_{i:03d}" for i in range(1, 20)]
                            for month in range(1, 7) for day in [1, 15]
                        }
                    },
                    "behavioral_data": {
                        "dates_vials": {
                            f"2024{month:02d}{day:02d}": [f"rat_{i:03d}" for i in range(1, 10)]
                            for month in range(1, 4) for day in [5, 20]
                        }
                    }
                }
            }
            medium_path = base_path / "medium_config.yaml"
            self.add_mock_config(medium_path, medium_config, loading_delay=0.01)
            configs["medium"] = medium_path
            
            # Large configuration - complex structure
            large_config = {
                "project": {
                    "name": "large_test",
                    "mandatory_experiment_strings": ["experiment_", "session_", "trial_"],
                    "ignore_substrings": ["backup", "temp", "old", "test"]
                },
                "datasets": {}
            }
            
            # Generate large dataset configuration
            for dataset_idx in range(10):
                dataset_name = f"dataset_{dataset_idx:02d}"
                large_config["datasets"][dataset_name] = {
                    "dates_vials": {
                        f"2024{month:02d}{day:02d}": [
                            f"animal_{animal_type}_{i:03d}" 
                            for animal_type in ["mouse", "rat"] 
                            for i in range(1, 50)
                        ]
                        for month in range(1, 13) 
                        for day in range(1, 32, 7)
                    }
                }
            
            large_path = base_path / "large_config.yaml"
            self.add_mock_config(large_path, large_config, loading_delay=0.05)
            configs["large"] = large_path
            
            # Error configuration - should raise exception
            error_path = base_path / "error_config.yaml"
            self.add_mock_config(
                error_path, 
                {},
                should_raise=yaml.YAMLError("Simulated YAML parsing error")
            )
            configs["error"] = error_path
            
            return configs
        
        def mock_yaml_safe_load(self, file_obj):
            """Mock yaml.safe_load with delay simulation and error handling."""
            file_path = getattr(file_obj, 'name', str(file_obj))
            
            if file_path in self.mock_configs:
                config_info = self.mock_configs[file_path]
                config_info["access_count"] += 1
                
                # Simulate loading delay for performance testing
                if config_info["loading_delay"] > 0:
                    time.sleep(config_info["loading_delay"])
                
                # Raise exception if configured
                if config_info["should_raise"]:
                    raise config_info["should_raise"]
                
                return config_info["data"]
            
            raise FileNotFoundError(f"Mock configuration not found: {file_path}")
        
        def activate_yaml_mocks(self):
            """Activate YAML loading mocks."""
            try:
                import yaml
                self.mocker.patch('yaml.safe_load', side_effect=self.mock_yaml_safe_load)
            except ImportError:
                # yaml not available, create a mock module
                mock_yaml = MagicMock()
                mock_yaml.safe_load = MagicMock(side_effect=self.mock_yaml_safe_load)
                mock_yaml.YAMLError = Exception
                self.mocker.patch.dict('sys.modules', {'yaml': mock_yaml})
        
        def get_config_access_statistics(self) -> Dict[str, Any]:
            """Get configuration access statistics for performance analysis."""
            total_accesses = sum(
                config["access_count"] for config in self.mock_configs.values()
            )
            
            access_summary = {
                path: config["access_count"] 
                for path, config in self.mock_configs.items()
            }
            
            return {
                "total_config_accesses": total_accesses,
                "configs_accessed": len([c for c in self.mock_configs.values() if c["access_count"] > 0]),
                "access_summary": access_summary
            }
    
    return MockConfigurationProvider(mocker)


# ============================================================================
# PERFORMANCE ANALYSIS FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def statistical_analysis_engine(benchmark_execution_environment):
    """
    Session-scoped fixture providing statistical analysis engine for benchmark validation.
    
    Features:
    - Confidence interval calculation for performance measurements
    - Regression detection with historical baseline comparison
    - Outlier detection and measurement cleaning for reliable analysis
    - Statistical significance testing for performance changes
    
    Returns:
        StatisticalAnalysisEngine: Comprehensive statistical analysis utility
    """
    return benchmark_execution_environment["statistical_engine"]


@pytest.fixture(scope="session")
def performance_regression_detector(benchmark_execution_environment):
    """
    Session-scoped fixture providing performance regression detection utilities.
    
    Features:
    - Automated regression detection with statistical significance testing
    - Performance baseline maintenance with rolling historical data
    - Confidence-based regression alerting for CI/CD integration
    - Trend analysis for gradual performance degradation detection
    
    Returns:
        RegressionDetector: Performance regression detection utility
    """
    return benchmark_execution_environment["regression_detector"]


@pytest.fixture(scope="function")
def performance_baseline_manager(benchmark_execution_environment):
    """
    Function-scoped fixture providing performance baseline management.
    
    Features:
    - Performance baseline establishment for new benchmark tests
    - Historical performance data management with retention policies
    - Cross-platform baseline normalization for consistent comparison
    - SLA compliance validation against established baselines
    
    Returns:
        PerformanceBaselineManager: Baseline management utility
    """
    class PerformanceBaselineManager:
        def __init__(self, execution_env):
            self.execution_env = execution_env
            self.statistical_engine = execution_env["statistical_engine"]
            self.regression_detector = execution_env["regression_detector"]
            self.baselines = {}
            self.sla_targets = execution_env["baseline_performance_targets"]
            
        def establish_baseline(self, 
                             test_name: str, 
                             measurements: List[float],
                             sla_category: Optional[str] = None) -> Dict[str, Any]:
            """
            Establish performance baseline for a test.
            
            Args:
                test_name: Name of the test for baseline tracking
                measurements: Initial performance measurements
                sla_category: SLA category for threshold validation
                
            Returns:
                Dict containing baseline establishment results
            """
            # Calculate statistical baseline
            confidence_interval = self.statistical_engine.calculate_confidence_interval(measurements)
            
            # Clean outliers for reliable baseline
            cleaned_measurements, outlier_indices = self.statistical_engine.detect_outliers(measurements)
            
            # Determine SLA threshold if category provided
            sla_threshold = None
            if sla_category and sla_category in self.sla_targets:
                sla_threshold = self.sla_targets[sla_category]
            
            # Store baseline
            baseline_data = {
                "test_name": test_name,
                "establishment_date": datetime.now().isoformat(),
                "measurements": cleaned_measurements,
                "confidence_interval": confidence_interval.to_dict(),
                "outliers_removed": len(outlier_indices),
                "sla_category": sla_category,
                "sla_threshold": sla_threshold,
                "baseline_mean": confidence_interval.mean,
                "baseline_std": confidence_interval.std_error * np.sqrt(confidence_interval.sample_size)
            }
            
            self.baselines[test_name] = baseline_data
            
            # Update regression detector baseline
            self.statistical_engine.update_baseline(test_name, cleaned_measurements)
            
            print(f"Baseline established for {test_name}:")
            print(f"  Mean: {confidence_interval.mean:.4f}s")
            print(f"  95% CI: [{confidence_interval.lower_bound:.4f}, {confidence_interval.upper_bound:.4f}]")
            if sla_threshold:
                sla_compliance = confidence_interval.mean <= sla_threshold
                print(f"  SLA Compliance: {'✓' if sla_compliance else '✗'} ({confidence_interval.mean:.4f} <= {sla_threshold:.4f})")
            
            return baseline_data
        
        def validate_against_baseline(self, 
                                    test_name: str, 
                                    new_measurements: List[float]) -> Dict[str, Any]:
            """
            Validate new measurements against established baseline.
            
            Args:
                test_name: Name of the test to validate
                new_measurements: New performance measurements
                
            Returns:
                Dict containing validation results and regression analysis
            """
            if test_name not in self.baselines:
                return {
                    "validation_status": "no_baseline",
                    "message": f"No baseline established for {test_name}"
                }
            
            baseline = self.baselines[test_name]
            
            # Perform regression detection
            regression_result = self.regression_detector.detect_regression(
                test_name=test_name,
                current_measurements=new_measurements,
                baseline_measurements=baseline["measurements"]
            )
            
            # Calculate current statistics
            current_ci = self.statistical_engine.calculate_confidence_interval(new_measurements)
            
            # SLA validation if applicable
            sla_validation = None
            if baseline["sla_threshold"]:
                sla_compliant = current_ci.mean <= baseline["sla_threshold"]
                sla_validation = {
                    "compliant": sla_compliant,
                    "threshold": baseline["sla_threshold"],
                    "measured_value": current_ci.mean,
                    "margin": (baseline["sla_threshold"] - current_ci.mean) / baseline["sla_threshold"] * 100
                }
            
            validation_result = {
                "validation_status": "completed",
                "test_name": test_name,
                "baseline_comparison": {
                    "baseline_mean": baseline["baseline_mean"],
                    "current_mean": current_ci.mean,
                    "percent_change": ((current_ci.mean - baseline["baseline_mean"]) / baseline["baseline_mean"]) * 100,
                    "within_baseline_ci": baseline["confidence_interval"]["lower_bound"] <= current_ci.mean <= baseline["confidence_interval"]["upper_bound"]
                },
                "regression_analysis": regression_result,
                "sla_validation": sla_validation,
                "current_statistics": current_ci.to_dict()
            }
            
            print(f"Baseline validation for {test_name}:")
            print(f"  Current: {current_ci.mean:.4f}s")
            print(f"  Baseline: {baseline['baseline_mean']:.4f}s")
            print(f"  Change: {validation_result['baseline_comparison']['percent_change']:+.1f}%")
            if regression_result['regression_analysis']['regression_detected']:
                print(f"  ⚠️  Regression detected with {regression_result['regression_analysis']['confidence']:.2f} confidence")
            
            return validation_result
        
        def get_all_baselines(self) -> Dict[str, Any]:
            """Get all established baselines."""
            return self.baselines.copy()
    
    return PerformanceBaselineManager(benchmark_execution_environment)


# ============================================================================
# ARTIFACT GENERATION FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def benchmark_artifact_generator(benchmark_execution_environment):
    """
    Session-scoped fixture providing performance artifact generation for CI/CD integration.
    
    Features:
    - JSON/CSV performance report generation with comprehensive statistics
    - GitHub Actions artifact management with 90-day retention compliance
    - Performance trend analysis and historical comparison support
    - Cross-platform artifact compatibility and validation
    
    Returns:
        PerformanceArtifactGenerator: Comprehensive artifact generation utility
    """
    return benchmark_execution_environment["artifact_generator"]


@pytest.fixture(scope="function")
def ci_integration_manager(benchmark_execution_environment):
    """
    Function-scoped fixture providing CI/CD integration management.
    
    Features:
    - GitHub Actions workflow integration and environment detection
    - Performance alerting configuration based on regression thresholds
    - Artifact collection and organization for benchmark results
    - Cross-platform CI/CD validation and normalization
    
    Returns:
        CICDIntegrationManager: CI/CD integration and artifact management utility
    """
    return benchmark_execution_environment["cicd_manager"]


# ============================================================================
# ENVIRONMENT AND CROSS-PLATFORM FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def environment_analyzer(benchmark_execution_environment):
    """
    Session-scoped fixture providing environment analysis and normalization.
    
    Features:
    - Cross-platform environment characteristic detection
    - CPU and memory performance normalization factors
    - CI vs local environment differentiation with appropriate adjustments
    - Hardware abstraction for consistent benchmark results
    
    Returns:
        EnvironmentAnalyzer: Environment analysis and normalization utility
    """
    return benchmark_execution_environment["environment_analyzer"]


@pytest.fixture(scope="function") 
def cross_platform_validator(benchmark_execution_environment):
    """
    Function-scoped fixture providing cross-platform performance validation.
    
    Features:
    - Performance consistency validation across Ubuntu, Windows, macOS
    - Platform-specific normalization factor application
    - Cross-platform variance analysis and reporting
    - Hardware compatibility validation for benchmark execution
    
    Returns:
        CrossPlatformValidator: Cross-platform performance validation utility
    """
    return benchmark_execution_environment["cross_platform_validator"]


@pytest.fixture(scope="function")
def platform_performance_normalizer(environment_analyzer):
    """
    Function-scoped fixture providing platform performance normalization utilities.
    
    Features:
    - Performance measurement normalization based on detected environment
    - Cross-platform consistency enforcement for benchmark comparison
    - CI environment adjustment with virtualization overhead compensation
    - Hardware performance scaling for diverse computational environments
    
    Returns:
        Function for normalizing performance measurements
    """
    def normalize_measurement(measurement: float, 
                            baseline_environment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Normalize performance measurement for current environment.
        
        Args:
            measurement: Raw performance measurement in seconds
            baseline_environment: Optional baseline environment for comparison
            
        Returns:
            Dict containing normalized measurement and normalization metadata
        """
        # Calculate normalization factors
        normalization_factors = environment_analyzer.calculate_normalization_factors(baseline_environment)
        
        # Apply normalization
        normalized_measurement = environment_analyzer.normalize_performance_measurement(
            measurement, normalization_factors
        )
        
        return {
            "raw_measurement": measurement,
            "normalized_measurement": normalized_measurement,
            "normalization_factors": normalization_factors,
            "improvement_factor": measurement / normalized_measurement if normalized_measurement > 0 else 1.0,
            "environment_metadata": environment_analyzer.analyze_current_environment()
        }
    
    return normalize_measurement


# ============================================================================
# CONDITIONAL SKIP FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def platform_skip_conditions():
    """
    Session-scoped fixture providing platform-specific skip conditions for benchmarks.
    
    Features:
    - Platform-specific benchmark capability detection
    - Memory constraint validation for large dataset scenarios
    - CI environment capability assessment
    - Cross-platform compatibility validation
    
    Returns:
        Dict containing skip condition functions for various scenarios
    """
    current_platform = platform.system()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    is_ci = os.getenv("CI", "").lower() == "true"
    
    def skip_if_insufficient_memory(required_gb: float):
        """Skip test if insufficient memory available."""
        available_gb = psutil.virtual_memory().available / (1024**3)
        return pytest.mark.skipif(
            available_gb < required_gb,
            reason=f"Insufficient memory: {available_gb:.1f}GB available, {required_gb:.1f}GB required"
        )
    
    def skip_if_platform_not_supported(supported_platforms: List[str]):
        """Skip test if current platform not in supported list."""
        return pytest.mark.skipif(
            current_platform not in supported_platforms,
            reason=f"Platform {current_platform} not in supported platforms: {supported_platforms}"
        )
    
    def skip_if_ci_environment():
        """Skip test if running in CI environment."""
        return pytest.mark.skipif(
            is_ci,
            reason="Test disabled in CI environment"
        )
    
    def skip_if_slow_filesystem():
        """Skip test if filesystem appears slow (for CI environments)."""
        # Simple filesystem speed test
        test_file = Path(tempfile.gettempdir()) / "fs_speed_test.tmp"
        try:
            start_time = time.time()
            with open(test_file, 'wb') as f:
                f.write(b'0' * (1024 * 1024))  # Write 1MB
            write_time = time.time() - start_time
            test_file.unlink()
            
            slow_filesystem = write_time > 0.5  # >0.5s for 1MB indicates slow FS
        except Exception:
            slow_filesystem = False
        
        return pytest.mark.skipif(
            slow_filesystem,
            reason="Slow filesystem detected - skipping I/O intensive test"
        )
    
    return {
        "skip_if_insufficient_memory": skip_if_insufficient_memory,
        "skip_if_platform_not_supported": skip_if_platform_not_supported,
        "skip_if_ci_environment": skip_if_ci_environment,
        "skip_if_slow_filesystem": skip_if_slow_filesystem,
        "platform_info": {
            "current_platform": current_platform,
            "memory_gb": memory_gb,
            "is_ci": is_ci
        }
    }


# ============================================================================
# PYTEST-BENCHMARK INTEGRATION FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def benchmark_plugin_config(benchmark_config):
    """
    Session-scoped fixture providing pytest-benchmark plugin configuration.
    
    Features:
    - Statistical analysis integration with confidence intervals
    - Regression detection configuration with historical comparison
    - Memory profiling integration for comprehensive performance analysis
    - Cross-platform benchmark normalization and artifact generation
    
    Returns:
        Dict containing complete pytest-benchmark configuration
    """
    pytest_config = benchmark_config.pytest_benchmark.get_pytest_benchmark_config()
    
    # Enhance with statistical analysis settings
    pytest_config.update({
        "benchmark_group_by": "group",
        "benchmark_sort": "mean",
        "benchmark_columns": ["min", "max", "mean", "stddev", "median", "ops", "rounds"],
        "benchmark_verbose": True,
        "benchmark_warmup": True,
        "benchmark_warmup_iterations": benchmark_config.statistical_analysis.warmup_iterations,
        "benchmark_min_rounds": benchmark_config.statistical_analysis.min_iterations,
        "benchmark_max_time": 30.0,  # Maximum 30 seconds per benchmark
        "benchmark_calibration_precision": benchmark_config.statistical_analysis.calibration_precision,
        "benchmark_disable_gc": False,  # Keep GC enabled for realistic performance
        "benchmark_timer": "time.perf_counter"
    })
    
    return pytest_config


@pytest.fixture(scope="function")
def benchmark_validator(benchmark_config, statistical_analysis_engine):
    """
    Function-scoped fixture providing benchmark result validation utilities.
    
    Features:
    - SLA compliance validation against technical specification thresholds
    - Statistical significance testing for benchmark reliability
    - Performance regression detection with confidence analysis
    - Memory efficiency validation for large dataset scenarios
    
    Returns:
        BenchmarkValidator: Comprehensive benchmark validation utility
    """
    class BenchmarkValidator:
        def __init__(self, config: BenchmarkConfig, statistical_engine: StatisticalAnalysisEngine):
            self.config = config
            self.statistical_engine = statistical_engine
            self.sla_thresholds = {
                "data_loading": config.sla.DATA_LOADING_TIME_PER_100MB_SECONDS,
                "transformation": config.sla.DATAFRAME_TRANSFORM_TIME_PER_1M_ROWS_MS / 1000.0,  # Convert to seconds
                "discovery": config.sla.FILE_DISCOVERY_TIME_FOR_10K_FILES_SECONDS,
                "config_loading": config.sla.CONFIG_LOADING_TIME_MS / 1000.0,  # Convert to seconds
            }
        
        def validate_sla_compliance(self, 
                                  test_category: str,
                                  measurement: float,
                                  scale_factor: float = 1.0) -> Dict[str, Any]:
            """
            Validate measurement against SLA thresholds.
            
            Args:
                test_category: Category of test for SLA lookup
                measurement: Measured performance value in seconds
                scale_factor: Scale factor for threshold adjustment
                
            Returns:
                Dict containing SLA compliance validation results
            """
            if test_category not in self.sla_thresholds:
                return {
                    "compliant": None,
                    "message": f"Unknown test category: {test_category}",
                    "threshold": None,
                    "measurement": measurement
                }
            
            threshold = self.sla_thresholds[test_category] * scale_factor
            compliant = measurement <= threshold
            margin_percent = ((threshold - measurement) / threshold) * 100
            
            return {
                "compliant": compliant,
                "threshold": threshold,
                "measurement": measurement,
                "scale_factor": scale_factor,
                "margin_percent": margin_percent,
                "message": f"{'✓ PASS' if compliant else '✗ FAIL'}: {measurement:.4f}s {'<=' if compliant else '>'} {threshold:.4f}s ({margin_percent:+.1f}% margin)"
            }
        
        def validate_statistical_reliability(self, measurements: List[float]) -> Dict[str, Any]:
            """
            Validate statistical reliability of benchmark measurements.
            
            Args:
                measurements: List of performance measurements
                
            Returns:
                Dict containing statistical reliability analysis
            """
            if len(measurements) < 3:
                return {
                    "reliable": False,
                    "message": "Insufficient measurements for statistical analysis",
                    "measurement_count": len(measurements)
                }
            
            # Calculate confidence interval
            confidence_interval = self.statistical_engine.calculate_confidence_interval(measurements)
            
            # Calculate coefficient of variation
            mean_val = np.mean(measurements)
            std_val = np.std(measurements, ddof=1)
            cv = (std_val / mean_val) * 100 if mean_val > 0 else float('inf')
            
            # Detect outliers
            cleaned_measurements, outlier_indices = self.statistical_engine.detect_outliers(measurements)
            outlier_percentage = (len(outlier_indices) / len(measurements)) * 100
            
            # Determine reliability
            reliable = (
                cv <= self.config.statistical_analysis.acceptable_variance_ratio * 100 and
                outlier_percentage <= 10.0 and  # <10% outliers
                len(cleaned_measurements) >= 3
            )
            
            return {
                "reliable": reliable,
                "coefficient_of_variation": cv,
                "confidence_interval": confidence_interval.to_dict(),
                "outlier_percentage": outlier_percentage,
                "outliers_detected": len(outlier_indices),
                "cleaned_measurement_count": len(cleaned_measurements),
                "reliability_factors": {
                    "cv_acceptable": cv <= self.config.statistical_analysis.acceptable_variance_ratio * 100,
                    "outliers_acceptable": outlier_percentage <= 10.0,
                    "sample_size_adequate": len(cleaned_measurements) >= 3
                },
                "message": f"Statistical reliability: {'✓ RELIABLE' if reliable else '✗ UNRELIABLE'} (CV: {cv:.1f}%, Outliers: {outlier_percentage:.1f}%)"
            }
        
        def validate_memory_efficiency(self, 
                                     memory_stats: Dict[str, Any],
                                     expected_data_size_mb: float) -> Dict[str, Any]:
            """
            Validate memory efficiency for large dataset scenarios.
            
            Args:
                memory_stats: Memory usage statistics from MemoryProfiler
                expected_data_size_mb: Expected data size in MB
                
            Returns:
                Dict containing memory efficiency validation results
            """
            if "memory_multiplier" not in memory_stats:
                return {
                    "efficient": None,
                    "message": "Memory statistics not available for efficiency validation"
                }
            
            memory_multiplier = memory_stats["memory_multiplier"]
            efficiency_threshold = self.config.sla.TRANSFORMATION_MEMORY_MULTIPLIER
            
            efficient = memory_multiplier <= efficiency_threshold
            
            return {
                "efficient": efficient,
                "memory_multiplier": memory_multiplier,
                "efficiency_threshold": efficiency_threshold,
                "data_size_mb": expected_data_size_mb,
                "peak_memory_mb": memory_stats.get("peak_memory_mb", 0),
                "memory_overhead_mb": memory_stats.get("memory_overhead_mb", 0),
                "leak_detected": memory_stats.get("leak_analysis", {}).get("leak_detected", False),
                "message": f"Memory efficiency: {'✓ EFFICIENT' if efficient else '✗ INEFFICIENT'} ({memory_multiplier:.1f}x <= {efficiency_threshold:.1f}x threshold)"
            }
        
        def comprehensive_validation(self, 
                                   test_category: str,
                                   measurements: List[float],
                                   memory_stats: Optional[Dict[str, Any]] = None,
                                   expected_data_size_mb: float = 0) -> Dict[str, Any]:
            """
            Perform comprehensive validation of benchmark results.
            
            Args:
                test_category: Category of test for SLA validation
                measurements: Performance measurements
                memory_stats: Optional memory usage statistics
                expected_data_size_mb: Expected data size for memory validation
                
            Returns:
                Dict containing comprehensive validation results
            """
            # SLA compliance validation
            mean_measurement = np.mean(measurements)
            sla_validation = self.validate_sla_compliance(test_category, mean_measurement)
            
            # Statistical reliability validation
            reliability_validation = self.validate_statistical_reliability(measurements)
            
            # Memory efficiency validation (if stats available)
            memory_validation = None
            if memory_stats and expected_data_size_mb > 0:
                memory_validation = self.validate_memory_efficiency(memory_stats, expected_data_size_mb)
            
            # Overall validation status
            overall_valid = (
                sla_validation.get("compliant", False) and
                reliability_validation.get("reliable", False) and
                (memory_validation is None or memory_validation.get("efficient", True))
            )
            
            validation_result = {
                "overall_valid": overall_valid,
                "test_category": test_category,
                "measurement_count": len(measurements),
                "mean_measurement": mean_measurement,
                "sla_validation": sla_validation,
                "reliability_validation": reliability_validation,
                "memory_validation": memory_validation,
                "validation_timestamp": datetime.now().isoformat(),
                "summary": {
                    "sla_compliant": sla_validation.get("compliant", False),
                    "statistically_reliable": reliability_validation.get("reliable", False),
                    "memory_efficient": memory_validation.get("efficient", True) if memory_validation else True
                }
            }
            
            print(f"\n=== Benchmark Validation Summary ===")
            print(f"Test Category: {test_category}")
            print(f"Mean Performance: {mean_measurement:.4f}s")
            print(f"SLA Compliance: {sla_validation.get('message', 'N/A')}")
            print(f"Statistical Reliability: {reliability_validation.get('message', 'N/A')}")
            if memory_validation:
                print(f"Memory Efficiency: {memory_validation.get('message', 'N/A')}")
            print(f"Overall Result: {'✓ PASS' if overall_valid else '✗ FAIL'}")
            print("=" * 40)
            
            return validation_result
    
    return BenchmarkValidator(benchmark_config, statistical_analysis_engine)


# ============================================================================
# COMPREHENSIVE BENCHMARK COORDINATION FIXTURE
# ============================================================================

@pytest.fixture(scope="function")
def benchmark_coordinator(
    benchmark_config,
    benchmark_execution_environment,
    synthetic_data_generator,
    performance_baseline_manager,
    benchmark_validator,
    memory_leak_detector
):
    """
    Function-scoped fixture providing comprehensive benchmark coordination and orchestration.
    
    This fixture serves as the primary interface for benchmark test execution, providing
    unified access to all benchmark utilities, analysis engines, and validation tools
    with complete performance measurement and artifact generation capabilities.
    
    Features:
    - Unified benchmark execution with comprehensive analysis and validation
    - Statistical analysis with confidence intervals and regression detection
    - Memory profiling with leak detection for large dataset scenarios
    - Performance artifact generation for CI/CD integration and reporting
    - Cross-platform normalization and environment-aware benchmark execution
    - SLA compliance validation with detailed performance margin analysis
    
    Returns:
        BenchmarkCoordinator: Complete benchmark orchestration utility
    """
    class BenchmarkCoordinator:
        def __init__(self,
                     config: BenchmarkConfig,
                     execution_env: Dict[str, Any],
                     data_generator: Any,
                     baseline_manager: Any,
                     validator: Any,
                     leak_detector: Any):
            self.config = config
            self.execution_env = execution_env
            self.data_generator = data_generator
            self.baseline_manager = baseline_manager
            self.validator = validator
            self.leak_detector = leak_detector
            
            # Initialize comprehensive utilities coordinator
            self.utils_coordinator = BenchmarkUtilsCoordinator(config)
        
        def execute_comprehensive_benchmark(self,
                                          test_name: str,
                                          test_function: callable,
                                          test_category: str,
                                          data_size_config: Optional[Tuple[int, int, float]] = None,
                                          enable_memory_profiling: bool = False,
                                          enable_regression_detection: bool = True,
                                          establish_baseline: bool = False,
                                          *args, **kwargs) -> Dict[str, Any]:
            """
            Execute comprehensive benchmark with full analysis and validation.
            
            Args:
                test_name: Name of the benchmark test
                test_function: Function to benchmark
                test_category: Category for SLA validation
                data_size_config: Optional (rows, cols, mb) for data generation
                enable_memory_profiling: Whether to enable memory profiling
                enable_regression_detection: Whether to detect performance regressions
                establish_baseline: Whether to establish new performance baseline
                *args, **kwargs: Arguments to pass to test function
                
            Returns:
                Dict containing comprehensive benchmark results and analysis
            """
            print(f"\n=== Executing Comprehensive Benchmark: {test_name} ===")
            
            # Generate test data if configuration provided
            test_data = None
            if data_size_config:
                rows, cols, estimated_mb = data_size_config
                print(f"Generating test data: {rows:,} x {cols} ({estimated_mb:.1f}MB)")
                test_data = self.data_generator.generate_synthetic_dataset(
                    rows=rows, 
                    cols=cols,
                    data_type="large_dataset" if estimated_mb > 100 else "neural"
                )
                # Add data to function arguments
                kwargs["test_data"] = test_data
            
            # Setup memory profiling if enabled
            memory_results = None
            if enable_memory_profiling:
                data_size_estimate = int(data_size_config[2] * 1024 * 1024) if data_size_config else 1024*1024
                
                with memory_profiling_context(
                    data_size_estimate=data_size_estimate,
                    precision=self.config.memory_profiling.memory_profiling_precision,
                    enable_line_profiling=self.config.memory_profiling.enable_line_profiling
                ) as profiler:
                    # Execute the benchmark function multiple times for statistical analysis
                    measurements = []
                    for iteration in range(self.config.statistical_analysis.min_iterations):
                        start_time = time.perf_counter()
                        try:
                            result = test_function(*args, **kwargs)
                        except Exception as e:
                            print(f"Benchmark execution failed at iteration {iteration}: {e}")
                            raise
                        end_time = time.perf_counter()
                        
                        measurement = end_time - start_time
                        measurements.append(measurement)
                        profiler.update_peak_memory()
                        
                        # Brief pause between iterations
                        time.sleep(0.01)
                    
                    memory_results = profiler.end_profiling()
            else:
                # Execute benchmark without memory profiling
                measurements = []
                for iteration in range(self.config.statistical_analysis.min_iterations):
                    start_time = time.perf_counter()
                    try:
                        result = test_function(*args, **kwargs)
                    except Exception as e:
                        print(f"Benchmark execution failed at iteration {iteration}: {e}")
                        raise
                    end_time = time.perf_counter()
                    
                    measurement = end_time - start_time
                    measurements.append(measurement)
                    
                    # Brief pause between iterations
                    time.sleep(0.01)
            
            # Comprehensive validation
            expected_data_size_mb = data_size_config[2] if data_size_config else 0
            validation_results = self.validator.comprehensive_validation(
                test_category=test_category,
                measurements=measurements,
                memory_stats=memory_results,
                expected_data_size_mb=expected_data_size_mb
            )
            
            # Baseline management
            baseline_results = None
            if establish_baseline:
                baseline_results = self.baseline_manager.establish_baseline(
                    test_name=test_name,
                    measurements=measurements,
                    sla_category=test_category
                )
            elif enable_regression_detection:
                baseline_results = self.baseline_manager.validate_against_baseline(
                    test_name=test_name,
                    new_measurements=measurements
                )
            
            # Memory leak detection for large datasets
            leak_analysis = None
            if enable_memory_profiling and data_size_config and data_size_config[2] > 100:
                print("Performing memory leak detection for large dataset...")
                leak_analysis = self.leak_detector.detect_leak_in_iterations(
                    operation_func=test_function,
                    iterations=5,  # Reduced iterations for benchmark context
                    *args, **kwargs
                )
            
            # Compile comprehensive results
            benchmark_results = {
                "test_name": test_name,
                "test_category": test_category,
                "execution_timestamp": datetime.now().isoformat(),
                "measurements": measurements,
                "statistics": {
                    "mean": np.mean(measurements),
                    "median": np.median(measurements),
                    "std": np.std(measurements, ddof=1),
                    "min": np.min(measurements),
                    "max": np.max(measurements),
                    "cv_percent": (np.std(measurements, ddof=1) / np.mean(measurements)) * 100,
                    "iterations": len(measurements)
                },
                "validation_results": validation_results,
                "baseline_results": baseline_results,
                "memory_results": memory_results,
                "leak_analysis": leak_analysis,
                "environment_info": {
                    "platform": platform.system(),
                    "normalization_factors": self.execution_env["normalization_factors"],
                    "environment_suitability": self.execution_env["environment_report"]["benchmarking_suitability"]
                },
                "test_data_info": {
                    "data_generated": data_size_config is not None,
                    "data_size_config": data_size_config,
                    "estimated_size_mb": expected_data_size_mb
                } if data_size_config else None
            }
            
            # Generate artifacts if in CI environment
            if self.execution_env["cicd_manager"].is_github_actions:
                artifact_results = self.execution_env["cicd_manager"].collect_benchmark_artifacts(
                    benchmark_results={test_name: benchmark_results},
                    artifact_dirs=self.execution_env["artifact_directories"]
                )
                benchmark_results["artifacts_generated"] = artifact_results
            
            print(f"Benchmark '{test_name}' completed:")
            print(f"  Mean: {benchmark_results['statistics']['mean']:.4f}s")
            print(f"  CV: {benchmark_results['statistics']['cv_percent']:.1f}%")
            print(f"  Validation: {'✓ PASS' if validation_results['overall_valid'] else '✗ FAIL'}")
            if baseline_results:
                regression_detected = baseline_results.get('regression_analysis', {}).get('regression_analysis', {}).get('regression_detected', False)
                print(f"  Regression: {'⚠️ DETECTED' if regression_detected else '✓ NONE'}")
            
            return benchmark_results
        
        def execute_category_benchmarks(self, 
                                      category: BenchmarkCategory,
                                      test_suite: Dict[str, callable]) -> Dict[str, Any]:
            """
            Execute complete benchmark suite for a specific category.
            
            Args:
                category: Benchmark category to execute
                test_suite: Dict mapping test names to test functions
                
            Returns:
                Dict containing category benchmark results and analysis
            """
            print(f"\n=== Executing Category Benchmarks: {category.value} ===")
            
            category_config = get_category_config(category)
            category_results = {}
            
            for test_name, test_function in test_suite.items():
                try:
                    # Configure test based on category
                    if category == BenchmarkCategory.DATA_LOADING:
                        data_sizes = category_config["data_sizes"]
                        for size_name, (size_desc, size_mb) in data_sizes.items():
                            if size_mb <= 100:  # Limit data size for CI performance
                                test_result = self.execute_comprehensive_benchmark(
                                    test_name=f"{test_name}_{size_name}",
                                    test_function=test_function,
                                    test_category="data_loading",
                                    data_size_config=(int(size_mb * 1000), 50, size_mb),
                                    enable_memory_profiling=size_mb > 10
                                )
                                category_results[f"{test_name}_{size_name}"] = test_result
                    
                    elif category == BenchmarkCategory.TRANSFORMATION:
                        scale_configs = category_config["scale_configs"]
                        for scale_name, scale_config in scale_configs.items():
                            if scale_config["rows"] <= 100_000:  # Limit for CI performance
                                test_result = self.execute_comprehensive_benchmark(
                                    test_name=f"{test_name}_{scale_name}",
                                    test_function=test_function,
                                    test_category="transformation",
                                    data_size_config=(scale_config["rows"], 50, scale_config["rows"] * 50 * 4 / (1024*1024)),
                                    enable_memory_profiling=True
                                )
                                category_results[f"{test_name}_{scale_name}"] = test_result
                    
                    else:
                        # Generic benchmark execution
                        test_result = self.execute_comprehensive_benchmark(
                            test_name=test_name,
                            test_function=test_function,
                            test_category=category.value.replace("-", "_"),
                            enable_memory_profiling=False
                        )
                        category_results[test_name] = test_result
                
                except Exception as e:
                    print(f"Error executing {test_name}: {e}")
                    category_results[test_name] = {
                        "error": str(e),
                        "status": "failed"
                    }
            
            # Analyze category results
            category_analysis = self.utils_coordinator.execute_comprehensive_analysis(
                benchmark_results=category_results,
                enable_regression_detection=True,
                enable_ci_integration=self.execution_env["cicd_manager"].is_github_actions
            )
            
            print(f"Category '{category.value}' benchmarks completed: {len(category_results)} tests")
            print(f"Analysis status: {category_analysis['overall_status']}")
            
            return {
                "category": category.value,
                "test_results": category_results,
                "category_analysis": category_analysis,
                "execution_summary": {
                    "total_tests": len(category_results),
                    "successful_tests": len([r for r in category_results.values() if "error" not in r]),
                    "failed_tests": len([r for r in category_results.values() if "error" in r])
                }
            }
    
    return BenchmarkCoordinator(
        benchmark_config,
        benchmark_execution_environment,
        synthetic_data_generator,
        performance_baseline_manager,
        benchmark_validator,
        memory_leak_detector
    )


# ============================================================================
# FIXTURE CLEANUP AND RESOURCE MANAGEMENT
# ============================================================================

@pytest.fixture(scope="function", autouse=True)
def benchmark_resource_cleanup():
    """
    Auto-used function-scoped fixture ensuring proper resource cleanup after benchmarks.
    
    Features:
    - Memory cleanup with garbage collection for large dataset scenarios
    - Temporary file and directory cleanup for filesystem benchmarks
    - Process resource monitoring and cleanup for performance isolation
    - Statistical engine cache cleanup for consistent baseline management
    """
    # Pre-test setup
    initial_memory = psutil.Process().memory_info().rss
    temp_files_created = []
    
    yield temp_files_created
    
    # Post-test cleanup
    try:
        # Force garbage collection to clean up large datasets
        gc.collect()
        
        # Clean up any temporary files
        for temp_file in temp_files_created:
            try:
                if isinstance(temp_file, Path) and temp_file.exists():
                    if temp_file.is_file():
                        temp_file.unlink()
                    elif temp_file.is_dir():
                        import shutil
                        shutil.rmtree(temp_file)
            except Exception as e:
                warnings.warn(f"Failed to cleanup temporary file {temp_file}: {e}")
        
        # Check for memory leaks
        final_memory = psutil.Process().memory_info().rss
        memory_growth = (final_memory - initial_memory) / (1024 * 1024)
        
        if memory_growth > 50:  # >50MB growth indicates potential leak
            warnings.warn(f"Significant memory growth detected: {memory_growth:.1f}MB")
        
    except Exception as e:
        warnings.warn(f"Error during benchmark cleanup: {e}")


# ============================================================================
# MODULE METADATA AND EXPORTS
# ============================================================================

__all__ = [
    # Configuration fixtures
    "benchmark_config",
    "comprehensive_column_config", 
    "benchmark_execution_environment",
    
    # Data generation fixtures
    "synthetic_data_generator",
    "large_test_directory",
    "pattern_test_directory", 
    "benchmark_data_sizes",
    
    # Memory profiling fixtures
    "memory_profiler_context",
    "memory_leak_detector",
    
    # Mock provider fixtures
    "mock_filesystem_provider",
    "mock_configuration_provider",
    
    # Performance analysis fixtures
    "statistical_analysis_engine",
    "performance_regression_detector",
    "performance_baseline_manager",
    
    # Artifact generation fixtures
    "benchmark_artifact_generator",
    "ci_integration_manager",
    
    # Environment fixtures
    "environment_analyzer",
    "cross_platform_validator",
    "platform_performance_normalizer",
    
    # Conditional skip fixtures
    "platform_skip_conditions",
    
    # pytest-benchmark integration fixtures
    "benchmark_plugin_config",
    "benchmark_validator",
    
    # Comprehensive coordination fixture
    "benchmark_coordinator",
    
    # Resource management fixtures
    "benchmark_resource_cleanup"
]

# Module metadata
__version__ = "1.0.0"
__author__ = "FlyRigLoader Benchmark Test Suite"
__description__ = "Centralized pytest fixture configuration for comprehensive benchmark testing with statistical analysis, memory profiling, and CI/CD integration"