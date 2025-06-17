"""
Centralized configuration module for the flyrigloader benchmark test suite.

This module consolidates all performance SLA thresholds, statistical analysis parameters,
environment normalization settings, execution categories, and CI/CD integration configuration
to support comprehensive performance validation across development and CI environments.

Provides unified configuration management for:
- Performance SLA thresholds per technical specification requirements
- Statistical analysis parameters for confidence intervals and regression detection
- Environment normalization for cross-platform performance consistency
- Benchmark execution categories and selective execution capabilities
- CI/CD integration settings for GitHub Actions and artifact management
- Memory profiling configuration for large dataset scenarios
"""

import os
import platform
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field

import psutil


# ============================================================================
# PERFORMANCE SLA THRESHOLDS
# ============================================================================

class PerformanceSLA:
    """
    Performance Service Level Agreement thresholds per technical specification.
    
    All SLA values are derived from the technical specification requirements:
    - TST-PERF-001: Data loading performance requirements
    - TST-PERF-002: DataFrame transformation performance requirements  
    - F-003-RQ-004: Format detection performance requirements
    - F-002-RQ-001: File discovery performance requirements
    """
    
    # Data Loading SLA (TST-PERF-001)
    DATA_LOADING_TIME_PER_100MB_SECONDS = 1.0  # <1s per 100MB
    DATA_LOADING_FORMAT_DETECTION_MS = 100.0   # F-003-RQ-004: <100ms format detection
    
    # DataFrame Transformation SLA (TST-PERF-002)  
    DATAFRAME_TRANSFORM_TIME_PER_1M_ROWS_MS = 500.0  # <500ms per 1M rows
    TRANSFORMATION_HANDLER_TIME_MS = 50.0             # F-006-RQ-003: <50ms per handler
    METADATA_MERGE_TIME_MS = 10.0                     # F-006-RQ-004: <10ms merge time
    
    # File Discovery SLA (F-002-RQ-001)
    FILE_DISCOVERY_TIME_FOR_10K_FILES_SECONDS = 5.0  # <5s for 10,000 files
    RECURSIVE_TRAVERSAL_MAX_SECONDS = 5.0             # F-002-RQ-001 base requirement
    
    # Configuration Management SLA (F-001-RQ series)
    CONFIG_LOADING_TIME_MS = 100.0                    # F-001-RQ-001: <100ms file loading
    CONFIG_VALIDATION_TIME_MS = 10.0                  # F-001-RQ-003: <10ms validation
    CONFIG_MERGE_TIME_MS = 5.0                        # F-001-RQ-004: <5ms merge operation
    CONFIG_DICT_VALIDATION_MS = 50.0                  # F-001-RQ-002: <50ms dict validation
    
    # Memory Constraints
    CONFIG_MEMORY_LIMIT_MB = 10.0                     # <10MB for large configurations
    TRANSFORMATION_MEMORY_MULTIPLIER = 2.0            # <2x data size overhead (Section 2.4.5)


# ============================================================================
# STATISTICAL ANALYSIS CONFIGURATION
# ============================================================================

@dataclass
class StatisticalAnalysisConfig:
    """
    Configuration for statistical analysis and regression detection in benchmarks.
    
    Provides comprehensive settings for confidence interval calculation,
    regression detection, baseline comparison, and statistical significance testing.
    """
    
    # Confidence Interval Settings
    confidence_level: float = 0.95                    # 95% confidence interval
    min_iterations: int = 3                           # Minimum benchmark iterations
    warmup_iterations: int = 1                        # Warmup rounds before measurement
    max_iterations: int = 10                          # Maximum iterations for accuracy
    
    # Variance and Stability Thresholds
    acceptable_variance_ratio: float = 0.1            # 10% variance tolerance
    stability_threshold: float = 0.05                 # 5% for statistical stability
    outlier_detection_stddev: float = 2.0             # 2 standard deviations for outliers
    
    # Regression Detection Parameters
    regression_threshold_percent: float = 20.0        # 20% performance degradation threshold
    baseline_comparison_samples: int = 5              # Historical samples for baseline
    significance_level: float = 0.05                  # p-value threshold for significance
    
    # Performance Consistency Requirements
    cross_platform_variance_limit: float = 0.3       # 30% variance across platforms
    ci_environment_variance_limit: float = 0.5       # 50% variance in CI vs local
    
    # Benchmark Execution Tuning
    calibration_precision: float = 0.01               # 1% precision target for calibration
    statistical_power: float = 0.8                    # 80% statistical power for tests
    
    def get_benchmark_kwargs(self, test_type: str = "standard") -> Dict[str, Any]:
        """
        Get pytest-benchmark configuration for specific test types.
        
        Args:
            test_type: Type of benchmark test ("standard", "precise", "fast")
            
        Returns:
            Dict with pytest-benchmark configuration parameters
        """
        if test_type == "precise":
            return {
                "min_rounds": self.max_iterations,
                "max_time": 60.0,  # Max 60 seconds for precise tests
                "calibration_precision": self.calibration_precision,
                "warmup": True
            }
        elif test_type == "fast":
            return {
                "min_rounds": self.min_iterations,
                "max_time": 10.0,  # Max 10 seconds for fast tests
                "warmup_rounds": 1
            }
        else:  # standard
            return {
                "min_rounds": self.min_iterations,
                "warmup_rounds": self.warmup_iterations,
                "max_time": 30.0  # Max 30 seconds for standard tests
            }


# ============================================================================
# ENVIRONMENT NORMALIZATION CONFIGURATION
# ============================================================================

@dataclass
class EnvironmentNormalizationConfig:
    """
    Configuration for environment normalization across development and CI platforms.
    
    Provides CPU and memory constraint detection with normalization factors
    to ensure consistent benchmark results across diverse computational environments.
    """
    
    # CPU Normalization Parameters
    reference_cpu_cores: int = 4                      # Reference CPU core count
    reference_cpu_frequency_ghz: float = 2.5          # Reference CPU frequency
    cpu_virtualization_overhead: float = 0.15         # 15% overhead for virtualized environments
    
    # Memory Normalization Parameters  
    reference_memory_gb: int = 8                       # Reference memory allocation
    memory_pressure_threshold: float = 0.8            # 80% memory usage threshold
    
    # CI Environment Detection
    ci_environment_indicators: List[str] = field(default_factory=lambda: [
        "CI", "CONTINUOUS_INTEGRATION", "GITHUB_ACTIONS", 
        "TRAVIS", "CIRCLECI", "JENKINS"
    ])
    
    # Platform-Specific Normalization Factors
    platform_performance_factors: Dict[str, float] = field(default_factory=lambda: {
        "Windows": 1.0,      # Baseline platform
        "Linux": 0.95,       # Slightly faster on Linux
        "Darwin": 1.05       # Slightly slower on macOS
    })
    
    # Virtualization Detection
    virtualization_indicators: List[str] = field(default_factory=lambda: [
        "docker", "container", "vm", "virtual", "qemu", "kvm"
    ])
    
    def detect_environment_characteristics(self) -> Dict[str, Any]:
        """
        Detect current environment characteristics for normalization.
        
        Returns:
            Dict containing environment characteristics and normalization factors
        """
        # CPU characteristics
        cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        cpu_frequency = cpu_freq.current if cpu_freq else self.reference_cpu_frequency_ghz * 1000
        
        # Memory characteristics
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_usage = memory.percent / 100.0
        
        # Environment detection
        is_ci = any(os.getenv(indicator) for indicator in self.ci_environment_indicators)
        platform_name = platform.system()
        
        # Virtualization detection (basic heuristics)
        is_virtualized = self._detect_virtualization()
        
        # Calculate normalization factors
        cpu_factor = self._calculate_cpu_normalization_factor(cpu_count, cpu_frequency)
        memory_factor = self._calculate_memory_normalization_factor(memory_gb, memory_usage)
        platform_factor = self.platform_performance_factors.get(platform_name, 1.0)
        
        # CI environment adjustment
        ci_factor = 1.0 + self.cpu_virtualization_overhead if is_ci else 1.0
        
        # Virtualization adjustment
        virtualization_factor = 1.0 + self.cpu_virtualization_overhead if is_virtualized else 1.0
        
        # Combined normalization factor
        combined_factor = cpu_factor * memory_factor * platform_factor * ci_factor * virtualization_factor
        
        return {
            "cpu_count": cpu_count,
            "cpu_frequency_mhz": cpu_frequency,
            "memory_gb": memory_gb,
            "memory_usage_percent": memory_usage * 100,
            "platform": platform_name,
            "is_ci_environment": is_ci,
            "is_virtualized": is_virtualized,
            "normalization_factors": {
                "cpu": cpu_factor,
                "memory": memory_factor,
                "platform": platform_factor,
                "ci": ci_factor,
                "virtualization": virtualization_factor,
                "combined": combined_factor
            }
        }
    
    def _calculate_cpu_normalization_factor(self, cpu_count: int, cpu_frequency: float) -> float:
        """Calculate CPU performance normalization factor."""
        # Normalize based on core count and frequency
        core_factor = cpu_count / self.reference_cpu_cores
        freq_factor = (cpu_frequency / 1000.0) / self.reference_cpu_frequency_ghz
        
        # Conservative approach: use minimum of core and frequency factors
        return min(core_factor, freq_factor)
    
    def _calculate_memory_normalization_factor(self, memory_gb: float, memory_usage: float) -> float:
        """Calculate memory performance normalization factor."""
        # Memory factor based on available memory
        memory_factor = memory_gb / self.reference_memory_gb
        
        # Adjust for memory pressure
        if memory_usage > self.memory_pressure_threshold:
            pressure_penalty = 1.0 + (memory_usage - self.memory_pressure_threshold)
            memory_factor /= pressure_penalty
        
        return memory_factor
    
    def _detect_virtualization(self) -> bool:
        """Detect if running in a virtualized environment."""
        # Check for virtualization indicators in various system properties
        try:
            # Check DMI information on Linux
            if platform.system() == "Linux":
                dmi_paths = ["/sys/class/dmi/id/product_name", "/sys/class/dmi/id/sys_vendor"]
                for path in dmi_paths:
                    try:
                        with open(path, 'r') as f:
                            content = f.read().lower()
                            if any(indicator in content for indicator in self.virtualization_indicators):
                                return True
                    except (IOError, OSError):
                        continue
            
            # Check for container environment
            if os.path.exists("/.dockerenv") or os.getenv("container"):
                return True
                
        except Exception:
            pass
        
        return False


# ============================================================================
# BENCHMARK EXECUTION CATEGORIES
# ============================================================================

class BenchmarkCategory(Enum):
    """
    Enumeration of benchmark execution categories for selective execution.
    
    Each category groups related performance tests for targeted execution
    via the CLI runner's --category flag.
    """
    
    DATA_LOADING = "data-loading"
    TRANSFORMATION = "transformation"  
    DISCOVERY = "discovery"
    CONFIG = "config"
    MEMORY_PROFILING = "memory-profiling"
    INTEGRATION = "integration"
    ALL = "all"


@dataclass
class BenchmarkExecutionConfig:
    """
    Configuration for benchmark execution management and selective execution.
    
    Defines execution parameters for different benchmark categories and
    provides configuration for pytest-benchmark integration.
    """
    
    # Category Execution Settings
    category_timeout_seconds: Dict[BenchmarkCategory, float] = field(default_factory=lambda: {
        BenchmarkCategory.DATA_LOADING: 120.0,        # 2 minutes for data loading tests
        BenchmarkCategory.TRANSFORMATION: 180.0,      # 3 minutes for transformation tests
        BenchmarkCategory.DISCOVERY: 60.0,            # 1 minute for discovery tests
        BenchmarkCategory.CONFIG: 30.0,               # 30 seconds for config tests
        BenchmarkCategory.MEMORY_PROFILING: 300.0,    # 5 minutes for memory profiling
        BenchmarkCategory.INTEGRATION: 240.0,         # 4 minutes for integration tests
        BenchmarkCategory.ALL: 600.0                  # 10 minutes for complete suite
    })
    
    # Test Data Size Configurations
    data_size_scenarios: Dict[str, Tuple[str, float]] = field(default_factory=lambda: {
        "small_1mb": ("small_1mb", 1.0),
        "small_5mb": ("small_5mb", 5.0),
        "medium_10mb": ("medium_10mb", 10.0),
        "medium_50mb": ("medium_50mb", 50.0),
        "large_100mb": ("large_100mb", 100.0),
        "large_500mb": ("large_500mb", 500.0),
        "xlarge_1gb": ("xlarge_1gb", 1000.0)
    })
    
    # Transformation Scale Configurations
    transformation_scale_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "small_scale": {
            "rows": 10_000,
            "expected_transform_time_ms": 5.0,
            "memory_multiplier_threshold": 1.5,
            "use_case": "Unit test validation"
        },
        "medium_scale": {
            "rows": 100_000,
            "expected_transform_time_ms": 50.0,
            "memory_multiplier_threshold": 1.8,
            "use_case": "Standard experimental session"
        },
        "large_scale": {
            "rows": 1_000_000,
            "expected_transform_time_ms": 500.0,  # TST-PERF-002 SLA limit
            "memory_multiplier_threshold": 2.0,
            "use_case": "Maximum SLA validation"
        },
        "stress_scale": {
            "rows": 2_000_000,
            "expected_transform_time_ms": 1000.0,
            "memory_multiplier_threshold": 2.0,
            "use_case": "Performance regression detection"
        }
    })
    
    # Discovery Test Configurations
    discovery_test_configs: Dict[str, Any] = field(default_factory=lambda: {
        "sla_time_limit": 5.0,              # 5 seconds for 10,000 files per F-002-RQ-001
        "acceptable_variance": 0.1,          # 10% variance in performance
        "file_count_scenarios": [1000, 2000, 5000, 10000],
        "pattern_complexity_levels": ["simple", "moderate", "complex"]
    })
    
    # Configuration Test Settings
    config_test_settings: Dict[str, Any] = field(default_factory=lambda: {
        "config_sizes": ["small", "medium", "large"],
        "loading_timeout_ms": 100.0,
        "validation_timeout_ms": 10.0,
        "merge_timeout_ms": 5.0,
        "memory_limit_mb": 10.0
    })
    
    def get_category_tests(self, category: BenchmarkCategory) -> List[str]:
        """
        Get list of test modules for a specific benchmark category.
        
        Args:
            category: Benchmark category to get tests for
            
        Returns:
            List of test module names for the category
        """
        category_mapping = {
            BenchmarkCategory.DATA_LOADING: ["test_benchmark_data_loading.py"],
            BenchmarkCategory.TRANSFORMATION: ["test_benchmark_transformations.py"],
            BenchmarkCategory.DISCOVERY: ["test_benchmark_discovery.py"],
            BenchmarkCategory.CONFIG: ["test_benchmark_config.py"],
            BenchmarkCategory.MEMORY_PROFILING: [
                "test_benchmark_data_loading.py",
                "test_benchmark_transformations.py"
            ],
            BenchmarkCategory.INTEGRATION: [
                "test_benchmark_discovery.py",
                "test_benchmark_config.py"
            ],
            BenchmarkCategory.ALL: [
                "test_benchmark_data_loading.py",
                "test_benchmark_transformations.py", 
                "test_benchmark_discovery.py",
                "test_benchmark_config.py"
            ]
        }
        
        return category_mapping.get(category, [])


# ============================================================================
# MEMORY PROFILING CONFIGURATION
# ============================================================================

@dataclass
class MemoryProfilingConfig:
    """
    Configuration for memory profiling and leak detection in benchmarks.
    
    Provides settings for pytest-memory-profiler integration, large dataset
    memory analysis, and memory leak detection procedures.
    """
    
    # Large Dataset Thresholds
    large_dataset_threshold_mb: float = 500.0         # >500MB considered large dataset
    memory_leak_detection_threshold_mb: float = 10.0  # 10MB memory growth threshold
    memory_profiling_precision: int = 3               # Precision for memory measurements
    
    # Memory Profiling Settings
    enable_line_profiling: bool = True                # Enable line-by-line profiling
    enable_leak_detection: bool = True                # Enable memory leak detection
    profile_gc_behavior: bool = True                  # Profile garbage collection impact
    
    # Memory Leak Detection Parameters
    leak_detection_iterations: int = 10               # Iterations for leak detection
    gc_collection_between_iterations: bool = True     # Force GC between iterations
    baseline_memory_samples: int = 3                  # Baseline memory measurements
    
    # Memory Efficiency Validation
    memory_efficiency_targets: Dict[str, float] = field(default_factory=lambda: {
        "data_loading_overhead_factor": 1.5,          # Max 1.5x data size in memory
        "transformation_overhead_factor": 2.0,        # Max 2x data size during transform
        "discovery_memory_limit_mb": 100.0,           # Max 100MB for discovery operations
        "config_memory_limit_mb": 10.0                # Max 10MB for config operations
    })
    
    # pytest-memory-profiler Integration
    memory_profiler_settings: Dict[str, Any] = field(default_factory=lambda: {
        "precision": 3,
        "stream": None,  # Use default stream
        "backend": "psutil"  # Use psutil backend for cross-platform compatibility
    })
    
    def get_memory_profiling_decorator_args(self, test_type: str = "standard") -> Dict[str, Any]:
        """
        Get memory profiling decorator arguments for different test types.
        
        Args:
            test_type: Type of memory profiling ("standard", "detailed", "leak_detection")
            
        Returns:
            Dict with memory profiler decorator arguments
        """
        base_args = self.memory_profiler_settings.copy()
        
        if test_type == "detailed":
            base_args.update({
                "precision": 4,
                "interval": 0.1  # More frequent sampling
            })
        elif test_type == "leak_detection":
            base_args.update({
                "precision": 4,
                "interval": 0.05,  # High-frequency sampling for leak detection
                "timeout": 300.0   # 5-minute timeout for leak detection tests
            })
        
        return base_args


# ============================================================================
# CI/CD INTEGRATION CONFIGURATION
# ============================================================================

@dataclass
class CICDIntegrationConfig:
    """
    Configuration for CI/CD integration including GitHub Actions workflow settings,
    artifact management, and performance alerting thresholds.
    """
    
    # GitHub Actions Integration
    github_actions_settings: Dict[str, Any] = field(default_factory=lambda: {
        "workflow_dispatch_enabled": True,
        "pr_label_triggers": ["benchmark", "performance"],
        "artifact_retention_days": 90,  # 90-day retention for benchmark artifacts
        "performance_report_retention_days": 30
    })
    
    # Artifact Management
    artifact_settings: Dict[str, Any] = field(default_factory=lambda: {
        "output_formats": ["json", "csv", "html"],
        "include_system_info": True,
        "include_environment_info": True,
        "compress_artifacts": True,
        "artifact_naming_pattern": "benchmark-{category}-{timestamp}-{platform}"
    })
    
    # Performance Alerting Thresholds
    performance_alerting: Dict[str, float] = field(default_factory=lambda: {
        "regression_threshold_percent": 25.0,         # 25% performance regression alert
        "sla_violation_threshold_percent": 10.0,      # 10% over SLA threshold alert
        "memory_usage_alert_threshold_percent": 50.0, # 50% over expected memory usage
        "test_failure_rate_threshold_percent": 5.0    # 5% test failure rate alert
    })
    
    # Benchmark Report Generation
    report_generation: Dict[str, Any] = field(default_factory=lambda: {
        "include_historical_comparison": True,
        "include_regression_analysis": True,
        "include_statistical_summary": True,
        "include_environment_details": True,
        "generate_trend_charts": True,
        "chart_history_days": 30
    })
    
    # Cross-Platform Compatibility
    cross_platform_settings: Dict[str, Any] = field(default_factory=lambda: {
        "test_platforms": ["ubuntu-latest", "windows-latest", "macos-latest"],
        "python_versions": ["3.8", "3.9", "3.10", "3.11"],
        "normalize_cross_platform_results": True,
        "platform_variance_tolerance": 0.3  # 30% variance across platforms
    })


# ============================================================================
# PYTEST-BENCHMARK INTEGRATION CONFIGURATION  
# ============================================================================

@dataclass
class PytestBenchmarkConfig:
    """
    Configuration for pytest-benchmark integration including statistical analysis,
    regression detection, and artifact generation settings.
    """
    
    # Benchmark Storage and Comparison
    benchmark_storage: Dict[str, Any] = field(default_factory=lambda: {
        "storage_type": "json",
        "storage_path": "benchmark-results",
        "save_data": True,
        "autosave": True,
        "compare": True
    })
    
    # Statistical Analysis Settings
    statistical_settings: Dict[str, Any] = field(default_factory=lambda: {
        "statistics": ["mean", "stddev", "min", "max", "median"],
        "histogram": True,
        "warmup": True,
        "disable_gc": False,  # Keep GC enabled for realistic performance
        "timer": "time.perf_counter"
    })
    
    # Regression Detection
    regression_settings: Dict[str, Any] = field(default_factory=lambda: {
        "compare_fail": ["stddev"],  # Fail on increased standard deviation
        "performance_regressions": True,
        "skip_first": False,
        "sort": "mean"
    })
    
    # Output and Reporting
    output_settings: Dict[str, Any] = field(default_factory=lambda: {
        "verbose": True,
        "json": True,
        "csv": True,
        "columns": ["min", "max", "mean", "stddev", "median", "ops"],
        "group_by": "group"
    })
    
    def get_pytest_benchmark_config(self) -> Dict[str, Any]:
        """
        Get complete pytest-benchmark configuration dictionary.
        
        Returns:
            Dict with complete pytest-benchmark configuration
        """
        return {
            **self.benchmark_storage,
            **self.statistical_settings,
            **self.regression_settings,
            **self.output_settings
        }


# ============================================================================
# CONSOLIDATED BENCHMARK CONFIGURATION
# ============================================================================

@dataclass
class BenchmarkConfig:
    """
    Consolidated benchmark configuration providing unified access to all
    benchmark settings, SLA thresholds, and execution parameters.
    """
    
    # Core Configuration Components
    sla: PerformanceSLA = field(default_factory=PerformanceSLA)
    statistical_analysis: StatisticalAnalysisConfig = field(default_factory=StatisticalAnalysisConfig)
    environment_normalization: EnvironmentNormalizationConfig = field(default_factory=EnvironmentNormalizationConfig)
    execution: BenchmarkExecutionConfig = field(default_factory=BenchmarkExecutionConfig)
    memory_profiling: MemoryProfilingConfig = field(default_factory=MemoryProfilingConfig)
    cicd_integration: CICDIntegrationConfig = field(default_factory=CICDIntegrationConfig)
    pytest_benchmark: PytestBenchmarkConfig = field(default_factory=PytestBenchmarkConfig)
    
    def get_sla_threshold(self, test_type: str, scale: Optional[str] = None) -> float:
        """
        Get SLA threshold for specific test type and scale.
        
        Args:
            test_type: Type of test ("data_loading", "transformation", "discovery", "config")
            scale: Optional scale modifier ("small", "medium", "large")
            
        Returns:
            SLA threshold value in appropriate units
        """
        base_thresholds = {
            "data_loading": self.sla.DATA_LOADING_TIME_PER_100MB_SECONDS,
            "transformation": self.sla.DATAFRAME_TRANSFORM_TIME_PER_1M_ROWS_MS,
            "discovery": self.sla.FILE_DISCOVERY_TIME_FOR_10K_FILES_SECONDS,
            "config_loading": self.sla.CONFIG_LOADING_TIME_MS,
            "config_validation": self.sla.CONFIG_VALIDATION_TIME_MS,
            "config_merge": self.sla.CONFIG_MERGE_TIME_MS
        }
        
        base_threshold = base_thresholds.get(test_type, 1.0)
        
        # Apply scale modifiers if provided
        if scale and test_type in ["data_loading", "transformation"]:
            scale_factors = {
                "small": 0.1,    # 10% of base threshold for small scale
                "medium": 0.5,   # 50% of base threshold for medium scale  
                "large": 1.0,    # Full threshold for large scale
                "stress": 2.0    # 2x threshold for stress testing
            }
            scale_factor = scale_factors.get(scale, 1.0)
            return base_threshold * scale_factor
        
        return base_threshold
    
    def validate_performance_result(
        self, 
        test_type: str, 
        measured_value: float, 
        scale: Optional[str] = None,
        units: str = "seconds"
    ) -> Tuple[bool, str]:
        """
        Validate performance result against SLA thresholds.
        
        Args:
            test_type: Type of test performed
            measured_value: Measured performance value
            scale: Optional scale for the test
            units: Units of measurement ("seconds", "ms")
            
        Returns:
            Tuple of (passes_sla, validation_message)
        """
        threshold = self.get_sla_threshold(test_type, scale)
        
        # Convert units if necessary
        if units == "ms" and test_type in ["data_loading", "discovery"]:
            measured_value = measured_value / 1000.0  # Convert to seconds
        elif units == "seconds" and test_type in ["transformation", "config_loading", "config_validation", "config_merge"]:
            measured_value = measured_value * 1000.0  # Convert to milliseconds
            
        passes_sla = measured_value <= threshold
        
        if passes_sla:
            percentage = (measured_value / threshold) * 100
            message = f"✓ SLA PASS: {measured_value:.3f}{units} ≤ {threshold:.3f}{units} ({percentage:.1f}% of threshold)"
        else:
            excess_percentage = ((measured_value / threshold) - 1) * 100
            message = f"✗ SLA FAIL: {measured_value:.3f}{units} > {threshold:.3f}{units} (+{excess_percentage:.1f}% over threshold)"
        
        return passes_sla, message


# ============================================================================
# DEFAULT CONFIGURATION INSTANCE
# ============================================================================

# Global default configuration instance
DEFAULT_BENCHMARK_CONFIG = BenchmarkConfig()


# ============================================================================
# CONFIGURATION FACTORY FUNCTIONS
# ============================================================================

def get_benchmark_config(environment: str = "auto") -> BenchmarkConfig:
    """
    Get benchmark configuration for specific environment.
    
    Args:
        environment: Target environment ("local", "ci", "auto")
        
    Returns:
        BenchmarkConfig instance configured for the environment
    """
    config = BenchmarkConfig()
    
    if environment == "auto":
        # Auto-detect environment
        env_chars = config.environment_normalization.detect_environment_characteristics()
        environment = "ci" if env_chars["is_ci_environment"] else "local"
    
    if environment == "ci":
        # Adjust thresholds for CI environment
        config.statistical_analysis.acceptable_variance_ratio = 0.15  # 15% variance in CI
        config.statistical_analysis.min_iterations = 2  # Fewer iterations in CI
        config.execution.category_timeout_seconds[BenchmarkCategory.ALL] = 300.0  # 5 min timeout
        
    elif environment == "local":
        # Optimize for local development
        config.statistical_analysis.min_iterations = 3
        config.statistical_analysis.max_iterations = 5
        config.memory_profiling.enable_line_profiling = True
        
    return config


def get_category_config(category: BenchmarkCategory) -> Dict[str, Any]:
    """
    Get configuration specific to a benchmark category.
    
    Args:
        category: Benchmark category
        
    Returns:
        Dict with category-specific configuration
    """
    config = DEFAULT_BENCHMARK_CONFIG
    
    base_config = {
        "timeout": config.execution.category_timeout_seconds[category],
        "statistical_settings": config.statistical_analysis.get_benchmark_kwargs(),
        "memory_settings": config.memory_profiling.get_memory_profiling_decorator_args()
    }
    
    # Category-specific additions
    if category == BenchmarkCategory.DATA_LOADING:
        base_config.update({
            "data_sizes": config.execution.data_size_scenarios,
            "sla_thresholds": {
                "loading_time_per_100mb": config.sla.DATA_LOADING_TIME_PER_100MB_SECONDS,
                "format_detection_ms": config.sla.DATA_LOADING_FORMAT_DETECTION_MS
            }
        })
    elif category == BenchmarkCategory.TRANSFORMATION:
        base_config.update({
            "scale_configs": config.execution.transformation_scale_configs,
            "sla_thresholds": {
                "transform_time_per_1m_rows_ms": config.sla.DATAFRAME_TRANSFORM_TIME_PER_1M_ROWS_MS,
                "handler_time_ms": config.sla.TRANSFORMATION_HANDLER_TIME_MS,
                "memory_multiplier": config.sla.TRANSFORMATION_MEMORY_MULTIPLIER
            }
        })
    elif category == BenchmarkCategory.DISCOVERY:
        base_config.update({
            "discovery_config": config.execution.discovery_test_configs,
            "sla_thresholds": {
                "discovery_time_10k_files": config.sla.FILE_DISCOVERY_TIME_FOR_10K_FILES_SECONDS
            }
        })
    elif category == BenchmarkCategory.CONFIG:
        base_config.update({
            "config_settings": config.execution.config_test_settings,
            "sla_thresholds": {
                "loading_time_ms": config.sla.CONFIG_LOADING_TIME_MS,
                "validation_time_ms": config.sla.CONFIG_VALIDATION_TIME_MS,
                "merge_time_ms": config.sla.CONFIG_MERGE_TIME_MS
            }
        })
    
    return base_config


if __name__ == "__main__":
    # Example usage and configuration validation
    config = get_benchmark_config("auto")
    
    print("Benchmark Configuration Summary:")
    print(f"Environment: {config.environment_normalization.detect_environment_characteristics()}")
    print(f"SLA Thresholds: Data Loading {config.sla.DATA_LOADING_TIME_PER_100MB_SECONDS}s/100MB")
    print(f"Statistical Settings: {config.statistical_analysis.confidence_level*100}% confidence")
    print(f"Memory Profiling: {'Enabled' if config.memory_profiling.enable_line_profiling else 'Disabled'}")