"""
FlyRigLoader Benchmark Testing Framework

This package provides comprehensive performance testing and benchmarking capabilities for the
FlyRigLoader data processing pipeline. These performance tests have been relocated from the
default test suite to maintain rapid developer feedback cycles while preserving comprehensive
performance validation capabilities.

The benchmark framework supports:
- Statistical analysis with confidence intervals and regression detection
- Memory profiling for large dataset processing scenarios
- Cross-platform performance validation (Ubuntu, Windows, macOS)
- Environment normalization for consistent results across development and CI
- CLI-based benchmark execution for manual performance validation
- GitHub Actions integration for optional comprehensive performance analysis

Usage Examples:
    # Execute all performance benchmarks via CLI
    python scripts/benchmarks/run_benchmarks.py
    
    # Execute specific benchmark categories
    python scripts/benchmarks/run_benchmarks.py --category data-loading
    python scripts/benchmarks/run_benchmarks.py --category transformation
    
    # Execute with detailed memory profiling
    python scripts/benchmarks/run_benchmarks.py --memory-profile --verbose
    
    # CI-mode execution with artifact generation
    python scripts/benchmarks/run_benchmarks.py --ci-mode --report-artifacts

Architecture:
    This package maintains isolation from the default test suite execution to ensure:
    - Default pytest execution completes in <30 seconds for rapid developer feedback
    - Performance-intensive tests are excluded from routine development workflows
    - Comprehensive performance validation remains available through dedicated execution
    - CI/CD integration provides optional performance regression detection

Performance SLA Validation:
    - TST-PERF-001: Data loading <1s per 100MB
    - TST-PERF-002: DataFrame transformation <500ms per 1M rows
    - F-003-RQ-004: Format detection overhead <100ms
    - Memory efficiency requirements for large dataset processing scenarios
"""

import os
import sys
import platform
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

# Package metadata
__version__ = "1.0.0"
__author__ = "FlyRigLoader Development Team"
__description__ = "Performance testing and benchmarking framework for FlyRigLoader"

# Performance validation capabilities
BENCHMARK_CAPABILITIES = {
    "statistical_analysis": True,
    "memory_profiling": True,
    "environment_normalization": True,
    "cross_platform_support": True,
    "cli_execution": True,
    "ci_integration": True,
    "regression_detection": True,
    "artifact_generation": True,
}

# Performance SLA thresholds for validation
PERFORMANCE_SLA_THRESHOLDS = {
    "TST-PERF-001": {"description": "Data loading", "threshold": "1s per 100MB"},
    "TST-PERF-002": {"description": "DataFrame transformation", "threshold": "500ms per 1M rows"},
    "F-003-RQ-004": {"description": "Format detection overhead", "threshold": "100ms"},
    "memory_efficiency": {"description": "Memory usage", "threshold": "<10MB for large configs"},
}


def _detect_environment() -> Dict[str, Any]:
    """
    Detect the current execution environment for cross-platform compatibility.
    
    Returns:
        Dict containing environment information including platform, Python version,
        CI detection, and resource constraints for benchmark normalization.
    """
    env_info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "is_ci": bool(os.getenv("CI", False)),
        "is_github_actions": bool(os.getenv("GITHUB_ACTIONS", False)),
        "pytest_current_test": os.getenv("PYTEST_CURRENT_TEST"),
        "is_test_mode": "pytest" in sys.modules or bool(os.getenv("PYTEST_CURRENT_TEST")),
    }
    
    # Detect resource constraints for environment normalization
    try:
        import psutil
        env_info.update({
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
        })
    except ImportError:
        # Fallback to os module if psutil not available
        env_info.update({
            "cpu_count": os.cpu_count(),
            "cpu_count_physical": None,
            "memory_total": None,
            "memory_available": None,
        })
    
    return env_info


def _initialize_logging() -> Optional[Any]:
    """
    Initialize logging integration with flyrigloader's Loguru-based logging system.
    
    Returns:
        Logger instance if flyrigloader is available, None otherwise.
    """
    logger = None
    
    try:
        # Import flyrigloader logger for integration
        from flyrigloader import logger as flyrig_logger
        logger = flyrig_logger
        
        # Configure benchmark-specific logging context
        logger.info("Initializing FlyRigLoader benchmark framework")
        logger.debug(f"Environment detected: {_detect_environment()['platform']}")
        
    except ImportError:
        # Fallback to standard logging if flyrigloader not available
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logger = logging.getLogger("scripts.benchmarks")
        logger.info("Using fallback logging for benchmark framework")
    
    return logger


def _get_benchmark_config_defaults() -> Dict[str, Any]:
    """
    Initialize benchmark configuration defaults and environment detection.
    
    Returns:
        Dictionary containing default configuration values for cross-platform
        compatibility and benchmark execution settings.
    """
    env_info = _detect_environment()
    
    # Base configuration defaults
    config_defaults = {
        "execution": {
            "parallel_workers": min(4, env_info.get("cpu_count", 2)),
            "timeout_seconds": 300,
            "retry_count": 3,
            "memory_limit_mb": 1024,
        },
        "statistical_analysis": {
            "confidence_level": 0.95,
            "regression_threshold": 0.1,
            "outlier_detection": True,
            "baseline_comparison": True,
        },
        "memory_profiling": {
            "enabled": True,
            "line_by_line": False,
            "leak_detection": True,
            "large_dataset_threshold_mb": 500,
        },
        "environment_normalization": {
            "cpu_normalization": True,
            "memory_normalization": True,
            "platform_scaling": True,
            "ci_environment_detection": env_info["is_ci"],
        },
        "artifact_generation": {
            "json_reports": True,
            "csv_exports": True,
            "retention_days": 90,
            "trend_analysis": True,
        },
    }
    
    # CI-specific adjustments
    if env_info["is_ci"]:
        config_defaults["execution"]["parallel_workers"] = 2
        config_defaults["execution"]["timeout_seconds"] = 600
        config_defaults["memory_profiling"]["line_by_line"] = False
    
    # Platform-specific adjustments
    if env_info["platform"] == "Windows":
        config_defaults["execution"]["timeout_seconds"] *= 1.5  # Windows overhead
    elif env_info["platform"] == "Darwin":  # macOS
        config_defaults["memory_profiling"]["line_by_line"] = True  # Better support
    
    return config_defaults


def _setup_pytest_discovery() -> None:
    """
    Set up proper package structure to support pytest discovery within scripts/benchmarks/.
    
    This ensures that performance tests can be discovered and executed via pytest
    when explicitly requested, while remaining excluded from default test execution.
    """
    # Add current directory to Python path for pytest discovery
    benchmark_dir = Path(__file__).parent
    if str(benchmark_dir) not in sys.path:
        sys.path.insert(0, str(benchmark_dir))
    
    # Set up pytest markers for benchmark categorization
    try:
        import pytest
        
        # Define custom markers for benchmark categories
        BENCHMARK_MARKERS = {
            "benchmark": "Performance and benchmark tests (excluded by default)",
            "performance": "Performance validation tests",
            "memory_profiling": "Memory usage and leak detection tests",
            "data_loading": "Data loading performance tests",
            "transformation": "DataFrame transformation performance tests",
            "discovery": "File discovery performance tests",
            "config": "Configuration loading performance tests",
        }
        
        # Register markers if pytest is available
        for marker, description in BENCHMARK_MARKERS.items():
            pytest.mark.__dict__[marker] = pytest.mark.usefixtures(marker)
            
    except ImportError:
        # pytest not available, skip marker setup
        pass


def get_benchmark_runner() -> Optional[Callable]:
    """
    Get the main benchmark runner function for external integration.
    
    Returns:
        Benchmark runner function if available, None if modules not yet created.
    """
    try:
        from .run_benchmarks import main as benchmark_runner
        return benchmark_runner
    except ImportError:
        # Module not yet created during refactoring
        return None


def get_benchmark_utilities() -> Optional[Any]:
    """
    Get benchmark utilities module for external integration.
    
    Returns:
        Benchmark utilities module if available, None if not yet created.
    """
    try:
        from . import utils as benchmark_utils
        return benchmark_utils
    except ImportError:
        # Module not yet created during refactoring
        return None


def get_benchmark_config() -> Optional[Any]:
    """
    Get benchmark configuration module for external integration.
    
    Returns:
        Benchmark configuration module if available, None if not yet created.
    """
    try:
        from . import config as benchmark_config
        return benchmark_config
    except ImportError:
        # Module not yet created during refactoring
        return None


# Initialize package on import
_package_logger = _initialize_logging()
_environment_info = _detect_environment()
_config_defaults = _get_benchmark_config_defaults()
_setup_pytest_discovery()

# Package-level exports (available immediately)
__all__ = [
    # Metadata
    "__version__",
    "__author__", 
    "__description__",
    
    # Configuration and capabilities
    "BENCHMARK_CAPABILITIES",
    "PERFORMANCE_SLA_THRESHOLDS",
    
    # Environment and setup functions
    "_detect_environment",
    "_initialize_logging", 
    "_get_benchmark_config_defaults",
    "_setup_pytest_discovery",
    
    # External integration functions
    "get_benchmark_runner",
    "get_benchmark_utilities", 
    "get_benchmark_config",
    
    # Runtime information
    "_package_logger",
    "_environment_info",
    "_config_defaults",
]

# Dynamic exports (available when modules are created)
__dynamic_exports__ = []

# Attempt to import and expose key functions from dependency modules
# These will be available once the refactoring is complete
try:
    from .run_benchmarks import main as run_benchmarks_main
    from .run_benchmarks import execute_benchmark_category, generate_performance_report
    __dynamic_exports__.extend([
        "run_benchmarks_main",
        "execute_benchmark_category", 
        "generate_performance_report",
    ])
    __all__.extend(__dynamic_exports__[-3:])
except ImportError:
    # Dependencies not yet created, will be available after refactoring
    pass

try:
    from .utils import (
        StatisticalAnalyzer,
        MemoryProfiler, 
        EnvironmentNormalizer,
        ArtifactGenerator,
        RegressionDetector,
    )
    __dynamic_exports__.extend([
        "StatisticalAnalyzer",
        "MemoryProfiler",
        "EnvironmentNormalizer", 
        "ArtifactGenerator",
        "RegressionDetector",
    ])
    __all__.extend(__dynamic_exports__[-5:])
except ImportError:
    # Dependencies not yet created, will be available after refactoring
    pass

try:
    from .config import (
        BenchmarkConfig,
        get_sla_thresholds,
        get_execution_config,
        get_statistical_config,
        get_memory_profiling_config,
    )
    __dynamic_exports__.extend([
        "BenchmarkConfig",
        "get_sla_thresholds",
        "get_execution_config", 
        "get_statistical_config",
        "get_memory_profiling_config",
    ])
    __all__.extend(__dynamic_exports__[-5:])
except ImportError:
    # Dependencies not yet created, will be available after refactoring
    pass

# Log successful package initialization
if _package_logger:
    try:
        _package_logger.info(
            f"FlyRigLoader benchmark framework initialized successfully"
        )
        _package_logger.debug(
            f"Platform: {_environment_info['platform']}, "
            f"Python: {_environment_info['python_version']}, "
            f"CI: {_environment_info['is_ci']}"
        )
        _package_logger.debug(
            f"Available capabilities: {list(BENCHMARK_CAPABILITIES.keys())}"
        )
        if __dynamic_exports__:
            _package_logger.debug(
                f"Dynamic exports available: {__dynamic_exports__}"
            )
        else:
            _package_logger.debug(
                "Dynamic exports will be available after benchmark module creation"
            )
    except Exception as e:
        # Fallback logging if flyrigloader logger fails
        import logging
        logging.getLogger("scripts.benchmarks").warning(
            f"Logging integration issue: {e}, but package initialized successfully"
        )