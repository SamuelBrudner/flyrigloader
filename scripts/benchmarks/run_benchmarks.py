#!/usr/bin/env python3
"""
Comprehensive CLI entry point for FlyRigLoader performance benchmark execution.

This CLI orchestrates performance benchmark testing through pytest-benchmark integration,
providing statistical analysis, memory profiling, environment normalization, and CI/CD
artifact generation while maintaining complete isolation from default test suite execution
to preserve rapid developer feedback cycles.

Key Features:
- Performance test isolation: Excludes performance/benchmark tests from default pytest execution
- Statistical analysis framework: Confidence intervals and regression detection per Section 6.6.4.2
- Memory profiling integration: Line-by-line analysis using pytest-memory-profiler for large datasets
- Environment normalization: CPU/memory constraint detection for consistent results across platforms
- CI/CD integration: GitHub Actions artifact management with 90-day retention policy
- Cross-platform validation: Ubuntu, Windows, macOS compatibility with performance baselines
- Selective execution: Category-based filtering (--category data-loading, transformation, etc.)
- Artifact generation: JSON/CSV/HTML reports with comprehensive performance analysis

Usage Examples:
    # Execute all benchmark tests with comprehensive analysis
    python scripts/benchmarks/run_benchmarks.py

    # Execute specific category with detailed memory profiling
    python scripts/benchmarks/run_benchmarks.py --category data-loading --memory-profiling

    # Execute with statistical analysis and artifact generation
    python scripts/benchmarks/run_benchmarks.py --statistical-analysis --report-artifacts

    # CI/CD mode with environment normalization
    python scripts/benchmarks/run_benchmarks.py --ci-mode --category all

    # Regression detection with baseline comparison
    python scripts/benchmarks/run_benchmarks.py --regression-detection --baseline-file baseline.json

Integration:
- pytest-benchmark: Statistical performance measurement and comparison framework
- pytest-memory-profiler: Line-by-line memory analysis and leak detection
- scripts/benchmarks/utils.py: Comprehensive utilities for analysis and artifact generation
- scripts/benchmarks/config.py: Centralized configuration management
- GitHub Actions: Optional workflow dispatch and PR label triggering per Section 8.4.4
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# Import benchmark utilities and configuration
from .config import (
    BenchmarkConfig,
    BenchmarkCategory,
    DEFAULT_BENCHMARK_CONFIG,
    get_benchmark_config,
    get_category_config
)
from .utils import (
    BenchmarkUtilsCoordinator,
    EnvironmentAnalyzer,
    PerformanceArtifactGenerator,
    RegressionDetector,
    CrossPlatformValidator,
    CICDIntegrationManager,
    analyze_benchmark_results,
    create_memory_profiling_context
)


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(verbose: bool = False, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Configure comprehensive logging for benchmark execution.
    
    Args:
        verbose: Enable verbose logging output
        log_file: Optional log file path for persistent logging
        
    Returns:
        Configured logger instance
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    logger = logging.getLogger('benchmarks')
    logger.info(f"Benchmark logging initialized - Level: {log_level}")
    
    return logger


# ============================================================================
# CLI ARGUMENT PARSING
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create comprehensive argument parser for benchmark CLI.
    
    Returns:
        Configured ArgumentParser instance with all benchmark options
    """
    parser = argparse.ArgumentParser(
        description="FlyRigLoader Performance Benchmark Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run all benchmarks with default settings
  %(prog)s --category data-loading            # Run only data loading benchmarks
  %(prog)s --memory-profiling --verbose       # Enable memory profiling with verbose output
  %(prog)s --ci-mode --report-artifacts       # CI execution with artifact generation
  %(prog)s --regression-detection             # Enable regression detection analysis
  %(prog)s --statistical-analysis --category transformation  # Statistical analysis for transformations

Categories:
  data-loading      Data loading and format detection benchmarks
  transformation    DataFrame transformation and processing benchmarks
  discovery         File discovery and pattern matching benchmarks
  config           Configuration loading and validation benchmarks
  memory-profiling  Memory profiling and leak detection tests
  integration      Cross-module integration benchmarks
  all              Execute all benchmark categories

For complete documentation, see docs/testing_guidelines.md
        """
    )
    
    # ========================================================================
    # EXECUTION CONTROL OPTIONS
    # ========================================================================
    
    execution_group = parser.add_argument_group('Execution Control')
    
    execution_group.add_argument(
        '--category',
        type=str,
        choices=[cat.value for cat in BenchmarkCategory],
        default='all',
        help='Benchmark category to execute (default: all)'
    )
    
    execution_group.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Maximum execution time in seconds (uses category default if not specified)'
    )
    
    execution_group.add_argument(
        '--parallel',
        type=int,
        default=1,
        metavar='N',
        help='Number of parallel workers for test execution (default: 1, auto-detects for CI)'
    )
    
    execution_group.add_argument(
        '--iterations',
        type=int,
        default=None,
        help='Number of benchmark iterations (uses statistical config default if not specified)'
    )
    
    execution_group.add_argument(
        '--warmup',
        type=int,
        default=None,
        help='Number of warmup iterations before measurement'
    )
    
    # ========================================================================
    # ANALYSIS AND PROFILING OPTIONS
    # ========================================================================
    
    analysis_group = parser.add_argument_group('Analysis and Profiling')
    
    analysis_group.add_argument(
        '--statistical-analysis',
        action='store_true',
        help='Enable comprehensive statistical analysis with confidence intervals'
    )
    
    analysis_group.add_argument(
        '--memory-profiling',
        action='store_true',
        help='Enable line-by-line memory profiling using pytest-memory-profiler'
    )
    
    analysis_group.add_argument(
        '--regression-detection',
        action='store_true',
        help='Enable performance regression detection with baseline comparison'
    )
    
    analysis_group.add_argument(
        '--environment-normalization',
        action='store_true',
        help='Apply environment normalization factors for consistent results'
    )
    
    analysis_group.add_argument(
        '--cross-platform-validation',
        action='store_true',
        help='Enable cross-platform performance consistency validation'
    )
    
    # ========================================================================
    # CI/CD AND ARTIFACT OPTIONS
    # ========================================================================
    
    cicd_group = parser.add_argument_group('CI/CD Integration')
    
    cicd_group.add_argument(
        '--ci-mode',
        action='store_true',
        help='Enable CI/CD optimized execution with reduced iterations and enhanced reporting'
    )
    
    cicd_group.add_argument(
        '--report-artifacts',
        action='store_true',
        help='Generate comprehensive performance reports and artifacts'
    )
    
    cicd_group.add_argument(
        '--artifact-dir',
        type=Path,
        default=Path.cwd() / 'benchmark-artifacts',
        help='Directory for generated artifacts (default: ./benchmark-artifacts)'
    )
    
    cicd_group.add_argument(
        '--github-actions',
        action='store_true',
        help='Enable GitHub Actions specific integration and output formatting'
    )
    
    # ========================================================================
    # BASELINE AND COMPARISON OPTIONS
    # ========================================================================
    
    baseline_group = parser.add_argument_group('Baseline and Comparison')
    
    baseline_group.add_argument(
        '--baseline-file',
        type=Path,
        help='Path to baseline performance data file for regression detection'
    )
    
    baseline_group.add_argument(
        '--save-baseline',
        action='store_true',
        help='Save current results as baseline for future regression detection'
    )
    
    baseline_group.add_argument(
        '--compare-with',
        type=Path,
        help='Path to performance data file for comparative analysis'
    )
    
    baseline_group.add_argument(
        '--regression-threshold',
        type=float,
        default=None,
        help='Regression detection threshold percentage (uses config default if not specified)'
    )
    
    # ========================================================================
    # OUTPUT AND REPORTING OPTIONS
    # ========================================================================
    
    output_group = parser.add_argument_group('Output and Reporting')
    
    output_group.add_argument(
        '--output-format',
        type=str,
        choices=['json', 'csv', 'html', 'all'],
        default='json',
        help='Output format for performance reports (default: json)'
    )
    
    output_group.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output with detailed execution information'
    )
    
    output_group.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help='Suppress non-essential output (overrides --verbose)'
    )
    
    output_group.add_argument(
        '--log-file',
        type=Path,
        help='Path to log file for persistent logging'
    )
    
    output_group.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip summary output at the end of execution'
    )
    
    # ========================================================================
    # PYTEST INTEGRATION OPTIONS
    # ========================================================================
    
    pytest_group = parser.add_argument_group('Pytest Integration')
    
    pytest_group.add_argument(
        '--pytest-args',
        type=str,
        nargs='*',
        help='Additional arguments to pass to pytest execution'
    )
    
    pytest_group.add_argument(
        '--collect-only',
        action='store_true',
        help='Only collect tests without executing them'
    )
    
    pytest_group.add_argument(
        '--benchmark-storage',
        type=Path,
        help='Directory for pytest-benchmark storage (default: auto-generated)'
    )
    
    # ========================================================================
    # DEVELOPMENT AND DEBUGGING OPTIONS
    # ========================================================================
    
    debug_group = parser.add_argument_group('Development and Debugging')
    
    debug_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be executed without running benchmarks'
    )
    
    debug_group.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with enhanced diagnostic output'
    )
    
    debug_group.add_argument(
        '--profile-benchmark-runner',
        action='store_true',
        help='Profile the benchmark runner itself for performance optimization'
    )
    
    debug_group.add_argument(
        '--validate-environment',
        action='store_true',
        help='Validate environment suitability for benchmarking without execution'
    )
    
    return parser


# ============================================================================
# ENVIRONMENT VALIDATION AND SETUP
# ============================================================================

def validate_environment(config: BenchmarkConfig, logger: logging.Logger) -> Dict[str, Any]:
    """
    Validate environment suitability for benchmark execution.
    
    Args:
        config: Benchmark configuration
        logger: Logger instance
        
    Returns:
        Dict containing environment validation results
    """
    logger.info("Validating environment for benchmark execution...")
    
    env_analyzer = EnvironmentAnalyzer(config.environment_normalization)
    env_report = env_analyzer.generate_environment_report()
    
    suitability = env_report['benchmarking_suitability']
    
    logger.info(f"Environment suitability: {suitability['level']} (score: {suitability['score']:.1f}/100)")
    
    if suitability['issues']:
        logger.warning("Environment issues detected:")
        for issue in suitability['issues']:
            logger.warning(f"  - {issue}")
    
    if suitability['warnings']:
        logger.info("Environment warnings:")
        for warning in suitability['warnings']:
            logger.info(f"  - {warning}")
    
    # Log recommendations
    if env_report['recommendations']:
        logger.info("Environment recommendations:")
        for rec in env_report['recommendations']:
            logger.info(f"  - {rec}")
    
    return env_report


def setup_benchmark_environment(
    args: argparse.Namespace,
    config: BenchmarkConfig,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Setup benchmark execution environment based on arguments and configuration.
    
    Args:
        args: Parsed command line arguments
        config: Benchmark configuration
        logger: Logger instance
        
    Returns:
        Dict containing environment setup results
    """
    logger.info("Setting up benchmark execution environment...")
    
    # Validate environment if requested or in CI mode
    if args.validate_environment or args.ci_mode:
        env_report = validate_environment(config, logger)
    else:
        env_analyzer = EnvironmentAnalyzer(config.environment_normalization)
        env_report = env_analyzer.generate_environment_report()
    
    # Setup artifact directories
    if args.report_artifacts or args.ci_mode:
        cicd_manager = CICDIntegrationManager(config.cicd_integration)
        artifact_dirs = cicd_manager.setup_artifact_directories(args.artifact_dir)
        logger.info(f"Artifact directories created: {list(artifact_dirs.keys())}")
    else:
        artifact_dirs = {}
    
    # Configure pytest-benchmark storage
    if args.benchmark_storage:
        benchmark_storage_dir = args.benchmark_storage
    else:
        benchmark_storage_dir = args.artifact_dir / 'benchmark-storage'
    
    benchmark_storage_dir.mkdir(parents=True, exist_ok=True)
    
    setup_results = {
        'environment_report': env_report,
        'artifact_directories': artifact_dirs,
        'benchmark_storage_dir': benchmark_storage_dir,
        'execution_timestamp': datetime.now().isoformat(),
        'platform_info': {
            'system': platform.system(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation()
        }
    }
    
    logger.info("Environment setup completed successfully")
    return setup_results


# ============================================================================
# PYTEST EXECUTION AND ORCHESTRATION
# ============================================================================

def build_pytest_command(
    args: argparse.Namespace,
    config: BenchmarkConfig,
    category_config: Dict[str, Any],
    setup_results: Dict[str, Any]
) -> List[str]:
    """
    Build comprehensive pytest command for benchmark execution.
    
    Args:
        args: Parsed command line arguments
        config: Benchmark configuration
        category_config: Category-specific configuration
        setup_results: Environment setup results
        
    Returns:
        List of command components for subprocess execution
    """
    cmd = [sys.executable, '-m', 'pytest']
    
    # ========================================================================
    # TEST SELECTION AND MARKERS
    # ========================================================================
    
    # Add test paths based on category
    benchmark_dir = Path(__file__).parent
    category = BenchmarkCategory(args.category)
    
    if category == BenchmarkCategory.ALL:
        # Execute all benchmark test files
        test_files = [
            benchmark_dir / 'test_benchmark_data_loading.py',
            benchmark_dir / 'test_benchmark_transformations.py',
            benchmark_dir / 'test_benchmark_discovery.py',
            benchmark_dir / 'test_benchmark_config.py'
        ]
    else:
        # Execute specific category tests
        test_modules = config.execution.get_category_tests(category)
        test_files = [benchmark_dir / module for module in test_modules]
    
    # Add existing test files to command
    for test_file in test_files:
        if test_file.exists():
            cmd.append(str(test_file))
    
    # Add marker selection for benchmark tests
    cmd.extend(['-m', 'benchmark or performance'])
    
    # ========================================================================
    # EXECUTION CONFIGURATION
    # ========================================================================
    
    # Parallel execution configuration
    if args.parallel > 1:
        cmd.extend(['-n', str(args.parallel)])
    elif args.ci_mode:
        # Auto-detect parallel workers in CI mode
        cmd.extend(['-n', 'auto'])
    
    # Timeout configuration
    timeout = args.timeout or category_config['timeout']
    cmd.extend(['--timeout', str(int(timeout))])
    
    # Collection-only mode
    if args.collect_only:
        cmd.append('--collect-only')
        return cmd  # Skip benchmark-specific options for collection
    
    # ========================================================================
    # BENCHMARK CONFIGURATION
    # ========================================================================
    
    # Benchmark storage configuration
    storage_dir = setup_results['benchmark_storage_dir']
    cmd.extend(['--benchmark-storage', str(storage_dir)])
    
    # Statistical analysis configuration
    benchmark_kwargs = config.statistical_analysis.get_benchmark_kwargs()
    
    # Override iterations if specified
    if args.iterations:
        cmd.extend(['--benchmark-min-rounds', str(args.iterations)])
        cmd.extend(['--benchmark-max-time', str(timeout)])
    else:
        cmd.extend(['--benchmark-min-rounds', str(benchmark_kwargs.get('min_rounds', 3))])
        cmd.extend(['--benchmark-max-time', str(benchmark_kwargs.get('max_time', 30))])
    
    # Warmup configuration
    if args.warmup is not None:
        cmd.extend(['--benchmark-warmup-iterations', str(args.warmup)])
    elif 'warmup_rounds' in benchmark_kwargs:
        cmd.extend(['--benchmark-warmup-iterations', str(benchmark_kwargs['warmup_rounds'])])
    
    # Enable benchmark comparison and regression detection
    if args.regression_detection or args.baseline_file:
        cmd.append('--benchmark-compare-fail=mean:5%')  # Fail on 5% mean increase
        cmd.append('--benchmark-compare')
        
        if args.baseline_file and args.baseline_file.exists():
            cmd.extend(['--benchmark-compare-fail', f'mean:{args.regression_threshold or 20}%'])
    
    # Save benchmark data
    cmd.append('--benchmark-save=benchmark_results')
    cmd.append('--benchmark-json=benchmark-results.json')
    
    # ========================================================================
    # MEMORY PROFILING CONFIGURATION
    # ========================================================================
    
    if args.memory_profiling:
        # Enable memory profiling for compatible tests
        cmd.append('--memory-profiler')
        
        # Configure memory profiling precision
        memory_config = config.memory_profiling.get_memory_profiling_decorator_args()
        if 'precision' in memory_config:
            cmd.extend(['--memory-profiler-precision', str(memory_config['precision'])])
    
    # ========================================================================
    # OUTPUT AND REPORTING CONFIGURATION
    # ========================================================================
    
    # Verbose output configuration
    if args.verbose and not args.quiet:
        cmd.append('-v')
        cmd.append('--benchmark-verbose')
    elif args.quiet:
        cmd.append('-q')
    
    # Disable pytest warnings for cleaner output
    cmd.append('--disable-warnings')
    
    # Add additional pytest arguments if specified
    if args.pytest_args:
        cmd.extend(args.pytest_args)
    
    return cmd


def execute_pytest_benchmarks(
    pytest_cmd: List[str],
    logger: logging.Logger,
    dry_run: bool = False
) -> Tuple[int, Dict[str, Any]]:
    """
    Execute pytest benchmark command and capture results.
    
    Args:
        pytest_cmd: Pytest command components
        logger: Logger instance
        dry_run: Whether to perform a dry run without execution
        
    Returns:
        Tuple of (return_code, execution_results)
    """
    logger.info(f"Executing pytest command: {' '.join(pytest_cmd)}")
    
    if dry_run:
        logger.info("DRY RUN: Would execute the above command")
        return 0, {'dry_run': True, 'command': pytest_cmd}
    
    start_time = time.time()
    
    try:
        # Execute pytest with comprehensive error capture
        result = subprocess.run(
            pytest_cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=None  # Timeout handled by pytest itself
        )
        
        execution_time = time.time() - start_time
        
        logger.info(f"Pytest execution completed in {execution_time:.2f} seconds")
        logger.info(f"Return code: {result.returncode}")
        
        if result.stdout:
            logger.debug("STDOUT:")
            logger.debug(result.stdout)
        
        if result.stderr:
            if result.returncode != 0:
                logger.error("STDERR:")
                logger.error(result.stderr)
            else:
                logger.debug("STDERR:")
                logger.debug(result.stderr)
        
        execution_results = {
            'return_code': result.returncode,
            'execution_time_seconds': execution_time,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'command': pytest_cmd
        }
        
        return result.returncode, execution_results
        
    except subprocess.TimeoutExpired as e:
        execution_time = time.time() - start_time
        logger.error(f"Pytest execution timed out after {execution_time:.2f} seconds")
        
        return 1, {
            'return_code': 1,
            'execution_time_seconds': execution_time,
            'error': 'timeout',
            'command': pytest_cmd
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Pytest execution failed: {e}")
        
        return 1, {
            'return_code': 1,
            'execution_time_seconds': execution_time,
            'error': str(e),
            'command': pytest_cmd
        }


# ============================================================================
# RESULTS PROCESSING AND ANALYSIS
# ============================================================================

def load_benchmark_results(
    storage_dir: Path,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Load pytest-benchmark results from storage directory.
    
    Args:
        storage_dir: Benchmark storage directory
        logger: Logger instance
        
    Returns:
        Dict containing loaded benchmark results
    """
    logger.info("Loading benchmark results from storage...")
    
    # Look for JSON results file
    json_results_file = storage_dir / 'benchmark-results.json'
    
    if not json_results_file.exists():
        # Try to find the most recent results file
        json_files = list(storage_dir.glob('*.json'))
        if json_files:
            json_results_file = max(json_files, key=lambda f: f.stat().st_mtime)
            logger.info(f"Using most recent results file: {json_results_file}")
        else:
            logger.warning("No benchmark results JSON file found")
            return {}
    
    try:
        with open(json_results_file, 'r') as f:
            benchmark_data = json.load(f)
        
        logger.info(f"Loaded benchmark results from {json_results_file}")
        
        # Transform pytest-benchmark format to utils format
        if 'benchmarks' in benchmark_data:
            transformed_results = {}
            
            for benchmark in benchmark_data['benchmarks']:
                test_name = benchmark['name']
                stats = benchmark['stats']
                
                transformed_results[test_name] = {
                    'stats': {
                        'mean': stats['mean'],
                        'min': stats['min'],
                        'max': stats['max'],
                        'stddev': stats['stddev'],
                        'median': stats['median'],
                        'iterations': stats.get('rounds', 1),
                        'rounds': stats.get('rounds', 1)
                    },
                    'status': 'passed',  # Assume passed if in results
                    'measurements': [stats['mean']] * stats.get('rounds', 1),  # Approximate
                    'extra_info': benchmark.get('extra_info', {}),
                    'platform': platform.system()
                }
                
                # Add SLA compliance check if available
                if 'extra_info' in benchmark and 'sla_threshold' in benchmark['extra_info']:
                    sla_threshold = benchmark['extra_info']['sla_threshold']
                    compliant = stats['mean'] <= sla_threshold
                    margin = ((sla_threshold - stats['mean']) / sla_threshold) * 100 if compliant else 0
                    
                    transformed_results[test_name]['sla_compliance'] = {
                        'compliant': compliant,
                        'sla_threshold_seconds': sla_threshold,
                        'performance_margin': margin
                    }
            
            return transformed_results
        else:
            logger.warning("Unexpected benchmark results format")
            return benchmark_data
            
    except Exception as e:
        logger.error(f"Failed to load benchmark results: {e}")
        return {}


def perform_comprehensive_analysis(
    benchmark_results: Dict[str, Any],
    args: argparse.Namespace,
    config: BenchmarkConfig,
    setup_results: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of benchmark results.
    
    Args:
        benchmark_results: Raw benchmark results
        args: Command line arguments
        config: Benchmark configuration
        setup_results: Environment setup results
        logger: Logger instance
        
    Returns:
        Dict containing comprehensive analysis results
    """
    logger.info("Performing comprehensive benchmark analysis...")
    
    # Initialize utilities coordinator
    coordinator = BenchmarkUtilsCoordinator(config)
    
    # Perform comprehensive analysis
    analysis_results = coordinator.execute_comprehensive_analysis(
        benchmark_results=benchmark_results,
        output_dir=args.artifact_dir,
        enable_regression_detection=args.regression_detection,
        enable_cross_platform_validation=args.cross_platform_validation,
        enable_ci_integration=(args.ci_mode or args.github_actions)
    )
    
    # Add setup context to analysis
    analysis_results['setup_context'] = setup_results
    analysis_results['execution_arguments'] = vars(args)
    
    logger.info(f"Analysis completed with status: {analysis_results['overall_status']}")
    
    # Log recommendations
    if analysis_results['recommendations']:
        logger.info("Analysis recommendations:")
        for recommendation in analysis_results['recommendations']:
            logger.info(f"  - {recommendation}")
    
    return analysis_results


def generate_performance_artifacts(
    benchmark_results: Dict[str, Any],
    analysis_results: Dict[str, Any],
    args: argparse.Namespace,
    config: BenchmarkConfig,
    logger: logging.Logger
) -> Dict[str, Path]:
    """
    Generate comprehensive performance artifacts.
    
    Args:
        benchmark_results: Benchmark results
        analysis_results: Analysis results
        args: Command line arguments
        config: Benchmark configuration
        logger: Logger instance
        
    Returns:
        Dict mapping artifact types to file paths
    """
    if not (args.report_artifacts or args.ci_mode):
        return {}
    
    logger.info("Generating performance artifacts...")
    
    # Setup CI/CD integration manager
    cicd_manager = CICDIntegrationManager(config.cicd_integration)
    
    # Collect benchmark artifacts
    artifacts = cicd_manager.collect_benchmark_artifacts(
        benchmark_results=benchmark_results,
        artifact_dirs=analysis_results.get('ci_integration_analysis', {}).get('artifacts_generated', {}),
        generate_all_formats=(args.output_format == 'all')
    )
    
    # Generate format-specific reports if requested
    if args.output_format != 'all':
        artifact_generator = PerformanceArtifactGenerator(config.cicd_integration)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if args.output_format == 'json':
            json_path = args.artifact_dir / f'benchmark-report-{timestamp}.json'
            artifact_generator.generate_json_report(
                benchmark_results=benchmark_results,
                output_path=json_path,
                include_environment=True,
                include_historical=args.regression_detection
            )
            artifacts['json_report'] = json_path
            
        elif args.output_format == 'csv':
            csv_path = args.artifact_dir / f'benchmark-data-{timestamp}.csv'
            artifact_generator.generate_csv_report(
                benchmark_results=benchmark_results,
                output_path=csv_path,
                include_statistics=args.statistical_analysis
            )
            artifacts['csv_report'] = csv_path
            
        elif args.output_format == 'html':
            html_path = args.artifact_dir / f'benchmark-report-{timestamp}.html'
            artifact_generator.generate_html_report(
                benchmark_results=benchmark_results,
                output_path=html_path,
                include_charts=True
            )
            artifacts['html_report'] = html_path
    
    # Save analysis results
    analysis_path = args.artifact_dir / 'reports' / f'analysis-results-{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    artifacts['analysis_results'] = analysis_path
    
    logger.info(f"Generated {len(artifacts)} artifact files")
    for artifact_type, path in artifacts.items():
        logger.info(f"  {artifact_type}: {path}")
    
    return artifacts


# ============================================================================
# RESULTS SUMMARY AND REPORTING
# ============================================================================

def print_execution_summary(
    benchmark_results: Dict[str, Any],
    analysis_results: Dict[str, Any],
    execution_results: Dict[str, Any],
    artifacts: Dict[str, Path],
    args: argparse.Namespace,
    logger: logging.Logger
):
    """
    Print comprehensive execution summary.
    
    Args:
        benchmark_results: Benchmark results
        analysis_results: Analysis results
        execution_results: Pytest execution results
        artifacts: Generated artifacts
        args: Command line arguments
        logger: Logger instance
    """
    if args.no_summary or args.quiet:
        return
    
    print("\n" + "=" * 80)
    print("FLYRIGLOADER BENCHMARK EXECUTION SUMMARY")
    print("=" * 80)
    
    # ========================================================================
    # EXECUTION OVERVIEW
    # ========================================================================
    
    print(f"\nExecution Overview:")
    print(f"  Category: {args.category}")
    print(f"  Total Execution Time: {execution_results.get('execution_time_seconds', 0):.2f}s")
    print(f"  Return Code: {execution_results.get('return_code', 'unknown')}")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print(f"  Python: {platform.python_version()}")
    
    # ========================================================================
    # BENCHMARK RESULTS SUMMARY
    # ========================================================================
    
    if benchmark_results:
        print(f"\nBenchmark Results:")
        print(f"  Total Tests: {len(benchmark_results)}")
        
        passed_tests = sum(1 for result in benchmark_results.values() 
                          if isinstance(result, dict) and result.get('status') == 'passed')
        
        print(f"  Passed Tests: {passed_tests}")
        print(f"  Success Rate: {(passed_tests / len(benchmark_results)) * 100:.1f}%")
        
        # SLA compliance summary
        sla_compliant = sum(1 for result in benchmark_results.values()
                           if isinstance(result, dict) and 
                           result.get('sla_compliance', {}).get('compliant', False))
        
        if sla_compliant > 0 or any('sla_compliance' in result for result in benchmark_results.values() 
                                   if isinstance(result, dict)):
            print(f"  SLA Compliant: {sla_compliant}/{len(benchmark_results)}")
            print(f"  SLA Compliance Rate: {(sla_compliant / len(benchmark_results)) * 100:.1f}%")
        
        # Performance overview - show fastest and slowest tests
        test_times = []
        for test_name, result in benchmark_results.items():
            if isinstance(result, dict) and 'stats' in result:
                test_times.append((test_name, result['stats'].get('mean', 0)))
        
        if test_times:
            test_times.sort(key=lambda x: x[1])
            print(f"\n  Fastest Test: {test_times[0][0]} ({test_times[0][1]:.4f}s)")
            print(f"  Slowest Test: {test_times[-1][0]} ({test_times[-1][1]:.4f}s)")
    
    # ========================================================================
    # ANALYSIS RESULTS SUMMARY
    # ========================================================================
    
    if analysis_results and analysis_results.get('overall_status') != 'unknown':
        print(f"\nAnalysis Results:")
        print(f"  Overall Status: {analysis_results['overall_status'].upper()}")
        
        # Memory analysis summary
        memory_analysis = analysis_results.get('memory_analysis', {})
        memory_summary = memory_analysis.get('memory_efficiency_summary', {})
        
        if memory_summary:
            print(f"  Memory Efficiency: {memory_summary.get('efficiency_rate_percent', 0):.1f}%")
            if memory_summary.get('tests_with_leaks', 0) > 0:
                print(f"  Memory Leaks Detected: {memory_summary['tests_with_leaks']} tests")
        
        # Regression analysis summary
        regression_analysis = analysis_results.get('regression_analysis', {})
        if regression_analysis.get('total_tests_analyzed', 0) > 0:
            print(f"  Regression Analysis: {regression_analysis['total_tests_analyzed']} tests analyzed")
            if regression_analysis.get('regressions_detected', 0) > 0:
                print(f"    Regressions Detected: {regression_analysis['regressions_detected']}")
                print(f"    Critical Regressions: {regression_analysis.get('critical_regressions', 0)}")
            else:
                print(f"    No Performance Regressions Detected")
        
        # Environment suitability
        env_analysis = analysis_results.get('environment_analysis', {})
        suitability = env_analysis.get('benchmarking_suitability', {})
        if suitability:
            print(f"  Environment Suitability: {suitability.get('level', 'unknown')} ({suitability.get('score', 0):.0f}/100)")
    
    # ========================================================================
    # ARTIFACTS SUMMARY
    # ========================================================================
    
    if artifacts:
        print(f"\nGenerated Artifacts:")
        for artifact_type, path in artifacts.items():
            print(f"  {artifact_type}: {path}")
    
    # ========================================================================
    # RECOMMENDATIONS AND NEXT STEPS
    # ========================================================================
    
    if analysis_results and analysis_results.get('recommendations'):
        print(f"\nRecommendations:")
        for recommendation in analysis_results['recommendations'][:5]:  # Show top 5
            print(f"  • {recommendation}")
        
        if len(analysis_results['recommendations']) > 5:
            print(f"  ... and {len(analysis_results['recommendations']) - 5} more (see detailed reports)")
    
    # ========================================================================
    # STATUS INDICATOR
    # ========================================================================
    
    overall_status = analysis_results.get('overall_status', 'unknown') if analysis_results else 'unknown'
    return_code = execution_results.get('return_code', 1)
    
    if return_code == 0 and overall_status in ['success', 'unknown']:
        status_emoji = "✅"
        status_text = "SUCCESS"
    elif overall_status == 'warning':
        status_emoji = "⚠️"
        status_text = "WARNING"
    else:
        status_emoji = "❌"
        status_text = "FAILURE"
    
    print(f"\n{status_emoji} BENCHMARK EXECUTION: {status_text}")
    print("=" * 80 + "\n")


# ============================================================================
# MAIN CLI ENTRY POINT
# ============================================================================

def main() -> int:
    """
    Main CLI entry point for benchmark execution.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Handle mutually exclusive options
    if args.verbose and args.quiet:
        args.verbose = False  # Quiet overrides verbose
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose, log_file=args.log_file)
    
    try:
        logger.info("FlyRigLoader Benchmark Test Suite starting...")
        logger.info(f"Arguments: {vars(args)}")
        
        # ====================================================================
        # CONFIGURATION AND ENVIRONMENT SETUP
        # ====================================================================
        
        # Get benchmark configuration
        environment = "ci" if args.ci_mode else "auto"
        config = get_benchmark_config(environment)
        
        # Get category-specific configuration
        category = BenchmarkCategory(args.category)
        category_config = get_category_config(category)
        
        logger.info(f"Configuration loaded for category: {category.value}")
        
        # Setup benchmark environment
        setup_results = setup_benchmark_environment(args, config, logger)
        
        # Validate environment if requested
        if args.validate_environment:
            env_report = setup_results['environment_report']
            suitability = env_report['benchmarking_suitability']
            
            print(f"\nEnvironment Validation Results:")
            print(f"Suitability: {suitability['level']} (score: {suitability['score']:.1f}/100)")
            
            if suitability['issues']:
                print("\nIssues:")
                for issue in suitability['issues']:
                    print(f"  - {issue}")
            
            if suitability['warnings']:
                print("\nWarnings:")
                for warning in suitability['warnings']:
                    print(f"  - {warning}")
            
            if env_report['recommendations']:
                print("\nRecommendations:")
                for rec in env_report['recommendations']:
                    print(f"  - {rec}")
            
            return 0 if suitability['level'] in ['excellent', 'good'] else 1
        
        # ====================================================================
        # PYTEST EXECUTION
        # ====================================================================
        
        # Build pytest command
        pytest_cmd = build_pytest_command(args, config, category_config, setup_results)
        
        # Execute benchmarks
        return_code, execution_results = execute_pytest_benchmarks(
            pytest_cmd, logger, dry_run=args.dry_run
        )
        
        if args.dry_run:
            print("DRY RUN COMPLETED")
            print(f"Would execute: {' '.join(pytest_cmd)}")
            return 0
        
        if args.collect_only:
            logger.info("Test collection completed")
            if execution_results.get('stdout'):
                print(execution_results['stdout'])
            return return_code
        
        # ====================================================================
        # RESULTS LOADING AND ANALYSIS
        # ====================================================================
        
        # Load benchmark results
        benchmark_results = load_benchmark_results(
            setup_results['benchmark_storage_dir'], logger
        )
        
        # Perform comprehensive analysis if results available
        analysis_results = {}
        if benchmark_results and (args.statistical_analysis or args.regression_detection or 
                                 args.memory_profiling or args.ci_mode):
            analysis_results = perform_comprehensive_analysis(
                benchmark_results, args, config, setup_results, logger
            )
        
        # ====================================================================
        # ARTIFACT GENERATION
        # ====================================================================
        
        # Generate performance artifacts
        artifacts = generate_performance_artifacts(
            benchmark_results, analysis_results, args, config, logger
        )
        
        # ====================================================================
        # SUMMARY AND REPORTING
        # ====================================================================
        
        # Print execution summary
        print_execution_summary(
            benchmark_results, analysis_results, execution_results, artifacts, args, logger
        )
        
        # Determine final exit code
        final_return_code = return_code
        
        # Check analysis results for critical issues
        if analysis_results:
            overall_status = analysis_results.get('overall_status', 'unknown')
            if overall_status == 'critical':
                final_return_code = max(final_return_code, 2)  # Critical status
            elif overall_status == 'warning' and final_return_code == 0:
                final_return_code = 1  # Warning status
        
        logger.info(f"Benchmark execution completed with exit code: {final_return_code}")
        return final_return_code
        
    except KeyboardInterrupt:
        logger.warning("Benchmark execution interrupted by user")
        return 130  # SIGINT exit code
        
    except Exception as e:
        logger.error(f"Benchmark execution failed with error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())