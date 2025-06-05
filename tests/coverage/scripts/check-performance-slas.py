#!/usr/bin/env python3
"""
FlyRigLoader Performance SLA Validation Script

Comprehensive performance SLA validation implementing automated benchmark analysis against 
defined service level agreements with regression detection and quality gate enforcement.

This script validates performance requirements per TST-PERF-001 and TST-PERF-002 specifications,
integrates with pytest-benchmark results for statistical analysis, and enforces automated 
quality gates for CI/CD pipeline integration.

Requirements Implementation:
- TST-PERF-001: Data loading SLA validation within 1s per 100MB
- TST-PERF-002: DataFrame transformation SLA validation within 500ms per 1M rows
- TST-PERF-003: Benchmark reporting integration with statistical performance reports
- Section 0.2.5: Performance regression detection per Infrastructure Updates
- Section 4.1.1.5: Test execution workflow with automated quality gate enforcement

Author: FlyRigLoader Test Infrastructure Team
Created: 2024-12-19
Last Updated: 2024-12-19
Version: 1.0.0
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import tempfile
import os
import traceback

# Performance analysis and statistical computing
import numpy as np

# Data structures for validation results
from collections import defaultdict, namedtuple

# YAML configuration loading for quality gates
try:
    import yaml
except ImportError:
    print("WARNING: PyYAML not available. YAML configuration loading disabled.")
    yaml = None


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Performance SLA thresholds per technical specifications
SLA_THRESHOLDS = {
    # TST-PERF-001: Data loading SLA validation within 1s per 100MB
    'data_loading': {
        'max_time_per_100mb': 1.0,  # seconds
        'tolerance_percentage': 10.0,  # 10% tolerance for system variance
        'statistical_confidence': 0.95,  # 95% confidence interval
        'minimum_sample_size': 3,  # Minimum benchmark runs required
        'description': 'Data loading SLA validation within 1s per 100MB per TST-PERF-001'
    },
    
    # TST-PERF-002: DataFrame transformation SLA validation within 500ms per 1M rows
    'data_transformation': {
        'max_time_per_million_rows': 0.5,  # seconds (500ms)
        'tolerance_percentage': 15.0,  # 15% tolerance for data complexity variance
        'statistical_confidence': 0.95,  # 95% confidence interval
        'minimum_sample_size': 3,  # Minimum benchmark runs required
        'description': 'DataFrame transformation SLA validation within 500ms per 1M rows per TST-PERF-002'
    },
    
    # General performance thresholds
    'format_detection': {
        'max_overhead_ms': 100.0,  # 100ms maximum detection overhead
        'tolerance_percentage': 20.0,  # 20% tolerance for filesystem variance
        'description': 'Format detection overhead validation per F-003-RQ-004'
    },
    
    # Performance regression detection thresholds
    'regression_detection': {
        'max_performance_degradation': 20.0,  # 20% maximum allowed degradation
        'trend_analysis_window': 10,  # Analyze last 10 builds for trends
        'baseline_comparison_enabled': True,
        'description': 'Performance regression detection per Section 0.2.5'
    }
}

# Quality gate enforcement configuration
QUALITY_GATE_CONFIG = {
    'fail_on_sla_violation': True,
    'block_merge_on_performance_regression': True,
    'require_statistical_significance': True,
    'generate_detailed_reports': True,
    'machine_readable_output': True
}

# Benchmark categories and their mappings
BENCHMARK_CATEGORIES = {
    'data_loading_sla': 'data_loading',
    'format_detection': 'format_detection', 
    'scaling_analysis': 'data_loading',
    'transformation_benchmarks': 'data_transformation',
    'benchmark_transformations': 'data_transformation'
}


# ============================================================================
# DATA STRUCTURES AND MODELS
# ============================================================================

@dataclass
class PerformanceMetric:
    """Individual performance measurement with statistical properties."""
    name: str
    execution_time: float
    expected_max_time: float
    sla_category: str
    data_size_mb: Optional[float] = None
    row_count: Optional[int] = None
    sample_count: int = 1
    std_deviation: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    passes_sla: bool = False
    margin_percentage: float = 0.0
    error_message: Optional[str] = None


@dataclass
class BenchmarkAnalysis:
    """Comprehensive analysis of benchmark results with SLA validation."""
    benchmark_name: str
    group: str
    category: str
    metrics: List[PerformanceMetric]
    statistical_summary: Dict[str, float]
    sla_compliance: bool
    violation_details: List[str]
    recommendations: List[str]
    baseline_comparison: Optional[Dict[str, Any]] = None


@dataclass 
class PerformanceReport:
    """Complete performance validation report with quality gate status."""
    timestamp: str
    validation_status: str  # 'PASS', 'FAIL', 'WARNING'
    total_benchmarks: int
    passing_benchmarks: int
    failing_benchmarks: int
    sla_compliance_rate: float
    benchmark_analyses: List[BenchmarkAnalysis]
    quality_gate_decision: str
    performance_trends: Dict[str, Any]
    recommendations: List[str]
    execution_summary: Dict[str, Any]


class PerformanceSLAValidator:
    """
    Comprehensive performance SLA validation engine.
    
    Validates pytest-benchmark results against defined Service Level Agreements
    with statistical analysis, regression detection, and quality gate enforcement.
    """
    
    def __init__(
        self, 
        quality_gates_config_path: Optional[Path] = None,
        enable_regression_detection: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the SLA validator with configuration and analysis settings.
        
        Args:
            quality_gates_config_path: Path to quality gates YAML configuration
            enable_regression_detection: Whether to perform regression analysis
            verbose: Enable detailed logging output
        """
        self.quality_gates_config_path = quality_gates_config_path
        self.enable_regression_detection = enable_regression_detection
        self.verbose = verbose
        
        # Load and merge configuration
        self.sla_config = self._load_configuration()
        
        # Performance tracking and analysis
        self.benchmark_analyses: List[BenchmarkAnalysis] = []
        self.violation_count = 0
        self.total_benchmarks = 0
        
        # Historical data for regression detection
        self.historical_baseline: Dict[str, Any] = {}
        self.performance_trends: Dict[str, List[float]] = defaultdict(list)
        
        if self.verbose:
            print(f"[SLA-VALIDATOR] Initialized with configuration: {self.sla_config.keys()}")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """
        Load and merge SLA configuration from multiple sources.
        
        Loads configuration from:
        1. Default SLA thresholds (built-in)
        2. Quality gates YAML configuration (if provided)
        3. Environment variables for CI/CD integration
        
        Returns:
            Merged configuration dictionary with SLA thresholds and enforcement rules
        """
        # Start with default configuration
        config = {
            'sla_thresholds': SLA_THRESHOLDS.copy(),
            'quality_gates': QUALITY_GATE_CONFIG.copy(),
            'benchmark_categories': BENCHMARK_CATEGORIES.copy()
        }
        
        # Load quality gates configuration if provided
        if self.quality_gates_config_path and self.quality_gates_config_path.exists():
            try:
                if yaml is None:
                    print("WARNING: Cannot load YAML configuration - PyYAML not available")
                else:
                    with open(self.quality_gates_config_path, 'r') as f:
                        quality_gates_config = yaml.safe_load(f)
                    
                    # Merge performance configuration
                    if 'performance' in quality_gates_config:
                        perf_config = quality_gates_config['performance']
                        
                        # Update SLA categories with configuration values
                        if 'sla_categories' in perf_config:
                            for category, settings in perf_config['sla_categories'].items():
                                if category in config['sla_thresholds']:
                                    config['sla_thresholds'][category].update(settings)
                        
                        # Update enforcement configuration
                        if 'enforcement' in perf_config:
                            config['quality_gates'].update(perf_config['enforcement'])
                
                if self.verbose:
                    print(f"[SLA-VALIDATOR] Loaded quality gates configuration from {self.quality_gates_config_path}")
                    
            except Exception as e:
                print(f"WARNING: Failed to load quality gates configuration: {e}")
        
        # Override with environment variables for CI/CD integration
        env_overrides = {
            'SLA_DATA_LOADING_MAX_TIME': ('sla_thresholds', 'data_loading', 'max_time_per_100mb'),
            'SLA_TRANSFORMATION_MAX_TIME': ('sla_thresholds', 'data_transformation', 'max_time_per_million_rows'),
            'SLA_FAIL_ON_VIOLATION': ('quality_gates', 'fail_on_sla_violation'),
            'SLA_REGRESSION_THRESHOLD': ('sla_thresholds', 'regression_detection', 'max_performance_degradation')
        }
        
        for env_var, config_path in env_overrides.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    # Navigate nested configuration
                    current = config
                    for key in config_path[:-1]:
                        current = current[key]
                    
                    # Parse value based on type
                    if isinstance(current[config_path[-1]], bool):
                        current[config_path[-1]] = env_value.lower() in ('true', '1', 'yes')
                    elif isinstance(current[config_path[-1]], (int, float)):
                        current[config_path[-1]] = float(env_value)
                    else:
                        current[config_path[-1]] = env_value
                        
                    if self.verbose:
                        print(f"[SLA-VALIDATOR] Override from {env_var}: {config_path} = {env_value}")
                        
                except (KeyError, ValueError) as e:
                    print(f"WARNING: Invalid environment override {env_var}={env_value}: {e}")
        
        return config
    
    def load_benchmark_results(self, results_path: Path) -> Dict[str, Any]:
        """
        Load and parse pytest-benchmark results from JSON file.
        
        Args:
            results_path: Path to pytest-benchmark JSON results file
            
        Returns:
            Parsed benchmark results dictionary
            
        Raises:
            FileNotFoundError: If results file doesn't exist
            ValueError: If results file is invalid or corrupted
        """
        if not results_path.exists():
            raise FileNotFoundError(f"Benchmark results file not found: {results_path}")
        
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            if self.verbose:
                benchmark_count = len(results.get('benchmarks', []))
                print(f"[SLA-VALIDATOR] Loaded {benchmark_count} benchmark results from {results_path}")
            
            # Validate results structure
            if 'benchmarks' not in results:
                raise ValueError("Invalid benchmark results: missing 'benchmarks' key")
            
            return results
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in benchmark results file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load benchmark results: {e}")
    
    def analyze_benchmark(self, benchmark_data: Dict[str, Any]) -> BenchmarkAnalysis:
        """
        Analyze individual benchmark against SLA requirements.
        
        Performs comprehensive analysis including:
        - SLA compliance validation
        - Statistical significance assessment
        - Performance scaling analysis
        - Regression detection (if enabled)
        
        Args:
            benchmark_data: Individual benchmark result from pytest-benchmark
            
        Returns:
            Comprehensive benchmark analysis with SLA validation results
        """
        benchmark_name = benchmark_data.get('name', 'unknown')
        group = benchmark_data.get('group', 'default')
        
        # Determine SLA category based on benchmark group/name
        category = self._determine_sla_category(benchmark_name, group)
        
        # Extract statistical data
        stats = benchmark_data.get('stats', {})
        
        # Create performance metrics
        metrics = self._extract_performance_metrics(benchmark_data, category)
        
        # Calculate statistical summary
        statistical_summary = self._calculate_statistical_summary(stats)
        
        # Validate SLA compliance
        sla_compliance, violation_details = self._validate_sla_compliance(metrics, category)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, violation_details, category)
        
        # Perform baseline comparison if regression detection enabled
        baseline_comparison = None
        if self.enable_regression_detection:
            baseline_comparison = self._compare_with_baseline(benchmark_name, statistical_summary)
        
        analysis = BenchmarkAnalysis(
            benchmark_name=benchmark_name,
            group=group,
            category=category,
            metrics=metrics,
            statistical_summary=statistical_summary,
            sla_compliance=sla_compliance,
            violation_details=violation_details,
            recommendations=recommendations,
            baseline_comparison=baseline_comparison
        )
        
        if self.verbose:
            status = "PASS" if sla_compliance else "FAIL"
            print(f"[SLA-VALIDATOR] {benchmark_name}: {status} ({category} category)")
        
        return analysis
    
    def _determine_sla_category(self, benchmark_name: str, group: str) -> str:
        """
        Determine SLA category based on benchmark name and group.
        
        Maps benchmark tests to their corresponding SLA categories for validation.
        
        Args:
            benchmark_name: Name of the benchmark test
            group: Benchmark group classification
            
        Returns:
            SLA category string for threshold lookup
        """
        # Check direct group mapping first
        if group in self.sla_config['benchmark_categories']:
            return self.sla_config['benchmark_categories'][group]
        
        # Check name-based patterns for more specific categorization
        name_patterns = {
            'data_loading': ['load', 'pickle', 'file', 'read'],
            'data_transformation': ['transform', 'dataframe', 'convert', 'make_dataframe'],
            'format_detection': ['detection', 'format', 'auto_detect']
        }
        
        benchmark_name_lower = benchmark_name.lower()
        for category, patterns in name_patterns.items():
            if any(pattern in benchmark_name_lower for pattern in patterns):
                return category
        
        # Default to data_loading if no specific match
        return 'data_loading'
    
    def _extract_performance_metrics(
        self, 
        benchmark_data: Dict[str, Any], 
        category: str
    ) -> List[PerformanceMetric]:
        """
        Extract performance metrics from benchmark data with SLA validation.
        
        Args:
            benchmark_data: Raw benchmark result data
            category: SLA category for threshold lookup
            
        Returns:
            List of performance metrics with SLA compliance status
        """
        stats = benchmark_data.get('stats', {})
        params = benchmark_data.get('params', {})
        
        # Extract execution time statistics
        mean_time = stats.get('mean', 0.0)
        std_dev = stats.get('stddev', 0.0)
        rounds = stats.get('rounds', 1)
        
        # Calculate confidence interval
        confidence_interval = None
        if rounds > 1 and std_dev > 0:
            margin = 1.96 * (std_dev / np.sqrt(rounds))  # 95% confidence
            confidence_interval = (mean_time - margin, mean_time + margin)
        
        # Extract data size information for SLA calculation
        data_size_mb = self._extract_data_size(params, benchmark_data)
        row_count = self._extract_row_count(params, benchmark_data)
        
        # Calculate expected maximum time based on SLA
        expected_max_time = self._calculate_expected_max_time(category, data_size_mb, row_count)
        
        # Determine SLA compliance
        sla_config = self.sla_config['sla_thresholds'].get(category, {})
        tolerance = sla_config.get('tolerance_percentage', 10.0) / 100.0
        adjusted_max_time = expected_max_time * (1 + tolerance)
        
        passes_sla = mean_time <= adjusted_max_time
        margin_percentage = ((mean_time - expected_max_time) / expected_max_time) * 100.0
        
        # Generate error message if SLA violation
        error_message = None
        if not passes_sla:
            error_message = (
                f"SLA violation: {mean_time:.3f}s exceeds {adjusted_max_time:.3f}s "
                f"(+{tolerance*100:.1f}% tolerance) for {category} category"
            )
        
        metric = PerformanceMetric(
            name=benchmark_data.get('name', 'unknown'),
            execution_time=mean_time,
            expected_max_time=expected_max_time,
            sla_category=category,
            data_size_mb=data_size_mb,
            row_count=row_count,
            sample_count=rounds,
            std_deviation=std_dev,
            confidence_interval=confidence_interval,
            passes_sla=passes_sla,
            margin_percentage=margin_percentage,
            error_message=error_message
        )
        
        return [metric]
    
    def _extract_data_size(self, params: Dict[str, Any], benchmark_data: Dict[str, Any]) -> Optional[float]:
        """Extract data size in MB from benchmark parameters."""
        # Check various parameter names that might contain size information
        size_keys = ['size_mb', 'data_size_mb', 'file_size_mb', 'size_name']
        
        for key in size_keys:
            if key in params:
                value = params[key]
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    # Try to extract numeric value from string (e.g., "medium_10mb" -> 10.0)
                    import re
                    match = re.search(r'(\d+(?:\.\d+)?)mb', value.lower())
                    if match:
                        return float(match.group(1))
        
        # Fallback: try to estimate from benchmark name
        benchmark_name = benchmark_data.get('name', '').lower()
        import re
        match = re.search(r'(\d+(?:\.\d+)?)mb', benchmark_name)
        if match:
            return float(match.group(1))
        
        return None
    
    def _extract_row_count(self, params: Dict[str, Any], benchmark_data: Dict[str, Any]) -> Optional[int]:
        """Extract row count from benchmark parameters for transformation SLA validation."""
        # Check for row count parameters
        row_keys = ['row_count', 'rows', 'n_rows', 'num_rows']
        
        for key in row_keys:
            if key in params:
                value = params[key]
                if isinstance(value, (int, float)):
                    return int(value)
        
        # Fallback: estimate from data size for transformation benchmarks
        # Typical experimental data: ~100 bytes per row
        data_size_mb = self._extract_data_size(params, benchmark_data)
        if data_size_mb:
            estimated_rows = int((data_size_mb * 1024 * 1024) / 100)
            return estimated_rows
        
        return None
    
    def _calculate_expected_max_time(
        self, 
        category: str, 
        data_size_mb: Optional[float], 
        row_count: Optional[int]
    ) -> float:
        """
        Calculate expected maximum execution time based on SLA category and data size.
        
        Args:
            category: SLA category ('data_loading', 'data_transformation', etc.)
            data_size_mb: Data size in megabytes
            row_count: Number of rows for transformation operations
            
        Returns:
            Expected maximum execution time in seconds
        """
        sla_config = self.sla_config['sla_thresholds'].get(category, {})
        
        if category == 'data_loading':
            # TST-PERF-001: 1 second per 100MB
            max_time_per_100mb = sla_config.get('max_time_per_100mb', 1.0)
            if data_size_mb:
                return (data_size_mb / 100.0) * max_time_per_100mb
            else:
                # Default assumption for unknown size
                return max_time_per_100mb
                
        elif category == 'data_transformation':
            # TST-PERF-002: 500ms per 1M rows
            max_time_per_million = sla_config.get('max_time_per_million_rows', 0.5)
            if row_count:
                return (row_count / 1_000_000) * max_time_per_million
            elif data_size_mb:
                # Estimate rows from data size (100 bytes per row average)
                estimated_rows = (data_size_mb * 1024 * 1024) / 100
                return (estimated_rows / 1_000_000) * max_time_per_million
            else:
                # Default assumption
                return max_time_per_million
                
        elif category == 'format_detection':
            # F-003-RQ-004: 100ms detection overhead
            return sla_config.get('max_overhead_ms', 100.0) / 1000.0
        
        else:
            # Default: 1 second for unknown categories
            return 1.0
    
    def _calculate_statistical_summary(self, stats: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive statistical summary from benchmark stats."""
        summary = {
            'mean': stats.get('mean', 0.0),
            'median': stats.get('median', 0.0),
            'min': stats.get('min', 0.0),
            'max': stats.get('max', 0.0),
            'stddev': stats.get('stddev', 0.0),
            'rounds': stats.get('rounds', 1),
            'iterations': stats.get('iterations', 1)
        }
        
        # Calculate additional statistical metrics
        if summary['mean'] > 0:
            summary['coefficient_of_variation'] = summary['stddev'] / summary['mean']
        else:
            summary['coefficient_of_variation'] = 0.0
        
        # Calculate relative standard error
        if summary['rounds'] > 1 and summary['mean'] > 0:
            summary['standard_error'] = summary['stddev'] / np.sqrt(summary['rounds'])
            summary['relative_standard_error'] = summary['standard_error'] / summary['mean']
        else:
            summary['standard_error'] = 0.0
            summary['relative_standard_error'] = 0.0
        
        return summary
    
    def _validate_sla_compliance(
        self, 
        metrics: List[PerformanceMetric], 
        category: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate SLA compliance for performance metrics.
        
        Args:
            metrics: List of performance metrics to validate
            category: SLA category for threshold lookup
            
        Returns:
            Tuple of (compliance_status, violation_details)
        """
        violation_details = []
        all_passed = True
        
        sla_config = self.sla_config['sla_thresholds'].get(category, {})
        minimum_samples = sla_config.get('minimum_sample_size', 3)
        
        for metric in metrics:
            # Check minimum sample size requirement
            if metric.sample_count < minimum_samples:
                violation_details.append(
                    f"Insufficient samples: {metric.sample_count} < {minimum_samples} required for statistical significance"
                )
                all_passed = False
            
            # Check SLA compliance
            if not metric.passes_sla:
                violation_details.append(metric.error_message)
                all_passed = False
            
            # Check statistical confidence if available
            if metric.confidence_interval and metric.confidence_interval[1] > metric.expected_max_time:
                violation_details.append(
                    f"Upper confidence bound {metric.confidence_interval[1]:.3f}s exceeds SLA {metric.expected_max_time:.3f}s"
                )
                all_passed = False
        
        return all_passed, violation_details
    
    def _generate_recommendations(
        self, 
        metrics: List[PerformanceMetric], 
        violation_details: List[str], 
        category: str
    ) -> List[str]:
        """
        Generate actionable optimization recommendations based on performance analysis.
        
        Args:
            metrics: Performance metrics for analysis
            violation_details: List of SLA violations
            category: SLA category for context-specific recommendations
            
        Returns:
            List of actionable optimization recommendations
        """
        recommendations = []
        
        for metric in metrics:
            if not metric.passes_sla:
                if category == 'data_loading':
                    recommendations.extend([
                        "Consider implementing lazy loading for large datasets",
                        "Optimize pickle deserialization with faster protocols",
                        "Implement streaming data loading for memory efficiency",
                        "Consider data compression to reduce I/O overhead"
                    ])
                    
                elif category == 'data_transformation':
                    recommendations.extend([
                        "Optimize DataFrame operations with vectorized computations",
                        "Consider chunked processing for large datasets",
                        "Implement column-wise processing to reduce memory overhead",
                        "Use dtype optimization to reduce memory usage"
                    ])
                    
                elif category == 'format_detection':
                    recommendations.extend([
                        "Implement format caching to avoid repeated detection",
                        "Optimize file header reading for faster detection",
                        "Consider format hints to skip detection when possible"
                    ])
            
            # Statistical quality recommendations
            if metric.sample_count < 5:
                recommendations.append("Increase benchmark sample size for better statistical confidence")
            
            if metric.std_deviation and metric.mean > 0:
                cv = metric.std_deviation / metric.mean
                if cv > 0.1:  # High variability
                    recommendations.append("High performance variability detected - investigate system factors")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _compare_with_baseline(
        self, 
        benchmark_name: str, 
        current_stats: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """
        Compare current performance with historical baseline for regression detection.
        
        Args:
            benchmark_name: Name of benchmark for baseline lookup
            current_stats: Current benchmark statistics
            
        Returns:
            Baseline comparison analysis or None if no baseline available
        """
        # For this implementation, we'll simulate baseline comparison
        # In a real deployment, this would load from a persistent baseline database
        
        if benchmark_name not in self.historical_baseline:
            # No baseline available - this becomes the new baseline
            self.historical_baseline[benchmark_name] = current_stats.copy()
            return {
                'status': 'baseline_established',
                'message': 'No historical baseline - current performance becomes baseline',
                'regression_detected': False
            }
        
        baseline_stats = self.historical_baseline[benchmark_name]
        current_mean = current_stats.get('mean', 0.0)
        baseline_mean = baseline_stats.get('mean', 0.0)
        
        if baseline_mean == 0:
            return None
        
        # Calculate performance change percentage
        performance_change = ((current_mean - baseline_mean) / baseline_mean) * 100.0
        
        # Check regression threshold
        regression_config = self.sla_config['sla_thresholds']['regression_detection']
        max_degradation = regression_config.get('max_performance_degradation', 20.0)
        
        regression_detected = performance_change > max_degradation
        
        comparison = {
            'status': 'regression_detected' if regression_detected else 'performance_stable',
            'baseline_mean': baseline_mean,
            'current_mean': current_mean,
            'performance_change_percent': performance_change,
            'regression_detected': regression_detected,
            'threshold_percent': max_degradation
        }
        
        if regression_detected:
            comparison['message'] = (
                f"Performance regression: {performance_change:.1f}% slower than baseline "
                f"(threshold: {max_degradation:.1f}%)"
            )
        else:
            comparison['message'] = f"Performance stable: {performance_change:.1f}% change from baseline"
        
        return comparison
    
    def validate_all_benchmarks(self, results_path: Path) -> PerformanceReport:
        """
        Validate all benchmarks in results file against SLA requirements.
        
        Performs comprehensive validation including:
        - Individual benchmark SLA compliance
        - Statistical significance analysis
        - Performance regression detection
        - Quality gate decision making
        
        Args:
            results_path: Path to pytest-benchmark JSON results
            
        Returns:
            Comprehensive performance validation report
        """
        if self.verbose:
            print(f"[SLA-VALIDATOR] Starting comprehensive SLA validation for {results_path}")
        
        # Load benchmark results
        try:
            results = self.load_benchmark_results(results_path)
        except Exception as e:
            # Create failure report for loading errors
            return PerformanceReport(
                timestamp=datetime.now(timezone.utc).isoformat(),
                validation_status='FAIL',
                total_benchmarks=0,
                passing_benchmarks=0,
                failing_benchmarks=0,
                sla_compliance_rate=0.0,
                benchmark_analyses=[],
                quality_gate_decision='BLOCK_MERGE',
                performance_trends={},
                recommendations=[f"Failed to load benchmark results: {e}"],
                execution_summary={'error': str(e)}
            )
        
        # Analyze each benchmark
        benchmark_analyses = []
        passing_count = 0
        failing_count = 0
        
        for benchmark_data in results.get('benchmarks', []):
            try:
                analysis = self.analyze_benchmark(benchmark_data)
                benchmark_analyses.append(analysis)
                
                if analysis.sla_compliance:
                    passing_count += 1
                else:
                    failing_count += 1
                    
            except Exception as e:
                # Create failure analysis for problematic benchmarks
                failing_count += 1
                error_analysis = BenchmarkAnalysis(
                    benchmark_name=benchmark_data.get('name', 'unknown'),
                    group=benchmark_data.get('group', 'default'),
                    category='unknown',
                    metrics=[],
                    statistical_summary={},
                    sla_compliance=False,
                    violation_details=[f"Analysis failed: {e}"],
                    recommendations=["Fix benchmark analysis errors before re-validation"]
                )
                benchmark_analyses.append(error_analysis)
                
                if self.verbose:
                    print(f"[SLA-VALIDATOR] Failed to analyze {benchmark_data.get('name')}: {e}")
        
        total_benchmarks = len(benchmark_analyses)
        sla_compliance_rate = (passing_count / total_benchmarks * 100.0) if total_benchmarks > 0 else 0.0
        
        # Determine overall validation status and quality gate decision
        validation_status, quality_gate_decision = self._determine_quality_gate_decision(
            passing_count, failing_count, sla_compliance_rate
        )
        
        # Generate comprehensive recommendations
        recommendations = self._generate_comprehensive_recommendations(benchmark_analyses)
        
        # Calculate performance trends
        performance_trends = self._calculate_performance_trends(benchmark_analyses)
        
        # Create execution summary
        execution_summary = {
            'validation_start_time': datetime.now(timezone.utc).isoformat(),
            'total_benchmarks_analyzed': total_benchmarks,
            'sla_compliance_rate': sla_compliance_rate,
            'regression_detection_enabled': self.enable_regression_detection,
            'configuration_source': str(self.quality_gates_config_path) if self.quality_gates_config_path else 'default'
        }
        
        report = PerformanceReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            validation_status=validation_status,
            total_benchmarks=total_benchmarks,
            passing_benchmarks=passing_count,
            failing_benchmarks=failing_count,
            sla_compliance_rate=sla_compliance_rate,
            benchmark_analyses=benchmark_analyses,
            quality_gate_decision=quality_gate_decision,
            performance_trends=performance_trends,
            recommendations=recommendations,
            execution_summary=execution_summary
        )
        
        if self.verbose:
            print(f"[SLA-VALIDATOR] Validation complete: {validation_status} ({passing_count}/{total_benchmarks} passed)")
        
        return report
    
    def _determine_quality_gate_decision(
        self, 
        passing_count: int, 
        failing_count: int, 
        compliance_rate: float
    ) -> Tuple[str, str]:
        """
        Determine quality gate decision based on SLA compliance and configuration.
        
        Args:
            passing_count: Number of benchmarks that passed SLA validation
            failing_count: Number of benchmarks that failed SLA validation  
            compliance_rate: Overall SLA compliance rate percentage
            
        Returns:
            Tuple of (validation_status, quality_gate_decision)
        """
        quality_config = self.sla_config['quality_gates']
        
        # Determine validation status
        if failing_count == 0:
            validation_status = 'PASS'
        elif compliance_rate >= 80.0:  # 80% threshold for warning vs failure
            validation_status = 'WARNING'
        else:
            validation_status = 'FAIL'
        
        # Determine quality gate decision based on enforcement configuration
        if quality_config.get('fail_on_sla_violation', True):
            if failing_count > 0:
                quality_gate_decision = 'BLOCK_MERGE'
            else:
                quality_gate_decision = 'ALLOW_MERGE'
        else:
            # Lenient mode - only block on severe violations
            if compliance_rate < 50.0:  # Less than 50% compliance
                quality_gate_decision = 'BLOCK_MERGE'
            else:
                quality_gate_decision = 'ALLOW_MERGE_WITH_WARNING'
        
        return validation_status, quality_gate_decision
    
    def _generate_comprehensive_recommendations(
        self, 
        benchmark_analyses: List[BenchmarkAnalysis]
    ) -> List[str]:
        """Generate comprehensive optimization recommendations across all benchmarks."""
        all_recommendations = []
        
        # Collect recommendations from all analyses
        for analysis in benchmark_analyses:
            all_recommendations.extend(analysis.recommendations)
        
        # Add global recommendations based on overall patterns
        failing_categories = defaultdict(int)
        for analysis in benchmark_analyses:
            if not analysis.sla_compliance:
                failing_categories[analysis.category] += 1
        
        global_recommendations = []
        
        if failing_categories.get('data_loading', 0) > 0:
            global_recommendations.append(
                "Multiple data loading SLA violations detected - consider I/O optimization strategies"
            )
        
        if failing_categories.get('data_transformation', 0) > 0:
            global_recommendations.append(
                "DataFrame transformation performance issues - consider vectorization and memory optimization"
            )
        
        # System-level recommendations
        total_failing = sum(failing_categories.values())
        if total_failing > len(benchmark_analyses) * 0.5:
            global_recommendations.append(
                "Widespread performance issues detected - consider system-level optimization"
            )
        
        # Combine and deduplicate
        all_recommendations.extend(global_recommendations)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]  # Limit to top 10 recommendations
    
    def _calculate_performance_trends(
        self, 
        benchmark_analyses: List[BenchmarkAnalysis]
    ) -> Dict[str, Any]:
        """Calculate performance trends and regression patterns."""
        trends = {
            'category_performance': {},
            'regression_summary': {},
            'statistical_quality': {}
        }
        
        # Category-wise performance summary
        for category in ['data_loading', 'data_transformation', 'format_detection']:
            category_analyses = [a for a in benchmark_analyses if a.category == category]
            if category_analyses:
                compliance_rate = sum(1 for a in category_analyses if a.sla_compliance) / len(category_analyses)
                avg_performance = statistics.mean([
                    m.execution_time for a in category_analyses for m in a.metrics
                ])
                
                trends['category_performance'][category] = {
                    'compliance_rate': compliance_rate * 100.0,
                    'average_execution_time': avg_performance,
                    'benchmark_count': len(category_analyses)
                }
        
        # Regression detection summary
        if self.enable_regression_detection:
            regression_count = sum(
                1 for a in benchmark_analyses 
                if a.baseline_comparison and a.baseline_comparison.get('regression_detected', False)
            )
            trends['regression_summary'] = {
                'regressions_detected': regression_count,
                'total_with_baseline': sum(1 for a in benchmark_analyses if a.baseline_comparison),
                'regression_rate': (regression_count / len(benchmark_analyses) * 100.0) if benchmark_analyses else 0.0
            }
        
        # Statistical quality assessment
        high_variance_count = 0
        low_sample_count = 0
        
        for analysis in benchmark_analyses:
            for metric in analysis.metrics:
                if metric.std_deviation and metric.execution_time > 0:
                    cv = metric.std_deviation / metric.execution_time
                    if cv > 0.15:  # High coefficient of variation
                        high_variance_count += 1
                
                if metric.sample_count < 5:
                    low_sample_count += 1
        
        trends['statistical_quality'] = {
            'high_variance_benchmarks': high_variance_count,
            'low_sample_benchmarks': low_sample_count,
            'total_metrics': sum(len(a.metrics) for a in benchmark_analyses)
        }
        
        return trends
    
    def generate_report(
        self, 
        report: PerformanceReport, 
        output_path: Optional[Path] = None,
        format_type: str = 'json'
    ) -> str:
        """
        Generate performance validation report in specified format.
        
        Args:
            report: Performance validation report to format
            output_path: Optional path to save report (if None, returns as string)
            format_type: Report format ('json', 'html', 'markdown')
            
        Returns:
            Formatted report content as string
        """
        if format_type == 'json':
            content = self._generate_json_report(report)
        elif format_type == 'html':
            content = self._generate_html_report(report)
        elif format_type == 'markdown':
            content = self._generate_markdown_report(report)
        else:
            raise ValueError(f"Unsupported report format: {format_type}")
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(content)
            
            if self.verbose:
                print(f"[SLA-VALIDATOR] Report saved to {output_path}")
        
        return content
    
    def _generate_json_report(self, report: PerformanceReport) -> str:
        """Generate JSON format performance report."""
        # Convert dataclasses to dictionaries for JSON serialization
        report_dict = asdict(report)
        
        # Add metadata
        report_dict['metadata'] = {
            'generator': 'FlyRigLoader Performance SLA Validator',
            'version': '1.0.0',
            'schema_version': 'performance-report-v1.0',
            'generation_time': datetime.now(timezone.utc).isoformat()
        }
        
        return json.dumps(report_dict, indent=2, default=str)
    
    def _generate_html_report(self, report: PerformanceReport) -> str:
        """Generate HTML format performance report with styling."""
        status_color = {
            'PASS': '#28a745',
            'WARNING': '#ffc107', 
            'FAIL': '#dc3545'
        }
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlyRigLoader Performance SLA Validation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .status {{ font-weight: bold; color: {status_color.get(report.validation_status, '#000')}; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #fff; border: 1px solid #dee2e6; border-radius: 4px; padding: 15px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #495057; }}
        .metric-label {{ color: #6c757d; font-size: 14px; }}
        .benchmark-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .benchmark-table th, .benchmark-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .benchmark-table th {{ background-color: #f8f9fa; font-weight: 600; }}
        .pass {{ color: #28a745; }}
        .fail {{ color: #dc3545; }}
        .recommendations {{ background: #e3f2fd; padding: 15px; border-radius: 4px; margin: 20px 0; }}
        .recommendations ul {{ margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>FlyRigLoader Performance SLA Validation Report</h1>
        <p><strong>Validation Status:</strong> <span class="status">{report.validation_status}</span></p>
        <p><strong>Generated:</strong> {report.timestamp}</p>
        <p><strong>Quality Gate Decision:</strong> {report.quality_gate_decision}</p>
    </div>
    
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value">{report.total_benchmarks}</div>
            <div class="metric-label">Total Benchmarks</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{report.passing_benchmarks}</div>
            <div class="metric-label">Passing</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{report.failing_benchmarks}</div>
            <div class="metric-label">Failing</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{report.sla_compliance_rate:.1f}%</div>
            <div class="metric-label">SLA Compliance Rate</div>
        </div>
    </div>
    
    <h2>Benchmark Analysis Results</h2>
    <table class="benchmark-table">
        <thead>
            <tr>
                <th>Benchmark Name</th>
                <th>Category</th>
                <th>Status</th>
                <th>Execution Time</th>
                <th>Expected Max</th>
                <th>Margin</th>
            </tr>
        </thead>
        <tbody>
"""
        
        for analysis in report.benchmark_analyses:
            for metric in analysis.metrics:
                status_class = 'pass' if metric.passes_sla else 'fail'
                status_text = 'PASS' if metric.passes_sla else 'FAIL'
                
                html_content += f"""
            <tr>
                <td>{analysis.benchmark_name}</td>
                <td>{analysis.category}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{metric.execution_time:.3f}s</td>
                <td>{metric.expected_max_time:.3f}s</td>
                <td>{metric.margin_percentage:+.1f}%</td>
            </tr>
"""
        
        html_content += """
        </tbody>
    </table>
"""
        
        if report.recommendations:
            html_content += """
    <div class="recommendations">
        <h3>Optimization Recommendations</h3>
        <ul>
"""
            for rec in report.recommendations:
                html_content += f"            <li>{rec}</li>\n"
            
            html_content += """
        </ul>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        return html_content
    
    def _generate_markdown_report(self, report: PerformanceReport) -> str:
        """Generate Markdown format performance report."""
        status_emoji = {
            'PASS': '✅',
            'WARNING': '⚠️',
            'FAIL': '❌'
        }
        
        markdown_content = f"""# FlyRigLoader Performance SLA Validation Report

{status_emoji.get(report.validation_status, '🔍')} **Validation Status:** {report.validation_status}
🕒 **Generated:** {report.timestamp}
🚦 **Quality Gate Decision:** {report.quality_gate_decision}

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Benchmarks | {report.total_benchmarks} |
| Passing Benchmarks | {report.passing_benchmarks} |
| Failing Benchmarks | {report.failing_benchmarks} |
| SLA Compliance Rate | {report.sla_compliance_rate:.1f}% |

## Benchmark Analysis Results

| Benchmark Name | Category | Status | Execution Time | Expected Max | Margin |
|----------------|----------|--------|----------------|--------------|--------|
"""
        
        for analysis in report.benchmark_analyses:
            for metric in analysis.metrics:
                status_emoji_cell = '✅' if metric.passes_sla else '❌'
                
                markdown_content += f"| {analysis.benchmark_name} | {analysis.category} | {status_emoji_cell} | {metric.execution_time:.3f}s | {metric.expected_max_time:.3f}s | {metric.margin_percentage:+.1f}% |\n"
        
        if report.recommendations:
            markdown_content += f"""
## Optimization Recommendations

"""
            for i, rec in enumerate(report.recommendations, 1):
                markdown_content += f"{i}. {rec}\n"
        
        # Add performance trends if available
        if report.performance_trends.get('category_performance'):
            markdown_content += """
## Performance Trends by Category

"""
            for category, trends in report.performance_trends['category_performance'].items():
                compliance = trends['compliance_rate']
                avg_time = trends['average_execution_time']
                count = trends['benchmark_count']
                
                markdown_content += f"- **{category.replace('_', ' ').title()}:** {compliance:.1f}% compliance ({count} benchmarks, avg: {avg_time:.3f}s)\n"
        
        markdown_content += f"""
---
*Report generated by FlyRigLoader Performance SLA Validator v1.0.0*
"""
        
        return markdown_content


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser for the SLA validation script."""
    parser = argparse.ArgumentParser(
        description='FlyRigLoader Performance SLA Validation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic SLA validation
  python check-performance-slas.py benchmark-results.json
  
  # With custom quality gates configuration
  python check-performance-slas.py benchmark-results.json --config quality-gates.yml
  
  # Generate HTML report
  python check-performance-slas.py benchmark-results.json --output report.html --format html
  
  # Verbose output with regression detection
  python check-performance-slas.py benchmark-results.json --verbose --enable-regression
  
  # Fail fast mode for CI/CD
  python check-performance-slas.py benchmark-results.json --fail-fast
        """
    )
    
    parser.add_argument(
        'benchmark_results',
        type=Path,
        help='Path to pytest-benchmark JSON results file'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=Path,
        help='Path to quality gates YAML configuration file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output path for validation report (default: stdout)'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'html', 'markdown'],
        default='json',
        help='Report format (default: json)'
    )
    
    parser.add_argument(
        '--fail-fast',
        action='store_true',
        help='Exit immediately on first SLA violation'
    )
    
    parser.add_argument(
        '--enable-regression',
        action='store_true',
        default=True,
        help='Enable performance regression detection (default: True)'
    )
    
    parser.add_argument(
        '--disable-regression',
        action='store_true',
        help='Disable performance regression detection'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='FlyRigLoader Performance SLA Validator 1.0.0'
    )
    
    return parser


def main() -> int:
    """
    Main entry point for the performance SLA validation script.
    
    Returns:
        Exit code: 0 for success, 1 for SLA violations, 2 for errors
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Handle regression detection flags
    enable_regression = args.enable_regression and not args.disable_regression
    
    try:
        # Initialize SLA validator
        validator = PerformanceSLAValidator(
            quality_gates_config_path=args.config,
            enable_regression_detection=enable_regression,
            verbose=args.verbose
        )
        
        if args.verbose:
            print(f"[SLA-VALIDATOR] Starting validation of {args.benchmark_results}")
        
        # Validate all benchmarks
        report = validator.validate_all_benchmarks(args.benchmark_results)
        
        # Generate and output report
        report_content = validator.generate_report(
            report=report,
            output_path=args.output,
            format_type=args.format
        )
        
        # Print report to stdout if no output file specified
        if not args.output:
            print(report_content)
        
        # Determine exit code based on validation results
        if report.validation_status == 'FAIL':
            if args.verbose:
                print(f"[SLA-VALIDATOR] SLA validation FAILED: {report.failing_benchmarks}/{report.total_benchmarks} benchmarks failed")
            return 1
        elif report.validation_status == 'WARNING':
            if args.verbose:
                print(f"[SLA-VALIDATOR] SLA validation WARNING: {report.failing_benchmarks}/{report.total_benchmarks} benchmarks failed")
            return 0  # Warnings don't cause failure
        else:
            if args.verbose:
                print(f"[SLA-VALIDATOR] SLA validation PASSED: {report.passing_benchmarks}/{report.total_benchmarks} benchmarks passed")
            return 0
    
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 2


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)