"""
Comprehensive utilities module for the flyrigloader benchmark test suite.

This module provides statistical analysis, memory profiling, environment normalization, 
artifact generation, and regression detection capabilities for the benchmark test suite, 
supporting cross-platform performance validation and CI/CD integration.

Features:
- Statistical analysis framework with confidence intervals and regression detection
- Memory profiling capabilities with line-by-line analysis using pytest-memory-profiler
- Environment normalization for consistent results across development and CI environments
- Artifact generation for JSON/CSV performance reports with CI/CD integration
- Cross-platform performance validation with hardware normalization factors
- Regression detection with automated performance alerts and statistical significance testing
- Memory leak detection for large dataset processing scenarios
- Benchmark execution coordination for pytest-benchmark orchestration
- Performance report generation with comprehensive statistical summaries
- CI/CD integration utilities for GitHub Actions artifact management

Integration:
- pytest-benchmark for performance measurement orchestration
- pytest-memory-profiler for detailed memory analysis and leak detection
- psutil for cross-platform system resource monitoring and normalization
- scipy.stats for statistical analysis and confidence interval calculation
- GitHub Actions for CI/CD artifact management and performance alerting
- Cross-platform compatibility across Ubuntu, Windows, macOS environments
"""

import gc
import json
import csv
import os
import platform
import subprocess
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import psutil
from scipy import stats
import pytest

# Import configuration from the dependency
from .config import (
    BenchmarkConfig, 
    DEFAULT_BENCHMARK_CONFIG,
    get_benchmark_config,
    get_category_config,
    BenchmarkCategory,
    StatisticalAnalysisConfig,
    EnvironmentNormalizationConfig,
    MemoryProfilingConfig,
    CICDIntegrationConfig
)


# ============================================================================
# MEMORY PROFILING UTILITIES (Enhanced from test_benchmark_transformations.py)
# ============================================================================

class MemoryProfiler:
    """
    Enhanced utility class for monitoring memory usage during DataFrame transformation
    operations to validate memory efficiency requirements with line-by-line analysis.
    
    Extracted and enhanced from test_benchmark_transformations.py to provide shared
    memory profiling capabilities across the benchmark test suite.
    
    Features:
    - Memory usage tracking with peak detection
    - Memory leak detection for iterative operations
    - Line-by-line memory profiling integration
    - Large dataset memory analysis (>500MB scenarios)
    - Cross-platform memory monitoring via psutil
    - Memory efficiency validation against Section 2.4.5 requirements
    """
    
    def __init__(self, precision: int = 3, enable_line_profiling: bool = True):
        """
        Initialize memory profiler with configurable precision.
        
        Args:
            precision: Decimal precision for memory measurements
            enable_line_profiling: Whether to enable detailed line-by-line profiling
        """
        self.precision = precision
        self.enable_line_profiling = enable_line_profiling
        self.initial_memory = None
        self.peak_memory = None
        self.final_memory = None
        self.data_size_estimate = None
        self.memory_timeline = []
        self.gc_events = []
        self.process = psutil.Process()
        self._monitoring_active = False
        self._monitoring_thread = None
        self._monitoring_queue = queue.Queue()
    
    def start_profiling(self, data_size_estimate: int, monitor_interval: float = 0.1):
        """
        Start memory profiling session with optional continuous monitoring.
        
        Args:
            data_size_estimate: Estimated size of input data in bytes
            monitor_interval: Interval for continuous memory monitoring in seconds
        """
        gc.collect()  # Clean up before measurement
        self.initial_memory = self.process.memory_info().rss
        self.peak_memory = self.initial_memory
        self.data_size_estimate = data_size_estimate
        self.memory_timeline = [(time.time(), self.initial_memory)]
        self.gc_events = []
        
        # Start continuous monitoring if enabled
        if self.enable_line_profiling:
            self._start_continuous_monitoring(monitor_interval)
    
    def _start_continuous_monitoring(self, interval: float):
        """Start continuous memory monitoring in background thread."""
        self._monitoring_active = True
        
        def monitor_memory():
            while self._monitoring_active:
                try:
                    current_time = time.time()
                    current_memory = self.process.memory_info().rss
                    self.memory_timeline.append((current_time, current_memory))
                    self.peak_memory = max(self.peak_memory, current_memory)
                    time.sleep(interval)
                except Exception as e:
                    self._monitoring_queue.put(f"Monitoring error: {e}")
                    break
        
        self._monitoring_thread = threading.Thread(target=monitor_memory, daemon=True)
        self._monitoring_thread.start()
    
    def update_peak_memory(self):
        """Update peak memory usage measurement."""
        current_memory = self.process.memory_info().rss
        self.peak_memory = max(self.peak_memory, current_memory)
        self.memory_timeline.append((time.time(), current_memory))
    
    def record_gc_event(self, event_type: str = "manual"):
        """Record garbage collection event for analysis."""
        gc_time = time.time()
        memory_before = self.process.memory_info().rss
        gc.collect()
        memory_after = self.process.memory_info().rss
        
        self.gc_events.append({
            'timestamp': gc_time,
            'type': event_type,
            'memory_before_mb': memory_before / 1024 / 1024,
            'memory_after_mb': memory_after / 1024 / 1024,
            'memory_freed_mb': (memory_before - memory_after) / 1024 / 1024
        })
    
    def end_profiling(self) -> Dict[str, Any]:
        """
        End profiling session and return comprehensive memory usage statistics.
        
        Returns:
            Dict containing detailed memory usage metrics and analysis
        """
        # Stop continuous monitoring
        if self._monitoring_active:
            self._monitoring_active = False
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=1.0)
        
        gc.collect()
        self.final_memory = self.process.memory_info().rss
        self.memory_timeline.append((time.time(), self.final_memory))
        
        # Calculate comprehensive statistics
        memory_overhead = self.peak_memory - self.initial_memory
        memory_multiplier = memory_overhead / max(self.data_size_estimate, 1)
        
        # Analyze memory timeline
        memory_values = [mem for _, mem in self.memory_timeline]
        memory_trend = self._analyze_memory_trend(memory_values)
        
        # Detect potential memory leaks
        leak_analysis = self._detect_memory_leaks()
        
        return {
            'initial_memory_mb': round(self.initial_memory / 1024 / 1024, self.precision),
            'peak_memory_mb': round(self.peak_memory / 1024 / 1024, self.precision),
            'final_memory_mb': round(self.final_memory / 1024 / 1024, self.precision),
            'memory_overhead_mb': round(memory_overhead / 1024 / 1024, self.precision),
            'data_size_mb': round(self.data_size_estimate / 1024 / 1024, self.precision),
            'memory_multiplier': round(memory_multiplier, self.precision),
            'meets_efficiency_requirement': memory_multiplier <= 2.0,
            'memory_trend': memory_trend,
            'leak_analysis': leak_analysis,
            'gc_events': self.gc_events,
            'timeline_samples': len(self.memory_timeline),
            'monitoring_duration_seconds': self.memory_timeline[-1][0] - self.memory_timeline[0][0] if len(self.memory_timeline) > 1 else 0
        }
    
    def _analyze_memory_trend(self, memory_values: List[int]) -> Dict[str, Any]:
        """Analyze memory usage trend over time."""
        if len(memory_values) < 2:
            return {'trend': 'insufficient_data', 'slope': 0, 'correlation': 0}
        
        # Calculate linear regression for memory trend
        x = np.arange(len(memory_values))
        y = np.array(memory_values)
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            trend_classification = 'stable'
            if slope > 1024 * 1024:  # >1MB per sample increase
                trend_classification = 'increasing'
            elif slope < -1024 * 1024:  # >1MB per sample decrease
                trend_classification = 'decreasing'
            
            return {
                'trend': trend_classification,
                'slope_mb_per_sample': slope / 1024 / 1024,
                'correlation': r_value,
                'p_value': p_value,
                'memory_variance_mb': np.std(y) / 1024 / 1024
            }
        except Exception as e:
            return {'trend': 'analysis_error', 'error': str(e)}
    
    def _detect_memory_leaks(self) -> Dict[str, Any]:
        """Detect potential memory leaks based on memory timeline analysis."""
        if len(self.memory_timeline) < 10:
            return {'leak_detected': False, 'confidence': 0, 'reason': 'insufficient_samples'}
        
        # Compare memory usage in first and last quarters
        quarter_size = len(self.memory_timeline) // 4
        initial_quarter = [mem for _, mem in self.memory_timeline[:quarter_size]]
        final_quarter = [mem for _, mem in self.memory_timeline[-quarter_size:]]
        
        initial_avg = np.mean(initial_quarter)
        final_avg = np.mean(final_quarter)
        memory_growth = final_avg - initial_avg
        
        # Define leak detection thresholds
        leak_threshold_mb = 10.0  # 10MB growth threshold
        leak_threshold_bytes = leak_threshold_mb * 1024 * 1024
        
        leak_detected = memory_growth > leak_threshold_bytes
        confidence = min(memory_growth / leak_threshold_bytes, 2.0) if leak_detected else 0
        
        return {
            'leak_detected': leak_detected,
            'confidence': confidence,
            'memory_growth_mb': memory_growth / 1024 / 1024,
            'threshold_mb': leak_threshold_mb,
            'initial_avg_mb': initial_avg / 1024 / 1024,
            'final_avg_mb': final_avg / 1024 / 1024
        }


@contextmanager
def memory_profiling_context(
    data_size_estimate: int,
    precision: int = 3,
    enable_line_profiling: bool = True,
    monitor_interval: float = 0.1
):
    """
    Context manager for convenient memory profiling with automatic cleanup.
    
    Args:
        data_size_estimate: Estimated size of data being processed
        precision: Decimal precision for measurements
        enable_line_profiling: Whether to enable continuous monitoring
        monitor_interval: Monitoring interval in seconds
        
    Yields:
        MemoryProfiler instance for manual monitoring control
        
    Returns:
        Dict with comprehensive memory analysis results
    """
    profiler = MemoryProfiler(precision=precision, enable_line_profiling=enable_line_profiling)
    
    try:
        profiler.start_profiling(data_size_estimate, monitor_interval)
        yield profiler
    finally:
        memory_stats = profiler.end_profiling()
        return memory_stats


def estimate_data_size(data_object: Any) -> int:
    """
    Estimate memory size of a data object for memory profiling.
    
    Args:
        data_object: Object to estimate size for (Dict, DataFrame, array, etc.)
        
    Returns:
        Estimated size in bytes
    """
    if hasattr(data_object, 'memory_usage'):
        # pandas DataFrame/Series
        return data_object.memory_usage(deep=True).sum()
    elif hasattr(data_object, 'nbytes'):
        # numpy array
        return data_object.nbytes
    elif isinstance(data_object, dict):
        # Dictionary with arrays/data
        total_size = 0
        for key, value in data_object.items():
            if hasattr(value, 'nbytes'):
                total_size += value.nbytes
            elif isinstance(value, (list, tuple)):
                total_size += len(value) * 64  # Conservative estimate
            else:
                total_size += 64  # Conservative estimate for other types
        return total_size
    else:
        # Fallback estimation
        return 1024  # 1KB default estimate


# ============================================================================
# STATISTICAL ANALYSIS FRAMEWORK
# ============================================================================

@dataclass
class ConfidenceInterval:
    """Statistical confidence interval with analysis metadata."""
    lower_bound: float
    upper_bound: float
    confidence_level: float
    mean: float
    std_error: float
    sample_size: int
    
    def contains(self, value: float) -> bool:
        """Check if a value falls within the confidence interval."""
        return self.lower_bound <= value <= self.upper_bound
    
    def margin_of_error(self) -> float:
        """Calculate margin of error for the confidence interval."""
        return (self.upper_bound - self.lower_bound) / 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert confidence interval to dictionary representation."""
        return {
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'confidence_level': self.confidence_level,
            'mean': self.mean,
            'std_error': self.std_error,
            'sample_size': self.sample_size,
            'margin_of_error': self.margin_of_error()
        }


class StatisticalAnalysisEngine:
    """
    Comprehensive statistical analysis engine for benchmark performance validation
    with confidence intervals, regression detection, and baseline comparison.
    
    Implements statistical analysis framework per Section 6.6.4.2 Enhanced Performance
    Analysis including confidence interval calculation and historical baseline comparison.
    """
    
    def __init__(self, config: StatisticalAnalysisConfig = None):
        """
        Initialize statistical analysis engine.
        
        Args:
            config: Statistical analysis configuration, uses default if None
        """
        self.config = config or DEFAULT_BENCHMARK_CONFIG.statistical_analysis
        self.baseline_cache = {}
        self.historical_data = {}
    
    def calculate_confidence_interval(
        self,
        measurements: List[float],
        confidence_level: float = None
    ) -> ConfidenceInterval:
        """
        Calculate confidence interval for performance measurements.
        
        Args:
            measurements: List of performance measurements
            confidence_level: Confidence level (0.0-1.0), uses config default if None
            
        Returns:
            ConfidenceInterval object with statistical analysis
        """
        if not measurements or len(measurements) < 2:
            raise ValueError("Need at least 2 measurements for confidence interval")
        
        confidence_level = confidence_level or self.config.confidence_level
        alpha = 1 - confidence_level
        
        # Calculate basic statistics
        mean_value = np.mean(measurements)
        std_dev = np.std(measurements, ddof=1)
        std_error = std_dev / np.sqrt(len(measurements))
        
        # Calculate t-statistic for confidence interval
        df = len(measurements) - 1
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        # Calculate confidence interval bounds
        margin_of_error = t_critical * std_error
        lower_bound = mean_value - margin_of_error
        upper_bound = mean_value + margin_of_error
        
        return ConfidenceInterval(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            mean=mean_value,
            std_error=std_error,
            sample_size=len(measurements)
        )
    
    def detect_outliers(
        self,
        measurements: List[float],
        method: str = 'iqr',
        threshold: float = None
    ) -> Tuple[List[float], List[int]]:
        """
        Detect and remove outliers from performance measurements.
        
        Args:
            measurements: List of performance measurements
            method: Outlier detection method ('iqr', 'zscore', 'modified_zscore')
            threshold: Custom threshold, uses config default if None
            
        Returns:
            Tuple of (cleaned_measurements, outlier_indices)
        """
        if len(measurements) < 3:
            return measurements, []
        
        measurements_array = np.array(measurements)
        outlier_indices = []
        
        if method == 'iqr':
            q1 = np.percentile(measurements_array, 25)
            q3 = np.percentile(measurements_array, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_indices = np.where(
                (measurements_array < lower_bound) | (measurements_array > upper_bound)
            )[0].tolist()
        
        elif method == 'zscore':
            threshold = threshold or self.config.outlier_detection_stddev
            z_scores = np.abs(stats.zscore(measurements_array))
            outlier_indices = np.where(z_scores > threshold)[0].tolist()
        
        elif method == 'modified_zscore':
            threshold = threshold or 3.5
            median = np.median(measurements_array)
            mad = np.median(np.abs(measurements_array - median))
            modified_z_scores = 0.6745 * (measurements_array - median) / mad
            outlier_indices = np.where(np.abs(modified_z_scores) > threshold)[0].tolist()
        
        # Create cleaned measurements list
        cleaned_measurements = [
            measurements[i] for i in range(len(measurements)) 
            if i not in outlier_indices
        ]
        
        return cleaned_measurements, outlier_indices
    
    def perform_significance_test(
        self,
        baseline_measurements: List[float],
        current_measurements: List[float],
        test_type: str = 'ttest'
    ) -> Dict[str, Any]:
        """
        Perform statistical significance test between baseline and current measurements.
        
        Args:
            baseline_measurements: Historical baseline measurements
            current_measurements: Current performance measurements
            test_type: Type of statistical test ('ttest', 'mannwhitney', 'anova')
            
        Returns:
            Dict containing test results and statistical significance analysis
        """
        if len(baseline_measurements) < 2 or len(current_measurements) < 2:
            return {
                'test_type': test_type,
                'statistic': None,
                'p_value': None,
                'significant': False,
                'error': 'Insufficient data for significance testing'
            }
        
        try:
            if test_type == 'ttest':
                # Independent samples t-test
                statistic, p_value = stats.ttest_ind(current_measurements, baseline_measurements)
                test_description = 'Independent samples t-test'
                
            elif test_type == 'mannwhitney':
                # Mann-Whitney U test (non-parametric)
                statistic, p_value = stats.mannwhitneyu(
                    current_measurements, baseline_measurements, alternative='two-sided'
                )
                test_description = 'Mann-Whitney U test'
                
            elif test_type == 'welch_ttest':
                # Welch's t-test (unequal variances)
                statistic, p_value = stats.ttest_ind(
                    current_measurements, baseline_measurements, equal_var=False
                )
                test_description = "Welch's t-test (unequal variances)"
            
            else:
                raise ValueError(f"Unsupported test type: {test_type}")
            
            # Determine statistical significance
            significant = p_value < self.config.significance_level
            
            # Calculate effect size (Cohen's d)
            baseline_mean = np.mean(baseline_measurements)
            current_mean = np.mean(current_measurements)
            pooled_std = np.sqrt(
                ((len(baseline_measurements) - 1) * np.var(baseline_measurements, ddof=1) +
                 (len(current_measurements) - 1) * np.var(current_measurements, ddof=1)) /
                (len(baseline_measurements) + len(current_measurements) - 2)
            )
            cohens_d = (current_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
            
            return {
                'test_type': test_description,
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': significant,
                'significance_level': self.config.significance_level,
                'baseline_mean': baseline_mean,
                'current_mean': current_mean,
                'mean_difference': current_mean - baseline_mean,
                'percent_change': ((current_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else 0,
                'effect_size_cohens_d': cohens_d,
                'sample_sizes': {
                    'baseline': len(baseline_measurements),
                    'current': len(current_measurements)
                }
            }
            
        except Exception as e:
            return {
                'test_type': test_type,
                'statistic': None,
                'p_value': None,
                'significant': False,
                'error': f'Statistical test failed: {str(e)}'
            }
    
    def detect_performance_regression(
        self,
        baseline_measurements: List[float],
        current_measurements: List[float],
        regression_threshold: float = None
    ) -> Dict[str, Any]:
        """
        Detect performance regression using statistical analysis and threshold comparison.
        
        Args:
            baseline_measurements: Historical baseline performance measurements
            current_measurements: Current performance measurements
            regression_threshold: Regression threshold percentage, uses config default if None
            
        Returns:
            Dict containing regression detection results and analysis
        """
        regression_threshold = regression_threshold or self.config.regression_threshold_percent
        
        # Clean outliers from both datasets
        cleaned_baseline, baseline_outliers = self.detect_outliers(baseline_measurements)
        cleaned_current, current_outliers = self.detect_outliers(current_measurements)
        
        if len(cleaned_baseline) < 2 or len(cleaned_current) < 2:
            return {
                'regression_detected': False,
                'confidence': 0,
                'reason': 'insufficient_data_after_outlier_removal',
                'outliers_removed': {
                    'baseline': len(baseline_outliers),
                    'current': len(current_outliers)
                }
            }
        
        # Calculate baseline statistics
        baseline_mean = np.mean(cleaned_baseline)
        baseline_ci = self.calculate_confidence_interval(cleaned_baseline)
        
        # Calculate current statistics
        current_mean = np.mean(cleaned_current)
        current_ci = self.calculate_confidence_interval(cleaned_current)
        
        # Check for regression based on threshold
        percent_change = ((current_mean - baseline_mean) / baseline_mean) * 100
        threshold_regression = percent_change > regression_threshold
        
        # Perform statistical significance test
        significance_test = self.perform_significance_test(cleaned_baseline, cleaned_current)
        
        # Check if current performance is outside baseline confidence interval
        ci_regression = not baseline_ci.contains(current_mean)
        
        # Determine overall regression detection
        regression_detected = threshold_regression and (significance_test['significant'] or ci_regression)
        
        # Calculate regression confidence score
        confidence_factors = []
        if threshold_regression:
            confidence_factors.append(min(percent_change / regression_threshold, 2.0))
        if significance_test['significant']:
            confidence_factors.append(1.0 - significance_test['p_value'])
        if ci_regression:
            confidence_factors.append(1.0)
        
        confidence = np.mean(confidence_factors) if confidence_factors else 0
        
        return {
            'regression_detected': regression_detected,
            'confidence': confidence,
            'percent_change': percent_change,
            'regression_threshold': regression_threshold,
            'threshold_exceeded': threshold_regression,
            'statistically_significant': significance_test['significant'],
            'outside_baseline_ci': ci_regression,
            'baseline_stats': {
                'mean': baseline_mean,
                'confidence_interval': baseline_ci.to_dict(),
                'sample_size': len(cleaned_baseline),
                'outliers_removed': len(baseline_outliers)
            },
            'current_stats': {
                'mean': current_mean,
                'confidence_interval': current_ci.to_dict(),
                'sample_size': len(cleaned_current),
                'outliers_removed': len(current_outliers)
            },
            'significance_test': significance_test
        }
    
    def update_baseline(
        self,
        test_name: str,
        measurements: List[float],
        max_baseline_samples: int = None
    ):
        """
        Update performance baseline with new measurements.
        
        Args:
            test_name: Name of the test for baseline tracking
            measurements: New performance measurements to add to baseline
            max_baseline_samples: Maximum baseline samples to retain
        """
        max_samples = max_baseline_samples or self.config.baseline_comparison_samples
        
        if test_name not in self.baseline_cache:
            self.baseline_cache[test_name] = []
        
        # Add new measurements to baseline
        self.baseline_cache[test_name].extend(measurements)
        
        # Trim to maximum baseline samples if needed
        if len(self.baseline_cache[test_name]) > max_samples:
            self.baseline_cache[test_name] = self.baseline_cache[test_name][-max_samples:]
    
    def get_baseline_measurements(self, test_name: str) -> List[float]:
        """
        Get baseline measurements for a specific test.
        
        Args:
            test_name: Name of the test
            
        Returns:
            List of baseline measurements, empty if no baseline exists
        """
        return self.baseline_cache.get(test_name, [])


# ============================================================================
# ENVIRONMENT NORMALIZATION UTILITIES
# ============================================================================

class EnvironmentAnalyzer:
    """
    Environment normalization utilities for consistent benchmark results across
    development and CI environments with CPU/memory normalization factors.
    
    Provides comprehensive environment analysis, hardware abstraction, and
    normalization factors for reproducible performance validation across
    Ubuntu, Windows, macOS with varying hardware configurations.
    """
    
    def __init__(self, config: EnvironmentNormalizationConfig = None):
        """
        Initialize environment analyzer.
        
        Args:
            config: Environment normalization configuration, uses default if None
        """
        self.config = config or DEFAULT_BENCHMARK_CONFIG.environment_normalization
        self._environment_cache = {}
        self._baseline_environment = None
    
    def analyze_current_environment(self) -> Dict[str, Any]:
        """
        Analyze current execution environment for normalization.
        
        Returns:
            Dict containing comprehensive environment analysis
        """
        if 'current_analysis' in self._environment_cache:
            return self._environment_cache['current_analysis']
        
        # Get base environment characteristics from config
        base_characteristics = self.config.detect_environment_characteristics()
        
        # Enhance with additional analysis
        enhanced_analysis = self._enhance_environment_analysis(base_characteristics)
        
        # Cache results for performance
        self._environment_cache['current_analysis'] = enhanced_analysis
        return enhanced_analysis
    
    def _enhance_environment_analysis(self, base_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance environment analysis with additional system information."""
        enhanced = base_characteristics.copy()
        
        # Add Python environment details
        enhanced['python_info'] = {
            'version': platform.python_version(),
            'implementation': platform.python_implementation(),
            'compiler': platform.python_compiler()
        }
        
        # Add system load information
        try:
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            enhanced['system_load'] = {
                '1min': load_avg[0],
                '5min': load_avg[1],
                '15min': load_avg[2]
            }
        except (OSError, AttributeError):
            enhanced['system_load'] = {'1min': 0, '5min': 0, '15min': 0}
        
        # Add disk I/O information
        try:
            disk_usage = psutil.disk_usage('/')
            enhanced['disk_info'] = {
                'total_gb': disk_usage.total / (1024**3),
                'available_gb': disk_usage.free / (1024**3),
                'usage_percent': (disk_usage.used / disk_usage.total) * 100
            }
        except Exception:
            enhanced['disk_info'] = {'total_gb': 0, 'available_gb': 0, 'usage_percent': 0}
        
        # Add network interface information
        try:
            network_stats = psutil.net_io_counters()
            enhanced['network_info'] = {
                'bytes_sent': network_stats.bytes_sent,
                'bytes_recv': network_stats.bytes_recv,
                'packets_sent': network_stats.packets_sent,
                'packets_recv': network_stats.packets_recv
            }
        except Exception:
            enhanced['network_info'] = {'bytes_sent': 0, 'bytes_recv': 0, 'packets_sent': 0, 'packets_recv': 0}
        
        return enhanced
    
    def calculate_normalization_factors(
        self,
        baseline_environment: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        Calculate normalization factors based on current vs baseline environment.
        
        Args:
            baseline_environment: Baseline environment for comparison, uses default if None
            
        Returns:
            Dict containing normalization factors for different performance aspects
        """
        current_env = self.analyze_current_environment()
        baseline_env = baseline_environment or self._get_default_baseline_environment()
        
        # CPU normalization factor
        cpu_factor = self._calculate_cpu_normalization_factor(current_env, baseline_env)
        
        # Memory normalization factor
        memory_factor = self._calculate_memory_normalization_factor(current_env, baseline_env)
        
        # I/O normalization factor
        io_factor = self._calculate_io_normalization_factor(current_env, baseline_env)
        
        # Overall combined factor
        combined_factor = (cpu_factor + memory_factor + io_factor) / 3
        
        return {
            'cpu': cpu_factor,
            'memory': memory_factor,
            'io': io_factor,
            'combined': combined_factor,
            'platform': current_env['normalization_factors']['platform'],
            'ci': current_env['normalization_factors']['ci'],
            'virtualization': current_env['normalization_factors']['virtualization']
        }
    
    def _calculate_cpu_normalization_factor(
        self,
        current_env: Dict[str, Any],
        baseline_env: Dict[str, Any]
    ) -> float:
        """Calculate CPU performance normalization factor."""
        current_cpu_score = self._calculate_cpu_performance_score(current_env)
        baseline_cpu_score = self._calculate_cpu_performance_score(baseline_env)
        
        return baseline_cpu_score / max(current_cpu_score, 0.1)
    
    def _calculate_memory_normalization_factor(
        self,
        current_env: Dict[str, Any],
        baseline_env: Dict[str, Any]
    ) -> float:
        """Calculate memory performance normalization factor."""
        current_memory_score = self._calculate_memory_performance_score(current_env)
        baseline_memory_score = self._calculate_memory_performance_score(baseline_env)
        
        return baseline_memory_score / max(current_memory_score, 0.1)
    
    def _calculate_io_normalization_factor(
        self,
        current_env: Dict[str, Any],
        baseline_env: Dict[str, Any]
    ) -> float:
        """Calculate I/O performance normalization factor."""
        # Simple I/O factor based on disk availability and system load
        current_disk_factor = (100 - current_env['disk_info']['usage_percent']) / 100
        baseline_disk_factor = (100 - baseline_env.get('disk_info', {}).get('usage_percent', 50)) / 100
        
        current_load_factor = 1.0 / (1.0 + current_env['system_load']['1min'])
        baseline_load_factor = 1.0 / (1.0 + baseline_env.get('system_load', {}).get('1min', 0.5))
        
        current_io_score = (current_disk_factor + current_load_factor) / 2
        baseline_io_score = (baseline_disk_factor + baseline_load_factor) / 2
        
        return baseline_io_score / max(current_io_score, 0.1)
    
    def _calculate_cpu_performance_score(self, env: Dict[str, Any]) -> float:
        """Calculate CPU performance score for environment."""
        cpu_count = env['cpu_count']
        cpu_freq = env['cpu_frequency_mhz'] / 1000.0  # Convert to GHz
        
        # Normalize by reference values
        cpu_count_score = cpu_count / self.config.reference_cpu_cores
        cpu_freq_score = cpu_freq / self.config.reference_cpu_frequency_ghz
        
        # Combined CPU score (geometric mean)
        return np.sqrt(cpu_count_score * cpu_freq_score)
    
    def _calculate_memory_performance_score(self, env: Dict[str, Any]) -> float:
        """Calculate memory performance score for environment."""
        memory_gb = env['memory_gb']
        memory_usage = env['memory_usage_percent'] / 100.0
        
        # Memory availability score
        memory_size_score = memory_gb / self.config.reference_memory_gb
        memory_availability_score = 1.0 - memory_usage
        
        # Combined memory score
        return memory_size_score * memory_availability_score
    
    def _get_default_baseline_environment(self) -> Dict[str, Any]:
        """Get default baseline environment for normalization."""
        if self._baseline_environment is None:
            self._baseline_environment = {
                'cpu_count': self.config.reference_cpu_cores,
                'cpu_frequency_mhz': self.config.reference_cpu_frequency_ghz * 1000,
                'memory_gb': self.config.reference_memory_gb,
                'memory_usage_percent': 20.0,  # Assume 20% baseline usage
                'platform': 'Linux',
                'is_ci_environment': False,
                'is_virtualized': False,
                'disk_info': {'usage_percent': 50.0},
                'system_load': {'1min': 0.5}
            }
        return self._baseline_environment
    
    def normalize_performance_measurement(
        self,
        measurement: float,
        normalization_factors: Dict[str, float] = None
    ) -> float:
        """
        Normalize a performance measurement using environment factors.
        
        Args:
            measurement: Raw performance measurement
            normalization_factors: Normalization factors, calculated if None
            
        Returns:
            Normalized performance measurement
        """
        if normalization_factors is None:
            normalization_factors = self.calculate_normalization_factors()
        
        # Apply combined normalization factor
        normalized_measurement = measurement * normalization_factors['combined']
        
        return normalized_measurement
    
    def generate_environment_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive environment report for performance analysis.
        
        Returns:
            Dict containing detailed environment analysis and recommendations
        """
        current_env = self.analyze_current_environment()
        normalization_factors = self.calculate_normalization_factors()
        
        # Analyze environment suitability for benchmarking
        suitability_analysis = self._analyze_benchmarking_suitability(current_env)
        
        return {
            'environment_analysis': current_env,
            'normalization_factors': normalization_factors,
            'benchmarking_suitability': suitability_analysis,
            'recommendations': self._generate_performance_recommendations(current_env),
            'timestamp': datetime.now().isoformat(),
            'analysis_version': '1.0'
        }
    
    def _analyze_benchmarking_suitability(self, env: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how suitable the current environment is for benchmarking."""
        suitability_score = 100.0
        issues = []
        warnings = []
        
        # Check memory usage
        if env['memory_usage_percent'] > 80:
            suitability_score -= 20
            issues.append("High memory usage may affect benchmark consistency")
        elif env['memory_usage_percent'] > 60:
            suitability_score -= 10
            warnings.append("Moderate memory usage detected")
        
        # Check system load
        if env['system_load']['1min'] > 2.0:
            suitability_score -= 25
            issues.append("High system load may affect benchmark accuracy")
        elif env['system_load']['1min'] > 1.0:
            suitability_score -= 10
            warnings.append("Moderate system load detected")
        
        # Check disk usage
        if env['disk_info']['usage_percent'] > 90:
            suitability_score -= 15
            issues.append("Very high disk usage may affect I/O performance")
        elif env['disk_info']['usage_percent'] > 80:
            suitability_score -= 5
            warnings.append("High disk usage detected")
        
        # Check virtualization
        if env['is_virtualized']:
            suitability_score -= 10
            warnings.append("Virtualized environment may have performance variance")
        
        # Check CI environment
        if env['is_ci_environment']:
            suitability_score -= 5
            warnings.append("CI environment detected - results may vary")
        
        suitability_score = max(suitability_score, 0)
        
        if suitability_score >= 90:
            suitability_level = "excellent"
        elif suitability_score >= 75:
            suitability_level = "good"
        elif suitability_score >= 60:
            suitability_level = "fair"
        else:
            suitability_level = "poor"
        
        return {
            'score': suitability_score,
            'level': suitability_level,
            'issues': issues,
            'warnings': warnings
        }
    
    def _generate_performance_recommendations(self, env: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on environment analysis."""
        recommendations = []
        
        if env['memory_usage_percent'] > 70:
            recommendations.append("Close unnecessary applications to reduce memory usage")
        
        if env['system_load']['1min'] > 1.5:
            recommendations.append("Wait for system load to decrease before running benchmarks")
        
        if env['disk_info']['usage_percent'] > 85:
            recommendations.append("Free up disk space to improve I/O performance")
        
        if env['is_ci_environment']:
            recommendations.append("Use environment normalization factors for CI comparison")
        
        if env['cpu_count'] < self.config.reference_cpu_cores:
            recommendations.append("Consider using parallel test execution with fewer workers")
        
        if not recommendations:
            recommendations.append("Environment is well-suited for performance benchmarking")
        
        return recommendations


# ============================================================================
# PERFORMANCE ARTIFACT GENERATION UTILITIES
# ============================================================================

class PerformanceArtifactGenerator:
    """
    Performance artifact generation utilities for JSON/CSV output formats
    with CI/CD integration support and comprehensive reporting capabilities.
    
    Supports GitHub Actions artifact management, performance trend analysis,
    and automated report generation with 90-day retention policy compliance.
    """
    
    def __init__(self, config: CICDIntegrationConfig = None):
        """
        Initialize performance artifact generator.
        
        Args:
            config: CI/CD integration configuration, uses default if None
        """
        self.config = config or DEFAULT_BENCHMARK_CONFIG.cicd_integration
        self.artifact_metadata = {
            'generator_version': '1.0',
            'generation_timestamp': datetime.now().isoformat(),
            'retention_policy_days': self.config.github_actions_settings['artifact_retention_days']
        }
    
    def generate_json_report(
        self,
        benchmark_results: Dict[str, Any],
        output_path: Path,
        include_environment: bool = True,
        include_historical: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive JSON performance report.
        
        Args:
            benchmark_results: Dictionary containing benchmark results
            output_path: Path to save the JSON report
            include_environment: Whether to include environment analysis
            include_historical: Whether to include historical comparison
            
        Returns:
            Dict containing the generated report structure
        """
        report = {
            'metadata': self.artifact_metadata.copy(),
            'benchmark_results': benchmark_results,
            'summary': self._generate_performance_summary(benchmark_results)
        }
        
        # Add environment analysis if requested
        if include_environment:
            env_analyzer = EnvironmentAnalyzer()
            report['environment'] = env_analyzer.generate_environment_report()
        
        # Add historical comparison if requested
        if include_historical:
            report['historical_analysis'] = self._generate_historical_analysis(benchmark_results)
        
        # Add CI/CD integration metadata
        report['ci_integration'] = {
            'artifact_settings': self.config.artifact_settings,
            'github_actions': self._get_github_actions_context()
        }
        
        # Write JSON report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def generate_csv_report(
        self,
        benchmark_results: Dict[str, Any],
        output_path: Path,
        include_statistics: bool = True
    ) -> pd.DataFrame:
        """
        Generate CSV performance report for spreadsheet analysis.
        
        Args:
            benchmark_results: Dictionary containing benchmark results
            output_path: Path to save the CSV report
            include_statistics: Whether to include detailed statistics
            
        Returns:
            DataFrame containing the generated report data
        """
        # Flatten benchmark results for CSV format
        csv_data = []
        
        for test_name, test_results in benchmark_results.items():
            if isinstance(test_results, dict) and 'stats' in test_results:
                row = {
                    'test_name': test_name,
                    'mean_seconds': test_results['stats'].get('mean', 0),
                    'min_seconds': test_results['stats'].get('min', 0),
                    'max_seconds': test_results['stats'].get('max', 0),
                    'stddev_seconds': test_results['stats'].get('stddev', 0),
                    'median_seconds': test_results['stats'].get('median', 0),
                    'iterations': test_results['stats'].get('iterations', 0),
                    'rounds': test_results['stats'].get('rounds', 0)
                }
                
                # Add SLA compliance if available
                if 'sla_compliance' in test_results:
                    sla_info = test_results['sla_compliance']
                    row.update({
                        'sla_threshold_seconds': sla_info.get('sla_threshold_seconds', 0),
                        'sla_compliant': sla_info.get('compliant', False),
                        'performance_margin_percent': sla_info.get('performance_margin', 0)
                    })
                
                # Add memory statistics if available
                if 'memory_stats' in test_results:
                    memory_info = test_results['memory_stats']
                    row.update({
                        'peak_memory_mb': memory_info.get('peak_memory_mb', 0),
                        'memory_multiplier': memory_info.get('memory_multiplier', 0),
                        'memory_efficient': memory_info.get('meets_efficiency_requirement', False)
                    })
                
                # Add regression analysis if available
                if 'regression_analysis' in test_results:
                    regression_info = test_results['regression_analysis']
                    row.update({
                        'regression_detected': regression_info.get('regression_detected', False),
                        'percent_change': regression_info.get('percent_change', 0),
                        'regression_confidence': regression_info.get('confidence', 0)
                    })
                
                # Add timestamp and environment info
                row.update({
                    'timestamp': self.artifact_metadata['generation_timestamp'],
                    'platform': platform.system(),
                    'python_version': platform.python_version()
                })
                
                csv_data.append(row)
        
        # Create DataFrame and save CSV
        df = pd.DataFrame(csv_data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        return df
    
    def generate_html_report(
        self,
        benchmark_results: Dict[str, Any],
        output_path: Path,
        include_charts: bool = True
    ) -> str:
        """
        Generate HTML performance report with visualization.
        
        Args:
            benchmark_results: Dictionary containing benchmark results
            output_path: Path to save the HTML report
            include_charts: Whether to include performance charts
            
        Returns:
            String containing the generated HTML content
        """
        # Generate comprehensive HTML report
        html_content = self._generate_html_template(benchmark_results, include_charts)
        
        # Write HTML report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return html_content
    
    def _generate_performance_summary(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary statistics."""
        total_tests = len(benchmark_results)
        passed_tests = 0
        failed_tests = 0
        sla_compliant_tests = 0
        memory_efficient_tests = 0
        
        total_execution_time = 0
        performance_scores = []
        
        for test_name, test_results in benchmark_results.items():
            if isinstance(test_results, dict):
                # Count test outcomes
                if test_results.get('status') == 'passed':
                    passed_tests += 1
                else:
                    failed_tests += 1
                
                # Check SLA compliance
                if test_results.get('sla_compliance', {}).get('compliant', False):
                    sla_compliant_tests += 1
                
                # Check memory efficiency
                if test_results.get('memory_stats', {}).get('meets_efficiency_requirement', False):
                    memory_efficient_tests += 1
                
                # Calculate execution time
                if 'stats' in test_results:
                    test_time = test_results['stats'].get('mean', 0) * test_results['stats'].get('iterations', 1)
                    total_execution_time += test_time
                    
                    # Calculate performance score (inverse of execution time)
                    if test_results['stats'].get('mean', 0) > 0:
                        performance_scores.append(1.0 / test_results['stats']['mean'])
        
        # Calculate overall metrics
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        sla_compliance_rate = (sla_compliant_tests / total_tests) * 100 if total_tests > 0 else 0
        memory_efficiency_rate = (memory_efficient_tests / total_tests) * 100 if total_tests > 0 else 0
        avg_performance_score = np.mean(performance_scores) if performance_scores else 0
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'pass_rate_percent': round(pass_rate, 2),
            'sla_compliant_tests': sla_compliant_tests,
            'sla_compliance_rate_percent': round(sla_compliance_rate, 2),
            'memory_efficient_tests': memory_efficient_tests,
            'memory_efficiency_rate_percent': round(memory_efficiency_rate, 2),
            'total_execution_time_seconds': round(total_execution_time, 3),
            'average_performance_score': round(avg_performance_score, 3),
            'performance_grade': self._calculate_performance_grade(pass_rate, sla_compliance_rate, memory_efficiency_rate)
        }
    
    def _calculate_performance_grade(
        self,
        pass_rate: float,
        sla_compliance_rate: float,
        memory_efficiency_rate: float
    ) -> str:
        """Calculate overall performance grade."""
        overall_score = (pass_rate + sla_compliance_rate + memory_efficiency_rate) / 3
        
        if overall_score >= 95:
            return 'A+'
        elif overall_score >= 90:
            return 'A'
        elif overall_score >= 85:
            return 'B+'
        elif overall_score >= 80:
            return 'B'
        elif overall_score >= 75:
            return 'C+'
        elif overall_score >= 70:
            return 'C'
        elif overall_score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _generate_historical_analysis(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate historical performance comparison analysis."""
        # This would typically load historical data from storage
        # For now, return placeholder structure
        return {
            'comparison_available': False,
            'baseline_date': None,
            'trend_analysis': 'insufficient_historical_data',
            'performance_trends': {},
            'recommendations': ['Collect more historical data for trend analysis']
        }
    
    def _get_github_actions_context(self) -> Dict[str, Any]:
        """Get GitHub Actions environment context."""
        github_context = {}
        
        # GitHub Actions environment variables
        github_env_vars = [
            'GITHUB_WORKFLOW', 'GITHUB_RUN_ID', 'GITHUB_RUN_NUMBER',
            'GITHUB_REF', 'GITHUB_SHA', 'GITHUB_REPOSITORY',
            'GITHUB_ACTOR', 'GITHUB_EVENT_NAME'
        ]
        
        for var in github_env_vars:
            github_context[var.lower()] = os.getenv(var)
        
        # Add runner information
        github_context['runner_os'] = os.getenv('RUNNER_OS')
        github_context['runner_arch'] = os.getenv('RUNNER_ARCH')
        
        return github_context
    
    def _generate_html_template(self, benchmark_results: Dict[str, Any], include_charts: bool) -> str:
        """Generate HTML template for performance report."""
        summary = self._generate_performance_summary(benchmark_results)
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlyRigLoader Performance Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric {{ background-color: #e9e9e9; padding: 15px; border-radius: 5px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .metric-label {{ font-size: 12px; color: #666; }}
        .test-results {{ margin: 20px 0; }}
        .test-item {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .test-passed {{ border-left: 5px solid #4CAF50; }}
        .test-failed {{ border-left: 5px solid #f44336; }}
        .grade-A {{ color: #4CAF50; }}
        .grade-B {{ color: #FF9800; }}
        .grade-C {{ color: #FFC107; }}
        .grade-D {{ color: #FF5722; }}
        .grade-F {{ color: #f44336; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>FlyRigLoader Performance Benchmark Report</h1>
        <p>Generated: {self.artifact_metadata['generation_timestamp']}</p>
        <p>Platform: {platform.system()} - Python {platform.python_version()}</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <div class="metric-value">{summary['total_tests']}</div>
            <div class="metric-label">Total Tests</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary['pass_rate_percent']}%</div>
            <div class="metric-label">Pass Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary['sla_compliance_rate_percent']}%</div>
            <div class="metric-label">SLA Compliance</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary['memory_efficiency_rate_percent']}%</div>
            <div class="metric-label">Memory Efficiency</div>
        </div>
        <div class="metric">
            <div class="metric-value grade-{summary['performance_grade'].replace('+', '')}">{summary['performance_grade']}</div>
            <div class="metric-label">Performance Grade</div>
        </div>
    </div>
    
    <div class="test-results">
        <h2>Test Results</h2>
        {self._generate_test_results_html(benchmark_results)}
    </div>
</body>
</html>
        """
        
        return html_template
    
    def _generate_test_results_html(self, benchmark_results: Dict[str, Any]) -> str:
        """Generate HTML for individual test results."""
        html_parts = []
        
        for test_name, test_results in benchmark_results.items():
            if isinstance(test_results, dict):
                status = test_results.get('status', 'unknown')
                stats = test_results.get('stats', {})
                
                status_class = 'test-passed' if status == 'passed' else 'test-failed'
                
                html_parts.append(f"""
                <div class="test-item {status_class}">
                    <h3>{test_name}</h3>
                    <p><strong>Status:</strong> {status}</p>
                    <p><strong>Mean Time:</strong> {stats.get('mean', 0):.4f}s</p>
                    <p><strong>Min/Max:</strong> {stats.get('min', 0):.4f}s / {stats.get('max', 0):.4f}s</p>
                    <p><strong>Standard Deviation:</strong> {stats.get('stddev', 0):.4f}s</p>
                    <p><strong>Iterations:</strong> {stats.get('iterations', 0)}</p>
                </div>
                """)
        
        return ''.join(html_parts)


# ============================================================================
# REGRESSION DETECTION UTILITIES
# ============================================================================

class RegressionDetector:
    """
    Automated performance regression detection with statistical significance testing
    for performance baseline maintenance and alerting.
    
    Implements comprehensive regression detection algorithms with confidence
    intervals, trend analysis, and automated alerting for CI/CD integration.
    """
    
    def __init__(
        self,
        statistical_engine: StatisticalAnalysisEngine = None,
        config: StatisticalAnalysisConfig = None
    ):
        """
        Initialize regression detector.
        
        Args:
            statistical_engine: Statistical analysis engine, creates new if None
            config: Statistical analysis configuration, uses default if None
        """
        self.config = config or DEFAULT_BENCHMARK_CONFIG.statistical_analysis
        self.statistical_engine = statistical_engine or StatisticalAnalysisEngine(self.config)
        self.regression_history = {}
        self.alert_thresholds = {
            'critical': 50.0,  # >50% performance degradation
            'warning': 25.0,   # >25% performance degradation
            'info': 10.0       # >10% performance degradation
        }
    
    def detect_regression(
        self,
        test_name: str,
        current_measurements: List[float],
        baseline_measurements: List[float] = None,
        custom_threshold: float = None
    ) -> Dict[str, Any]:
        """
        Detect performance regression for a specific test.
        
        Args:
            test_name: Name of the test for regression tracking
            current_measurements: Current performance measurements
            baseline_measurements: Baseline measurements, uses cached if None
            custom_threshold: Custom regression threshold, uses config default if None
            
        Returns:
            Dict containing comprehensive regression analysis
        """
        # Use cached baseline if not provided
        if baseline_measurements is None:
            baseline_measurements = self.statistical_engine.get_baseline_measurements(test_name)
        
        if not baseline_measurements:
            return {
                'test_name': test_name,
                'regression_detected': False,
                'status': 'no_baseline',
                'message': 'No baseline measurements available for regression detection',
                'recommendations': ['Establish baseline measurements by running tests multiple times']
            }
        
        # Perform regression detection using statistical engine
        regression_analysis = self.statistical_engine.detect_performance_regression(
            baseline_measurements=baseline_measurements,
            current_measurements=current_measurements,
            regression_threshold=custom_threshold
        )
        
        # Enhance with trend analysis
        trend_analysis = self._analyze_performance_trend(test_name, current_measurements)
        
        # Generate regression alerts
        alerts = self._generate_regression_alerts(test_name, regression_analysis)
        
        # Update regression history
        self._update_regression_history(test_name, regression_analysis)
        
        # Compile comprehensive regression report
        regression_report = {
            'test_name': test_name,
            'timestamp': datetime.now().isoformat(),
            'regression_analysis': regression_analysis,
            'trend_analysis': trend_analysis,
            'alerts': alerts,
            'recommendations': self._generate_regression_recommendations(regression_analysis, trend_analysis),
            'historical_context': self._get_historical_regression_context(test_name)
        }
        
        return regression_report
    
    def batch_regression_detection(
        self,
        benchmark_results: Dict[str, Any],
        baseline_data: Dict[str, List[float]] = None
    ) -> Dict[str, Any]:
        """
        Perform regression detection for multiple tests in batch.
        
        Args:
            benchmark_results: Dictionary containing benchmark results for multiple tests
            baseline_data: Dictionary mapping test names to baseline measurements
            
        Returns:
            Dict containing batch regression analysis results
        """
        batch_results = {
            'timestamp': datetime.now().isoformat(),
            'total_tests_analyzed': 0,
            'regressions_detected': 0,
            'critical_regressions': 0,
            'warning_regressions': 0,
            'test_results': {},
            'overall_status': 'unknown',
            'summary_alerts': []
        }
        
        for test_name, test_results in benchmark_results.items():
            if isinstance(test_results, dict) and 'measurements' in test_results:
                current_measurements = test_results['measurements']
                baseline_measurements = baseline_data.get(test_name) if baseline_data else None
                
                # Perform regression detection for this test
                regression_result = self.detect_regression(
                    test_name=test_name,
                    current_measurements=current_measurements,
                    baseline_measurements=baseline_measurements
                )
                
                batch_results['test_results'][test_name] = regression_result
                batch_results['total_tests_analyzed'] += 1
                
                # Count regressions by severity
                if regression_result['regression_analysis']['regression_detected']:
                    batch_results['regressions_detected'] += 1
                    
                    # Determine severity based on percent change
                    percent_change = abs(regression_result['regression_analysis']['percent_change'])
                    if percent_change > self.alert_thresholds['critical']:
                        batch_results['critical_regressions'] += 1
                    elif percent_change > self.alert_thresholds['warning']:
                        batch_results['warning_regressions'] += 1
        
        # Determine overall status
        if batch_results['critical_regressions'] > 0:
            batch_results['overall_status'] = 'critical'
        elif batch_results['warning_regressions'] > 0:
            batch_results['overall_status'] = 'warning'
        elif batch_results['regressions_detected'] > 0:
            batch_results['overall_status'] = 'info'
        else:
            batch_results['overall_status'] = 'success'
        
        # Generate summary alerts
        batch_results['summary_alerts'] = self._generate_batch_summary_alerts(batch_results)
        
        return batch_results
    
    def _analyze_performance_trend(
        self,
        test_name: str,
        measurements: List[float]
    ) -> Dict[str, Any]:
        """Analyze performance trend within current measurements."""
        if len(measurements) < 3:
            return {
                'trend': 'insufficient_data',
                'confidence': 0,
                'direction': 'unknown'
            }
        
        # Calculate linear regression for trend
        x = np.arange(len(measurements))
        y = np.array(measurements)
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Determine trend direction and significance
            if p_value < 0.05:  # Statistically significant trend
                if slope > 0:
                    direction = 'deteriorating'
                    trend = 'increasing_time'
                else:
                    direction = 'improving'
                    trend = 'decreasing_time'
                confidence = 1.0 - p_value
            else:
                direction = 'stable'
                trend = 'no_significant_trend'
                confidence = 0
            
            return {
                'trend': trend,
                'direction': direction,
                'confidence': confidence,
                'slope': slope,
                'correlation': r_value,
                'p_value': p_value,
                'percent_change_per_measurement': (slope / np.mean(y)) * 100 if np.mean(y) != 0 else 0
            }
            
        except Exception as e:
            return {
                'trend': 'analysis_error',
                'error': str(e),
                'confidence': 0,
                'direction': 'unknown'
            }
    
    def _generate_regression_alerts(
        self,
        test_name: str,
        regression_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate appropriate alerts based on regression analysis."""
        alerts = []
        
        if not regression_analysis['regression_detected']:
            return alerts
        
        percent_change = abs(regression_analysis['percent_change'])
        
        # Determine alert level
        if percent_change > self.alert_thresholds['critical']:
            alert_level = 'critical'
            alert_message = f"CRITICAL: {test_name} performance degraded by {percent_change:.1f}%"
        elif percent_change > self.alert_thresholds['warning']:
            alert_level = 'warning'
            alert_message = f"WARNING: {test_name} performance degraded by {percent_change:.1f}%"
        else:
            alert_level = 'info'
            alert_message = f"INFO: {test_name} performance degraded by {percent_change:.1f}%"
        
        alerts.append({
            'level': alert_level,
            'message': alert_message,
            'test_name': test_name,
            'percent_change': percent_change,
            'confidence': regression_analysis['confidence'],
            'timestamp': datetime.now().isoformat()
        })
        
        return alerts
    
    def _generate_regression_recommendations(
        self,
        regression_analysis: Dict[str, Any],
        trend_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on regression and trend analysis."""
        recommendations = []
        
        if regression_analysis['regression_detected']:
            recommendations.append("Investigate recent changes that may have affected performance")
            recommendations.append("Run additional measurements to confirm regression")
            
            if regression_analysis['confidence'] > 0.8:
                recommendations.append("High confidence regression detected - immediate investigation recommended")
            
            if trend_analysis['direction'] == 'deteriorating':
                recommendations.append("Performance trend is deteriorating - monitor closely")
        
        if not regression_analysis['statistically_significant']:
            recommendations.append("Regression not statistically significant - may be normal variance")
        
        if regression_analysis['baseline_stats']['sample_size'] < 5:
            recommendations.append("Increase baseline sample size for more reliable regression detection")
        
        return recommendations
    
    def _update_regression_history(
        self,
        test_name: str,
        regression_analysis: Dict[str, Any]
    ):
        """Update regression detection history for trend tracking."""
        if test_name not in self.regression_history:
            self.regression_history[test_name] = []
        
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'regression_detected': regression_analysis['regression_detected'],
            'percent_change': regression_analysis['percent_change'],
            'confidence': regression_analysis['confidence']
        }
        
        self.regression_history[test_name].append(history_entry)
        
        # Keep only recent history (last 30 entries)
        if len(self.regression_history[test_name]) > 30:
            self.regression_history[test_name] = self.regression_history[test_name][-30:]
    
    def _get_historical_regression_context(self, test_name: str) -> Dict[str, Any]:
        """Get historical context for regression detection."""
        if test_name not in self.regression_history:
            return {
                'historical_entries': 0,
                'recent_regressions': 0,
                'pattern': 'no_history'
            }
        
        history = self.regression_history[test_name]
        recent_regressions = sum(1 for entry in history[-10:] if entry['regression_detected'])
        
        return {
            'historical_entries': len(history),
            'recent_regressions': recent_regressions,
            'pattern': 'frequent_regressions' if recent_regressions > 3 else 'stable'
        }
    
    def _generate_batch_summary_alerts(self, batch_results: Dict[str, Any]) -> List[str]:
        """Generate summary alerts for batch regression detection."""
        alerts = []
        
        if batch_results['critical_regressions'] > 0:
            alerts.append(f"CRITICAL: {batch_results['critical_regressions']} tests have critical performance regressions")
        
        if batch_results['warning_regressions'] > 0:
            alerts.append(f"WARNING: {batch_results['warning_regressions']} tests have performance regressions")
        
        if batch_results['regressions_detected'] == 0:
            alerts.append("SUCCESS: No performance regressions detected")
        
        regression_rate = (batch_results['regressions_detected'] / batch_results['total_tests_analyzed']) * 100
        if regression_rate > 25:
            alerts.append(f"HIGH REGRESSION RATE: {regression_rate:.1f}% of tests showing regressions")
        
        return alerts


# ============================================================================
# CROSS-PLATFORM PERFORMANCE VALIDATION UTILITIES
# ============================================================================

class CrossPlatformValidator:
    """
    Cross-platform performance validation utilities ensuring consistent results
    across Ubuntu, Windows, macOS with hardware abstraction and normalization.
    
    Provides platform-specific performance baselines, hardware normalization
    factors, and cross-platform consistency validation for CI/CD environments.
    """
    
    def __init__(
        self,
        env_config: EnvironmentNormalizationConfig = None,
        platforms: List[str] = None
    ):
        """
        Initialize cross-platform validator.
        
        Args:
            env_config: Environment normalization configuration
            platforms: List of target platforms, uses default if None
        """
        self.env_config = env_config or DEFAULT_BENCHMARK_CONFIG.environment_normalization
        self.platforms = platforms or ['Windows', 'Linux', 'Darwin']
        self.platform_baselines = {}
        self.platform_factors = self.env_config.platform_performance_factors
    
    def validate_cross_platform_consistency(
        self,
        platform_results: Dict[str, Dict[str, Any]],
        tolerance: float = None
    ) -> Dict[str, Any]:
        """
        Validate performance consistency across multiple platforms.
        
        Args:
            platform_results: Dict mapping platform names to benchmark results
            tolerance: Allowed variance percentage, uses config default if None
            
        Returns:
            Dict containing cross-platform validation analysis
        """
        tolerance = tolerance or self.env_config.cross_platform_variance_limit * 100
        
        validation_results = {
            'platforms_tested': list(platform_results.keys()),
            'consistency_analysis': {},
            'overall_consistent': True,
            'variance_violations': [],
            'platform_recommendations': {}
        }
        
        # Analyze each test across platforms
        all_test_names = set()
        for platform_data in platform_results.values():
            all_test_names.update(platform_data.keys())
        
        for test_name in all_test_names:
            test_analysis = self._analyze_test_cross_platform_consistency(
                test_name=test_name,
                platform_results=platform_results,
                tolerance=tolerance
            )
            
            validation_results['consistency_analysis'][test_name] = test_analysis
            
            if not test_analysis['consistent']:
                validation_results['overall_consistent'] = False
                validation_results['variance_violations'].append({
                    'test_name': test_name,
                    'variance_percent': test_analysis['variance_percent'],
                    'tolerance_percent': tolerance
                })
        
        # Generate platform-specific recommendations
        validation_results['platform_recommendations'] = self._generate_platform_recommendations(
            platform_results, validation_results['consistency_analysis']
        )
        
        return validation_results
    
    def _analyze_test_cross_platform_consistency(
        self,
        test_name: str,
        platform_results: Dict[str, Dict[str, Any]],
        tolerance: float
    ) -> Dict[str, Any]:
        """Analyze consistency of a specific test across platforms."""
        # Extract performance measurements for this test from each platform
        platform_measurements = {}
        normalized_measurements = {}
        
        for platform, results in platform_results.items():
            if test_name in results and 'mean_time' in results[test_name]:
                raw_time = results[test_name]['mean_time']
                platform_measurements[platform] = raw_time
                
                # Apply platform normalization factor
                normalization_factor = self.platform_factors.get(platform, 1.0)
                normalized_measurements[platform] = raw_time * normalization_factor
        
        if len(platform_measurements) < 2:
            return {
                'consistent': True,
                'reason': 'insufficient_platforms',
                'platforms_available': len(platform_measurements)
            }
        
        # Calculate variance statistics
        raw_values = list(platform_measurements.values())
        normalized_values = list(normalized_measurements.values())
        
        raw_variance = (np.std(raw_values) / np.mean(raw_values)) * 100 if np.mean(raw_values) > 0 else 0
        normalized_variance = (np.std(normalized_values) / np.mean(normalized_values)) * 100 if np.mean(normalized_values) > 0 else 0
        
        # Determine consistency
        consistent = normalized_variance <= tolerance
        
        return {
            'consistent': consistent,
            'raw_variance_percent': raw_variance,
            'normalized_variance_percent': normalized_variance,
            'tolerance_percent': tolerance,
            'platform_measurements': platform_measurements,
            'normalized_measurements': normalized_measurements,
            'improvement_from_normalization': raw_variance - normalized_variance,
            'platform_factor_effectiveness': raw_variance > normalized_variance
        }
    
    def _generate_platform_recommendations(
        self,
        platform_results: Dict[str, Dict[str, Any]],
        consistency_analysis: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Generate platform-specific performance recommendations."""
        recommendations = {}
        
        for platform in platform_results.keys():
            platform_recs = []
            
            # Analyze this platform's performance relative to others
            platform_performance_issues = 0
            for test_name, analysis in consistency_analysis.items():
                if not analysis.get('consistent', True):
                    platform_measurements = analysis.get('platform_measurements', {})
                    if platform in platform_measurements:
                        # Check if this platform is an outlier
                        platform_time = platform_measurements[platform]
                        other_times = [time for p, time in platform_measurements.items() if p != platform]
                        
                        if other_times:
                            avg_other_time = np.mean(other_times)
                            if platform_time > avg_other_time * 1.2:  # 20% slower
                                platform_performance_issues += 1
            
            # Generate recommendations based on issues found
            if platform_performance_issues > 0:
                platform_recs.append(f"Platform showing slower performance in {platform_performance_issues} tests")
                platform_recs.append("Consider optimizing for this platform or adjusting normalization factors")
            
            if platform == 'Windows' and platform_performance_issues > 0:
                platform_recs.append("Windows-specific optimizations may be needed")
            elif platform == 'Darwin' and platform_performance_issues > 0:
                platform_recs.append("macOS-specific performance tuning recommended")
            
            if not platform_recs:
                platform_recs.append("Platform performance is consistent with others")
            
            recommendations[platform] = platform_recs
        
        return recommendations
    
    def normalize_platform_results(
        self,
        platform_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Apply platform normalization factors to benchmark results.
        
        Args:
            platform_results: Raw platform benchmark results
            
        Returns:
            Dict containing normalized platform results
        """
        normalized_results = {}
        
        for platform, results in platform_results.items():
            normalization_factor = self.platform_factors.get(platform, 1.0)
            normalized_platform_results = {}
            
            for test_name, test_data in results.items():
                if isinstance(test_data, dict):
                    normalized_test_data = test_data.copy()
                    
                    # Normalize timing measurements
                    timing_fields = ['mean_time', 'min_time', 'max_time', 'median_time']
                    for field in timing_fields:
                        if field in normalized_test_data:
                            normalized_test_data[field] = normalized_test_data[field] * normalization_factor
                    
                    # Add normalization metadata
                    normalized_test_data['normalization_applied'] = True
                    normalized_test_data['normalization_factor'] = normalization_factor
                    normalized_test_data['original_platform'] = platform
                    
                    normalized_platform_results[test_name] = normalized_test_data
                else:
                    normalized_platform_results[test_name] = test_data
            
            normalized_results[platform] = normalized_platform_results
        
        return normalized_results
    
    def generate_cross_platform_report(
        self,
        platform_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive cross-platform performance analysis report.
        
        Args:
            platform_results: Platform benchmark results
            
        Returns:
            Dict containing detailed cross-platform analysis
        """
        # Normalize results
        normalized_results = self.normalize_platform_results(platform_results)
        
        # Validate consistency
        consistency_analysis = self.validate_cross_platform_consistency(platform_results)
        
        # Generate platform comparison statistics
        platform_stats = self._generate_platform_statistics(platform_results, normalized_results)
        
        # Compile comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'platforms_analyzed': list(platform_results.keys()),
            'raw_results': platform_results,
            'normalized_results': normalized_results,
            'consistency_analysis': consistency_analysis,
            'platform_statistics': platform_stats,
            'normalization_effectiveness': self._analyze_normalization_effectiveness(
                platform_results, normalized_results
            ),
            'recommendations': self._generate_comprehensive_platform_recommendations(
                consistency_analysis, platform_stats
            )
        }
        
        return report
    
    def _generate_platform_statistics(
        self,
        raw_results: Dict[str, Dict[str, Any]],
        normalized_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate statistical comparison across platforms."""
        stats = {
            'platform_performance_ranking': {},
            'variance_reduction_summary': {},
            'relative_performance_factors': {}
        }
        
        # Calculate average performance per platform
        platform_averages = {}
        for platform, results in normalized_results.items():
            times = []
            for test_data in results.values():
                if isinstance(test_data, dict) and 'mean_time' in test_data:
                    times.append(test_data['mean_time'])
            
            if times:
                platform_averages[platform] = np.mean(times)
        
        # Rank platforms by performance
        sorted_platforms = sorted(platform_averages.items(), key=lambda x: x[1])
        stats['platform_performance_ranking'] = {
            platform: {'rank': i+1, 'avg_time': avg_time}
            for i, (platform, avg_time) in enumerate(sorted_platforms)
        }
        
        return stats
    
    def _analyze_normalization_effectiveness(
        self,
        raw_results: Dict[str, Dict[str, Any]],
        normalized_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze effectiveness of platform normalization."""
        effectiveness = {
            'variance_reduction': {},
            'overall_improvement': False,
            'normalization_impact': {}
        }
        
        # Compare variance before and after normalization for each test
        all_test_names = set()
        for platform_data in raw_results.values():
            all_test_names.update(platform_data.keys())
        
        total_variance_reduction = 0
        valid_tests = 0
        
        for test_name in all_test_names:
            raw_times = []
            normalized_times = []
            
            for platform in raw_results.keys():
                if (test_name in raw_results[platform] and 
                    test_name in normalized_results[platform]):
                    
                    raw_data = raw_results[platform][test_name]
                    norm_data = normalized_results[platform][test_name]
                    
                    if isinstance(raw_data, dict) and 'mean_time' in raw_data:
                        raw_times.append(raw_data['mean_time'])
                        normalized_times.append(norm_data['mean_time'])
            
            if len(raw_times) >= 2:
                raw_variance = np.std(raw_times) / np.mean(raw_times) if np.mean(raw_times) > 0 else 0
                norm_variance = np.std(normalized_times) / np.mean(normalized_times) if np.mean(normalized_times) > 0 else 0
                
                variance_reduction = (raw_variance - norm_variance) / raw_variance if raw_variance > 0 else 0
                effectiveness['variance_reduction'][test_name] = variance_reduction
                
                total_variance_reduction += variance_reduction
                valid_tests += 1
        
        if valid_tests > 0:
            avg_variance_reduction = total_variance_reduction / valid_tests
            effectiveness['overall_improvement'] = avg_variance_reduction > 0
            effectiveness['average_variance_reduction'] = avg_variance_reduction
        
        return effectiveness
    
    def _generate_comprehensive_platform_recommendations(
        self,
        consistency_analysis: Dict[str, Any],
        platform_stats: Dict[str, Any]
    ) -> List[str]:
        """Generate comprehensive recommendations for cross-platform optimization."""
        recommendations = []
        
        if not consistency_analysis['overall_consistent']:
            recommendations.append("Cross-platform performance inconsistencies detected")
            recommendations.append("Consider adjusting normalization factors for affected platforms")
        
        # Analyze platform ranking
        ranking = platform_stats.get('platform_performance_ranking', {})
        if ranking:
            fastest_platform = min(ranking.items(), key=lambda x: x[1]['avg_time'])[0]
            slowest_platform = max(ranking.items(), key=lambda x: x[1]['avg_time'])[0]
            
            recommendations.append(f"Fastest platform: {fastest_platform}")
            recommendations.append(f"Slowest platform: {slowest_platform}")
            
            if len(ranking) > 2:
                recommendations.append("Consider platform-specific optimizations for consistent performance")
        
        variance_violations = consistency_analysis.get('variance_violations', [])
        if len(variance_violations) > 0:
            recommendations.append(f"{len(variance_violations)} tests exceed cross-platform variance tolerance")
            recommendations.append("Focus optimization efforts on these high-variance tests")
        
        return recommendations


# ============================================================================
# CI/CD INTEGRATION UTILITIES
# ============================================================================

class CICDIntegrationManager:
    """
    CI/CD integration utilities for GitHub Actions artifact management,
    performance alerting, and automated benchmark workflow coordination.
    
    Provides comprehensive integration with GitHub Actions workflows,
    automated artifact collection, performance trend monitoring, and
    CI/CD pipeline optimization for benchmark test execution.
    """
    
    def __init__(self, config: CICDIntegrationConfig = None):
        """
        Initialize CI/CD integration manager.
        
        Args:
            config: CI/CD integration configuration, uses default if None
        """
        self.config = config or DEFAULT_BENCHMARK_CONFIG.cicd_integration
        self.is_github_actions = self._detect_github_actions_environment()
        self.workflow_context = self._get_workflow_context()
    
    def _detect_github_actions_environment(self) -> bool:
        """Detect if running in GitHub Actions environment."""
        return os.getenv('GITHUB_ACTIONS') == 'true'
    
    def _get_workflow_context(self) -> Dict[str, str]:
        """Get GitHub Actions workflow context information."""
        context = {}
        
        github_env_vars = [
            'GITHUB_WORKFLOW', 'GITHUB_RUN_ID', 'GITHUB_RUN_NUMBER',
            'GITHUB_REF', 'GITHUB_SHA', 'GITHUB_REPOSITORY',
            'GITHUB_ACTOR', 'GITHUB_EVENT_NAME', 'GITHUB_HEAD_REF',
            'GITHUB_BASE_REF', 'RUNNER_OS', 'RUNNER_ARCH'
        ]
        
        for var in github_env_vars:
            context[var.lower()] = os.getenv(var, '')
        
        return context
    
    def setup_artifact_directories(self, base_path: Path = None) -> Dict[str, Path]:
        """
        Setup artifact directories for benchmark output with retention policy compliance.
        
        Args:
            base_path: Base path for artifacts, uses current directory if None
            
        Returns:
            Dict mapping artifact types to their directory paths
        """
        if base_path is None:
            base_path = Path.cwd() / 'benchmark-artifacts'
        
        # Create artifact directory structure
        artifact_dirs = {
            'reports': base_path / 'reports',
            'raw_data': base_path / 'raw-data',
            'charts': base_path / 'charts',
            'logs': base_path / 'logs',
            'historical': base_path / 'historical'
        }
        
        # Create directories
        for artifact_type, dir_path in artifact_dirs.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Add README file explaining retention policy
            readme_path = dir_path / 'README.md'
            if not readme_path.exists():
                self._create_artifact_readme(readme_path, artifact_type)
        
        return artifact_dirs
    
    def _create_artifact_readme(self, readme_path: Path, artifact_type: str):
        """Create README file for artifact directory with retention policy information."""
        retention_days = self.config.github_actions_settings['artifact_retention_days']
        
        readme_content = f"""# {artifact_type.title()} Artifacts

This directory contains {artifact_type} generated by the FlyRigLoader benchmark test suite.

## Retention Policy
- Artifacts are retained for {retention_days} days as per CI/CD configuration
- Historical data is automatically cleaned up after retention period
- Critical performance baselines may be retained longer for regression detection

## Contents
Generated automatically by the benchmark test suite with comprehensive performance analysis.

## Integration
These artifacts are automatically collected by GitHub Actions workflows and made available for download and analysis.
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
    
    def collect_benchmark_artifacts(
        self,
        benchmark_results: Dict[str, Any],
        artifact_dirs: Dict[str, Path] = None,
        generate_all_formats: bool = True
    ) -> Dict[str, Path]:
        """
        Collect and organize benchmark artifacts for CI/CD integration.
        
        Args:
            benchmark_results: Comprehensive benchmark results
            artifact_dirs: Artifact directories, creates default if None
            generate_all_formats: Whether to generate all output formats
            
        Returns:
            Dict mapping artifact types to their file paths
        """
        if artifact_dirs is None:
            artifact_dirs = self.setup_artifact_directories()
        
        # Generate timestamp for artifact naming
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        platform_name = platform.system().lower()
        
        # Create artifact generator
        artifact_generator = PerformanceArtifactGenerator(self.config)
        
        collected_artifacts = {}
        
        # Generate JSON report
        if generate_all_formats:
            json_path = artifact_dirs['reports'] / f'benchmark-report-{platform_name}-{timestamp}.json'
            artifact_generator.generate_json_report(
                benchmark_results=benchmark_results,
                output_path=json_path,
                include_environment=True,
                include_historical=True
            )
            collected_artifacts['json_report'] = json_path
        
        # Generate CSV report
        if generate_all_formats:
            csv_path = artifact_dirs['reports'] / f'benchmark-data-{platform_name}-{timestamp}.csv'
            artifact_generator.generate_csv_report(
                benchmark_results=benchmark_results,
                output_path=csv_path,
                include_statistics=True
            )
            collected_artifacts['csv_report'] = csv_path
        
        # Generate HTML report
        if generate_all_formats:
            html_path = artifact_dirs['reports'] / f'benchmark-report-{platform_name}-{timestamp}.html'
            artifact_generator.generate_html_report(
                benchmark_results=benchmark_results,
                output_path=html_path,
                include_charts=True
            )
            collected_artifacts['html_report'] = html_path
        
        # Save raw benchmark data
        raw_data_path = artifact_dirs['raw_data'] / f'raw-results-{platform_name}-{timestamp}.json'
        with open(raw_data_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        collected_artifacts['raw_data'] = raw_data_path
        
        # Generate GitHub Actions summary if in CI
        if self.is_github_actions:
            summary_path = self._generate_github_actions_summary(benchmark_results, artifact_dirs)
            if summary_path:
                collected_artifacts['github_summary'] = summary_path
        
        return collected_artifacts
    
    def _generate_github_actions_summary(
        self,
        benchmark_results: Dict[str, Any],
        artifact_dirs: Dict[str, Path]
    ) -> Optional[Path]:
        """Generate GitHub Actions job summary with performance highlights."""
        try:
            # Generate performance summary
            artifact_generator = PerformanceArtifactGenerator(self.config)
            summary_stats = artifact_generator._generate_performance_summary(benchmark_results)
            
            # Create GitHub Actions summary markdown
            summary_md = f"""# FlyRigLoader Benchmark Results

## Performance Summary
- **Total Tests**: {summary_stats['total_tests']}
- **Pass Rate**: {summary_stats['pass_rate_percent']}%
- **SLA Compliance**: {summary_stats['sla_compliance_rate_percent']}%
- **Memory Efficiency**: {summary_stats['memory_efficiency_rate_percent']}%
- **Performance Grade**: {summary_stats['performance_grade']}

## Environment
- **Platform**: {platform.system()}
- **Python**: {platform.python_version()}
- **Workflow**: {self.workflow_context.get('github_workflow', 'Unknown')}
- **Run ID**: {self.workflow_context.get('github_run_id', 'Unknown')}

## Test Results
"""
            
            # Add individual test results
            for test_name, test_results in benchmark_results.items():
                if isinstance(test_results, dict) and 'stats' in test_results:
                    stats = test_results['stats']
                    status = "✅" if test_results.get('status') == 'passed' else "❌"
                    summary_md += f"- {status} **{test_name}**: {stats.get('mean', 0):.4f}s (±{stats.get('stddev', 0):.4f}s)\\n"
            
            # Write summary to GitHub Actions environment
            github_step_summary = os.getenv('GITHUB_STEP_SUMMARY')
            if github_step_summary:
                with open(github_step_summary, 'a') as f:
                    f.write(summary_md)
                return Path(github_step_summary)
            
            # Fallback: save to artifact directory
            summary_path = artifact_dirs['reports'] / 'github-actions-summary.md'
            with open(summary_path, 'w') as f:
                f.write(summary_md)
            return summary_path
            
        except Exception as e:
            warnings.warn(f"Failed to generate GitHub Actions summary: {e}")
            return None
    
    def configure_performance_alerting(
        self,
        regression_results: Dict[str, Any],
        alert_thresholds: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Configure performance alerting based on regression detection results.
        
        Args:
            regression_results: Results from regression detection analysis
            alert_thresholds: Custom alert thresholds, uses config defaults if None
            
        Returns:
            Dict containing alerting configuration and actions
        """
        if alert_thresholds is None:
            alert_thresholds = self.config.performance_alerting
        
        alerting_config = {
            'alerts_generated': [],
            'github_actions_outputs': {},
            'notification_channels': [],
            'alert_summary': {}
        }
        
        # Analyze regression results for alerting
        critical_regressions = []
        warning_regressions = []
        
        if 'test_results' in regression_results:
            for test_name, test_result in regression_results['test_results'].items():
                if test_result['regression_analysis']['regression_detected']:
                    percent_change = abs(test_result['regression_analysis']['percent_change'])
                    
                    if percent_change > alert_thresholds['regression_threshold_percent']:
                        if percent_change > alert_thresholds['sla_violation_threshold_percent'] * 2:
                            critical_regressions.append({
                                'test_name': test_name,
                                'percent_change': percent_change,
                                'confidence': test_result['regression_analysis']['confidence']
                            })
                        else:
                            warning_regressions.append({
                                'test_name': test_name,
                                'percent_change': percent_change,
                                'confidence': test_result['regression_analysis']['confidence']
                            })
        
        # Generate alerts
        if critical_regressions:
            alerting_config['alerts_generated'].append({
                'level': 'critical',
                'message': f"Critical performance regressions detected in {len(critical_regressions)} tests",
                'tests': critical_regressions
            })
        
        if warning_regressions:
            alerting_config['alerts_generated'].append({
                'level': 'warning',
                'message': f"Performance regressions detected in {len(warning_regressions)} tests",
                'tests': warning_regressions
            })
        
        # Configure GitHub Actions outputs for workflow decisions
        if self.is_github_actions:
            alerting_config['github_actions_outputs'] = {
                'performance-status': 'critical' if critical_regressions else ('warning' if warning_regressions else 'success'),
                'regression-count': str(len(critical_regressions) + len(warning_regressions)),
                'critical-regressions': str(len(critical_regressions)),
                'alert-summary': self._generate_alert_summary_text(critical_regressions, warning_regressions)
            }
            
            # Set GitHub Actions outputs
            self._set_github_actions_outputs(alerting_config['github_actions_outputs'])
        
        return alerting_config
    
    def _generate_alert_summary_text(
        self,
        critical_regressions: List[Dict],
        warning_regressions: List[Dict]
    ) -> str:
        """Generate text summary of performance alerts."""
        if not critical_regressions and not warning_regressions:
            return "No performance regressions detected"
        
        summary_parts = []
        
        if critical_regressions:
            critical_tests = [reg['test_name'] for reg in critical_regressions]
            summary_parts.append(f"CRITICAL: {', '.join(critical_tests)}")
        
        if warning_regressions:
            warning_tests = [reg['test_name'] for reg in warning_regressions]
            summary_parts.append(f"WARNING: {', '.join(warning_tests)}")
        
        return "; ".join(summary_parts)
    
    def _set_github_actions_outputs(self, outputs: Dict[str, str]):
        """Set GitHub Actions workflow outputs."""
        try:
            github_output = os.getenv('GITHUB_OUTPUT')
            if github_output:
                with open(github_output, 'a') as f:
                    for key, value in outputs.items():
                        f.write(f"{key}={value}\n")
        except Exception as e:
            warnings.warn(f"Failed to set GitHub Actions outputs: {e}")
    
    def generate_ci_performance_report(
        self,
        benchmark_results: Dict[str, Any],
        output_dir: Path = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive CI/CD performance report with all integration features.
        
        Args:
            benchmark_results: Complete benchmark results
            output_dir: Output directory for reports, creates default if None
            
        Returns:
            Dict containing comprehensive CI integration analysis
        """
        if output_dir is None:
            output_dir = Path.cwd() / 'ci-performance-reports'
        
        # Setup artifact collection
        artifact_dirs = self.setup_artifact_directories(output_dir)
        
        # Collect all artifacts
        artifacts = self.collect_benchmark_artifacts(
            benchmark_results=benchmark_results,
            artifact_dirs=artifact_dirs,
            generate_all_formats=True
        )
        
        # Perform regression analysis
        regression_detector = RegressionDetector()
        regression_results = regression_detector.batch_regression_detection(benchmark_results)
        
        # Configure alerting
        alerting_config = self.configure_performance_alerting(regression_results)
        
        # Generate cross-platform analysis if multiple platforms detected
        cross_platform_analysis = None
        if len(set(result.get('platform', 'unknown') for result in benchmark_results.values())) > 1:
            validator = CrossPlatformValidator()
            # This would need platform-specific results - placeholder for now
            cross_platform_analysis = {'status': 'multiple_platforms_detected', 'analysis': 'pending'}
        
        # Compile comprehensive CI report
        ci_report = {
            'ci_metadata': {
                'is_github_actions': self.is_github_actions,
                'workflow_context': self.workflow_context,
                'generation_timestamp': datetime.now().isoformat(),
                'retention_policy_days': self.config.github_actions_settings['artifact_retention_days']
            },
            'artifacts_generated': artifacts,
            'regression_analysis': regression_results,
            'alerting_configuration': alerting_config,
            'cross_platform_analysis': cross_platform_analysis,
            'recommendations': self._generate_ci_recommendations(
                benchmark_results, regression_results, alerting_config
            )
        }
        
        # Save CI report
        ci_report_path = artifact_dirs['reports'] / 'ci-integration-report.json'
        with open(ci_report_path, 'w') as f:
            json.dump(ci_report, f, indent=2, default=str)
        
        ci_report['report_path'] = str(ci_report_path)
        
        return ci_report
    
    def _generate_ci_recommendations(
        self,
        benchmark_results: Dict[str, Any],
        regression_results: Dict[str, Any],
        alerting_config: Dict[str, Any]
    ) -> List[str]:
        """Generate CI/CD-specific recommendations based on analysis."""
        recommendations = []
        
        # Analyze test execution efficiency
        total_time = sum(
            result.get('stats', {}).get('mean', 0) * result.get('stats', {}).get('iterations', 1)
            for result in benchmark_results.values()
            if isinstance(result, dict)
        )
        
        if total_time > 300:  # >5 minutes
            recommendations.append("Consider optimizing benchmark execution time for faster CI feedback")
        
        # Analyze regression patterns
        if regression_results['regressions_detected'] > 0:
            recommendations.append("Performance regressions detected - review recent changes")
            
            if regression_results['critical_regressions'] > 0:
                recommendations.append("Critical regressions require immediate attention")
        
        # Analyze alerting effectiveness
        if not alerting_config['alerts_generated']:
            recommendations.append("Performance monitoring is healthy - no alerts generated")
        
        # GitHub Actions specific recommendations
        if self.is_github_actions:
            recommendations.append("Artifacts configured for GitHub Actions with 90-day retention")
            
            if self.workflow_context.get('github_event_name') == 'pull_request':
                recommendations.append("PR performance validation completed - review artifacts before merge")
        
        return recommendations


# ============================================================================
# MAIN COORDINATION UTILITIES
# ============================================================================

class BenchmarkUtilsCoordinator:
    """
    Main coordination class for all benchmark utilities providing unified interface
    for statistical analysis, memory profiling, environment normalization,
    artifact generation, regression detection, and CI/CD integration.
    
    Serves as the primary entry point for benchmark test suite utilities with
    comprehensive orchestration of performance validation workflows.
    """
    
    def __init__(self, config: BenchmarkConfig = None):
        """
        Initialize benchmark utilities coordinator.
        
        Args:
            config: Benchmark configuration, uses default if None
        """
        self.config = config or DEFAULT_BENCHMARK_CONFIG
        
        # Initialize all utility components
        self.statistical_engine = StatisticalAnalysisEngine(self.config.statistical_analysis)
        self.environment_analyzer = EnvironmentAnalyzer(self.config.environment_normalization)
        self.artifact_generator = PerformanceArtifactGenerator(self.config.cicd_integration)
        self.regression_detector = RegressionDetector(self.statistical_engine, self.config.statistical_analysis)
        self.cross_platform_validator = CrossPlatformValidator(self.config.environment_normalization)
        self.cicd_manager = CICDIntegrationManager(self.config.cicd_integration)
    
    def execute_comprehensive_analysis(
        self,
        benchmark_results: Dict[str, Any],
        output_dir: Path = None,
        enable_regression_detection: bool = True,
        enable_cross_platform_validation: bool = False,
        enable_ci_integration: bool = None
    ) -> Dict[str, Any]:
        """
        Execute comprehensive benchmark analysis with all utilities.
        
        Args:
            benchmark_results: Raw benchmark results from pytest-benchmark
            output_dir: Output directory for artifacts, creates default if None
            enable_regression_detection: Whether to perform regression analysis
            enable_cross_platform_validation: Whether to validate cross-platform consistency
            enable_ci_integration: Whether to enable CI/CD integration, auto-detects if None
            
        Returns:
            Dict containing comprehensive analysis results and recommendations
        """
        if enable_ci_integration is None:
            enable_ci_integration = self.cicd_manager.is_github_actions
        
        analysis_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'configuration': {
                'regression_detection_enabled': enable_regression_detection,
                'cross_platform_validation_enabled': enable_cross_platform_validation,
                'ci_integration_enabled': enable_ci_integration
            },
            'environment_analysis': {},
            'statistical_analysis': {},
            'memory_analysis': {},
            'regression_analysis': {},
            'cross_platform_analysis': {},
            'ci_integration_analysis': {},
            'artifacts_generated': {},
            'recommendations': [],
            'overall_status': 'unknown'
        }
        
        try:
            # 1. Environment Analysis
            analysis_results['environment_analysis'] = self.environment_analyzer.generate_environment_report()
            
            # 2. Statistical Analysis of benchmark results
            analysis_results['statistical_analysis'] = self._perform_statistical_analysis(benchmark_results)
            
            # 3. Memory Analysis (if memory data available)
            analysis_results['memory_analysis'] = self._perform_memory_analysis(benchmark_results)
            
            # 4. Regression Detection
            if enable_regression_detection:
                analysis_results['regression_analysis'] = self.regression_detector.batch_regression_detection(benchmark_results)
            
            # 5. Cross-Platform Validation
            if enable_cross_platform_validation:
                # This would require platform-specific results
                analysis_results['cross_platform_analysis'] = {'status': 'not_implemented', 'reason': 'requires_platform_specific_data'}
            
            # 6. CI/CD Integration
            if enable_ci_integration:
                analysis_results['ci_integration_analysis'] = self.cicd_manager.generate_ci_performance_report(
                    benchmark_results=benchmark_results,
                    output_dir=output_dir
                )
                analysis_results['artifacts_generated'] = analysis_results['ci_integration_analysis'].get('artifacts_generated', {})
            
            # 7. Generate comprehensive recommendations
            analysis_results['recommendations'] = self._generate_comprehensive_recommendations(analysis_results)
            
            # 8. Determine overall status
            analysis_results['overall_status'] = self._determine_overall_status(analysis_results)
            
        except Exception as e:
            analysis_results['error'] = str(e)
            analysis_results['overall_status'] = 'error'
            warnings.warn(f"Comprehensive analysis failed: {e}")
        
        return analysis_results
    
    def _perform_statistical_analysis(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis on benchmark results."""
        statistical_results = {
            'confidence_intervals': {},
            'outlier_analysis': {},
            'consistency_metrics': {}
        }
        
        for test_name, test_data in benchmark_results.items():
            if isinstance(test_data, dict) and 'measurements' in test_data:
                measurements = test_data['measurements']
                
                # Calculate confidence interval
                try:
                    ci = self.statistical_engine.calculate_confidence_interval(measurements)
                    statistical_results['confidence_intervals'][test_name] = ci.to_dict()
                except Exception as e:
                    statistical_results['confidence_intervals'][test_name] = {'error': str(e)}
                
                # Detect outliers
                try:
                    cleaned_measurements, outlier_indices = self.statistical_engine.detect_outliers(measurements)
                    statistical_results['outlier_analysis'][test_name] = {
                        'outliers_detected': len(outlier_indices),
                        'outlier_indices': outlier_indices,
                        'outlier_percentage': (len(outlier_indices) / len(measurements)) * 100,
                        'cleaned_measurement_count': len(cleaned_measurements)
                    }
                except Exception as e:
                    statistical_results['outlier_analysis'][test_name] = {'error': str(e)}
        
        return statistical_results
    
    def _perform_memory_analysis(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform memory analysis on benchmark results."""
        memory_analysis = {
            'memory_efficiency_summary': {},
            'leak_detection_summary': {},
            'large_dataset_analysis': {}
        }
        
        total_tests_with_memory = 0
        efficient_tests = 0
        tests_with_leaks = 0
        
        for test_name, test_data in benchmark_results.items():
            if isinstance(test_data, dict) and 'memory_stats' in test_data:
                memory_stats = test_data['memory_stats']
                total_tests_with_memory += 1
                
                if memory_stats.get('meets_efficiency_requirement', False):
                    efficient_tests += 1
                
                if memory_stats.get('leak_analysis', {}).get('leak_detected', False):
                    tests_with_leaks += 1
                
                # Analyze large dataset scenarios
                if memory_stats.get('data_size_mb', 0) > 100:  # >100MB datasets
                    memory_analysis['large_dataset_analysis'][test_name] = {
                        'data_size_mb': memory_stats.get('data_size_mb', 0),
                        'memory_multiplier': memory_stats.get('memory_multiplier', 0),
                        'efficient': memory_stats.get('meets_efficiency_requirement', False)
                    }
        
        if total_tests_with_memory > 0:
            memory_analysis['memory_efficiency_summary'] = {
                'total_tests_analyzed': total_tests_with_memory,
                'efficient_tests': efficient_tests,
                'efficiency_rate_percent': (efficient_tests / total_tests_with_memory) * 100,
                'tests_with_leaks': tests_with_leaks,
                'leak_rate_percent': (tests_with_leaks / total_tests_with_memory) * 100
            }
        
        return memory_analysis
    
    def _generate_comprehensive_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations based on all analysis results."""
        recommendations = []
        
        # Environment recommendations
        env_analysis = analysis_results.get('environment_analysis', {})
        if 'recommendations' in env_analysis:
            recommendations.extend(env_analysis['recommendations'])
        
        # Statistical analysis recommendations
        statistical_analysis = analysis_results.get('statistical_analysis', {})
        outlier_tests = [
            test for test, analysis in statistical_analysis.get('outlier_analysis', {}).items()
            if analysis.get('outlier_percentage', 0) > 10
        ]
        if outlier_tests:
            recommendations.append(f"High outlier rate detected in {len(outlier_tests)} tests - investigate measurement consistency")
        
        # Memory analysis recommendations
        memory_analysis = analysis_results.get('memory_analysis', {})
        memory_summary = memory_analysis.get('memory_efficiency_summary', {})
        if memory_summary.get('efficiency_rate_percent', 100) < 90:
            recommendations.append("Memory efficiency below 90% - optimize memory usage in failing tests")
        
        if memory_summary.get('leak_rate_percent', 0) > 0:
            recommendations.append("Memory leaks detected - implement proper cleanup in affected tests")
        
        # Regression analysis recommendations
        regression_analysis = analysis_results.get('regression_analysis', {})
        if regression_analysis.get('regressions_detected', 0) > 0:
            recommendations.append("Performance regressions detected - review recent changes and optimize")
        
        # CI/CD integration recommendations
        ci_analysis = analysis_results.get('ci_integration_analysis', {})
        if 'recommendations' in ci_analysis:
            recommendations.extend(ci_analysis['recommendations'])
        
        # Overall recommendations
        if not recommendations:
            recommendations.append("All benchmark analyses passed - performance is healthy")
        
        return recommendations
    
    def _determine_overall_status(self, analysis_results: Dict[str, Any]) -> str:
        """Determine overall status based on all analysis results."""
        # Check for critical issues
        regression_analysis = analysis_results.get('regression_analysis', {})
        if regression_analysis.get('critical_regressions', 0) > 0:
            return 'critical'
        
        memory_analysis = analysis_results.get('memory_analysis', {})
        memory_summary = memory_analysis.get('memory_efficiency_summary', {})
        if memory_summary.get('leak_rate_percent', 0) > 5:  # >5% tests with leaks
            return 'critical'
        
        # Check for warnings
        if regression_analysis.get('regressions_detected', 0) > 0:
            return 'warning'
        
        if memory_summary.get('efficiency_rate_percent', 100) < 85:  # <85% efficiency
            return 'warning'
        
        env_analysis = analysis_results.get('environment_analysis', {})
        if env_analysis.get('benchmarking_suitability', {}).get('level') in ['poor', 'fair']:
            return 'warning'
        
        # All checks passed
        return 'success'


# ============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON OPERATIONS
# ============================================================================

def create_memory_profiling_context(data_size_estimate: int, **kwargs):
    """Convenience function to create memory profiling context."""
    return memory_profiling_context(data_size_estimate, **kwargs)


def analyze_benchmark_results(
    benchmark_results: Dict[str, Any],
    config: BenchmarkConfig = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for comprehensive benchmark analysis.
    
    Args:
        benchmark_results: Benchmark results from pytest-benchmark
        config: Benchmark configuration, uses default if None
        **kwargs: Additional options for analysis
        
    Returns:
        Dict containing comprehensive analysis results
    """
    coordinator = BenchmarkUtilsCoordinator(config)
    return coordinator.execute_comprehensive_analysis(benchmark_results, **kwargs)


def detect_performance_regression(
    test_name: str,
    current_measurements: List[float],
    baseline_measurements: List[float] = None,
    config: StatisticalAnalysisConfig = None
) -> Dict[str, Any]:
    """
    Convenience function for performance regression detection.
    
    Args:
        test_name: Name of the test
        current_measurements: Current performance measurements
        baseline_measurements: Baseline measurements for comparison
        config: Statistical analysis configuration
        
    Returns:
        Dict containing regression analysis results
    """
    detector = RegressionDetector(config=config)
    return detector.detect_regression(test_name, current_measurements, baseline_measurements)


def normalize_environment_performance(
    measurement: float,
    baseline_environment: Dict[str, Any] = None,
    config: EnvironmentNormalizationConfig = None
) -> float:
    """
    Convenience function for environment-normalized performance measurement.
    
    Args:
        measurement: Raw performance measurement
        baseline_environment: Baseline environment for normalization
        config: Environment normalization configuration
        
    Returns:
        Normalized performance measurement
    """
    analyzer = EnvironmentAnalyzer(config)
    normalization_factors = analyzer.calculate_normalization_factors(baseline_environment)
    return analyzer.normalize_performance_measurement(measurement, normalization_factors)


def generate_performance_artifacts(
    benchmark_results: Dict[str, Any],
    output_dir: Path,
    formats: List[str] = None,
    config: CICDIntegrationConfig = None
) -> Dict[str, Path]:
    """
    Convenience function for generating performance artifacts.
    
    Args:
        benchmark_results: Benchmark results to generate artifacts for
        output_dir: Output directory for artifacts
        formats: List of formats to generate ('json', 'csv', 'html')
        config: CI/CD integration configuration
        
    Returns:
        Dict mapping format names to generated file paths
    """
    if formats is None:
        formats = ['json', 'csv', 'html']
    
    generator = PerformanceArtifactGenerator(config)
    artifacts = {}
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if 'json' in formats:
        json_path = output_dir / f'benchmark-report-{timestamp}.json'
        generator.generate_json_report(benchmark_results, json_path)
        artifacts['json'] = json_path
    
    if 'csv' in formats:
        csv_path = output_dir / f'benchmark-data-{timestamp}.csv'
        generator.generate_csv_report(benchmark_results, csv_path)
        artifacts['csv'] = csv_path
    
    if 'html' in formats:
        html_path = output_dir / f'benchmark-report-{timestamp}.html'
        generator.generate_html_report(benchmark_results, html_path)
        artifacts['html'] = html_path
    
    return artifacts


# ============================================================================
# MODULE INITIALIZATION AND EXPORTS
# ============================================================================

__all__ = [
    # Memory Profiling
    'MemoryProfiler',
    'memory_profiling_context',
    'estimate_data_size',
    
    # Statistical Analysis
    'StatisticalAnalysisEngine',
    'ConfidenceInterval',
    
    # Environment Normalization
    'EnvironmentAnalyzer',
    
    # Artifact Generation
    'PerformanceArtifactGenerator',
    
    # Regression Detection
    'RegressionDetector',
    
    # Cross-Platform Validation
    'CrossPlatformValidator',
    
    # CI/CD Integration
    'CICDIntegrationManager',
    
    # Main Coordination
    'BenchmarkUtilsCoordinator',
    
    # Convenience Functions
    'create_memory_profiling_context',
    'analyze_benchmark_results',
    'detect_performance_regression',
    'normalize_environment_performance',
    'generate_performance_artifacts'
]


# Module metadata
__version__ = '1.0.0'
__author__ = 'FlyRigLoader Benchmark Test Suite'
__description__ = 'Comprehensive utilities for benchmark test suite with statistical analysis, memory profiling, and CI/CD integration'