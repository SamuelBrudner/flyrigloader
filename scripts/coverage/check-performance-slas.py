#!/usr/bin/env python3
"""
Performance SLA Validation Engine

This module implements comprehensive statistical analysis, regression detection, and quality gate 
enforcement to ensure data loading and transformation performance requirements essential for 
neuroscience research workflows.

Key Features:
- Statistical analysis with confidence intervals for performance validation
- Integration with pytest-benchmark framework and optional performance testing  
- Quality gate enforcement for performance regression detection
- Cross-platform performance normalization and environment constraints
- Memory profiling integration for large dataset processing scenarios
- CI/CD integration for performance monitoring and alerting

Performance SLA Targets:
- TST-PERF-001: Data loading <1s per 100MB
- TST-PERF-002: DataFrame transformation <500ms per 1M rows  
- TST-PERF-003: File discovery <5s for 10,000 files
- TST-PERF-004: Configuration loading <100ms

Usage:
    python scripts/coverage/check-performance-slas.py [options]
    
    Options:
        --benchmark-results PATH    Path to pytest-benchmark results JSON
        --config PATH              Path to performance SLA configuration
        --quality-gates PATH       Path to quality gates configuration  
        --output-dir PATH          Output directory for reports
        --verbose                  Enable verbose logging
        --ci-mode                  Enable CI/CD integration mode
        --threshold-override FLOAT Override SLA threshold multiplier
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from statistics import mean, stdev, median
from math import sqrt

import psutil
import yaml

# Suppress warnings for statistical calculations
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceSLA:
    """Performance SLA specification with statistical validation."""
    name: str
    description: str
    threshold_ms: float
    unit: str = "ms"
    confidence_level: float = 0.95
    regression_threshold: float = 0.15  # 15% degradation threshold
    max_variance: float = 0.20  # 20% maximum variance
    baseline_samples: int = 5  # Minimum samples for baseline
    
    def __post_init__(self):
        """Validate SLA configuration parameters."""
        if self.threshold_ms <= 0:
            raise ValueError(f"Threshold must be positive: {self.threshold_ms}")
        if not 0.5 <= self.confidence_level <= 0.99:
            raise ValueError(f"Confidence level must be between 0.5 and 0.99: {self.confidence_level}")
        if self.regression_threshold <= 0:
            raise ValueError(f"Regression threshold must be positive: {self.regression_threshold}")


@dataclass  
class BenchmarkResult:
    """Individual benchmark measurement result."""
    name: str
    duration_ms: float
    memory_mb: Optional[float] = None
    timestamp: Optional[str] = None
    environment: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set default timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class StatisticalAnalysis:
    """Statistical analysis results for performance measurements."""
    mean: float
    median: float
    std_dev: float
    variance: float
    min_value: float
    max_value: float
    confidence_interval: Tuple[float, float]
    coefficient_of_variation: float
    outliers: List[float]
    sample_size: int


@dataclass
class SLAValidationResult:
    """SLA validation result with comprehensive analysis."""
    sla_name: str
    passed: bool
    measured_value: float
    threshold_value: float
    margin: float
    statistical_analysis: StatisticalAnalysis
    regression_detected: bool
    confidence_score: float
    environment_normalized: bool
    baseline_comparison: Optional[Dict[str, float]] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        """Initialize warnings list if None."""
        if self.warnings is None:
            self.warnings = []


class EnvironmentNormalizer:
    """Cross-platform performance normalization utilities."""
    
    def __init__(self):
        self.cpu_info = self._get_cpu_info()
        self.memory_info = self._get_memory_info()
        self.platform_info = self._get_platform_info()
        self.normalization_factors = self._calculate_normalization_factors()
        
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information for normalization."""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'cpu_percent': psutil.cpu_percent(interval=1)
            }
        except Exception as e:
            logger.warning(f"Failed to get CPU info: {e}")
            return {'cpu_count': 1, 'cpu_count_logical': 1}
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information for normalization."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'percent_used': memory.percent
            }
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            return {'total_gb': 8.0, 'available_gb': 6.0, 'percent_used': 25.0}
    
    def _get_platform_info(self) -> Dict[str, str]:
        """Get platform information for normalization."""
        import platform
        return {
            'system': platform.system(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
            'platform': platform.platform()
        }
    
    def _calculate_normalization_factors(self) -> Dict[str, float]:
        """Calculate environment-specific normalization factors."""
        # GitHub Actions standard: 2-core CPU, 7GB RAM
        reference_cpu_count = 2
        reference_memory_gb = 7.0
        
        # Calculate CPU normalization factor
        cpu_factor = reference_cpu_count / max(self.cpu_info.get('cpu_count', 1), 1)
        
        # Calculate memory normalization factor  
        memory_factor = reference_memory_gb / max(self.memory_info.get('total_gb', 8.0), 4.0)
        
        # Platform-specific adjustments
        platform_factor = 1.0
        system = self.platform_info.get('system', '').lower()
        if system == 'windows':
            platform_factor = 1.1  # Windows overhead
        elif system == 'darwin':  
            platform_factor = 0.95  # macOS optimization
        
        return {
            'cpu_factor': min(max(cpu_factor, 0.5), 2.0),  # Clamp between 0.5-2.0
            'memory_factor': min(max(memory_factor, 0.7), 1.5),  # Clamp between 0.7-1.5
            'platform_factor': platform_factor,
            'combined_factor': cpu_factor * platform_factor
        }
    
    def normalize_measurement(self, measurement_ms: float) -> float:
        """Normalize performance measurement for environment differences."""
        normalized = measurement_ms * self.normalization_factors['combined_factor']
        logger.debug(f"Normalized {measurement_ms}ms to {normalized}ms using factor {self.normalization_factors['combined_factor']}")
        return normalized
    
    def get_environment_summary(self) -> Dict[str, Any]:
        """Get comprehensive environment summary for reporting."""
        return {
            'cpu_info': self.cpu_info,
            'memory_info': self.memory_info, 
            'platform_info': self.platform_info,
            'normalization_factors': self.normalization_factors
        }


class StatisticalAnalyzer:
    """Statistical analysis and confidence interval calculations."""
    
    @staticmethod
    def calculate_confidence_interval(data: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for sample data."""
        if len(data) < 2:
            return (data[0], data[0]) if data else (0.0, 0.0)
            
        sample_mean = mean(data)
        sample_std = stdev(data) if len(data) > 1 else 0.0
        n = len(data)
        
        # Use t-distribution for small samples (n < 30) or z-distribution for large samples
        if n < 30:
            # Approximation for t-distribution critical value
            # For 95% confidence and small samples, use ~2.0 as approximation
            critical_value = 2.0 if confidence_level >= 0.95 else 1.64
        else:
            # Z-distribution critical values
            critical_value = 1.96 if confidence_level >= 0.95 else 1.64
            
        margin_of_error = critical_value * (sample_std / sqrt(n))
        
        return (sample_mean - margin_of_error, sample_mean + margin_of_error)
    
    @staticmethod 
    def detect_outliers(data: List[float], threshold: float = 2.0) -> List[float]:
        """Detect outliers using modified z-score method."""
        if len(data) < 3:
            return []
            
        median_val = median(data)
        mad = median([abs(x - median_val) for x in data])  # Median Absolute Deviation
        
        if mad == 0:
            return []
            
        modified_z_scores = [0.6745 * (x - median_val) / mad for x in data]
        outliers = [data[i] for i, score in enumerate(modified_z_scores) if abs(score) > threshold]
        
        return outliers
    
    @classmethod
    def analyze_performance_data(cls, data: List[float], confidence_level: float = 0.95) -> StatisticalAnalysis:
        """Perform comprehensive statistical analysis on performance data."""
        if not data:
            raise ValueError("Cannot analyze empty performance data")
            
        # Remove None values and convert to float
        clean_data = [float(x) for x in data if x is not None]
        
        if not clean_data:
            raise ValueError("No valid performance data after cleaning")
            
        sample_mean = mean(clean_data)
        sample_median = median(clean_data)
        sample_std = stdev(clean_data) if len(clean_data) > 1 else 0.0
        sample_variance = sample_std ** 2
        min_val = min(clean_data)
        max_val = max(clean_data)
        
        confidence_interval = cls.calculate_confidence_interval(clean_data, confidence_level)
        coefficient_of_variation = (sample_std / sample_mean) if sample_mean > 0 else 0.0
        outliers = cls.detect_outliers(clean_data)
        
        return StatisticalAnalysis(
            mean=sample_mean,
            median=sample_median,
            std_dev=sample_std,
            variance=sample_variance,
            min_value=min_val,
            max_value=max_val,
            confidence_interval=confidence_interval,
            coefficient_of_variation=coefficient_of_variation,
            outliers=outliers,
            sample_size=len(clean_data)
        )


class BaselineManager:
    """Manages performance baselines for regression detection."""
    
    def __init__(self, baseline_file: Path):
        self.baseline_file = baseline_file
        self.baselines = self._load_baselines()
        
    def _load_baselines(self) -> Dict[str, Any]:
        """Load performance baselines from file."""
        if not self.baseline_file.exists():
            logger.info(f"Baseline file {self.baseline_file} does not exist, starting fresh")
            return {}
            
        try:
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load baselines from {self.baseline_file}: {e}")
            return {}
    
    def save_baselines(self):
        """Save current baselines to file."""
        try:
            self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.baseline_file, 'w') as f:
                json.dump(self.baselines, f, indent=2)
            logger.info(f"Baselines saved to {self.baseline_file}")
        except IOError as e:
            logger.error(f"Failed to save baselines to {self.baseline_file}: {e}")
    
    def update_baseline(self, sla_name: str, measurement: float, metadata: Optional[Dict] = None):
        """Update baseline for a specific SLA."""
        if sla_name not in self.baselines:
            self.baselines[sla_name] = {
                'measurements': [],
                'last_updated': None,
                'metadata': {}
            }
            
        self.baselines[sla_name]['measurements'].append({
            'value': measurement,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        })
        
        # Keep only recent measurements (last 30 days)
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        self.baselines[sla_name]['measurements'] = [
            m for m in self.baselines[sla_name]['measurements']
            if datetime.fromisoformat(m['timestamp']) > cutoff_date
        ]
        
        self.baselines[sla_name]['last_updated'] = datetime.utcnow().isoformat()
        
    def get_baseline_stats(self, sla_name: str) -> Optional[Dict[str, float]]:
        """Get baseline statistics for regression comparison."""
        if sla_name not in self.baselines:
            return None
            
        measurements = [m['value'] for m in self.baselines[sla_name]['measurements']]
        
        if len(measurements) < 3:  # Need minimum samples for reliable baseline
            return None
            
        try:
            analysis = StatisticalAnalyzer.analyze_performance_data(measurements)
            return {
                'mean': analysis.mean,
                'median': analysis.median,
                'std_dev': analysis.std_dev,
                'confidence_interval_lower': analysis.confidence_interval[0],
                'confidence_interval_upper': analysis.confidence_interval[1],
                'sample_size': analysis.sample_size
            }
        except Exception as e:
            logger.warning(f"Failed to calculate baseline stats for {sla_name}: {e}")
            return None


class PerformanceSLAValidator:
    """Main performance SLA validation engine."""
    
    def __init__(self, config_path: Optional[Path] = None, quality_gates_path: Optional[Path] = None):
        self.config_path = config_path or Path("scripts/coverage/performance-sla-config.yml")
        self.quality_gates_path = quality_gates_path or Path("scripts/coverage/quality-gates.yml") 
        
        # Initialize components
        self.environment = EnvironmentNormalizer()
        self.analyzer = StatisticalAnalyzer()
        self.baseline_manager = BaselineManager(Path("scripts/coverage/performance-baselines.json"))
        
        # Load configuration
        self.slas = self._load_sla_configuration()
        self.quality_gates = self._load_quality_gates()
        
        logger.info(f"Initialized SLA validator with {len(self.slas)} SLAs")
        
    def _load_sla_configuration(self) -> Dict[str, PerformanceSLA]:
        """Load SLA configuration from YAML file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            else:
                logger.warning(f"SLA config file {self.config_path} not found, using defaults")
                config_data = {}
                
            return self._create_default_slas(config_data)
        except Exception as e:
            logger.warning(f"Failed to load SLA config: {e}, using defaults")
            return self._create_default_slas({})
    
    def _create_default_slas(self, config_data: Dict) -> Dict[str, PerformanceSLA]:
        """Create default SLA specifications based on technical requirements."""
        defaults = {
            'data_loading_100mb': PerformanceSLA(
                name='TST-PERF-001: Data Loading Performance',
                description='Data loading must complete within 1 second per 100MB',
                threshold_ms=1000.0,  # 1 second per 100MB
                confidence_level=0.95,
                regression_threshold=0.15
            ),
            'transformation_1m_rows': PerformanceSLA(
                name='TST-PERF-002: DataFrame Transformation Performance', 
                description='DataFrame transformation must complete within 500ms per 1M rows',
                threshold_ms=500.0,  # 500ms per 1M rows
                confidence_level=0.95,
                regression_threshold=0.15
            ),
            'file_discovery_10k': PerformanceSLA(
                name='TST-PERF-003: File Discovery Performance',
                description='File discovery must complete within 5 seconds for 10,000 files',
                threshold_ms=5000.0,  # 5 seconds for 10k files
                confidence_level=0.95,
                regression_threshold=0.20
            ),
            'config_loading': PerformanceSLA(
                name='TST-PERF-004: Configuration Loading Performance',
                description='Configuration loading must complete within 100ms',
                threshold_ms=100.0,  # 100ms
                confidence_level=0.95,
                regression_threshold=0.10
            )
        }
        
        # Override with config file values if present
        slas = {}
        for key, default_sla in defaults.items():
            if key in config_data:
                config_override = config_data[key]
                slas[key] = PerformanceSLA(
                    name=config_override.get('name', default_sla.name),
                    description=config_override.get('description', default_sla.description),
                    threshold_ms=config_override.get('threshold_ms', default_sla.threshold_ms),
                    confidence_level=config_override.get('confidence_level', default_sla.confidence_level),
                    regression_threshold=config_override.get('regression_threshold', default_sla.regression_threshold),
                    max_variance=config_override.get('max_variance', default_sla.max_variance)
                )
            else:
                slas[key] = default_sla
                
        return slas
    
    def _load_quality_gates(self) -> Dict[str, Any]:
        """Load quality gate configuration."""
        try:
            if self.quality_gates_path.exists():
                with open(self.quality_gates_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"Quality gates file {self.quality_gates_path} not found, using defaults")
                return self._get_default_quality_gates()
        except Exception as e:
            logger.warning(f"Failed to load quality gates: {e}, using defaults")
            return self._get_default_quality_gates()
    
    def _get_default_quality_gates(self) -> Dict[str, Any]:
        """Get default quality gate configuration."""
        return {
            'performance': {
                'enforce_sla_compliance': True,
                'allow_regression_threshold': 0.15,
                'require_baseline_comparison': True,
                'max_variance_threshold': 0.20,
                'min_confidence_score': 0.90
            },
            'reporting': {
                'generate_detailed_reports': True,
                'include_environment_info': True,
                'include_statistical_analysis': True,
                'retention_days': 90
            }
        }
    
    def load_benchmark_results(self, results_path: Path) -> List[BenchmarkResult]:
        """Load pytest-benchmark results from JSON file."""
        if not results_path.exists():
            logger.warning(f"Benchmark results file {results_path} not found")
            return []
            
        try:
            with open(results_path, 'r') as f:
                data = json.load(f)
                
            results = []
            
            # Handle pytest-benchmark JSON format
            if 'benchmarks' in data:
                for benchmark in data['benchmarks']:
                    # Extract timing statistics
                    stats = benchmark.get('stats', {})
                    name = benchmark.get('name', 'unknown')
                    
                    # Get mean duration in milliseconds
                    mean_duration = stats.get('mean', 0) * 1000  # Convert to ms
                    
                    # Extract memory information if available
                    memory_mb = None
                    if 'extra_info' in benchmark and 'memory_usage' in benchmark['extra_info']:
                        memory_mb = benchmark['extra_info']['memory_usage']
                        
                    result = BenchmarkResult(
                        name=name,
                        duration_ms=mean_duration,
                        memory_mb=memory_mb,
                        environment=self.environment.get_environment_summary(),
                        metadata=benchmark
                    )
                    results.append(result)
                    
            logger.info(f"Loaded {len(results)} benchmark results from {results_path}")
            return results
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse benchmark results from {results_path}: {e}")
            return []
    
    def validate_sla(self, sla: PerformanceSLA, measurements: List[float], 
                     threshold_override: Optional[float] = None) -> SLAValidationResult:
        """Validate performance SLA against measurements."""
        if not measurements:
            return SLAValidationResult(
                sla_name=sla.name,
                passed=False,
                measured_value=0.0,
                threshold_value=sla.threshold_ms,
                margin=0.0,
                statistical_analysis=StatisticalAnalysis(0, 0, 0, 0, 0, 0, (0, 0), 0, [], 0),
                regression_detected=False,
                confidence_score=0.0,
                environment_normalized=True,
                warnings=["No measurements available for validation"]
            )
        
        # Normalize measurements for environment differences
        normalized_measurements = [
            self.environment.normalize_measurement(m) for m in measurements
        ]
        
        # Perform statistical analysis
        analysis = self.analyzer.analyze_performance_data(normalized_measurements, sla.confidence_level)
        
        # Apply threshold override if provided
        effective_threshold = threshold_override or sla.threshold_ms
        
        # Check SLA compliance using mean value
        measured_value = analysis.mean
        passed = measured_value <= effective_threshold
        margin = (measured_value - effective_threshold) / effective_threshold if effective_threshold > 0 else 0.0
        
        # Calculate confidence score based on consistency
        confidence_score = max(0.0, 1.0 - analysis.coefficient_of_variation)
        
        # Check for regression against baseline
        regression_detected = False
        baseline_comparison = None
        baseline_stats = self.baseline_manager.get_baseline_stats(sla.name)
        
        if baseline_stats:
            baseline_mean = baseline_stats['mean']
            regression_threshold = baseline_mean * (1 + sla.regression_threshold)
            regression_detected = measured_value > regression_threshold
            
            baseline_comparison = {
                'baseline_mean': baseline_mean,
                'current_mean': measured_value,
                'regression_threshold': regression_threshold,
                'regression_detected': regression_detected,
                'improvement_ratio': baseline_mean / measured_value if measured_value > 0 else 1.0
            }
        
        # Collect warnings
        warnings = []
        if analysis.coefficient_of_variation > sla.max_variance:
            warnings.append(f"High variance detected: {analysis.coefficient_of_variation:.2%} > {sla.max_variance:.2%}")
        if len(analysis.outliers) > 0:
            warnings.append(f"Outliers detected: {len(analysis.outliers)} values")
        if confidence_score < 0.8:
            warnings.append(f"Low confidence score: {confidence_score:.2f}")
        if regression_detected:
            warnings.append("Performance regression detected compared to baseline")
            
        # Update baseline with current measurement
        self.baseline_manager.update_baseline(
            sla.name, 
            measured_value,
            {'environment': self.environment.get_environment_summary()}
        )
        
        return SLAValidationResult(
            sla_name=sla.name,
            passed=passed,
            measured_value=measured_value,
            threshold_value=effective_threshold,
            margin=margin,
            statistical_analysis=analysis,
            regression_detected=regression_detected,
            confidence_score=confidence_score,
            environment_normalized=True,
            baseline_comparison=baseline_comparison,
            warnings=warnings
        )
    
    def map_benchmark_to_sla(self, benchmark_name: str) -> Optional[str]:
        """Map benchmark name to SLA key."""
        name_lower = benchmark_name.lower()
        
        # Data loading benchmarks
        if any(term in name_lower for term in ['data_loading', 'load_data', 'pickle_load']):
            return 'data_loading_100mb'
        
        # Transformation benchmarks  
        if any(term in name_lower for term in ['transformation', 'transform', 'dataframe']):
            return 'transformation_1m_rows'
            
        # Discovery benchmarks
        if any(term in name_lower for term in ['discovery', 'discover', 'file_discovery']):
            return 'file_discovery_10k'
            
        # Configuration benchmarks
        if any(term in name_lower for term in ['config', 'configuration', 'yaml']):
            return 'config_loading'
            
        logger.warning(f"Could not map benchmark '{benchmark_name}' to any SLA")
        return None
    
    def validate_all_slas(self, benchmark_results: List[BenchmarkResult], 
                         threshold_override: Optional[float] = None) -> Dict[str, SLAValidationResult]:
        """Validate all SLAs against benchmark results."""
        # Group benchmark results by SLA
        sla_measurements = {sla_key: [] for sla_key in self.slas.keys()}
        
        for result in benchmark_results:
            sla_key = self.map_benchmark_to_sla(result.name)
            if sla_key and sla_key in sla_measurements:
                sla_measurements[sla_key].append(result.duration_ms)
        
        # Validate each SLA
        validation_results = {}
        for sla_key, sla in self.slas.items():
            measurements = sla_measurements[sla_key]
            if measurements:
                validation_results[sla_key] = self.validate_sla(sla, measurements, threshold_override)
                logger.info(f"Validated SLA {sla_key}: {'PASS' if validation_results[sla_key].passed else 'FAIL'}")
            else:
                logger.warning(f"No measurements found for SLA {sla_key}")
                # Create empty validation result
                validation_results[sla_key] = SLAValidationResult(
                    sla_name=sla.name,
                    passed=True,  # Pass if no measurements (optional benchmarks)
                    measured_value=0.0,
                    threshold_value=sla.threshold_ms,
                    margin=0.0,
                    statistical_analysis=StatisticalAnalysis(0, 0, 0, 0, 0, 0, (0, 0), 0, [], 0),
                    regression_detected=False,
                    confidence_score=0.0,
                    environment_normalized=True,
                    warnings=["No benchmark measurements available"]
                )
        
        return validation_results
    
    def generate_performance_report(self, validation_results: Dict[str, SLAValidationResult], 
                                  output_dir: Path, detailed: bool = True) -> Path:
        """Generate comprehensive performance report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate report timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"performance_sla_report_{timestamp}.json"
        
        # Calculate overall compliance
        total_slas = len(validation_results)
        passed_slas = sum(1 for result in validation_results.values() if result.passed)
        overall_compliance = (passed_slas / total_slas) if total_slas > 0 else 0.0
        
        # Detect any regressions
        regressions_detected = any(result.regression_detected for result in validation_results.values())
        
        # Compile report data
        report_data = {
            'metadata': {
                'timestamp': datetime.utcnow().isoformat(),
                'report_version': '1.0',
                'environment': self.environment.get_environment_summary(),
                'sla_count': total_slas,
                'quality_gates': self.quality_gates
            },
            'summary': {
                'overall_compliance': overall_compliance,
                'passed_slas': passed_slas,
                'failed_slas': total_slas - passed_slas,
                'regressions_detected': regressions_detected,
                'average_confidence_score': mean([r.confidence_score for r in validation_results.values()]) if validation_results else 0.0
            },
            'sla_results': {}
        }
        
        # Add detailed results for each SLA
        for sla_key, result in validation_results.items():
            sla_data = {
                'name': result.sla_name,
                'passed': result.passed,
                'measured_value_ms': result.measured_value,
                'threshold_ms': result.threshold_value,
                'margin_percent': result.margin * 100,
                'confidence_score': result.confidence_score,
                'regression_detected': result.regression_detected,
                'warnings': result.warnings
            }
            
            if detailed:
                sla_data.update({
                    'statistical_analysis': asdict(result.statistical_analysis),
                    'baseline_comparison': result.baseline_comparison,
                    'environment_normalized': result.environment_normalized
                })
            
            report_data['sla_results'][sla_key] = sla_data
        
        # Write report to file
        try:
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            logger.info(f"Performance report generated: {report_file}")
            
            # Also generate a summary for CI logs
            self._log_summary_report(report_data)
            
            return report_file
            
        except IOError as e:
            logger.error(f"Failed to write performance report to {report_file}: {e}")
            raise
    
    def _log_summary_report(self, report_data: Dict[str, Any]):
        """Log a summary report for CI visibility."""
        summary = report_data['summary']
        
        logger.info("=" * 60)
        logger.info("PERFORMANCE SLA VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Overall Compliance: {summary['overall_compliance']:.1%}")
        logger.info(f"SLAs Passed: {summary['passed_slas']}/{summary['passed_slas'] + summary['failed_slas']}")
        logger.info(f"Regressions Detected: {'YES' if summary['regressions_detected'] else 'NO'}")
        logger.info(f"Average Confidence: {summary['average_confidence_score']:.2f}")
        
        # Log individual SLA results
        for sla_key, sla_result in report_data['sla_results'].items():
            status = "✓ PASS" if sla_result['passed'] else "✗ FAIL"
            logger.info(f"  {sla_key}: {status} ({sla_result['measured_value_ms']:.1f}ms)")
            
            if sla_result['warnings']:
                for warning in sla_result['warnings']:
                    logger.warning(f"    ⚠ {warning}")
        
        logger.info("=" * 60)
    
    def enforce_quality_gates(self, validation_results: Dict[str, SLAValidationResult]) -> bool:
        """Enforce quality gates based on validation results."""
        performance_config = self.quality_gates.get('performance', {})
        
        # Check if SLA compliance enforcement is enabled
        if not performance_config.get('enforce_sla_compliance', True):
            logger.info("SLA compliance enforcement is disabled")
            return True
        
        # Check overall SLA compliance
        failed_slas = [result for result in validation_results.values() if not result.passed]
        if failed_slas:
            logger.error(f"Quality gate FAILED: {len(failed_slas)} SLA(s) failed compliance")
            for result in failed_slas:
                logger.error(f"  - {result.sla_name}: {result.measured_value:.1f}ms > {result.threshold_value:.1f}ms")
            return False
        
        # Check regression threshold
        if performance_config.get('require_baseline_comparison', True):
            regressions = [result for result in validation_results.values() if result.regression_detected]
            if regressions:
                logger.error(f"Quality gate FAILED: {len(regressions)} performance regression(s) detected")
                for result in regressions:
                    logger.error(f"  - {result.sla_name}: Performance regression detected")
                return False
        
        # Check confidence scores
        min_confidence = performance_config.get('min_confidence_score', 0.8)
        low_confidence = [result for result in validation_results.values() 
                         if result.confidence_score < min_confidence]
        if low_confidence:
            logger.warning(f"Quality gate WARNING: {len(low_confidence)} SLA(s) have low confidence scores")
            for result in low_confidence:
                logger.warning(f"  - {result.sla_name}: Confidence {result.confidence_score:.2f} < {min_confidence}")
            # Don't fail on low confidence, just warn
        
        logger.info("Quality gate PASSED: All performance SLAs met")
        return True
    
    def save_state(self):
        """Save validator state including baselines."""
        self.baseline_manager.save_baselines()


def main():
    """Main entry point for performance SLA validation."""
    parser = argparse.ArgumentParser(
        description="Performance SLA Validation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation with default paths
  python scripts/coverage/check-performance-slas.py

  # Validation with custom benchmark results
  python scripts/coverage/check-performance-slas.py --benchmark-results /path/to/results.json

  # CI mode with threshold override
  python scripts/coverage/check-performance-slas.py --ci-mode --threshold-override 1.2

  # Verbose validation with custom output directory
  python scripts/coverage/check-performance-slas.py --verbose --output-dir /tmp/perf-reports
        """
    )
    
    parser.add_argument(
        '--benchmark-results',
        type=Path,
        default=Path('scripts/benchmarks/pytest-benchmark-results.json'),
        help='Path to pytest-benchmark results JSON file'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('scripts/coverage/performance-sla-config.yml'),
        help='Path to performance SLA configuration file'
    )
    
    parser.add_argument(
        '--quality-gates',
        type=Path,
        default=Path('scripts/coverage/quality-gates.yml'),
        help='Path to quality gates configuration file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('scripts/coverage/reports'),
        help='Output directory for performance reports'
    )
    
    parser.add_argument(
        '--threshold-override',
        type=float,
        help='Override SLA thresholds with multiplier (e.g., 1.2 for 20% more lenient)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--ci-mode',
        action='store_true',
        help='Enable CI/CD integration mode'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    try:
        # Initialize validator
        logger.info("Initializing Performance SLA Validator")
        validator = PerformanceSLAValidator(args.config, args.quality_gates)
        
        # Load benchmark results
        logger.info(f"Loading benchmark results from {args.benchmark_results}")
        benchmark_results = validator.load_benchmark_results(args.benchmark_results)
        
        if not benchmark_results:
            logger.warning("No benchmark results found - this may be expected for optional performance testing")
            # In CI mode, don't fail if no benchmarks are available (they're optional)
            if args.ci_mode:
                logger.info("CI mode: Passing validation due to optional benchmark execution")
                sys.exit(0)
        
        # Validate SLAs
        logger.info("Validating performance SLAs")
        validation_results = validator.validate_all_slas(benchmark_results, args.threshold_override)
        
        # Generate performance report
        logger.info(f"Generating performance report in {args.output_dir}")
        report_file = validator.generate_performance_report(
            validation_results, 
            args.output_dir, 
            detailed=not args.ci_mode
        )
        
        # Enforce quality gates
        logger.info("Enforcing quality gates")
        quality_gates_passed = validator.enforce_quality_gates(validation_results)
        
        # Save validator state
        validator.save_state()
        
        # Exit with appropriate code
        if quality_gates_passed:
            logger.info("Performance SLA validation completed successfully")
            sys.exit(0)
        else:
            logger.error("Performance SLA validation failed")
            sys.exit(1)
            
    except Exception as e:
        logger.exception(f"Performance SLA validation failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()